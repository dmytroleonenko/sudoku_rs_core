//! # Vicinity-Search Hill-Climbing Engine (R3.4 Stage 2)
//!
//! ## Inputs
//! `seeds: Vec<VicinityEntry>` — initial puzzle pool with solved solutions and
//! se_score pre-populated. `cfg: &VicinityConfig` — search parameters.
//!
//! ## Mutates
//! None. Pure function: all state is local to `explore`.
//!
//! ## Returns
//! `Vec<VicinityEntry>` — emitted puzzles with `se_score >= cfg.target_se`.
//! Puzzles are deduplicated by approximate canonical hash before emission.
//!
//! ## Performance budget
//! Target: < 200 ms per seed at default `budget_iters=200` on M3 Max.
//! Dominant cost is `rate()` per candidate: ~1–2 ms for T3 puzzles.
//! K=32 per n=1 step × 200 iters × ~1.5 ms/rate ≈ ~10s max per seed at full
//! budget. In practice, most iters discard candidates at L2 uniqueness before
//! rating, reducing actual cost substantially.
//!
//! ## Algorithm reference
//! Approximate BFS/hill-climb: BinaryHeap max-heap by `se_score`, HashSet for
//! canonical dedup, cascade filter L1(clue_count)→L2(uniqueness)→L4(rate).
//! Mutation: pick n distinct clue cells, replace each digit with one ≠ current
//! ≠ solution[cell] (approximate pencilmark; see APPROXIMATION note below).
//!
//! ## AlphaEvolve contract
//! Self-contained file. Public API: `VicinityConfig`, `VicinityEntry`, `explore`.
//! Internal helpers can be freely refactored. Must preserve the invariant:
//! outputs are pairwise canonical-distinct and all have `se_score ≥ target_se`.
//!
//! ## APPROXIMATION notes
//! 1. Pencilmark: we choose replacement digit ∈ {1..=9} \ {current, solution[cell]}.
//!    This is an approximation — a digit might create a contradiction elsewhere in
//!    the puzzle. Such puzzles fail L2 uniqueness and are discarded harmlessly.
//! 2. Minimality (L1): we only check that `clue_count` is unchanged (±0). True
//!    minimality (every clue is necessary) is not verified — it's expensive and
//!    not required for hill-climbing quality. TODO: add optional minimality check.
//! 3. L3 (backtrack-count prefilter): not implemented. The backtracker in
//!    `generic::search` does not expose a per-call branch counter. TODO once
//!    exposed.
//! 4. Single-threaded. Rayon parallelism is a TODO (shared `seen` HashSet +
//!    `frontier` behind Mutex, or per-seed independent run).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::seq::SliceRandom;

use crate::generic::{
    canonical::canonical_hash,
    grid::Grid,
    rater::rate,
    search::{count_solutions_up_to, solve_unique},
    techniques::Tier,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for one vicinity-search run.
#[derive(Clone, Debug)]
pub struct VicinityConfig {
    /// Accept candidates with `se_score >= target_se` as output.
    pub target_se: f64,
    /// Backtrack-count threshold for L3 filter; 0 = disable (L3 not yet implemented).
    pub bt_prefilter: u32,
    /// Which mute-n values to use, typically [1, 2].
    pub mute_set: Vec<u8>,
    /// Per-seed iteration cap.
    pub budget_iters: usize,
    /// Global cap on emitted puzzles.
    pub max_outputs: usize,
    /// PRNG seed.
    pub seed: u64,
}

impl Default for VicinityConfig {
    fn default() -> Self {
        Self {
            target_se: 7.5,
            bt_prefilter: 0,
            mute_set: vec![1, 2],
            budget_iters: 200,
            max_outputs: 1000,
            seed: 0,
        }
    }
}

/// One puzzle entry in the vicinity search.
#[derive(Clone, Debug)]
pub struct VicinityEntry {
    /// 81-char puzzle string: digits for clues, '.' for empty.
    pub puzzle: String,
    /// 81-char fully solved grid (canonical for that grid).
    pub solution: String,
    /// SE-equivalent difficulty score from rater.
    pub se_score: f64,
    /// Tier from rater cascade.
    pub tier: Tier,
    /// Number of mutations from the original seed.
    pub generation: u32,
}

// ---------------------------------------------------------------------------
// Heap wrapper — max-heap by se_score
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct HeapEntry {
    se_score: f64,
    entry: VicinityEntry,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.se_score.total_cmp(&other.se_score) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher se_score has priority.
        self.se_score.total_cmp(&other.se_score)
    }
}

// ---------------------------------------------------------------------------
// Mutation helpers
// ---------------------------------------------------------------------------

/// Sample indices of clue cells (cells where puzzle[i] != '.').
/// Returns a Vec of cell indices.
fn clue_indices(puzzle: &str) -> Vec<usize> {
    puzzle
        .bytes()
        .enumerate()
        .filter_map(|(i, b)| if b != b'.' { Some(i) } else { None })
        .collect()
}

/// Count clues in a puzzle string.
fn clue_count(puzzle: &str) -> usize {
    puzzle.bytes().filter(|&b| b != b'.').count()
}

/// Apply a single mutation to `current`: pick `n` distinct clue cells and
/// replace each digit with a random digit ≠ current (from {1..=9}).
///
/// The resulting puzzle has the same clue positions but different digit values.
/// Most candidates will fail L2 uniqueness (they're invalid puzzles), but some
/// will have exactly one solution (potentially a different one) and pass
/// through to the rater.
///
/// APPROXIMATION: we sample digits uniformly from {1..=9} \ {current_digit}.
/// The spec's "legal digits given solution pencilmarks" is approximated as
/// all digits, since the solution may change. Candidates that create
/// contradictions or non-unique grids are filtered at L2.
///
/// Returns up to K mutated puzzle strings; each has the same clue count.
fn mutate_n(
    puzzle: &str,
    n: usize,
    k: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<String> {
    let clues: Vec<usize> = clue_indices(puzzle);
    if clues.len() < n {
        return Vec::new();
    }

    let puzzle_bytes: Vec<u8> = puzzle.bytes().collect();
    let mut results = Vec::with_capacity(k);

    // Try up to k*8 attempts to gather k distinct mutations.
    let max_tries = k * 8;
    for _ in 0..max_tries {
        if results.len() >= k {
            break;
        }

        // Pick n distinct clue cells.
        let mut clue_pick: Vec<usize> = clues.clone();
        clue_pick.shuffle(rng);
        let picked = &clue_pick[..n];

        // For each picked cell, choose a replacement digit ≠ current.
        let mut new_bytes = puzzle_bytes.clone();
        let mut ok = true;
        for &cell in picked {
            let current_digit = puzzle_bytes[cell]; // '1'..'9'
            // Candidates: 1..=9 excluding current.
            let mut candidates: Vec<u8> = (b'1'..=b'9')
                .filter(|&d| d != current_digit)
                .collect();
            if candidates.is_empty() {
                ok = false;
                break;
            }
            candidates.shuffle(rng);
            new_bytes[cell] = candidates[0];
        }
        if !ok {
            continue;
        }

        let new_puzzle = match String::from_utf8(new_bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        if new_puzzle == puzzle {
            continue;
        }
        results.push(new_puzzle);
    }

    results
}

// ---------------------------------------------------------------------------
// Cascade filter
// ---------------------------------------------------------------------------

/// L1: clue count must stay identical to original.
/// L2: must have exactly one solution (also obtains the solution).
/// L4: must rate at se_score >= target_se - 0.3.
///
/// Returns `Some(VicinityEntry)` if all filters pass, `None` otherwise.
/// The returned entry has its `solution` field populated from the actual
/// unique solution found by the solver.
fn cascade_filter(
    puzzle_str: &str,
    target_se: f64,
    expected_clue_count: usize,
    generation: u32,
) -> Option<VicinityEntry> {
    // L1 — clue count must be unchanged.
    if clue_count(puzzle_str) != expected_clue_count {
        return None;
    }

    // Parse the puzzle grid.
    let grid = Grid::<9, 3, 3>::from_str(puzzle_str)?;

    // L2 — uniqueness + solve: must have exactly one solution.
    if count_solutions_up_to::<9, 3, 3>(&grid, 2) != 1 {
        return None;
    }
    let sol_grid = solve_unique::<9, 3, 3>(&grid)?;
    let solution_str = sol_grid.to_string_grid();

    // L4 — rate and check se_score threshold.
    let r = rate::<9, 3, 3>(&grid);
    if r.rater_error {
        return None;
    }
    if r.se_score < target_se - 0.3 {
        return None;
    }

    Some(VicinityEntry {
        puzzle: puzzle_str.to_string(),
        solution: solution_str,
        se_score: r.se_score,
        tier: r.tier,
        generation,
    })
}

// ---------------------------------------------------------------------------
// Samples per mutation level
// ---------------------------------------------------------------------------

const K_N1: usize = 32;
const K_N2: usize = 16;
const K_N3: usize = 8;

fn k_for_n(n: usize) -> usize {
    match n {
        1 => K_N1,
        2 => K_N2,
        _ => K_N3,
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run vicinity-search hill-climbing from `seeds` according to `cfg`.
///
/// For each seed, maintains a max-heap (by se_score) frontier and a canonical
/// hash dedup set. Emits puzzles with `se_score >= cfg.target_se`.
///
/// Single-threaded; see module doc for parallelism TODO.
pub fn explore(seeds: Vec<VicinityEntry>, cfg: &VicinityConfig) -> Vec<VicinityEntry> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.seed);
    let mut outputs: Vec<VicinityEntry> = Vec::new();
    // Global seen set shared across all seeds for cross-seed dedup.
    let mut seen: HashSet<u128> = HashSet::new();

    // Pre-seed global seen with the seeds themselves.
    for seed in &seeds {
        if let Some(g) = Grid::<9, 3, 3>::from_str(&seed.puzzle) {
            seen.insert(canonical_hash::<9, 3, 3>(&g));
        }
    }

    'seed_loop: for seed in seeds {
        if outputs.len() >= cfg.max_outputs {
            break;
        }

        let seed_clue_count = clue_count(&seed.puzzle);
        let mut frontier: BinaryHeap<HeapEntry> = BinaryHeap::new();
        frontier.push(HeapEntry {
            se_score: seed.se_score,
            entry: seed.clone(),
        });

        let mut iters: usize = 0;
        // Stall tracking for escalation from n=1 to n=2.
        let mut stall_n1: usize = 0;
        let stall_threshold = 10; // consecutive iterations with no new survivor at n=1

        while !frontier.is_empty() && outputs.len() < cfg.max_outputs {
            if iters >= cfg.budget_iters {
                break;
            }
            iters += 1;

            let current = match frontier.pop() {
                Some(h) => h.entry,
                None => break,
            };

            // Determine which mute level to try this iteration.
            // Escalate to n=2 if n=1 has been stalling.
            let mute_level: u8 = if cfg.mute_set.is_empty() {
                break;
            } else if cfg.mute_set.len() == 1 || stall_n1 < stall_threshold {
                cfg.mute_set[0]
            } else if cfg.mute_set.len() >= 2 {
                cfg.mute_set[1]
            } else {
                // All levels stalled — drop this seed.
                break 'seed_loop;
            };

            let n = mute_level as usize;
            let k = k_for_n(n);

            let candidates = mutate_n(&current.puzzle, n, k, &mut rng);

            let mut found_any = false;
            for cand in candidates {
                // Canonical dedup before the expensive cascade.
                let parsed = match Grid::<9, 3, 3>::from_str(&cand) {
                    Some(g) => g,
                    None => continue,
                };
                let hash = canonical_hash::<9, 3, 3>(&parsed);
                if seen.contains(&hash) {
                    continue;
                }
                // MIN-5 fix: insert into `seen` immediately after computing the
                // hash, before the expensive L2/L3/L4 cascade.  Previously
                // `seen.insert` was called only on success, so a candidate that
                // fails cascade_filter would be re-parsed and re-rated on every
                // encounter in the same generation.
                seen.insert(hash);

                // Cascade filter (L1 + L2 + solve + L4).
                let entry = match cascade_filter(
                    &cand,
                    cfg.target_se,
                    seed_clue_count,
                    current.generation + 1,
                ) {
                    Some(e) => e,
                    None => continue,
                };
                found_any = true;

                if entry.se_score >= cfg.target_se {
                    outputs.push(entry.clone());
                    if outputs.len() >= cfg.max_outputs {
                        break;
                    }
                }

                frontier.push(HeapEntry {
                    se_score: entry.se_score,
                    entry,
                });
            }

            if !found_any && n == 1 {
                stall_n1 += 1;
            } else if found_any {
                stall_n1 = 0;
            }
        }
    }

    outputs
}

// ---------------------------------------------------------------------------
// Parquet writer for VicinityEntry outputs
// ---------------------------------------------------------------------------

/// Schema: puzzle utf8, solution utf8, clue_count int32, se_score float64,
/// tier utf8, frontier_json utf8 (stub), generation int32.
pub fn vicinity_schema() -> std::sync::Arc<arrow_schema::Schema> {
    use arrow_schema::{DataType, Field, Schema};
    std::sync::Arc::new(Schema::new(vec![
        Field::new("puzzle", DataType::Utf8, false),
        Field::new("solution", DataType::Utf8, false),
        Field::new("clue_count", DataType::Int32, false),
        Field::new("se_score", DataType::Float64, false),
        Field::new("tier", DataType::Utf8, false),
        Field::new("frontier_json", DataType::Utf8, false),
        Field::new("generation", DataType::Int32, false),
    ]))
}

fn tier_name(t: Tier) -> &'static str {
    match t {
        Tier::T1 => "T1",
        Tier::T2 => "T2",
        Tier::T3 => "T3",
        Tier::T4Plus => "T4Plus",
    }
}

/// Write a slice of `VicinityEntry` to a parquet file at `path`.
pub fn write_vicinity_parquet(path: &Path, entries: &[VicinityEntry]) -> std::io::Result<()> {
    use arrow_array::builder::{Float64Builder, Int32Builder, StringBuilder};
    use arrow_array::{ArrayRef, RecordBatch};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};

    let schema = vicinity_schema();
    let n = entries.len();

    let mut puzzle_b = StringBuilder::with_capacity(n, n * 82);
    let mut solution_b = StringBuilder::with_capacity(n, n * 82);
    let mut clue_b = Int32Builder::with_capacity(n);
    let mut se_b = Float64Builder::with_capacity(n);
    let mut tier_b = StringBuilder::with_capacity(n, n * 8);
    let mut frontier_b = StringBuilder::with_capacity(n, n * 4);
    let mut gen_b = Int32Builder::with_capacity(n);

    for e in entries {
        puzzle_b.append_value(&e.puzzle);
        solution_b.append_value(&e.solution);
        clue_b.append_value(clue_count(&e.puzzle) as i32);
        se_b.append_value(e.se_score);
        tier_b.append_value(tier_name(e.tier));
        frontier_b.append_value("[]");
        gen_b.append_value(e.generation as i32);
    }

    let cols: Vec<ArrayRef> = vec![
        std::sync::Arc::new(puzzle_b.finish()),
        std::sync::Arc::new(solution_b.finish()),
        std::sync::Arc::new(clue_b.finish()),
        std::sync::Arc::new(se_b.finish()),
        std::sync::Arc::new(tier_b.finish()),
        std::sync::Arc::new(frontier_b.finish()),
        std::sync::Arc::new(gen_b.finish()),
    ];

    let batch = RecordBatch::try_new(schema.clone(), cols)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    writer
        .close()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

/// Write manifest sidecar `<out>.manifest.json`.
pub fn write_vicinity_manifest(
    out_path: &Path,
    label: &str,
    seed_count: usize,
    output_count: usize,
) -> std::io::Result<()> {
    let manifest_path = {
        let mut p = out_path.to_path_buf();
        let ext = p.extension().unwrap_or_default().to_string_lossy().into_owned();
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        p.set_file_name(format!("{}.{}.manifest.json", stem, ext));
        p
    };
    let json = format!(
        r#"{{
  "label": {label},
  "seed_count": {seed_count},
  "output_count": {output_count}
}}"#,
        label = serde_json::to_string(label).unwrap(),
        seed_count = seed_count,
        output_count = output_count,
    );
    std::fs::write(&manifest_path, json)?;
    eprintln!("vicinity manifest: {}", manifest_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A known T3 puzzle from the top1465 set (magictour). 26 clues.
    const T3_SEED: &str = "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.";
    /// Solution for T3_SEED.
    const T3_SOL: &str =  "859612437723854169164379528986147352375268914241593786432981675617425893598736241";

    fn make_seed(puzzle: &str, solution: &str) -> VicinityEntry {
        let grid = Grid::<9, 3, 3>::from_str(puzzle).expect("parse seed");
        let r = rate::<9, 3, 3>(&grid);
        VicinityEntry {
            puzzle: puzzle.to_string(),
            solution: solution.to_string(),
            se_score: r.se_score,
            tier: r.tier,
            generation: 0,
        }
    }

    #[test]
    fn mutate_1_produces_different_puzzle() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let candidates = mutate_n(T3_SEED, 1, 8, &mut rng);
        assert!(!candidates.is_empty(), "mutate_n should produce at least one candidate");
        // All candidates must differ from original.
        for c in &candidates {
            assert_ne!(c, T3_SEED, "mutated puzzle should differ from original");
        }
    }

    #[test]
    fn vicinity_dedup_via_canonical_hash() {
        let seed = make_seed(T3_SEED, T3_SOL);
        let cfg = VicinityConfig {
            target_se: 1.0,
            bt_prefilter: 0,
            mute_set: vec![1],
            budget_iters: 10,
            max_outputs: 20,
            seed: 42,
        };
        let outputs = explore(vec![seed], &cfg);
        // All outputs must be pairwise canonical-distinct.
        let hashes: Vec<u128> = outputs
            .iter()
            .filter_map(|e| Grid::<9, 3, 3>::from_str(&e.puzzle))
            .map(|g| canonical_hash::<9, 3, 3>(&g))
            .collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(
                    hashes[i], hashes[j],
                    "dedup failure: outputs[{}] and outputs[{}] have same canonical hash",
                    i, j
                );
            }
        }
    }

    #[test]
    fn vicinity_explore_smoke() {
        let seed = make_seed(T3_SEED, T3_SOL);
        // Use target_se=1.5 so that T2+ neighbours (very common from swaps) are accepted.
        // The cascade filter passes anything with se_score >= 1.5 - 0.3 = 1.2 onto the
        // frontier, and emits anything >= 1.5. Most swap-mutations of a T3 seed produce
        // T2 or T1 puzzles; we lower the bar here so the smoke test reliably passes.
        let cfg = VicinityConfig {
            target_se: 1.5,
            bt_prefilter: 0,
            mute_set: vec![1, 2],
            budget_iters: 50,
            max_outputs: 10,
            seed: 7,
        };
        let outputs = explore(vec![seed], &cfg);
        // Must emit at least 1 output (T3 seed should find neighbours easily).
        assert!(
            !outputs.is_empty(),
            "vicinity_explore_smoke: expected at least 1 output, got 0"
        );
        // All outputs must have se_score >= 1.2 (target 1.5, filter passes >=1.2).
        for e in &outputs {
            assert!(
                e.se_score >= 1.2,
                "output se_score {} < 1.2",
                e.se_score
            );
        }
    }
}
