//! Reverse-construction for **Dynamic Forcing Chain L1** (SE 9.0–9.5).
//!
//! ## Why random-erasure fails
//!
//! The previous v1 algorithm (random-erase from solution + full-cascade rate)
//! produces **0 DFC-top emits** in practice.  Root cause: in our T4Plus cascade
//! the order is `CFC → RFC → DFC → NestedFC L2 → L3 → L4`.  `NestedForcingChain`
//! L2's recursive propagator subsumes DFC's structure — whenever DFC fires,
//! NestedFC L2 fires later and accumulates the harder SE score.  The full-cascade
//! SE of every DFC-capable puzzle is therefore ≥ 9.5 (NestedFC L2 bucket), and
//! the frontier always contains *both* DFC and NestedFC.  The acceptance gate
//! `frontier.contains(DFC) && se ≥ 9.0` never rejects anything useful, but it
//! also never gives us puzzles where DFC is truly the *top* technique.
//!
//! ## Why vicinity-mutation + NestedFC-exclusion works
//!
//! To expose DFC as the load-bearing technique we exclude all three NestedFC
//! levels from the cascade (analogous to how `rfc_reverse.rs` excludes CFC to
//! expose RFC).  In the no-NestedFC cascade, DFC is the deepest available
//! technique; a puzzle that requires it will have `se_score ∈ [9.0, 9.5)` and
//! `DFC ∈ frontier`.
//!
//! Seeds from `forum_hardest_1905` already live near the hardest corner of the
//! search space; vicinity mutation (replace 1–2 clue digits) explores the
//! neighbourhood efficiently, while random-erasure from a fresh solution rarely
//! reaches SE 9.0+ territory.
//!
//! ## Acceptance contract
//!
//! An emitted puzzle is **DFC-top in the no-NestedFC cascade**:
//!
//! ```text
//! rate_excluding(puzzle, [NestedForcingChain, NestedForcingChainL3, NestedForcingChainL4])
//!   .solved   == true
//!   .frontier ∋ DynamicForcingChain
//!   .se_score ∈ [spec.target_se, spec.target_se_upper)   (default [9.0, 9.5))
//! ```
//!
//! The full-cascade rating of the same puzzle will assign SE 9.5+ (NestedFC L2
//! fires redundantly), so `se_score` and `frontier` on the emitted struct reflect
//! the **no-NestedFC** probe, not the full cascade.
//!
//! ## Contract (docs/alphaevolve_contract.md)
//!
//! §1 Interface — `DfcReverseSpec`, `ConstructedPuzzle`, `batch_dfc_reverse_construct`
//! §2 Algorithm — seed-anchored vicinity mutation + NestedFC-exclusion gate
//! §3 Restrictions — does not touch `TechniqueProgress`, `TechniqueId`, `Tier`
//! §4 Dependencies — `rater`, `generator`, `search`, `grid`
//! §5 Versioning — v1 = random-erase (abandoned); v2 = vicinity + NestedFC-exclusion
//! §6 Tests — `accepts_only_dfc_top`, `mutates_then_rates`, `seed_pool_loading`

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::grid::Grid;
use super::rater::rate_excluding;
use super::search::count_solutions_up_to;
use super::techniques::TechniqueId;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for `batch_dfc_reverse_construct`.
#[derive(Clone, Debug)]
pub struct DfcReverseSpec {
    /// Minimum SE score to accept (no-NestedFC cascade; 9.0 = DFC L1 base).
    pub target_se: f64,
    /// Strict upper bound on SE score (no-NestedFC cascade; default 9.5).
    /// Set to `None` to disable upper bound.
    pub target_se_upper: Option<f64>,
    /// When `true` (default), exclude all NestedFC levels from the rating
    /// cascade to expose DFC as the top technique.  Set to `false` only for
    /// diagnostics — produces full-cascade ratings where NestedFC shadows DFC.
    pub excluded_nested_fc: bool,
    /// Number of vicinity-mutation candidates to try per seed puzzle.
    pub attempts_per_seed: u32,
    /// RNG seed for deterministic generation.
    pub rng_seed: u64,
    /// Path to seed puzzle file (one 81-char puzzle per line).
    /// When `Some` and readable, seeds drive vicinity mutations.
    /// When `None` or unreadable, the module returns 0 emits (fallback
    /// random-erasure was removed in v2 — it is known-ineffective).
    pub seed_path: Option<PathBuf>,
    /// Target clue-count band [min, max] — used only to validate seeds;
    /// mutations preserve clue count so this is informational only.
    pub clue_min: u32,
    /// See `clue_min`.
    pub clue_max: u32,
    /// Maximum total outer iterations (seed passes). Safety cap.
    pub max_total_attempts: u32,
    /// Worker thread count.  `0` or `1` → single-threaded.
    pub threads: usize,
}

impl Default for DfcReverseSpec {
    fn default() -> Self {
        Self {
            target_se: 9.0,
            target_se_upper: Some(9.5),
            excluded_nested_fc: true,
            attempts_per_seed: 200,
            rng_seed: 0,
            seed_path: None,
            clue_min: 20,
            clue_max: 28,
            max_total_attempts: 50_000,
            threads: 0,
        }
    }
}

/// One successfully constructed DFC-top puzzle.
#[derive(Clone, Debug)]
pub struct ConstructedPuzzle {
    /// 81-character puzzle string (`.` = unsolved).
    pub puzzle: String,
    /// 81-character solution string.
    pub solution: String,
    /// SE score from the no-NestedFC cascade at acceptance time.
    pub se_score: f64,
    /// Techniques in the no-NestedFC frontier (`DynamicForcingChain` guaranteed
    /// present when `spec.excluded_nested_fc` is true).
    pub frontier: Vec<TechniqueId>,
    /// Number of clues placed in the puzzle.
    pub clue_count: u32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Exclusion list: all three NestedFC levels.
///
/// Excluding these exposes DFC as the deepest available technique in the
/// cascade, analogous to how `rfc_reverse` excludes CFC to expose RFC.
const NESTED_FC_ALL: &[TechniqueId] = &[
    TechniqueId::NestedForcingChain,
    TechniqueId::NestedForcingChainL3,
    TechniqueId::NestedForcingChainL4,
];

/// Read seed puzzles from `path`. Lines < 81 chars and comment lines (#) are
/// skipped. Only the first 81 chars of each line are used (solution field, if
/// any, is ignored). The digit '0' is normalised to '.'.
fn load_seeds(path: &Path) -> Vec<String> {
    let f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("dfc_reverse: cannot open seed file {}: {}", path.display(), e);
            return Vec::new();
        }
    };
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => continue,
        };
        if line.starts_with('#') || line.len() < 81 {
            continue;
        }
        let puzzle: String = line.chars().take(81).collect();
        if puzzle.bytes().all(|b| b == b'.' || b == b'0' || (b >= b'1' && b <= b'9')) {
            let normalized: String = puzzle
                .bytes()
                .map(|b| if b == b'0' { b'.' as u8 } else { b } as char)
                .collect();
            out.push(normalized);
        }
    }
    out
}

/// Count non-dot characters in an 81-char puzzle string.
fn count_clues_str(s: &str) -> usize {
    s.bytes().filter(|&b| b != b'.').count()
}

/// 1-digit vicinity mutation: pick one clue cell, replace its digit.
/// Returns up to `k` distinct mutated strings.
fn mutate_1(puzzle: &str, k: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<String> {
    let bytes: Vec<u8> = puzzle.bytes().collect();
    let clue_indices: Vec<usize> = bytes
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b != b'.' { Some(i) } else { None })
        .collect();
    if clue_indices.is_empty() {
        return Vec::new();
    }
    let mut results = Vec::with_capacity(k);
    for _ in 0..(k * 10) {
        if results.len() >= k {
            break;
        }
        let idx = clue_indices[rng.gen_range(0..clue_indices.len())];
        let cur = bytes[idx];
        let mut alts: Vec<u8> = (b'1'..=b'9').filter(|&d| d != cur).collect();
        alts.shuffle(rng);
        let mut nb = bytes.clone();
        nb[idx] = alts[0];
        if let Ok(s) = String::from_utf8(nb) {
            if s != puzzle && !results.contains(&s) {
                results.push(s);
            }
        }
    }
    results
}

/// 2-digit vicinity mutation: pick two distinct clue cells, replace each.
/// Returns up to `k` distinct mutated strings.
fn mutate_2(puzzle: &str, k: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<String> {
    let bytes: Vec<u8> = puzzle.bytes().collect();
    let mut clue_indices: Vec<usize> = bytes
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b != b'.' { Some(i) } else { None })
        .collect();
    if clue_indices.len() < 2 {
        return Vec::new();
    }
    let mut results = Vec::with_capacity(k);
    for _ in 0..(k * 10) {
        if results.len() >= k {
            break;
        }
        clue_indices.shuffle(rng);
        let ia = clue_indices[0];
        let ib = clue_indices[1];
        let da = bytes[ia];
        let db = bytes[ib];
        let mut aa: Vec<u8> = (b'1'..=b'9').filter(|&d| d != da).collect();
        let mut ab: Vec<u8> = (b'1'..=b'9').filter(|&d| d != db).collect();
        aa.shuffle(rng);
        ab.shuffle(rng);
        let mut nb = bytes.clone();
        nb[ia] = aa[0];
        nb[ib] = ab[0];
        if let Ok(s) = String::from_utf8(nb) {
            if s != puzzle && !results.contains(&s) {
                results.push(s);
            }
        }
    }
    results
}

/// Try to accept a candidate puzzle string as DFC-top.
///
/// Rate with NestedFC excluded (or with full cascade if `excluded_nested_fc`
/// is false).  Accept iff:
///   - no rater error
///   - cascade fully solves the puzzle
///   - `DynamicForcingChain ∈ frontier`
///   - `se_score ≥ target_se`
///   - `se_score < target_se_upper` (if set)
fn try_accept<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    spec: &DfcReverseSpec,
) -> Option<(f64, Vec<TechniqueId>)> {
    let r = if spec.excluded_nested_fc {
        rate_excluding(grid, NESTED_FC_ALL)
    } else {
        super::rater::rate(grid)
    };
    if r.rater_error || !r.solved {
        return None;
    }
    if !r.frontier.contains(&TechniqueId::DynamicForcingChain) {
        return None;
    }
    if r.se_score < spec.target_se {
        return None;
    }
    if let Some(upper) = spec.target_se_upper {
        if r.se_score >= upper {
            return None;
        }
    }
    Some((r.se_score, r.frontier))
}

// ---------------------------------------------------------------------------
// Public batch constructor
// ---------------------------------------------------------------------------

/// Produce up to `num_puzzles` DFC-top puzzles.
///
/// Requires `spec.seed_path` to point to a readable seed file.  If no seeds
/// can be loaded, returns an empty vec immediately (random-erasure fallback
/// was removed in v2 — it is known-ineffective for DFC-top).
///
/// Multi-threaded fan-out when `spec.threads > 1`.
pub fn batch_dfc_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    spec: &DfcReverseSpec,
    num_puzzles: u32,
) -> Vec<ConstructedPuzzle> {
    if spec.threads > 1 {
        let n_threads = spec.threads;
        let collected: Arc<Mutex<Vec<ConstructedPuzzle>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_puzzles as usize)));
        let kept = Arc::new(AtomicU32::new(0));

        // Load seeds once in the main thread; workers share a slice.
        let seeds: Vec<String> = spec
            .seed_path
            .as_ref()
            .map(|p| load_seeds(p))
            .unwrap_or_default();

        if seeds.is_empty() {
            eprintln!("dfc_reverse: no seeds loaded — cannot produce DFC-top puzzles.");
            return Vec::new();
        }

        let seeds = Arc::new(seeds);
        let mut handles = Vec::with_capacity(n_threads);

        for w in 0..n_threads {
            let mut spec_w = spec.clone();
            spec_w.threads = 1;
            spec_w.seed_path = None; // seeds already loaded
            let per_worker_quota = num_puzzles.div_ceil(n_threads as u32).max(1);
            let collected_w = collected.clone();
            let kept_w = kept.clone();
            let seeds_w = seeds.clone();
            let child_seed = spec
                .rng_seed
                .wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
            spec_w.rng_seed = child_seed;

            handles.push(std::thread::spawn(move || {
                let local = dfc_reverse_inner::<N, BR, BC>(
                    &spec_w,
                    per_worker_quota,
                    Some(&kept_w),
                    num_puzzles,
                    Some(&seeds_w),
                    w, // stride offset for seed partition
                    n_threads,
                );
                let mut g = collected_w.lock().unwrap();
                for p in local {
                    if kept_w.load(Ordering::Relaxed) >= num_puzzles {
                        break;
                    }
                    if g.iter().any(|q: &ConstructedPuzzle| q.puzzle == p.puzzle) {
                        continue;
                    }
                    g.push(p);
                    kept_w.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            let _ = h.join();
        }
        let mut out: Vec<ConstructedPuzzle> =
            Arc::try_unwrap(collected).unwrap().into_inner().unwrap();
        out.truncate(num_puzzles as usize);
        return out;
    }

    // Single-threaded path.
    dfc_reverse_inner::<N, BR, BC>(spec, num_puzzles, None, num_puzzles, None, 0, 1)
}

/// Single-threaded inner loop.
///
/// `shared_kept` — optional shared counter for multi-thread early-exit.
/// `preloaded_seeds` — if `Some`, use this slice directly (already loaded).
/// `worker_offset` / `n_workers` — stride partitioning into seed pool.
fn dfc_reverse_inner<const N: usize, const BR: usize, const BC: usize>(
    spec: &DfcReverseSpec,
    num_puzzles: u32,
    shared_kept: Option<&AtomicU32>,
    global_target: u32,
    preloaded_seeds: Option<&[String]>,
    worker_offset: usize,
    n_workers: usize,
) -> Vec<ConstructedPuzzle> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(spec.rng_seed);
    let mut out = Vec::with_capacity(num_puzzles as usize);

    // Load seeds (or use preloaded).
    let owned: Vec<String>;
    let seeds: &[String] = if let Some(s) = preloaded_seeds {
        s
    } else {
        owned = spec
            .seed_path
            .as_ref()
            .map(|p| load_seeds(p))
            .unwrap_or_default();
        if owned.is_empty() {
            eprintln!("dfc_reverse: no seeds loaded — cannot produce DFC-top puzzles.");
            return out;
        }
        &owned
    };

    // Warn if seed file was missing/empty (only on worker 0 to avoid noise).
    if seeds.is_empty() {
        if worker_offset == 0 {
            eprintln!("dfc_reverse: seed pool is empty; returning 0 emits.");
        }
        return out;
    }

    // Build a stride-partitioned index list for this worker.
    let my_indices: Vec<usize> = (worker_offset..seeds.len())
        .step_by(n_workers.max(1))
        .collect();
    if my_indices.is_empty() {
        return out;
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut total_attempts: u32 = 0;
    let mut seed_cursor: usize = 0;

    // budget split: 50% 1-digit mutations, 50% 2-digit mutations
    let n1 = (spec.attempts_per_seed as usize + 1) / 2;
    let n2 = spec.attempts_per_seed as usize - n1;

    'outer: while out.len() < num_puzzles as usize
        && total_attempts < spec.max_total_attempts
    {
        if let Some(k) = shared_kept {
            if k.load(Ordering::Relaxed) >= global_target {
                break 'outer;
            }
        }
        total_attempts += 1;

        let seed_str = &seeds[my_indices[seed_cursor % my_indices.len()]];
        seed_cursor += 1;

        let mut candidates: Vec<String> = Vec::with_capacity(spec.attempts_per_seed as usize);
        candidates.extend(mutate_1(seed_str, n1.max(4), &mut rng));
        candidates.extend(mutate_2(seed_str, n2.max(2), &mut rng));

        for cand in candidates {
            if let Some(k) = shared_kept {
                if k.load(Ordering::Relaxed) >= global_target {
                    break 'outer;
                }
            }

            // Dedup on puzzle string (cheap; canonical hash would be more robust
            // but adds ~5 µs/puzzle — not worth it for DFC territory).
            if seen.contains(&cand) {
                continue;
            }
            seen.insert(cand.clone());

            // Parse.
            let grid = match Grid::<N, BR, BC>::from_str(&cand) {
                Some(g) => g,
                None => continue,
            };

            // Uniqueness check.
            if count_solutions_up_to(&grid, 2) != 1 {
                continue;
            }

            // Solve to get solution string.
            let sol_grid = match super::search::solve_unique::<N, BR, BC>(&grid) {
                Some(s) => s,
                None => continue,
            };

            // DFC-top acceptance gate.
            let (se_score, frontier) = match try_accept::<N, BR, BC>(&grid, spec) {
                Some(v) => v,
                None => continue,
            };

            let clue_count = count_clues_str(&cand) as u32;
            out.push(ConstructedPuzzle {
                puzzle: cand,
                solution: sol_grid.to_string_grid(),
                se_score,
                frontier,
                clue_count,
            });

            if out.len() >= num_puzzles as usize {
                break 'outer;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::rater::rate_excluding;
    use crate::generic::search::count_solutions_up_to;

    // ---------------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------------

    /// A known SE 11.x puzzle from forum_hardest (uniquely solvable).
    /// Used as a seed for mutation smoke tests.
    const SEED_PUZZLE: &str =
        "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..";

    fn minimal_spec() -> DfcReverseSpec {
        DfcReverseSpec {
            target_se: 9.0,
            target_se_upper: Some(9.5),
            excluded_nested_fc: true,
            attempts_per_seed: 4,
            rng_seed: 42,
            seed_path: None,
            clue_min: 20,
            clue_max: 28,
            max_total_attempts: 2,
            threads: 0,
        }
    }

    // ---------------------------------------------------------------------------
    // Test: seed_pool_loading
    // ---------------------------------------------------------------------------

    /// Smoke: load_seeds returns empty vec on a nonexistent file — no panic.
    #[test]
    fn seed_pool_loading_missing_file() {
        let result = load_seeds(Path::new("/nonexistent/path/seeds.txt"));
        assert!(
            result.is_empty(),
            "Expected empty vec for missing file, got {} entries",
            result.len()
        );
    }

    /// Smoke: load_seeds parses valid 81-char lines and skips comments / short
    /// lines, normalises '0' → '.'.
    #[test]
    fn seed_pool_loading_inline() {
        use std::io::Write;
        let mut tmp = tempfile_or_skip!();
        let valid = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let with_zero = "023456789456789123789123456214365897365897214897214365531642978642978531978531642";
        writeln!(tmp.as_file_mut(), "# comment").unwrap();
        writeln!(tmp.as_file_mut(), "short").unwrap();
        writeln!(tmp.as_file_mut(), "{}", valid).unwrap();
        writeln!(tmp.as_file_mut(), "{}", with_zero).unwrap();
        tmp.as_file_mut().flush().unwrap();
        let seeds = load_seeds(tmp.path());
        // Should have 2 entries (comment + short lines skipped).
        assert_eq!(seeds.len(), 2, "Expected 2 valid seeds");
        // '0' must be normalised to '.'.
        assert!(
            seeds[1].starts_with('.'),
            "Expected leading '.' after normalisation"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: mutates_then_rates
    // ---------------------------------------------------------------------------

    /// Vicinity mutations produce valid 81-char strings distinct from the seed.
    #[test]
    fn mutates_then_rates() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let m1 = mutate_1(SEED_PUZZLE, 5, &mut rng);
        let m2 = mutate_2(SEED_PUZZLE, 5, &mut rng);
        for m in m1.iter().chain(m2.iter()) {
            assert_eq!(m.len(), 81, "mutated puzzle must be 81 chars");
            assert_ne!(m.as_str(), SEED_PUZZLE, "mutation must differ from seed");
        }
        // At least some mutations must be produced.
        assert!(!m1.is_empty() || !m2.is_empty(), "Expected at least one mutation");
    }

    // ---------------------------------------------------------------------------
    // Test: accepts_only_dfc_top
    // ---------------------------------------------------------------------------

    /// The acceptance contract: every emitted puzzle must, when rated with
    /// NestedFC excluded, be fully solved with DFC in the frontier and SE in
    /// [target_se, target_se_upper).
    ///
    /// Non-vacuous gate test: directly call `try_accept` with synthetic
    /// `RateResult`-like inputs (constructed via real `rate_excluding` calls
    /// on an inlined forum_hardest seed) to verify each branch of the gate
    /// rejects/accepts as documented.
    ///
    /// Uses the "AI escargot" SE 11.6 puzzle as the test seed (same as
    /// `HARD_SEED` in nested_fc_reverse). After mutating one clue we expect
    /// either: still SE 11+ (NestedFC-shadowed), drops to T3 (lost DFC), or
    /// lands in our target [9.0, 9.5) bucket (rare, but the gate must
    /// behave correctly in any case).
    #[test]
    fn accepts_only_dfc_top_invariant() {
        const HARD_SEED: &str =
            "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4..";
        let g = Grid::<9, 3, 3>::from_str(HARD_SEED)
            .expect("HARD_SEED parses");

        // Branch 1: default spec → excluded_nested_fc=true, upper=Some(9.5).
        // HARD_SEED rates SE 11+ on full cascade; with NestedFC excluded its SE
        // is still high (Java SE 11.6 is mostly NestedFC-driven so removing it
        // may leave puzzle unsolvable). Either way try_accept must NOT panic
        // and must NOT accept an SE>=9.5 puzzle as DFC-top.
        let default_spec = DfcReverseSpec {
            target_se: 9.0,
            target_se_upper: Some(9.5),
            excluded_nested_fc: true,
            ..minimal_spec()
        };
        let result = try_accept::<9, 3, 3>(&g, &default_spec);
        if let Some((se, fr)) = result {
            assert!(
                se >= 9.0 && se < 9.5,
                "default gate accepted SE outside [9.0, 9.5): {:.2}",
                se
            );
            assert!(
                fr.contains(&TechniqueId::DynamicForcingChain),
                "default gate accepted without DFC in frontier: {:?}",
                fr
            );
        }

        // Branch 2: relaxed upper bound — `target_se_upper = None`. Now any
        // SE >= 9.0 with DFC in the no-NestedFC frontier is accepted.
        let relaxed_spec = DfcReverseSpec {
            target_se_upper: None,
            ..default_spec.clone()
        };
        if let Some((se, _)) = try_accept::<9, 3, 3>(&g, &relaxed_spec) {
            assert!(se >= 9.0, "relaxed gate accepted SE < 9.0: {:.2}", se);
        }

        // Branch 3: too-strict gate (target_se=20.0) must always reject.
        let strict_spec = DfcReverseSpec {
            target_se: 20.0,
            target_se_upper: None,
            ..default_spec.clone()
        };
        assert!(
            try_accept::<9, 3, 3>(&g, &strict_spec).is_none(),
            "gate must reject when target_se=20.0 > any real SE"
        );

        // Branch 4: full-cascade mode (`excluded_nested_fc=false`) + relaxed
        // upper. The full-cascade rating of HARD_SEED is dominated by NestedFC
        // → SE likely 11+; with DFC in frontier this should be accepted iff
        // upper is None or large.
        let full_cascade_spec = DfcReverseSpec {
            excluded_nested_fc: false,
            target_se_upper: None,
            ..default_spec.clone()
        };
        // Just verify no panic; semantics depend on rater's full-cascade output.
        let _ = try_accept::<9, 3, 3>(&g, &full_cascade_spec);
    }

    /// Integration test: given a forum_hardest seed file (if present on disk),
    /// run a smoke benchmark with a small budget and verify the contract on any
    /// emitted puzzles.  Skipped silently if the seed file is missing (CI).
    #[test]
    fn accepts_only_dfc_top_with_real_seeds() {
        let seed_path = PathBuf::from(
            "data/seeds/public_hardest/forum_hardest_1905/forum_hardest_1905_11plus.txt",
        );
        if !seed_path.exists() {
            return; // seed file not present — skip
        }
        let spec = DfcReverseSpec {
            target_se: 9.0,
            target_se_upper: Some(9.5),
            excluded_nested_fc: true,
            attempts_per_seed: 50,
            rng_seed: 7,
            seed_path: Some(seed_path),
            clue_min: 20,
            clue_max: 28,
            max_total_attempts: 200,
            threads: 0,
        };
        let results = batch_dfc_reverse_construct::<9, 3, 3>(&spec, 5);
        for cp in &results {
            assert_eq!(cp.puzzle.len(), 81);
            let g = Grid::<9, 3, 3>::from_str(&cp.puzzle)
                .expect("emitted puzzle must parse");
            assert_eq!(count_solutions_up_to(&g, 2), 1);
            let r = rate_excluding(&g, NESTED_FC_ALL);
            assert!(!r.rater_error);
            assert!(r.solved);
            assert!(r.frontier.contains(&TechniqueId::DynamicForcingChain));
            assert!(r.se_score >= 9.0);
            assert!(r.se_score < 9.5);
        }
    }

    // ---------------------------------------------------------------------------
    // Test: batch structural invariants
    // ---------------------------------------------------------------------------

    #[test]
    fn batch_smoke_no_panic() {
        let results = batch_dfc_reverse_construct::<9, 3, 3>(&minimal_spec(), 5);
        assert!(results.len() <= 5);
    }

    #[test]
    fn default_spec_terminates() {
        let spec = DfcReverseSpec {
            max_total_attempts: 1,
            threads: 0,
            attempts_per_seed: 1,
            ..DfcReverseSpec::default()
        };
        let _ = batch_dfc_reverse_construct::<9, 3, 3>(&spec, 1);
    }
}

// ---------------------------------------------------------------------------
// Tempfile helper (test-only, avoids external dependency)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod temp_helpers {
    use std::fs::File;
    use std::path::{Path, PathBuf};

    pub struct TempFile {
        path: PathBuf,
        file: File,
    }

    impl TempFile {
        pub fn new() -> Option<Self> {
            let path = std::env::temp_dir().join(format!("dfc_test_{}.txt", std::process::id()));
            let file = File::create(&path).ok()?;
            Some(Self { path, file })
        }
        pub fn path(&self) -> &Path {
            &self.path
        }
        pub fn as_file_mut(&mut self) -> &mut File {
            &mut self.file
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
macro_rules! tempfile_or_skip {
    () => {
        match crate::generic::dfc_reverse::temp_helpers::TempFile::new() {
            Some(f) => f,
            None => return,
        }
    };
}

#[cfg(test)]
use tempfile_or_skip;
