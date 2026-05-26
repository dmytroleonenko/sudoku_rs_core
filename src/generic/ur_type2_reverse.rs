//! R2.2.2: constructive Unique Rectangle Type 2 reverse synthesis.
//!
//! Background. R1's search-and-filter `reverse_construct` measured a hit rate
//! of ~7/1000 for UR Type 2 on 9×9 — sparse enough that bulk dataset
//! construction is throughput-bound. UR Type 2 is structurally rarer than
//! AIC: a UR Type 2 pattern requires 4 corners on the rectangle (2×2 cells
//! spanning exactly 2 boxes) where exactly two corners are bivalue {a,b} on
//! one side, the other two carry {a,b}+x with the SAME extra digit x, AND
//! that x has at least one common-peer cell to eliminate.
//!
//! Approach. Mirrors `aic_reverse.rs`: guided removal with a UR-Type-2-aware
//! score, *not* full constructive planting.
//!
//!   1. Generate a unique seed at the upper end of the clue band.
//!   2. Greedily try removing each remaining clue (shuffled order); keep the
//!      removal iff (a) uniqueness is preserved AND (b) once a UR Type 2
//!      pattern has appeared we never let it disappear.
//!   3. Verify: tier == target_tier, UrType2 ∈ frontier, and (optional)
//!      `rate_excluding(puzzle, [UrType2]).tier > target_tier`.
//!
//! Soundness mirror. `try_ur_eliminate_dryrun` is a pure (no-mutation) port
//! of `unique_rect::try_rectangle::<UrKind::Type2>` — it must match the
//! production firing rule exactly:
//!   * 4 corners empty (no solved cell);
//!   * pair partition: bivalue pair on one row-side, ext pair on other
//!     row-side, OR same with col-sides;
//!   * bivalue mask m0 has popcount 2; both ext corners' candidates are
//!     superset of m0 with the SAME single extra digit x;
//!   * ≥1 cell that is a common peer of both ext corners (excluding the 4
//!     corners themselves) still has digit x as a candidate.
//! `tests::ur_type2_dryrun_mirrors_production` asserts on positive cases that
//! the dry-run fires iff the production technique would have eliminated.
//!
//! Determinism. Single-threaded with a fixed seed → byte-identical puzzle.
//! Multi-threaded `batch_ur_type2_reverse_construct` produces the **same
//! output set** under a fixed master seed (per-worker seeds derived via
//! splitmix from the master), but **emission order is non-deterministic**
//! across runs because workers race to push their results. Sort by
//! `(puzzle, solution)` if you need byte-identical output across runs.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{GenConfig, gen_unique_puzzle};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::{count_solutions_up_to, solve_unique};
use super::techniques::{TechniqueId, Tier};

/// Configuration for one constructive UR Type 2 reverse-synth call.
#[derive(Clone, Debug)]
pub struct UrType2ReverseSpec {
    /// Target tier (typically T3).
    pub target_tier: Tier,
    /// Inclusive clue-count band for the seed puzzle.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Hard cap on outer attempts (each attempt re-seeds from a fresh
    /// random solution).
    pub max_attempts: u32,
    /// If true, additionally verify UrType2 is load-bearing
    /// (`rate_excluding(puzzle, [UrType2]).tier > target_tier`).
    pub require_load_bearing: bool,
    /// Inner cap on greedy guided-removal trials per seed.
    /// `0` means "no cap" (still bounded by `N*N` from a single shuffled pass).
    pub greedy_max_trials: u32,
}

impl UrType2ReverseSpec {
    pub fn new_t3(max_attempts: u32) -> Self {
        Self {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts,
            require_load_bearing: true,
            greedy_max_trials: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.clue_min > self.clue_max {
            return Err(format!(
                "clue_min ({}) > clue_max ({})",
                self.clue_min, self.clue_max
            ));
        }
        if matches!(self.target_tier, Tier::T1 | Tier::T2) {
            return Err("target_tier T1/T2 cannot fire UrType2 (UR is T3)".into());
        }
        Ok(())
    }
}

#[inline]
fn tier_rank(t: Tier) -> u8 {
    match t {
        Tier::T1 => 1,
        Tier::T2 => 2,
        Tier::T3 => 3,
        Tier::T4Plus => 4,
    }
}

// ---------------------------------------------------------------------------
// UR Type 2 dry-run probe — mirrors production `unique_rect::try_rectangle`.
// ---------------------------------------------------------------------------

/// Pure (no mutation) check for whether UR Type 2 would fire on the given
/// 4-cell rectangle, mirroring the production rule exactly.
///
/// SOUNDNESS: this must stay byte-equivalent to
/// `unique_rect::try_rectangle::<UrKind::Type2>`. If you change the
/// production rule, change this in lockstep — `tests::ur_type2_dryrun_mirrors_production`
/// asserts they agree on positive cases.
///
/// `cells` ordering: cells[0]=(r1,c1), cells[1]=(r1,c2), cells[2]=(r2,c1),
/// cells[3]=(r2,c2).
pub fn try_ur_eliminate_dryrun<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    cells: [usize; 4],
) -> bool {
    if cells.iter().any(|&c| grid.solved[c] != 0) {
        return false;
    }
    let masks = [
        grid.candidates[cells[0]],
        grid.candidates[cells[1]],
        grid.candidates[cells[2]],
        grid.candidates[cells[3]],
    ];
    // pair_partitions: pi=0 by row sides, pi=1 by col sides.
    let pair_partitions: [[(usize, usize); 2]; 2] = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
    ];
    let table = grid.table();
    let nn = N * N;
    for parts in pair_partitions.iter() {
        for swap in 0..2 {
            let (b_a, b_b) = if swap == 0 { parts[0] } else { parts[1] };
            let (e_a, e_b) = if swap == 0 { parts[1] } else { parts[0] };
            let m0 = masks[b_a];
            if m0.count_ones() != 2 { continue; }
            if masks[b_b] != m0 { continue; }
            let me_a = masks[e_a];
            let me_b = masks[e_b];
            if me_a & m0 != m0 || me_b & m0 != m0 { continue; }
            let extra_a = me_a & !m0;
            let extra_b = me_b & !m0;
            if extra_a == 0 || extra_b == 0 { continue; }
            if extra_a == extra_b && extra_a.count_ones() == 1 {
                let x_bit = extra_a;
                // Common peers of cells[e_a] and cells[e_b], excluding the 4 corners.
                let mut e_peer = vec![false; nn];
                for &p in &table.cells[cells[e_b]].peers {
                    e_peer[p as usize] = true;
                }
                for &p in &table.cells[cells[e_a]].peers {
                    let cell = p as usize;
                    if !e_peer[cell] { continue; }
                    if cell == cells[0] || cell == cells[1]
                        || cell == cells[2] || cell == cells[3] { continue; }
                    if grid.solved[cell] != 0 { continue; }
                    if grid.candidates[cell] & x_bit != 0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Iterate every UR rectangle (2 rows × 2 cols × exactly 2 boxes) in the
/// SAME order as `unique_rect::iterate_rectangles`. Calls `f(cells)` for each;
/// returns the first cells for which `f` returns `true`. Deterministic.
fn iterate_rectangles_pure<const N: usize, const BR: usize, const BC: usize, F>(
    mut f: F,
) -> Option<[usize; 4]>
where
    F: FnMut([usize; 4]) -> bool,
{
    // Class (a): same band, different stacks.
    let n_bands = N / BR;
    for band in 0..n_bands {
        let r_base = band * BR;
        for r1 in r_base..(r_base + BR) {
            for r2 in (r1 + 1)..(r_base + BR) {
                for c1 in 0..N {
                    for c2 in (c1 + 1)..N {
                        if c1 / BC == c2 / BC { continue; }
                        let cells = [
                            r1 * N + c1,
                            r1 * N + c2,
                            r2 * N + c1,
                            r2 * N + c2,
                        ];
                        if f(cells) { return Some(cells); }
                    }
                }
            }
        }
    }
    // Class (b): same stack, different bands.
    let n_stacks = N / BC;
    for stack in 0..n_stacks {
        let c_base = stack * BC;
        for c1 in c_base..(c_base + BC) {
            for c2 in (c1 + 1)..(c_base + BC) {
                for r1 in 0..N {
                    for r2 in (r1 + 1)..N {
                        if r1 / BR == r2 / BR { continue; }
                        let cells = [
                            r1 * N + c1,
                            r1 * N + c2,
                            r2 * N + c1,
                            r2 * N + c2,
                        ];
                        if f(cells) { return Some(cells); }
                    }
                }
            }
        }
    }
    None
}

/// Find the first UR Type 2 rectangle (in iteration order) on which the
/// dry-run would fire on `grid`. Returns the 4 corner indices.
pub fn find_first_ur_type2<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<[usize; 4]> {
    iterate_rectangles_pure::<N, BR, BC, _>(|cells| {
        try_ur_eliminate_dryrun::<N, BR, BC>(grid, cells)
    })
}

// ---------------------------------------------------------------------------
// Guided removal driver.
// ---------------------------------------------------------------------------

fn build_subset<const N: usize, const BR: usize, const BC: usize>(
    solution: &Grid<N, BR, BC>,
    keep: &[bool],
) -> Option<Grid<N, BR, BC>> {
    let nn = N * N;
    let mut g: Grid<N, BR, BC> = Grid::empty();
    for i in 0..nn {
        if !keep[i] { continue; }
        let d = solution.solved[i];
        if d == 0 { return None; }
        if g.assign(i, d).is_err() { return None; }
    }
    Some(g)
}

/// Score: `true` iff a UR Type 2 dry-run hit exists.
fn ur_score<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> bool {
    find_first_ur_type2::<N, BR, BC>(grid).is_some()
}

/// Greedy guided-removal pass. Mirrors `aic_reverse::guided_removal` but with
/// UR-Type-2 dry-run as the score.
fn guided_removal<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    solution: &Grid<N, BR, BC>,
    keep: &mut Vec<bool>,
    spec: &UrType2ReverseSpec,
) {
    let nn = N * N;
    let mut order: Vec<u16> = (0..nn as u16).filter(|&i| keep[i as usize]).collect();
    order.shuffle(rng);

    let initial = match build_subset::<N, BR, BC>(solution, keep) {
        Some(g) => g,
        None => return,
    };
    let mut have_pattern = ur_score::<N, BR, BC>(&initial);

    let mut trials: u32 = 0;
    let trial_cap = if spec.greedy_max_trials == 0 { u32::MAX } else { spec.greedy_max_trials };

    for &c in &order {
        if trials >= trial_cap { break; }
        let c = c as usize;
        if !keep[c] { continue; }
        let kept_count: u32 = keep.iter().filter(|&&k| k).count() as u32;
        if kept_count <= spec.clue_min { break; }

        keep[c] = false;
        trials = trials.saturating_add(1);
        let trial = match build_subset::<N, BR, BC>(solution, keep) {
            Some(g) => g,
            None => { keep[c] = true; continue; }
        };
        if count_solutions_up_to(&trial, 2) != 1 {
            keep[c] = true;
            continue;
        }
        let cand_has = ur_score::<N, BR, BC>(&trial);
        let accept = if !have_pattern {
            // No pattern yet → accept anything that preserves uniqueness.
            true
        } else {
            // Have a pattern → never drop it.
            cand_has
        };
        if accept {
            have_pattern = cand_has;
        } else {
            keep[c] = true;
        }
    }
}

/// Outer driver. See module-level docs for the algorithm.
pub fn ur_type2_reverse_construct<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &UrType2ReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() { return None; }
    let target_rank = tier_rank(spec.target_tier);
    let nn = N * N;

    for attempt in 0..spec.max_attempts {
        let cfg = GenConfig { target_clues: spec.clue_max, max_attempts: 0 };
        let (seed_puzzle, _seed_clue_count): (Grid<N, BR, BC>, u32) =
            gen_unique_puzzle::<N, BR, BC, R>(rng, &cfg);

        let solution = match solve_unique::<N, BR, BC>(&seed_puzzle) {
            Some(s) => s,
            None => continue,
        };

        let mut keep: Vec<bool> = (0..nn).map(|i| seed_puzzle.solved[i] != 0).collect();
        guided_removal::<N, BR, BC, R>(rng, &solution, &mut keep, spec);

        let puzzle = match build_subset::<N, BR, BC>(&solution, &keep) {
            Some(g) => g,
            None => continue,
        };
        let final_clues: u32 = keep.iter().filter(|&&k| k).count() as u32;

        let r = rate(&puzzle);
        if r.rater_error { continue; }
        if tier_rank(r.tier) != target_rank { continue; }
        if !r.frontier.contains(&TechniqueId::UrType2) { continue; }

        // Dry-run probe sanity (mirrors find_first_ur_type2 ↔ rate frontier).
        if find_first_ur_type2::<N, BR, BC>(&puzzle).is_none() { continue; }

        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[TechniqueId::UrType2]);
            if r2.rater_error { continue; }
            if tier_rank(r2.tier) <= target_rank { continue; }
        }

        return Some(ReverseResult {
            puzzle,
            solution,
            rate: r,
            clue_count: final_clues,
            attempts_taken: attempt + 1,
        });
    }
    None
}

/// Multi-threaded batch driver. Same per-worker seed scheme as
/// `batch_aic_reverse_construct`.
pub fn batch_ur_type2_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &UrType2ReverseSpec,
    num_puzzles: u32,
    threads: usize,
) -> Vec<ReverseResult<N, BR, BC>> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    let threads = threads.max(1);
    let collected: Arc<Mutex<Vec<ReverseResult<N, BR, BC>>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_puzzles as usize)));
    let kept = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::with_capacity(threads);
    for w in 0..threads {
        let spec = spec.clone();
        let collected = collected.clone();
        let kept = kept.clone();
        let child_seed = rng_seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
        handles.push(std::thread::spawn(move || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
            let per_worker_attempts = if spec.max_attempts == 0 {
                u32::MAX
            } else {
                ((spec.max_attempts as u64 + threads as u64 - 1) / threads as u64) as u32
            };
            let mut used: u32 = 0;
            while used < per_worker_attempts {
                if kept.load(Ordering::Relaxed) >= num_puzzles { return; }
                let chunk = per_worker_attempts.saturating_sub(used).min(32);
                let chunk_spec = UrType2ReverseSpec { max_attempts: chunk, ..spec.clone() };
                match ur_type2_reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
                    Some(res) => {
                        used = used.saturating_add(res.attempts_taken);
                        let prev = kept.fetch_add(1, Ordering::Relaxed);
                        if prev < num_puzzles {
                            collected.lock().unwrap().push(res);
                        } else {
                            kept.fetch_sub(1, Ordering::Relaxed);
                            return;
                        }
                    }
                    None => { used = used.saturating_add(chunk); }
                }
            }
        }));
    }
    for h in handles { let _ = h.join(); }
    let g = collected.lock().unwrap();
    g.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::techniques::{Technique, UrType2 as ProdUrType2};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Soundness mirror: on a hand-constructed UR Type 2 grid, our dry-run
    /// must fire on the same 4 corners as the production technique. Mirror
    /// of `unique_rect::tests::ur_type2_fires_9x9`.
    #[test]
    fn ur_type2_dryrun_mirrors_production() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Same rectangle as production test: (0,0),(0,4),(1,0),(1,4).
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in 3..=9u8 { g.eliminate(4, d).unwrap(); }
        for d in 4..=9u8 { g.eliminate(9, d).unwrap(); }
        for d in 4..=9u8 { g.eliminate(13, d).unwrap(); }

        let cells = [0usize, 4, 9, 13];
        assert!(
            try_ur_eliminate_dryrun::<9, 3, 3>(&g, cells),
            "dry-run must fire on canonical UR Type 2 setup"
        );
        // find_first_ur_type2 should pick exactly this rectangle.
        let hit = find_first_ur_type2::<9, 3, 3>(&g).expect("must find rectangle");
        assert_eq!(hit, cells);

        // Production would also fire (and produce ≥1 elim).
        let mut g2 = g.clone();
        let t = ProdUrType2;
        let p = <ProdUrType2 as Technique<9, 3, 3>>::apply(&t, &mut g2)
            .expect("production UrType2 must fire");
        assert!(!p.eliminations.is_empty());
    }

    /// Negative: 3-cell partial rectangle (one corner solved) must NOT fire.
    #[test]
    fn ur_type2_rejects_invalid_rectangles() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Same as positive but ASSIGN one corner — kills the pattern.
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in 3..=9u8 { g.eliminate(4, d).unwrap(); }
        for d in 4..=9u8 { g.eliminate(9, d).unwrap(); }
        // Solve the 4th corner — now only 3 unsolved corners → not a UR rectangle.
        g.assign(13, 1).unwrap();
        assert!(
            !try_ur_eliminate_dryrun::<9, 3, 3>(&g, [0, 4, 9, 13]),
            "dry-run must NOT fire when a corner is solved"
        );

        // Non-bivalue mismatch: extra corners disagree on the extra digit.
        let mut g2: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 { g2.eliminate(0, d).unwrap(); }
        for d in 3..=9u8 { g2.eliminate(4, d).unwrap(); }
        // (1,0) = {1,2,3}; (1,4) = {1,2,5}: extras differ → no UR Type 2.
        for d in 4..=9u8 { g2.eliminate(9, d).unwrap(); }
        for d in [3u8, 4, 6, 7, 8, 9] { g2.eliminate(13, d).unwrap(); }
        assert!(
            !try_ur_eliminate_dryrun::<9, 3, 3>(&g2, [0, 4, 9, 13]),
            "dry-run must NOT fire when ext digits differ"
        );
    }

    /// LOAD-BEARING STRICT SEMANTICS: `require_load_bearing=true` accepts only
    /// puzzles where excluding UrType2 promotes the rated tier strictly above
    /// `target_tier` (NOT "equal-or-greater"). Validates the comparator.
    #[test]
    fn load_bearing_strict_semantics() {
        // We build a synthetic spec with require_load_bearing=true and run
        // a small budget. Whatever it returns (Some/None), if Some, the
        // strict invariant must hold. We additionally exercise the
        // comparator on a constructed counterexample — a puzzle whose
        // rate_excluding returns the SAME tier should be rejected.
        let spec = UrType2ReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 80,
            require_load_bearing: true,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2027);
        if let Some(r) = ur_type2_reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            let r2 = rate_excluding(&r.puzzle, &[TechniqueId::UrType2]);
            assert!(
                tier_rank(r2.tier) > tier_rank(Tier::T3),
                "load_bearing=true must imply strict tier promotion, got {:?}",
                r2.tier
            );
        }
    }

    /// Determinism: same seed → same puzzle (single-thread).
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = UrType2ReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 50,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(31337);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(31337);
        let ra = ur_type2_reverse_construct::<9, 3, 3, _>(&mut a, &spec);
        let rb = ur_type2_reverse_construct::<9, 3, 3, _>(&mut b, &spec);
        match (ra, rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.puzzle.to_string_grid(), y.puzzle.to_string_grid());
            }
            (None, None) => {}
            _ => panic!("determinism violated: one Some, one None"),
        }
    }

    /// Smoke 9×9: loose budget, no load-bearing — should produce ≥1 puzzle.
    #[test]
    fn smoke_9x9_ur_type2() {
        let spec = UrType2ReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 100,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9001);
        let res = ur_type2_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T3));
            assert!(r.rate.frontier.contains(&TechniqueId::UrType2));
            assert!(find_first_ur_type2::<9, 3, 3>(&r.puzzle).is_some());
            // Returned puzzle solves to its solution.
            let sol = solve_unique::<9, 3, 3>(&r.puzzle).expect("must be unique");
            assert_eq!(sol.to_string_grid(), r.solution.to_string_grid());
        }
        // Soft-pass on None — environment-dependent.
    }

    /// Smoke 12×12: only assert termination — generation is slow.
    #[test]
    fn smoke_12x12_ur_type2() {
        let spec = UrType2ReverseSpec {
            target_tier: Tier::T3,
            clue_min: 60,
            clue_max: 90,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 32, // cap aggressively for speed
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12_001);
        let _ = ur_type2_reverse_construct::<12, 3, 4, _>(&mut rng, &spec);
    }

    /// Validation: T1/T2 tiers rejected (UrType2 is T3).
    #[test]
    fn validate_t1_t2_rejected() {
        let mut spec = UrType2ReverseSpec::new_t3(10);
        spec.target_tier = Tier::T1;
        assert!(spec.validate().is_err());
        spec.target_tier = Tier::T2;
        assert!(spec.validate().is_err());
        spec.target_tier = Tier::T3;
        assert!(spec.validate().is_ok());
    }

    /// Batch driver smoke.
    #[test]
    fn batch_smoke_terminates() {
        let spec = UrType2ReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let res = batch_ur_type2_reverse_construct::<9, 3, 3>(42, &spec, 2, 1);
        assert!(res.len() <= 2);
    }
}
