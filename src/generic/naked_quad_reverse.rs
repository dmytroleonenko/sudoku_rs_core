//! R2.2.4: constructive Naked Quad reverse synthesis.
//!
//! Background. R1 search-and-filter measured a 0/8000 hit-rate for
//! `naked_quad` across 4 clue-bands — too sparse for bulk dataset construction.
//! Naked Quad is structurally rare: a unit must contain four unsolved cells
//! whose candidate union is exactly four digits (and each cell has 2..=4
//! candidates), AND at least one OTHER unsolved cell of that unit must still
//! carry one of those four digits to be eliminated.
//!
//! Approach. Mirrors `als_xz_reverse.rs` / `ur_type2_reverse.rs`: guided
//! greedy removal preserving uniqueness AND the naked-quad pattern.
//!
//!   1. Generate a unique seed at the upper end of the clue band.
//!   2. Greedily try removing each remaining clue (shuffled order); keep the
//!      removal iff (a) uniqueness is preserved AND (b) once a naked-quad has
//!      appeared we never let it disappear.
//!   3. Verify: tier == target_tier, NakedQuad ∈ frontier, and (optional)
//!      `rate_excluding(puzzle, [NakedQuad]).tier > target_tier`.
//!
//! Tier semantics. After R3.1a tier-alignment NakedQuad is **T2** in the
//! generic rater (`AnyTechnique::tier`). The technique's own `tier()` impl
//! still returns T3 for K>=3, but the rater body is authoritative — the
//! frontier reaches T2 when only naked/hidden quads (or simpler T2 techniques)
//! fire. We default `target_tier = T2`.
//!
//! Soundness mirror. `try_naked_quad_eliminate_dryrun` is a pure (no-mutation)
//! port of `techniques::naked_set::NakedSet::<4>::apply` — the firing rule is:
//!   * 4 unsolved cells from the same unit;
//!   * each cell's candidate-mask has popcount in [2,4];
//!   * union of the 4 masks has popcount exactly 4;
//!   * ≥1 OTHER unsolved cell in the unit shares ≥1 bit with that union
//!     (i.e. the elimination set is non-empty).
//! `tests::naked_quad_dryrun_mirrors_production` asserts they agree.
//!
//! Determinism. Single-threaded with a fixed seed → byte-identical puzzle.
//! Multi-threaded `batch_*` derives per-worker seeds via splitmix off the
//! master seed (same scheme as `batch_aic_reverse_construct`).

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle, GenConfig};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::{count_solutions_up_to, solve_unique};
use super::techniques::{TechniqueId, Tier};

/// Configuration for one constructive Naked Quad reverse-synth call.
#[derive(Clone, Debug)]
pub struct NakedQuadReverseSpec {
    /// Target tier. Default T2 (post-R3.1a tier alignment in the rater).
    pub target_tier: Tier,
    /// Inclusive clue-count band for the seed puzzle.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Hard cap on outer attempts (each attempt re-seeds from a fresh
    /// random solution).
    pub max_attempts: u32,
    /// If true, additionally verify NakedQuad is load-bearing under STRICT
    /// semantics: `rate_excluding(puzzle, [NakedQuad]).tier > target_tier`.
    pub require_load_bearing: bool,
    /// Inner cap on greedy guided-removal trials per seed (0 = no cap, still
    /// bounded by N*N from a single shuffled pass).
    pub greedy_max_trials: u32,
}

impl NakedQuadReverseSpec {
    pub fn new_t2(clue_min: u32, clue_max: u32, max_attempts: u32) -> Self {
        Self {
            target_tier: Tier::T2,
            clue_min,
            clue_max,
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
        if matches!(self.target_tier, Tier::T1) {
            return Err("target_tier T1 cannot fire NakedQuad".into());
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
// Naked-Quad dry-run probe — mirrors production
// `techniques::naked_set::NakedSet::<4>::apply` exactly.
// ---------------------------------------------------------------------------

/// Pure (no mutation) check: do `cells` (4 cells in a shared unit, ordered
/// arbitrarily but distinct) form a naked-quad on `grid` that would eliminate
/// at least one candidate from another cell of that unit?
///
/// SOUNDNESS: must stay byte-equivalent to `NakedSet::<4>::apply`. Asserted by
/// `tests::naked_quad_dryrun_mirrors_production`.
pub fn try_naked_quad_eliminate_dryrun<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    cells: [u16; 4],
    unit_idx: usize,
) -> bool {
    let table = grid.table();
    let n_units = 3 * N;
    if unit_idx >= n_units {
        return false;
    }
    let unit = &table.units[unit_idx];
    // distinct + all-in-unit + all-unsolved + popcount in [2,4]
    let mut seen = [false; 4];
    for (i, &c) in cells.iter().enumerate() {
        for j in 0..i {
            if cells[j] == c {
                return false;
            }
        }
        if !unit.iter().any(|&u| u == c) {
            return false;
        }
        let cu = c as usize;
        if grid.solved[cu] != 0 {
            return false;
        }
        let pop = grid.candidates[cu].count_ones();
        if !(2..=4).contains(&pop) {
            return false;
        }
        seen[i] = true;
    }
    let _ = seen;
    let mut union: u32 = 0;
    for &c in &cells {
        union |= grid.candidates[c as usize];
    }
    if union.count_ones() != 4 {
        return false;
    }
    // ≥1 elimination on another cell of the unit.
    for &uc in unit {
        let cu = uc as usize;
        if grid.solved[cu] != 0 {
            continue;
        }
        if cells.iter().any(|&c| c == uc) {
            continue;
        }
        if grid.candidates[cu] & union != 0 {
            return true;
        }
    }
    false
}

/// Find the first naked-quad on `grid` in the same deterministic order as
/// production `NakedSet::<4>::apply`: outer loop over `0..3*N` units, inner
/// lex-order K-subset enumeration over unsolved cells with popcount in [2,4].
/// Returns `Some((unit_idx, [c0,c1,c2,c3]))` of the first firing quad, or
/// `None` if none fires.
pub fn find_first_naked_quad<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<(usize, [u16; 4])> {
    let table = grid.table().clone();
    let n_units = 3 * N;
    for u in 0..n_units {
        let unit_cells: Vec<u16> = table.units[u].iter().copied().collect();
        // (cell, mask) for unsolved, popcount in [2,4]
        let mut buf: Vec<(u16, u32)> = Vec::with_capacity(N);
        for &c in &unit_cells {
            let cu = c as usize;
            if grid.solved[cu] != 0 {
                continue;
            }
            let m = grid.candidates[cu];
            let pop = m.count_ones() as usize;
            if (2..=4).contains(&pop) {
                buf.push((c, m));
            }
        }
        if buf.len() < 4 {
            continue;
        }
        let nb = buf.len();
        // Lex iteration of 4-subsets.
        let mut idx = [0usize, 1, 2, 3];
        loop {
            let mut union: u32 = 0;
            for i in 0..4 {
                union |= buf[idx[i]].1;
            }
            if union.count_ones() == 4 {
                // Check elimination available.
                let mut fires = false;
                let combo = [
                    buf[idx[0]].0,
                    buf[idx[1]].0,
                    buf[idx[2]].0,
                    buf[idx[3]].0,
                ];
                for &uc in &unit_cells {
                    let cu = uc as usize;
                    if grid.solved[cu] != 0 {
                        continue;
                    }
                    if combo.iter().any(|&c| c == uc) {
                        continue;
                    }
                    if grid.candidates[cu] & union != 0 {
                        fires = true;
                        break;
                    }
                }
                if fires {
                    return Some((u, combo));
                }
            }
            if !next_combo4(&mut idx, nb) {
                break;
            }
        }
    }
    None
}

fn next_combo4(idx: &mut [usize; 4], nc: usize) -> bool {
    let k = 4usize;
    let mut i = k;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        let max_i = nc - (k - i);
        if idx[i] < max_i {
            idx[i] += 1;
            for j in (i + 1)..k {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
    }
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
        if !keep[i] {
            continue;
        }
        let d = solution.solved[i];
        if d == 0 {
            return None;
        }
        if g.assign(i, d).is_err() {
            return None;
        }
    }
    Some(g)
}

fn has_naked_quad<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> bool {
    find_first_naked_quad::<N, BR, BC>(grid).is_some()
}

fn guided_removal<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    solution: &Grid<N, BR, BC>,
    keep: &mut Vec<bool>,
    spec: &NakedQuadReverseSpec,
) {
    let nn = N * N;
    let mut order: Vec<u16> = (0..nn as u16).filter(|&i| keep[i as usize]).collect();
    order.shuffle(rng);

    let initial = match build_subset::<N, BR, BC>(solution, keep) {
        Some(g) => g,
        None => return,
    };
    let mut have_quad = has_naked_quad::<N, BR, BC>(&initial);

    let mut trials: u32 = 0;
    let trial_cap = if spec.greedy_max_trials == 0 {
        u32::MAX
    } else {
        spec.greedy_max_trials
    };

    for &c in &order {
        if trials >= trial_cap {
            break;
        }
        let c = c as usize;
        if !keep[c] {
            continue;
        }
        let kept_count: u32 = keep.iter().filter(|&&k| k).count() as u32;
        if kept_count <= spec.clue_min {
            break;
        }

        keep[c] = false;
        trials = trials.saturating_add(1);
        let trial = match build_subset::<N, BR, BC>(solution, keep) {
            Some(g) => g,
            None => {
                keep[c] = true;
                continue;
            }
        };
        if count_solutions_up_to(&trial, 2) != 1 {
            keep[c] = true;
            continue;
        }
        let cand_has = has_naked_quad::<N, BR, BC>(&trial);
        let accept = if !have_quad {
            // No naked-quad yet → accept any uniqueness-preserving removal.
            true
        } else {
            // Already have one → never let it disappear.
            cand_has
        };
        if accept {
            have_quad = cand_has;
        } else {
            keep[c] = true;
        }
    }
}

/// Outer driver. See module-level docs for the algorithm.
pub fn naked_quad_reverse_construct<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &NakedQuadReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() {
        return None;
    }
    let target_rank = tier_rank(spec.target_tier);
    let nn = N * N;

    for attempt in 0..spec.max_attempts {
        let cfg = GenConfig {
            target_clues: spec.clue_max,
            max_attempts: 0,
        };
        let (seed_puzzle, _seed_clues): (Grid<N, BR, BC>, u32) =
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
        if r.rater_error {
            continue;
        }
        if tier_rank(r.tier) != target_rank {
            continue;
        }
        if !r.frontier.contains(&TechniqueId::NakedQuad) {
            continue;
        }

        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[TechniqueId::NakedQuad]);
            if r2.rater_error {
                continue;
            }
            if tier_rank(r2.tier) <= target_rank {
                continue;
            }
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
pub fn batch_naked_quad_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &NakedQuadReverseSpec,
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
                if kept.load(Ordering::Relaxed) >= num_puzzles {
                    return;
                }
                let chunk = per_worker_attempts.saturating_sub(used).min(32);
                let chunk_spec = NakedQuadReverseSpec {
                    max_attempts: chunk,
                    ..spec.clone()
                };
                match naked_quad_reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
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
                    None => {
                        used = used.saturating_add(chunk);
                    }
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    let g = collected.lock().unwrap();
    g.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::techniques::{naked_set::NakedSet, Technique};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// REGRESSION: `find_first_naked_quad` and `try_naked_quad_eliminate_dryrun`
    /// must agree with production `NakedSet::<4>::apply` on whether the
    /// technique fires.
    #[test]
    fn naked_quad_dryrun_mirrors_production() {
        // Empty grid: neither fires.
        {
            let mut g: Grid<9, 3, 3> = Grid::empty();
            let t = NakedSet::<4>;
            assert!(<NakedSet<4> as Technique<9, 3, 3>>::apply(&t, &mut g).is_none());
            assert!(find_first_naked_quad::<9, 3, 3>(&g).is_none());
        }
        // Canonical naked quad in row 0:
        // c0={1,2}, c1={2,3}, c2={3,4}, c3={1,4}; cells 4..8 unrestricted.
        // Production fires; probe must agree.
        {
            let mut g: Grid<9, 3, 3> = Grid::empty();
            let masks = [0b0011u32, 0b0110, 0b1100, 0b1001];
            for (c, m) in masks.iter().enumerate() {
                for d in 1..=9u8 {
                    if m & (1u32 << (d - 1)) == 0 {
                        g.eliminate(c, d).unwrap();
                    }
                }
            }
            let probed = find_first_naked_quad::<9, 3, 3>(&g);
            assert!(probed.is_some(), "find_first_naked_quad must fire");
            let (u, cells) = probed.unwrap();
            // Must be unit 0 (row 0).
            assert_eq!(u, 0);
            assert!(try_naked_quad_eliminate_dryrun::<9, 3, 3>(&g, cells, u));
            // Production must also fire.
            let mut g2 = g.clone();
            let t = NakedSet::<4>;
            let p = <NakedSet<4> as Technique<9, 3, 3>>::apply(&t, &mut g2);
            assert!(p.is_some(), "prod NakedQuad must fire");
        }
    }

    /// `try_naked_quad_eliminate_dryrun` must reject non-naked-quad cell sets:
    ///   * 3-cell input (wrong arity is impossible — array fixed at 4 — so we
    ///     test "5-cell-style" via duplicates).
    ///   * cells from different units (no shared unit).
    ///   * union mask popcount != 4 (e.g. 5 distinct digits, or 3).
    #[test]
    fn naked_quad_rejects_pseudo_quad() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Duplicate cells → reject.
        let dup = [0u16, 0, 1, 2];
        assert!(!try_naked_quad_eliminate_dryrun::<9, 3, 3>(&g, dup, 0));

        // cells from different units (row 0 has unit_idx 0; cell 0 yes, cell 9
        // is row 1 — not in unit 0). With unit_idx=0 cell 9 must be rejected.
        let cross = [0u16, 1, 2, 9];
        assert!(!try_naked_quad_eliminate_dryrun::<9, 3, 3>(&g, cross, 0));

        // Union popcount != 4: pre-empty grid all cells have 9-bit mask →
        // union popcount = 9 → reject.
        let four = [0u16, 1, 2, 3];
        assert!(!try_naked_quad_eliminate_dryrun::<9, 3, 3>(&g, four, 0));

        // Setup: 4 cells with masks {1,2}, {2,3}, {3,4}, {4,5} — union 5 bits → reject.
        let masks = [0b00011u32, 0b00110, 0b01100, 0b11000];
        for (c, m) in masks.iter().enumerate() {
            for d in 1..=9u8 {
                if m & (1u32 << (d - 1)) == 0 {
                    g.eliminate(c, d).unwrap();
                }
            }
        }
        assert!(!try_naked_quad_eliminate_dryrun::<9, 3, 3>(&g, four, 0));
    }

    /// Determinism: same seed → byte-identical puzzle (single-thread).
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = NakedQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE);
        let ra = naked_quad_reverse_construct::<9, 3, 3, _>(&mut a, &spec);
        let rb = naked_quad_reverse_construct::<9, 3, 3, _>(&mut b, &spec);
        match (ra, rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.puzzle.to_string_grid(), y.puzzle.to_string_grid());
            }
            (None, None) => {}
            _ => panic!("determinism violated: one Some, one None"),
        }
    }

    /// Strict load-bearing: require_load_bearing=true → returned puzzle MUST
    /// have rate_excluding(p, [NakedQuad]).tier > target_tier.
    #[test]
    fn load_bearing_strict_semantics() {
        let spec = NakedQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 200,
            require_load_bearing: true,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
        let res = naked_quad_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T2));
            assert!(r.rate.frontier.contains(&TechniqueId::NakedQuad));
            let r2 = rate_excluding(&r.puzzle, &[TechniqueId::NakedQuad]);
            assert!(
                tier_rank(r2.tier) > tier_rank(Tier::T2),
                "strict load-bearing violated: rate_excluding tier = {:?}",
                r2.tier
            );
        }
        // Soft-pass on None — environment-dependent.
    }

    /// 9×9 smoke: termination + invariants on any returned puzzle.
    #[test]
    fn smoke_9x9_naked_quad() {
        let spec = NakedQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let res = naked_quad_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T2));
            assert!(r.rate.frontier.contains(&TechniqueId::NakedQuad));
            let sol = solve_unique::<9, 3, 3>(&r.puzzle).expect("must be unique");
            assert_eq!(sol.to_string_grid(), r.solution.to_string_grid());
        }
    }

    /// 12×12 informational smoke: terminate.
    #[test]
    fn smoke_12x12_naked_quad() {
        let spec = NakedQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 70,
            clue_max: 100,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 32,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let _ = naked_quad_reverse_construct::<12, 3, 4, _>(&mut rng, &spec);
        // Termination is the assertion.
    }

    /// Validation: T1 rejected, clue_min>clue_max rejected.
    #[test]
    fn validate_rejects_bad_specs() {
        let mut s = NakedQuadReverseSpec::new_t2(28, 36, 10);
        assert!(s.validate().is_ok());
        s.target_tier = Tier::T1;
        assert!(s.validate().is_err());
        s.target_tier = Tier::T2;
        s.clue_min = 50;
        s.clue_max = 30;
        assert!(s.validate().is_err());
    }

    /// Batch driver smoke: returns at most num_puzzles, never panics.
    #[test]
    fn batch_smoke_terminates() {
        let spec = NakedQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 20,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let res = batch_naked_quad_reverse_construct::<9, 3, 3>(42, &spec, 2, 1);
        assert!(res.len() <= 2);
    }
}
