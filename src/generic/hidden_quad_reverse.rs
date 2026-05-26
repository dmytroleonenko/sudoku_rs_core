//! R2.2.4: constructive Hidden Quad reverse synthesis.
//!
//! Background. R1 search-and-filter measured 0/8000 hit-rate for `hidden_quad`
//! across 4 clue-bands. Hidden Quad is sparse: a unit must have four digits
//! whose candidate-positions are confined to exactly four cells, AND those
//! four cells must still carry a candidate OUTSIDE those four digits to be
//! eliminated.
//!
//! Approach. Mirrors `naked_quad_reverse.rs` / `als_xz_reverse.rs`: guided
//! greedy removal preserving uniqueness AND the hidden-quad pattern.
//!
//!   1. Generate a unique seed at the upper end of the clue band.
//!   2. Greedily try removing each remaining clue (shuffled order); keep the
//!      removal iff (a) uniqueness is preserved AND (b) once a hidden-quad
//!      has appeared we never let it disappear.
//!   3. Verify: tier == target_tier, HiddenQuad ∈ frontier, and (optional)
//!      `rate_excluding(puzzle, [HiddenQuad]).tier > target_tier`.
//!
//! Tier semantics. After R3.1a tier-alignment HiddenQuad is **T2** in the
//! generic rater (`AnyTechnique::tier`). Default `target_tier = T2`.
//!
//! Soundness mirror. `try_hidden_quad_eliminate_dryrun` is a pure (no-mutation)
//! port of `techniques::hidden_set::HiddenSet::<4>::apply` — the firing rule
//! is:
//!   * 4 digits not yet placed in the unit, each with where_d-popcount in [2,4];
//!   * union of their where_d positions has popcount exactly 4;
//!   * ≥1 of the 4 cells at those positions has a candidate OUTSIDE the 4
//!     digits' mask (i.e. the elimination set is non-empty).
//! `tests::hidden_quad_dryrun_mirrors_production` asserts they agree.
//!
//! Determinism. Single-threaded with a fixed seed → byte-identical puzzle.
//! Multi-threaded `batch_*` derives per-worker seeds via splitmix off the
//! master seed.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle, GenConfig};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::{count_solutions_up_to, solve_unique};
use super::techniques::{TechniqueId, Tier};

/// Configuration for one constructive Hidden Quad reverse-synth call.
#[derive(Clone, Debug)]
pub struct HiddenQuadReverseSpec {
    /// Target tier. Default T2 (post-R3.1a).
    pub target_tier: Tier,
    /// Inclusive clue-count band for the seed puzzle.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Hard cap on outer attempts.
    pub max_attempts: u32,
    /// If true, additionally verify HiddenQuad is load-bearing under STRICT
    /// semantics: `rate_excluding(puzzle, [HiddenQuad]).tier > target_tier`.
    pub require_load_bearing: bool,
    /// Inner cap on greedy guided-removal trials per seed.
    pub greedy_max_trials: u32,
}

impl HiddenQuadReverseSpec {
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
            return Err("target_tier T1 cannot fire HiddenQuad".into());
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
// Hidden-Quad dry-run probe — mirrors production
// `techniques::hidden_set::HiddenSet::<4>::apply` exactly.
// ---------------------------------------------------------------------------

/// Pure (no mutation) check: does the digit-quad `digits = [d1,d2,d3,d4]`
/// (1-based, distinct, all in [1, N], all unplaced in `unit_idx`) form a
/// hidden-quad on `grid` that would eliminate ≥1 candidate?
///
/// SOUNDNESS: must stay byte-equivalent to `HiddenSet::<4>::apply`.
pub fn try_hidden_quad_eliminate_dryrun<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    digits: [u8; 4],
    unit_idx: usize,
) -> bool {
    let table = grid.table();
    let n_units = 3 * N;
    if unit_idx >= n_units {
        return false;
    }
    // Distinct + bounds + unplaced.
    for (i, &d) in digits.iter().enumerate() {
        if d < 1 || (d as usize) > N {
            return false;
        }
        for j in 0..i {
            if digits[j] == d {
                return false;
            }
        }
    }
    let unit = &table.units[unit_idx];
    // Build placed mask + where_d for the 4 digits.
    let mut placed: u32 = 0;
    let mut where_d = [0u32; 4]; // bitmap over positions 0..N
    for (k, &uc) in unit.iter().enumerate() {
        let cu = uc as usize;
        if grid.solved[cu] != 0 {
            placed |= 1u32 << (grid.solved[cu] - 1);
            continue;
        }
        let cand = grid.candidates[cu];
        for i in 0..4 {
            let bit = 1u32 << (digits[i] - 1);
            if cand & bit != 0 {
                where_d[i] |= 1u32 << k;
            }
        }
    }
    for &d in &digits {
        if placed & (1u32 << (d - 1)) != 0 {
            return false;
        }
    }
    // Each digit's where_d popcount in [2,4].
    for i in 0..4 {
        let pop = where_d[i].count_ones();
        if !(2..=4).contains(&pop) {
            return false;
        }
    }
    let mut union_pos: u32 = 0;
    for i in 0..4 {
        union_pos |= where_d[i];
    }
    if union_pos.count_ones() != 4 {
        return false;
    }
    // digit_mask
    let mut digit_mask: u32 = 0;
    for &d in &digits {
        digit_mask |= 1u32 << (d - 1);
    }
    // ≥1 of the 4 cells has a candidate outside digit_mask.
    let mut bits = union_pos;
    while bits != 0 {
        let bb = bits & bits.wrapping_neg();
        bits ^= bb;
        let k = bb.trailing_zeros() as usize;
        let cu = unit[k] as usize;
        if grid.candidates[cu] & !digit_mask != 0 {
            return true;
        }
    }
    false
}

/// Find the first hidden-quad on `grid` in deterministic order matching
/// production `HiddenSet::<4>::apply`: outer loop over 0..3*N units, inner
/// lex-order 4-subset enumeration over available digits (popcount-of-where_d
/// in [2,4]). Returns `Some((unit_idx, [d1,d2,d3,d4]))`.
pub fn find_first_hidden_quad<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<(usize, [u8; 4])> {
    let table = grid.table().clone();
    let n_units = 3 * N;
    let mut where_d: Vec<u32> = vec![0u32; N + 1]; // 1..=N
    let mut avail: Vec<u8> = Vec::with_capacity(N);

    for u in 0..n_units {
        for v in where_d.iter_mut() {
            *v = 0;
        }
        avail.clear();
        let unit_cells: Vec<usize> = table.units[u].iter().map(|&x| x as usize).collect();
        let mut placed: u32 = 0;
        for k in 0..N {
            let c = unit_cells[k];
            if grid.solved[c] != 0 {
                placed |= 1u32 << (grid.solved[c] - 1);
                continue;
            }
            let mut bits = grid.candidates[c];
            while bits != 0 {
                let bit = bits & bits.wrapping_neg();
                bits ^= bit;
                let d = (bit.trailing_zeros() as usize) + 1;
                where_d[d] |= 1u32 << k;
            }
        }
        for d in 1u8..=(N as u8) {
            if placed & (1u32 << (d - 1)) != 0 {
                continue;
            }
            let pop = where_d[d as usize].count_ones() as usize;
            if (2..=4).contains(&pop) {
                avail.push(d);
            }
        }
        if avail.len() < 4 {
            continue;
        }
        let na = avail.len();
        let mut idx = [0usize, 1, 2, 3];
        loop {
            let mut union_pos: u32 = 0;
            let mut digit_mask: u32 = 0;
            for i in 0..4 {
                let d = avail[idx[i]];
                union_pos |= where_d[d as usize];
                digit_mask |= 1u32 << (d - 1);
            }
            if union_pos.count_ones() == 4 {
                // Check elimination: ≥1 of the 4 cells has bits outside digit_mask.
                let mut fires = false;
                let mut bits = union_pos;
                while bits != 0 {
                    let bb = bits & bits.wrapping_neg();
                    bits ^= bb;
                    let k = bb.trailing_zeros() as usize;
                    let cu = unit_cells[k];
                    if grid.candidates[cu] & !digit_mask != 0 {
                        fires = true;
                        break;
                    }
                }
                if fires {
                    let digits = [
                        avail[idx[0]],
                        avail[idx[1]],
                        avail[idx[2]],
                        avail[idx[3]],
                    ];
                    return Some((u, digits));
                }
            }
            if !next_combo4(&mut idx, na) {
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

fn has_hidden_quad<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> bool {
    find_first_hidden_quad::<N, BR, BC>(grid).is_some()
}

fn guided_removal<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    solution: &Grid<N, BR, BC>,
    keep: &mut Vec<bool>,
    spec: &HiddenQuadReverseSpec,
) {
    let nn = N * N;
    let mut order: Vec<u16> = (0..nn as u16).filter(|&i| keep[i as usize]).collect();
    order.shuffle(rng);

    let initial = match build_subset::<N, BR, BC>(solution, keep) {
        Some(g) => g,
        None => return,
    };
    let mut have_hq = has_hidden_quad::<N, BR, BC>(&initial);

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
        let cand_has = has_hidden_quad::<N, BR, BC>(&trial);
        let accept = if !have_hq { true } else { cand_has };
        if accept {
            have_hq = cand_has;
        } else {
            keep[c] = true;
        }
    }
}

/// Outer driver. See module-level docs.
pub fn hidden_quad_reverse_construct<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &HiddenQuadReverseSpec,
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
        if !r.frontier.contains(&TechniqueId::HiddenQuad) {
            continue;
        }

        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[TechniqueId::HiddenQuad]);
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
pub fn batch_hidden_quad_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &HiddenQuadReverseSpec,
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
                let chunk_spec = HiddenQuadReverseSpec {
                    max_attempts: chunk,
                    ..spec.clone()
                };
                match hidden_quad_reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
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
    use crate::generic::techniques::{hidden_set::HiddenSet, Technique};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// REGRESSION: `find_first_hidden_quad` and `try_hidden_quad_eliminate_dryrun`
    /// must agree with production `HiddenSet::<4>::apply`.
    #[test]
    fn hidden_quad_dryrun_mirrors_production() {
        // Empty: neither fires.
        {
            let mut g: Grid<9, 3, 3> = Grid::empty();
            let t = HiddenSet::<4>;
            assert!(<HiddenSet<4> as Technique<9, 3, 3>>::apply(&t, &mut g).is_none());
            assert!(find_first_hidden_quad::<9, 3, 3>(&g).is_none());
        }
        // Canonical hidden quad in row 0: digits 1,2,3,4 confined to cells 0..3
        // (eliminate them from cells 4..8).
        {
            let mut g: Grid<9, 3, 3> = Grid::empty();
            for c in 4..9usize {
                for d in [1u8, 2, 3, 4] {
                    g.eliminate(c, d).unwrap();
                }
            }
            // probe fires
            let probed = find_first_hidden_quad::<9, 3, 3>(&g);
            assert!(probed.is_some(), "find_first_hidden_quad must fire");
            let (u, digits) = probed.unwrap();
            assert_eq!(u, 0);
            assert!(try_hidden_quad_eliminate_dryrun::<9, 3, 3>(&g, digits, u));
            // production fires
            let mut g2 = g.clone();
            let t = HiddenSet::<4>;
            assert!(<HiddenSet<4> as Technique<9, 3, 3>>::apply(&t, &mut g2).is_some());
        }
    }

    /// `try_hidden_quad_eliminate_dryrun` must reject pseudo-quads:
    ///   * duplicate digits.
    ///   * out-of-range digits.
    ///   * 3 digits expanded to 4 with a duplicate (we test via duplicate above;
    ///     here use unit_idx out of range).
    ///   * digits whose where_d union has popcount ≠ 4.
    #[test]
    fn hidden_quad_rejects_pseudo_quad() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Duplicate digit.
        let dup = [1u8, 1, 2, 3];
        assert!(!try_hidden_quad_eliminate_dryrun::<9, 3, 3>(&g, dup, 0));
        // Out-of-range digit (10 > N=9).
        let oob = [1u8, 2, 3, 10];
        assert!(!try_hidden_quad_eliminate_dryrun::<9, 3, 3>(&g, oob, 0));
        // Out-of-range unit_idx.
        let ok = [1u8, 2, 3, 4];
        assert!(!try_hidden_quad_eliminate_dryrun::<9, 3, 3>(&g, ok, 999));

        // Empty grid → digits 1..4 each appear in 9 positions → where_d popcount=9
        // → reject (popcount > 4).
        assert!(!try_hidden_quad_eliminate_dryrun::<9, 3, 3>(&g, ok, 0));

        // Setup confining digits 1,2,3 (3-digit set) to cells 0..3 (4 positions);
        // union popcount = 4 but we only have 3 digits — try 4-tuple including
        // unrelated digit 5 which still fires in 9 positions → union pop = 9 → reject.
        for c in 4..9usize {
            for d in [1u8, 2, 3] {
                g.eliminate(c, d).unwrap();
            }
        }
        let mixed = [1u8, 2, 3, 5];
        assert!(!try_hidden_quad_eliminate_dryrun::<9, 3, 3>(&g, mixed, 0));
    }

    /// Determinism.
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = HiddenQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(0xBADC0DE);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(0xBADC0DE);
        let ra = hidden_quad_reverse_construct::<9, 3, 3, _>(&mut a, &spec);
        let rb = hidden_quad_reverse_construct::<9, 3, 3, _>(&mut b, &spec);
        match (ra, rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.puzzle.to_string_grid(), y.puzzle.to_string_grid());
            }
            (None, None) => {}
            _ => panic!("determinism violated: one Some, one None"),
        }
    }

    /// Strict load-bearing.
    #[test]
    fn load_bearing_strict_semantics() {
        let spec = HiddenQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 200,
            require_load_bearing: true,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
        let res = hidden_quad_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T2));
            assert!(r.rate.frontier.contains(&TechniqueId::HiddenQuad));
            let r2 = rate_excluding(&r.puzzle, &[TechniqueId::HiddenQuad]);
            assert!(
                tier_rank(r2.tier) > tier_rank(Tier::T2),
                "strict load-bearing violated: rate_excluding tier = {:?}",
                r2.tier
            );
        }
    }

    #[test]
    fn smoke_9x9_hidden_quad() {
        let spec = HiddenQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let res = hidden_quad_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T2));
            assert!(r.rate.frontier.contains(&TechniqueId::HiddenQuad));
            let sol = solve_unique::<9, 3, 3>(&r.puzzle).expect("must be unique");
            assert_eq!(sol.to_string_grid(), r.solution.to_string_grid());
        }
    }

    #[test]
    fn smoke_12x12_hidden_quad() {
        let spec = HiddenQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 70,
            clue_max: 100,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 32,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let _ = hidden_quad_reverse_construct::<12, 3, 4, _>(&mut rng, &spec);
    }

    #[test]
    fn validate_rejects_bad_specs() {
        let mut s = HiddenQuadReverseSpec::new_t2(28, 36, 10);
        assert!(s.validate().is_ok());
        s.target_tier = Tier::T1;
        assert!(s.validate().is_err());
        s.target_tier = Tier::T2;
        s.clue_min = 50;
        s.clue_max = 30;
        assert!(s.validate().is_err());
    }

    #[test]
    fn batch_smoke_terminates() {
        let spec = HiddenQuadReverseSpec {
            target_tier: Tier::T2,
            clue_min: 28,
            clue_max: 36,
            max_attempts: 20,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let res = batch_hidden_quad_reverse_construct::<9, 3, 3>(42, &spec, 2, 1);
        assert!(res.len() <= 2);
    }
}
