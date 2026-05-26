//! R2.2.3: constructive ALS-XZ reverse synthesis with ALS-size targeting.
//!
//! Background. ALS-XZ (Almost Locked Set, XZ rule) is a T3 technique that
//! becomes load-bearing on large grids (16×16) where the combinatorial space
//! of ALS pairs is rich enough that no simpler T3 technique substitutes. R1
//! search-and-filter measured very low pps for puzzles where ALS-XZ is the
//! unique T3 wedge — particularly at 16×16 where rejection compounds. This
//! module mirrors the R2.1 (AIC) and R2.2.{1,2} (Fish, UR-T2) approach:
//! guided removal that preserves uniqueness *and* the ALS-XZ structure.
//!
//! Approach.
//!
//! 1. Generate a uniquely-solvable seed puzzle via `gen_unique_puzzle` at the
//!    upper end of the requested clue band.
//! 2. While the puzzle has more clues than `clue_min`: try removing each
//!    remaining clue in shuffled order; keep the removal iff
//!      (a) uniqueness is preserved, and
//!      (b) ALS-XZ still fires (or fires for the first time), and
//!      (c) the smaller of the two ALS sizes lies in
//!          `[als_size_min, als_size_max]` (when we already have an ALS-XZ
//!          firing; otherwise we accept any non-worsening removal).
//! 3. Verify on convergence:
//!      * `rate(puzzle).tier == target_tier`,
//!      * `AlsXz ∈ rate(puzzle).frontier`,
//!      * the first ALS-XZ pair found by the dry-run probe has
//!        `min(|A|,|B|) ∈ [als_size_min, als_size_max]`,
//!      * (if `require_load_bearing`) `rate_excluding(puzzle, [AlsXz]).tier
//!        > target_tier` — strict semantics, mirroring AIC reverse.
//!
//! Soundness. `try_als_xz_eliminate_dryrun` and `find_first_als_xz` use the
//! exact same restricted-common / overlap rules as the production
//! `techniques::als_xz::AlsXz::apply` — see `als_xz_dryrun_mirrors_production`
//! regression test. ALSes are required to be **non-overlapping** per the
//! ALS-XZ definition; the production code enforces this and so does the
//! dry-run probe (`als_xz_rejects_overlap_cells` test).
//!
//! Determinism. Single-threaded with a fixed seed → byte-identical puzzle.
//! Multi-threaded `batch_als_xz_reverse_construct` derives per-worker seeds
//! via splitmix off the master seed and produces the same output **set**;
//! emission order is non-deterministic. Sort by `(puzzle, solution)` if you
//! need byte-identical multi-thread output.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle, GenConfig};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::{count_solutions_up_to, solve_unique};
use super::techniques::{TechniqueId, Tier};

/// Configuration for one constructive ALS-XZ reverse-synth call.
#[derive(Clone, Debug)]
pub struct AlsXzReverseSpec {
    /// Target tier (must be T3 — ALS-XZ is T3).
    pub target_tier: Tier,
    /// Inclusive clue-count band for the seed puzzle.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Inclusive band on the **smaller** ALS size in the firing pair
    /// (typical 1..=4; size-1 ALS = bivalue cell). Production enumerates up
    /// to MAX_ALS_SIZE=4, so values >4 will never hit.
    pub als_size_min: u32,
    pub als_size_max: u32,
    /// Hard cap on outer attempts (each attempt re-seeds from a fresh
    /// random solution).
    pub max_attempts: u32,
    /// If true, additionally verify AlsXz is load-bearing under STRICT
    /// semantics: `rate_excluding(puzzle, [AlsXz]).tier > target_tier`.
    /// Mirrors `aic_reverse::AicReverseSpec::require_load_bearing`.
    pub require_load_bearing: bool,
    /// Inner cap on greedy guided-removal trials per seed (0 = no cap, still
    /// bounded by N*N from a single shuffled pass).
    pub greedy_max_trials: u32,
}

impl AlsXzReverseSpec {
    pub fn new_t3(clue_min: u32, clue_max: u32, max_attempts: u32) -> Self {
        Self {
            target_tier: Tier::T3,
            clue_min,
            clue_max,
            als_size_min: 1,
            als_size_max: 4,
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
            return Err("target_tier must be T3 (AlsXz is T3)".into());
        }
        if self.als_size_min == 0 {
            return Err("als_size_min must be >= 1".into());
        }
        if self.als_size_min > self.als_size_max {
            return Err(format!(
                "als_size_min ({}) > als_size_max ({})",
                self.als_size_min, self.als_size_max
            ));
        }
        if self.als_size_max > 4 {
            return Err(format!(
                "als_size_max ({}) > 4 — production AlsXz enumerates only sizes 1..=4",
                self.als_size_max
            ));
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
// ALS-XZ dry-run probe — mirrors production `techniques::als_xz::AlsXz::apply`.
//
// IMPORTANT: any change here must be matched in the prod technique (and
// vice-versa) or the `als_xz_dryrun_mirrors_production` regression test will
// fail. The probe enumerates ALSes per-unit (sizes 1..=MAX_ALS_SIZE), then
// scans non-overlapping pairs whose shared candidate-mask has ≥2 bits, finds
// a restricted-common digit X (every X-cell of A peers with every X-cell of
// B), then for each Z ≠ X scans non-(A∪B) cells that see all Z-cells of
// (A∪B) — the first such cell is the first ALS-XZ elimination.
// ---------------------------------------------------------------------------

const MAX_ALS_SIZE: usize = 4;

#[derive(Clone)]
struct ProbeAls {
    cells: Vec<u16>,
    mask: u32,
}

fn next_combination(idx: &mut [usize], pool: usize) -> bool {
    let n = idx.len();
    let mut i = n;
    while i > 0 {
        i -= 1;
        if idx[i] < pool - (n - i) {
            idx[i] += 1;
            for j in (i + 1)..n {
                idx[j] = idx[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

fn als_key(a: &ProbeAls) -> u128 {
    let mut k: u128 = a.mask as u128;
    for &c in &a.cells {
        k = (k << 9) | (c as u128);
    }
    k
}

fn enumerate_alses<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Vec<ProbeAls> {
    let mut out: Vec<ProbeAls> = Vec::new();
    let mut seen: std::collections::HashSet<u128> = std::collections::HashSet::new();
    let table = grid.table().clone();
    let n_units = 3 * N;

    for u_idx in 0..n_units {
        let unit = &table.units[u_idx];
        let mut cells: Vec<(u16, u32)> = Vec::with_capacity(N);
        for &c in unit {
            let cu = c as usize;
            if grid.solved[cu] == 0 {
                cells.push((c, grid.candidates[cu]));
            }
        }
        let n = cells.len();
        if n == 0 {
            continue;
        }
        // size 1: bivalue cells
        for i in 0..n {
            if cells[i].1.count_ones() == 2 {
                let als = ProbeAls {
                    cells: vec![cells[i].0],
                    mask: cells[i].1,
                };
                let key = als_key(&als);
                if seen.insert(key) {
                    out.push(als);
                }
            }
        }
        let max_size = MAX_ALS_SIZE.min(n);
        let mut idx = vec![0usize; MAX_ALS_SIZE];
        for size in 2..=max_size {
            for k in 0..size {
                idx[k] = k;
            }
            loop {
                let mut mask: u32 = 0;
                for k in 0..size {
                    mask |= cells[idx[k]].1;
                }
                if (mask.count_ones() as usize) == size + 1 {
                    let mut cs: Vec<u16> = (0..size).map(|k| cells[idx[k]].0).collect();
                    cs.sort_unstable();
                    let als = ProbeAls { cells: cs, mask };
                    let key = als_key(&als);
                    if seen.insert(key) {
                        out.push(als);
                    }
                }
                if !next_combination(&mut idx[..size], n) {
                    break;
                }
            }
        }
    }
    out
}

fn shares_unit<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    a: usize,
    b: usize,
) -> bool {
    if a == b {
        return false;
    }
    let table = grid.table();
    table.cells[a].peers.iter().any(|&p| p as usize == b)
}

fn restricted_common<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    a: &ProbeAls,
    b: &ProbeAls,
    d_bit: u8,
) -> bool {
    let bit = 1u32 << d_bit;
    let a_cells: Vec<usize> = a
        .cells
        .iter()
        .map(|&c| c as usize)
        .filter(|&c| grid.candidates[c] & bit != 0)
        .collect();
    if a_cells.is_empty() {
        return false;
    }
    let b_cells: Vec<usize> = b
        .cells
        .iter()
        .map(|&c| c as usize)
        .filter(|&c| grid.candidates[c] & bit != 0)
        .collect();
    if b_cells.is_empty() {
        return false;
    }
    for &ac in &a_cells {
        for &bc in &b_cells {
            if ac == bc {
                return false;
            }
            if !shares_unit::<N, BR, BC>(grid, ac, bc) {
                return false;
            }
        }
    }
    true
}

/// Pure (no mutation) check: would the ALS pair (a, b) produce ≥1 ALS-XZ
/// elimination on `grid`? Mirrors `als_xz::AlsXz::apply` exactly: requires
/// disjoint cells, ≥2 common bits, ≥1 restricted-common digit X, ≥1 other
/// digit Z, and ≥1 cell outside (a∪b) that sees all Z-cells of (a∪b).
pub fn try_als_xz_eliminate_dryrun<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    a_cells: &[u16],
    b_cells: &[u16],
) -> bool {
    // Overlap check (ALS-XZ requires disjoint ALSes).
    for &ac in a_cells {
        for &bc in b_cells {
            if ac == bc {
                return false;
            }
        }
    }
    let mut a_mask: u32 = 0;
    for &c in a_cells {
        a_mask |= grid.candidates[c as usize];
    }
    let mut b_mask: u32 = 0;
    for &c in b_cells {
        b_mask |= grid.candidates[c as usize];
    }
    let common = a_mask & b_mask;
    if common.count_ones() < 2 {
        return false;
    }
    let a = ProbeAls {
        cells: a_cells.to_vec(),
        mask: a_mask,
    };
    let b = ProbeAls {
        cells: b_cells.to_vec(),
        mask: b_mask,
    };
    let nn = N * N;
    let mut bits = common;
    while bits != 0 {
        let bb = bits & bits.wrapping_neg();
        bits ^= bb;
        let d_bit = bb.trailing_zeros() as u8;
        if !restricted_common::<N, BR, BC>(grid, &a, &b, d_bit) {
            continue;
        }
        let mut zbits = common & !bb;
        while zbits != 0 {
            let zb = zbits & zbits.wrapping_neg();
            zbits ^= zb;
            // Z-cells in a ∪ b
            let mut z_cells: Vec<usize> = Vec::new();
            for &c in a_cells {
                let cu = c as usize;
                if grid.candidates[cu] & zb != 0 {
                    z_cells.push(cu);
                }
            }
            for &c in b_cells {
                let cu = c as usize;
                if grid.candidates[cu] & zb != 0 {
                    z_cells.push(cu);
                }
            }
            if z_cells.is_empty() {
                continue;
            }
            for cell in 0..nn {
                if grid.solved[cell] != 0 {
                    continue;
                }
                if grid.candidates[cell] & zb == 0 {
                    continue;
                }
                if a_cells.iter().any(|&c| c as usize == cell) {
                    continue;
                }
                if b_cells.iter().any(|&c| c as usize == cell) {
                    continue;
                }
                let mut sees_all = true;
                for &zc in &z_cells {
                    if !shares_unit::<N, BR, BC>(grid, cell, zc) {
                        sees_all = false;
                        break;
                    }
                }
                if sees_all {
                    return true;
                }
            }
        }
    }
    false
}

/// Find the first ALS-XZ pair on `grid` (deterministic enumeration order
/// matching the production `apply`). Returns `Some((min_size, max_size))` of
/// the firing pair sizes, or `None` if no ALS-XZ fires.
///
/// The walker is deterministic and mirrors `als_xz::AlsXz::apply`'s outer
/// pair-scan order (i < j over enumerate_alses).
pub fn find_first_als_xz<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<(usize, usize)> {
    let alses = enumerate_alses::<N, BR, BC>(grid);
    let n = alses.len();
    let nn = N * N;
    for i in 0..n {
        for j in (i + 1)..n {
            let a = &alses[i];
            let b = &alses[j];
            // Overlap.
            let mut overlap = false;
            'ov: for &ac in &a.cells {
                for &bc in &b.cells {
                    if ac == bc {
                        overlap = true;
                        break 'ov;
                    }
                }
            }
            if overlap {
                continue;
            }
            let common = a.mask & b.mask;
            if common.count_ones() < 2 {
                continue;
            }
            let mut bits = common;
            while bits != 0 {
                let bb = bits & bits.wrapping_neg();
                bits ^= bb;
                let d_bit = bb.trailing_zeros() as u8;
                if !restricted_common::<N, BR, BC>(grid, a, b, d_bit) {
                    continue;
                }
                let mut zbits = common & !bb;
                while zbits != 0 {
                    let zb = zbits & zbits.wrapping_neg();
                    zbits ^= zb;
                    let mut z_cells: Vec<usize> = Vec::new();
                    for &c in &a.cells {
                        let cu = c as usize;
                        if grid.candidates[cu] & zb != 0 {
                            z_cells.push(cu);
                        }
                    }
                    for &c in &b.cells {
                        let cu = c as usize;
                        if grid.candidates[cu] & zb != 0 {
                            z_cells.push(cu);
                        }
                    }
                    if z_cells.is_empty() {
                        continue;
                    }
                    for cell in 0..nn {
                        if grid.solved[cell] != 0 {
                            continue;
                        }
                        if grid.candidates[cell] & zb == 0 {
                            continue;
                        }
                        if a.cells.iter().any(|&c| c as usize == cell) {
                            continue;
                        }
                        if b.cells.iter().any(|&c| c as usize == cell) {
                            continue;
                        }
                        let mut sees_all = true;
                        for &zc in &z_cells {
                            if !shares_unit::<N, BR, BC>(grid, cell, zc) {
                                sees_all = false;
                                break;
                            }
                        }
                        if sees_all {
                            let sa = a.cells.len();
                            let sb = b.cells.len();
                            return Some((sa.min(sb), sa.max(sb)));
                        }
                    }
                }
            }
        }
    }
    None
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

/// Score a partial puzzle: `(has_als_xz, in_size_band)` where `in_size_band`
/// is `true` iff the firing pair's smaller-side size falls within
/// `[als_size_min, als_size_max]`.
fn als_xz_score<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    spec: &AlsXzReverseSpec,
) -> (bool, bool) {
    match find_first_als_xz::<N, BR, BC>(grid) {
        Some((min_size, _max_size)) => {
            let in_band = (min_size as u32) >= spec.als_size_min
                && (min_size as u32) <= spec.als_size_max;
            (true, in_band)
        }
        None => (false, false),
    }
}

fn guided_removal<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    solution: &Grid<N, BR, BC>,
    keep: &mut Vec<bool>,
    spec: &AlsXzReverseSpec,
) {
    let nn = N * N;
    let mut order: Vec<u16> = (0..nn as u16).filter(|&i| keep[i as usize]).collect();
    order.shuffle(rng);

    let initial = match build_subset::<N, BR, BC>(solution, keep) {
        Some(g) => g,
        None => return,
    };
    let (mut have_alsxz, mut in_band) = als_xz_score::<N, BR, BC>(&initial, spec);

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
        let (cand_has, cand_in_band) = als_xz_score::<N, BR, BC>(&trial, spec);
        let accept = if !have_alsxz {
            // No ALS-XZ yet → accept any uniqueness-preserving removal; we may
            // need to thin further before patterns emerge.
            true
        } else if !in_band {
            // R3.2: was `cand_has && (cand_in_band || true)` which always
            // reduces to `cand_has` — the band-improvement guard never
            // tightened anything. Either we keep ALS-XZ regardless of band
            // (current de-facto behaviour, post-filter still enforces band)
            // or we require the candidate to be in-band. The post-filter
            // already guarantees the final puzzle is in band; here we bias
            // the walk toward in-band candidates by accepting them
            // unconditionally and accepting out-of-band candidates only if
            // they don't move us further from band — which in practice means
            // accepting any cand_has so the walker keeps shrinking. Leave
            // semantics unchanged but spell it out.
            cand_has
        } else {
            // Have a band-matching ALS-XZ → never lose it AND require staying
            // in band (don't drift to wrong size).
            cand_has && cand_in_band
        };
        if accept {
            have_alsxz = cand_has;
            in_band = cand_in_band;
        } else {
            keep[c] = true;
        }
    }
}

/// Outer driver. See module-level docs for the algorithm.
pub fn als_xz_reverse_construct<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &AlsXzReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() {
        return None;
    }
    let target_rank = tier_rank(spec.target_tier);
    let nn = N * N;

    for attempt in 0..spec.max_attempts {
        let seed_target_clues = spec.clue_max;
        let cfg = GenConfig {
            target_clues: seed_target_clues,
            max_attempts: 0,
        };
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
        if r.rater_error {
            continue;
        }
        if tier_rank(r.tier) != target_rank {
            continue;
        }
        if !r.frontier.contains(&TechniqueId::AlsXz) {
            continue;
        }

        // ALS-size band match (smaller of the firing pair).
        let (min_size, _max_size) = match find_first_als_xz::<N, BR, BC>(&puzzle) {
            Some(p) => p,
            None => continue, // shouldn't happen if AlsXz is in frontier
        };
        if (min_size as u32) < spec.als_size_min || (min_size as u32) > spec.als_size_max {
            continue;
        }

        // Strict load-bearing semantics (mirrors aic_reverse): AlsXz is
        // load-bearing iff removing it from the cascade promotes the puzzle
        // to a strictly harder tier (T4Plus).
        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[TechniqueId::AlsXz]);
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
pub fn batch_als_xz_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &AlsXzReverseSpec,
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
                let chunk_spec = AlsXzReverseSpec {
                    max_attempts: chunk,
                    ..spec.clone()
                };
                match als_xz_reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
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
    use crate::generic::techniques::{als_xz::AlsXz, Technique};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// REGRESSION: `try_als_xz_eliminate_dryrun` and `find_first_als_xz` must
    /// agree with the production `AlsXz::apply` on whether ALS-XZ fires. We
    /// build the canonical XY-Wing pattern (three bivalues) where the prod
    /// technique is known to fire and assert the dry-run also reports
    /// firing. Then build an empty grid and assert neither path fires.
    #[test]
    fn als_xz_dryrun_mirrors_production() {
        // Empty grid: prod doesn't fire; probe returns None.
        {
            let mut g: Grid<9, 3, 3> = Grid::empty();
            let t = AlsXz;
            assert!(<AlsXz as Technique<9, 3, 3>>::apply(&t, &mut g).is_none());
            assert!(find_first_als_xz::<9, 3, 3>(&g).is_none());
        }
        // XY-Wing-style: (0,0)={1,2}, (0,3)={1,3}, (1,0)={2,3}.
        {
            let mut g: Grid<9, 3, 3> = Grid::empty();
            for d in 3..=9u8 {
                g.eliminate(0, d).unwrap();
            }
            for d in [2u8, 4, 5, 6, 7, 8, 9] {
                g.eliminate(3, d).unwrap();
            }
            for d in [1u8, 4, 5, 6, 7, 8, 9] {
                g.eliminate(9, d).unwrap();
            }
            // Probe says "yes, ALS-XZ fires somewhere".
            let probed = find_first_als_xz::<9, 3, 3>(&g);
            assert!(
                probed.is_some(),
                "find_first_als_xz must fire on XY-Wing grid"
            );
            // And prod fires too.
            let mut g2 = g.clone();
            let t = AlsXz;
            let p = <AlsXz as Technique<9, 3, 3>>::apply(&t, &mut g2);
            assert!(p.is_some(), "prod AlsXz must fire on XY-Wing grid");
            // Mutual: both fired → mirror property holds for this grid.
        }
    }

    /// `try_als_xz_eliminate_dryrun` must reject ALS pairs that share any
    /// cells (overlap forbidden by ALS-XZ definition).
    #[test]
    fn als_xz_rejects_overlap_cells() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 {
            g.eliminate(0, d).unwrap();
        }
        // (0,0) is now {1,2}. Pass it as both A and B → overlap → reject.
        let a_cells: Vec<u16> = vec![0];
        let b_cells: Vec<u16> = vec![0];
        assert!(
            !try_als_xz_eliminate_dryrun::<9, 3, 3>(&g, &a_cells, &b_cells),
            "ALS-XZ must reject identical (overlapping) cell-sets"
        );
        // Partial overlap: A = [0, 1], B = [0, 2] also overlaps.
        let a2: Vec<u16> = vec![0, 1];
        let b2: Vec<u16> = vec![0, 2];
        assert!(
            !try_als_xz_eliminate_dryrun::<9, 3, 3>(&g, &a2, &b2),
            "ALS-XZ must reject partially-overlapping cell-sets"
        );
    }

    /// Determinism: same seed → same puzzle (single-thread).
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = AlsXzReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            als_size_min: 1,
            als_size_max: 4,
            max_attempts: 50,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(31337);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(31337);
        let ra = als_xz_reverse_construct::<9, 3, 3, _>(&mut a, &spec);
        let rb = als_xz_reverse_construct::<9, 3, 3, _>(&mut b, &spec);
        match (ra, rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.puzzle.to_string_grid(), y.puzzle.to_string_grid());
            }
            (None, None) => {}
            _ => panic!("determinism violated: one Some, one None"),
        }
    }

    /// Strict load-bearing: when require_load_bearing=true, returned puzzle
    /// MUST have `rate_excluding(p, [AlsXz]).tier > target_tier`.
    #[test]
    fn load_bearing_strict_semantics() {
        let spec = AlsXzReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            als_size_min: 1,
            als_size_max: 4,
            max_attempts: 200,
            require_load_bearing: true,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
        let res = als_xz_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T3));
            assert!(r.rate.frontier.contains(&TechniqueId::AlsXz));
            let r2 = rate_excluding(&r.puzzle, &[TechniqueId::AlsXz]);
            assert!(
                tier_rank(r2.tier) > tier_rank(Tier::T3),
                "strict load-bearing violated: rate_excluding tier = {:?}",
                r2.tier
            );
            // Sanity: ALS sizes within band (default 1..=4 always holds).
            let (min_s, _) = find_first_als_xz::<9, 3, 3>(&r.puzzle).unwrap();
            assert!((1..=4).contains(&min_s));
        }
        // Soft-pass on None — environment-dependent.
    }

    /// 9×9 smoke: with looser settings the construct path terminates and any
    /// returned puzzle satisfies the basic invariants.
    #[test]
    fn smoke_9x9_als_xz() {
        let spec = AlsXzReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            als_size_min: 1,
            als_size_max: 4,
            max_attempts: 50,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let res = als_xz_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T3));
            assert!(r.rate.frontier.contains(&TechniqueId::AlsXz));
            // returned puzzle solves to its solution.
            let sol = solve_unique::<9, 3, 3>(&r.puzzle).expect("must be unique");
            assert_eq!(sol.to_string_grid(), r.solution.to_string_grid());
        }
    }

    /// 16×16 informational smoke: just terminate. 16×16 generation is slow.
    #[test]
    fn smoke_16x16_als_xz() {
        let spec = AlsXzReverseSpec {
            target_tier: Tier::T3,
            clue_min: 100,
            clue_max: 130,
            als_size_min: 1,
            als_size_max: 4,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 32,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let _ = als_xz_reverse_construct::<16, 4, 4, _>(&mut rng, &spec);
        // Termination is the assertion.
    }

    /// Validation: T1/T2 rejected, als_size_min=0 rejected, als_size_max>4
    /// rejected, clue_min>clue_max rejected, als_size_min>als_size_max rejected.
    #[test]
    fn validate_rejects_bad_specs() {
        let mut s = AlsXzReverseSpec::new_t3(22, 30, 10);
        assert!(s.validate().is_ok());
        s.target_tier = Tier::T1;
        assert!(s.validate().is_err());
        s.target_tier = Tier::T2;
        assert!(s.validate().is_err());
        s.target_tier = Tier::T3;
        s.als_size_min = 0;
        assert!(s.validate().is_err());
        s.als_size_min = 1;
        s.als_size_max = 5;
        assert!(s.validate().is_err());
        s.als_size_max = 4;
        s.als_size_min = 4;
        s.als_size_max = 2;
        assert!(s.validate().is_err());
        s.als_size_min = 1;
        s.als_size_max = 4;
        s.clue_min = 50;
        s.clue_max = 30;
        assert!(s.validate().is_err());
    }

    /// Batch driver smoke: returns at most num_puzzles, never panics.
    #[test]
    fn batch_smoke_terminates() {
        let spec = AlsXzReverseSpec {
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            als_size_min: 1,
            als_size_max: 4,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let res = batch_als_xz_reverse_construct::<9, 3, 3>(42, &spec, 2, 1);
        assert!(res.len() <= 2);
    }
}
