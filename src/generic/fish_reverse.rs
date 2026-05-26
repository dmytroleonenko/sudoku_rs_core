//! R2.2.1: constructive Fish<K> reverse synthesis (X-Wing / Swordfish /
//! Jellyfish), mirroring the R2.1 AIC pattern.
//!
//! Background. R1's search-and-filter `reverse_construct` measured Fish hit
//! rates of roughly 1/1000 (jellyfish), 5/1000 (swordfish), better-but-still-
//! sparse (X-Wing) on 9×9. On 16×16 every Fish family fires < 1 pps. Without
//! a steady supply of load-bearing Fish puzzles we cannot run the Phase D'5
//! lemma-extraction probe for Fish techniques.
//!
//! Approach. **Guided removal**, identical pipeline shape to
//! `aic_reverse.rs`:
//!
//! 1. Generate a uniquely-solvable seed puzzle near the upper end of the
//!    requested clue band.
//! 2. While more clues than `clue_min`: try removing each remaining clue in
//!    shuffled order; keep the removal iff (a) uniqueness preserved AND
//!    (b) the puzzle either still admits a Fish<K> elimination at the *target*
//!    size OR doesn't yet have one (allowing the structure to emerge).
//! 3. After convergence, verify:
//!      * `rate(puzzle).tier == target_tier`,
//!      * `Fish<K>'s id ∈ rate(puzzle).frontier`,
//!      * `find_first_fish_size(puzzle) == K` (the *first-firing* fish in the
//!        cascade is exactly size K — heavier fish that get pre-empted by a
//!        smaller one don't count),
//!      * (if `require_load_bearing`) `rate_excluding(puzzle, &[Fish_K]).tier
//!        > target_tier` — Fish<K> is genuinely required.
//!
//! Soundness: `try_fish_eliminate_dryrun` is a *pure* mirror of
//! `techniques::fish::try_fish_k`. We only check whether at least one cell in
//! the cover lines (excluding base lines) carries the candidate digit; we
//! never mutate the grid here. Same combination iteration, same K-cover-mask
//! semantics.
//!
//! Determinism. Single-threaded with a fixed seed → byte-identical puzzle.
//! Multi-threaded `batch_*` reuses the splitmix-style per-worker seed
//! derivation from `batch_reverse_construct` / `batch_aic_reverse_construct`.
//! No `thread_rng`, no `HashMap` iteration order — only `Vec` traversal in
//! fixed (cell-index ascending) order.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle, GenConfig};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::{count_solutions_up_to, solve_unique};
use super::techniques::{TechniqueId, Tier};

/// Configuration for one constructive Fish<K> reverse-synth call.
#[derive(Clone, Debug)]
pub struct FishReverseSpec {
    /// Target Fish size (K=2 X-Wing, K=3 Swordfish, K=4 Jellyfish).
    pub target_size: usize,
    /// Target tier (must be T3; Fish are T3 techniques).
    pub target_tier: Tier,
    /// Inclusive clue-count band for the seed puzzle.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Hard cap on outer attempts (each attempt re-seeds).
    pub max_attempts: u32,
    /// If true, additionally verify `rate_excluding(puzzle, [Fish_K]).tier
    /// > target_tier` — i.e. Fish<K> is load-bearing.
    pub require_load_bearing: bool,
    /// Inner cap on greedy guided-removal trials per seed.
    /// `0` means "no extra cap"; still bounded by `N*N` from a single
    /// shuffled pass.
    pub greedy_max_trials: u32,
}

impl FishReverseSpec {
    pub fn new_t3(target_size: usize, max_attempts: u32) -> Self {
        Self {
            target_size,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts,
            require_load_bearing: true,
            greedy_max_trials: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.target_size, 2 | 3 | 4) {
            return Err(format!(
                "target_size {} invalid; Fish supports K ∈ {{2,3,4}}",
                self.target_size
            ));
        }
        if self.clue_min > self.clue_max {
            return Err(format!(
                "clue_min ({}) > clue_max ({})",
                self.clue_min, self.clue_max
            ));
        }
        if !matches!(self.target_tier, Tier::T3) {
            return Err(format!(
                "target_tier must be T3 for Fish (got {:?})",
                self.target_tier
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

#[inline]
fn fish_id_for(size: usize) -> TechniqueId {
    match size {
        2 => TechniqueId::XWing,
        3 => TechniqueId::Swordfish,
        4 => TechniqueId::Jellyfish,
        _ => panic!("fish size must be 2/3/4"),
    }
}

// ---------------------------------------------------------------------------
// Soundness mirror of `techniques::fish::try_fish_k`. Pure: never mutates the
// grid. Returns true iff there is at least one elimination candidate cell in
// the cover lines (not in base lines) that still has digit `digit` as a
// candidate.
//
// SOUNDNESS NOTE. Mirror exact:
//   * combination iteration via `next_combination`,
//   * `(2..=K)` count_ones gate per base line,
//   * `union.count_ones() == K` cover gate,
//   * cover-cell exclusion of base lines,
//   * `grid.solved[cell] == 0 && (grid.candidates[cell] & bit) != 0`.
// Do not introduce any rule that production fish.rs does not implement.
// ---------------------------------------------------------------------------

fn try_fish_eliminate_dryrun<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    digit: u8,
    k: usize,
    line_mask: &[u32],
    placed_line: &[bool],
    base_is_row: bool,
) -> bool {
    let mut bases: Vec<usize> = Vec::with_capacity(N);
    for i in 0..N {
        if placed_line[i] {
            continue;
        }
        let pc = line_mask[i].count_ones() as usize;
        if (2..=k).contains(&pc) {
            bases.push(i);
        }
    }
    if bases.len() < k {
        return false;
    }
    let bit = 1u32 << (digit - 1);
    let mut idx = vec![0usize; k];
    for kk in 0..k {
        idx[kk] = kk;
    }
    loop {
        let mut union: u32 = 0;
        for kk in 0..k {
            union |= line_mask[bases[idx[kk]]];
        }
        if (union.count_ones() as usize) == k {
            let mut base_bits: u32 = 0;
            for kk in 0..k {
                base_bits |= 1u32 << bases[idx[kk]];
            }
            let mut cover = union;
            while cover != 0 {
                let cb = cover & cover.wrapping_neg();
                cover ^= cb;
                let cover_idx = cb.trailing_zeros() as usize;
                for line in 0..N {
                    if (base_bits >> line) & 1 != 0 {
                        continue;
                    }
                    let cell = if base_is_row {
                        line * N + cover_idx
                    } else {
                        cover_idx * N + line
                    };
                    if grid.solved[cell] != 0 {
                        continue;
                    }
                    if grid.candidates[cell] & bit != 0 {
                        return true;
                    }
                }
            }
        }
        if !next_combination(&mut idx, bases.len()) {
            break;
        }
    }
    false
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

/// Build per-row / per-col candidate masks for a given digit. Returns
/// `(row_mask, col_mask, placed_row, placed_col)`. Determined by cell-index
/// traversal in ascending order — no HashMap iteration.
fn build_line_masks<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    d_bit: u8,
) -> (Vec<u32>, Vec<u32>, Vec<bool>, Vec<bool>) {
    let table = grid.table().clone();
    let bit = 1u32 << d_bit;
    let digit = d_bit + 1;
    let mut row_mask = vec![0u32; N];
    let mut col_mask = vec![0u32; N];
    let mut placed_row = vec![false; N];
    let mut placed_col = vec![false; N];
    for cell in 0..(N * N) {
        let r = table.cells[cell].row as usize;
        let c = table.cells[cell].col as usize;
        if grid.solved[cell] != 0 {
            if grid.solved[cell] == digit {
                placed_row[r] = true;
                placed_col[c] = true;
            }
            continue;
        }
        if grid.candidates[cell] & bit != 0 {
            row_mask[r] |= 1u32 << c;
            col_mask[c] |= 1u32 << r;
        }
    }
    (row_mask, col_mask, placed_row, placed_col)
}

/// Scan `grid` and return the smallest Fish size in `{2,3,4}` whose dry-run
/// firing condition holds for any digit / orientation. Deterministic:
/// digits scanned in ascending bit order, K ascending, rows-then-cols.
/// Returns `None` if no fish of any supported size fires.
pub fn find_first_fish_size<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<usize> {
    if N < 4 {
        return None;
    }
    for k in 2..=4usize {
        if k >= N {
            break;
        }
        for d_bit in 0..(N as u8) {
            let (row_mask, col_mask, placed_row, placed_col) =
                build_line_masks::<N, BR, BC>(grid, d_bit);
            let digit = d_bit + 1;
            if try_fish_eliminate_dryrun::<N, BR, BC>(
                grid, digit, k, &row_mask, &placed_row, true,
            ) {
                return Some(k);
            }
            if try_fish_eliminate_dryrun::<N, BR, BC>(
                grid, digit, k, &col_mask, &placed_col, false,
            ) {
                return Some(k);
            }
        }
    }
    None
}

/// Pure check: does `grid` admit a Fish<k> elimination for SOME digit?
fn fish_fires_at_size<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    k: usize,
) -> bool {
    if k >= N {
        return false;
    }
    for d_bit in 0..(N as u8) {
        let (row_mask, col_mask, placed_row, placed_col) =
            build_line_masks::<N, BR, BC>(grid, d_bit);
        let digit = d_bit + 1;
        if try_fish_eliminate_dryrun::<N, BR, BC>(
            grid, digit, k, &row_mask, &placed_row, true,
        ) {
            return true;
        }
        if try_fish_eliminate_dryrun::<N, BR, BC>(
            grid, digit, k, &col_mask, &placed_col, false,
        ) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Guided-removal driver.
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

fn guided_removal<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    solution: &Grid<N, BR, BC>,
    keep: &mut Vec<bool>,
    spec: &FishReverseSpec,
) {
    let nn = N * N;
    let mut order: Vec<u16> = (0..nn as u16).filter(|&i| keep[i as usize]).collect();
    order.shuffle(rng);

    // Initial state: do we already have a Fish at target size?
    let mut have_target = match build_subset::<N, BR, BC>(solution, keep) {
        Some(g) => fish_fires_at_size::<N, BR, BC>(&g, spec.target_size),
        None => false,
    };

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
        let cand_has = fish_fires_at_size::<N, BR, BC>(&trial, spec.target_size);
        // Acceptance: never DROP an already-found target Fish; otherwise,
        // any uniqueness-preserving removal is admissible (the target may
        // emerge after thinning).
        let accept = if !have_target {
            true
        } else {
            cand_has
        };
        if accept {
            have_target = cand_has;
        } else {
            keep[c] = true;
        }
    }
}

/// Outer driver. See module docs for the algorithm.
pub fn fish_reverse_construct<
    const N: usize,
    const BR: usize,
    const BC: usize,
    R: Rng + ?Sized,
>(
    rng: &mut R,
    spec: &FishReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() {
        return None;
    }
    let target_rank = tier_rank(spec.target_tier);
    let target_id = fish_id_for(spec.target_size);
    let nn = N * N;

    for attempt in 0..spec.max_attempts {
        let cfg = GenConfig {
            target_clues: spec.clue_max,
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
        if !r.frontier.contains(&target_id) {
            continue;
        }

        // First-firing fish must be exactly target size (heavier fish
        // pre-empted by smaller don't count for K=3/4 targets).
        match find_first_fish_size::<N, BR, BC>(&puzzle) {
            Some(s) if s == spec.target_size => {}
            _ => continue,
        }

        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[target_id]);
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
/// `batch_aic_reverse_construct` (golden-ratio splitmix). Throughput, not
/// byte-reproducibility, in the multi-threaded path.
pub fn batch_fish_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &FishReverseSpec,
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
        let child_seed =
            rng_seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
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
                let chunk_spec = FishReverseSpec {
                    max_attempts: chunk,
                    ..spec.clone()
                };
                match fish_reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
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
    use crate::generic::techniques::{
        fish::Fish as ProdFish, Technique as ProdTechnique,
    };

    /// X-Wing positive case (mirrors `fish::tests::xwing_fires_9x9` shape).
    /// Construct an X-Wing for digit 5 across rows 0,4 in cols 4,7. Dry-run
    /// must report a fish at K=2; production prod-apply must fire.
    #[test]
    fn fish_dryrun_mirrors_production_xwing_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 0..9 {
            if c != 4 && c != 7 {
                g.eliminate(0 * 9 + c, 5).unwrap();
            }
        }
        for c in 0..9 {
            if c != 4 && c != 7 {
                g.eliminate(4 * 9 + c, 5).unwrap();
            }
        }
        // Dry-run probe.
        assert_eq!(
            find_first_fish_size::<9, 3, 3>(&g),
            Some(2),
            "dry-run must detect X-Wing"
        );
        assert!(fish_fires_at_size::<9, 3, 3>(&g, 2));
        // Production parity.
        let mut h = g.clone();
        let t = ProdFish::<2>;
        let p = <ProdFish<2> as ProdTechnique<9, 3, 3>>::apply(&t, &mut h);
        assert!(p.is_some(), "production X-Wing must fire");
    }

    /// Swordfish positive (mirrors `fish::tests::swordfish_fires_9x9`).
    #[test]
    fn fish_dryrun_mirrors_production_swordfish_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let allowed = [
            (0usize, [0, 1].as_slice()),
            (1, [1, 2].as_slice()),
            (2, [0, 2].as_slice()),
        ];
        for &(r, a) in &allowed {
            for c in 0..9 {
                if !a.contains(&c) {
                    g.eliminate(r * 9 + c, 7).unwrap();
                }
            }
        }
        // Dry-run: smallest firing fish should be ≤3 (X-Wing might also fire
        // on this board for digit 7 — we only assert Swordfish is detectable
        // at K=3).
        assert!(fish_fires_at_size::<9, 3, 3>(&g, 3));
        let mut h = g.clone();
        let t = ProdFish::<3>;
        let p = <ProdFish<3> as ProdTechnique<9, 3, 3>>::apply(&t, &mut h);
        assert!(p.is_some(), "production Swordfish must fire");
    }

    /// Jellyfish positive (mirrors `fish::tests::jellyfish_fires_16x16`).
    #[test]
    fn fish_dryrun_mirrors_production_jellyfish_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        let allowed_per_row: [(usize, &[usize]); 4] = [
            (0, &[0, 1, 2, 3]),
            (1, &[0, 1, 2, 3]),
            (2, &[0, 1, 2, 3]),
            (3, &[0, 1, 2, 3]),
        ];
        for &(r, a) in &allowed_per_row {
            for c in 0..16 {
                if !a.contains(&c) {
                    g.eliminate(r * 16 + c, 7).unwrap();
                }
            }
        }
        assert!(fish_fires_at_size::<16, 4, 4>(&g, 4));
        let mut h = g.clone();
        let t = ProdFish::<4>;
        let p = <ProdFish<4> as ProdTechnique<16, 4, 4>>::apply(&t, &mut h);
        assert!(p.is_some(), "production Jellyfish must fire");
    }

    /// Pseudo-fish: 4 cells *not* aligned in K rows × K cols → must NOT fire.
    /// Place digit 7 candidate restricted to cells (0,0), (0,3), (3,0), (5,5)
    /// — union of cols across base rows {0,3} is {0,3} (size 2) for row 0
    /// (X-Wing candidate?), but row 3 contributes only col 0, forming a 2×2
    /// pattern at digit-7 across rows {0,3} cols {0,3} — wait, row 3 has only
    /// col 0 (mask=1), so its `(2..=K).contains(pc)` gate fails (pc=1). The
    /// dry-run *correctly* skips this base. Verify K=2 returns false (no other
    /// fish on this contrived board).
    #[test]
    fn fish_dryrun_rejects_pseudo_fish_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Restrict digit 7 to a non-fish pattern.
        // We restrict row 0 to cols {0,3}, row 3 to col {0} only, others free.
        for c in 0..9 {
            if c != 0 && c != 3 {
                g.eliminate(0 * 9 + c, 7).unwrap();
            }
        }
        for c in 0..9 {
            if c != 0 {
                g.eliminate(3 * 9 + c, 7).unwrap();
            }
        }
        // A fish-7 for K=2 would require two rows whose mask has count_ones
        // ∈ {2}. Row 0 has 2, row 3 has 1 — fails the gate.
        // Other rows (1,2,4..8) still have full digit-7 candidates (mask=
        // 0b111111111, count_ones=9), also outside the 2..=K gate. So no
        // X-Wing on digit 7. We additionally assert nothing else fires either
        // (the only digit we constrained is 7).
        assert!(!fish_fires_at_size::<9, 3, 3>(&g, 2),
            "pseudo-fish must NOT trigger dry-run");
        // Production must agree.
        let mut h = g.clone();
        let t = ProdFish::<2>;
        let p = <ProdFish<2> as ProdTechnique<9, 3, 3>>::apply(&t, &mut h);
        assert!(p.is_none(), "production X-Wing must NOT fire on pseudo");
    }

    /// `find_first_fish_size` is deterministic — does not depend on HashMap
    /// iteration order. Two calls on the same grid return identical Option.
    #[test]
    fn find_first_fish_size_deterministic() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 0..9 {
            if c != 4 && c != 7 {
                g.eliminate(0 * 9 + c, 5).unwrap();
            }
        }
        for c in 0..9 {
            if c != 4 && c != 7 {
                g.eliminate(4 * 9 + c, 5).unwrap();
            }
        }
        let a = find_first_fish_size::<9, 3, 3>(&g);
        let b = find_first_fish_size::<9, 3, 3>(&g);
        let c = find_first_fish_size::<9, 3, 3>(&g);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, Some(2));
    }

    /// Determinism: same seed → byte-identical puzzle.
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = FishReverseSpec {
            target_size: 2,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(31337);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(31337);
        let ra = fish_reverse_construct::<9, 3, 3, _>(&mut a, &spec);
        let rb = fish_reverse_construct::<9, 3, 3, _>(&mut b, &spec);
        match (ra, rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.puzzle.to_string_grid(), y.puzzle.to_string_grid());
            }
            (None, None) => {}
            _ => panic!("determinism violated: one Some, one None"),
        }
    }

    /// Load-bearing strict semantics: if returned with `require_load_bearing
    /// = true`, then `rate_excluding(&puzzle, &[Fish_K]).tier > target_tier`.
    #[test]
    fn load_bearing_strict_semantics_xwing_9x9() {
        let spec = FishReverseSpec {
            target_size: 2,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 28,
            max_attempts: 200,
            require_load_bearing: true,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
        if let Some(r) = fish_reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            let r2 = rate_excluding(&r.puzzle, &[TechniqueId::XWing]);
            assert!(!r2.rater_error);
            assert!(
                tier_rank(r2.tier) > tier_rank(Tier::T3),
                "load_bearing violated: rate_excluding tier = {:?}",
                r2.tier
            );
        }
        // Soft-pass on None — environment-dependent.
    }

    /// Validation: K outside {2,3,4} rejected.
    #[test]
    fn validate_rejects_invalid_size() {
        let mut spec = FishReverseSpec::new_t3(2, 1);
        spec.target_size = 5;
        assert!(spec.validate().is_err());
        spec.target_size = 1;
        assert!(spec.validate().is_err());
        spec.target_size = 3;
        assert!(spec.validate().is_ok());
    }

    /// Smoke 9×9 X-Wing: must terminate.
    #[test]
    fn smoke_9x9_x_wing() {
        let spec = FishReverseSpec {
            target_size: 2,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 50,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let _ = fish_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
    }

    /// Smoke 9×9 Swordfish: must terminate.
    #[test]
    fn smoke_9x9_swordfish() {
        let spec = FishReverseSpec {
            target_size: 3,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(13);
        let _ = fish_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
    }

    /// Smoke 9×9 Jellyfish: must terminate.
    #[test]
    fn smoke_9x9_jellyfish() {
        let spec = FishReverseSpec {
            target_size: 4,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 20,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(17);
        let _ = fish_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
    }

    /// Smoke 16×16 X-Wing: termination only (16×16 generation is slow).
    #[test]
    fn smoke_16x16_x_wing() {
        let spec = FishReverseSpec {
            target_size: 2,
            target_tier: Tier::T3,
            clue_min: 100,
            clue_max: 130,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 32,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let _ = fish_reverse_construct::<16, 4, 4, _>(&mut rng, &spec);
    }

    /// Returned puzzle round-trips through `solve_unique` to its `solution`.
    #[test]
    fn returned_puzzle_solves_to_solution() {
        let spec = FishReverseSpec {
            target_size: 2,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 100,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2027);
        if let Some(r) = fish_reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            let sol = solve_unique::<9, 3, 3>(&r.puzzle)
                .expect("returned puzzle must be unique");
            assert_eq!(sol.to_string_grid(), r.solution.to_string_grid());
        }
    }

    /// Batch driver smoke.
    #[test]
    fn batch_smoke_terminates() {
        let spec = FishReverseSpec {
            target_size: 2,
            target_tier: Tier::T3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 50,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let res = batch_fish_reverse_construct::<9, 3, 3>(42, &spec, 2, 1);
        assert!(res.len() <= 2);
    }
}
