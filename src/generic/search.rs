//! Generic backtracking search over `Grid<N, BR, BC>`.
//!
//! Mirrors the role of `crate::backtracker::{solve_unique, count_solutions_up_to}`
//! for the const-generic substrate. Uses the singles propagator from
//! `super::backtracker::propagate_singles` plus a simple MRV (smallest-domain)
//! cell-pick. Branching is deterministic in `count_solutions_up_to` /
//! `solve_unique` and randomized in `search_random`.
//!
//! Public API:
//!   - `count_solutions_up_to(&Grid, limit) -> usize` — short-circuits as soon
//!     as `limit` distinct solutions have been found. Used for uniqueness
//!     checks (limit=2).
//!   - `solve_unique(&Grid) -> Option<Grid>` — returns the unique completion
//!     iff exactly one exists.
//!   - `search_random(&mut Grid, &mut Rng) -> Result<(), ()>` — randomized
//!     completion: at each branch we pick an MRV cell, shuffle the surviving
//!     digits, and recurse. On contradiction we backtrack and try the next
//!     digit; only if all digits at every branch fail do we return `Err(())`.
//!     Completeness is preserved (every digit at every branch is eventually
//!     attempted on retry — randomization only changes the order).

use rand::seq::SliceRandom;
use rand::Rng;

use super::backtracker::propagate_singles;
use super::grid::Grid;

/// MRV: pick the unsolved cell with the smallest candidate-domain. Returns
/// `(cell, mask)` or `None` if there is no unsolved cell (i.e. the grid is
/// fully solved). Ties broken by lowest index — deterministic.
#[inline]
fn pick_mrv<const N: usize, const BR: usize, const BC: usize>(
    g: &Grid<N, BR, BC>,
) -> Option<(usize, u32)> {
    let nn = N * N;
    let mut best: Option<(usize, u32, u32)> = None; // (cell, mask, popcount)
    for c in 0..nn {
        if g.solved[c] != 0 {
            continue;
        }
        let m = g.candidates[c];
        let pc = m.count_ones();
        match best {
            None => best = Some((c, m, pc)),
            Some((_, _, p)) if pc < p => best = Some((c, m, pc)),
            _ => {}
        }
        // Can't do better than 1; early-out is tempting but `pc==0` would have
        // been caught by propagate_singles. Leave loop simple.
    }
    best.map(|(c, m, _)| (c, m))
}

/// Recursive deterministic counter. Increments `count` for each solution and
/// short-circuits when `count >= limit`. Tries digits in ascending bit order.
/// Increments `branches` once per attempted digit at a branching cell — same
/// semantics as the legacy `count_solutions_with_steps` step counter.
fn count_search<const N: usize, const BR: usize, const BC: usize>(
    g: &mut Grid<N, BR, BC>,
    limit: usize,
    count: &mut usize,
    capture: Option<&mut Option<Grid<N, BR, BC>>>,
    branches: &mut u64,
) {
    if *count >= limit {
        return;
    }
    if propagate_singles(g).is_err() {
        return;
    }
    if g.is_solved() {
        if let Some(cap) = capture {
            if cap.is_none() {
                *cap = Some(g.clone());
            }
        }
        *count += 1;
        return;
    }
    let (cell, mask) = match pick_mrv(g) {
        Some(x) => x,
        None => return,
    };
    let mut bits = mask;
    // Re-borrow capture each recursion so we can pass it down without moving.
    let mut cap_ref = capture;
    while bits != 0 && *count < limit {
        let bit = bits & bits.wrapping_neg();
        bits ^= bit;
        let d = bit.trailing_zeros() as u8 + 1;
        let mut child = g.clone();
        if child.assign(cell, d).is_ok() {
            *branches += 1;
            count_search(&mut child, limit, count, cap_ref.as_deref_mut(), branches);
        }
    }
}

/// Count solutions of `grid`, stopping as soon as `limit` have been found.
/// `limit == 0` returns 0 trivially. For uniqueness checks pass `limit = 2`.
pub fn count_solutions_up_to<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    limit: usize,
) -> usize {
    if limit == 0 {
        return 0;
    }
    let mut work = grid.clone();
    let mut count: usize = 0;
    let mut branches: u64 = 0;
    // No solution capture — pure count path skips the per-solution clone.
    count_search(&mut work, limit, &mut count, None, &mut branches);
    count
}

/// Same as [`count_solutions_up_to`] but also returns the cumulative number of
/// branch steps taken (each successful `assign` at a branching cell counts as
/// one). Mirrors `crate::backtracker::count_solutions_with_steps`.
pub fn count_solutions_with_steps<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    limit: usize,
) -> (usize, u64) {
    if limit == 0 {
        return (0, 0);
    }
    let mut work = grid.clone();
    let mut count: usize = 0;
    let mut branches: u64 = 0;
    count_search(&mut work, limit, &mut count, None, &mut branches);
    (count, branches)
}

/// Returns `Some(solution)` iff `grid` has exactly one completion.
pub fn solve_unique<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<Grid<N, BR, BC>> {
    let mut work = grid.clone();
    let mut count: usize = 0;
    let mut cap: Option<Grid<N, BR, BC>> = None;
    let mut branches: u64 = 0;
    count_search(&mut work, 2, &mut count, Some(&mut cap), &mut branches);
    if count == 1 {
        cap
    } else {
        None
    }
}

/// Randomized DFS to a fully-solved grid. On entry `g` may be partially
/// constrained (e.g. empty); on `Ok(())` `g` holds a valid completion. On
/// `Err(())` no completion exists from the input state (caller's input was
/// already over-constrained — never happens for a freshly-created empty grid).
///
/// The search is complete: at every branch we attempt every surviving digit in
/// shuffled order, only returning `Err` when *all* of them fail.
pub fn search_random<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    g: &mut Grid<N, BR, BC>,
    rng: &mut R,
) -> Result<(), ()> {
    if propagate_singles(g).is_err() {
        return Err(());
    }
    if g.is_solved() {
        return Ok(());
    }
    let (cell, mask) = match pick_mrv(g) {
        Some(x) => x,
        None => return Err(()),
    };
    // Collect surviving digits and shuffle.
    let mut digits: Vec<u8> = Vec::with_capacity(N);
    let mut bits = mask;
    while bits != 0 {
        let bit = bits & bits.wrapping_neg();
        bits ^= bit;
        digits.push(bit.trailing_zeros() as u8 + 1);
    }
    digits.shuffle(rng);
    for d in digits {
        let mut child = g.clone();
        if child.assign(cell, d).is_ok() {
            if search_random(&mut child, rng).is_ok() {
                *g = child;
                return Ok(());
            }
        }
    }
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn count_solutions_easy_9x9_unique() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        assert_eq!(count_solutions_up_to(&g, 2), 1);
        let sol = solve_unique(&g).expect("unique");
        assert!(sol.is_solved());
        assert!(sol.is_consistent());
    }

    #[test]
    fn count_solutions_empty_short_circuits() {
        let g: Grid<9, 3, 3> = Grid::empty();
        // many solutions; we only care that we cap at limit
        assert_eq!(count_solutions_up_to(&g, 2), 2);
        assert_eq!(count_solutions_up_to(&g, 5), 5);
    }

    #[test]
    fn count_solutions_zero_limit() {
        let g: Grid<9, 3, 3> = Grid::empty();
        assert_eq!(count_solutions_up_to(&g, 0), 0);
    }

    #[test]
    fn solve_unique_none_for_empty() {
        let g: Grid<9, 3, 3> = Grid::empty();
        assert!(solve_unique(&g).is_none());
    }

    #[test]
    fn search_random_completes_empty_9x9() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut g: Grid<9, 3, 3> = Grid::empty();
        search_random(&mut g, &mut rng).unwrap();
        assert!(g.is_solved());
        assert!(g.is_consistent());
    }

    #[test]
    fn search_random_completes_empty_6x6() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2);
        let mut g: Grid<6, 2, 3> = Grid::empty();
        search_random(&mut g, &mut rng).unwrap();
        assert!(g.is_solved());
        assert!(g.is_consistent());
    }

    #[test]
    fn search_random_completes_empty_12x12() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        let mut g: Grid<12, 3, 4> = Grid::empty();
        search_random(&mut g, &mut rng).unwrap();
        assert!(g.is_solved());
        assert!(g.is_consistent());
    }

    #[test]
    fn search_random_completes_empty_16x16() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let mut g: Grid<16, 4, 4> = Grid::empty();
        search_random(&mut g, &mut rng).unwrap();
        assert!(g.is_solved());
        assert!(g.is_consistent());
    }

    #[test]
    fn count_solutions_unique_easy_9x9() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        assert_eq!(count_solutions_up_to(&g, 5), 1);
    }
}
