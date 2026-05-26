//! Generic random unique-solution sudoku generator.
//!
//! Mirrors `crate::generator::gen_unique_puzzle` for the const-generic
//! substrate. Pipeline:
//!   1. `random_solution` — start from an empty grid and fill via
//!      `search_random` (MRV + shuffled digit order). For 9×9 the legacy
//!      generator additionally seeds row 0 with a shuffled permutation; we
//!      don't replicate that here because `BR*BC == N` doesn't always permit
//!      a single all-distinct row outside the natural-order case (the
//!      randomized DFS already explores the same space — it's slightly slower
//!      to first descent but identical in distribution).
//!   2. `gen_unique_puzzle` — start with the full solution; visit cells in
//!      shuffled order; for each, try clearing it and check that the
//!      remaining grid still has exactly one solution
//!      (`count_solutions_up_to(&trial, 2) == 1`); if so, accept the removal.
//!      Stop early once we've reached `cfg.target_clues`. There is no infinite
//!      loop: the outer pass visits each of the `N*N` cells at most once and
//!      then returns whatever clue count we landed at — even if the target
//!      was unreachable for that solution (e.g. 6×6, target 4: terminates at
//!      whatever uniqueness floor that particular solution allowed).
//!
//! Notes:
//!   - The `cfg.max_attempts` cap is a safety knob; under the current
//!     deterministic visit-each-cell-once strategy total trials are bounded
//!     by `N*N`, so we use it only as a hard ceiling on cell-removal trials
//!     (set to `0` to mean "no extra cap"; the natural `N*N` bound still
//!     applies).
//!   - RNG consumption is deterministic for a fixed seed within a single
//!     build of this crate (no `HashMap` iteration, no global state read in
//!     this module). Cross-version reproducibility is *not* guaranteed.

use rand::seq::SliceRandom;
use rand::Rng;

use super::grid::Grid;
use super::search::{count_solutions_up_to, search_random};

/// Tunables for `gen_unique_puzzle`.
#[derive(Debug, Clone, Copy)]
pub struct GenConfig {
    /// Soft target clue count. The generator stops removing cells once the
    /// puzzle has at most this many clues. The actual clue count may exceed
    /// this if uniqueness can't be preserved at lower counts for the sampled
    /// solution.
    pub target_clues: u32,
    /// Hard cap on the number of cell-removal *trials* (each cell is tried at
    /// most once anyway, so values >= N*N are equivalent to "no cap"). `0`
    /// disables the cap (still bounded by N*N from the visit order).
    pub max_attempts: u32,
}

impl GenConfig {
    pub fn new(target_clues: u32) -> Self {
        GenConfig {
            target_clues,
            max_attempts: 0,
        }
    }
}

/// Sample a fully-solved random grid of size `(N, BR, BC)`.
pub fn random_solution<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
) -> Grid<N, BR, BC> {
    let mut g: Grid<N, BR, BC> = Grid::empty();
    search_random(&mut g, rng).expect("empty grid is satisfiable");
    g
}

/// Generate one random unique-solution puzzle. Returns the puzzle and its
/// final clue count.
///
/// Termination: the loop visits each of the `N*N` cells at most once (see
/// shuffled `order`); after one pass we exit with whatever clue count we
/// reached. Both `target_clues` and `max_attempts` are *early-exit* guards;
/// the natural `N*N` bound on attempts ensures we always return.
pub fn gen_unique_puzzle<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    cfg: &GenConfig,
) -> (Grid<N, BR, BC>, u32) {
    let (puzzle, clue_count, _solution) = gen_unique_puzzle_with_solution::<N, BR, BC, R>(rng, cfg);
    (puzzle, clue_count)
}

/// Same as [`gen_unique_puzzle`] but also returns the full solution grid that
/// the puzzle was carved from. Lets downstream callers (reverse-construct
/// drivers) skip a redundant `solve_unique` call on the freshly generated
/// puzzle, since the solution is already known internally.
pub fn gen_unique_puzzle_with_solution<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    cfg: &GenConfig,
) -> (Grid<N, BR, BC>, u32, Grid<N, BR, BC>) {
    let nn = N * N;
    let solution: Grid<N, BR, BC> = random_solution::<N, BR, BC, R>(rng);
    let mut puzzle = solution.clone();
    let mut order: Vec<u16> = (0..nn as u16).collect();
    order.shuffle(rng);

    let mut clue_count = nn as u32;
    let mut trials: u32 = 0;
    for &c in &order {
        if clue_count <= cfg.target_clues {
            break;
        }
        if cfg.max_attempts != 0 && trials >= cfg.max_attempts {
            break;
        }
        let c = c as usize;
        if puzzle.solved[c] == 0 {
            continue;
        }
        // Build a trial grid from the current puzzle minus cell `c`.
        let mut trial: Grid<N, BR, BC> = Grid::empty();
        let mut feasible = true;
        for i in 0..nn {
            if i == c {
                continue;
            }
            let d = puzzle.solved[i];
            if d == 0 {
                continue;
            }
            if trial.assign(i, d).is_err() {
                // Should not happen: we're rebuilding a known-consistent
                // subset, but stay defensive.
                feasible = false;
                break;
            }
        }
        trials += 1;
        if !feasible {
            continue;
        }
        if count_solutions_up_to(&trial, 2) == 1 {
            puzzle = trial;
            clue_count -= 1;
        }
    }
    (puzzle, clue_count, solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::search::{count_solutions_up_to, solve_unique};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn smoke_size<const N: usize, const BR: usize, const BC: usize>(
        seed: u64,
        target: u32,
        clue_lo: u32,
        clue_hi: u32,
    ) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let cfg = GenConfig::new(target);
        let mut total = 0u32;
        for k in 0..10 {
            let (p, clues): (Grid<N, BR, BC>, u32) = gen_unique_puzzle(&mut rng, &cfg);
            assert_eq!(
                count_solutions_up_to(&p, 2),
                1,
                "size {} sample {} not unique (clues={})",
                N,
                k,
                clues
            );
            assert!(solve_unique(&p).is_some(), "solve_unique failed at N={}", N);
            assert!(
                clues >= clue_lo && clues <= clue_hi,
                "size {} sample {}: clues {} not in [{}, {}]",
                N,
                k,
                clues,
                clue_lo,
                clue_hi
            );
            total += clues;
        }
        let mean = total as f64 / 10.0;
        eprintln!("smoke N={} mean clues={:.1}", N, mean);
    }

    #[test]
    fn smoke_6x6() {
        smoke_size::<6, 2, 3>(11, 15, 12, 22);
    }

    #[test]
    fn smoke_9x9() {
        smoke_size::<9, 3, 3>(12, 30, 25, 40);
    }

    #[test]
    fn smoke_12x12() {
        smoke_size::<12, 3, 4>(13, 60, 48, 80);
    }

    #[test]
    fn smoke_16x16() {
        smoke_size::<16, 4, 4>(14, 120, 100, 160);
    }

    #[test]
    fn determinism_same_seed_same_puzzle_9x9() {
        let cfg = GenConfig::new(30);
        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(1234);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(1234);
        let (a, ca): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng_a, &cfg);
        let (b, cb): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng_b, &cfg);
        assert_eq!(ca, cb);
        assert_eq!(a.to_string_grid(), b.to_string_grid());
    }

    #[test]
    fn determinism_same_seed_same_puzzle_6x6() {
        let cfg = GenConfig::new(15);
        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(7);
        let (a, _): (Grid<6, 2, 3>, u32) = gen_unique_puzzle(&mut rng_a, &cfg);
        let (b, _): (Grid<6, 2, 3>, u32) = gen_unique_puzzle(&mut rng_b, &cfg);
        assert_eq!(a.to_string_grid(), b.to_string_grid());
    }

    #[test]
    fn random_solution_is_valid() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let g: Grid<9, 3, 3> = random_solution(&mut rng);
        assert!(g.is_solved());
        assert!(g.is_consistent());
    }

    #[test]
    fn random_solution_16x16_is_valid() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let g: Grid<16, 4, 4> = random_solution(&mut rng);
        assert!(g.is_solved());
        assert!(g.is_consistent());
    }

}
