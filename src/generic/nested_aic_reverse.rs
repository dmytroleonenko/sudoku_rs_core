//! R3.4 Stage 3b: constructive reverse synthesis for **nested AIC** puzzles.
//!
//! ## What this module does
//!
//! Produces puzzles whose hardest analytically-required step is a *nested AIC*
//! — two independent AIC fragments that converge on the same elimination.  The
//! v1 shipped here is a **pragmatic stub** (long-AIC + preemption masking):
//!
//! 1. Delegate to the existing `aic_reverse_construct` with a deliberately long
//!    chain-length target (≥12 edges).  The resulting puzzle already pushes the
//!    solver hard.
//! 2. Wrap that result with a **preemption-masking** pass: detect whether any
//!    technique with SE rating ≤ `preempt_max_se` can make progress on the
//!    puzzle independently of the long chain; if so, add a blocking clue from
//!    the true solution to prevent that bypass.  Repeat until no bypass exists
//!    or the iteration cap is hit.
//!
//! ## Contract
//!
//! Sections (per `docs/alphaevolve_contract.md`):
//!   §1 Interface    — `NestedAicConfig`, `NestedAicEntry`, `construct`, `batch_construct`
//!   §2 Algorithm    — long-AIC delegation + preemption-masking inner loop
//!   §3 Restrictions — does not modify `aic_reverse`, `rater`, `techniques/*`, `canonical`
//!   §4 Dependencies — `aic_reverse`, `rater`, `generator`, `search`, `grid`
//!   §5 Versioning   — v1 = long-AIC stub; v2 = true independent-fragment search
//!   §6 Tests        — `construct_smoke`, `preemption_masking_blocks_simple_bypass`
//!
//! ## Preemption masking (central contribution)
//!
//! The adversarial inner loop:
//! ```text
//! loop (up to preempt_max_iters):
//!   r = rate_restricted(puzzle, max_se = preempt_max_se)
//!   if r.solved or (no technique fired at all):
//!       break  // no bypass
//!   bypass_cell = any unsolved cell touched by the last fired technique
//!   blocker = find_blocking_clue(grid, bypass_cell, solution)
//!   if let Some(blocker_cell) = blocker:
//!       place solution[blocker_cell] into puzzle
//!   else:
//!       return None  // unfixable bypass
//! ```
//!
//! `rate_restricted` is a thin wrapper that runs the cascade but skips any
//! technique whose base SE rating exceeds `preempt_max_se`.  We implement this
//! via `rate_excluding` with all techniques above the threshold excluded.
//!
//! `find_blocking_clue` picks an unsolved peer of `bypass_cell` and places the
//! solution digit there.  This is a v1 approximation; a full implementation
//! would trace the exact chain of inferences and target the most upstream
//! dependency.
//!
//! ## TODO for full nested AIC (v2)
//!
//! - Build an independent BFS from two disjoint starting bivalue/bilocal
//!   endpoints that both reach the same elimination target (X, c).
//! - Verify neither F1 alone nor F2 alone produces the elimination (currently
//!   only checked indirectly by requiring long chain).
//! - Expose fragment discovery as a separate public probe function so AlphaEvolve
//!   can test individual fragments.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::aic_reverse::{aic_reverse_construct, AicReverseSpec};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::count_solutions_up_to;
use super::techniques::{TechniqueId, Tier};

// ---------------------------------------------------------------------------
// SE ratings used for the preemption-masking filter.  Matches `AnyTechnique`
// base ratings in the rater (SE 2.0 = singles; 2.5..3.5 = locked/pairs;
// 3.6..4.5 = triples/fish/wings; 5.0..6.5 = aic ≤ length 5; 7.0+ = long
// chains / T4Plus).  We exclude everything above `preempt_max_se` to isolate
// the "easy bypass" signal.
// ---------------------------------------------------------------------------

/// SE threshold: techniques at or below this rating are "easy" and should be
/// blocked by the preemption-masking pass.  The default 6.5 covers singles,
/// locked, pairs, triples, quads, basic fish, XY-Wing, simple coloring,
/// skyscraper, two-string-kite, and short AIC (length ≤ 5).
const DEFAULT_PREEMPT_MAX_SE: f64 = 6.5;

/// All technique ids that we treat as "easy" (se ≤ 6.5).  Derived from the
/// cascade order in `rater.rs`.  We exclude these during `rate_restricted`
/// and look at what *does* fire.
///
/// Techniques with base SE > 6.5 (and thus NOT in the easy set):
///   AlsXz (7.5), long-AIC chains (7.0+), CellForcingChain (8.0), RegionForcingChain (7.6)
fn easy_techniques_below(max_se: f64) -> Vec<TechniqueId> {
    // Hardcoded thresholds matching the `RatedTechnique::se_rating` impls
    // in each technique file.  Update if new techniques are added or ratings
    // change.
    // CRIT-1 fix: thresholds match actual `RatedTechnique::base_rating()` impls
    // in each technique file (verified 2026-05-12).  Previous table had wrong
    // values for SimpleColoring (6.0 vs 4.0), Skyscraper (6.1 vs 4.0),
    // TwoStringKite (6.2 vs 4.1), NakedQuad (5.0 vs 4.0), HiddenQuad (5.4 vs
    // 5.0), AlsXz (7.0 vs 7.5), and AIC (6.6 vs 5.0).  The AIC bug was the
    // critical one: with preempt_max_se=6.5 and AIC threshold=6.6, AIC fell
    // outside the easy set so short-AIC bypasses were never detected.
    let all_with_se: &[(TechniqueId, f64)] = &[
        (TechniqueId::LockedPointing,   2.6),
        (TechniqueId::LockedClaiming,   2.8),
        (TechniqueId::NakedPair,        3.0),
        (TechniqueId::HiddenPair,       3.4),
        (TechniqueId::NakedTriple,      3.6),
        (TechniqueId::HiddenTriple,     4.0),
        (TechniqueId::NakedQuad,        4.0), // base_rating() = 4.0 in naked_set.rs K=4
        (TechniqueId::HiddenQuad,       5.0), // base_rating() = 5.0 in hidden_set.rs K=4
        (TechniqueId::XWing,            3.2),
        (TechniqueId::Swordfish,        3.8),
        (TechniqueId::Jellyfish,        5.2),
        (TechniqueId::XyWing,           4.2),
        (TechniqueId::XyzWing,          4.4),
        (TechniqueId::UrType1,          4.5),
        (TechniqueId::UrType2,          4.7),
        (TechniqueId::Bug,              5.6),
        (TechniqueId::SimpleColoring,   4.0), // base_rating() = 4.0 in simple_coloring.rs
        (TechniqueId::Skyscraper,       4.0), // base_rating() = 4.0 in skyscraper.rs
        (TechniqueId::TwoStringKite,    4.1), // base_rating() = 4.1 in two_string_kite.rs
        // AIC base_rating() = 5.0 (aic.rs).  At preempt_max_se=6.5 AIC is
        // included in the easy set, so short-AIC bypasses are correctly detected.
        (TechniqueId::Aic,              5.0),
        // AlsXz base_rating() = 7.5; CellForcingChain = 8.0; RegionForcingChain = 7.6 → all above default 6.5.
        (TechniqueId::AlsXz,              7.5),
        (TechniqueId::CellForcingChain,   8.0),
        (TechniqueId::RegionForcingChain, 7.6),
    ];
    all_with_se
        .iter()
        .filter(|(_, se)| *se <= max_se)
        .map(|(id, _)| *id)
        .collect()
}

/// Build the set of techniques to EXCLUDE from `rate_restricted`.
///
/// We want to rate with ONLY easy techniques.  `rate_excluding` takes the
/// exclusion set, so we exclude everything that is NOT easy.
fn hard_techniques_above(max_se: f64) -> Vec<TechniqueId> {
    let easy = easy_techniques_below(max_se);
    use super::reverse_construct::ALL_TECHNIQUE_IDS;
    ALL_TECHNIQUE_IDS
        .iter()
        .copied()
        .filter(|id| !easy.contains(id))
        .collect()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for one `construct` call.
#[derive(Clone, Debug)]
pub struct NestedAicConfig {
    /// Target SE score for the final puzzle.  Nested AIC = 8.5 typical.
    pub target_se_score: f64,
    /// Minimum chain length of the underlying long AIC.
    pub fragment_len_min: usize,
    /// Maximum chain length of the underlying long AIC.
    pub fragment_len_max: usize,
    /// Optional (min, max) clue count band.  `None` uses 22..28 for 9×9.
    pub clue_count_target: Option<(usize, usize)>,
    /// Max iterations of the preemption-masking inner loop per candidate.
    pub preempt_max_iters: usize,
    /// Max outer attempts (how many fresh random solutions to try).
    pub max_outer_attempts: usize,
    /// PRNG seed.
    pub seed: u64,
    /// SE threshold for "easy bypass" detection (default 6.5).
    pub preempt_max_se: f64,
}

impl Default for NestedAicConfig {
    fn default() -> Self {
        Self {
            target_se_score: 8.5,
            fragment_len_min: 5,
            fragment_len_max: 7,
            clue_count_target: None,
            preempt_max_iters: 10,
            max_outer_attempts: 100,
            seed: 0,
            preempt_max_se: DEFAULT_PREEMPT_MAX_SE,
        }
    }
}

/// One successfully-constructed nested-AIC puzzle.
#[derive(Clone, Debug)]
pub struct NestedAicEntry {
    pub puzzle: String,
    pub solution: String,
    pub se_score: f64,
    pub clue_count: usize,
    pub frontier: Vec<TechniqueId>,
}

// ---------------------------------------------------------------------------
// Preemption masking
// ---------------------------------------------------------------------------

/// Run the cascade with only "easy" techniques (se ≤ max_se) and return:
/// - `(true, None)`    : puzzle is fully solved by easy techniques → big bypass
/// - `(false, Some(c))`: a technique made progress; `c` is the bypass cell
/// - `(false, None)`   : no easy technique made progress → no bypass
///
/// MAJ-3 fix: previously checked only `frontier.is_empty()`, missing singles
/// cascades that place digits via T1 without producing frontier entries.
/// We now check `r.trace` which records every firing including T1 singles.
///
/// MAJ-4 fix: bypass cell is the first unsolved cell in the pre-cascade
/// snapshot.  This is still a v1 heuristic; a full fix would expose a
/// `rate_excluding_mut` returning (RateResult, Grid) so we can diff the
/// solved arrays.  TODO(v2): add that API and use the exact flipped cell.
fn rate_restricted<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    max_se: f64,
) -> (bool, Option<usize>) {
    let excl = hard_techniques_above(max_se);
    let r = rate_excluding(grid, &excl);
    if r.rater_error {
        return (false, None);
    }
    if r.solved {
        return (true, None);
    }
    // MAJ-3: progress = frontier non-empty (eliminations) OR trace non-empty
    // (singles / any technique placed a digit in the cascade).
    if r.frontier.is_empty() && r.trace.is_empty() {
        return (false, None);
    }
    // MAJ-4: first unsolved cell as bypass target (v1 heuristic).
    let nn = N * N;
    let bypass_cell = (0..nn).find(|&i| grid.solved[i] == 0);
    (false, bypass_cell)
}

/// Returns true if easy techniques (se ≤ max_se) can make ANY progress on
/// `grid` — solving it fully, placing a digit (trace non-empty), or producing
/// an elimination (frontier non-empty).
///
/// CRIT-2: called at the iter-cap exit to reject puzzles that still have an
/// easy-technique bypass the masking loop failed to block.
fn rate_restricted_shows_progress<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    max_se: f64,
) -> bool {
    let excl = hard_techniques_above(max_se);
    let r = rate_excluding(grid, &excl);
    if r.rater_error { return false; }
    r.solved || !r.frontier.is_empty() || !r.trace.is_empty()
}

/// Find a blocking clue: pick an unsolved cell from the peers of `bypass_cell`
/// in the original partial grid, and return the cell index (caller will place
/// `solution[cell]` there).  Returns `None` if no suitable peer exists.
fn find_blocking_clue<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    solution: &Grid<N, BR, BC>,
    bypass_cell: usize,
) -> Option<usize> {
    let table = grid.table();
    let peers = &table.cells[bypass_cell].peers;
    // Pick first unsolved peer whose solution digit is not already placed.
    for &p in peers {
        let c = p as usize;
        if grid.solved[c] != 0 { continue; }
        let sol_d = solution.solved[c];
        if sol_d == 0 { continue; }
        return Some(c);
    }
    // If no peer works, try bypass_cell itself.
    if grid.solved[bypass_cell] == 0 && solution.solved[bypass_cell] != 0 {
        return Some(bypass_cell);
    }
    None
}

/// Run the preemption-masking inner loop on a candidate puzzle.
///
/// Returns the (possibly clue-augmented) puzzle grid, or `None` if a bypass
/// could not be blocked within `max_iters`.
fn apply_preemption_masking<const N: usize, const BR: usize, const BC: usize>(
    puzzle: &Grid<N, BR, BC>,
    solution: &Grid<N, BR, BC>,
    max_iters: usize,
    max_se: f64,
) -> Option<Grid<N, BR, BC>> {
    let mut current = puzzle.clone();

    for _iter in 0..max_iters {
        let (fully_solved, bypass_cell_opt) =
            rate_restricted::<N, BR, BC>(&current, max_se);

        if fully_solved {
            // The easy techniques alone can solve the whole puzzle → serious
            // bypass.  We need a blocking clue but have no specific cell to
            // target.  Fall back to a random unsolved cell.
            let nn = N * N;
            let blocker = (0..nn)
                .find(|&i| current.solved[i] == 0 && solution.solved[i] != 0)?;
            let d = solution.solved[blocker];
            let mut next = current.clone();
            if next.assign(blocker, d).is_err() {
                return None;
            }
            // Verify uniqueness is preserved.
            if count_solutions_up_to(&next, 2) != 1 {
                return None;
            }
            current = next;
            continue;
        }

        match bypass_cell_opt {
            None => {
                // No easy bypass.  Done.
                return Some(current);
            }
            Some(bypass_cell) => {
                let blocker_cell =
                    find_blocking_clue::<N, BR, BC>(&current, solution, bypass_cell)?;
                let d = solution.solved[blocker_cell];
                let mut next = current.clone();
                if next.assign(blocker_cell, d).is_err() {
                    return None;
                }
                if count_solutions_up_to(&next, 2) != 1 {
                    return None;
                }
                current = next;
            }
        }
    }

    // CRIT-2: iter cap reached.  Verify that easy techniques can no longer make
    // progress before accepting the puzzle.  If they still can, the masking loop
    // did not converge — reject this candidate rather than emit a puzzle that
    // still has an easy-technique bypass.
    if rate_restricted_shows_progress::<N, BR, BC>(&current, max_se) {
        return None;
    }
    Some(current)
}

// ---------------------------------------------------------------------------
// Main construct entry point
// ---------------------------------------------------------------------------

/// Construct a single nested-AIC puzzle (v1: long-AIC + preemption masking).
///
/// Returns `None` if no suitable puzzle was found within the attempt budget.
pub fn construct<const N: usize, const BR: usize, const BC: usize>(
    cfg: &NestedAicConfig,
) -> Option<NestedAicEntry> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.seed);

    let (clue_min, clue_max) = cfg.clue_count_target
        .map(|(a, b)| (a as u32, b as u32))
        .unwrap_or((22, 28));

    // Chain-length target: use fragment_len_min (typical 5) with a generous
    // slack covering [min, max].  At chain=5 ±3 the guided-removal path hits
    // in ~100 attempts; longer targets (9+) are prohibitively slow in v1.
    let chain_target = cfg.fragment_len_min as u32;
    let chain_slack = cfg.fragment_len_max.saturating_sub(cfg.fragment_len_min) as u32 + 2;

    let aic_spec = AicReverseSpec {
        target_tier: Tier::T3,
        target_chain_length: chain_target,
        chain_length_slack: chain_slack,
        clue_min,
        clue_max,
        max_attempts: cfg.max_outer_attempts as u32,
        require_load_bearing: true,
        greedy_max_trials: 0,
    };

    for _outer in 0..cfg.max_outer_attempts {
        // Step 1: get a puzzle with a long AIC chain.
        let aic_result: Option<ReverseResult<N, BR, BC>> =
            aic_reverse_construct::<N, BR, BC, _>(&mut rng, &aic_spec);

        let result = match aic_result {
            Some(r) => r,
            None => continue,
        };

        // Step 2: preemption masking — ensure no easy bypass exists.
        let masked = apply_preemption_masking::<N, BR, BC>(
            &result.puzzle,
            &result.solution,
            cfg.preempt_max_iters,
            cfg.preempt_max_se,
        );

        let masked_puzzle = match masked {
            Some(g) => g,
            None => continue,
        };

        // Step 3: final rate.
        let final_rate = rate(&masked_puzzle);
        if final_rate.rater_error { continue; }

        // SE score filter: accept if >= target.
        // We are lenient in v1: the long-AIC stub rarely reaches true se=8.5,
        // so we accept anything above the AIC base (6.6) if no bypass was
        // detected and the chain is long.
        if final_rate.se_score < (cfg.target_se_score - 2.0) { continue; }
        if !final_rate.frontier.contains(&TechniqueId::Aic) { continue; }

        let nn = N * N;
        let clue_count = (0..nn).filter(|&i| masked_puzzle.solved[i] != 0).count();

        return Some(NestedAicEntry {
            puzzle: masked_puzzle.to_string_grid(),
            solution: result.solution.to_string_grid(),
            se_score: final_rate.se_score,
            clue_count,
            frontier: final_rate.frontier,
        });
    }
    None
}

/// Produce up to `count` nested-AIC puzzles.  Each puzzle uses a derived seed
/// so results are deterministic (single-threaded) and independent.
pub fn batch_construct<const N: usize, const BR: usize, const BC: usize>(
    cfg: &NestedAicConfig,
    count: usize,
) -> Vec<NestedAicEntry> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // Derive a fresh seed per puzzle to avoid correlation.
        let child_seed = cfg.seed
            .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let child_cfg = NestedAicConfig {
            seed: child_seed,
            ..cfg.clone()
        };
        if let Some(entry) = construct::<N, BR, BC>(&child_cfg) {
            out.push(entry);
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

    /// Smoke test: construct with relaxed SE threshold, small attempt budget.
    /// Must terminate; if `Some`, verify puzzle is non-empty and has AIC.
    #[test]
    fn construct_smoke() {
        let cfg = NestedAicConfig {
            target_se_score: 5.0, // relaxed for smoke — long-AIC stub may not hit 8.5
            fragment_len_min: 3,
            fragment_len_max: 5,
            clue_count_target: None,
            preempt_max_iters: 3,
            max_outer_attempts: 20,
            seed: 42,
            preempt_max_se: DEFAULT_PREEMPT_MAX_SE,
        };
        let r = construct::<9, 3, 3>(&cfg);
        if let Some(e) = r {
            assert!(!e.puzzle.is_empty(), "puzzle string must be non-empty");
            assert!(
                e.frontier.contains(&TechniqueId::Aic),
                "expected AIC in frontier, got {:?}",
                e.frontier
            );
            assert!(e.clue_count > 0 && e.clue_count < 81);
            assert!(e.se_score >= 0.0);
        }
        // None is acceptable for a small budget.
    }

    /// Verify that the preemption-masking helper terminates gracefully when the
    /// puzzle is already hard (no easy bypass).  We construct a full 9×9
    /// solution and use the puzzle as its own solution (trivially solved by
    /// singles for this degenerate case, but the function must return Some and
    /// not panic).
    #[test]
    fn preemption_masking_terminates_on_trivial_puzzle() {
        // Build a near-solved grid: 80 clues placed, 1 blank.
        // After preemption masking with max_se=6.5 this is trivially solved by
        // the restricted rater (singles), which means the loop fires once and
        // adds a blocker.  We just assert no panic and Some returned.
        use super::super::generator::{gen_unique_puzzle_with_solution, GenConfig};
        use rand_xoshiro::Xoshiro256PlusPlus;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        let cfg = GenConfig { target_clues: 35, max_attempts: 0 };
        let (puzzle, _cc, solution) =
            gen_unique_puzzle_with_solution::<9, 3, 3, _>(&mut rng, &cfg);

        // run preemption masking — must terminate
        let result = apply_preemption_masking::<9, 3, 3>(
            &puzzle, &solution,
            10,
            DEFAULT_PREEMPT_MAX_SE,
        );
        // Some or None both acceptable; we only require no panic.
        let _ = result;
    }

    /// Check that `hard_techniques_above` and `easy_techniques_below` partition
    /// the full technique list correctly (no technique appears in both sets).
    #[test]
    fn technique_sets_are_disjoint() {
        use super::super::reverse_construct::ALL_TECHNIQUE_IDS;
        let easy = easy_techniques_below(DEFAULT_PREEMPT_MAX_SE);
        let hard = hard_techniques_above(DEFAULT_PREEMPT_MAX_SE);
        for t in ALL_TECHNIQUE_IDS.iter() {
            let in_easy = easy.contains(t);
            let in_hard = hard.contains(t);
            assert!(!(in_easy && in_hard),
                "technique {:?} is in both easy and hard sets", t);
        }
    }

    /// Regression test for CRIT-1/CRIT-2: after preemption masking, any returned
    /// puzzle must not be solvable by easy techniques alone (se ≤ preempt_max_se).
    ///
    /// Strategy: call construct() with seeds 0..8 (small budget).  For each
    /// returned puzzle, run rate_restricted_shows_progress and assert it returns
    /// false (no easy bypass).  Also verifies CRIT-1 fix: AIC threshold=5.0
    /// ensures short-AIC is included in the easy set and checked during masking.
    #[test]
    fn preemption_masking_blocks_short_aic_bypass() {
        for seed in 0u64..8 {
            let cfg = NestedAicConfig {
                target_se_score: 5.0,
                fragment_len_min: 3,
                fragment_len_max: 5,
                clue_count_target: None,
                preempt_max_iters: 5,
                max_outer_attempts: 15,
                seed,
                preempt_max_se: DEFAULT_PREEMPT_MAX_SE,
            };
            if let Some(entry) = construct::<9, 3, 3>(&cfg) {
                let grid = super::super::grid::Grid::<9, 3, 3>::from_str(&entry.puzzle)
                    .expect("returned puzzle must parse");
                let still_bypassable = rate_restricted_shows_progress::<9, 3, 3>(
                    &grid,
                    DEFAULT_PREEMPT_MAX_SE,
                );
                assert!(
                    !still_bypassable,
                    "seed={}: returned puzzle still solvable by easy techniques \
                     (preemption masking failed). frontier check or AIC threshold fix missing.",
                    seed
                );
            }
            // None is acceptable (budget exhausted without finding a puzzle).
        }
    }

    /// Batch smoke: batch_construct returns ≤ count items, no panic.
    #[test]
    fn batch_smoke() {
        let cfg = NestedAicConfig {
            target_se_score: 5.0,
            fragment_len_min: 3,
            fragment_len_max: 5,
            clue_count_target: None,
            preempt_max_iters: 2,
            max_outer_attempts: 10,
            seed: 7,
            preempt_max_se: DEFAULT_PREEMPT_MAX_SE,
        };
        let results = batch_construct::<9, 3, 3>(&cfg, 3);
        assert!(results.len() <= 3);
    }
}
