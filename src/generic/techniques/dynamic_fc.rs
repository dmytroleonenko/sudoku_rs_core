//! # Dynamic Forcing Chain (DynamicFC)
//!
//! ## Inputs
//! Reads `Grid<N,BR,BC>` candidate bitmasks. Outer pivots: unsolved cells with
//! 2–4 candidates. Inner pivots: first bivalue cell found in each hypothesis.
//!
//! ## Mutates
//! Eliminates candidates or places digits in `grid` when consensus is found
//! across all outer hypotheses, or when a hypothesis contradicts. Does not
//! touch state outside the passed-in `Grid`.
//!
//! ## Returns
//! `Some(TechniqueProgress)` on any elimination/placement; `None` otherwise.
//! Sets `progress.contradiction = true` on cascade contradiction.
//! Sets `progress.chain_len = Some(depth * pivots_tried)` as a chain proxy.
//!
//! ## Performance budget
//! < 200 ms/grid on 9×9. Up to 16 outer pivots × 4 cands × 1 inner bivalue
//! × 2 sub-cands = 128 hypothesis evaluations. 1-second wall-time abort guard.
//!
//! ## Algorithm reference
//! SE `isDynamic=true, isMultiple=true, level=1` (Chaining.java). Base 9.0
//! (SE formula: 8.5 + 0.5 * level → 9.0 at level=1).
//! Outer pivot C with k candidates: hypothesize each dᵢ, propagate singles,
//! then branch on the first bivalue cell found (depth=1). Intersect surviving
//! sub-branches per hypothesis, then intersect all outer hypothesis sets.
//! Contradiction in any outer hypothesis → eliminate that candidate. Level=0
//! = Cell FC (already present). Level=2+ (Nested FC) deferred to Phase D.
//! Limitation: only the FIRST bivalue cell branched per hypothesis (v1 perf).
//!
//! ## AlphaEvolve contract
//! Self-contained file. `Technique` / `RatedTechnique` traits are stable
//! contract; do not change their signatures. May freely refactor internals.
//!   * Outer candidate limit: 2–4. Inner branch: first bivalue only (TODO PhaseD).
//!   * Propagation: naked + hidden singles via `propagate_singles`.
//!   * Contradiction sub-case: folded in (eliminated immediately).

use std::time::Instant;

use super::super::grid::{AssignErr, Grid};
use super::chaining_propagator::{propagate_hypothesis, PropagatorConfig};
use super::{RatedTechnique, Technique, TechniqueId, TechniqueProgress, Tier};

/// Maximum outer candidates per pivot cell.
const MAX_OUTER_CANDIDATES: u32 = 4;

/// Maximum outer pivot cells attempted per `apply()` call.
const DEFAULT_MAX_PIVOTS: usize = 16;

/// Wall-time budget (µs). If exceeded, `apply()` aborts early and returns None.
/// This is a safety net for degenerate puzzle states.
const BUDGET_MICROS: u128 = 1_000_000; // 1 second

pub struct DynamicForcingChain {
    pub max_depth: u8,     // default 1 (level=1; 0 = equivalent to Cell FC)
    pub max_pivots: usize, // cap on outer pivots tried per call (default 16)
}

impl Default for DynamicForcingChain {
    fn default() -> Self {
        Self {
            max_depth: 1,
            max_pivots: DEFAULT_MAX_PIVOTS,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal consequence type — identical structure to Cell FC / Region FC.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Action {
    Place,
    Eliminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Consequence {
    cell: usize,
    digit: u8,
    action: Action,
}

// ---------------------------------------------------------------------------
// Core propagation helpers.
// ---------------------------------------------------------------------------

/// Collect the delta (Consequences) between `before` and `after` a mutation.
/// `before` = original base grid; `after` = hypothesis grid after propagation.
fn collect_consequences<const N: usize, const BR: usize, const BC: usize>(
    before: &Grid<N, BR, BC>,
    after: &Grid<N, BR, BC>,
) -> Vec<Consequence> {
    let nn = N * N;
    let mut out = Vec::new();
    for c in 0..nn {
        if after.solved[c] != 0 && before.solved[c] == 0 {
            out.push(Consequence {
                cell: c,
                digit: after.solved[c],
                action: Action::Place,
            });
        }
        if after.solved[c] == 0 && before.solved[c] == 0 {
            let removed = before.candidates[c] & !after.candidates[c];
            let mut bits = removed;
            while bits != 0 {
                let bit = bits & bits.wrapping_neg();
                bits ^= bit;
                let d = bit.trailing_zeros() as u8 + 1;
                out.push(Consequence {
                    cell: c,
                    digit: d,
                    action: Action::Eliminate,
                });
            }
        }
    }
    out
}

/// Intersect `sets` in place: keep only consequences present in ALL sets.
/// Returns the consensus (empty if any set is empty).
fn intersect_all(sets: &[Vec<Consequence>]) -> Vec<Consequence> {
    if sets.is_empty() {
        return Vec::new();
    }
    let mut consensus = sets[0].clone();
    for set in &sets[1..] {
        if consensus.is_empty() {
            break;
        }
        consensus.retain(|c| set.contains(c));
    }
    consensus
}

/// Find the first bivalue (exactly 2 candidates) unsolved cell in `grid`.
/// Returns `(cell_idx, [d_a, d_b])` or `None`.
fn find_first_bivalue<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<(usize, u8, u8)> {
    let nn = N * N;
    for c in 0..nn {
        if grid.solved[c] != 0 {
            continue;
        }
        let mask = grid.candidates[c];
        if mask.count_ones() == 2 {
            let lo = mask.trailing_zeros() as u8 + 1;
            // Second bit
            let second_bit = mask & !(1u32 << (lo - 1));
            let d2 = second_bit.trailing_zeros() as u8 + 1;
            return Some((c, lo, d2));
        }
    }
    None
}

/// Run a *dynamic* hypothesis at `depth` levels of bivalue branching.
///
/// Returns `None` on contradiction; `Some(consequences_vs_outer_base)`.
/// At depth=0 this is identical to `run_static_hypothesis`.
/// At depth=1 it additionally branches on the first bivalue cell found in
/// the post-propagation grid and intersects the sub-branch results.
///
/// `outer_base` is the snapshot of the grid *before* the outer `assign` — used
/// to compute the consequence delta that will be intersected across outer
/// hypotheses.
fn run_dynamic_hypothesis<const N: usize, const BR: usize, const BC: usize>(
    outer_base: &Grid<N, BR, BC>,
    pivot_cell: usize,
    digit: u8,
    depth: u8,
) -> Option<Vec<Consequence>> {
    let mut g = outer_base.clone();
    if g.assign(pivot_cell, digit).is_err() {
        return None;
    }
    // Propagate to fixpoint using the richer SE-equivalent cascade (Phase F):
    // singles → locked candidates → naked/hidden pairs → ALS-XZ.
    if propagate_hypothesis(&mut g, &PropagatorConfig::default()).is_err() {
        return None;
    }

    // Level-0: no dynamic branching. Consequences are the delta vs outer_base.
    if depth == 0 {
        return Some(collect_consequences(outer_base, &g));
    }

    // Level-1+: find first bivalue cell and branch on it.
    if let Some((bv_cell, d_a, d_b)) = find_first_bivalue(&g) {
        // We'll collect surviving sub-branch consequences (relative to `g`).
        let mut sub_sets: Vec<Vec<Consequence>> = Vec::with_capacity(2);

        // A snapshot of `g` before sub-branching (for consequence delta).
        let g_snapshot = g.clone();

        for &sub_digit in &[d_a, d_b] {
            // Run the sub-branch at depth-1. Base for sub = current `g`.
            match run_dynamic_hypothesis(&g_snapshot, bv_cell, sub_digit, depth - 1) {
                None => {
                    // Contradiction in this sub-branch: eliminate sub_digit
                    // from bv_cell in our working `g` persistently.
                    match g.eliminate(bv_cell, sub_digit) {
                        Ok(_) => {
                            // Re-propagate after the elimination.
                            if propagate_hypothesis(&mut g, &PropagatorConfig::default()).is_err() {
                                // Propagation contradiction: outer hypothesis fails.
                                return None;
                            }
                        }
                        Err(AssignErr::Contradiction) => {
                            return None;
                        }
                    }
                }
                Some(inner_consequences) => {
                    // Translate inner consequences (relative to g_snapshot) to
                    // include only what wasn't already implied before the sub-branch.
                    sub_sets.push(inner_consequences);
                }
            }
        }

        // Merge inner sub-branch insights into our consequence set.
        if sub_sets.len() == 1 {
            // Only one sub-branch survived → its consequences are certain.
            // They are already relative to g_snapshot. We'll fold them into
            // the outer consequence delta below.
        } else if sub_sets.len() >= 2 {
            // Multiple sub-branches survived → intersect them.
            let inner_consensus = intersect_all(&sub_sets);
            // Apply inner consensus to `g` so it propagates into the outer delta.
            for c in inner_consensus {
                match c.action {
                    Action::Place => {
                        let _ = g.assign(c.cell, c.digit);
                    }
                    Action::Eliminate => {
                        let _ = g.eliminate(c.cell, c.digit);
                    }
                }
            }
            // Re-propagate after applying inner consensus.
            if propagate_hypothesis(&mut g, &PropagatorConfig::default()).is_err() {
                return None;
            }
        }
        // If sub_sets is empty: both sub-branches contradicted → outer is contradiction.
        if sub_sets.is_empty() && d_a != 0 {
            // Check: if bv_cell still has any candidates left (the eliminations
            // above may have emptied it).
            if g.solved[bv_cell] == 0 && g.candidates[bv_cell] == 0 {
                return None;
            }
        }
    }
    // No bivalue cell found — depth>0 but no inner branching opportunity.
    // Fall through to collect flat consequences.

    Some(collect_consequences(outer_base, &g))
}

// ---------------------------------------------------------------------------
// Trait impls.
// ---------------------------------------------------------------------------

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC>
    for DynamicForcingChain
{
    fn id(&self) -> TechniqueId {
        TechniqueId::DynamicForcingChain
    }

    fn tier(&self) -> Tier {
        Tier::T4Plus
    }

    fn name(&self) -> &'static str {
        "DynamicForcingChain"
    }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let start = Instant::now();
        let mut pivots_tried: usize = 0;

        for pivot in 0..nn {
            // Abort if budget exceeded.
            if start.elapsed().as_micros() > BUDGET_MICROS {
                return None;
            }
            if pivots_tried >= self.max_pivots {
                break;
            }

            if grid.solved[pivot] != 0 {
                continue;
            }
            let cand_mask = grid.candidates[pivot];
            let k = cand_mask.count_ones();
            if k < 2 || k > MAX_OUTER_CANDIDATES {
                continue;
            }

            pivots_tried += 1;

            // Collect candidates.
            let mut digits: [u8; 4] = [0; 4];
            let mut n_digits = 0usize;
            let mut bits = cand_mask;
            while bits != 0 {
                let bit = bits & bits.wrapping_neg();
                bits ^= bit;
                digits[n_digits] = bit.trailing_zeros() as u8 + 1;
                n_digits += 1;
            }
            let digits = &digits[..n_digits];

            // Run one dynamic hypothesis per candidate.
            let mut hyp_results: Vec<Option<Vec<Consequence>>> = Vec::with_capacity(n_digits);
            let mut any_contradiction = false;

            for &d in digits {
                let result = run_dynamic_hypothesis(grid, pivot, d, self.max_depth);
                if result.is_none() {
                    any_contradiction = true;
                }
                hyp_results.push(result);
            }

            // --- Contradiction sub-case ---
            if any_contradiction {
                let mut prog = TechniqueProgress::default();
                prog.chain_len = Some((self.max_depth as u32 + 1) * pivots_tried as u32);
                for (i, res) in hyp_results.iter().enumerate() {
                    if res.is_none() {
                        let d = digits[i];
                        match grid.eliminate(pivot, d) {
                            Ok(true) => {
                                prog.eliminations.push((pivot, d));
                            }
                            Ok(false) => {}
                            Err(AssignErr::Contradiction) => {
                                prog.contradiction = true;
                                return Some(prog);
                            }
                        }
                    }
                }
                if prog.fired() {
                    return Some(prog);
                }
                continue;
            }

            // --- Consensus intersection ---
            let valid_sets: Vec<Vec<Consequence>> = hyp_results
                .into_iter()
                .map(|r| r.unwrap())
                .collect();
            let consensus = intersect_all(&valid_sets);

            if consensus.is_empty() {
                continue;
            }

            // Filter: skip already-satisfied consequences.
            let valid_consensus: Vec<Consequence> = consensus
                .into_iter()
                .filter(|c| match c.action {
                    Action::Place => grid.solved[c.cell] == 0,
                    Action::Eliminate => {
                        grid.solved[c.cell] == 0
                            && (grid.candidates[c.cell] & (1u32 << (c.digit - 1))) != 0
                    }
                })
                .collect();

            if valid_consensus.is_empty() {
                continue;
            }

            let mut prog = TechniqueProgress::default();
            prog.chain_len = Some((self.max_depth as u32 + 1) * pivots_tried as u32);

            for c in valid_consensus {
                match c.action {
                    Action::Place => match grid.assign(c.cell, c.digit) {
                        Ok(()) => {
                            prog.placements.push((c.cell, c.digit));
                        }
                        Err(AssignErr::Contradiction) => {
                            prog.contradiction = true;
                            return Some(prog);
                        }
                    },
                    Action::Eliminate => match grid.eliminate(c.cell, c.digit) {
                        Ok(true) => {
                            prog.eliminations.push((c.cell, c.digit));
                        }
                        Ok(false) => {}
                        Err(AssignErr::Contradiction) => {
                            prog.contradiction = true;
                            return Some(prog);
                        }
                    },
                }
            }

            if prog.fired() {
                return Some(prog);
            }
        }

        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for DynamicForcingChain
{
    fn base_rating(&self) -> f64 {
        // SE: ForcingChainHint.getDifficulty() = 8.5 + 0.5*level. level=1 → 9.0.
        // (Phase D will add level=2 nested FC at base 9.5.)
        9.0
    }

    fn se_rating(&self, progress: &TechniqueProgress) -> f64 {
        // Use same step schedule as AIC (SE getLengthDifficulty()).
        // chain_len is a proxy (depth * pivots); use base when not set.
        let chain_len = match progress.chain_len {
            None => return 9.0,
            Some(cl) => cl as i32,
        };
        let se_length = (chain_len - 1).max(0);

        let mut added = 0.0_f64;
        let mut ceil = 4_i32;
        let mut is_odd = false;
        while se_length > ceil {
            added += 0.1;
            if !is_odd {
                ceil = (ceil * 3) / 2;
            } else {
                ceil = (ceil * 4) / 3;
            }
            is_odd = !is_odd;
        }

        // Base 9.0 (level=1); no upper cap — SE DFC keeps growing with chain length.
        9.0 + added
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::backtracker::propagate_singles;
    use crate::generic::grid::Grid;

    fn grid_9(s: &str) -> Grid<9, 3, 3> {
        Grid::from_str(s).expect("invalid puzzle string")
    }

    /// DynamicFC returns None on an easy (T1) puzzle solvable by singles.
    #[test]
    fn dynamic_fc_does_not_fire_on_easy() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let mut g = grid_9(p);
        propagate_singles(&mut g).expect("should not contradict");
        assert!(g.is_solved(), "easy puzzle must solve by singles");
        let dfc = DynamicForcingChain::default();
        let result = dfc.apply(&mut g);
        assert!(
            result.is_none(),
            "DynamicFC must return None on a fully solved grid"
        );
    }

    /// Contradiction sub-case: a depth-1 hypothesis that leads to contradiction
    /// causes the corresponding candidate to be eliminated from the outer pivot.
    ///
    /// Construction: start from a fully solved 9×9; un-solve cell 0, give it
    /// candidates {1,2,3}. Digit 2 and digit 3 are already placed at row-peer
    /// cells 1 and 2 respectively, so assigning them at cell 0 immediately
    /// contradicts. DFC should eliminate both and return Some(progress).
    #[test]
    fn dynamic_fc_contradiction_subcase() {
        let sol = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let mut g = grid_9(sol);
        assert!(g.is_solved());

        // Un-solve cell 0; give it fake 3-candidate mask {1,2,3}.
        g.solved[0] = 0;
        g.solved_count -= 1;
        g.candidates[0] = 0b111; // bits 0,1,2 = digits 1,2,3

        assert_eq!(g.solved[1], 2, "cell 1 must be solved with digit 2");
        assert_eq!(g.solved[2], 3, "cell 2 must be solved with digit 3");

        let dfc = DynamicForcingChain::default();
        let result = dfc.apply(&mut g);

        assert!(
            result.is_some(),
            "DynamicFC should fire the contradiction sub-case"
        );
        let prog = result.unwrap();
        assert!(prog.fired(), "progress should report at least one change");
        assert!(
            prog.eliminations.contains(&(0, 2)),
            "expected elimination of digit 2 from cell 0; got {:?}",
            prog.eliminations
        );
        assert!(
            prog.eliminations.contains(&(0, 3)),
            "expected elimination of digit 3 from cell 0; got {:?}",
            prog.eliminations
        );
        assert!(!prog.contradiction, "no grid-level contradiction expected");
    }

    /// `base_rating()` must return 9.0 (SE level=1 dynamic FC).
    #[test]
    fn dynamic_fc_rated_technique_base_rating() {
        let dfc = DynamicForcingChain::default();
        let rating = <DynamicForcingChain as RatedTechnique<9, 3, 3>>::base_rating(&dfc);
        assert!(
            (rating - 9.0).abs() < 1e-9,
            "base_rating should be 9.0, got {}",
            rating
        );
    }

    /// After firing, `progress.chain_len` should be `Some(...)` not `None`.
    #[test]
    fn dynamic_fc_chain_len_set() {
        let sol = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let mut g = grid_9(sol);
        g.solved[0] = 0;
        g.solved_count -= 1;
        g.candidates[0] = 0b111;

        let dfc = DynamicForcingChain::default();
        let prog = dfc.apply(&mut g).expect("should fire");
        assert!(
            prog.chain_len.is_some(),
            "chain_len should be Some after firing"
        );
    }

    /// DynamicFC returns None on a fully solved grid (no pivot to try).
    #[test]
    fn dynamic_fc_no_op_on_solved_grid() {
        let sol = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let mut g = grid_9(sol);
        assert!(g.is_solved());
        let dfc = DynamicForcingChain::default();
        assert!(dfc.apply(&mut g).is_none());
    }

    /// Tier and id are correct.
    #[test]
    fn dynamic_fc_tier_and_id() {
        let dfc = DynamicForcingChain::default();
        assert_eq!(
            <DynamicForcingChain as Technique<9, 3, 3>>::tier(&dfc),
            Tier::T4Plus
        );
        assert_eq!(
            <DynamicForcingChain as Technique<9, 3, 3>>::id(&dfc),
            TechniqueId::DynamicForcingChain
        );
    }
}
