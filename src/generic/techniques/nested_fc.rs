//! # Nested Forcing Chain (NestedFC)
//!
//! ## Inputs
//! Reads `Grid<N,BR,BC>` candidate bitmasks. Outer pivots: unsolved cells with
//! 2–4 candidates (up to `max_pivots=12`). Inner depth-N recursion: branches
//! on up to `max_bivalue_branches=2` bivalue cells per hypothesis level.
//!
//! ## Mutates
//! Eliminates candidates or places digits in `grid` when consensus is found
//! across all outer hypotheses, or when a hypothesis leads to contradiction.
//! Does not touch state outside the passed-in `Grid`.
//!
//! ## Returns
//! `Some(TechniqueProgress)` on any elimination/placement; `None` otherwise.
//! Sets `progress.contradiction = true` on cascade contradiction.
//! Sets `progress.chain_len = Some(depth * pivots_tried)` as a chain proxy.
//!
//! ## Performance budget
//! Level=2: < 1000 ms/grid. 12 × 4 × 2 × 2 = 192 hypothesis evaluations max.
//! 5-second wall-time abort guard.
//! Level=3: < 10 000 ms/grid. 12 × 4 × 2 × 2 × 2 = 384 evaluations max.
//! 10-second wall-time abort guard (configurable via `budget_micros` field).
//!
//! ## Algorithm reference
//! SE `isDynamic=true, isMultiple=true, level=N` (Chaining.java).
//! Base rating = 8.5 + 0.5 * level.
//!   level=2 → 9.5, level=3 → 10.0.
//! Outer pivot C with k candidates: hypothesize each dᵢ, propagate singles to
//! fixpoint, then recursively branch on up to `max_bivalue_branches` bivalue
//! cells found in the post-propagation grid. At each depth level,
//! contradictions in sub-branches force eliminations; surviving sub-branches are
//! intersected for consensus consequences that are applied to the parent grid.
//! The outer hypothesis consequences are then intersected across all outer dᵢ.
//! Approximation vs SE: SE level=2+ allows locked-candidates/pair propagation
//! *inside* hypotheses; our propagator is singles only, so we compensate by
//! branching on 2 bivalue cells instead of 1 (depth=2 with 2 branches/level).
//!
//! ## AlphaEvolve contract
//! Self-contained file. `Technique` / `RatedTechnique` traits are stable
//! contract; do not change their signatures. May freely refactor internals.
//!   * Outer candidate limit: 2–4. Inner branch: up to 2 bivalue cells per depth.
//!   * Propagation: naked + hidden singles via `propagate_singles`.
//!   * Contradiction sub-case: folded in (eliminated immediately).
//!   * Level=4+ deferred (TODO).
//!   * ALS propagation inside hypotheses deferred.

use std::time::Instant;

use super::super::grid::{AssignErr, Grid};
use super::chaining_propagator::{propagate_hypothesis, PropagatorConfig};
use super::{RatedTechnique, Technique, TechniqueId, TechniqueProgress, Tier};

/// Maximum outer candidates per pivot cell.
const MAX_OUTER_CANDIDATES: u32 = 4;

/// Maximum outer pivot cells attempted per `apply()` call.
const DEFAULT_MAX_PIVOTS: usize = 12;

/// Maximum bivalue branches to explore at each depth level.
const DEFAULT_MAX_BIVALUE_BRANCHES: u8 = 2;

/// Wall-time budget (µs) for level=2. Abort early if exceeded.
const BUDGET_MICROS_L2: u128 = 5_000_000; // 5 seconds

/// Wall-time budget (µs) for level=3. Abort early if exceeded.
const BUDGET_MICROS_L3: u128 = 10_000_000; // 10 seconds

/// Wall-time budget (µs) for level=4. Abort early if exceeded.
const BUDGET_MICROS_L4: u128 = 20_000_000; // 20 seconds

pub struct NestedForcingChain {
    pub max_pivots: usize,
    pub max_bivalue_branches: u8,
    /// Recursion depth for inner bivalue branching. Also determines `id()`,
    /// `name()`, `base_rating()`, and wall-time budget.
    /// depth=2 → NestedForcingChainL2 (SE 9.5), depth=3 → NestedForcingChainL3 (SE 10.0).
    pub max_depth: u8,
}

impl Default for NestedForcingChain {
    /// Default is level=2 (SE 9.5) for backward compatibility.
    fn default() -> Self {
        Self::level_2()
    }
}

impl NestedForcingChain {
    /// Level=2 (SE base 9.5): depth=2, 2 bivalue branches per level.
    pub fn level_2() -> Self {
        Self {
            max_pivots: DEFAULT_MAX_PIVOTS,
            max_bivalue_branches: DEFAULT_MAX_BIVALUE_BRANCHES,
            max_depth: 2,
        }
    }

    /// Level=3 (SE base 10.0): depth=3, 2 bivalue branches per level.
    pub fn level_3() -> Self {
        Self {
            max_pivots: DEFAULT_MAX_PIVOTS,
            max_bivalue_branches: DEFAULT_MAX_BIVALUE_BRANCHES,
            max_depth: 3,
        }
    }

    /// Level=4 (SE base 10.5): depth=4, 2 bivalue branches per level.
    pub fn level_4() -> Self {
        Self {
            max_pivots: DEFAULT_MAX_PIVOTS,
            max_bivalue_branches: DEFAULT_MAX_BIVALUE_BRANCHES,
            max_depth: 4,
        }
    }

    /// Wall-time budget in µs based on configured depth.
    fn budget_micros(&self) -> u128 {
        match self.max_depth {
            4.. => BUDGET_MICROS_L4,
            3 => BUDGET_MICROS_L3,
            _ => BUDGET_MICROS_L2,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal consequence type — same as DynamicFC.
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

/// Collect the delta between `before` and `after` a mutation.
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

/// Intersect `sets`: keep only consequences present in ALL sets.
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

/// Find up to `max_count` bivalue (exactly 2 candidates) unsolved cells.
fn find_bivalue_cells<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    max_count: u8,
) -> Vec<(usize, u8, u8)> {
    let nn = N * N;
    let mut out = Vec::new();
    for c in 0..nn {
        if out.len() >= max_count as usize {
            break;
        }
        if grid.solved[c] != 0 {
            continue;
        }
        let mask = grid.candidates[c];
        if mask.count_ones() == 2 {
            let lo = mask.trailing_zeros() as u8 + 1;
            let second_bit = mask & !(1u32 << (lo - 1));
            let d2 = second_bit.trailing_zeros() as u8 + 1;
            out.push((c, lo, d2));
        }
    }
    out
}

/// Run a nested hypothesis at `depth` levels of bivalue branching, exploring
/// up to `max_bivalue_branches` bivalue cells at each depth level.
///
/// Returns `None` on contradiction; `Some(consequences_vs_outer_base)`.
/// At depth=0: flat singles propagation only (identical to static hypothesis).
/// At depth=1: branches on first bivalue cell (same as DynamicFC at depth=1).
/// At depth=2 (NestedFC): branches on up to 2 bivalue cells per level.
///
/// `outer_base` is the snapshot before the outer `assign` — the consequence
/// delta is computed relative to it so it can be intersected across outer
/// hypotheses.
fn run_nested_hypothesis<const N: usize, const BR: usize, const BC: usize>(
    outer_base: &Grid<N, BR, BC>,
    pivot_cell: usize,
    digit: u8,
    depth: u8,
    max_bivalue_branches: u8,
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

    // Depth=0: no inner branching.
    if depth == 0 {
        return Some(collect_consequences(outer_base, &g));
    }

    // Find up to `max_bivalue_branches` bivalue cells in current state.
    let bv_cells = find_bivalue_cells(&g, max_bivalue_branches);

    if bv_cells.is_empty() {
        // No bivalue cells to branch on — fall through to flat consequences.
        return Some(collect_consequences(outer_base, &g));
    }

    // For each bivalue cell, branch on its two candidates and apply
    // contradiction eliminations or intersect surviving sub-branches.
    //
    // `bv_cells` is captured BEFORE this loop, so a previous sibling's
    // consensus may have already solved `bv_cell` to one of its bivalue
    // digits. `Grid::assign` doesn't guard against `solved[cell] != 0`,
    // so re-assigning would double-increment `solved_count` and corrupt
    // `is_solved()`. Skip stale bv_cells.
    for (bv_cell, d_a, d_b) in &bv_cells {
        if g.solved[*bv_cell] != 0 {
            continue;
        }
        let g_snapshot = g.clone();

        let mut sub_sets: Vec<Vec<Consequence>> = Vec::with_capacity(2);

        for &sub_digit in &[*d_a, *d_b] {
            match run_nested_hypothesis(
                &g_snapshot,
                *bv_cell,
                sub_digit,
                depth - 1,
                max_bivalue_branches,
            ) {
                None => {
                    // Sub-branch contradicts: eliminate sub_digit from bv_cell.
                    match g.eliminate(*bv_cell, sub_digit) {
                        Ok(_) => {
                            if propagate_hypothesis(&mut g, &PropagatorConfig::default()).is_err() {
                                return None;
                            }
                        }
                        Err(AssignErr::Contradiction) => {
                            return None;
                        }
                    }
                }
                Some(inner_cons) => {
                    sub_sets.push(inner_cons);
                }
            }
        }

        if sub_sets.is_empty() {
            // Both sub-branches of this bv_cell contradicted.
            // The eliminations above should have propagated a contradiction
            // or forced the cell — check grid state.
            if g.solved[*bv_cell] == 0 && g.candidates[*bv_cell] == 0 {
                return None;
            }
        } else if sub_sets.len() == 1 {
            // Only one sub-branch survived; its consequences are certain.
            // They are relative to g_snapshot; apply them to g.
            for c in &sub_sets[0] {
                match c.action {
                    Action::Place => {
                        let _ = g.assign(c.cell, c.digit);
                    }
                    Action::Eliminate => {
                        let _ = g.eliminate(c.cell, c.digit);
                    }
                }
            }
            if propagate_hypothesis(&mut g, &PropagatorConfig::default()).is_err() {
                return None;
            }
        } else {
            // Multiple sub-branches survived: intersect and apply consensus.
            let inner_consensus = intersect_all(&sub_sets);
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
            if propagate_hypothesis(&mut g, &PropagatorConfig::default()).is_err() {
                return None;
            }
        }
    }

    Some(collect_consequences(outer_base, &g))
}

// ---------------------------------------------------------------------------
// Trait impls.
// ---------------------------------------------------------------------------

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC>
    for NestedForcingChain
{
    fn id(&self) -> TechniqueId {
        match self.max_depth {
            4 => TechniqueId::NestedForcingChainL4,
            3 => TechniqueId::NestedForcingChainL3,
            _ => TechniqueId::NestedForcingChain, // depth=2 (and safe default)
        }
    }

    fn tier(&self) -> Tier {
        Tier::T4Plus
    }

    fn name(&self) -> &'static str {
        match self.max_depth {
            4 => "NestedForcingChain[4]",
            3 => "NestedForcingChain[3]",
            _ => "NestedForcingChain[2]",
        }
    }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let start = Instant::now();
        let budget = self.budget_micros();
        let mut pivots_tried: usize = 0;

        for pivot in 0..nn {
            if start.elapsed().as_micros() > budget {
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

            // Run one nested hypothesis per candidate (depth=2).
            let mut hyp_results: Vec<Option<Vec<Consequence>>> = Vec::with_capacity(n_digits);
            let mut any_contradiction = false;

            for &d in digits {
                let result = run_nested_hypothesis(
                    grid,
                    pivot,
                    d,
                    self.max_depth,
                    self.max_bivalue_branches,
                );
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
    for NestedForcingChain
{
    fn base_rating(&self) -> f64 {
        // SE: ForcingChainHint.getDifficulty() = 8.5 + 0.5*level.
        // level=2 → 9.5, level=3 → 10.0.
        8.5 + 0.5 * (self.max_depth as f64)
    }

    fn se_rating(&self, progress: &TechniqueProgress) -> f64 {
        // SE: 8.5 + 0.5 * level (level = max_depth).
        let base = 8.5 + 0.5 * (self.max_depth as f64);
        // Same step schedule as AIC / DFC (SE getLengthDifficulty()).
        let chain_len = match progress.chain_len {
            None => return base,
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

        base + added
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

    /// NestedFC returns None on an easy (T1) puzzle solvable by singles.
    #[test]
    fn nested_fc_does_not_fire_on_easy() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let mut g = grid_9(p);
        propagate_singles(&mut g).expect("should not contradict");
        assert!(g.is_solved(), "easy puzzle must solve by singles");
        let nfc = NestedForcingChain::default();
        let result = nfc.apply(&mut g);
        assert!(
            result.is_none(),
            "NestedFC must return None on a fully solved grid"
        );
    }

    /// Contradiction sub-case: a depth-2 hypothesis that contradicts causes
    /// the candidate to be eliminated from the outer pivot.
    ///
    /// Construction: start from a fully solved 9×9; un-solve cell 0, give it
    /// candidates {1,2,3}. Digits 2 and 3 are placed at row-peer cells 1 and 2,
    /// so assigning them at cell 0 immediately contradicts. NestedFC should
    /// eliminate both and return Some(progress).
    #[test]
    fn nested_fc_contradiction_subcase() {
        let sol = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let mut g = grid_9(sol);
        assert!(g.is_solved());

        // Un-solve cell 0; give it fake 3-candidate mask {1,2,3}.
        g.solved[0] = 0;
        g.solved_count -= 1;
        g.candidates[0] = 0b111; // bits 0,1,2 = digits 1,2,3

        assert_eq!(g.solved[1], 2, "cell 1 must be solved with digit 2");
        assert_eq!(g.solved[2], 3, "cell 2 must be solved with digit 3");

        let nfc = NestedForcingChain::default();
        let result = nfc.apply(&mut g);

        assert!(
            result.is_some(),
            "NestedFC should fire the contradiction sub-case"
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

    /// `base_rating()` must return 9.5 (SE level=2 nested FC).
    #[test]
    fn nested_fc_rated_technique_base_rating() {
        let nfc = NestedForcingChain::default();
        let rating = <NestedForcingChain as RatedTechnique<9, 3, 3>>::base_rating(&nfc);
        assert!(
            (rating - 9.5).abs() < 1e-9,
            "base_rating should be 9.5, got {}",
            rating
        );
    }

    /// After firing, `progress.chain_len` should be `Some(..)` not `None`.
    #[test]
    fn nested_fc_chain_len_set() {
        let sol = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let mut g = grid_9(sol);
        g.solved[0] = 0;
        g.solved_count -= 1;
        g.candidates[0] = 0b111;

        let nfc = NestedForcingChain::default();
        let prog = nfc.apply(&mut g).expect("should fire");
        assert!(
            prog.chain_len.is_some(),
            "chain_len should be Some after firing"
        );
    }

    /// Tier and id are correct for level=2 (default).
    #[test]
    fn nested_fc_tier_and_id() {
        let nfc = NestedForcingChain::default();
        assert_eq!(
            <NestedForcingChain as Technique<9, 3, 3>>::tier(&nfc),
            Tier::T4Plus
        );
        assert_eq!(
            <NestedForcingChain as Technique<9, 3, 3>>::id(&nfc),
            TechniqueId::NestedForcingChain
        );
    }

    // -----------------------------------------------------------------------
    // Level=3 tests
    // -----------------------------------------------------------------------

    /// NestedFC L3 returns None on a fully solved (easy) grid.
    #[test]
    fn nested_fc_l3_does_not_fire_on_easy() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let mut g = grid_9(p);
        propagate_singles(&mut g).expect("should not contradict");
        assert!(g.is_solved(), "easy puzzle must solve by singles");
        let nfc3 = NestedForcingChain::level_3();
        let result = nfc3.apply(&mut g);
        assert!(
            result.is_none(),
            "NestedFC L3 must return None on a fully solved grid"
        );
    }

    /// Contradiction sub-case at depth=3: same construction as the L2 test —
    /// digits 2 and 3 at peer cells immediately contradict; L3 should also
    /// detect and eliminate them.
    #[test]
    fn nested_fc_l3_contradiction_subcase() {
        let sol = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let mut g = grid_9(sol);
        assert!(g.is_solved());

        g.solved[0] = 0;
        g.solved_count -= 1;
        g.candidates[0] = 0b111; // digits 1, 2, 3

        let nfc3 = NestedForcingChain::level_3();
        let result = nfc3.apply(&mut g);

        assert!(
            result.is_some(),
            "NestedFC L3 should fire the contradiction sub-case"
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

    /// `base_rating()` must return 10.0 for level=3 instance.
    #[test]
    fn nested_fc_l3_base_rating() {
        let nfc3 = NestedForcingChain::level_3();
        let rating = <NestedForcingChain as RatedTechnique<9, 3, 3>>::base_rating(&nfc3);
        assert!(
            (rating - 10.0).abs() < 1e-9,
            "L3 base_rating should be 10.0, got {}",
            rating
        );
    }

    /// `id()` returns `NestedForcingChainL3` for the level=3 instance.
    #[test]
    fn nested_fc_l3_id() {
        let nfc3 = NestedForcingChain::level_3();
        assert_eq!(
            <NestedForcingChain as Technique<9, 3, 3>>::id(&nfc3),
            TechniqueId::NestedForcingChainL3,
        );
    }

    // -----------------------------------------------------------------------
    // Level=4 tests
    // -----------------------------------------------------------------------

    /// `id()` returns `NestedForcingChainL4` and `base_rating()` returns 10.5.
    #[test]
    fn level_4_id_and_rating() {
        let nfc4 = NestedForcingChain::level_4();
        assert_eq!(
            <NestedForcingChain as Technique<9, 3, 3>>::id(&nfc4),
            TechniqueId::NestedForcingChainL4,
        );
        let rating = <NestedForcingChain as RatedTechnique<9, 3, 3>>::base_rating(&nfc4);
        assert!(
            (rating - 10.5).abs() < 1e-9,
            "L4 base_rating should be 10.5, got {}",
            rating
        );
    }

    /// NestedFC L4 returns None on a fully solved grid (no pivot needed).
    #[test]
    fn nested_fc_l4_does_not_fire_on_solved() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let mut g = grid_9(p);
        propagate_singles(&mut g).expect("should not contradict");
        assert!(g.is_solved(), "easy puzzle must solve by singles");
        let nfc4 = NestedForcingChain::level_4();
        let result = nfc4.apply(&mut g);
        assert!(
            result.is_none(),
            "NestedFC L4 must return None on a fully solved grid"
        );
    }
}
