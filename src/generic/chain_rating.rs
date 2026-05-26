//! Chain rating for the GBraid technique (spec §17 — GBraid-only collapse,
//! 2026-05-20). Per Berthier's containment theorem (Whip ⊆ Braid ⊆ GBraid,
//! GWhip ⊆ GBraid), GBraid alone preserves T&E(1) solving power.
//!
//! Pre-collapse this driver salience-interleaved whip[k] → gwhip[k] → braid[k]
//! → gbraid[k] at each k. Post-collapse only gbraid[k] is probed.
//!
//! ## Non-mutating contract
//!
//! `find_first_gbraid` does NOT call `rs.eliminate_candidate`. It reads
//! `ctx.rs.cand_alive` but never modifies it, so `rate_chain` can call it
//! without snapshotting / restoring the RS.
//!
//! ## Relationship to `rater::rate_excluding`
//!
//! `rate_chain` classifies the *post-singles* state with a single chain
//! technique. `rater::rate_excluding` runs the full T2/T3 cascade before the
//! T4Plus chain pass; that cascade is strictly more permissive at the chain
//! layer than `rate_chain`. The asymmetry is conservative for load-bearing:
//! if the richer cascade still cannot solve `p` without GBraid, the CLIPS
//! cascade likewise cannot.

use std::sync::OnceLock;

use super::backtracker::propagate_singles;
use super::csp_tables::{build_csp_tables_n9, CspLinkGraph, CspVarTable, LinkGraph, W9};
use super::glabel_tables::{build_glabel_tables_n9, GLabelTable, GLinkGraph, WG9};
use super::grid::{AssignErr, Grid};
use super::resolution_state::ResolutionState;
use super::techniques::whip::ChainContext;
use super::techniques::gbraid::find_first_gbraid;

// ─── Tables cache ─────────────────────────────────────────────────────────────

/// Cached immutable tables for N=9. Built once on first call to `rate_chain`.
static TABLES_N9: OnceLock<(CspVarTable, LinkGraph<W9>, CspLinkGraph, GLabelTable<W9>, GLinkGraph<W9, WG9>)> =
    OnceLock::new();

fn get_tables_n9() -> &'static (CspVarTable, LinkGraph<W9>, CspLinkGraph, GLabelTable<W9>, GLinkGraph<W9, WG9>) {
    TABLES_N9.get_or_init(|| {
        let (csp, link, cspl) = build_csp_tables_n9();
        let (glab, glnk) = build_glabel_tables_n9(&csp);
        (csp, link, cspl, glab, glnk)
    })
}

// ─── ChainRating ──────────────────────────────────────────────────────────────

/// The chain rating axis. Post-collapse only the `GB` variant is constructed
/// by `rate_chain` / the rater pipeline. The other three variants (`W`, `GW`,
/// `B`) are retained for backwards-compatibility with the on-disk JSONL
/// format and external consumers; they are never emitted by current code.
///
/// The inner `u8` is the chain length k at which the technique first fired.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ChainRating {
    /// Whip[k]. **Deprecated post-collapse (spec §17).** Retained for
    /// backwards-compatibility; never constructed by current code.
    W(u8),
    /// g-Whip[k]. **Deprecated post-collapse (spec §17).**
    GW(u8),
    /// Braid[k]. **Deprecated post-collapse (spec §17).**
    B(u8),
    /// g-Braid[k] — the sole chain rating emitted by `rate_chain` post-collapse.
    GB(u8),
}

// ─── rate_chain ───────────────────────────────────────────────────────────────

/// Rate a 9×9 puzzle on the GBraid chain axis.
///
/// Returns the SMALLEST k at which `gbraid[k]` fires, packaged as
/// `Some(ChainRating::GB(k))`, or `None` in three semantically distinct cases:
///
/// 1. **Inconsistent input** — the givens already violate a CSP constraint or
///    `propagate_singles` detected a contradiction.
/// 2. **Solved by singles alone** — `propagate_singles` reaches a complete
///    grid before any chain technique runs.
/// 3. **No gbraid fires within `k_max`** — propagation leaves a non-trivial
///    grid and `gbraid[3..=k_max]` produces no elimination.
///
/// ## BRT/singles pre-pass (spec §4.1)
///
/// Propagates singles to fixpoint before building `ChainContext`. Non-mutating
/// on the original `g` — propagation runs on a clone.
pub fn rate_chain(g: &Grid<9, 3, 3>, k_max: u8) -> Option<ChainRating> {
    if k_max == 0 {
        return None;
    }

    // Fast reject: inconsistent givens.
    if !g.is_consistent() {
        return None;
    }

    // BRT/singles pre-pass.
    let mut g_prepped = g.clone();
    match propagate_singles(&mut g_prepped) {
        Err(AssignErr::Contradiction) => return None,
        Ok(()) => {}
    }
    if g_prepped.is_solved() {
        return None; // singles alone suffice
    }
    let g = &g_prepped;

    let (csp, link, cspl, glab, glnk) = get_tables_n9();
    let mut rs = ResolutionState::from_grid_9x9(g, glab);
    let mut ctx = ChainContext { csp, link, cspl, glab, glnk, rs: &mut rs };
    if let Some((_elim, k_fired)) = find_first_gbraid(&mut ctx, k_max) {
        return Some(ChainRating::GB(k_fired));
    }

    None
}

// ─── chain_rating_to_se ──────────────────────────────────────────────────────

/// SE (Sudoku Explainer) rating mapping. Per spec §17 (GBraid-only collapse)
/// only the `GB(k)` arm is meaningful for emitted ratings; the other three
/// arms retain their pre-collapse formulas for backwards-compatibility with
/// any historical `ChainRating` values still parsed from on-disk data.
pub fn chain_rating_to_se(r: ChainRating) -> f32 {
    match r {
        // Pre-collapse arms retained for backwards-compatibility only.
        ChainRating::W(k)  => 6.6 + 0.2 * (k as f32 - 1.0),
        ChainRating::GW(k) => 6.8 + 0.2 * (k as f32 - 1.0),
        ChainRating::B(k)  => 8.0 + 0.3 * (k as f32 - 1.0),
        // The only arm constructed post-collapse.
        ChainRating::GB(k) => 8.5 + 0.3 * (k as f32 - 1.0),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Solved-by-singles puzzle returns `None`.
    #[test]
    fn test_fixture1_solved_by_singles() {
        let puzzle = "...456..9..6.......891..45.2.........7..9.....35......397...5.......4.72.....5361";
        let grid = Grid::<9, 3, 3>::from_str(puzzle).expect("parse");
        let rating = rate_chain(&grid, 5);
        assert_eq!(rating, None, "Fixture 1 (B=0): singles must suffice");
    }

    /// `None` on a fully solved grid.
    #[test]
    fn test_none_on_solved_grid() {
        let solved = "974856213328149657516723948153278469769435821842691375431967582685312794297584136";
        let grid = Grid::<9, 3, 3>::from_str(solved).expect("parse");
        assert_eq!(rate_chain(&grid, 10), None);
    }

    /// `k_max = 0` returns `None` cleanly.
    #[test]
    fn test_k_max_zero_returns_none() {
        let puzzle = "...456..9..6.......891..45.2.........7..9.....35......397...5.......4.72.....5361";
        let grid = Grid::<9, 3, 3>::from_str(puzzle).expect("parse");
        assert_eq!(rate_chain(&grid, 0), None);
    }

    /// `chain_rating_to_se` monotonicity in k for `GB`.
    #[test]
    fn test_chain_rating_to_se_monotonicity() {
        let gb3 = chain_rating_to_se(ChainRating::GB(3));
        let gb4 = chain_rating_to_se(ChainRating::GB(4));
        let gb10 = chain_rating_to_se(ChainRating::GB(10));
        assert!(gb3 < gb4);
        assert!(gb4 < gb10);
    }
}
