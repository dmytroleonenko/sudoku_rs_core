//! # Rater-pipeline wrapper for the GBraid chain technique.
//!
//! Per Berthier's containment theorem (Whip ⊆ Braid ⊆ GBraid, GWhip ⊆ GBraid),
//! GBraid alone preserves T&E(1) solving power. The previous four-arm
//! salience-interleaved driver (Whip → GWhip → Braid → GBraid at each k) has
//! been collapsed to a single GBraid pass per k (spec §17 — GBraid-only
//! collapse, 2026-05-20).
//!
//! ## Inputs
//! Reads from `Grid<N,BR,BC>` (handed to the chain rater via an `unsafe`
//! transmute guarded by `N=9 / BR=3 / BC=3` const checks), builds a
//! per-call `ResolutionState` + `ChainContext`, and accepts a configurable
//! `k_max` cap (default `CHAIN_RATER_K_MAX = 36`).
//!
//! ## Mutates
//! Mutates `Grid` in-place via `eliminate_candidate` to propagate the chain
//! elimination through peer cells. No global state. Allocates per-call
//! scratch for the chain context.
//!
//! ## Returns
//! `Option<TechniqueProgress>`. On a real firing: `eliminations` populated,
//! `technique_id_override` set to `GBraid`, `chain_len` set to the firing k.
//! `None` if no gbraid[k] fires within `k_max`.
//!
//! ## Performance budget
//! O(k_max) per `apply` call thanks to the persistent `GBraidSearchState`
//! struct (CR-FIN-15 perf pass).
//!
//! ## 9×9 only
//!
//! Chain techniques require static CSP / glabel tables built for N=9 only.
//! For other `(N, BR, BC)` `apply` returns `None` (no-op, harmless).
//!
//! ## SE rating
//!
//! Based on `chain_rating_to_se` in `chain_rating.rs`:
//!   GB[k] ≈ 8.5 + 0.3*(k-1).
//! `chain_len` in `TechniqueProgress` carries the firing k.

use std::collections::HashSet;
use std::sync::OnceLock;

use super::super::chain_model::{Chain, ChainElimination, Label};
use super::super::chain_rating::{chain_rating_to_se, ChainRating};
use super::super::csp_tables::{build_csp_tables_n9, CspLinkGraph, CspVarTable, LinkGraph, W9};
use super::super::glabel_tables::{build_glabel_tables_n9, GLabelTable, GLinkGraph, WG9};
use super::super::grid::Grid;
use super::super::resolution_state::ResolutionState;
use super::whip::{
    build_partial_whips_length_1, extend_partial_whips, ChainContext,
};
use super::gwhip::{
    build_partial_gwhips_length_1, extend_partial_gwhips,
};
use super::gbraid::{
    extend_partial_gbraids, try_terminate_gbraid,
};
use super::{RatedTechnique, Technique, TechniqueId, TechniqueProgress, Tier};

/// Default k_max for the chain technique in the rater cascade.
/// Spec §4.1 / §13 cap at k=36; we honour that as the default.
pub const CHAIN_RATER_K_MAX: u8 = 36;

// ---------------------------------------------------------------------------
// Static tables cache (9×9)
// ---------------------------------------------------------------------------

static TABLES_N9: OnceLock<(
    CspVarTable,
    LinkGraph<W9>,
    CspLinkGraph,
    GLabelTable<W9>,
    GLinkGraph<W9, WG9>,
)> = OnceLock::new();

fn get_tables() -> &'static (
    CspVarTable,
    LinkGraph<W9>,
    CspLinkGraph,
    GLabelTable<W9>,
    GLinkGraph<W9, WG9>,
) {
    TABLES_N9.get_or_init(|| {
        let (csp, link, cspl) = build_csp_tables_n9();
        let (glab, glnk) = build_glabel_tables_n9(&csp);
        (csp, link, cspl, glab, glnk)
    })
}

// ---------------------------------------------------------------------------
// Helper: decode a Label and apply the elimination.
// Returns a TechniqueProgress with the firing k in chain_len.
// ---------------------------------------------------------------------------

fn apply_label_elim(grid: &mut Grid<9, 3, 3>, target: Label, fired_k: u8) -> TechniqueProgress {
    let mut prog = TechniqueProgress::default();
    prog.chain_len = Some(fired_k as u32);
    let cell = (target / 9) as usize;
    let digit_bit = (target % 9) as u8;
    let digit = digit_bit + 1; // 1-based
    if grid.solved[cell] == 0
        && grid.candidates[cell] & (1u32 << digit_bit) != 0
    {
        match grid.eliminate(cell, digit) {
            Ok(_) => { prog.eliminations.push((cell, digit)); }
            Err(_) => { prog.contradiction = true; }
        }
    }
    prog
}

// ---------------------------------------------------------------------------
// Persistent GBraid search state (CR-FIN-15 perf pass, retained post-collapse)
// ---------------------------------------------------------------------------
//
// Tracks gbraid partials plus the companion partial-braid/whip/gwhip streams
// that gbraid extension depends on (CR-FIN-5 C2 cross-type subsumption guard).
// Per spec §17 the other three arms (Whip/GWhip/Braid) no longer have their
// own driver entry — but their partial-chain extension helpers are still used
// here as gbraid companion streams.
struct GBraidSearchState {
    gbraid_partials: Vec<Chain>,
    braid_partials: Vec<Chain>,
    whip_partials: Vec<Chain>,
    gwhip_partials: Vec<Chain>,
    dedup: HashSet<u64>,
    dedup_gwhip: HashSet<u64>,
    current_len: u8,
    exhausted: bool,
}

impl GBraidSearchState {
    fn new() -> Self {
        Self {
            gbraid_partials: Vec::new(),
            braid_partials: Vec::new(),
            whip_partials: Vec::new(),
            gwhip_partials: Vec::new(),
            dedup: HashSet::new(),
            dedup_gwhip: HashSet::new(),
            current_len: 0,
            exhausted: false,
        }
    }

    fn advance_to(&mut self, target_len: u8, ctx: &ChainContext<'_>) {
        use super::gbraid::{
            build_partial_braids_length_1_for_gbraid, build_partial_gbraids_length_1,
            extend_plain_braids,
        };
        if self.exhausted {
            return;
        }
        // Initial seed at length 1.
        if self.current_len == 0 && target_len >= 1 {
            self.gbraid_partials = build_partial_gbraids_length_1(ctx);
            self.braid_partials = build_partial_braids_length_1_for_gbraid(ctx);
            self.whip_partials = build_partial_whips_length_1(ctx);
            self.gwhip_partials = build_partial_gwhips_length_1(ctx, &self.whip_partials);
            // FX-R7 mirror: cross-type union dedup over gbraid ∪ gwhip seeds.
            self.dedup = self
                .gbraid_partials
                .iter()
                .chain(self.gwhip_partials.iter())
                .map(|c| c.dedup_key_cross_type())
                .collect();
            self.dedup_gwhip = self.gwhip_partials.iter().map(|c| c.dedup_key()).collect();
            self.current_len = 1;
            if self.gbraid_partials.is_empty()
                && self.braid_partials.is_empty()
                && self.whip_partials.is_empty()
                && self.gwhip_partials.is_empty()
            {
                self.exhausted = true;
                return;
            }
        }
        while self.current_len < target_len {
            // CR-FIN-5 C2 mirror: build length-k plain (braid ∪ whip) before extending gbraid.
            let mut braid_next: Vec<Chain> = Vec::new();
            extend_plain_braids(&self.braid_partials, &self.whip_partials, ctx, &mut braid_next);

            let mut tmp_dedup: HashSet<u64> =
                self.whip_partials.iter().map(|c| c.dedup_key()).collect();
            let mut whip_next: Vec<Chain> = Vec::new();
            extend_partial_whips(&self.whip_partials, ctx, &mut tmp_dedup, &mut whip_next);

            let plain_next_union: Vec<Chain> = braid_next
                .iter()
                .chain(whip_next.iter())
                .cloned()
                .collect();

            let combined_g_prev: Vec<Chain> = self
                .gbraid_partials
                .iter()
                .chain(self.gwhip_partials.iter())
                .cloned()
                .collect();
            let mut gbraid_next: Vec<Chain> = Vec::new();
            extend_partial_gbraids(
                &self.braid_partials,
                &combined_g_prev,
                &plain_next_union,
                ctx,
                &mut self.dedup,
                &mut gbraid_next,
            );

            // FX-R6 mirror: gwhip extension consumes partials_whip_prev (length k-1)
            // PLUS partials_whip_next (length k) for the cross-type subsumption guard.
            let mut gwhip_next: Vec<Chain> = Vec::new();
            extend_partial_gwhips(
                &self.gwhip_partials,
                &self.whip_partials,
                &whip_next,
                ctx,
                &mut self.dedup_gwhip,
                &mut gwhip_next,
            );

            self.gbraid_partials = gbraid_next;
            self.gwhip_partials = gwhip_next;
            self.braid_partials = braid_next;
            self.whip_partials = whip_next;
            self.current_len += 1;

            if self.gbraid_partials.is_empty()
                && self.braid_partials.is_empty()
                && self.whip_partials.is_empty()
                && self.gwhip_partials.is_empty()
            {
                self.exhausted = true;
                return;
            }
        }
    }

    fn probe_at_k(&self, k: u8, ctx: &ChainContext<'_>) -> Option<(ChainElimination, u8)> {
        // gbraid k-floor = 3 (CR-FIN-7 C2).
        if k < 3 || self.exhausted {
            return None;
        }
        debug_assert!(
            self.current_len == k - 1,
            "GBraidSearchState::probe_at_k: current_len ({}) != k-1 ({})",
            self.current_len, k - 1
        );
        for chain in &self.gbraid_partials {
            if let Some(e) = try_terminate_gbraid(chain, ctx) {
                return Some((e, k));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// ChainCombinedTechnique — GBraid-only driver (spec §17, 2026-05-20)
// ---------------------------------------------------------------------------

/// GBraid-only chain driver. The name is retained for API stability with the
/// rater pipeline (`T4PLUS_LIST` references `ChainCombined`) even though only
/// the GBraid arm is now probed.
///
/// Iterates k from 3 to `k_max` (GBraid k-floor = 3 per CR-FIN-7 C2) and
/// returns the first elimination produced by `gbraid[k]`. Per Berthier's
/// containment theorem (Whip ⊆ Braid ⊆ GBraid, GWhip ⊆ GBraid) this preserves
/// T&E(1) solving power.
///
/// On a hit, `TechniqueProgress::technique_id_override` is set to
/// `TechniqueId::GBraid`.
///
/// `id()` returns `TechniqueId::GBraid` (no placeholder needed post-collapse).
#[derive(Clone, Copy)]
pub struct ChainCombinedTechnique {
    /// Upper bound on the chain length probed inside `apply`. Defaults to
    /// `CHAIN_RATER_K_MAX = 36`.
    pub k_max: u8,
    /// `true` iff the GBraid arm has been excluded by `rate_excluding`. When
    /// set, `apply` is a no-op (returns `None`).
    pub excluded: bool,
}

impl ChainCombinedTechnique {
    /// Default-configured driver: `k_max = CHAIN_RATER_K_MAX`, not excluded.
    pub const DEFAULT: Self = Self {
        k_max: CHAIN_RATER_K_MAX,
        excluded: false,
    };

    /// Construct with a specific `k_max` cap (capped at 36 per spec §4.1).
    pub fn with_k_max(k_max: u8) -> Self {
        Self {
            k_max: k_max.min(CHAIN_RATER_K_MAX),
            excluded: false,
        }
    }

    /// Construct with an exclusion list. Post-collapse only `TechniqueId::GBraid`
    /// is meaningful; pre-collapse `Whip`/`GWhip`/`Braid` ids in the list are
    /// accepted but ignored (the driver no longer probes those arms). When
    /// `GBraid` is in the list, the driver is fully excluded.
    pub fn with_excluded(exclude: &[TechniqueId]) -> Self {
        let excluded = exclude.iter().any(|&id| matches!(id, TechniqueId::GBraid));
        Self {
            k_max: CHAIN_RATER_K_MAX,
            excluded,
        }
    }

    /// `true` iff the (now sole) GBraid arm is masked — caller should skip
    /// the driver entirely in that case.
    #[inline]
    pub fn all_arms_excluded(&self) -> bool {
        self.excluded
    }
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for ChainCombinedTechnique {
    fn id(&self) -> TechniqueId { TechniqueId::GBraid }
    fn tier(&self) -> Tier { Tier::T4Plus }
    fn name(&self) -> &'static str { "chain-combined" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        if N != 9 || BR != 3 || BC != 3 {
            return None;
        }
        if self.excluded {
            return None;
        }
        // SAFETY: verified N=9, BR=3, BC=3 above. `Grid` is `#[repr(C)]`.
        let grid9: &mut Grid<9, 3, 3> = unsafe {
            &mut *(grid as *mut Grid<N, BR, BC> as *mut Grid<9, 3, 3>)
        };
        let (csp, link, cspl, glab, glnk) = get_tables();
        let mut rs = ResolutionState::from_grid_9x9(grid9, glab);

        let k_max = self.k_max.min(CHAIN_RATER_K_MAX);
        let mut gbraid_state = GBraidSearchState::new();

        for k in 1u8..=k_max {
            // GBraid k-floor = 3. For k<3 we still need to advance the partials,
            // but skip the terminator probe.
            if k >= 2 {
                let ctx = ChainContext { csp, link, cspl, glab, glnk, rs: &mut rs };
                gbraid_state.advance_to(k - 1, &ctx);
            }
            if k < 3 {
                continue;
            }
            let ctx = ChainContext { csp, link, cspl, glab, glnk, rs: &mut rs };
            if let Some((elim, k_fired)) = gbraid_state.probe_at_k(k, &ctx) {
                let mut prog = apply_label_elim(grid9, elim.target, k_fired);
                prog.technique_id_override = Some(TechniqueId::GBraid);
                if !(prog.eliminations.is_empty() && !prog.contradiction) {
                    return Some(prog);
                }
                // Phantom-hit fall-through: continue to next k.
            }
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC> for ChainCombinedTechnique {
    fn base_rating(&self) -> f64 { 8.5 } // GB[3] base
    fn se_rating(&self, progress: &TechniqueProgress) -> f64 {
        let k_u8 = progress.chain_len.unwrap_or(3) as u8;
        chain_rating_to_se(ChainRating::GB(k_u8)) as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::backtracker::propagate_singles;
    use crate::generic::rater::rate_excluding;

    fn grid_after_singles(p: &str) -> Grid<9, 3, 3> {
        let mut g = Grid::<9, 3, 3>::from_str(p).expect("parse");
        let _ = propagate_singles(&mut g);
        g
    }

    /// `CHAIN_RATER_K_MAX` matches spec §4.1 cap.
    #[test]
    fn chain_rater_k_max_is_spec_capped() {
        assert_eq!(CHAIN_RATER_K_MAX, 36);
    }

    /// `with_excluded(&[GBraid])` masks the (now sole) chain arm.
    #[test]
    fn with_excluded_gbraid_masks_driver() {
        let t = ChainCombinedTechnique::with_excluded(&[TechniqueId::GBraid]);
        assert!(t.all_arms_excluded());
        let t2 = ChainCombinedTechnique::with_excluded(&[]);
        assert!(!t2.all_arms_excluded());
        // Legacy (pre-collapse) ids are accepted but do not mask the driver.
        let t3 = ChainCombinedTechnique::with_excluded(&[TechniqueId::Whip]);
        assert!(!t3.all_arms_excluded());
    }

    /// Excluding GBraid: `rate_excluding(&[GBraid])` skips the chain pass
    /// entirely; no chain id appears in the frontier.
    #[test]
    fn rate_excluding_gbraid_skips_chain() {
        let p = ".23....8.......12.7.91.........97.46...3....29..6......6.9..5.457.....98.....4...";
        let g = Grid::<9, 3, 3>::from_str(p).expect("parse");
        let r = rate_excluding(&g, &[TechniqueId::GBraid]);
        assert!(!r.rater_error);
        assert!(!r.frontier.contains(&TechniqueId::GBraid));
    }

    /// All-arms-excluded driver returns None on apply.
    #[test]
    fn all_arms_excluded_returns_none() {
        let p = ".23....8.......12.7.91.........97.46...3....29..6......6.9..5.457.....98.....4...";
        let mut g = grid_after_singles(p);
        let t = ChainCombinedTechnique::with_excluded(&[TechniqueId::GBraid]);
        let prog = <ChainCombinedTechnique as Technique<9, 3, 3>>::apply(&t, &mut g);
        assert!(prog.is_none());
    }
}
