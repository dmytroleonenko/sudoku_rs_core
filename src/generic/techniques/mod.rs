//! Const-generic `Technique<N, BR, BC>` trait + tier classification.
//!
//! Sub-stage 2: locked candidates (pointing/claiming) + naked/hidden sets
//! (pair/triple/quad). Singles propagation (T1) is handled by
//! `super::backtracker::propagate_singles`, not as a `Technique` impl.
//!
//! Subsequent sub-stages add Fish, AIC, ALS-XZ, UR, and remainder — they all
//! implement this same trait so the rater can iterate them uniformly.
//!
//! ## Stage R additions (R3.4)
//!
//! `RatedTechnique<N,BR,BC>` extends `Technique` with SE-equivalent rating.
//! `all_techniques_9x9()` is the single registry point for 9×9 technique
//! assembly; AlphaEvolve / OpenEvolve adds a new technique by:
//!   1. Creating `techniques/my_technique.rs` implementing `Technique`.
//!   2. Adding one `Box::new(MyTechnique)` line to `all_techniques_9x9()`.
//!   No other file needs editing.

use super::grid::Grid;

pub mod result;
pub mod locked;
pub mod naked_set;
pub mod hidden_set;
pub mod fish;
pub mod aic;
pub mod unique_rect;
pub mod als_xz;
pub mod xy_wing;
pub mod xyz_wing;
pub mod bug;
pub mod simple_coloring;
pub mod skyscraper;
pub mod two_string_kite;
pub mod chaining_propagator;
pub mod cell_fc;
pub mod region_fc;
pub mod dynamic_fc;
pub mod nested_fc;
pub mod whip;
pub mod gwhip;
pub mod braid;
pub mod gbraid;
pub mod chain_rated;

pub use result::TechniqueProgress;
pub use locked::{LockedPointing, LockedClaiming};
pub use naked_set::{NakedSet, NakedPair, NakedTriple, NakedQuad};
pub use hidden_set::{HiddenSet, HiddenPair, HiddenTriple, HiddenQuad};
pub use fish::{Fish, XWing, Swordfish, Jellyfish};
pub use aic::Aic;
pub use unique_rect::{UniqueRectangle, UrType1, UrType2};
pub use als_xz::AlsXz;
pub use xy_wing::XyWing;
pub use xyz_wing::XyzWing;
pub use bug::Bug;
pub use simple_coloring::SimpleColoring;
pub use skyscraper::Skyscraper;
pub use two_string_kite::TwoStringKite;
pub use cell_fc::CellForcingChain;
pub use region_fc::RegionForcingChain;
pub use dynamic_fc::DynamicForcingChain;
pub use nested_fc::NestedForcingChain;
pub use chain_rated::ChainCombinedTechnique;

/// Identifier for a single technique. Stable across sub-stages — variants must
/// not be renumbered (rater rules may key off these).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TechniqueId {
    LockedPointing,
    LockedClaiming,
    NakedPair,
    NakedTriple,
    NakedQuad,
    HiddenPair,
    HiddenTriple,
    HiddenQuad,
    // Sub-stage 3 additions:
    XWing,
    Swordfish,
    Jellyfish,
    Aic,
    UrType1,
    UrType2,
    AlsXz,
    XyWing,
    XyzWing,
    Bug,
    SimpleColoring,
    Skyscraper,
    TwoStringKite,
    CellForcingChain,
    RegionForcingChain,
    DynamicForcingChain,
    NestedForcingChain,
    NestedForcingChainL3,
    NestedForcingChainL4,
    // Chain techniques (Berthier W/B/gW/gB axis)
    Whip,
    GWhip,
    Braid,
    GBraid,
}

/// Difficulty tier (rater bucketing). T1 = pure singles propagation, no
/// technique needed. T2 = locked candidates + naked/hidden pairs. T3 =
/// triples/quads + simple chains. T4Plus = everything more advanced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    T1,
    T2,
    T3,
    T4Plus,
}

pub trait Technique<const N: usize, const BR: usize, const BC: usize>: Send + Sync {
    fn id(&self) -> TechniqueId;
    fn tier(&self) -> Tier;
    fn name(&self) -> &'static str;
    /// Apply the technique once. Returns `None` if no eliminations / placements
    /// were made (the technique did not fire on this grid state). Returns
    /// `Some(progress)` if at least one change was made; the grid is mutated
    /// in place. `progress.contradiction` is set if a contradiction was
    /// triggered while eliminating.
    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress>;
}

// ---------------------------------------------------------------------------
// RatedTechnique — SE-equivalent difficulty extension.
//
// Stage 0 will replace the blanket impl with per-technique overrides that
// return calibrated SE weights (from SudokuExplainer 1.2.1 / HoDoKu tables).
// For Stage R the blanket impl returning tier-bucket constants is sufficient.
// ---------------------------------------------------------------------------

/// Extension of `Technique` that carries SE-equivalent difficulty rating.
///
/// `se_rating` returns the difficulty weight for a *single fire* of this
/// technique (not cumulative across the solve). The rater accumulates
/// `max(se_rating per fire)` as the puzzle's overall `se_score`.
///
/// `base_rating` is the lower bound / default — used for technique ordering
/// (cheap-first) and as the default `se_rating`. Override `se_rating` for
/// chain-length-dependent techniques (AIC, FC) where the cost scales with
/// `progress` content (e.g. chain length stored in `progress.eliminations`).
pub trait RatedTechnique<const N: usize, const BR: usize, const BC: usize>:
    Technique<N, BR, BC>
{
    /// SE-equivalent difficulty for one fire of this technique.
    /// Default delegates to `base_rating()`. Override for variable-cost
    /// techniques (AIC, Forcing Chains) once Stage 0 lands.
    fn se_rating(&self, _progress: &TechniqueProgress) -> f64 {
        self.base_rating()
    }

    /// Lower bound on `se_rating`. Used for technique ordering and as the
    /// default se_rating. Stage 0 will replace this blanket impl.
    fn base_rating(&self) -> f64;
}

// Stage 0: blanket impl removed. Each technique file provides its own
// explicit `impl RatedTechnique<N,BR,BC> for X` with calibrated SE weights
// (HoDoKu / SudokuExplainer 1.2.1 table). See individual technique files.

// ---------------------------------------------------------------------------
// Registry — single assembly point for all known techniques.
//
// AlphaEvolve / OpenEvolve workflow:
//   1. Create `techniques/my_technique.rs` implementing `Technique<9,3,3>`.
//   2. Add `pub mod my_technique;` + `pub use` above.
//   3. Add `Box::new(MyTechnique)` in `all_techniques_9x9()` below.
//   No other file needs editing.
//
// Order must be cheap-first (T2 before T3, faster techniques first within
// tier) — this mirrors the cascade order in `rater.rs` and `T2_LIST`/
// `T3_LIST` constants there. Rater.rs continues to use its own static-
// dispatch `AnyTechnique` enum for zero-cost monomorphisation; `all_techniques_9x9`
// is for benchmarks, lint tests, and future dynamic-dispatch callers.
// ---------------------------------------------------------------------------

/// Single point of assembly for all techniques known to the rater (9×9).
///
/// Returns techniques in cascade order (cheap-first / T2 before T3).
/// Add new techniques here; no other file needs editing.
///
/// # Note on generic variants
/// Some techniques (`NakedSet<K>`, `HiddenSet<K>`, `Fish<K>`) are internally
/// generic over K. The 9×9 registry instantiates the concrete arities used by
/// the rater. A fully generic `all_techniques<N,BR,BC>()` is deferred (TODO)
/// because several techniques are constrained to specific N values or require
/// additional const bounds not yet in scope.
pub fn all_techniques_9x9() -> Vec<Box<dyn Technique<9, 3, 3>>> {
    vec![
        // T2 — cheap, restart after each fire
        Box::new(LockedPointing),
        Box::new(LockedClaiming),
        Box::new(naked_set::NakedSet::<2>),   // NakedPair
        Box::new(hidden_set::HiddenSet::<2>), // HiddenPair
        Box::new(naked_set::NakedSet::<3>),   // NakedTriple
        Box::new(hidden_set::HiddenSet::<3>), // HiddenTriple
        Box::new(naked_set::NakedSet::<4>),   // NakedQuad
        Box::new(hidden_set::HiddenSet::<4>), // HiddenQuad
        // T3 — more expensive, cascade order mirrors T3_LIST in rater.rs
        Box::new(fish::Fish::<2>),  // XWing
        Box::new(fish::Fish::<3>),  // Swordfish
        Box::new(fish::Fish::<4>),  // Jellyfish
        Box::new(UrType1),
        Box::new(UrType2),
        Box::new(XyWing),
        Box::new(XyzWing),
        Box::new(SimpleColoring),
        Box::new(Skyscraper),
        Box::new(TwoStringKite),
        Box::new(Bug),
        Box::new(AlsXz),
        Box::new(Aic),
        // T4Plus — expensive, positioned last
        Box::new(CellForcingChain),
        Box::new(RegionForcingChain),
        Box::new(DynamicForcingChain::default()),
        Box::new(NestedForcingChain::default()),
        Box::new(NestedForcingChain::level_3()),
        Box::new(NestedForcingChain::level_4()),
        // Berthier chain technique: GBraid-only post-collapse (spec §17).
        // Per Berthier's containment theorem (Whip ⊆ Braid ⊆ GBraid,
        // GWhip ⊆ GBraid), GBraid alone preserves T&E(1) solving power.
        Box::new(chain_rated::ChainCombinedTechnique::DEFAULT),
    ]
}
