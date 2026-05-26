//! Common result type for `Technique<N, BR, BC>` impls.
//!
//! Mirrors the semantics of `crate::techniques::Action::Fired` but in a more
//! compact form: a technique either returns `None` (no-op, no change) or
//! `Some(TechniqueProgress)` with the placements/eliminations it produced.
//!
//! Contradictions during elimination are propagated as
//! `TechniqueProgress::contradiction = true` and the caller (rater/solver)
//! decides how to handle them.

#[derive(Debug, Clone, Default)]
pub struct TechniqueProgress {
    /// Cells that were placed (cell_idx, digit). For locked/naked/hidden sets
    /// this list is empty — these techniques only eliminate.
    pub placements: Vec<(usize, u8)>,
    /// Eliminations performed (cell_idx, digit).
    pub eliminations: Vec<(usize, u8)>,
    /// True iff a contradiction was detected during application.
    pub contradiction: bool,
    /// Chain length proxy for chain-length-dependent techniques (AIC, FC).
    /// `None` for non-chain techniques.
    pub chain_len: Option<u32>,
    /// Set by AIC to distinguish XY-Chain (bivalue cells with digit change)
    /// from plain X-Chain (single digit). `Some(true)` → XY-Chain (base 7.0);
    /// `Some(false)` → X-Chain (base 6.6); `None` → not applicable.
    pub is_xy_chain: Option<bool>,
    /// Set by CellForcingChain to distinguish Y-Chain (k=2 branches, SE 6.6)
    /// from full CFC (k≥3 branches, SE 8.0). `None` → not set.
    pub k_branches: Option<u8>,
    /// Override for the technique id actually fired. Used by `ChainCombinedTechnique`
    /// (CR-FIN-2): the combined driver fires as one of W/GW/B/GB; this field
    /// carries the actual id so the rater can attribute tier/se_rating correctly.
    /// `None` → use the wrapper's own `id()`.
    pub technique_id_override: Option<super::TechniqueId>,
}

impl TechniqueProgress {
    pub fn fired(&self) -> bool {
        !self.placements.is_empty() || !self.eliminations.is_empty()
    }
}
