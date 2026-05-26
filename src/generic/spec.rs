//! `TechniqueChainSpec` — declarative spec for constrained-removal generation.
//!
//! Path B subset (P2): tier band, required-technique set, clue-count band.
//! No AIC/Fish chain-length parameters (those live in Path A — deferred to P3).
//!
//! Semantics: a `RateResult` "matches" a spec iff
//!   - `result.tier == spec.target_tier` (exact match, not "≥"); and
//!   - every `id` in `spec.required_techniques` appears in `result.frontier`; and
//!   - `clue_count` is within `[spec.clue_min, spec.clue_max]` (inclusive).
//!
//! Required-techniques semantics is membership-only: we don't yet check
//! load-bearing-ness here. The `gen_constrained` driver runs the load-bearing
//! probe (`rate_excluding(grid, &required)`) on accepted puzzles and rejects
//! ones where excluding the required set still solves at the target tier.

use super::rater::RateResult;
use super::techniques::{TechniqueId, Tier};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TechniqueChainSpec {
    pub target_tier: Tier,
    /// All of these technique ids must appear in the rated frontier.
    /// Empty list disables the technique check (any frontier is OK as long
    /// as tier matches).
    pub required_techniques: Vec<TechniqueId>,
    /// Inclusive clue-count band.
    pub clue_min: u32,
    pub clue_max: u32,
}

impl TechniqueChainSpec {
    /// Convenience constructor for an unconstrained spec (any tier, any clue
    /// count, no required techniques).
    pub fn any(target_tier: Tier) -> Self {
        Self {
            target_tier,
            required_techniques: Vec::new(),
            clue_min: 0,
            clue_max: u32::MAX,
        }
    }

    pub fn matches(&self, result: &RateResult, clue_count: u32) -> bool {
        if result.rater_error {
            return false;
        }
        if !tier_eq(result.tier, self.target_tier) {
            return false;
        }
        if clue_count < self.clue_min || clue_count > self.clue_max {
            return false;
        }
        for req in &self.required_techniques {
            if !result.frontier.iter().any(|id| *id == *req) {
                return false;
            }
        }
        true
    }
}

#[inline]
fn tier_eq(a: Tier, b: Tier) -> bool {
    matches!(
        (a, b),
        (Tier::T1, Tier::T1)
            | (Tier::T2, Tier::T2)
            | (Tier::T3, Tier::T3)
            | (Tier::T4Plus, Tier::T4Plus)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::rater::RateResult;

    fn mk_result(tier: Tier, frontier: Vec<TechniqueId>) -> RateResult {
        RateResult {
            tier,
            frontier,
            solved: true,
            trace: Vec::new(),
            rater_error: false,
            wave_depth: 0,
            backtrack_steps: 0,
            unique_solution: false,
            se_score: 0.0,
        }
    }

    #[test]
    fn matches_exact_tier_no_techs() {
        let s = TechniqueChainSpec::any(Tier::T2);
        assert!(s.matches(&mk_result(Tier::T2, vec![]), 25));
        assert!(!s.matches(&mk_result(Tier::T1, vec![]), 25));
        assert!(!s.matches(&mk_result(Tier::T3, vec![]), 25));
    }

    #[test]
    fn matches_clue_band() {
        let s = TechniqueChainSpec {
            target_tier: Tier::T3,
            required_techniques: vec![],
            clue_min: 22,
            clue_max: 28,
        };
        assert!(s.matches(&mk_result(Tier::T3, vec![]), 22));
        assert!(s.matches(&mk_result(Tier::T3, vec![]), 28));
        assert!(!s.matches(&mk_result(Tier::T3, vec![]), 21));
        assert!(!s.matches(&mk_result(Tier::T3, vec![]), 29));
    }

    #[test]
    fn matches_required_techs() {
        let s = TechniqueChainSpec {
            target_tier: Tier::T3,
            required_techniques: vec![TechniqueId::Aic, TechniqueId::XyWing],
            clue_min: 0,
            clue_max: 81,
        };
        assert!(s.matches(
            &mk_result(Tier::T3, vec![TechniqueId::Aic, TechniqueId::XyWing, TechniqueId::NakedTriple]),
            24,
        ));
        // Missing one of the two required techs.
        assert!(!s.matches(&mk_result(Tier::T3, vec![TechniqueId::Aic]), 24));
        // None of the required techs.
        assert!(!s.matches(&mk_result(Tier::T3, vec![TechniqueId::NakedTriple]), 24));
    }

    #[test]
    fn rater_error_never_matches() {
        let s = TechniqueChainSpec::any(Tier::T1);
        let mut r = mk_result(Tier::T1, vec![]);
        r.rater_error = true;
        assert!(!s.matches(&r, 25));
    }
}
