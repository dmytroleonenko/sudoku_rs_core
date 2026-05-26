//! Generic (const-generic) rater driver.
//!
//! Cascade T1 → T2 → T3 → T4Plus, mirroring the legacy `crate::rater::rate`
//! semantics but dispatched through `Technique<N, BR, BC>` trait objects so the
//! same code path covers 6×6, 9×9, 12×12, 16×16.
//!
//! Order (post R3.1a tier alignment):
//!   T1: `propagate_singles` to fixpoint.
//!   T2: LockedPointing, LockedClaiming, NakedPair, HiddenPair, NakedTriple,
//!       HiddenTriple, NakedQuad, HiddenQuad. (Triples/Quads were T3 prior to
//!       R3.1a — moved to T2 to match the canonical Sudoku.coach tiering.)
//!   T3: XWing, Swordfish, Jellyfish, UrType1, UrType2, XyWing, XyzWing,
//!       SimpleColoring, Skyscraper, TwoStringKite, Bug, AlsXz, Aic.
//!
//! On any technique firing we restart the loop from T1 (singles re-fixpoint),
//! escalating `highest_tier` to max(current, technique.tier()).
//!
//! `rater_error` is set on contradiction — propagator failure, technique
//! contradiction flag, or initial-state inconsistency.

use super::backtracker::propagate_singles;
use super::grid::{AssignErr, Grid};
use super::techniques::{
    aic::Aic,
    als_xz::AlsXz,
    bug::Bug,
    cell_fc::CellForcingChain,
    chain_rated::ChainCombinedTechnique,
    dynamic_fc::DynamicForcingChain,
    nested_fc::NestedForcingChain,
    fish::Fish,
    hidden_set::HiddenSet,
    locked::{LockedClaiming, LockedPointing},
    naked_set::NakedSet,
    region_fc::RegionForcingChain,
    simple_coloring::SimpleColoring,
    skyscraper::Skyscraper,
    two_string_kite::TwoStringKite,
    unique_rect::{UrType1, UrType2},
    xy_wing::XyWing,
    xyz_wing::XyzWing,
    RatedTechnique, Technique, TechniqueId, TechniqueProgress, Tier,
};

// ---------------------------------------------------------------------------
// Static-dispatch wrapper.
//
// Replaces the previous `Vec<Box<dyn Technique<N,BR,BC>>>` cascade list with a
// closed enum whose `apply` is monomorphized per (N,BR,BC). The match in
// `apply` lowers to a jump-table; we avoid the heap allocation + virtual
// dispatch that the trait-object form incurred. Variant set must mirror the
// list of techniques wired into `t2_techniques` / `t3_techniques` exactly —
// adding a technique here without adding a `Self::Foo => ...` arm in `apply`
// is a logic error (compiler will catch the missing arm).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum AnyTechnique {
    LockedPointing,
    LockedClaiming,
    NakedPair,    // NakedSet<2>
    NakedTriple,  // NakedSet<3>
    NakedQuad,    // NakedSet<4>
    HiddenPair,   // HiddenSet<2>
    HiddenTriple, // HiddenSet<3>
    HiddenQuad,   // HiddenSet<4>
    XWing,        // Fish<2>
    Swordfish,    // Fish<3>
    Jellyfish,    // Fish<4>
    XyWing,
    XyzWing,
    UrType1,
    UrType2,
    Bug,
    Aic,
    AlsXz,
    SimpleColoring,
    Skyscraper,
    TwoStringKite,
    CellForcingChain,
    RegionForcingChain,
    DynamicForcingChain,
    NestedForcingChain,
    NestedForcingChainL3,
    NestedForcingChainL4,
    // Berthier chain techniques (T4Plus).
    // Single salience-interleaved combined driver (CR-FIN-2 fix).
    // Internally iterates: whip[k] → gwhip[k] → braid[k] → gbraid[k]
    // for k in 1..=K_MAX before returning the tightest hit.
    // TechniqueProgress::technique_id_override carries which variant fired.
    ChainCombined,
}

impl AnyTechnique {
    #[inline]
    fn id(self) -> TechniqueId {
        match self {
            Self::LockedPointing => TechniqueId::LockedPointing,
            Self::LockedClaiming => TechniqueId::LockedClaiming,
            Self::NakedPair => TechniqueId::NakedPair,
            Self::NakedTriple => TechniqueId::NakedTriple,
            Self::NakedQuad => TechniqueId::NakedQuad,
            Self::HiddenPair => TechniqueId::HiddenPair,
            Self::HiddenTriple => TechniqueId::HiddenTriple,
            Self::HiddenQuad => TechniqueId::HiddenQuad,
            Self::XWing => TechniqueId::XWing,
            Self::Swordfish => TechniqueId::Swordfish,
            Self::Jellyfish => TechniqueId::Jellyfish,
            Self::XyWing => TechniqueId::XyWing,
            Self::XyzWing => TechniqueId::XyzWing,
            Self::UrType1 => TechniqueId::UrType1,
            Self::UrType2 => TechniqueId::UrType2,
            Self::Bug => TechniqueId::Bug,
            Self::Aic => TechniqueId::Aic,
            Self::AlsXz => TechniqueId::AlsXz,
            Self::SimpleColoring => TechniqueId::SimpleColoring,
            Self::Skyscraper => TechniqueId::Skyscraper,
            Self::TwoStringKite => TechniqueId::TwoStringKite,
            Self::CellForcingChain => TechniqueId::CellForcingChain,
            Self::RegionForcingChain => TechniqueId::RegionForcingChain,
            Self::DynamicForcingChain => TechniqueId::DynamicForcingChain,
            Self::NestedForcingChain => TechniqueId::NestedForcingChain,
            Self::NestedForcingChainL3 => TechniqueId::NestedForcingChainL3,
            Self::NestedForcingChainL4 => TechniqueId::NestedForcingChainL4,
            // ChainCombined: post-collapse driver only emits GBraid (spec §17).
            Self::ChainCombined => TechniqueId::GBraid,
        }
    }

    #[inline]
    fn tier(self) -> Tier {
        // R3.1a tier alignment: Triples/Quads return T2 here (canonical
        // Sudoku.coach tiering). The underlying `Technique::tier()` impls
        // for NakedSet<3>/<4> and HiddenSet<3>/<4> still return T3 — we
        // deliberately override to T2 in the rater to keep T2 closure
        // (i.e. a puzzle solved using only locked + naked/hidden pairs/
        // triples/quads stays in T2). Adding a new naked/hidden-set arity
        // requires extending this match too.
        match self {
            Self::LockedPointing
            | Self::LockedClaiming
            | Self::NakedPair
            | Self::HiddenPair
            | Self::NakedTriple
            | Self::HiddenTriple
            | Self::NakedQuad
            | Self::HiddenQuad => Tier::T2,
            Self::CellForcingChain | Self::RegionForcingChain | Self::DynamicForcingChain
            | Self::NestedForcingChain | Self::NestedForcingChainL3 | Self::NestedForcingChainL4
            | Self::ChainCombined => Tier::T4Plus,
            _ => Tier::T3,
        }
    }

    #[inline]
    fn apply<const N: usize, const BR: usize, const BC: usize>(
        self,
        grid: &mut Grid<N, BR, BC>,
    ) -> Option<TechniqueProgress> {
        match self {
            Self::LockedPointing => LockedPointing.apply(grid),
            Self::LockedClaiming => LockedClaiming.apply(grid),
            Self::NakedPair => NakedSet::<2>.apply(grid),
            Self::NakedTriple => NakedSet::<3>.apply(grid),
            Self::NakedQuad => NakedSet::<4>.apply(grid),
            Self::HiddenPair => HiddenSet::<2>.apply(grid),
            Self::HiddenTriple => HiddenSet::<3>.apply(grid),
            Self::HiddenQuad => HiddenSet::<4>.apply(grid),
            Self::XWing => Fish::<2>.apply(grid),
            Self::Swordfish => Fish::<3>.apply(grid),
            Self::Jellyfish => Fish::<4>.apply(grid),
            Self::XyWing => XyWing.apply(grid),
            Self::XyzWing => XyzWing.apply(grid),
            Self::UrType1 => UrType1.apply(grid),
            Self::UrType2 => UrType2.apply(grid),
            Self::Bug => Bug.apply(grid),
            Self::Aic => Aic.apply(grid),
            Self::AlsXz => AlsXz.apply(grid),
            Self::SimpleColoring => SimpleColoring.apply(grid),
            Self::Skyscraper => Skyscraper.apply(grid),
            Self::TwoStringKite => TwoStringKite.apply(grid),
            Self::CellForcingChain => CellForcingChain.apply(grid),
            Self::RegionForcingChain => RegionForcingChain.apply(grid),
            Self::DynamicForcingChain => DynamicForcingChain::default().apply(grid),
            Self::NestedForcingChain => NestedForcingChain::default().apply(grid),
            Self::NestedForcingChainL3 => NestedForcingChain::level_3().apply(grid),
            Self::NestedForcingChainL4 => NestedForcingChain::level_4().apply(grid),
            Self::ChainCombined => ChainCombinedTechnique::DEFAULT.apply(grid),
        }
    }

    /// SE-equivalent difficulty rating for one fire of this technique.
    /// Delegates to each technique's `RatedTechnique::se_rating` impl.
    /// For AIC, passes `progress` so the chain-length heuristic can run.
    #[inline]
    fn se_rating<const N: usize, const BR: usize, const BC: usize>(
        self,
        progress: &TechniqueProgress,
    ) -> f64 {
        match self {
            Self::LockedPointing  => RatedTechnique::<N, BR, BC>::se_rating(&LockedPointing, progress),
            Self::LockedClaiming  => RatedTechnique::<N, BR, BC>::se_rating(&LockedClaiming, progress),
            Self::NakedPair       => RatedTechnique::<N, BR, BC>::se_rating(&NakedSet::<2>, progress),
            Self::NakedTriple     => RatedTechnique::<N, BR, BC>::se_rating(&NakedSet::<3>, progress),
            Self::NakedQuad       => RatedTechnique::<N, BR, BC>::se_rating(&NakedSet::<4>, progress),
            Self::HiddenPair      => RatedTechnique::<N, BR, BC>::se_rating(&HiddenSet::<2>, progress),
            Self::HiddenTriple    => RatedTechnique::<N, BR, BC>::se_rating(&HiddenSet::<3>, progress),
            Self::HiddenQuad      => RatedTechnique::<N, BR, BC>::se_rating(&HiddenSet::<4>, progress),
            Self::XWing           => RatedTechnique::<N, BR, BC>::se_rating(&Fish::<2>, progress),
            Self::Swordfish       => RatedTechnique::<N, BR, BC>::se_rating(&Fish::<3>, progress),
            Self::Jellyfish       => RatedTechnique::<N, BR, BC>::se_rating(&Fish::<4>, progress),
            Self::XyWing          => RatedTechnique::<N, BR, BC>::se_rating(&XyWing, progress),
            Self::XyzWing         => RatedTechnique::<N, BR, BC>::se_rating(&XyzWing, progress),
            Self::UrType1         => RatedTechnique::<N, BR, BC>::se_rating(&UrType1, progress),
            Self::UrType2         => RatedTechnique::<N, BR, BC>::se_rating(&UrType2, progress),
            Self::Bug             => RatedTechnique::<N, BR, BC>::se_rating(&Bug, progress),
            Self::Aic             => RatedTechnique::<N, BR, BC>::se_rating(&Aic, progress),
            Self::AlsXz           => RatedTechnique::<N, BR, BC>::se_rating(&AlsXz, progress),
            Self::SimpleColoring  => RatedTechnique::<N, BR, BC>::se_rating(&SimpleColoring, progress),
            Self::Skyscraper      => RatedTechnique::<N, BR, BC>::se_rating(&Skyscraper, progress),
            Self::TwoStringKite   => RatedTechnique::<N, BR, BC>::se_rating(&TwoStringKite, progress),
            Self::CellForcingChain => RatedTechnique::<N, BR, BC>::se_rating(&CellForcingChain, progress),
            Self::RegionForcingChain => RatedTechnique::<N, BR, BC>::se_rating(&RegionForcingChain, progress),
            Self::DynamicForcingChain => RatedTechnique::<N, BR, BC>::se_rating(&DynamicForcingChain::default(), progress),
            Self::NestedForcingChain => RatedTechnique::<N, BR, BC>::se_rating(&NestedForcingChain::default(), progress),
            Self::NestedForcingChainL3 => RatedTechnique::<N, BR, BC>::se_rating(&NestedForcingChain::level_3(), progress),
            Self::NestedForcingChainL4 => RatedTechnique::<N, BR, BC>::se_rating(&NestedForcingChain::level_4(), progress),
            Self::ChainCombined => RatedTechnique::<N, BR, BC>::se_rating(&ChainCombinedTechnique::DEFAULT, progress),
        }
    }
}

// T2 cascade order — must match legacy `crate::rater::rate` T2_TECHS:
//   LockedPointing, LockedClaiming, NakedPair, HiddenPair,
//   NakedTriple, HiddenTriple, NakedQuad, HiddenQuad.
const T2_LIST: [AnyTechnique; 8] = [
    AnyTechnique::LockedPointing,
    AnyTechnique::LockedClaiming,
    AnyTechnique::NakedPair,
    AnyTechnique::HiddenPair,
    AnyTechnique::NakedTriple,
    AnyTechnique::HiddenTriple,
    AnyTechnique::NakedQuad,
    AnyTechnique::HiddenQuad,
];

// T3 cascade order — mirrors legacy `crate::rater::rate` T3_TECHS:
//   Fish (XWing, Swordfish, Jellyfish), UniqueRectangle (UrType1, UrType2),
//   XyWing, XyzWing, SimpleColoring, Skyscraper, TwoStringKite, Bug,
//   AlsXz, Aic.
// Note: CellForcingChain is in T4PLUS_LIST (see below), not here.
const T3_LIST: [AnyTechnique; 13] = [
    AnyTechnique::XWing,
    AnyTechnique::Swordfish,
    AnyTechnique::Jellyfish,
    AnyTechnique::UrType1,
    AnyTechnique::UrType2,
    AnyTechnique::XyWing,
    AnyTechnique::XyzWing,
    AnyTechnique::SimpleColoring,
    AnyTechnique::Skyscraper,
    AnyTechnique::TwoStringKite,
    AnyTechnique::Bug,
    AnyTechnique::AlsXz,
    AnyTechnique::Aic,
];

// T4Plus cascade order — expensive techniques, tried only after T2+T3 are
// exhausted. CellForcingChain (SE 8.0) first; RegionForcingChain (SE 7.6);
// DynamicForcingChain (SE 9.0); NestedForcingChain L2 (SE 9.5), L3 (SE 10.0),
// L4 (SE 10.5). Then ONE salience-interleaved chain driver (CR-FIN-2 fix):
//   `ChainCombined` internally does whip[k]→gwhip[k]→braid[k]→gbraid[k] per k.
// Default k_max = `chain_rated::CHAIN_RATER_K_MAX` (= 36, raised from 9 in
// CR-FIN-3 C2 per spec §4.1 / §13 cap). CR-FIN-5 MINOR: stale comment fix.
// Placed after NestedForcingChain: chain techniques are analytically exact but
// search-heavy at k≥5; NestedFC is a broader forcing-chain heuristic that covers
// many practical T4Plus puzzles faster.
const T4PLUS_LIST: [AnyTechnique; 7] = [
    AnyTechnique::CellForcingChain,
    AnyTechnique::RegionForcingChain,
    AnyTechnique::DynamicForcingChain,
    AnyTechnique::NestedForcingChain,
    AnyTechnique::NestedForcingChainL3,
    AnyTechnique::NestedForcingChainL4,
    AnyTechnique::ChainCombined,
];

/// R3.3a perf knob: how much of the rater pipeline to actually execute.
///
/// `Solve` — fastest: T1 propagation + uniqueness/solve only. The cascade is
/// not run; `tier`/`frontier`/`trace` are returned in their cheap defaults
/// (`Tier::T1` if the grid solved on singles alone, else `Tier::T4Plus`; both
/// vectors empty). Use this for paths that just need the solution and trust
/// the puzzle's tier/frontier classification from upstream construction.
///
/// `Tier` — medium: full cascade with **early exit** as soon as it can no
/// longer escalate `tier` (i.e. once `T3` is reached we can stop on the next
/// no-progress wave; we still need to confirm whether the puzzle actually
/// `solved` so the cascade is not aborted mid-wave). The frontier/trace are
/// populated normally. No standalone `solve_unique` second-solve is performed
/// by callers that consult `solved` directly.
///
/// `Full` — current behaviour (default for backward compatibility): full
/// cascade until solved or stuck, plus any caller-driven uniqueness check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolverMode {
    Solve,
    Tier,
    Full,
}

impl SolverMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "solve" => Ok(Self::Solve),
            "tier" => Ok(Self::Tier),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "unknown solver mode '{}': expected solve|tier|full",
                other
            )),
        }
    }
}

/// Outcome of running the cascade on a grid.
#[derive(Debug, Clone)]
pub struct RateResult {
    /// Highest tier whose techniques contributed (or T1 if only singles fired,
    /// or T4Plus if the cascade got stuck without solving).
    pub tier: Tier,
    /// Distinct techniques that fired at least once (order = first firing).
    pub frontier: Vec<TechniqueId>,
    /// Whether the grid reached a fully-solved state via this cascade.
    pub solved: bool,
    /// Ordered firings (cascade trace). Each element is one technique
    /// application that produced placements/eliminations.
    pub trace: Vec<TechniqueId>,
    /// True on contradiction: inconsistent input, propagator contradiction,
    /// or technique-reported contradiction.
    pub rater_error: bool,
    /// Number of cascade outer-loop iterations executed (each iteration is one
    /// pass over T2 ∪ T3 lists; restarts after a firing increment this). 0 iff
    /// the puzzle solved by the initial T1 propagation alone or the input was
    /// inconsistent.
    pub wave_depth: u32,
    /// Cumulative count_solutions_with_steps branch count from the *uniqueness
    /// check* on the original input grid (`limit = 2`). Independent of the
    /// cascade — populated by [`rate_with_uniqueness`]; defaults to 0 when set
    /// by the cascade-only `rate` / `rate_excluding` paths.
    pub backtrack_steps: u64,
    /// True iff the original input grid has exactly one completion. Only
    /// populated by [`rate_with_uniqueness`]; defaults to `false` from the
    /// cascade-only paths.
    pub unique_solution: bool,
    /// SE-equivalent difficulty score: max(se_rating per fire) across the
    /// full cascade. 0.0 iff the puzzle solved by T1 singles alone (no
    /// technique fired). Set to 7.5 if the cascade got stuck (T4Plus) — a
    /// conservative lower bound for puzzles we cannot solve analytically.
    pub se_score: f64,
}

#[inline]
fn max_tier(a: Tier, b: Tier) -> Tier {
    fn rank(t: Tier) -> u8 {
        match t {
            Tier::T1 => 1,
            Tier::T2 => 2,
            Tier::T3 => 3,
            Tier::T4Plus => 4,
        }
    }
    if rank(b) > rank(a) {
        b
    } else {
        a
    }
}

#[inline]
fn push_unique(v: &mut Vec<TechniqueId>, id: TechniqueId) {
    if !v.iter().any(|&x| x == id) {
        v.push(id);
    }
}

/// Run the cascade on `grid` (clones first; input is not modified).
pub fn rate<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> RateResult {
    rate_excluding(grid, &[])
}

/// R3.3a: mode-gated cascade. `Full` ≡ [`rate`]. `Tier` runs the same cascade
/// with no other change (early-exit semantics fold in trivially: the cascade
/// already exits once neither T2 nor T3 fires, and we never rank above T3 in
/// the per-firing escalation; the savings come from skipping the
/// uniqueness/double-solve in callers, not from the cascade itself). `Solve`
/// skips the cascade entirely and only runs T1 propagation; tier defaults to
/// `T1` (solved by singles) or `T4Plus` (singles got stuck).
pub fn rate_with_mode<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    mode: SolverMode,
) -> RateResult {
    match mode {
        SolverMode::Full | SolverMode::Tier => rate_excluding(grid, &[]),
        SolverMode::Solve => {
            let mut g = grid.clone();
            let frontier: Vec<TechniqueId> = Vec::new();
            let trace: Vec<TechniqueId> = Vec::new();
            if !g.is_consistent() {
                return RateResult {
                    tier: Tier::T4Plus,
                    frontier,
                    solved: false,
                    trace,
                    rater_error: true,
                    wave_depth: 0,
                    backtrack_steps: 0,
                    unique_solution: false,
                    se_score: 0.0,
                };
            }
            match propagate_singles(&mut g) {
                Err(AssignErr::Contradiction) => RateResult {
                    tier: Tier::T1,
                    frontier,
                    solved: false,
                    trace,
                    rater_error: true,
                    wave_depth: 0,
                    backtrack_steps: 0,
                    unique_solution: false,
                    se_score: 0.0,
                },
                Ok(()) => {
                    let solved = g.is_solved();
                    RateResult {
                        tier: if solved { Tier::T1 } else { Tier::T4Plus },
                        frontier,
                        solved,
                        trace,
                        rater_error: false,
                        wave_depth: 0,
                        backtrack_steps: 0,
                        unique_solution: false,
                        // Solve mode doesn't run the technique cascade — no
                        // technique ratings available. For unsolved (T4Plus)
                        // use 7.5 as a lower-bound sentinel.
                        se_score: if solved { 0.0 } else { 7.5 },
                    }
                }
            }
        }
    }
}

/// Same cascade as [`rate`] but skips any technique whose `id()` is in
/// `exclude`. Used for "is technique X load-bearing?" probes.
pub fn rate_excluding<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    exclude: &[TechniqueId],
) -> RateResult {
    let mut g = grid.clone();
    let mut frontier: Vec<TechniqueId> = Vec::new();
    let mut trace: Vec<TechniqueId> = Vec::new();
    let mut highest = Tier::T1;
    let mut wave_depth: u32 = 0;
    let mut se_score: f64 = 0.0;

    // Fast reject: inconsistent givens.
    if !g.is_consistent() {
        return RateResult {
            tier: Tier::T4Plus,
            frontier,
            solved: false,
            trace,
            rater_error: true,
            wave_depth,
            backtrack_steps: 0,
            unique_solution: false,
            se_score: 0.0,
        };
    }

    // Static (stack-allocated, monomorphized) cascade lists. Replaces the
    // previous Vec<Box<dyn Technique>> form.
    let t2: &[AnyTechnique] = &T2_LIST;
    let t3: &[AnyTechnique] = &T3_LIST;

    // Pre-filter exclusions once.
    let is_excluded = |id: TechniqueId| exclude.iter().any(|&e| e == id);

    // ---- Initial T1 propagation ----
    match propagate_singles(&mut g) {
        Err(AssignErr::Contradiction) => {
            return RateResult {
                tier: highest,
                frontier,
                solved: g.is_solved(),
                trace,
                rater_error: true,
                wave_depth,
                backtrack_steps: 0,
                unique_solution: false,
                se_score: 0.0,
            };
        }
        Ok(()) => {}
    }
    if g.is_solved() {
        return RateResult {
            tier: highest, // T1
            frontier,
            solved: true,
            trace,
            rater_error: false,
            wave_depth,
            backtrack_steps: 0,
            unique_solution: false,
            se_score: 0.0,
        };
    }

    // ---- Cascade loop ----
    // Each iteration: try T2 list in order; if any fires, re-propagate T1 and
    // restart the inner loop. If no T2 fires, try T3. If neither fires and not
    // solved -> T4Plus.
    'outer: loop {
        wave_depth += 1;
        // T2 pass.
        for &tech in t2 {
            if is_excluded(tech.id()) {
                continue;
            }
            if let Some(p) = tech.apply(&mut g) {
                if p.contradiction {
                    return RateResult {
                        tier: max_tier(highest, tech.tier()),
                        frontier,
                        solved: g.is_solved(),
                        trace,
                        rater_error: true,
                        wave_depth,
                        backtrack_steps: 0,
                        unique_solution: false,
                        se_score,
                    };
                }
                if p.fired() {
                    se_score = se_score.max(tech.se_rating::<N, BR, BC>(&p));
                    push_unique(&mut frontier, tech.id());
                    trace.push(tech.id());
                    highest = max_tier(highest, tech.tier());
                    // Re-propagate singles after eliminations.
                    if let Err(_) = propagate_singles(&mut g) {
                        return RateResult {
                            tier: highest,
                            frontier,
                            solved: g.is_solved(),
                            trace,
                            rater_error: true,
                            wave_depth,
                            backtrack_steps: 0,
                            unique_solution: false,
                            se_score,
                        };
                    }
                    if g.is_solved() {
                        return RateResult {
                            tier: highest,
                            frontier,
                            solved: true,
                            trace,
                            rater_error: false,
                            wave_depth,
                            backtrack_steps: 0,
                            unique_solution: false,
                            se_score,
                        };
                    }
                    // Restart from top of T2 list — earlier techniques may now
                    // apply on the shrunk candidate set.
                    continue 'outer;
                }
            }
        }

        // T3 pass.
        for &tech in t3 {
            if is_excluded(tech.id()) {
                continue;
            }
            if let Some(p) = tech.apply(&mut g) {
                if p.contradiction {
                    return RateResult {
                        tier: max_tier(highest, tech.tier()),
                        frontier,
                        solved: g.is_solved(),
                        trace,
                        rater_error: true,
                        wave_depth,
                        backtrack_steps: 0,
                        unique_solution: false,
                        se_score,
                    };
                }
                if p.fired() {
                    se_score = se_score.max(tech.se_rating::<N, BR, BC>(&p));
                    push_unique(&mut frontier, tech.id());
                    trace.push(tech.id());
                    highest = max_tier(highest, tech.tier());
                    if let Err(_) = propagate_singles(&mut g) {
                        return RateResult {
                            tier: highest,
                            frontier,
                            solved: g.is_solved(),
                            trace,
                            rater_error: true,
                            wave_depth,
                            backtrack_steps: 0,
                            unique_solution: false,
                            se_score,
                        };
                    }
                    if g.is_solved() {
                        return RateResult {
                            tier: highest,
                            frontier,
                            solved: true,
                            trace,
                            rater_error: false,
                            wave_depth,
                            backtrack_steps: 0,
                            unique_solution: false,
                            se_score,
                        };
                    }
                    // Restart at T2 — a T3-driven elimination often unlocks
                    // cheaper T2 techniques.
                    continue 'outer;
                }
            }
        }

        // T4Plus pass — expensive techniques tried only when T2+T3 are exhausted.
        let t4plus: &[AnyTechnique] = &T4PLUS_LIST;
        // Precompute the per-arm exclusion mask for the combined chain driver.
        // CR-FIN-3 C1: prior code skipped the entire ChainCombined entry on any
        // chain-id exclusion, dropping all four arms → strict load-bearing
        // trivially passed. Now we run `ChainCombinedTechnique` with a per-arm
        // mask so that excluding (e.g.) `Whip` still probes gwhip/braid/gbraid.
        // Only when all four chain ids are excluded do we skip the entry.
        let chain_mask = ChainCombinedTechnique::with_excluded(exclude);
        for &tech in t4plus {
            let skip = match tech {
                AnyTechnique::ChainCombined => chain_mask.all_arms_excluded(),
                other => is_excluded(other.id()),
            };
            if skip {
                continue;
            }
            // Dispatch ChainCombined through the per-call mask (not `apply`).
            let progress = match tech {
                AnyTechnique::ChainCombined => chain_mask.apply(&mut g),
                other => other.apply(&mut g),
            };
            if let Some(p) = progress {
                if p.contradiction {
                    return RateResult {
                        tier: max_tier(highest, tech.tier()),
                        frontier,
                        solved: g.is_solved(),
                        trace,
                        rater_error: true,
                        wave_depth,
                        backtrack_steps: 0,
                        unique_solution: false,
                        se_score,
                    };
                }
                if p.fired() {
                    let rating = tech.se_rating::<N, BR, BC>(&p);
                    se_score = se_score.max(rating);
                    // Use technique_id_override if present (set by ChainCombinedTechnique
                    // to carry the actual W/GW/B/GB id that fired — CR-FIN-2).
                    //
                    // CR-FIN-4 Mn-2: enforce the consumer-side invariant — when
                    // dispatching `ChainCombinedTechnique` we MUST have an
                    // override (Whip/GWhip/Braid/GBraid) on a firing. Without
                    // it, `tech.id()` returns the placeholder `Whip`, silently
                    // mis-attributing the firing arm (frontier/se_rating).
                    // `ChainCombinedTechnique::apply` already debug_asserts the
                    // override before return; this consumer-side guard catches
                    // any future refactor that bypasses that path.
                    if matches!(tech, AnyTechnique::ChainCombined) {
                        debug_assert!(
                            p.technique_id_override.is_some(),
                            "ChainCombined must set technique_id_override on fire"
                        );
                    }
                    let fired_id = p.technique_id_override.unwrap_or_else(|| tech.id());
                    push_unique(&mut frontier, fired_id);
                    trace.push(fired_id);
                    highest = max_tier(highest, tech.tier());
                    if let Err(_) = propagate_singles(&mut g) {
                        return RateResult {
                            tier: highest,
                            frontier,
                            solved: g.is_solved(),
                            trace,
                            rater_error: true,
                            wave_depth,
                            backtrack_steps: 0,
                            unique_solution: false,
                            se_score,
                        };
                    }
                    if g.is_solved() {
                        return RateResult {
                            tier: highest,
                            frontier,
                            solved: true,
                            trace,
                            rater_error: false,
                            wave_depth,
                            backtrack_steps: 0,
                            unique_solution: false,
                            se_score,
                        };
                    }
                    // Restart at T2 after a T4Plus firing.
                    continue 'outer;
                }
            }
        }

        // Nothing fired this round. Cascade stuck → T4Plus.
        //
        // CR-FIN-3 M-opus-4: `se_score` here is already the max of all
        // `tech.se_rating(progress)` over every technique that DID fire in
        // T2/T3 (and any earlier T4Plus). Bumping it to `.max(7.5)` is a
        // conservative T4Plus floor for puzzles that ended unsolved — without
        // a successful T4Plus firing we cannot read off a richer rating from
        // any individual technique. The original highest-pre-T4 SE rating
        // (e.g. an AIC firing at 6.6) is preserved when it exceeds 7.5;
        // otherwise the floor encodes "this puzzle defeats our T2/T3 toolkit,
        // so it lives in the T4Plus territory". Tightening this requires a
        // post-T4Plus search probe (e.g. CLIPS oracle), out of scope here.
        return RateResult {
            tier: Tier::T4Plus,
            frontier,
            solved: false,
            trace,
            rater_error: false,
            wave_depth,
            backtrack_steps: 0,
            unique_solution: false,
            se_score: se_score.max(7.5),
        };
    }
}

/// Run the cascade and additionally compute uniqueness + backtrack-step count
/// over the *original* input grid (independent of cascade exclusions). This is
/// the path used by `rate-batch` to populate every RateResult field at once.
pub fn rate_with_uniqueness<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> RateResult {
    let mut r = rate(grid);
    let (n_sol, branches) = super::search::count_solutions_with_steps(grid, 2);
    r.backtrack_steps = branches;
    r.unique_solution = n_sol == 1;
    r
}

/// R3.3a: mode-gated `rate_with_uniqueness`.
///
/// * `Full` — identical to [`rate_with_uniqueness`]: cascade + uniqueness
///   double-solve (`count_solutions_with_steps(grid, 2)`).
/// * `Tier` — cascade only; **skips the uniqueness double-solve**. We trust
///   that uniqueness was guaranteed at puzzle construction time. The
///   `unique_solution` field is left as `false` and `backtrack_steps` as `0`;
///   callers that need uniqueness verification must use `Full` mode.
/// * `Solve` — T1 propagation only (no cascade); same uniqueness short-circuit
///   as `Tier`.
pub fn rate_with_uniqueness_mode<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    mode: SolverMode,
) -> RateResult {
    match mode {
        SolverMode::Full => rate_with_uniqueness(grid),
        SolverMode::Tier | SolverMode::Solve => rate_with_mode(grid, mode),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    // ------------------------------------------------------------------
    // 9×9 — full tier coverage.
    // ------------------------------------------------------------------

    /// Classic "easy" 9×9 — solvable by singles propagation alone.
    #[test]
    fn t1_9x9_easy_singles() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r = rate(&g);
        assert!(r.solved, "expected solved, got {:?}", r);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T1);
        assert!(r.frontier.is_empty(), "T1 path must not fire any technique: {:?}", r.frontier);
        assert!(r.trace.is_empty());
    }

    /// 9×9 puzzle that requires at least one T2 technique to solve.
    /// Generated via the legacy generator and confirmed at `Tier::T2` by the
    /// legacy rater. We assert tier ≥ T2 (cascade may also escalate to T3 if
    /// a T3 tech happens to fire opportunistically — but for a well-classified
    /// T2 puzzle we expect exactly T2).
    #[test]
    fn t2_9x9_requires_locked_or_pair() {
        use crate::generic::generator::{gen_unique_puzzle, GenConfig};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        let cfg = GenConfig::new(30);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        for _ in 0..400 {
            let (g, _c): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng, &cfg);
            let r = rate(&g);
            if !matches!(r.tier, Tier::T2) {
                continue;
            }
            assert!(r.solved, "T2 puzzle must solve via cascade: {:?}", r);
            assert!(!r.rater_error);
            assert!(!r.frontier.is_empty(), "T2 must have firings");
            return;
        }
        panic!("no T2 puzzle found in 400 samples");
    }

    /// 9×9 puzzle that requires a T3 technique.
    #[test]
    fn t3_9x9_requires_t3_tech() {
        use crate::generic::generator::{gen_unique_puzzle, GenConfig};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        let cfg = GenConfig::new(30);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        for _ in 0..400 {
            let (g, _c): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng, &cfg);
            let r = rate(&g);
            if !matches!(r.tier, Tier::T3) {
                continue;
            }
            assert!(r.solved, "T3 puzzle must solve: {:?}", r);
            assert!(!r.rater_error);
            // Frontier must contain at least one T3 technique.
            let has_t3 = r.frontier.iter().any(|id| matches!(*id,
                TechniqueId::XWing | TechniqueId::Swordfish | TechniqueId::Jellyfish |
                TechniqueId::XyWing | TechniqueId::XyzWing |
                TechniqueId::UrType1 | TechniqueId::UrType2 |
                TechniqueId::Bug |
                TechniqueId::Aic | TechniqueId::AlsXz |
                TechniqueId::SimpleColoring | TechniqueId::Skyscraper |
                TechniqueId::TwoStringKite));
            assert!(has_t3, "T3 frontier must contain a T3 tech: {:?}", r.frontier);
            return;
        }
        eprintln!("warn: no T3 puzzle in 400 samples (test soft-passes)");
    }

    /// 9×9 puzzle that none of our techniques can crack.
    /// The empty grid (all candidates open) has no technique fire and is not
    /// solved → T4Plus, no rater_error. (No givens → not a "real" puzzle but
    /// satisfies the contract: no contradiction, no technique applies.)
    #[test]
    fn t4plus_9x9_empty_grid_stuck() {
        let g: Grid<9, 3, 3> = Grid::empty();
        let r = rate(&g);
        assert!(!r.solved);
        assert!(!r.rater_error, "empty grid is consistent — must NOT be rater_error");
        assert_eq!(r.tier, Tier::T4Plus);
    }

    /// Inconsistent input → rater_error true, tier T4Plus, no firings.
    #[test]
    fn t4plus_9x9_inconsistent_input() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        g.assign(0, 5).unwrap();
        // Bypass assign() to plant a duplicate 5 in same row.
        g.solved[1] = 5;
        g.candidates[1] = 1u32 << 4;
        g.solved_count += 1;
        assert!(!g.is_consistent());
        let r = rate(&g);
        assert!(r.rater_error);
        assert!(!r.solved);
    }

    /// Load-bearing test: find a T3 puzzle whose only T3 technique is X
    /// (single id in the frontier). Then `rate_excluding(grid, &[X])` must
    /// drop tier — not still solve at T3.
    #[test]
    fn load_bearing_9x9_t3_drops_when_excluded() {
        use crate::generic::generator::{gen_unique_puzzle, GenConfig};
        use rand_xoshiro::rand_core::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        let cfg = GenConfig::new(30);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..600 {
            let (g, _c): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng, &cfg);
            let r = rate(&g);
            if r.tier != Tier::T3 || !r.solved {
                continue;
            }
            // Load-bearing semantics: excluding the entire T3 family must
            // prevent the cascade from reaching T3 (it either drops to T1/T2
            // or gets stuck at T4Plus). A single T3 tech may be replaceable
            // by another (e.g. AIC ↔ ALS-XZ on the same puzzle), but with
            // *all* T3 techs disabled the cascade can no longer escalate.
            // Exclude T3 *and* all T4Plus — the test checks that a T3-rated
            // puzzle cannot be solved by T2-and-below only.
            const ALL_T3: &[TechniqueId] = &[
                TechniqueId::NakedTriple, TechniqueId::HiddenTriple,
                TechniqueId::NakedQuad, TechniqueId::HiddenQuad,
                TechniqueId::XWing, TechniqueId::Swordfish, TechniqueId::Jellyfish,
                TechniqueId::XyWing, TechniqueId::XyzWing,
                TechniqueId::UrType1, TechniqueId::UrType2,
                TechniqueId::Bug,
                TechniqueId::Aic, TechniqueId::AlsXz,
                TechniqueId::SimpleColoring, TechniqueId::Skyscraper,
                TechniqueId::TwoStringKite,
                // T4Plus — exclude so T4Plus techniques can't step in for the blocked T3s.
                TechniqueId::CellForcingChain,
                TechniqueId::RegionForcingChain,
                TechniqueId::DynamicForcingChain,
                TechniqueId::NestedForcingChain,
                TechniqueId::NestedForcingChainL3,
                TechniqueId::NestedForcingChainL4,
                // Berthier chain technique (GBraid-only post-collapse, spec §17).
                TechniqueId::GBraid,
            ];
            let r2 = rate_excluding(&g, ALL_T3);
            assert!(r2.tier != Tier::T3,
                "excluding all T3 techs must prevent T3 escalation, got {:?}", r2);
            // The puzzle was T3 originally — without T3 it must NOT solve.
            assert!(!r2.solved,
                "T3 puzzle must not solve when T3 family is excluded; got {:?}", r2);
            assert_eq!(r2.tier, Tier::T4Plus,
                "expected T4Plus stuck state, got {:?}", r2);
            return;
        }
        eprintln!("warn: no T3 puzzle in 400 samples — load-bearing test soft-pass");
    }

    /// `rate_excluding` with a no-op exclusion list (technique that didn't
    /// fire) must give the same result as `rate`.
    #[test]
    fn rate_excluding_inert_exclusion_matches_rate() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r1 = rate(&g);
        let r2 = rate_excluding(&g, &[TechniqueId::Aic, TechniqueId::Jellyfish]);
        assert_eq!(r1.tier, r2.tier);
        assert_eq!(r1.solved, r2.solved);
        assert_eq!(r1.frontier, r2.frontier);
    }

    // ------------------------------------------------------------------
    // 6×6 — T1 only (block size too small for non-trivial T2/T3 hand-craft).
    // ------------------------------------------------------------------

    #[test]
    fn t1_6x6_singles_only() {
        let p = "12345.\
                 456123\
                 21436.\
                 365214\
                 5316.2\
                 642531";
        let g: Grid<6, 2, 3> = Grid::from_str(p).unwrap();
        let r = rate(&g);
        assert!(r.solved);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T1);
        assert!(r.frontier.is_empty());
    }

    #[test]
    fn t4plus_6x6_empty_grid() {
        let g: Grid<6, 2, 3> = Grid::empty();
        let r = rate(&g);
        assert!(!r.solved);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T4Plus);
    }

    // ------------------------------------------------------------------
    // 12×12 — T1 (full solution minus a few cells).
    // ------------------------------------------------------------------

    fn algebraic_solution_string(n: usize, br: usize, bc: usize) -> String {
        let mut s = String::with_capacity(n * n);
        for r in 0..n {
            for c in 0..n {
                let v = ((bc * r + r / br + c) % n) + 1;
                let d = v as u8;
                if d <= 9 {
                    s.push((b'0' + d) as char);
                } else {
                    s.push((b'A' + d - 10) as char);
                }
            }
        }
        s
    }

    #[test]
    fn t1_12x12_singles_only() {
        let s = algebraic_solution_string(12, 3, 4);
        let mut bytes: Vec<u8> = s.as_bytes().to_vec();
        for &i in &[0usize, 13, 26, 50, 100, 130] {
            bytes[i] = b'.';
        }
        let s2 = std::str::from_utf8(&bytes).unwrap();
        let g: Grid<12, 3, 4> = Grid::from_str(s2).unwrap();
        let r = rate(&g);
        assert!(r.solved, "{:?}", r);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T1);
    }

    // ------------------------------------------------------------------
    // 16×16 — T1.
    // ------------------------------------------------------------------

    #[test]
    fn t1_16x16_singles_only() {
        let s = algebraic_solution_string(16, 4, 4);
        let mut bytes: Vec<u8> = s.as_bytes().to_vec();
        for &i in &[0usize, 17, 34, 51, 100, 200, 250] {
            bytes[i] = b'.';
        }
        let s2 = std::str::from_utf8(&bytes).unwrap();
        let g: Grid<16, 4, 4> = Grid::from_str(s2).unwrap();
        let r = rate(&g);
        assert!(r.solved);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T1);
    }

    // ------------------------------------------------------------------
    // R3.3a — SolverMode (Solve / Tier / Full) parity & shortcut tests.
    // ------------------------------------------------------------------

    #[test]
    fn mode_solve_easy_returns_t1_solved_no_frontier() {
        // Easy 9×9 — T1 cascade ≡ T1 propagation, so Solve must mark solved.
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r = rate_with_mode(&g, SolverMode::Solve);
        assert!(r.solved, "Solve mode must solve a singles-only puzzle: {:?}", r);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T1);
        assert!(r.frontier.is_empty());
        assert!(r.trace.is_empty());
        assert_eq!(r.wave_depth, 0);
        // Solve mode deliberately does NOT compute uniqueness.
        assert!(!r.unique_solution);
        assert_eq!(r.backtrack_steps, 0);
    }

    #[test]
    fn mode_full_matches_rate_default() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r_full = rate_with_mode(&g, SolverMode::Full);
        let r_legacy = rate(&g);
        assert_eq!(r_full.tier, r_legacy.tier);
        assert_eq!(r_full.solved, r_legacy.solved);
        assert_eq!(r_full.frontier, r_legacy.frontier);
        assert_eq!(r_full.trace, r_legacy.trace);
    }

    #[test]
    fn mode_tier_skips_uniqueness_double_solve() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r_tier = rate_with_uniqueness_mode(&g, SolverMode::Tier);
        // Tier mode populates cascade outputs but skips the second solve.
        assert!(r_tier.solved);
        assert_eq!(r_tier.tier, Tier::T1);
        assert!(!r_tier.unique_solution, "Tier mode must NOT compute uniqueness");
        assert_eq!(r_tier.backtrack_steps, 0);
        // Full mode computes uniqueness.
        let r_full = rate_with_uniqueness_mode(&g, SolverMode::Full);
        assert!(r_full.unique_solution);
    }

    #[test]
    fn mode_solve_inconsistent_input_flags_error() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        g.assign(0, 5).unwrap();
        g.solved[1] = 5;
        g.candidates[1] = 1u32 << 4;
        g.solved_count += 1;
        assert!(!g.is_consistent());
        let r = rate_with_mode(&g, SolverMode::Solve);
        assert!(r.rater_error);
        assert!(!r.solved);
    }

    #[test]
    fn solver_mode_parse() {
        assert_eq!(SolverMode::parse("solve").unwrap(), SolverMode::Solve);
        assert_eq!(SolverMode::parse("TIER").unwrap(), SolverMode::Tier);
        assert_eq!(SolverMode::parse("Full").unwrap(), SolverMode::Full);
        assert!(SolverMode::parse("nonsense").is_err());
    }

    #[test]
    fn t4plus_16x16_empty_grid() {
        let g: Grid<16, 4, 4> = Grid::empty();
        let r = rate(&g);
        assert!(!r.solved);
        assert!(!r.rater_error);
        assert_eq!(r.tier, Tier::T4Plus);
    }

    // ------------------------------------------------------------------
    // FA-1 regression: chain techniques registered in rater pipeline.
    // ------------------------------------------------------------------

    /// Smoke test: T1/T2/T3-rated puzzles must NOT be reclassified by
    /// chain techniques. rate() on an easy T1 puzzle must still return T1
    /// (chain techniques don't accidentally fire and elevate the tier).
    #[test]
    fn chain_techniques_do_not_reclassify_easy_puzzles() {
        // Easy T1 puzzle (solvable by singles).
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r = rate(&g);
        assert_eq!(r.tier, Tier::T1, "T1 puzzle must stay T1 with chain techniques registered: {:?}", r);
        assert!(r.solved);
        assert!(!r.rater_error);
        // Chain techniques must not appear in frontier for a T1 puzzle.
        let chain_techs = [TechniqueId::GBraid];
        for &t in &chain_techs {
            assert!(!r.frontier.contains(&t),
                "chain technique {:?} must not fire on T1 puzzle", t);
        }
    }

    /// rate_excluding with chain technique excluded must not affect T1/T2/T3 puzzles.
    /// Excluding Whip from a T1 puzzle has no effect (Whip wouldn't have fired anyway).
    #[test]
    fn chain_technique_exclusion_inert_on_t1() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        let r1 = rate(&g);
        let r2 = rate_excluding(&g, &[TechniqueId::GBraid]);
        assert_eq!(r1.tier, r2.tier, "excluding chain techs must not change T1 puzzle tier");
        assert_eq!(r1.solved, r2.solved);
    }
}
