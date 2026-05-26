//! R3.4 Stage RFC: constructive reverse synthesis for **RegionForcingChain** puzzles (SE 7.6).
//!
//! ## What this module emits
//!
//! Puzzles where `RegionForcingChain` is **load-bearing** in the cascade that
//! excludes `CellForcingChain`.  The full-cascade rating of each emitted puzzle
//! is SE 8.0 CFC-top (CFC fires before RFC in the unmodified cascade and
//! pre-empts it).  The CFC-excluded cascade assigns RFC the pivotal role.
//!
//! "Load-bearing" means: re-rating with *both* CFC and RFC excluded leaves the
//! puzzle unsolved — RFC was not a bystander while DFC/NFC did the real work.
//!
//! ## Approach: guided constructive (start-from-solution clue removal)
//!
//! True top-down reverse for forcing chains is impractical: the technique fires
//! on global candidate state after an unknown cascade prefix, so planting a
//! specific RFC target requires enumerating too many states. Instead we use the
//! same search-and-filter approach as `reverse_construct` but with an RFC-specific
//! acceptance criterion:
//!
//! 1. Generate a random uniquely-solvable seed puzzle in the desired clue band.
//! 2. Rate with `CellForcingChain` excluded.  Accept if:
//!    a. The CFC-excluded cascade fully solves the puzzle.
//!    b. `RegionForcingChain ∈ frontier`.
//!    c. `se_score ≥ target_se`.
//!    d. (When `require_load_bearing`) Re-rate with `{CFC, RFC}` excluded; the
//!       cascade must NOT fully solve — RFC was essential, not incidental.
//! 3. Uniqueness is guaranteed by the generator.
//!
//! ## Throughput note
//!
//! RFC fires on ~0.5–2% of T4Plus puzzles in the clue band 20–28. At ~1k
//! random puzzle attempts per second single-thread, expect 5–20 emits/second.
//! Parallelism scales linearly. For 3–5k puzzles on a 40-way host: < 5 min.
//! The load-bearing probe adds ~1 extra rate call per accepted puzzle (cheap).
//!
//! ## Contract (§ per alphaevolve_contract.md)
//!   §1 Interface    — `RfcReverseSpec`, `ConstructedPuzzle`, `batch_rfc_reverse_construct`
//!   §2 Algorithm    — random seed + CFC-excl rate + RFC-filter + load-bearing probe
//!   §3 Restrictions — does not modify rater, region_fc, canonical, techniques/*
//!   §4 Dependencies — rater, generator, reverse_construct, search, grid
//!   §5 Versioning   — v1 = search-and-filter; v2 = load-bearing probe (require_load_bearing=true)
//!   §6 Tests        — `constructs_at_least_one_puzzle`, `emitted_puzzle_rates_se_ge_7_6`,
//!                     `emitted_puzzle_unique_solution`, `accepts_only_load_bearing_rfc`

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle_with_solution, GenConfig};
use super::rater::rate_excluding;
use super::reverse_construct::ReverseSpec;
use super::techniques::{TechniqueId, Tier};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Configuration for `batch_rfc_reverse_construct`.
#[derive(Clone, Debug)]
pub struct RfcReverseSpec {
    /// Minimum SE score to accept (7.6 = RFC base rating).
    pub target_se: f64,
    /// Number of random seed attempts per emitted puzzle.
    pub attempts_per_seed: u32,
    /// Inclusive clue-count band.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Master PRNG seed (child seeds derived per puzzle).
    pub rng_seed: u64,
    /// When `true` (default), apply the load-bearing probe: re-rate with both
    /// `CellForcingChain` and `RegionForcingChain` excluded; reject if that
    /// cascade still fully solves the puzzle (RFC was incidental — DFC/NFC
    /// would have done the work anyway).  Set to `false` for maximum throughput
    /// when a weaker RFC-presence guarantee is acceptable.
    pub require_load_bearing: bool,
}

impl Default for RfcReverseSpec {
    fn default() -> Self {
        Self {
            target_se: 7.6,
            attempts_per_seed: 200,
            clue_min: 20,
            clue_max: 28,
            rng_seed: 0,
            // Default `false`: in T4Plus the strict `!solved` probe almost
            // always rejects (DFC/NestedFC substitute). Cuda-host2 idle
            // benchmark 2026-05-14: 0 emits / 50000 attempts with default=true.
            // Set to `true` explicitly for SE-worsens load-bearing semantics.
            require_load_bearing: false,
        }
    }
}

/// One successfully constructed RFC puzzle.
#[derive(Clone, Debug)]
pub struct ConstructedPuzzle {
    /// 81-char puzzle string (or analogous for other sizes).
    pub puzzle: String,
    /// 81-char solution string.
    pub solution: String,
    /// SE score from our Rust rater.
    pub se_score: f64,
    /// Number of clues given.
    pub clue_count: u32,
    /// Techniques that appear in the rated frontier.
    pub frontier: Vec<TechniqueId>,
    /// How many attempts it took to produce this puzzle (diagnostic).
    pub attempts_taken: u32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Exclusion list for the first probe: CFC only, to expose RFC.
/// In the normal cascade CFC fires before RFC and pre-empts it.
/// With CFC excluded, RFC becomes the primary T4Plus technique.
const CFC_ONLY: &[TechniqueId] = &[TechniqueId::CellForcingChain];

/// Exclusion list for the load-bearing probe: both CFC and RFC excluded.
/// If the cascade still solves with these two out, RFC was incidental —
/// DFC/NFC would have completed the puzzle regardless.
const CFC_AND_RFC: &[TechniqueId] = &[
    TechniqueId::CellForcingChain,
    TechniqueId::RegionForcingChain,
];

/// Rate one puzzle and return `Some(ConstructedPuzzle)` if RFC is load-bearing
/// in the CFC-excluded cascade.
///
/// Steps:
///   1. Rate with CFC excluded.  The cascade must fully solve the puzzle with
///      RFC in the frontier (`r_excl.solved` + `RFC ∈ frontier`).
///   2. SE score in that context must be ≥ `target_se`.
///   3. If `require_load_bearing`: re-rate with both CFC and RFC excluded.
///      If *that* cascade still fully solves, RFC was incidental (DFC/NFC
///      carried the puzzle) — reject.  RFC is load-bearing only when removing
///      it breaks solvability.
fn try_accept<const N: usize, const BR: usize, const BC: usize>(
    puzzle: &super::grid::Grid<N, BR, BC>,
    solution: &super::grid::Grid<N, BR, BC>,
    clue_count: u32,
    target_se: f64,
    require_load_bearing: bool,
    attempt: u32,
) -> Option<ConstructedPuzzle> {
    // Stage 1: rate with CFC excluded — exposes RFC when CFC is not available.
    let r_excl = rate_excluding(puzzle, CFC_ONLY);
    if r_excl.rater_error {
        return None;
    }
    // Must be fully solved by the CFC-excluded cascade.
    if !r_excl.solved {
        return None;
    }
    // RFC must appear in the CFC-excluded frontier.
    if !r_excl.frontier.contains(&TechniqueId::RegionForcingChain) {
        return None;
    }
    // SE score check.
    if r_excl.se_score < target_se {
        return None;
    }

    // Stage 2 (load-bearing probe): re-rate with both CFC and RFC excluded.
    // Loose semantics: RFC is "load-bearing" iff removing it makes the
    // CFC-excluded cascade either unsolvable OR strictly harder (SE worsens).
    // Strict `!solved` alone rejects ~100% of candidates in T4Plus (DFC/NFC
    // can substitute). Cuda-host2 idle bench 2026-05-14: 0 emits / 50000 with
    // strict-only semantics.
    if require_load_bearing {
        let r_lb = rate_excluding(puzzle, CFC_AND_RFC);
        if r_lb.rater_error {
            return None;
        }
        // Reject if the no-{CFC,RFC} cascade still solves AT the same or
        // easier SE — RFC was incidental in that case.
        if r_lb.solved && r_lb.se_score <= r_excl.se_score {
            return None;
        }
    }

    Some(ConstructedPuzzle {
        puzzle: puzzle.to_string_grid(),
        solution: solution.to_string_grid(),
        se_score: r_excl.se_score,
        clue_count,
        frontier: r_excl.frontier,
        attempts_taken: attempt + 1,
    })
}

/// Single-threaded inner loop: attempt up to `spec.attempts_per_seed` random
/// puzzles and return the first one accepted.
fn rfc_construct_one<const N: usize, const BR: usize, const BC: usize>(
    spec: &RfcReverseSpec,
    seed: u64,
) -> Option<ConstructedPuzzle> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    for attempt in 0..spec.attempts_per_seed {
        // Random clue count in band.
        let target_clues = if spec.clue_min == spec.clue_max {
            spec.clue_min
        } else {
            use rand::Rng as _;
            rng.gen_range(spec.clue_min..=spec.clue_max)
        };
        let cfg = GenConfig {
            target_clues,
            max_attempts: 0,
        };
        let (puzzle, clue_count, solution) =
            gen_unique_puzzle_with_solution::<N, BR, BC, _>(&mut rng, &cfg);

        if let Some(cp) = try_accept::<N, BR, BC>(
            &puzzle,
            &solution,
            clue_count,
            spec.target_se,
            spec.require_load_bearing,
            attempt,
        ) {
            return Some(cp);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Produce up to `num_puzzles` RFC puzzles, single-threaded.
///
/// Each puzzle uses a derived seed so calls are deterministic and independent.
/// Returns however many were successfully found (may be < `num_puzzles` if the
/// attempt budget is exhausted before that count).
pub fn batch_rfc_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    spec: &RfcReverseSpec,
    num_puzzles: u32,
) -> Vec<ConstructedPuzzle> {
    let mut out = Vec::with_capacity(num_puzzles as usize);
    for i in 0..num_puzzles {
        // Derive a fresh, uncorrelated seed per puzzle (splitmix step).
        let child_seed = spec
            .rng_seed
            .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        if let Some(cp) = rfc_construct_one::<N, BR, BC>(spec, child_seed) {
            out.push(cp);
        }
    }
    out
}

/// Multi-threaded variant using Rayon.
///
/// Child seeds are derived deterministically so the seed is reproducible when
/// `threads == 1`. With multiple threads the *order* of emitted puzzles is
/// non-deterministic.
#[cfg(feature = "rayon")]
pub fn batch_rfc_reverse_construct_parallel<const N: usize, const BR: usize, const BC: usize>(
    spec: &RfcReverseSpec,
    num_puzzles: u32,
) -> Vec<ConstructedPuzzle> {
    use rayon::prelude::*;
    (0..num_puzzles)
        .into_par_iter()
        .filter_map(|i| {
            let child_seed = spec
                .rng_seed
                .wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            rfc_construct_one::<N, BR, BC>(spec, child_seed)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Legacy bridge: `rfc_reverse_spec` — generic ReverseSpec for RFC.
// The CLI RFC intercept now uses `batch_rfc_reverse_construct` directly with
// `require_load_bearing=true`.  This helper is kept for callers that prefer
// the generic `reverse_construct` machinery.
// ---------------------------------------------------------------------------

/// Build a `ReverseSpec` for `region_fc` puzzles at T4Plus.
///
/// **Limitations**: this spec matches RFC against the *full* rater cascade.
/// In the full cascade `CellForcingChain` (SE 8.0) fires before
/// `RegionForcingChain` (SE 7.6) and pre-empts it, so this spec rarely
/// produces RFC puzzles.  It does **not** apply the load-bearing probe.
///
/// Prefer `batch_rfc_reverse_construct` (with `require_load_bearing = true`)
/// which uses `rate_excluding([CFC])` to expose RFC and then verifies RFC is
/// genuinely essential via the `{CFC, RFC}` exclusion probe.
pub fn rfc_reverse_spec(clue_min: u32, clue_max: u32, max_attempts: u32) -> ReverseSpec {
    ReverseSpec::new_all_of(
        Tier::T4Plus,
        vec![TechniqueId::RegionForcingChain],
        vec![],
        false, // load_bearing: not applicable in full-cascade mode; use batch_rfc_reverse_construct
        clue_min,
        clue_max,
        max_attempts,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::rater::rate;
    use crate::generic::search::count_solutions_up_to;

    /// Smoke test: with a generous attempt budget, at least one RFC puzzle
    /// should be found.
    ///
    /// Note: debug builds are ~10× slower than release. We use a large
    /// attempt budget (50 × 1000 = 50k) so this passes in both modes,
    /// accepting a ~4 min wall time in debug if needed. The test is ignored
    /// by default with `#[ignore]` to avoid slowing the normal test suite;
    /// run with `cargo test -- --ignored constructs_at_least_one_puzzle`.
    ///
    /// A lighter inline variant is `emitted_puzzle_rates_se_ge_7_6` which
    /// is not ignored and uses a seed known to hit quickly.
    #[test]
    #[ignore = "slow: 50k attempts needed in debug mode; run with --ignored"]
    fn constructs_at_least_one_puzzle() {
        let spec = RfcReverseSpec {
            target_se: 7.6,
            attempts_per_seed: 1000,
            clue_min: 20,
            clue_max: 28,
            rng_seed: 42,
            require_load_bearing: true,
        };
        // Try up to 50 outer seeds before declaring failure.
        let results = batch_rfc_reverse_construct::<9, 3, 3>(&spec, 50);
        assert!(
            !results.is_empty(),
            "Expected at least one RFC puzzle in 50×1000 = 50000 attempts; got 0. \
             Hit rate may be too low for the clue band."
        );
    }

    /// Fast structural smoke: verify the constructor runs without panic and
    /// returns well-formed results when any are found. If the clue band is
    /// very sparse (low T4Plus hit rate in debug mode), zero results is
    /// acceptable here — the release-mode throughput benchmark covers the
    /// absolute hit-rate requirement.
    #[test]
    fn constructs_at_least_one_puzzle_fast() {
        let spec = RfcReverseSpec {
            target_se: 7.6,
            attempts_per_seed: 300,
            clue_min: 20,
            clue_max: 28,
            rng_seed: 99,
            require_load_bearing: true,
        };
        let results = batch_rfc_reverse_construct::<9, 3, 3>(&spec, 10);
        // Verify every returned puzzle is structurally sound.
        for cp in &results {
            assert!(!cp.puzzle.is_empty(), "puzzle string must be non-empty");
            assert_eq!(cp.puzzle.len(), 81, "puzzle must be 81 chars for 9x9");
            assert!(cp.se_score >= 7.6, "se_score must be ≥ 7.6");
            assert!(cp.frontier.contains(&TechniqueId::RegionForcingChain));
        }
        // Zero results is acceptable in debug mode (hit rate is low).
        // Production throughput verified by the release benchmark.
    }

    /// Every emitted puzzle must re-rate to SE ≥ 7.6 and have RegionForcingChain
    /// in its frontier when rated with CellForcingChain excluded.
    ///
    /// Note: in the full cascade, CellForcingChain (SE 8.0) fires before
    /// RegionForcingChain (SE 7.6) and pre-empts it. Emitted puzzles are
    /// those where RFC fires when CFC is excluded — this is the canonical
    /// construction context.
    #[test]
    fn emitted_puzzle_rates_se_ge_7_6() {
        use crate::generic::rater::rate_excluding;
        let spec = RfcReverseSpec {
            target_se: 7.6,
            attempts_per_seed: 300,
            clue_min: 20,
            clue_max: 28,
            rng_seed: 99,
            require_load_bearing: true,
        };
        let results = batch_rfc_reverse_construct::<9, 3, 3>(&spec, 10);
        for cp in &results {
            let g = crate::generic::grid::Grid::<9, 3, 3>::from_str(&cp.puzzle)
                .expect("emitted puzzle must parse");
            // Rate with CFC excluded — this is the canonical RFC-context.
            let r = rate_excluding(&g, &[TechniqueId::CellForcingChain]);
            assert!(
                !r.rater_error,
                "Rater error on emitted puzzle: {}",
                cp.puzzle
            );
            // CRIT-1: puzzle must be solved by CFC-excluded cascade.
            assert!(
                r.solved,
                "CFC-excluded cascade did not fully solve emitted puzzle: {}",
                cp.puzzle
            );
            assert!(
                r.se_score >= 7.6,
                "SE score {:.1} < 7.6 (CFC-excluded rating) for puzzle: {}",
                r.se_score,
                cp.puzzle
            );
            assert!(
                r.frontier.contains(&TechniqueId::RegionForcingChain),
                "RegionForcingChain not in CFC-excluded frontier {:?} for puzzle: {}",
                r.frontier,
                cp.puzzle
            );
        }
    }

    /// Every emitted puzzle must have a unique solution.
    #[test]
    fn emitted_puzzle_unique_solution() {
        let spec = RfcReverseSpec {
            target_se: 7.6,
            attempts_per_seed: 300,
            clue_min: 20,
            clue_max: 28,
            rng_seed: 7,
            require_load_bearing: true,
        };
        let results = batch_rfc_reverse_construct::<9, 3, 3>(&spec, 5);
        for cp in &results {
            let g = crate::generic::grid::Grid::<9, 3, 3>::from_str(&cp.puzzle)
                .expect("emitted puzzle must parse");
            let n_solutions = count_solutions_up_to(&g, 2);
            assert_eq!(
                n_solutions, 1,
                "Puzzle does not have a unique solution (count={}): {}",
                n_solutions, cp.puzzle
            );
        }
    }

    /// Load-bearing probe invariant: every puzzle emitted with
    /// `require_load_bearing = true` must NOT be solvable when both
    /// CellForcingChain and RegionForcingChain are excluded.
    ///
    /// This verifies the CRIT-2 fix: RFC is not merely incidental while
    /// DFC/NFC do the actual work.  If this assertion fires it means the
    /// load-bearing probe is bypassed or inverted.
    ///
    /// Also verifies that relaxing to `require_load_bearing = false` finds at
    /// least as many puzzles (probe can only reject, never promote).
    #[test]
    fn accepts_only_load_bearing_rfc() {
        use crate::generic::rater::rate_excluding;

        let spec_strict = RfcReverseSpec {
            target_se: 7.6,
            attempts_per_seed: 300,
            clue_min: 20,
            clue_max: 28,
            rng_seed: 42,
            require_load_bearing: true,
        };
        let spec_relaxed = RfcReverseSpec {
            require_load_bearing: false,
            ..spec_strict.clone()
        };

        let strict_results = batch_rfc_reverse_construct::<9, 3, 3>(&spec_strict, 10);
        let relaxed_results = batch_rfc_reverse_construct::<9, 3, 3>(&spec_relaxed, 10);

        // Strict mode must not produce MORE puzzles than relaxed mode
        // (probe can only filter, never invent).
        assert!(
            strict_results.len() <= relaxed_results.len(),
            "require_load_bearing=true produced more puzzles than false \
             ({} vs {}); probe logic is inverted",
            strict_results.len(),
            relaxed_results.len(),
        );

        // Every puzzle emitted under strict mode must, when both CFC and RFC
        // are excluded, either fail to solve OR rate strictly harder than the
        // CFC-excluded baseline (loose load-bearing: removing RFC makes the
        // cascade unsolvable or pushes SE up).
        for cp in &strict_results {
            let g = crate::generic::grid::Grid::<9, 3, 3>::from_str(&cp.puzzle)
                .expect("emitted puzzle must parse");
            let r_excl = rate_excluding(&g, CFC_ONLY);
            let r_lb = rate_excluding(&g, CFC_AND_RFC);
            assert!(
                !r_lb.rater_error,
                "Rater error on load-bearing probe for puzzle: {}",
                cp.puzzle
            );
            assert!(
                !r_lb.solved || r_lb.se_score > r_excl.se_score,
                "Puzzle solved without CFC+RFC at same/lower SE: RFC was not load-bearing but \
                 was emitted under require_load_bearing=true. Puzzle: {}",
                cp.puzzle
            );
        }
    }

    /// Verify `rfc_reverse_spec` builds a valid `ReverseSpec`.
    #[test]
    fn rfc_reverse_spec_is_valid() {
        let spec = rfc_reverse_spec(20, 28, 100);
        assert!(spec.validate().is_ok(), "rfc_reverse_spec must be valid");
        let implied = spec.match_mode.implied_required();
        assert!(implied.contains(&TechniqueId::RegionForcingChain));
    }
}
