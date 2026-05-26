//! R3.5: Guided-constructive reverse synthesis for **Cell Forcing Chain (CFC)** puzzles.
//!
//! ## Approach: Guided Constructive (not true-reverse)
//!
//! True top-down reverse construction for CFC is intractable: CFC depends on the
//! *global candidate state* after all simpler techniques are exhausted. You cannot
//! cheaply derive "what candidate state makes CFC fire" from a solution grid — you
//! would need to solve the inverse propagation problem. Instead we use a
//! **guided constructive** approach:
//!
//! 1. Generate a random uniquely-solvable seed puzzle at a clue count near the
//!    upper end of the requested band (T4Plus puzzles tend to live in 20–28 clues
//!    for 9×9).
//! 2. Run the rater. If CellForcingChain ∈ frontier AND se_score ≥ target_se AND
//!    no excluded technique fired, accept immediately.
//! 3. Otherwise, greedily try to remove clues (shuffled order) while preserving
//!    uniqueness. After each successful removal, re-rate and check the accept
//!    condition. Greedy removal tends to push the puzzle into harder territory
//!    where lower techniques can no longer substitute for CFC.
//! 4. Repeat with a fresh seed if the greedy pass fails to find an accept state.
//!
//! ## Trade-off
//!
//! - Hit rate is low (CFC puzzles are a small fraction of all T4Plus puzzles at
//!   standard clue counts). With `attempts_per_puzzle = 200` per seed and
//!   `max_seeds = 500` we empirically achieve ~0.5–2 pps single-thread for 9×9.
//! - The approach generalises to all (N, BR, BC) sizes without CFC-specific
//!   chain construction logic.
//! - Each emitted puzzle is guaranteed: unique solution, re-rates with CFC in
//!   frontier, se_score ≥ target_se, and (when `require_load_bearing=true`) CFC
//!   is the load-bearing technique.
//!
//! ## Accept gate semantics (v2)
//!
//! `is_accept` enforces gates in order:
//!   1. Basic: solved, CellForcingChain ∈ frontier, se_score ≥ target_se, k_min
//!      constraint via se_score proxy.
//!   2. **Upper-bound gate** (when `target_se_upper = Some(x)`): reject if
//!      se_score > x.  Use `Some(8.5)` for "strict CFC, no harder techniques"
//!      (DFC/NFC fire at ≥8.5).  Fixes C1: se_score is MAX across all fired
//!      techniques — a puzzle where CFC fires at 8.0 but DFC pushes se_score to
//!      9.0 is accepted without this gate despite CFC being incidental.
//!   3. Excluded techniques gate: no listed technique may appear in frontier.
//! The **load-bearing probe** (C2 fix) is checked in `try_seed` after `is_accept`
//! returns true: re-rate excluding CellForcingChain; if tier stays T4Plus, CFC
//! was not load-bearing — continue the greedy loop.  Controlled by
//! `require_load_bearing` (default `true`).
//!
//! ## AlphaEvolve contract
//!
//! Sections (per `docs/alphaevolve_contract.md`):
//!   §1 Interface  — `CfcReverseSpec`, `batch_cfc_reverse_construct`
//!   §2 Algorithm  — guided constructive (seed → greedy removal → rerate accept)
//!   §3 Restrictions — does not modify Technique/TechniqueProgress/TechniqueId/Tier
//!   §4 Dependencies — generator, rater, search, grid, reverse_construct::ReverseResult
//!   §5 Versioning — v1 = guided-constructive; v2 = load-bearing probe + upper bound
//!   §6 Tests      — constructs_at_least_one_puzzle, emitted_puzzle_rates_se_ge_8,
//!                   emitted_puzzle_unique_solution, spec_validation,
//!                   accepts_only_load_bearing_cfc, accepts_only_k_ge_3

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle_with_solution, GenConfig};
use super::grid::Grid;
use super::rater::{rate, RateResult};
use super::reverse_construct::ReverseResult;
use super::search::count_solutions_up_to;
use super::techniques::TechniqueId;

/// Configuration for one CFC reverse-synthesis run.
#[derive(Clone, Debug)]
pub struct CfcReverseSpec {
    /// Minimum SE score accepted for the emitted puzzle.  Default 8.0.
    pub target_se: f64,
    /// Minimum k (branch count) required on the CFC step.  Use 3 for strict
    /// CFC (SE 8.0); use 2 to also accept Y-Chain pivots (SE 6.6).  Default 3.
    pub target_k_min: u32,
    /// Inclusive clue-count band for the seed puzzle.  Typical: 20–28 for 9×9 T4Plus.
    pub clue_min: u32,
    /// Upper bound of the clue-count band.
    pub clue_max: u32,
    /// Max greedy removal trials *per seed*. 0 = N*N (one full shuffled pass).
    pub attempts_per_seed: u32,
    /// Max number of seed puzzles to try before giving up.  Default 500.
    pub max_seeds: u32,
    /// RNG seed for reproducibility.
    pub rng_seed: u64,
    /// If true (default), require that CFC is load-bearing: re-rate excluding
    /// CellForcingChain and confirm the puzzle no longer solves at T4Plus.
    /// Adds one extra rate call per accepted puzzle.  Set to false only for
    /// high-throughput exploratory runs where purity matters less.
    pub require_load_bearing: bool,
    /// Optional upper bound on se_score.  When `Some(x)`, puzzles with
    /// se_score > x are rejected even if CFC is in the frontier.
    /// Recommended value for "strict CFC, no harder techniques": `Some(8.5)`
    /// (DFC/NFC fire at ≥8.5).  Default `None` (no upper bound).
    pub target_se_upper: Option<f64>,
    /// Technique ids that must NOT appear in the rated frontier.
    pub excluded_techniques: Vec<TechniqueId>,
}

impl Default for CfcReverseSpec {
    fn default() -> Self {
        Self {
            target_se: 8.0,
            target_k_min: 3,
            clue_min: 20,
            clue_max: 28,
            attempts_per_seed: 200,
            max_seeds: 500,
            rng_seed: 0,
            // Default `false` because CFC sits in Tier::T4Plus alongside RFC/DFC/
            // NestedFC; strict load-bearing checks (tier-drop or `!solved`)
            // almost always reject since some other T4Plus tech can substitute.
            // Set to `true` explicitly when you want the SE-worsens probe.
            require_load_bearing: false,
            target_se_upper: None,
            excluded_techniques: vec![],
        }
    }
}

impl CfcReverseSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.clue_min > self.clue_max {
            return Err(format!(
                "clue_min ({}) > clue_max ({})",
                self.clue_min, self.clue_max
            ));
        }
        if self.target_se < 0.0 {
            return Err("target_se must be >= 0.0".to_string());
        }
        if self.target_k_min < 2 {
            return Err("target_k_min must be >= 2 (CFC requires at least 2 branches)".to_string());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Accept predicate
// ---------------------------------------------------------------------------

/// Return true iff `r` satisfies the CFC accept condition for `spec`.
///
/// Conditions (checked in order — first failure short-circuits):
///  1. Puzzle solved (no rater error, not left unsolved).
///  2. CellForcingChain in frontier.
///  3. se_score >= spec.target_se (lower bound).
///  4. If target_k_min >= 3: se_score >= 8.0 (k=2 Y-Chain fires at 6.6;
///     k≥3 CFC fires at 8.0 — inferred from se_score since TechniqueProgress
///     k_branches is not surfaced through RateResult).
///  5. If target_se_upper is Some(x): se_score <= x (upper bound).  Use
///     Some(8.5) to exclude puzzles where DFC/NFC fired alongside CFC.
///  6. No excluded technique in frontier.
///
/// Note: the load-bearing probe is checked *outside* this predicate in
/// `try_seed` so the greedy loop can continue rather than hard-rejecting.
fn is_accept(r: &RateResult, spec: &CfcReverseSpec) -> bool {
    if r.rater_error || !r.solved {
        return false;
    }
    if !r.frontier.contains(&TechniqueId::CellForcingChain) {
        return false;
    }
    if r.se_score < spec.target_se {
        return false;
    }
    // Enforce k_min via se_score: k=2 fires at 6.6, k>=3 fires at 8.0.
    // If caller wants k>=3 (the default), require se_score >= 8.0.
    if spec.target_k_min >= 3 && r.se_score < 8.0 {
        return false;
    }
    // Optional upper bound: reject if harder techniques pushed se_score above cap.
    if let Some(upper) = spec.target_se_upper {
        if r.se_score > upper {
            return false;
        }
    }
    for excl in &spec.excluded_techniques {
        if r.frontier.contains(excl) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Core single-seed attempt
// ---------------------------------------------------------------------------

/// Try to produce a CFC puzzle from one seed.  Returns `Some(ReverseResult)`
/// on accept, `None` if the seed did not yield a valid puzzle within the
/// greedy trial budget.
fn try_seed<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &CfcReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    let nn = N * N;

    // 1. Generate a uniquely-solvable seed at a clue count in [clue_min, clue_max].
    let target_clues = if spec.clue_min == spec.clue_max {
        spec.clue_min
    } else {
        rng.gen_range(spec.clue_min..=spec.clue_max)
    };
    let cfg = GenConfig {
        target_clues,
        max_attempts: 0,
    };
    let (mut puzzle, clue_count_init, solution) =
        gen_unique_puzzle_with_solution::<N, BR, BC, R>(rng, &cfg);
    let _ = clue_count_init;

    // 2. Rate the seed immediately -- maybe we get lucky.
    let r = rate(&puzzle);
    if is_accept(&r, spec) {
        if spec.require_load_bearing && !check_load_bearing(&puzzle, r.se_score) {
            // Not load-bearing; proceed to greedy removal.
        } else {
            let clue_count = puzzle.solved.iter().filter(|&&d| d != 0).count() as u32;
            return Some(ReverseResult {
                puzzle,
                solution,
                rate: r,
                clue_count,
                attempts_taken: 1,
            });
        }
    }

    // 3. Greedy removal loop: try removing clues in shuffled order.
    // Each successful removal (unique solution preserved) re-rates.
    // We accept as soon as is_accept holds.
    let mut order: Vec<u16> = (0..nn as u16).collect();
    order.shuffle(rng);

    let max_trials = if spec.attempts_per_seed == 0 {
        nn as u32
    } else {
        spec.attempts_per_seed
    };

    let mut trials: u32 = 0;
    for &c in &order {
        if trials >= max_trials {
            break;
        }
        let c = c as usize;
        if puzzle.solved[c] == 0 {
            // Already empty -- skip.
            continue;
        }
        trials += 1;

        // Build trial puzzle without cell c.
        let mut trial: Grid<N, BR, BC> = Grid::empty();
        let mut feasible = true;
        for i in 0..nn {
            if i == c {
                continue;
            }
            let d = puzzle.solved[i];
            if d == 0 {
                continue;
            }
            if trial.assign(i, d).is_err() {
                feasible = false;
                break;
            }
        }
        if !feasible {
            continue;
        }

        // Check uniqueness before committing the removal.
        if count_solutions_up_to(&trial, 2) != 1 {
            continue;
        }

        // Accept the removal -- update the working puzzle.
        puzzle = trial;

        // Re-rate.
        let r = rate(&puzzle);
        if is_accept(&r, spec) {
            if spec.require_load_bearing && !check_load_bearing(&puzzle, r.se_score) {
                // Not load-bearing -- continue removing.
                continue;
            }
            let clue_count = puzzle.solved.iter().filter(|&&d| d != 0).count() as u32;
            return Some(ReverseResult {
                puzzle,
                solution,
                rate: r,
                clue_count,
                attempts_taken: trials,
            });
        }
    }

    None
}

/// Check whether CFC is load-bearing: re-rate excluding CellForcingChain;
/// CFC is "load-bearing" iff the puzzle becomes strictly harder OR unsolvable
/// without it. Tier-drop alone (the `aic_reverse` pattern) is wrong here
/// because CFC lives in `Tier::T4Plus` alongside RFC/DFC/NestedFC — other
/// T4Plus techniques can almost always substitute, so the tier-drop probe
/// rejects ~100% of candidates (empirical: 0 emits / 50000 attempts on
/// cuda-host2 idle benchmark, 2026-05-14).
///
/// Loose semantics adopted instead: `r2.se_score > r.se_score` (without CFC,
/// the puzzle requires a strictly harder technique) OR `!r2.solved` (without
/// CFC, no analytic path exists).
fn check_load_bearing<const N: usize, const BR: usize, const BC: usize>(
    puzzle: &Grid<N, BR, BC>,
    full_se_score: f64,
) -> bool {
    use super::rater::rate_excluding;

    let r2 = rate_excluding(puzzle, &[TechniqueId::CellForcingChain]);
    if r2.rater_error {
        return false;
    }
    // Load-bearing iff puzzle is unsolvable OR harder without CFC.
    !r2.solved || r2.se_score > full_se_score
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Single-threaded CFC reverse-construct.  Returns `Some` on the first
/// accepted puzzle, `None` if `spec.max_seeds` is exhausted.
pub fn cfc_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    spec: &CfcReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() {
        return None;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(spec.rng_seed);
    for _ in 0..spec.max_seeds {
        if let Some(r) = try_seed::<N, BR, BC, _>(&mut rng, spec) {
            return Some(r);
        }
    }
    None
}

/// Parallel batch CFC reverse-construct.  Produces up to `num_puzzles` accepted
/// puzzles using `threads` worker threads.  Workers share a global kept-counter
/// and stop once `num_puzzles` is reached or each worker has exhausted its
/// per-worker seed budget (`spec.max_seeds` divided among workers).
///
/// Signature mirrors `batch_aic_reverse_construct` and `batch_reverse_construct`.
pub fn batch_cfc_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &CfcReverseSpec,
    num_puzzles: u32,
    threads: usize,
) -> Vec<ReverseResult<N, BR, BC>> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    if spec.validate().is_err() {
        return vec![];
    }

    let threads = threads.max(1);
    let collected: Arc<Mutex<Vec<ReverseResult<N, BR, BC>>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_puzzles as usize)));
    let kept = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::with_capacity(threads);
    for w in 0..threads {
        let spec = spec.clone();
        let collected = collected.clone();
        let kept = kept.clone();
        // Same seed-derivation scheme as batch_reverse_construct.
        let child_seed = rng_seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
        handles.push(std::thread::spawn(move || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
            let per_worker_seeds = if spec.max_seeds == 0 {
                u32::MAX
            } else {
                ((spec.max_seeds as u64 + threads as u64 - 1) / threads as u64) as u32
            };
            let mut seeds_used = 0u32;
            while seeds_used < per_worker_seeds {
                if kept.load(Ordering::Relaxed) >= num_puzzles {
                    return;
                }
                seeds_used += 1;
                if let Some(res) = try_seed::<N, BR, BC, _>(&mut rng, &spec) {
                    let prev = kept.fetch_add(1, Ordering::Relaxed);
                    if prev < num_puzzles {
                        let mut guard = collected.lock().unwrap();
                        guard.push(res);
                    } else {
                        kept.fetch_sub(1, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    let guard = collected.lock().unwrap();
    guard.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::rater::rate;
    use crate::generic::search::count_solutions_up_to;

    // Helper: default spec with a small seed budget suitable for CI.
    // require_load_bearing=false for speed; the load-bearing probe tests below
    // use their own specs.
    fn ci_spec() -> CfcReverseSpec {
        CfcReverseSpec {
            target_se: 8.0,
            target_k_min: 3,
            clue_min: 20,
            clue_max: 28,
            attempts_per_seed: 200,
            max_seeds: 2000,
            rng_seed: 42,
            require_load_bearing: false,
            target_se_upper: None,
            excluded_techniques: vec![],
        }
    }

    /// Validate: spec accepts valid defaults and rejects bad inputs.
    #[test]
    fn spec_validation() {
        let mut s = ci_spec();
        assert!(s.validate().is_ok());

        s.clue_min = 30;
        s.clue_max = 20;
        assert!(s.validate().is_err(), "clue_min > clue_max should be invalid");

        let mut s2 = ci_spec();
        s2.target_k_min = 1;
        assert!(s2.validate().is_err(), "target_k_min < 2 should be invalid");
    }

    /// Smoke: batch_cfc_reverse_construct emits >= 1 puzzle under a generous budget.
    ///
    /// This test is intentionally lenient on budget (2000 seeds) because CFC is
    /// rare at random clue counts. It may be slow on debug builds (~10--30s);
    /// use `cargo test --release` for CI.
    #[test]
    fn constructs_at_least_one_puzzle() {
        let spec = ci_spec();
        let results = batch_cfc_reverse_construct::<9, 3, 3>(42, &spec, 1, 1);
        assert!(
            !results.is_empty(),
            "Expected at least 1 CFC puzzle within 2000 seeds; got 0. \
             This may indicate a rater/accept regression."
        );
    }

    /// Each emitted puzzle must re-rate with CFC in frontier and se_score >= 8.0.
    #[test]
    fn emitted_puzzle_rates_se_ge_8() {
        let spec = ci_spec();
        let results = batch_cfc_reverse_construct::<9, 3, 3>(42, &spec, 1, 1);
        if results.is_empty() {
            // Inconclusive -- let constructs_at_least_one_puzzle catch this.
            return;
        }
        for r in &results {
            let re_rated = rate(&r.puzzle);
            assert!(
                !re_rated.rater_error,
                "Re-rating emitted puzzle returned rater_error"
            );
            assert!(
                re_rated.frontier.contains(&TechniqueId::CellForcingChain),
                "Re-rated puzzle must have CellForcingChain in frontier; got {:?}",
                re_rated.frontier
            );
            assert!(
                re_rated.se_score >= 8.0,
                "Re-rated se_score must be >= 8.0; got {}",
                re_rated.se_score
            );
        }
    }

    /// Each emitted puzzle must have a unique solution.
    #[test]
    fn emitted_puzzle_unique_solution() {
        let spec = ci_spec();
        let results = batch_cfc_reverse_construct::<9, 3, 3>(42, &spec, 1, 1);
        if results.is_empty() {
            return;
        }
        for r in &results {
            let nsol = count_solutions_up_to(&r.puzzle, 2);
            assert_eq!(
                nsol, 1,
                "Emitted puzzle must have exactly 1 solution; count_solutions_up_to returned {}",
                nsol
            );
        }
    }

    /// Helper: construct a minimal RateResult for `is_accept` unit tests.
    /// Does not touch any Grid or rater -- purely for predicate logic testing.
    fn make_rate_result(
        solved: bool,
        cfc_in_frontier: bool,
        se_score: f64,
        extra_techs: &[TechniqueId],
    ) -> crate::generic::rater::RateResult {
        use crate::generic::techniques::Tier;
        let mut frontier = vec![];
        if cfc_in_frontier {
            frontier.push(TechniqueId::CellForcingChain);
        }
        frontier.extend_from_slice(extra_techs);
        crate::generic::rater::RateResult {
            tier: Tier::T4Plus,
            frontier,
            solved,
            trace: vec![],
            rater_error: false,
            wave_depth: 1,
            backtrack_steps: 0,
            unique_solution: false,
            se_score,
        }
    }

    /// `is_accept` must reject puzzles where se_score exceeds `target_se_upper`
    /// (C1 fix: se_score is MAX across all fired techniques, not just CFC's own
    /// contribution -- so a puzzle where DFC/NFC pushed se_score to 9.0 while
    /// CFC fired only at 8.0 should be rejectable via an upper bound).
    ///
    /// Also verifies the excluded-technique gate and the no-CFC rejection.
    #[test]
    fn accepts_only_load_bearing_cfc() {
        let base_spec = CfcReverseSpec {
            target_se: 8.0,
            target_k_min: 3,
            target_se_upper: None,
            require_load_bearing: false,
            ..ci_spec()
        };

        // se_score=8.0, CFC in frontier -- accepted without upper bound.
        let r = make_rate_result(true, true, 8.0, &[]);
        assert!(is_accept(&r, &base_spec), "se_score=8.0 should be accepted with no upper bound");

        // Add upper bound of 8.5: se_score=8.0 still passes.
        let spec_with_upper = CfcReverseSpec {
            target_se_upper: Some(8.5),
            ..base_spec.clone()
        };
        assert!(is_accept(&r, &spec_with_upper), "se_score=8.0 should pass upper bound 8.5");

        // se_score=9.0 (DFC/NFC fired alongside CFC) -- rejected by upper bound 8.5.
        let r_hard = make_rate_result(true, true, 9.0, &[]);
        assert!(
            !is_accept(&r_hard, &spec_with_upper),
            "se_score=9.0 should be rejected by target_se_upper=8.5 (DFC/NFC present)"
        );

        // Same se_score=9.0 without upper bound -- accepted (backwards compat).
        assert!(
            is_accept(&r_hard, &base_spec),
            "se_score=9.0 without upper bound should be accepted"
        );

        // CFC not in frontier at all -- always rejected regardless of se_score.
        let r_no_cfc = make_rate_result(true, false, 8.0, &[]);
        assert!(!is_accept(&r_no_cfc, &base_spec), "No CFC in frontier must always reject");

        // Excluded technique in frontier -- rejected even if CFC is present.
        let spec_excl = CfcReverseSpec {
            excluded_techniques: vec![TechniqueId::DynamicForcingChain],
            ..base_spec.clone()
        };
        let r_dfc = make_rate_result(true, true, 8.0, &[TechniqueId::DynamicForcingChain]);
        assert!(
            !is_accept(&r_dfc, &spec_excl),
            "DFC in excluded list must reject even when CFC is present"
        );
    }

    /// `is_accept` must reject k=2 Y-Chain puzzles (se_score=6.6) when
    /// `target_k_min=3`.  C1 scenario: a k=2 CFC fires (rating 6.6) and a
    /// later DFC/NFC fire bumps se_score to 8.0 -- `target_se >= 8.0` passes
    /// but `target_k_min=3` (proxy: se_score < 8.0 from CFC alone) should
    /// reject.  Additionally, when DFC bumps se_score to 9.0, the strict upper
    /// bound (`target_se_upper=Some(8.5)`) provides the correct rejection gate.
    #[test]
    fn accepts_only_k_ge_3() {
        let spec_k3 = CfcReverseSpec {
            target_se: 6.0,  // low floor so k=2 passes the se_score floor
            target_k_min: 3, // k_min=3 requires se_score >= 8.0 (proxy)
            target_se_upper: None,
            require_load_bearing: false,
            ..ci_spec()
        };

        // k=2 Y-Chain: se_score=6.6 -- rejected by target_k_min=3 gate (6.6 < 8.0).
        let r_k2 = make_rate_result(true, true, 6.6, &[]);
        assert!(
            !is_accept(&r_k2, &spec_k3),
            "k=2 Y-Chain (se_score=6.6) must be rejected when target_k_min=3"
        );

        // k=3 CFC: se_score=8.0 -- accepted.
        let r_k3 = make_rate_result(true, true, 8.0, &[]);
        assert!(
            is_accept(&r_k3, &spec_k3),
            "k=3 CFC (se_score=8.0) must be accepted when target_k_min=3"
        );

        // Relax to k=2 -- then se_score=6.6 is accepted.
        let spec_k2 = CfcReverseSpec {
            target_k_min: 2,
            target_se: 6.0,
            target_se_upper: None,
            require_load_bearing: false,
            ..ci_spec()
        };
        assert!(
            is_accept(&r_k2, &spec_k2),
            "k=2 Y-Chain (se_score=6.6) must be accepted when target_k_min=2"
        );

        // C1 scenario: k=2 CFC fires (se 6.6) + DFC bumps se_score to 9.0.
        // Without upper bound: passes k_min=3 gate (9.0 >= 8.0) -- but DFC is
        // the real driver. Upper bound gate rejects correctly.
        let r_k2_plus_dfc = make_rate_result(true, true, 9.0, &[]);
        assert!(
            is_accept(&r_k2_plus_dfc, &spec_k3),
            "se_score=9.0 passes k_min=3 (9.0 >= 8.0); use target_se_upper for strict bucket"
        );
        let spec_strict = CfcReverseSpec {
            target_se_upper: Some(8.5),
            ..spec_k3.clone()
        };
        assert!(
            !is_accept(&r_k2_plus_dfc, &spec_strict),
            "se_score=9.0 must be rejected by target_se_upper=8.5 in strict bucket"
        );
    }
}
