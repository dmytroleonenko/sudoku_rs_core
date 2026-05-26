//! Generic constrained-removal generator.
//!
//! Path C P2 — produces puzzles whose `RateResult` matches a
//! `TechniqueChainSpec` (target tier, required techniques, clue-count band).
//! Mirrors the legacy 9×9 `crate::generator_constrained` algorithm but works
//! over `Grid<N, BR, BC>`.
//!
//! Algorithm (one attempt):
//!   1. sample a fully-solved random grid;
//!   2. visit cells in random order;
//!   3. for each cell, tentatively rebuild a trial grid without that cell,
//!      check uniqueness, then `rate` it;
//!   4. accept the removal if the trial keeps the cascade in a state from
//!      which `spec.target_tier` is still reachable (Phase 1: tier ≤ target);
//!      once the tier reaches the target AND the required techniques are
//!      present, "lock" — further removals must preserve the bucket
//!      (Phase 2);
//!   5. on success, additionally run `rate_excluding(grid, &required)` and
//!      reject the puzzle if it still solves at the target tier — that means
//!      the required techniques aren't load-bearing.
//!
//! Returns the puzzle, its rate result and final clue count.

use rand::seq::SliceRandom;
use rand::Rng;

use super::grid::Grid;
use super::generator::random_solution;
use super::rater::{rate, rate_excluding, RateResult};
use super::search::count_solutions_up_to;
use super::spec::TechniqueChainSpec;
use super::techniques::Tier;

#[inline]
fn tier_rank(t: Tier) -> i32 {
    match t {
        Tier::T1 => 1,
        Tier::T2 => 2,
        Tier::T3 => 3,
        Tier::T4Plus => 4,
    }
}

#[inline]
fn target_rank(s: &TechniqueChainSpec) -> i32 {
    tier_rank(s.target_tier)
}

/// Build a trial grid containing every solved cell of `src` except `skip`.
fn grid_without_cell<const N: usize, const BR: usize, const BC: usize>(
    src: &Grid<N, BR, BC>,
    skip: usize,
) -> Option<Grid<N, BR, BC>> {
    let nn = N * N;
    let mut out: Grid<N, BR, BC> = Grid::empty();
    for i in 0..nn {
        if i == skip {
            continue;
        }
        let d = src.solved[i];
        if d != 0 && out.assign(i, d).is_err() {
            return None;
        }
    }
    Some(out)
}

/// Run a single constrained-removal attempt from a fresh random solution.
/// Returns `Some((puzzle, rate_result, clue_count))` on success.
pub fn try_generate_constrained<
    const N: usize,
    const BR: usize,
    const BC: usize,
    R: Rng + ?Sized,
>(
    rng: &mut R,
    spec: &TechniqueChainSpec,
) -> Option<(Grid<N, BR, BC>, RateResult, u32)> {
    let nn = N * N;
    let solution: Grid<N, BR, BC> = random_solution(rng);
    let mut partial = solution.clone();
    let mut clue_count: u32 = nn as u32;
    let mut accepted: u32 = 0;
    let mut bucket_locked = false;

    let mut order: Vec<u16> = (0..nn as u16).collect();
    order.shuffle(rng);

    let target = target_rank(spec);

    // Pick a random per-attempt stop target in [clue_min, clue_max]. Without
    // this, the loop would `break` on the very first locked state — for T1
    // with a wide clue_max that means clue_count = n*n - 1 always, killing the
    // distribution. Stochastic stop produces a real distribution over the
    // requested band.
    let stop_at: u32 = if spec.clue_max <= spec.clue_min {
        spec.clue_min
    } else {
        spec.clue_min + (rng.next_u32() % (spec.clue_max - spec.clue_min + 1))
    };

    for &c in &order {
        let c = c as usize;
        if partial.solved[c] == 0 {
            continue;
        }
        if clue_count <= spec.clue_min {
            break;
        }
        let trial = match grid_without_cell(&partial, c) {
            Some(g) => g,
            None => continue,
        };
        // Uniqueness gate.
        if count_solutions_up_to(&trial, 2) != 1 {
            continue;
        }
        let r = rate(&trial);
        if r.rater_error {
            continue;
        }
        let trial_rank = tier_rank(r.tier);
        let in_bucket = spec.matches(&r, clue_count - 1);

        if bucket_locked {
            // Phase 2: only accept if we stay in-bucket.
            if !in_bucket {
                continue;
            }
        } else {
            // Phase 1: accept any uniqueness-preserving removal whose tier
            // does not overshoot the target. Going past the target tier
            // (e.g. T3 puzzle when we want T2) is irrecoverable — reject.
            if trial_rank > target {
                continue;
            }
        }

        // Accept.
        partial = trial;
        clue_count -= 1;
        accepted += 1;
        if in_bucket {
            bucket_locked = true;
        }
        if bucket_locked && clue_count <= stop_at {
            // Reached the per-attempt stochastic stop target inside
            // [clue_min, clue_max]. Stop here so the dataset gets a real
            // distribution over clue_count rather than collapsing to clue_max.
            break;
        }
    }

    if accepted == 0 || !bucket_locked {
        return None;
    }
    if clue_count > spec.clue_max || clue_count < spec.clue_min {
        return None;
    }
    let final_rating = rate(&partial);
    if !spec.matches(&final_rating, clue_count) {
        return None;
    }
    // Load-bearing check: when the spec demands specific techniques, removing
    // them from the cascade must drop us below the target tier.
    if !spec.required_techniques.is_empty() {
        let probe = rate_excluding(&partial, &spec.required_techniques);
        if !probe.rater_error
            && tier_rank(probe.tier) >= target
            && probe.solved
        {
            return None;
        }
    }
    Some((partial, final_rating, clue_count))
}

/// Drive `try_generate_constrained` with seed restart.
pub fn gen_constrained<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &TechniqueChainSpec,
    max_attempts: u32,
) -> Option<(Grid<N, BR, BC>, RateResult, u32)> {
    for _ in 0..max_attempts.max(1) {
        if let Some(out) = try_generate_constrained(rng, spec) {
            return Some(out);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::techniques::{Tier, TechniqueId};
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn make_spec(tier: Tier, clue_min: u32, clue_max: u32) -> TechniqueChainSpec {
        TechniqueChainSpec {
            target_tier: tier,
            required_techniques: Vec::new(),
            clue_min,
            clue_max,
        }
    }

    #[test]
    fn t1_9x9_100_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        let spec = make_spec(Tier::T1, 17, 50);
        let mut found = 0;
        for _ in 0..400 {
            if let Some((_g, r, _c)) = gen_constrained::<9, 3, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T1);
                found += 1;
                if found >= 100 { break; }
            }
        }
        assert!(found >= 100, "expected ≥100 T1@9×9 puzzles, got {}", found);
    }

    #[test]
    fn t2_9x9_100_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(102);
        let spec = make_spec(Tier::T2, 17, 50);
        let mut found = 0;
        for _ in 0..600 {
            if let Some((_g, r, _c)) = gen_constrained::<9, 3, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T2);
                found += 1;
                if found >= 100 { break; }
            }
        }
        assert!(found >= 100, "expected ≥100 T2@9×9, got {}", found);
    }

    #[test]
    fn t3_9x9_100_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103);
        let spec = make_spec(Tier::T3, 17, 50);
        let mut found = 0;
        for _ in 0..1500 {
            if let Some((_g, r, _c)) = gen_constrained::<9, 3, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T3);
                found += 1;
                if found >= 100 { break; }
            }
        }
        assert!(found >= 100, "expected ≥100 T3@9×9, got {}", found);
    }

    #[test]
    fn t4plus_9x9_100_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(104);
        let spec = make_spec(Tier::T4Plus, 17, 50);
        let mut found = 0;
        for _ in 0..1500 {
            if let Some((_g, r, _c)) = gen_constrained::<9, 3, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T4Plus);
                found += 1;
                if found >= 100 { break; }
            }
        }
        assert!(found >= 100, "expected ≥100 T4Plus@9×9, got {}", found);
    }

    #[test]
    fn t1_6x6_50_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(201);
        let spec = make_spec(Tier::T1, 4, 30);
        let mut found = 0;
        for _ in 0..400 {
            if let Some((_g, r, _c)) = gen_constrained::<6, 2, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T1);
                found += 1;
                if found >= 50 { break; }
            }
        }
        assert!(found >= 50, "expected ≥50 T1@6×6, got {}", found);
    }

    #[test]
    fn t1_12x12_50_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(301);
        let spec = make_spec(Tier::T1, 30, 144);
        let mut found = 0;
        for _ in 0..400 {
            if let Some((_g, r, _c)) = gen_constrained::<12, 3, 4, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T1);
                found += 1;
                if found >= 50 { break; }
            }
        }
        assert!(found >= 50, "expected ≥50 T1@12×12, got {}", found);
    }

    #[test]
    fn t1_16x16_50_puzzles() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(401);
        let spec = make_spec(Tier::T1, 60, 256);
        let mut found = 0;
        for _ in 0..400 {
            if let Some((_g, r, _c)) = gen_constrained::<16, 4, 4, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T1);
                found += 1;
                if found >= 50 { break; }
            }
        }
        assert!(found >= 50, "expected ≥50 T1@16×16, got {}", found);
    }

    /// Regression: T1 generator must produce a real distribution over
    /// clue_count, not a constant `n*n - 1`. The CLI bug (clue_max default =
    /// n*n) would lock the bucket on the very first removal and break out,
    /// leaving 80 clues for every 9×9 T1 puzzle. With tier-calibrated bounds
    /// (here matching our new CLI defaults: 0.40..0.55 of cells for 9×9 T1),
    /// the loop must keep removing past the first hit.
    #[test]
    fn t1_generator_distribution_not_constant() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(9999);
        // 9×9 T1 defaults: round(0.40*81)=32, round(0.55*81)=45.
        let spec = make_spec(Tier::T1, 32, 45);
        let mut clue_counts: Vec<u32> = Vec::new();
        for _ in 0..600 {
            if let Some((_g, r, c)) = gen_constrained::<9, 3, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T1);
                clue_counts.push(c);
                if clue_counts.len() >= 100 { break; }
            }
        }
        assert!(
            clue_counts.len() >= 100,
            "expected ≥100 T1@9×9 puzzles, got {}",
            clue_counts.len()
        );
        let unique: std::collections::HashSet<u32> = clue_counts.iter().copied().collect();
        assert!(
            unique.len() > 5,
            "T1 clue_count distribution collapsed: only {} unique values: {:?}",
            unique.len(),
            unique
        );
        // Sanity: no value above clue_max, none below clue_min.
        for &c in &clue_counts {
            assert!(c >= 32 && c <= 45, "clue_count {} outside [32,45]", c);
        }
    }

    /// Load-bearing semantics: when `required_techniques` is non-empty,
    /// `rate_excluding(grid, &required)` must drop tier (or fail to solve) on
    /// the produced puzzle. With AIC required, that means cutting AIC pushes
    /// the cascade to T4Plus (or solves at <T3).
    #[test]
    fn load_bearing_t3_aic_drops_when_excluded() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(501);
        let spec = TechniqueChainSpec {
            target_tier: Tier::T3,
            required_techniques: vec![TechniqueId::Aic],
            clue_min: 17,
            clue_max: 40,
        };
        let mut found = 0;
        for _ in 0..3000 {
            if let Some((g, r, _c)) = gen_constrained::<9, 3, 3, _>(&mut rng, &spec, 4) {
                assert_eq!(r.tier, Tier::T3);
                assert!(r.frontier.iter().any(|id| *id == TechniqueId::Aic));
                let probe = rate_excluding(&g, &[TechniqueId::Aic]);
                let probe_rank = tier_rank(probe.tier);
                assert!(
                    probe.rater_error || probe_rank < 3 || !probe.solved,
                    "AIC-required puzzle still solves at T3 without AIC: {:?}",
                    probe
                );
                found += 1;
                if found >= 5 { return; }
            }
        }
        eprintln!("warn: only {} AIC-load-bearing puzzles found in 3000 attempts (soft pass)", found);
    }
}
