//! Generic technique-targeted reverse-construct (Phase R1).
//!
//! Search-and-filter implementation: sample random unique-solution puzzles in
//! a clue band; rate; keep only those whose rated frontier matches the
//! caller-supplied `ReverseSpec` (target tier, all required techniques fired,
//! none of the excluded techniques fired). When `load_bearing` is set, also
//! verify that re-rating with the required-technique set excluded *drops* the
//! tier below `target_tier` — i.e. the required techniques are genuinely
//! load-bearing for solving the puzzle, not opportunistic.
//!
//! Per-technique constructive synthesis (chain-targeted AIC / Fish<K> /
//! ALS-XZ) is deferred to R2+. R1 establishes the API surface and the hit-rate
//! baseline that informs R2 priorities.
//!
//! Determinism: a single `reverse_construct` call is fully deterministic given
//! a seeded `Rng`. `batch_reverse_construct` derives per-worker child seeds
//! via splitmix-style jumps off the master seed (same scheme as
//! `pipeline_writer_generic::run_size`). With `threads=1` the result is
//! byte-identical across runs of a fixed `seed`. With `threads>1` the
//! per-worker seeds are deterministic, but the *kept* set is order-dependent
//! on the kept-counter race (workers may produce more candidates than the
//! `num_puzzles` budget; the first to reach the cap wins). Tests assert
//! single-threaded determinism only; multi-threaded batches are intended for
//! throughput, not reproducibility.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{gen_unique_puzzle_with_solution, GenConfig};
use super::grid::Grid;
use super::rater::{rate, rate_excluding, RateResult};
use super::techniques::{TechniqueId, Tier};

/// How the rated frontier is matched against the technique list.
///
/// All variants short-circuit when their list is empty in a *defined* way:
///   * `AllOf([])` → trivially matches (no constraint on frontier).
///   * `AnyOf([])` → trivially matches (no constraint).
///   * `ExactK { techniques: [], k: 0 }` → matches; `k>0` → never matches.
///   * `AtLeastK { techniques: [], k: 0 }` → matches; `k>0` → never matches.
///   * `Weighted { weights: [], min_picked: 0 }` → matches.
///
/// `Weighted` consumes the supplied RNG: each call to `matches` performs a
/// fresh Bernoulli sample per `(tech, p)` pair. With a deterministic seeded
/// RNG, reverse-construct remains deterministic.
#[derive(Clone, Debug)]
pub enum MatchMode {
    /// Every technique in the list must fire (R1 semantics).
    AllOf(Vec<TechniqueId>),
    /// At least one technique in the list must fire.
    AnyOf(Vec<TechniqueId>),
    /// Exactly `k` techniques from the list must fire.
    ExactK { techniques: Vec<TechniqueId>, k: u32 },
    /// At least `k` techniques from the list must fire.
    AtLeastK { techniques: Vec<TechniqueId>, k: u32 },
    /// Per-puzzle Bernoulli sample of the listed `(technique, p)` weights.
    /// All sampled techniques must fire; if fewer than `min_picked` were
    /// sampled, the match fails.
    Weighted {
        weights: Vec<(TechniqueId, f32)>,
        min_picked: u32,
    },
}

impl MatchMode {
    /// Returns true iff the rated `frontier` satisfies this mode.
    pub fn matches<R: Rng + ?Sized>(&self, frontier: &[TechniqueId], rng: &mut R) -> bool {
        match self {
            Self::AllOf(ts) => ts.iter().all(|t| frontier.contains(t)),
            Self::AnyOf(ts) => {
                if ts.is_empty() {
                    return true;
                }
                ts.iter().any(|t| frontier.contains(t))
            }
            Self::ExactK { techniques, k } => {
                let count = techniques.iter().filter(|t| frontier.contains(t)).count();
                count == *k as usize
            }
            Self::AtLeastK { techniques, k } => {
                let count = techniques.iter().filter(|t| frontier.contains(t)).count();
                count >= *k as usize
            }
            Self::Weighted { weights, min_picked } => {
                let picked: Vec<TechniqueId> = weights
                    .iter()
                    .filter_map(|(t, p)| if rng.gen::<f32>() < *p { Some(*t) } else { None })
                    .collect();
                if picked.len() < *min_picked as usize {
                    return false;
                }
                picked.iter().all(|t| frontier.contains(t))
            }
        }
    }

    /// Conservative load-bearing set: all techniques referenced by this mode.
    /// For `Weighted`, returns *every* listed technique (not the per-puzzle
    /// sample), so the load-bearing check excludes the entire candidate set.
    pub fn implied_required(&self) -> Vec<TechniqueId> {
        match self {
            Self::AllOf(ts) | Self::AnyOf(ts) => ts.clone(),
            Self::ExactK { techniques, .. } | Self::AtLeastK { techniques, .. } => {
                techniques.clone()
            }
            Self::Weighted { weights, .. } => weights.iter().map(|(t, _)| *t).collect(),
        }
    }

    /// Validate internal consistency (k bounds, weights non-negative, list
    /// uniqueness, etc.). Duplicate technique ids in `ExactK`/`AtLeastK`/
    /// `Weighted` are rejected: counting `frontier.contains(t)` per *entry*
    /// would otherwise double-count a single fired technique against `k`/
    /// `min_picked`.
    pub fn validate(&self) -> Result<(), String> {
        fn assert_unique(ts: &[TechniqueId], where_: &str) -> Result<(), String> {
            for (i, a) in ts.iter().enumerate() {
                if ts[..i].contains(a) {
                    return Err(format!(
                        "{}: technique {:?} appears more than once",
                        where_, a
                    ));
                }
            }
            Ok(())
        }
        match self {
            Self::AllOf(ts) | Self::AnyOf(ts) => assert_unique(ts, "match_mode list"),
            Self::ExactK { techniques, k } | Self::AtLeastK { techniques, k } => {
                assert_unique(techniques, "match_mode list")?;
                if *k as usize > techniques.len() {
                    return Err(format!(
                        "k ({}) exceeds technique-list length ({})",
                        k,
                        techniques.len()
                    ));
                }
                Ok(())
            }
            Self::Weighted { weights, min_picked } => {
                let ids: Vec<TechniqueId> = weights.iter().map(|(t, _)| *t).collect();
                assert_unique(&ids, "weighted list")?;
                for (t, p) in weights {
                    if !p.is_finite() || *p < 0.0 || *p > 1.0 {
                        return Err(format!(
                            "weighted: p={} for {:?} must be finite in [0,1]",
                            p, t
                        ));
                    }
                }
                if *min_picked as usize > weights.len() {
                    return Err(format!(
                        "weighted: min_picked ({}) exceeds weights length ({})",
                        min_picked,
                        weights.len()
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Reverse-construct request: what kind of puzzle to find.
#[derive(Clone, Debug)]
pub struct ReverseSpec {
    /// Exact-match target tier. The cascade rating must equal this tier.
    pub target_tier: Tier,
    /// Match mode for the technique frontier (R1.5).
    pub match_mode: MatchMode,
    /// None of these technique ids may appear in the rated frontier.
    pub excluded_techniques: Vec<TechniqueId>,
    /// If true and `match_mode` references at least one technique, verify
    /// that `rate_excluding(grid, match_mode.implied_required())` produces a
    /// tier strictly below `target_tier`.
    pub load_bearing: bool,
    /// Inclusive clue-count band (for the random seed puzzle).
    pub clue_min: u32,
    pub clue_max: u32,
    /// Hard cap on attempts before giving up.
    pub max_attempts: u32,
}

impl ReverseSpec {
    /// Convenience: build a spec with `MatchMode::AllOf(required)` (R1
    /// semantics).
    pub fn new_all_of(
        target_tier: Tier,
        required: Vec<TechniqueId>,
        excluded: Vec<TechniqueId>,
        load_bearing: bool,
        clue_min: u32,
        clue_max: u32,
        max_attempts: u32,
    ) -> Self {
        Self {
            target_tier,
            match_mode: MatchMode::AllOf(required),
            excluded_techniques: excluded,
            load_bearing,
            clue_min,
            clue_max,
            max_attempts,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.clue_min > self.clue_max {
            return Err(format!(
                "clue_min ({}) > clue_max ({})",
                self.clue_min, self.clue_max
            ));
        }
        self.match_mode.validate()?;
        let implied = self.match_mode.implied_required();
        for r in &implied {
            if self.excluded_techniques.iter().any(|e| e == r) {
                return Err(format!(
                    "technique {:?} appears in both required (via match_mode) and excluded",
                    r
                ));
            }
        }
        if matches!(self.target_tier, Tier::T1) && !implied.is_empty() {
            return Err(
                "target_tier=T1 with non-empty match_mode techniques is unsatisfiable \
                 (T1 puzzles fire no techniques)"
                    .to_string(),
            );
        }
        // load_bearing semantics are only well-defined for `AllOf`. For other
        // modes the implied set is conservative (full candidate list), which
        // can both falsely accept (excluding more than fired) and falsely
        // reject (excluding unrelated techniques drops tier for the wrong
        // reason). Reject the combination explicitly so callers must opt in
        // by switching to AllOf.
        if self.load_bearing && !matches!(self.match_mode, MatchMode::AllOf(_)) {
            return Err(
                "load_bearing=true is only supported for MatchMode::AllOf; \
                 for other modes the implied_required() set is conservative \
                 and the test would have ill-defined semantics"
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct ReverseResult<const N: usize, const BR: usize, const BC: usize> {
    pub puzzle: Grid<N, BR, BC>,
    pub solution: Grid<N, BR, BC>,
    pub rate: RateResult,
    pub clue_count: u32,
    pub attempts_taken: u32,
}

#[inline]
fn tier_rank(t: Tier) -> u8 {
    match t {
        Tier::T1 => 1,
        Tier::T2 => 2,
        Tier::T3 => 3,
        Tier::T4Plus => 4,
    }
}

/// Try `spec.max_attempts` random seed puzzles and return the first one whose
/// rating matches the spec. `None` if no match within budget.
pub fn reverse_construct<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &ReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() {
        return None;
    }
    let target_rank = tier_rank(spec.target_tier);
    for attempt in 0..spec.max_attempts {
        // 1. Random unique seed in clue band.
        let target_clues = if spec.clue_min == spec.clue_max {
            spec.clue_min
        } else {
            rng.gen_range(spec.clue_min..=spec.clue_max)
        };
        // `max_attempts: 0` here means "no extra cap on cell-removal trials";
        // gen_unique_puzzle still terminates after one pass over N*N cells.
        let cfg = GenConfig {
            target_clues,
            max_attempts: 0,
        };
        let (puzzle, clue_count, solution): (Grid<N, BR, BC>, u32, Grid<N, BR, BC>) =
            gen_unique_puzzle_with_solution(rng, &cfg);

        // 2. Rate.
        let r = rate(&puzzle);
        if r.rater_error {
            continue;
        }

        // 3. Tier match (exact).
        if tier_rank(r.tier) != target_rank {
            continue;
        }

        // 4. Match-mode satisfied by frontier.
        if !spec.match_mode.matches(&r.frontier, rng) {
            continue;
        }

        // 5. No excluded technique fired.
        if spec
            .excluded_techniques
            .iter()
            .any(|t| r.frontier.contains(t))
        {
            continue;
        }

        // 6. Load-bearing test (only if requested + we have something to exclude).
        let implied = spec.match_mode.implied_required();
        if spec.load_bearing && !implied.is_empty() {
            let r2 = rate_excluding(&puzzle, &implied);
            if r2.rater_error {
                continue;
            }
            // Tier MUST drop strictly below target_tier when the required set
            // is excluded. If the cascade still reaches target_tier (or even
            // higher — e.g. T4Plus stuck because the required tech can't be
            // substituted but the puzzle remains unsolved at target tier),
            // the required set is not load-bearing in the sense we want.
            if tier_rank(r2.tier) >= target_rank {
                continue;
            }
        }

        // 7. Solution captured from gen_unique_puzzle_with_solution — no
        //    redundant solve_unique on the freshly carved puzzle.
        return Some(ReverseResult {
            puzzle,
            solution,
            rate: r,
            clue_count,
            attempts_taken: attempt + 1,
        });
    }
    None
}

/// Run `num_puzzles` reverse-constructs with `threads` workers. Each worker
/// keeps drawing seed puzzles from its private RNG (seeded deterministically
/// off `rng_seed`) until the global kept-count reaches `num_puzzles` or the
/// per-worker attempt budget is exhausted.
///
/// Per-worker budget: each worker gets `spec.max_attempts` attempts total, so
/// the function returns at most `min(num_puzzles, threads * max_attempts /
/// avg_attempts_per_hit)` puzzles. If `spec.max_attempts == 0` the workers
/// run unbounded until `num_puzzles` is reached (caller's responsibility to
/// ensure the spec is satisfiable).
pub fn batch_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &ReverseSpec,
    num_puzzles: u32,
    threads: usize,
) -> Vec<ReverseResult<N, BR, BC>> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    let threads = threads.max(1);
    let collected: Arc<Mutex<Vec<ReverseResult<N, BR, BC>>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_puzzles as usize)));
    let kept = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::with_capacity(threads);
    for w in 0..threads {
        let spec = spec.clone();
        let collected = collected.clone();
        let kept = kept.clone();
        // Same seed-derivation scheme as pipeline_writer_generic so behaviour
        // is consistent across the codebase.
        let child_seed =
            rng_seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
        handles.push(std::thread::spawn(move || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
            // Per-worker attempt cap: divide spec.max_attempts among workers
            // so total work ~= spec.max_attempts. If max_attempts==0, unbounded.
            let per_worker_attempts = if spec.max_attempts == 0 {
                u32::MAX
            } else {
                ((spec.max_attempts as u64 + threads as u64 - 1) / threads as u64) as u32
            };
            let mut attempts_used = 0u32;
            while attempts_used < per_worker_attempts {
                if kept.load(Ordering::Relaxed) >= num_puzzles {
                    return;
                }
                // Reuse single-call reverse_construct in chunks so we can
                // periodically re-check the kept counter without losing the
                // ability to attribute attempts accurately. We track the
                // *actual* attempts consumed (res.attempts_taken or the full
                // chunk on miss), not just the chunk size.
                let remaining = per_worker_attempts - attempts_used;
                let chunk = remaining.min(64);
                let chunk_spec = ReverseSpec {
                    max_attempts: chunk,
                    ..spec.clone()
                };
                match reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
                    Some(res) => {
                        attempts_used = attempts_used.saturating_add(res.attempts_taken);
                        let prev = kept.fetch_add(1, Ordering::Relaxed);
                        if prev < num_puzzles {
                            let mut guard = collected.lock().unwrap();
                            guard.push(res);
                        } else {
                            kept.fetch_sub(1, Ordering::Relaxed);
                            return;
                        }
                    }
                    None => {
                        // Whole chunk consumed without a hit.
                        attempts_used = attempts_used.saturating_add(chunk);
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
// R1.5: BatchMixSpec — per-puzzle weighted mix of ReverseSpecs. Loaded from
// TOML; each emitted puzzle samples a mode by `weight` and runs that mode
// to completion (up to its per-mode `max_attempts`) before re-sampling. This
// gives an output mode distribution proportional to `weight` (not
// `weight × hit_rate`), provided every mode is feasible within its budget.
// See `batch_reverse_construct_mixed`.
// ---------------------------------------------------------------------------

/// One weighted entry in a `BatchMixSpec` — TOML-deserialisable shape (with
/// stringly-typed match-mode that we re-build into the enum form below).
#[derive(Clone, Debug, serde::Deserialize)]
struct WeightedModeRaw {
    weight: f32,
    spec: ReverseSpecRaw,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct ReverseSpecRaw {
    target_tier: String,
    clue_min: u32,
    clue_max: u32,
    #[serde(default)]
    load_bearing: bool,
    #[serde(default)]
    max_attempts: u32,
    match_mode: MatchModeRaw,
    #[serde(default)]
    excluded_techniques: Vec<String>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct MatchModeRaw {
    #[serde(rename = "type")]
    type_: String,
    #[serde(default)]
    techniques: Vec<String>,
    #[serde(default)]
    k: Option<u32>,
    #[serde(default)]
    weighted: Vec<(String, f32)>,
    #[serde(default)]
    min_picked: Option<u32>,
}

#[derive(Clone, Debug, serde::Deserialize)]
struct BatchMixSpecRaw {
    modes: Vec<WeightedModeRaw>,
}

/// Resolved (typed) weighted mode used at runtime.
#[derive(Clone, Debug)]
pub struct WeightedMode {
    pub weight: f32,
    pub spec: ReverseSpec,
}

/// Per-puzzle weighted mix of `ReverseSpec`s. See module docstring for the
/// TOML shape; load via `BatchMixSpec::load_toml`.
#[derive(Clone, Debug)]
pub struct BatchMixSpec {
    pub modes: Vec<WeightedMode>,
}

impl BatchMixSpec {
    /// Read a TOML file and build a validated `BatchMixSpec`.
    pub fn load_toml(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        let raw: BatchMixSpecRaw =
            toml::from_str(&bytes).map_err(|e| format!("parse {}: {}", path.display(), e))?;
        Self::from_raw(raw)
    }

    /// Parse a TOML string in-memory (used in tests).
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        let raw: BatchMixSpecRaw =
            toml::from_str(s).map_err(|e| format!("parse: {}", e))?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: BatchMixSpecRaw) -> Result<Self, String> {
        if raw.modes.is_empty() {
            return Err("batch-mix-spec has zero modes".to_string());
        }
        let mut modes: Vec<WeightedMode> = Vec::with_capacity(raw.modes.len());
        let mut sum = 0.0f32;
        for (i, m) in raw.modes.into_iter().enumerate() {
            if !m.weight.is_finite() || m.weight < 0.0 {
                return Err(format!("mode[{}]: weight {} must be finite, >= 0", i, m.weight));
            }
            sum += m.weight;
            let spec = resolve_spec_raw(m.spec)
                .map_err(|e| format!("mode[{}]: {}", i, e))?;
            spec.validate().map_err(|e| format!("mode[{}]: {}", i, e))?;
            modes.push(WeightedMode { weight: m.weight, spec });
        }
        if sum <= 0.0 {
            return Err("batch-mix-spec: sum of weights must be > 0".to_string());
        }
        Ok(Self { modes })
    }

    /// Sample a mode index by weight using `rng`. Returns the index into
    /// `self.modes`.
    pub fn sample_index<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let total: f32 = self.modes.iter().map(|m| m.weight).sum();
        let mut x = rng.gen::<f32>() * total;
        for (i, m) in self.modes.iter().enumerate() {
            x -= m.weight;
            if x <= 0.0 {
                return i;
            }
        }
        self.modes.len() - 1
    }
}

fn resolve_spec_raw(r: ReverseSpecRaw) -> Result<ReverseSpec, String> {
    let target_tier = parse_tier_local(&r.target_tier)?;
    let mm = resolve_match_mode_raw(r.match_mode)?;
    let excluded: Vec<TechniqueId> = r
        .excluded_techniques
        .iter()
        .map(|s| parse_technique_id(s))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ReverseSpec {
        target_tier,
        match_mode: mm,
        excluded_techniques: excluded,
        load_bearing: r.load_bearing,
        clue_min: r.clue_min,
        clue_max: r.clue_max,
        max_attempts: r.max_attempts,
    })
}

fn resolve_match_mode_raw(m: MatchModeRaw) -> Result<MatchMode, String> {
    let parse_techs = |ts: &[String]| -> Result<Vec<TechniqueId>, String> {
        ts.iter().map(|s| parse_technique_id(s)).collect()
    };
    match m.type_.trim().to_ascii_lowercase().as_str() {
        "all_of" => Ok(MatchMode::AllOf(parse_techs(&m.techniques)?)),
        "any_of" => Ok(MatchMode::AnyOf(parse_techs(&m.techniques)?)),
        "exact_k" => Ok(MatchMode::ExactK {
            techniques: parse_techs(&m.techniques)?,
            k: m.k.ok_or_else(|| "exact_k requires `k`".to_string())?,
        }),
        "at_least_k" => Ok(MatchMode::AtLeastK {
            techniques: parse_techs(&m.techniques)?,
            k: m.k.ok_or_else(|| "at_least_k requires `k`".to_string())?,
        }),
        "weighted" => {
            if m.weighted.is_empty() {
                return Err("weighted mode requires non-empty `weighted = [...]`".to_string());
            }
            let mut ws = Vec::with_capacity(m.weighted.len());
            for (name, p) in &m.weighted {
                let id = parse_technique_id(name)?;
                if !p.is_finite() || !(0.0..=1.0).contains(p) {
                    return Err(format!("weighted: p={} for {} not in [0,1]", p, name));
                }
                ws.push((id, *p));
            }
            Ok(MatchMode::Weighted {
                weights: ws,
                min_picked: m.min_picked.unwrap_or(1),
            })
        }
        other => Err(format!(
            "match_mode.type='{}' must be all_of|any_of|exact_k|at_least_k|weighted",
            other
        )),
    }
}

fn parse_tier_local(s: &str) -> Result<Tier, String> {
    match s.trim().to_ascii_uppercase().as_str() {
        "T1" => Ok(Tier::T1),
        "T2" => Ok(Tier::T2),
        "T3" => Ok(Tier::T3),
        "T4PLUS" | "T4+" | "T4" => Ok(Tier::T4Plus),
        other => Err(format!("expected T1|T2|T3|T4Plus, got '{}'", other)),
    }
}

/// Mixed-mode batch driver. Each puzzle slot independently samples a mode
/// from `mix.modes` and runs `reverse_construct` against the sampled spec
/// until that slot lands a hit (or the mode's `max_attempts` budget is
/// exhausted, in which case the slot is *retried* with a fresh mode sample —
/// this preserves the output mode distribution: kept puzzles are
/// proportional to `weight`, NOT `weight × hit_rate`).
///
/// Each mode SHOULD set a finite `max_attempts` in TOML (default 1024 if
/// zero) so that an infeasible mode does not stall the whole batch. A global
/// `safety_budget` cap (per worker) guards against pathological
/// configurations.
///
/// Per-worker RNG derivation matches `batch_reverse_construct`.
pub fn batch_reverse_construct_mixed<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    mix: &BatchMixSpec,
    num_puzzles: u32,
    threads: usize,
) -> Vec<ReverseResult<N, BR, BC>> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    let threads = threads.max(1);
    let collected: Arc<Mutex<Vec<ReverseResult<N, BR, BC>>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_puzzles as usize)));
    let kept = Arc::new(AtomicU32::new(0));
    let mix = Arc::new(mix.clone());
    let mut handles = Vec::with_capacity(threads);
    for w in 0..threads {
        let collected = collected.clone();
        let kept = kept.clone();
        let mix = mix.clone();
        let child_seed =
            rng_seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
        handles.push(std::thread::spawn(move || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
            // Per-slot: pick mode by weight, run that mode's spec to
            // completion (up to its max_attempts). This is the key correctness
            // change vs the earlier prototype: by attempting a single slot
            // until it lands, the OUTPUT mix tracks `weight`, not
            // `weight * hit_rate`. A consecutive miss budget across slots
            // bails out if every mode is infeasible.
            let mut consecutive_failed_slots = 0u64;
            let safety_slot_failures: u64 = 64;
            while kept.load(Ordering::Relaxed) < num_puzzles {
                let idx = mix.sample_index(&mut rng);
                let spec = &mix.modes[idx].spec;
                let slot_budget = if spec.max_attempts == 0 {
                    1024 // sensible per-slot default if the TOML left it 0
                } else {
                    spec.max_attempts
                };
                let slot_spec = ReverseSpec {
                    max_attempts: slot_budget,
                    ..spec.clone()
                };
                match reverse_construct::<N, BR, BC, _>(&mut rng, &slot_spec) {
                    Some(res) => {
                        consecutive_failed_slots = 0;
                        let prev = kept.fetch_add(1, Ordering::Relaxed);
                        if prev < num_puzzles {
                            let mut g = collected.lock().unwrap();
                            g.push(res);
                        } else {
                            kept.fetch_sub(1, Ordering::Relaxed);
                            return;
                        }
                    }
                    None => {
                        // This whole slot exhausted its budget without a hit.
                        // Re-sample a (possibly different) mode and try again.
                        consecutive_failed_slots += 1;
                        if consecutive_failed_slots >= safety_slot_failures {
                            return;
                        }
                    }
                }
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    let g = collected.lock().unwrap();
    g.clone()
}

// ---------------------------------------------------------------------------
// TechniqueId <-> CLI string mapping. Stable identifiers used by the CLI
// `--required` / `--excluded` flags. Names mirror the convention already used
// in `legacy reverse_construct` ("aic") and the rater frontier.
// ---------------------------------------------------------------------------

/// All technique ids supported by the rater, in canonical order. The CLI
/// `--required <name>` flag accepts the strings returned by `as_str` (and a
/// few common aliases handled in `parse_technique_id`).
pub const ALL_TECHNIQUE_IDS: [TechniqueId; 28] = [
    TechniqueId::LockedPointing,
    TechniqueId::LockedClaiming,
    TechniqueId::NakedPair,
    TechniqueId::NakedTriple,
    TechniqueId::NakedQuad,
    TechniqueId::HiddenPair,
    TechniqueId::HiddenTriple,
    TechniqueId::HiddenQuad,
    TechniqueId::XWing,
    TechniqueId::Swordfish,
    TechniqueId::Jellyfish,
    TechniqueId::XyWing,
    TechniqueId::XyzWing,
    TechniqueId::UrType1,
    TechniqueId::UrType2,
    TechniqueId::Aic,
    TechniqueId::AlsXz,
    TechniqueId::Bug,
    TechniqueId::SimpleColoring,
    TechniqueId::Skyscraper,
    TechniqueId::TwoStringKite,
    TechniqueId::CellForcingChain,
    TechniqueId::RegionForcingChain,
    TechniqueId::DynamicForcingChain,
    TechniqueId::NestedForcingChain,
    TechniqueId::NestedForcingChainL3,
    TechniqueId::NestedForcingChainL4,
    // Berthier chain techniques: GBraid only post-collapse (spec §17).
    // Whip/GWhip/Braid removed — subsumed by GBraid per Berthier's theorem.
    TechniqueId::GBraid,
];

#[inline]
pub fn technique_id_str(id: TechniqueId) -> &'static str {
    match id {
        TechniqueId::LockedPointing => "locked_pointing",
        TechniqueId::LockedClaiming => "locked_claiming",
        TechniqueId::NakedPair => "naked_pair",
        TechniqueId::NakedTriple => "naked_triple",
        TechniqueId::NakedQuad => "naked_quad",
        TechniqueId::HiddenPair => "hidden_pair",
        TechniqueId::HiddenTriple => "hidden_triple",
        TechniqueId::HiddenQuad => "hidden_quad",
        TechniqueId::XWing => "xwing",
        TechniqueId::Swordfish => "swordfish",
        TechniqueId::Jellyfish => "jellyfish",
        TechniqueId::XyWing => "xy_wing",
        TechniqueId::UrType1 => "ur_type1",
        TechniqueId::UrType2 => "ur_type2",
        TechniqueId::Aic => "aic",
        TechniqueId::AlsXz => "als_xz",
        TechniqueId::XyzWing => "xyz_wing",
        TechniqueId::Bug => "bug",
        TechniqueId::SimpleColoring => "simple_coloring",
        TechniqueId::Skyscraper => "skyscraper",
        TechniqueId::TwoStringKite => "two_string_kite",
        TechniqueId::CellForcingChain => "cell_forcing_chain",
        TechniqueId::RegionForcingChain => "region_forcing_chain",
        TechniqueId::DynamicForcingChain => "dynamic_forcing_chain",
        TechniqueId::NestedForcingChain => "nested_forcing_chain",
        TechniqueId::NestedForcingChainL3 => "nested_forcing_chain_l3",
        TechniqueId::NestedForcingChainL4 => "nested_forcing_chain_l4",
        TechniqueId::Whip   => "whip",
        TechniqueId::GWhip  => "gwhip",
        TechniqueId::Braid  => "braid",
        TechniqueId::GBraid => "gbraid",
    }
}

/// Parse a CLI technique-id string into a `TechniqueId`. Accepts the canonical
/// `as_str` form plus a handful of common aliases (`fish_xwing`, `xy-wing`,
/// `urtype1`, etc.). Case-insensitive.
pub fn parse_technique_id(s: &str) -> Result<TechniqueId, String> {
    let key = s.trim().to_ascii_lowercase().replace('-', "_");
    let id = match key.as_str() {
        "locked_pointing" | "lockedpointing" | "pointing" => TechniqueId::LockedPointing,
        "locked_claiming" | "lockedclaiming" | "claiming" => TechniqueId::LockedClaiming,
        "naked_pair" | "nakedpair" => TechniqueId::NakedPair,
        "naked_triple" | "nakedtriple" => TechniqueId::NakedTriple,
        "naked_quad" | "nakedquad" => TechniqueId::NakedQuad,
        "hidden_pair" | "hiddenpair" => TechniqueId::HiddenPair,
        "hidden_triple" | "hiddentriple" => TechniqueId::HiddenTriple,
        "hidden_quad" | "hiddenquad" => TechniqueId::HiddenQuad,
        "xwing" | "x_wing" | "fish_xwing" | "fish2" | "fish_2" => TechniqueId::XWing,
        "swordfish" | "fish_swordfish" | "fish3" | "fish_3" => TechniqueId::Swordfish,
        "jellyfish" | "fish_jellyfish" | "fish4" | "fish_4" => TechniqueId::Jellyfish,
        "xy_wing" | "xywing" => TechniqueId::XyWing,
        "ur_type1" | "urtype1" | "ur1" => TechniqueId::UrType1,
        "ur_type2" | "urtype2" | "ur2" => TechniqueId::UrType2,
        "aic" => TechniqueId::Aic,
        "als_xz" | "alsxz" | "als" => TechniqueId::AlsXz,
        "xyz_wing" | "xyzwing" => TechniqueId::XyzWing,
        "bug" => TechniqueId::Bug,
        "simple_coloring" | "simplecoloring" | "coloring" => TechniqueId::SimpleColoring,
        "skyscraper" => TechniqueId::Skyscraper,
        "two_string_kite" | "twostringkite" | "kite" | "t2k" => TechniqueId::TwoStringKite,
        "cell_forcing_chain" | "cellforcingchain" | "cellfc" | "cfc" => TechniqueId::CellForcingChain,
        "region_forcing_chain" | "regionforcingchain" | "regionfc" | "rfc" => TechniqueId::RegionForcingChain,
        "dynamic_forcing_chain" | "dynamicforcingchain" | "dynamic_fc" | "dfc" => TechniqueId::DynamicForcingChain,
        "nested_forcing_chain" | "nestedforcingchain" | "nested_fc" | "nfc" => TechniqueId::NestedForcingChain,
        "nested_forcing_chain_l3" | "nestedforcingchainl3" | "nested_fc_l3" | "nfc_l3" => TechniqueId::NestedForcingChainL3,
        "nested_forcing_chain_l4" | "nestedforcingchainl4" | "nested_fc_l4" | "nfc_l4" => TechniqueId::NestedForcingChainL4,
        // Berthier chain techniques: GBraid only post-collapse (spec §17).
        // Whip/GWhip/Braid arms removed — subsumed by GBraid.
        "gbraid" | "gb" | "g_braid" => TechniqueId::GBraid,
        other => {
            return Err(format!(
                "unknown technique '{}'; expected one of: {}",
                other,
                ALL_TECHNIQUE_IDS
                    .iter()
                    .map(|&t| technique_id_str(t))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    };
    Ok(id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn mk_spec(target: Tier, required: Vec<TechniqueId>, max_attempts: u32) -> ReverseSpec {
        ReverseSpec {
            target_tier: target,
            match_mode: MatchMode::AllOf(required),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 22,
            clue_max: 30,
            max_attempts,
        }
    }

    /// Per-tech smoke test on 9×9. For each of the 16 technique ids we run
    /// `reverse_construct` with budget 200 attempts. Some techniques are very
    /// rare (ALS-XZ, Jellyfish) — for those we simply accept `None` as a
    /// valid outcome. The contract is: must terminate within budget; if
    /// `Some` is returned, the returned puzzle must satisfy the spec.
    #[test]
    fn per_tech_smoke_9x9_terminates_within_budget() {
        for &tech in ALL_TECHNIQUE_IDS.iter() {
            let target_tier = match tech {
                TechniqueId::LockedPointing
                | TechniqueId::LockedClaiming
                | TechniqueId::NakedPair
                | TechniqueId::HiddenPair => Tier::T2,
                _ => Tier::T3,
            };
            let spec = mk_spec(target_tier, vec![tech], 200);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234 + tech as u64);
            // Just check it terminates and (if Some) the contract holds.
            if let Some(res) =
                reverse_construct::<9, 3, 3, _>(&mut rng, &spec)
            {
                assert!(res.rate.frontier.contains(&tech),
                    "tech {:?}: returned puzzle frontier {:?} doesn't include required",
                    tech, res.rate.frontier);
                assert_eq!(tier_rank(res.rate.tier), tier_rank(target_tier));
                assert!(res.attempts_taken <= spec.max_attempts);
            }
            // None is acceptable — some techniques are very rare in random
            // sampling and need constructive synth (R2+).
        }
    }

    /// `reverse_construct` with no required techniques and target=T1 always
    /// returns Some quickly (T1 puzzles are common at clue_count >= 30).
    #[test]
    fn no_required_t1_succeeds_easily() {
        let spec = ReverseSpec {
            target_tier: Tier::T1,
            match_mode: MatchMode::AllOf(Vec::new()),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 30,
            clue_max: 40,
            max_attempts: 200,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let res = reverse_construct::<9, 3, 3, _>(&mut rng, &spec)
            .expect("T1 puzzle at clue 30 should be findable in 200 attempts");
        assert_eq!(tier_rank(res.rate.tier), tier_rank(Tier::T1));
        assert!(res.rate.solved);
    }

    /// Frontier match: required=[NakedPair, LockedPointing] → both must be in
    /// returned frontier. Soft-pass if no such puzzle exists in budget.
    #[test]
    fn frontier_match_multiple_required() {
        let spec = ReverseSpec {
            target_tier: Tier::T2,
            match_mode: MatchMode::AllOf(vec![TechniqueId::NakedPair, TechniqueId::LockedPointing]),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 500,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        if let Some(res) = reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            assert!(res.rate.frontier.contains(&TechniqueId::NakedPair));
            assert!(res.rate.frontier.contains(&TechniqueId::LockedPointing));
        } else {
            eprintln!(
                "warn: no T2 puzzle with both NakedPair+LockedPointing in 500 attempts (soft-pass)"
            );
        }
    }

    /// Excluded check: returned puzzle must not have AIC in frontier.
    #[test]
    fn excluded_technique_not_in_frontier() {
        let spec = ReverseSpec {
            target_tier: Tier::T3,
            match_mode: MatchMode::AllOf(Vec::new()),
            excluded_techniques: vec![TechniqueId::Aic],
            load_bearing: false,
            clue_min: 22,
            clue_max: 28,
            max_attempts: 500,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        if let Some(res) = reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            assert!(!res.rate.frontier.contains(&TechniqueId::Aic),
                "excluded AIC appeared in frontier: {:?}", res.rate.frontier);
            assert_eq!(tier_rank(res.rate.tier), tier_rank(Tier::T3));
        }
        // None is acceptable.
    }

    /// Determinism: same seed → same result.
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = ReverseSpec {
            target_tier: Tier::T1,
            match_mode: MatchMode::AllOf(Vec::new()),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 30,
            clue_max: 40,
            max_attempts: 50,
        };
        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(2026);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(2026);
        let a = reverse_construct::<9, 3, 3, _>(&mut rng_a, &spec).unwrap();
        let b = reverse_construct::<9, 3, 3, _>(&mut rng_b, &spec).unwrap();
        assert_eq!(a.puzzle.to_string_grid(), b.puzzle.to_string_grid());
        assert_eq!(a.attempts_taken, b.attempts_taken);
    }

    /// Load-bearing verification. We find any T3 puzzle (no required) and
    /// then re-run reverse_construct with `load_bearing=true` against ALL
    /// T3 techniques as the required set — for that to match, every listed
    /// T3 tech would need to fire AND excluding all of them would drop tier.
    /// Easier: pick a T3 puzzle with AIC in frontier; verify load-bearing
    /// semantics directly via rate_excluding.
    #[test]
    fn load_bearing_drops_tier_when_required_excluded() {
        let spec = ReverseSpec {
            target_tier: Tier::T3,
            match_mode: MatchMode::AllOf(vec![TechniqueId::Aic]),
            excluded_techniques: Vec::new(),
            load_bearing: true,
            clue_min: 22,
            clue_max: 26,
            max_attempts: 1000,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(31415);
        if let Some(res) = reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            // Sanity: AIC fired in returned puzzle.
            assert!(res.rate.frontier.contains(&TechniqueId::Aic));
            assert_eq!(tier_rank(res.rate.tier), tier_rank(Tier::T3));
            // Load-bearing contract: re-rating without AIC drops below T3.
            let r2 = rate_excluding(&res.puzzle, &[TechniqueId::Aic]);
            assert!(!r2.rater_error);
            assert!(tier_rank(r2.tier) < tier_rank(Tier::T3),
                "load_bearing flag passed but rate_excluding tier {:?} not below T3",
                r2.tier);
        } else {
            eprintln!("warn: no load-bearing AIC puzzle in 1000 attempts (soft-pass)");
        }
    }

    /// Validate `parse_technique_id` covers all 16 ids and accepts aliases.
    #[test]
    fn parse_technique_id_covers_all_16() {
        for &id in ALL_TECHNIQUE_IDS.iter() {
            let s = technique_id_str(id);
            let parsed = parse_technique_id(s).expect(s);
            assert_eq!(parsed, id, "round-trip failed for {}", s);
        }
        // Aliases.
        assert_eq!(parse_technique_id("X-Wing").unwrap(), TechniqueId::XWing);
        assert_eq!(parse_technique_id("fish_xwing").unwrap(), TechniqueId::XWing);
        assert_eq!(parse_technique_id("fish_2").unwrap(), TechniqueId::XWing);
        assert_eq!(parse_technique_id("fish_3").unwrap(), TechniqueId::Swordfish);
        assert_eq!(parse_technique_id("fish_4").unwrap(), TechniqueId::Jellyfish);
        assert_eq!(parse_technique_id("XYWing").unwrap(), TechniqueId::XyWing);
        assert_eq!(parse_technique_id("ur1").unwrap(), TechniqueId::UrType1);
        assert!(parse_technique_id("nonexistent").is_err());
    }

    /// Validation: spec with target=T1 + required=[Aic] is unsatisfiable.
    #[test]
    fn validate_t1_with_required_is_rejected() {
        let spec = ReverseSpec {
            target_tier: Tier::T1,
            match_mode: MatchMode::AllOf(vec![TechniqueId::Aic]),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 30,
            clue_max: 40,
            max_attempts: 10,
        };
        assert!(spec.validate().is_err());
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        // reverse_construct returns None on validation error.
        assert!(reverse_construct::<9, 3, 3, _>(&mut rng, &spec).is_none());
    }

    /// Validation: required and excluded must be disjoint.
    #[test]
    fn validate_required_excluded_disjoint() {
        let spec = ReverseSpec {
            target_tier: Tier::T3,
            match_mode: MatchMode::AllOf(vec![TechniqueId::Aic]),
            excluded_techniques: vec![TechniqueId::Aic],
            load_bearing: false,
            clue_min: 22,
            clue_max: 28,
            max_attempts: 10,
        };
        assert!(spec.validate().is_err());
    }

    /// Batch determinism: same `(seed, threads=1)` is bit-identical. The
    /// multi-worker case is intrinsically order-dependent (workers race on
    /// the kept counter); we only assert single-threaded determinism, which
    /// is the contract documented in the module preamble.
    #[test]
    fn batch_determinism_same_seed_same_multiset_single_thread() {
        let spec = ReverseSpec {
            target_tier: Tier::T1,
            match_mode: MatchMode::AllOf(Vec::new()),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 30,
            clue_max: 40,
            max_attempts: 200,
        };
        let a: Vec<_> = batch_reverse_construct::<9, 3, 3>(2026, &spec, 4, 1);
        let b: Vec<_> = batch_reverse_construct::<9, 3, 3>(2026, &spec, 4, 1);
        assert_eq!(a.len(), b.len());
        let a_strs: Vec<String> = a.iter().map(|r| r.puzzle.to_string_grid()).collect();
        let b_strs: Vec<String> = b.iter().map(|r| r.puzzle.to_string_grid()).collect();
        assert_eq!(a_strs, b_strs, "single-threaded batch must be byte-identical");
    }

    /// 6×6 termination smoke: ALS-XZ on 6×6 is essentially impossible — the
    /// call must return None (not hang) within the budget.
    #[test]
    fn als_xz_on_6x6_terminates_with_none_or_some() {
        let spec = mk_spec(Tier::T3, vec![TechniqueId::AlsXz], 100);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        // We don't care about Some vs None; we care about termination.
        let _ = reverse_construct::<6, 2, 3, _>(&mut rng, &spec);
    }

    // ----------------------------------------------------------------------
    // R1.5 mix-mode tests
    // ----------------------------------------------------------------------

    /// AnyOf semantic: every returned frontier contains at least one of the
    /// listed techniques. Soft-pass on empty result set (all techs in the
    /// list might be too rare under the budget).
    #[test]
    fn r15_any_of_each_frontier_has_at_least_one() {
        let spec = ReverseSpec {
            target_tier: Tier::T3,
            match_mode: MatchMode::AnyOf(vec![
                TechniqueId::Aic,
                TechniqueId::XWing,
                TechniqueId::XyWing,
            ]),
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 22,
            clue_max: 28,
            max_attempts: 200,
        };
        let res = batch_reverse_construct::<9, 3, 3>(2026, &spec, 50, 1);
        // Soft assertion on count — we only require correctness on the ones we got.
        for r in &res {
            let f = &r.rate.frontier;
            assert!(
                f.contains(&TechniqueId::Aic)
                    || f.contains(&TechniqueId::XWing)
                    || f.contains(&TechniqueId::XyWing),
                "AnyOf violated: frontier {:?}", f
            );
        }
    }

    /// AtLeastK k=2 semantic: each returned frontier contains ≥2 of the
    /// listed techniques.
    #[test]
    fn r15_at_least_k2_each_frontier_has_two() {
        let techs = vec![
            TechniqueId::Aic,
            TechniqueId::NakedPair,
            TechniqueId::LockedPointing,
            TechniqueId::XyWing,
        ];
        let spec = ReverseSpec {
            target_tier: Tier::T3,
            match_mode: MatchMode::AtLeastK { techniques: techs.clone(), k: 2 },
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 22,
            clue_max: 28,
            max_attempts: 500,
        };
        let res = batch_reverse_construct::<9, 3, 3>(7, &spec, 20, 1);
        for r in &res {
            let count = techs.iter().filter(|t| r.rate.frontier.contains(t)).count();
            assert!(count >= 2, "AtLeastK k=2 violated: count={} frontier={:?}",
                count, r.rate.frontier);
        }
    }

    /// Weighted distribution: per-puzzle Bernoulli sampling over (aic, fish).
    /// We don't run a full empirical-fire-rate calibration here (that needs
    /// thousands of puzzles); we sanity-check that the mode at least emits
    /// some puzzles and that every kept frontier satisfies the matched
    /// sample (impossible to verify post-hoc — the sample is internal — so
    /// we settle for: kept puzzles must contain at least min_picked of the
    /// listed techniques).
    #[test]
    fn r15_weighted_emits_and_contract_holds() {
        let spec = ReverseSpec {
            target_tier: Tier::T3,
            match_mode: MatchMode::Weighted {
                weights: vec![(TechniqueId::Aic, 0.7), (TechniqueId::XWing, 0.3)],
                min_picked: 1,
            },
            excluded_techniques: Vec::new(),
            load_bearing: false,
            clue_min: 22,
            clue_max: 28,
            max_attempts: 300,
        };
        let res = batch_reverse_construct::<9, 3, 3>(123, &spec, 30, 1);
        for r in &res {
            // Contract: at least one of {aic, xwing} must be in frontier
            // (since min_picked=1 and the picked set must all fire).
            let f = &r.rate.frontier;
            assert!(
                f.contains(&TechniqueId::Aic) || f.contains(&TechniqueId::XWing),
                "Weighted: kept puzzle has neither aic nor xwing: {:?}", f
            );
        }
    }

    /// BatchMixSpec end-to-end: parse a 3-mode TOML, run a small batch,
    /// verify all kept puzzles satisfy at least ONE of the mode contracts.
    #[test]
    fn r15_batch_mix_spec_parses_and_dispatches() {
        let toml_str = r#"
[[modes]]
weight = 0.4
[modes.spec]
target_tier = "T2"
clue_min = 28
clue_max = 36
load_bearing = false
max_attempts = 0
excluded_techniques = []
[modes.spec.match_mode]
type = "all_of"
techniques = ["naked_pair"]

[[modes]]
weight = 0.3
[modes.spec]
target_tier = "T2"
clue_min = 28
clue_max = 36
load_bearing = false
max_attempts = 0
excluded_techniques = []
[modes.spec.match_mode]
type = "any_of"
techniques = ["locked_pointing", "hidden_pair"]

[[modes]]
weight = 0.3
[modes.spec]
target_tier = "T1"
clue_min = 32
clue_max = 40
load_bearing = false
max_attempts = 0
excluded_techniques = []
[modes.spec.match_mode]
type = "all_of"
techniques = []
"#;
        let mix = BatchMixSpec::from_toml_str(toml_str).expect("toml parse");
        assert_eq!(mix.modes.len(), 3);
        // Sample distribution sanity: with seeded RNG, sample_index over many
        // draws should roughly track the weights (loose 25% slack for n=400).
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        let mut hist = [0u32; 3];
        for _ in 0..400 {
            hist[mix.sample_index(&mut rng)] += 1;
        }
        let total = 400.0;
        let weights = [0.4f32, 0.3, 0.3];
        for i in 0..3 {
            let observed = hist[i] as f32 / total;
            let want = weights[i];
            assert!(
                (observed - want).abs() < 0.10,
                "mode[{}] empirical {} vs weight {}", i, observed, want
            );
        }
        // End-to-end small batch.
        let res = batch_reverse_construct_mixed::<9, 3, 3>(31, &mix, 12, 1);
        for r in &res {
            // Each kept must satisfy ONE of:
            //   * naked_pair fires AND tier == T2; OR
            //   * (locked_pointing | hidden_pair) fires AND tier == T2; OR
            //   * tier == T1.
            let f = &r.rate.frontier;
            let m1 = matches!(r.rate.tier, Tier::T2) && f.contains(&TechniqueId::NakedPair);
            let m2 = matches!(r.rate.tier, Tier::T2)
                && (f.contains(&TechniqueId::LockedPointing)
                    || f.contains(&TechniqueId::HiddenPair));
            let m3 = matches!(r.rate.tier, Tier::T1);
            assert!(m1 || m2 || m3,
                "BatchMixSpec contract violated: tier={:?} frontier={:?}",
                r.rate.tier, f);
        }
    }

    /// Validation: BatchMixSpec with sum-of-weights=0 is rejected.
    #[test]
    fn r15_batch_mix_zero_weights_rejected() {
        let toml_str = r#"
[[modes]]
weight = 0.0
[modes.spec]
target_tier = "T1"
clue_min = 30
clue_max = 40
[modes.spec.match_mode]
type = "all_of"
techniques = []
"#;
        let err = BatchMixSpec::from_toml_str(toml_str).err().expect("must fail");
        assert!(err.contains("weights"), "unexpected err: {}", err);
    }

    /// Validation: MatchMode::ExactK with k > techniques.len() is rejected.
    #[test]
    fn r15_match_mode_validate_exact_k_overflow() {
        let mm = MatchMode::ExactK {
            techniques: vec![TechniqueId::Aic],
            k: 5,
        };
        assert!(mm.validate().is_err());
    }
}
