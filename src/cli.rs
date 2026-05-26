//! CLI: solve | rate | gen | gen-dataset.

// Opt #3a: mimalloc as global allocator. Faster small-allocation throughput
// (no per-thread arena lock contention) + better fragmentation vs the system
// allocator on macOS. The SHC port hot path allocates short-lived Vec<u16>
// chains + HashMap dedup entries per probe, so allocator quality matters.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::{Parser, Subcommand};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;
use std::path::PathBuf;
use sudoku_rs_core::pipeline_writer_generic::{
    parse_clue_range, parse_size, parse_tier, run as run_generic_pipeline, GenericPipelineConfig,
};
use sudoku_rs_core::rerate_generic::{run as run_rerate_generic, RerateGenericConfig};
use sudoku_rs_core::generic::reverse_construct::{
    batch_reverse_construct as generic_batch_reverse,
    batch_reverse_construct_mixed as generic_batch_reverse_mixed,
    parse_technique_id as parse_generic_tech, technique_id_str as generic_tech_str,
    BatchMixSpec, MatchMode, ReverseSpec as GenericReverseSpec, ALL_TECHNIQUE_IDS,
};
use sudoku_rs_core::generic::aic_reverse::{
    batch_aic_reverse_construct as generic_batch_aic_reverse, AicReverseSpec,
};
use sudoku_rs_core::generic::fish_reverse::{
    batch_fish_reverse_construct as generic_batch_fish_reverse, FishReverseSpec,
};
use sudoku_rs_core::generic::ur_type2_reverse::{
    batch_ur_type2_reverse_construct as generic_batch_ur_type2_reverse, UrType2ReverseSpec,
};
use sudoku_rs_core::generic::als_xz_reverse::{
    batch_als_xz_reverse_construct as generic_batch_als_xz_reverse, AlsXzReverseSpec,
};
use sudoku_rs_core::generic::naked_quad_reverse::{
    batch_naked_quad_reverse_construct as generic_batch_naked_quad_reverse, NakedQuadReverseSpec,
};
use sudoku_rs_core::generic::hidden_quad_reverse::{
    batch_hidden_quad_reverse_construct as generic_batch_hidden_quad_reverse, HiddenQuadReverseSpec,
};
use sudoku_rs_core::generic::nested_aic_reverse::{
    batch_construct as nested_aic_batch_construct, NestedAicConfig,
};
use sudoku_rs_core::generic::nested_fc_reverse::{
    batch_nested_fc_reverse_construct, NestedFcReverseSpec,
};
use sudoku_rs_core::generic::rfc_reverse::{
    batch_rfc_reverse_construct, RfcReverseSpec,
};
use sudoku_rs_core::generic::dfc_reverse::{
    batch_dfc_reverse_construct as generic_batch_dfc_reverse, DfcReverseSpec,
};
use sudoku_rs_core::generic::cfc_reverse::{
    batch_cfc_reverse_construct as generic_batch_cfc_reverse, CfcReverseSpec,
};
use sudoku_rs_core::generic::techniques::TechniqueId as GTechniqueId;
use sudoku_rs_core::pipeline_writer_generic::{make_record as make_generic_record, write_generic_parquet};
use sudoku_rs_core::generic::grid::Grid as GGrid;
use sudoku_rs_core::generic::generator::{gen_unique_puzzle as g_gen_unique_puzzle, GenConfig as GGenConfig};
use sudoku_rs_core::generic::search::solve_unique as g_solve_unique;
use sudoku_rs_core::generic::rater::{
    rate_with_uniqueness, rate_with_uniqueness_mode, RateResult as GRateResult,
    SolverMode as GSolverMode,
};
use sudoku_rs_core::generic::techniques::Tier as GTier;
use sudoku_rs_core::io::{JsonlSink, NextLossy, OrderingSink, PuzzleSource, RatedPuzzle, RawPuzzle, TextFileSource};
use rayon::prelude::*;

#[derive(Parser)]
#[command(name = "sudoku_rs_core")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Solve { puzzle: String },
    Rate { puzzle: String },
    Gen {
        #[arg(default_value_t = 30)]
        clues: u32,
        #[arg(short, long, default_value_t = 1)]
        n: u32,
        #[arg(short, long, default_value_t = 0)]
        seed: u64,
    },
    /// Generate a parquet dataset shard set + manifest.
    ///
    /// Two modes:
    ///   * Legacy (no `--size`): 9×9-only random / constrained-removal pipeline
    ///     writing sharded parquet under `--out-dir`. Backwards compatible.
    ///   * Generic (`--size NxBRxBC` provided): const-generic pipeline over
    ///     the size given, writing a single parquet at `--output`. Adds a
    ///     `size` column to the schema. Used by Path C P2 onward for
    ///     6×6 / 9×9 / 12×12 / 16×16 dataset generation.
    GenDataset {
        // ---- New (generic) mode flags ----
        /// Generic mode: grid size as `NxBRxBC` (e.g. `9x3x3`, `6x2x3`,
        /// `12x3x4`, `16x4x4`). Presence of this flag activates the generic
        /// pipeline. The `--num`, `--threads`, `--target-clues` and `--output`
        /// flags are also part of the generic mode.
        #[arg(long)]
        size: Option<String>,
        /// Generic mode: number of puzzles to generate. Mirrors legacy
        /// `--target-total`.
        #[arg(long)]
        num: Option<usize>,
        /// Generic mode: worker thread count. 0 = auto.
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Generic mode: clue-count band as `MIN..MAX` (inclusive).
        #[arg(long = "target-clues")]
        target_clues: Option<String>,
        /// Generic mode: output parquet file path.
        #[arg(long)]
        output: Option<PathBuf>,

        // ---- Legacy flags (unchanged) ----
        #[arg(long, default_value_t = 30)]
        n_clues: u32,
        #[arg(long, default_value_t = 1000)]
        target_total: usize,
        #[arg(long, default_value_t = 500)]
        flush_every: usize,
        #[arg(long, default_value_t = 100)]
        worker_buffer: usize,
        #[arg(long, default_value_t = 0)]
        num_workers: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        out_dir: Option<PathBuf>,
        #[arg(long, default_value_t = true)]
        drop_unresolved: bool,
        /// Per-tier targets, formatted "T1=100,T2=100,T3=0,T4=0".
        #[arg(long, default_value = "T1=1000,T2=0,T3=0,T4=0")]
        target_per_tier: String,
        /// DEPRECATED. Historical flag pointing at the external tdoku binary.
        /// Backtracker counts and uniqueness are now computed in-process by
        /// the Rust backtracker; this flag is accepted for backward
        /// compatibility but ignored (a warning is emitted).
        #[arg(long, hide = true)]
        tdoku_bin: Option<PathBuf>,
        /// Generation mode. `random` (default) = legacy random rejection
        /// sampling. `constrained-removal` = iterative cell-removal that
        /// guarantees every emitted puzzle satisfies `--target-tier` and
        /// `--min-chain-length`.
        #[arg(long, default_value = "random")]
        mode: String,
        /// Constrained-removal: minimum tier (T1|T2|T3|T4). Ignored in
        /// random mode.
        #[arg(long, default_value = "T1")]
        target_tier: String,
        /// Constrained-removal: minimum chain length (proxy: rater trace
        /// length). Ignored in random mode.
        #[arg(long, default_value_t = 0)]
        min_chain_length: i32,
        /// Constrained-removal: HARD cap on final clue count. Algorithm stops
        /// accepting more removals once `clue_count <= max_clues`; if the seed
        /// runs through its whole permutation without ever trimming below the
        /// cap, the puzzle is rejected and a fresh seed is tried (see
        /// `generator_constrained::try_generate`). Ignored in random mode.
        #[arg(long, default_value_t = 81)]
        max_clues: u32,
        /// Constrained-removal: hard floor on clue count (algorithm refuses
        /// to go below). Ignored in random mode.
        #[arg(long, default_value_t = 17)]
        min_clues: u32,
        /// Per-worker consecutive-failure cutoff. If a worker fails to produce
        /// a kept record this many times in a row, it exits as "stagnated";
        /// if ALL workers stagnate before `target_total` is reached, the run
        /// aborts with a non-zero exit code (prevents infinite spin on
        /// infeasible specs).
        #[arg(long, default_value_t = 200)]
        max_consecutive_failures: u32,
        /// Generic `--mode reverse`: required technique ids (repeatable). All
        /// listed techniques must fire in the rated frontier of every emitted
        /// puzzle. Names: locked_pointing, locked_claiming, naked_pair,
        /// naked_triple, naked_quad, hidden_pair, hidden_triple, hidden_quad,
        /// xwing, swordfish, jellyfish, xy_wing, ur_type1, ur_type2, aic,
        /// als_xz. Aliases: fish_xwing/fish_2, x-wing, etc. Ignored unless
        /// `--mode reverse`.
        #[arg(long = "required")]
        required: Vec<String>,
        /// Generic `--mode reverse`: excluded technique ids (repeatable).
        /// None of these may appear in the rated frontier. Same naming as
        /// `--required`. Ignored unless `--mode reverse`.
        #[arg(long = "excluded")]
        excluded: Vec<String>,
        /// Generic `--mode reverse`: if set, additionally verify the required
        /// set is load-bearing — i.e. `rate_excluding(grid, required)` drops
        /// the tier strictly below `--target-tier`. Off by default; pass
        /// `--load-bearing=true` or `--load-bearing` to enable.
        #[arg(long = "load-bearing", default_value_t = false,
              action = clap::ArgAction::Set,
              num_args = 0..=1, default_missing_value = "true")]
        load_bearing: bool,
        /// Generic `--mode reverse`: explicit negation (overrides
        /// `--load-bearing`). Mostly cosmetic since default is already off,
        /// but keeps the negation flag uniform across reverse paths.
        #[arg(long = "no-load-bearing", default_value_t = false)]
        no_load_bearing: bool,
        /// Generic `--mode reverse`: hard cap on attempts per puzzle slot
        /// (0 = unbounded, terminates only when target reached).
        #[arg(long = "max-attempts", default_value_t = 0)]
        max_attempts: u32,
        /// R1.5: how the required-techniques list is matched against the rated
        /// frontier. `all_of` (default, R1 semantics) | `any_of` |
        /// `exact_k` | `at_least_k` | `weighted`.
        #[arg(long = "match-mode", default_value = "all_of")]
        match_mode: String,
        /// R1.5: K parameter for `exact_k` / `at_least_k` modes.
        #[arg(long = "k", default_value_t = 1)]
        k: u32,
        /// R1.5: weighted mode comma-separated `tech:p` pairs (e.g.
        /// `aic:0.5,fish_xwing:0.5`). Overrides `--required` when set with
        /// `--match-mode weighted`.
        #[arg(long = "weighted")]
        weighted: Option<String>,
        /// R1.5: minimum sampled techniques for `weighted` mode.
        #[arg(long = "min-picked", default_value_t = 1)]
        min_picked: u32,
        /// R1.5: TOML batch-mix-spec path. When set, all other --required /
        /// --match-mode / --weighted / --excluded / --load-bearing /
        /// --max-attempts / --target-tier flags are ignored: each emitted
        /// puzzle uses a per-puzzle weighted-sampled spec from the file.
        #[arg(long = "batch-mix-spec")]
        batch_mix_spec: Option<PathBuf>,
        /// R2.1: AIC chain length target (in edges). When set together with
        /// `--required aic` (only) AND `--match-mode all_of`, dispatches to
        /// the constructive `aic_reverse_construct` path which uses guided
        /// removal to hit the target chain length. Falls back to plain
        /// search-and-filter when 0 or when other techniques are required.
        #[arg(long = "aic-chain-length", default_value_t = 0)]
        aic_chain_length: u32,
        /// R2.1: ± slack on `--aic-chain-length`. Ignored when
        /// `--aic-chain-length == 0`.
        #[arg(long = "aic-chain-slack", default_value_t = 1)]
        aic_chain_slack: u32,
        /// R3.2: shortcut — set `--excluded` to ALL_TECHNIQUE_IDS minus
        /// `(required ∪ {tech} ∪ techniques_easier_than(tech))`. Effectively
        /// "frontier max-level = level(tech)" — purest bucket. Mutually
        /// exclusive with explicit `--excluded`.
        #[arg(long = "max-technique")]
        max_technique: Option<String>,
        /// R3.2: shortcut — set `--excluded` to all T3 techniques when value
        /// is `T2`, all T4Plus when value is `T3`, etc. Mutex with
        /// `--max-technique` and explicit `--excluded`.
        #[arg(long = "max-tier")]
        max_tier: Option<String>,
    },
    /// Reverse-construct hit-rate benchmark. For each technique in
    /// `--techniques` (default = all 16), runs `--attempts` reverse-construct
    /// attempts at `--target-tier` and reports observed pps + reject rate.
    /// Informational baseline for R2 priority.
    ///
    /// `--required X --excluded X` — silently strips X from the per-row
    /// excluded list (the row's target is being benchmarked, so the override
    /// is expected; no conflict-exit here, unlike `gen-text`/`gen-dataset`).
    BenchReverseHitrate {
        /// Grid size as `NxBRxBC`. Default 9x3x3.
        #[arg(long, default_value = "9x3x3")]
        size: String,
        /// Target tier (T1|T2|T3|T4Plus). Default T3.
        #[arg(long = "target-tier", default_value = "T3")]
        target_tier: String,
        /// Required techniques (repeatable). Default = each of the 16 techniques
        /// (one row per tech in the output table).
        #[arg(long = "required")]
        required: Vec<String>,
        /// R3.2: excluded techniques (repeatable). Applied as a constant
        /// excluded-set on every benchmarked row — useful pre-flight check
        /// for "purest bucket" hit-rate (e.g. AIC alone with all easier
        /// techniques excluded).
        #[arg(long = "excluded")]
        excluded: Vec<String>,
        /// Attempts per technique row.
        #[arg(long, default_value_t = 1000)]
        attempts: u32,
        /// Worker threads (0 = auto).
        #[arg(long, default_value_t = 1)]
        threads: usize,
        /// Clue band MIN..MAX (inclusive).
        #[arg(long = "target-clues", default_value = "22..28")]
        target_clues: String,
        /// PRNG seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional JSON output path; if set, writes a structured summary.
        #[arg(long, value_name = "FILE")]
        json_out: Option<PathBuf>,
    },
    /// Construct puzzles that REQUIRE a specific T3 technique.
    ///
    /// Two modes:
    ///   * Legacy 9×9 (no `--size`): supports `--technique aic` only; writes
    ///     a JSONL file under `--output-dir` (back-compat with iter-1).
    ///   * Generic (`--size NxBRxBC` + `--output FILE`): supports
    ///     `--technique aic | ur-type2`. Writes one parquet at `--output`.
    ///     `--target` is the puzzle count, `--threads` worker threads.
    ReverseConstruct {
        /// Technique to target. Legacy 9x9 path: "aic". Generic path:
        /// "aic" | "fish" | "ur-type2" | "als-xz" (requires --size + --output).
        #[arg(long, default_value = "aic")]
        technique: String,
        /// AIC: target chain length in edges (must be odd >= 3). Ignored for ur-type2.
        #[arg(long = "chain-length", default_value_t = 7)]
        chain_length: u32,
        /// AIC: +/- slack on chain length (default 0 = exact). Ignored for ur-type2.
        #[arg(long = "chain-length-slack", default_value_t = 0)]
        chain_length_slack: u32,
        /// Legacy mode: number of puzzles to construct.
        #[arg(long = "target-count", default_value_t = 10)]
        target_count: usize,
        /// Legacy AIC mode: output directory; produces <technique>_len<k>.jsonl.
        /// Optional in generic mode.
        #[arg(long = "output-dir")]
        output_dir: Option<PathBuf>,
        /// Worker threads; 0 = num cpus.
        #[arg(long = "num-workers", default_value_t = 0)]
        num_workers: usize,
        /// Per-seed attempt budget before discarding seed.
        #[arg(long = "max-attempts-per-puzzle", default_value_t = 50)]
        max_attempts_per_puzzle: u32,
        /// PRNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Min clue count to sample (random in [min,max]).
        #[arg(long = "clue-min", default_value_t = 22)]
        clue_min: u32,
        /// Max clue count to sample.
        #[arg(long = "clue-max", default_value_t = 26)]
        clue_max: u32,
        // ---- Generic mode flags (R2.2.x) ----
        /// Generic mode: grid size as `NxBRxBC` (e.g. `9x3x3`, `12x3x4`,
        /// `16x4x4`). Triggers generic mode together with `--output`.
        #[arg(long)]
        size: Option<String>,
        /// Generic mode: output parquet file path (single file).
        #[arg(long)]
        output: Option<PathBuf>,
        /// Generic mode: total puzzles to emit (alias of `--target-count`).
        #[arg(long = "target", default_value_t = 100)]
        target: u32,
        /// Generic mode: worker threads (alias of `--num-workers`).
        #[arg(long = "threads", default_value_t = 0)]
        threads: usize,
        /// Generic mode: require the target technique be load-bearing
        /// (`rate_excluding(puzzle, [target]).tier > target_tier`). On by
        /// default; pass `--no-load-bearing` to disable. Mutually exclusive
        /// with `--fish-no-load-bearing` (legacy alias for fish path).
        #[arg(long = "load-bearing", default_value_t = true,
              action = clap::ArgAction::Set,
              num_args = 0..=1, default_missing_value = "true")]
        load_bearing: bool,
        /// Generic mode: disable the load-bearing requirement (unified flag
        /// covering AIC / Fish / UR / ALS reverse paths). Equivalent to
        /// `--load-bearing=false`. Overrides `--load-bearing` if both given.
        #[arg(long = "no-load-bearing", default_value_t = false)]
        no_load_bearing: bool,
        /// Generic mode: hard cap on outer attempts per puzzle slot
        /// (0 = unbounded, terminates only when target reached).
        #[arg(long = "max-attempts", default_value_t = 0)]
        max_attempts: u32,
        /// Fish target size K in {2,3,4} (X-Wing / Swordfish / Jellyfish).
        /// Required for `--technique fish`.
        #[arg(long = "fish-size", default_value_t = 2)]
        fish_size: u32,
        /// Fish path: skip the load-bearing requirement (overrides
        /// `--load-bearing` for `--technique fish`).
        #[arg(long = "fish-no-load-bearing", default_value_t = false)]
        fish_no_load_bearing: bool,
        /// R2.2.3: ALS-XZ minimum ALS size (smaller of the firing pair).
        /// Default 1 = include bivalues. Range 1..=4. Ignored unless
        /// `--technique als-xz`.
        #[arg(long = "als-size-min", default_value_t = 1)]
        als_size_min: u32,
        /// R2.2.3: ALS-XZ maximum ALS size. Default 4 = production cap.
        /// Ignored unless `--technique als-xz`.
        #[arg(long = "als-size-max", default_value_t = 4)]
        als_size_max: u32,
    },
    /// Re-rate existing parquet shards with the current rater (idempotent).
    /// Preserves puzzle/solution/puzzle_id/clue_count/split; recomputes all
    /// classification metadata. Skips rows with `search_unique == false`.
    ///
    /// Two modes:
    ///   * Legacy (no `--size`, or `--size 9x3x3` without `--output`): the
    ///     byte-identical 9×9-only shard rewriter from P0.
    ///   * Generic (`--size NxBRxBC` + `--output FILE`): generic rerate
    ///     over (6,2,3), (9,3,3), (12,3,4), (16,4,4); reads parquet OR text
    ///     puzzle files, writes one parquet at `--output`.
    ReRate {
        /// Glob pattern matching input parquet (or text) files. Supports
        /// `*`, `?`, `**`. Repeat the flag to combine multiple globs.
        #[arg(long = "input-glob", value_name = "GLOB")]
        input_glob: Vec<String>,
        // ---- Legacy (9×9 sharded) flags ----
        #[arg(long = "output-dir")]
        output_dir: Option<PathBuf>,
        /// DEPRECATED. See note on `gen-dataset --tdoku-bin`. Ignored.
        #[arg(long = "tdoku-bin", hide = true)]
        tdoku_bin: Option<PathBuf>,
        /// Number of rayon worker threads. 0 = rayon default.
        #[arg(long = "num-workers", default_value_t = 0)]
        num_workers: usize,
        /// Records per output parquet shard. (Legacy mode only.)
        #[arg(long = "shard-size", default_value_t = 1000)]
        shard_size: usize,
        // ---- Generic mode flags ----
        /// Generic mode: grid size as `NxBRxBC` (e.g. `9x3x3`, `16x4x4`).
        #[arg(long)]
        size: Option<String>,
        /// Generic mode: output parquet file path (single file).
        #[arg(long)]
        output: Option<PathBuf>,
        /// R3.3a perf knob: `solve` | `tier` | `full`. Default `full` for
        /// backward compatibility — re-rate's whole purpose is recomputing
        /// tier/frontier so `tier` is the typical fast choice; `solve` skips
        /// the cascade entirely and only verifies the puzzle's solution.
        #[arg(long = "mode", default_value = "full")]
        mode: String,
        /// Render a tqdm progress bar to stderr while rating. Off by default;
        /// zero overhead when unset.
        #[arg(long, default_value_t = false)]
        progress: bool,
    },
    /// R3.2: Generate puzzles and write a plain text file (one puzzle per
    /// line). Supports the same `--required` / `--excluded` / `--match-mode`
    /// / `--max-tier` / `--max-technique` filters as `gen-dataset --mode
    /// reverse`. Encoding: `0`/`.` = blank, `1-9` digits, `A-G` for 10-16.
    /// `--include-solution` adds `<puzzle> <solution>` per line.
    /// `--output -` writes to stdout.
    GenText {
        #[arg(long)]
        num: u32,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "9x3x3")]
        size: String,
        /// Required techniques (repeatable).
        #[arg(long = "required")]
        required: Vec<String>,
        /// Excluded techniques (repeatable).
        #[arg(long = "excluded")]
        excluded: Vec<String>,
        /// Match mode: `all_of` | `any_of`. (exact_k / at_least_k / weighted
        /// not exposed via gen-text — use `gen-dataset --mode reverse` for
        /// those.)
        #[arg(long = "match-mode", default_value = "all_of")]
        match_mode: String,
        /// Pure bucket: exclude all techniques except
        /// `{required ∪ {tech} ∪ techniques_easier_than(tech)}`. Same-tier
        /// siblings of `tech` are excluded.
        #[arg(long = "max-technique")]
        max_technique: Option<String>,
        /// Shortcut: exclude all techniques above tier T (e.g. T2 → exclude
        /// all T3).
        #[arg(long = "max-tier")]
        max_tier: Option<String>,
        /// Shortcut: equivalent to `--excluded ALL \ required`. Strict purity
        /// of the frontier — ⊆ required.
        #[arg(long = "excluded-all-others", default_value_t = false)]
        excluded_all_others: bool,
        /// Target tier (T1|T2|T3|T4Plus).
        #[arg(long = "target-tier", default_value = "T3")]
        target_tier: String,
        /// Clue band MIN..MAX (inclusive). Default = tier-derived band, same
        /// fractions as `gen-dataset` generic mode.
        #[arg(long = "target-clues")]
        target_clues: Option<String>,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Append `<space><solution>` per line.
        #[arg(long = "include-solution", default_value_t = false)]
        include_solution: bool,
        /// Per-slot attempt cap (0 = unbounded).
        #[arg(long = "max-attempts-per", default_value_t = 100_000)]
        max_attempts_per: u32,
        /// RFC intercept: require the load-bearing probe (re-rate with both
        /// CFC and RFC excluded; loose probe rejects only when the cascade
        /// still solves at the same or easier SE). Default **false** —
        /// see commit `c0012e1`: in T4Plus the strict probe rejects ~100%
        /// of candidates because DFC/NestedFC can substitute. Pass
        /// `--load-bearing` explicitly to enable the loose probe.
        #[arg(long = "load-bearing", default_value_t = false)]
        load_bearing: bool,
        /// Disable the RFC/CFC load-bearing probe. Overrides `--load-bearing`.
        /// Kept for backward compatibility; default is already disabled.
        #[arg(long = "no-load-bearing", default_value_t = false)]
        no_load_bearing: bool,
    },
    /// Rate a batch of puzzles read from a text file, writing JSONL output.
    ///
    /// Pipeline: TextFileSource → chunked par_iter rate → OrderingSink<JsonlSink>.
    /// Output preserves input order regardless of `--threads`.
    RateBatch {
        /// Input path: one puzzle per line (length N×N, '.' or '0' = blank).
        #[arg(long)]
        input: PathBuf,
        /// Output JSONL path.
        #[arg(long)]
        output: PathBuf,
        /// Grid size as `NxBRxBC`. Supported: 6x2x3, 9x3x3, 12x3x4, 16x4x4.
        #[arg(long, default_value = "9x3x3")]
        size: String,
        /// Worker thread count. 0 = rayon default.
        #[arg(long, default_value_t = 0)]
        threads: usize,
        /// Treat parse errors as fatal (default: skip with stderr warning,
        /// emitting a `rater_error: true` row in the output to keep indices
        /// aligned).
        #[arg(long, default_value_t = false)]
        strict: bool,
        /// Render a tqdm progress bar to stderr while rating. Off by default;
        /// zero overhead when unset.
        #[arg(long, default_value_t = false)]
        progress: bool,
        /// R3.3a perf knob: `solve` (T1 propagate only, fastest), `tier`
        /// (full cascade, skip uniqueness double-solve — recommended for
        /// rate-batch since uniqueness is guaranteed by upstream
        /// construction), or `full` (cascade + uniqueness double-solve;
        /// matches pre-R3.3a behaviour).
        #[arg(long = "mode", default_value = "full")]
        mode: String,
        /// Excluded technique ids (repeatable). When non-empty, the rater runs
        /// `rate_excluding(grid, ...)` instead of the default cascade — useful
        /// for measuring downstream coverage of T4Plus chain techniques in
        /// isolation, e.g. `--excluded cell_forcing_chain --excluded
        /// region_forcing_chain --excluded dynamic_forcing_chain --excluded
        /// nested_forcing_chain --excluded nested_forcing_chain_l3 --excluded
        /// nested_forcing_chain_l4` to see what whip/braid/g-whip/g-braid solve.
        /// Ignored when empty.
        #[arg(long = "excluded")]
        excluded: Vec<String>,
    },
    /// Phase 3 (SHC port): rate a batch of 9×9 puzzles via the clean-room
    /// Rust SHC port (`shc::rate_b`). Output JSONL: one object per input line
    /// with fields `{puzzle, b, wall_ms}` or `{puzzle, error, wall_ms}`.
    ///
    /// Input format: one 81-char puzzle per line; an optional `;B=<N>` tail
    /// (or anything after a `;`) is ignored. Lines starting with `*` (e.g.
    /// SHC.jar header) are skipped.
    RateShcBatch {
        /// Input path: one puzzle per line.
        #[arg(long)]
        input: PathBuf,
        /// Output JSONL path.
        #[arg(long)]
        output: PathBuf,
        /// Max chain length budget passed to `rate_b` (Java `-max-length`).
        #[arg(long = "max-length", default_value_t = 30)]
        max_length: u16,
        /// Partial-braid dedup buffer (Java `-buffer-size`).
        #[arg(long = "buffer-size", default_value_t = 1_000_000usize)]
        buffer_size: usize,
        /// Worker thread count. 1 = single-threaded (matches SHC.jar baseline).
        #[arg(long, default_value_t = 1usize)]
        threads: usize,
        /// Classification mode:
        ///   * `b`        → plain B-rating (Rating_inf_TE1, default)
        ///   * `te-depth` → T&E-depth (0/1/2/3, Rating_TE_depth)
        ///   * `bxb`      → BxB (T&E(1) outer, Rating_sup_TE1 with profmax=1)
        ///   * `bxbb`     → BxBB (T&E(2) outer, Rating_sup_TE1 with profmax=2)
        #[arg(long = "classification", default_value = "b")]
        classification: String,
    },
    /// R3.4 Stage S: ingest a plain-text puzzle list into a parquet seed pool.
    ///
    /// Input: one 81-char 9×9 puzzle per line (`'.'` or `'0'` = empty cell).
    /// Lines starting with `#` and blank lines are skipped. Optionally checks
    /// uniqueness and solves each puzzle (`--solve`).
    ///
    /// Output: parquet with columns `puzzle utf8`, `clue_count int32`,
    /// `unique bool?`, `solution utf8?`. A manifest sidecar
    /// `<out>.manifest.json` is written alongside the parquet.
    IngestSeeds {
        /// Input path: one 81-char puzzle per line, '.' or '0' = empty.
        #[arg(long = "in", value_name = "PATH")]
        input: PathBuf,
        /// Output parquet file path.
        #[arg(long = "out", value_name = "PATH")]
        output: PathBuf,
        /// Label written into the manifest sidecar. Default: stem of --in.
        #[arg(long, default_value = "")]
        label: String,
        /// If set, solve each puzzle (must be unique) and populate the
        /// `unique` and `solution` columns. Adds per-puzzle backtracking cost.
        #[arg(long, default_value_t = false)]
        solve: bool,
        /// Cap on number of input lines processed (0 = unlimited).
        #[arg(long = "max", default_value_t = 0)]
        max: usize,
    },
    /// R3.4 Stage 2: vicinity-search hill-climbing over a seed pool.
    ///
    /// Reads seed puzzles (txt or parquet), mutates them, filters via
    /// uniqueness + rater cascade, and emits hard puzzles (se_score >=
    /// target_se) to a parquet output file.
    ///
    /// Input: `--seed-in` accepts:
    ///   - `.txt`: one 81-char puzzle per line ('.' or '0' = empty)
    ///   - `.parquet`: must have `puzzle` and `solution` utf8 columns
    ///
    /// Output: parquet with columns `puzzle, solution, clue_count, se_score,
    /// tier, frontier_json, generation`. A manifest sidecar is written.
    GenVicinity {
        /// Input seed file: .txt (one puzzle per line) or .parquet.
        #[arg(long = "seed-in", value_name = "PATH")]
        seed_in: PathBuf,
        /// Output parquet file path.
        #[arg(long = "out", value_name = "PATH")]
        out: PathBuf,
        /// Accept candidates with se_score >= this value.
        #[arg(long = "target-se", default_value_t = 7.5)]
        target_se: f64,
        /// Backtrack-count threshold for L3 prefilter; 0 = disable.
        #[arg(long = "bt-prefilter", default_value_t = 0)]
        bt_prefilter: u32,
        /// Mutation counts to try, comma-separated (e.g. "1,2").
        #[arg(long = "mute-n", default_value = "1,2")]
        mute_n: String,
        /// Per-seed iteration cap.
        #[arg(long = "budget-iters", default_value_t = 200)]
        budget_iters: usize,
        /// Global cap on emitted puzzles.
        #[arg(long = "max-outputs", default_value_t = 1000)]
        max_outputs: usize,
        /// PRNG seed.
        #[arg(long = "seed", default_value_t = 0)]
        seed: u64,
        /// Cap on seeds processed (0 = all).
        #[arg(long = "max-seeds", default_value_t = 0)]
        max_seeds: usize,
        /// Label written to manifest sidecar.
        #[arg(long, default_value = "")]
        label: String,
        /// Fitness function: "se" (SE rating via CLIPS cascade) or "bxb" (BxB
        /// rating via SHC TE engine). Default: "se" (backward compat).
        #[arg(long = "fitness", default_value = "se", value_parser = ["se", "bxb"])]
        fitness: String,
        /// Target BxB rating when --fitness=bxb. Accept candidate iff
        /// shc::te::rate_bxb(...) >= this. Range 2..=14. Default: 6.
        #[arg(long = "target-bxb", default_value_t = 6)]
        target_bxb: u16,
        /// max-length for the SHC BxB rater (passed to rate_bxb). Default: 14
        /// (matches SHC.jar BxB default).
        #[arg(long = "bxb-max-length", default_value_t = 14)]
        bxb_max_length: u16,
        /// buffer-size for the SHC BxB rater (BraidArena). Default: 4096.
        #[arg(long = "bxb-buffer-size", default_value_t = 4096)]
        bxb_buffer_size: usize,
        /// Worker thread count for the BxB-fitness path. 0 = rayon default.
        /// Ignored by the SE-fitness path (which has its own parallelization).
        #[arg(long = "threads", default_value_t = 0)]
        threads: usize,
        /// Mutation operator for --fitness=bxb. "v1" = naive digit-flip on
        /// clue cells (preserves clue count, almost never preserves
        /// uniqueness). "v2" = uniqueness-aware solution-aware mutator
        /// (swap_add_remove / remove_only / add_only). Default: v2.
        #[arg(long = "mutator", default_value = "v2", value_parser = ["v1", "v2", "v3", "v4"])]
        mutator: String,
        /// Allow candidates with clue count ≠ seed clue count. Default: true
        /// (required for the v2 add_only / remove_only moves). Set to false to
        /// preserve the v1 invariant.
        #[arg(long = "allow-clue-count-change", default_value_t = true)]
        allow_clue_count_change: bool,
    },
    /// UA-constructor pilot per `docs/specs/ua_constructor_pilot.md`.
    ///
    /// For each of `--attempts` independent random full grids, enumerate
    /// Unavoidable Sets of size ≤ `--max-ua-size`, compute a greedy minimum
    /// hitting set as the clue positions, verify uniqueness, rate via
    /// `shc::te::rate_bxb`. Emits one JSONL line per attempt.
    GenUaPilot {
        #[arg(long, default_value_t = 100)]
        attempts: u64,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long = "max-ua-size", default_value_t = 12)]
        max_ua_size: usize,
        #[arg(long = "target-bxb", default_value_t = 6)]
        target_bxb: i16,
        #[arg(long, default_value_t = 1)]
        threads: usize,
        #[arg(long)]
        output: PathBuf,
    },
    /// Kill-switch pre-test: for each solved-grid in `--input-solutions`,
    /// enumerate UAs up to `--max-ua-size` and report n_uas. Used to verify
    /// the UA enumerator is not under-counting before the full pilot.
    UaKillSwitch {
        /// Path to a file of 81-char solution grids (one per line).
        #[arg(long)]
        input_solutions: PathBuf,
        #[arg(long = "max-ua-size", default_value_t = 12)]
        max_ua_size: usize,
        #[arg(long, default_value_t = 0)]
        max: usize,
    },
}

/// Expand a `glob`-pattern path into a sorted, deduplicated list of files.
/// Inlined here after the legacy `rerate::expand_glob` was retired in R3.1a.
fn expand_glob(pattern: &str) -> std::io::Result<Vec<PathBuf>> {
    use glob::glob;
    let mut out = Vec::new();
    let entries = glob(pattern).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("bad glob '{}': {}", pattern, e))
    })?;
    for entry in entries {
        match entry {
            Ok(p) => {
                if p.is_file() {
                    out.push(p);
                }
            }
            Err(e) => {
                eprintln!("warning: glob error on '{}': {}", pattern, e);
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

#[allow(dead_code)]
fn parse_target_per_tier(s: &str) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for k in ["T1", "T2", "T3", "T4"] {
        m.insert(k.into(), 0);
    }
    for piece in s.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let (k, v) = piece.split_once('=').expect("format: Tier=N");
        m.insert(k.trim().to_string(), v.trim().parse().expect("integer"));
    }
    m
}

/// Parse `--weighted "tech:p,tech:p,..."` into a list of `(TechniqueId, f32)`.
fn parse_weighted_arg(s: &str) -> Result<Vec<(GTechniqueId, f32)>, String> {
    let mut out = Vec::new();
    for piece in s.split(',') {
        let piece = piece.trim();
        if piece.is_empty() { continue; }
        let (name, p_str) = piece
            .split_once(':')
            .ok_or_else(|| format!("entry '{}' missing ':p'", piece))?;
        let id = parse_generic_tech(name.trim())?;
        let p: f32 = p_str
            .trim()
            .parse()
            .map_err(|e| format!("entry '{}': bad probability '{}': {}", piece, p_str, e))?;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(format!("entry '{}': p={} must be in [0,1]", piece, p));
        }
        out.push((id, p));
    }
    if out.is_empty() {
        return Err("weighted spec is empty".to_string());
    }
    Ok(out)
}

/// R3.2: TechniqueId → Tier classification. Mirrors the cascade lists in
/// `generic::rater::{T2_LIST, T3_LIST}`. Centralised here so the CLI shortcuts
/// (`--max-tier`, `--max-technique`) don't need internal access to the
/// `AnyTechnique` enum.
fn tier_of(id: GTechniqueId) -> GTier {
    use GTechniqueId::*;
    match id {
        LockedPointing | LockedClaiming | NakedPair | HiddenPair | NakedTriple
        | HiddenTriple | NakedQuad | HiddenQuad => GTier::T2,
        // Everything else (Fish, XyWing, XyzWing, UrType1, UrType2, Aic,
        // AlsXz, Bug, SimpleColoring, Skyscraper, TwoStringKite) → T3.
        _ => GTier::T3,
    }
}

#[inline]
fn tier_rank(t: GTier) -> u8 {
    match t {
        GTier::T1 => 1,
        GTier::T2 => 2,
        GTier::T3 => 3,
        GTier::T4Plus => 4,
    }
}

/// R3.2: build the excluded-set from a `--max-technique` shortcut.
/// "frontier max-level = level(tech)" purest bucket: excluded = ALL \
/// (required ∪ {tech} ∪ techniques_easier_than(tech)).
fn excluded_from_max_technique(
    max_tech: GTechniqueId,
    required: &[GTechniqueId],
) -> Vec<GTechniqueId> {
    let max_tier = tier_of(max_tech);
    ALL_TECHNIQUE_IDS
        .iter()
        .copied()
        .filter(|&t| {
            if t == max_tech {
                return false;
            }
            if required.contains(&t) {
                return false;
            }
            // Drop "easier than" max_tech. "Easier" = strictly lower tier.
            // (Within the same tier we still exclude unless required, because
            // max-technique semantics is "ONLY max_tech and easier may
            // appear" — we keep easier, exclude same-tier-but-different.)
            if tier_rank(tier_of(t)) < tier_rank(max_tier) {
                return false;
            }
            true
        })
        .collect()
}

/// R3.2: build the excluded-set from a `--max-tier` shortcut.
/// `--max-tier T2` → exclude all techniques with `tier > T2`.
fn excluded_from_max_tier(max_tier: GTier, required: &[GTechniqueId]) -> Vec<GTechniqueId> {
    ALL_TECHNIQUE_IDS
        .iter()
        .copied()
        .filter(|&t| {
            if required.contains(&t) {
                return false;
            }
            tier_rank(tier_of(t)) > tier_rank(max_tier)
        })
        .collect()
}

/// R3.2: build excluded for `--excluded-all-others` — ALL \ required.
fn excluded_all_others_fn(required: &[GTechniqueId]) -> Vec<GTechniqueId> {
    ALL_TECHNIQUE_IDS
        .iter()
        .copied()
        .filter(|t| !required.contains(t))
        .collect()
}

/// Dispatch `batch_reverse_construct_mixed` over (N,BR,BC) and write parquet.
fn run_reverse_mode_mixed(
    n: usize,
    br: usize,
    bc: usize,
    mix: &BatchMixSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;
    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        mix: &BatchMixSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_reverse_mixed::<N, BR, BC>(seed, mix, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }
    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(mix, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(mix, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(mix, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(mix, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R2.2.1: Dispatch `batch_fish_reverse_construct` over (N,BR,BC) and write
/// parquet. Returns the number of records written.
fn run_fish_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &FishReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &FishReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_fish_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R2.1: Dispatch `batch_aic_reverse_construct` over (N,BR,BC) and write
/// parquet. Returns the number of records written.
fn run_aic_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &AicReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &AicReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_aic_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R2.2.3: Dispatch `batch_als_xz_reverse_construct` over (N,BR,BC) and
/// write parquet. Returns the number of records written.
fn run_als_xz_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &AlsXzReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &AlsXzReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_als_xz_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R3.5: Dispatch `batch_cfc_reverse_construct` over (N,BR,BC). Returns lines.
fn run_cfc_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &CfcReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    include_solution: bool,
) -> Vec<String> {
    fn build_lines<const N: usize, const BR: usize, const BC: usize>(
        spec: &CfcReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        include_solution: bool,
    ) -> Vec<String> {
        let res = generic_batch_cfc_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                let p = r.puzzle.to_string_grid();
                if include_solution {
                    let s = r.solution.to_string_grid();
                    format!("{} {}", p, s)
                } else {
                    p
                }
            })
            .collect()
    }
    match (n, br, bc) {
        (6, 2, 3) => build_lines::<6, 2, 3>(spec, num, threads, seed, include_solution),
        (9, 3, 3) => build_lines::<9, 3, 3>(spec, num, threads, seed, include_solution),
        (12, 3, 4) => build_lines::<12, 3, 4>(spec, num, threads, seed, include_solution),
        (16, 4, 4) => build_lines::<16, 4, 4>(spec, num, threads, seed, include_solution),
        _ => {
            eprintln!("error: unsupported size {}x{}x{} for cfc-reverse", n, br, bc);
            vec![]
        }
    }
}

/// R2.2.4: Dispatch `batch_naked_quad_reverse_construct` over (N,BR,BC) and
/// write parquet. Returns the number of records written.
fn run_naked_quad_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &NakedQuadReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &NakedQuadReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_naked_quad_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R2.2.4: Dispatch `batch_hidden_quad_reverse_construct` over (N,BR,BC) and
/// write parquet. Returns the number of records written.
fn run_hidden_quad_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &HiddenQuadReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &HiddenQuadReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_hidden_quad_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R2.2.2: Dispatch `batch_ur_type2_reverse_construct` over (N,BR,BC) and
/// write parquet. Returns the number of records written.
fn run_ur_type2_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &UrType2ReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &UrType2ReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_ur_type2_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// Dispatch `batch_reverse_construct` over (N,BR,BC) and write parquet.
/// Returns the number of records written.
fn run_reverse_mode(
    n: usize,
    br: usize,
    bc: usize,
    spec: &GenericReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    out_path: &std::path::Path,
) -> std::io::Result<usize> {
    use sudoku_rs_core::pipeline_writer_generic::GRecord;

    fn build_records<const N: usize, const BR: usize, const BC: usize>(
        spec: &GenericReverseSpec,
        num: u32,
        threads: usize,
        seed: u64,
        size_str: &str,
        br: usize,
        bc: usize,
    ) -> Vec<GRecord> {
        let res = generic_batch_reverse::<N, BR, BC>(seed, spec, num, threads);
        res.iter()
            .map(|r| {
                make_generic_record::<N, BR, BC>(
                    &r.puzzle, &r.solution, &r.rate, r.clue_count, size_str, br, bc,
                )
            })
            .collect()
    }

    let size_str = format!("{}x{}x{}", n, br, bc);
    let records: Vec<GRecord> = match (n, br, bc) {
        (6, 2, 3) => build_records::<6, 2, 3>(spec, num, threads, seed, &size_str, br, bc),
        (9, 3, 3) => build_records::<9, 3, 3>(spec, num, threads, seed, &size_str, br, bc),
        (12, 3, 4) => build_records::<12, 3, 4>(spec, num, threads, seed, &size_str, br, bc),
        (16, 4, 4) => build_records::<16, 4, 4>(spec, num, threads, seed, &size_str, br, bc),
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unsupported size {}x{}x{}", n, br, bc),
            ))
        }
    };
    let n_kept = records.len();
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    write_generic_parquet(out_path, &records)?;
    Ok(n_kept)
}

/// R3.2: gen-text inner driver. Returns rendered text lines.
/// `include_solution` adds `<space><solution>` per line.
fn run_gen_text<const N: usize, const BR: usize, const BC: usize>(
    spec: &GenericReverseSpec,
    num: u32,
    threads: usize,
    seed: u64,
    include_solution: bool,
) -> Vec<String> {
    let res = generic_batch_reverse::<N, BR, BC>(seed, spec, num, threads);
    res.iter()
        .map(|r| {
            let p = r.puzzle.to_string_grid();
            if include_solution {
                let s = r.solution.to_string_grid();
                format!("{} {}", p, s)
            } else {
                p
            }
        })
        .collect()
}

/// Run a hit-rate benchmark over the supplied technique list (or all 16) at
/// the given size+tier+clue-band. Prints a markdown table and optionally
/// writes a JSON summary.
fn run_bench_reverse_hitrate(
    size_str: &str,
    target_tier_str: &str,
    techniques: Vec<String>,
    excluded: Vec<String>,
    attempts: u32,
    threads: usize,
    target_clues: &str,
    seed: u64,
    json_out: Option<PathBuf>,
) -> std::io::Result<()> {
    let (n, br, bc) = sudoku_rs_core::pipeline_writer_generic::parse_size(size_str)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let tier = sudoku_rs_core::pipeline_writer_generic::parse_tier(target_tier_str)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let (clue_min, clue_max) =
        sudoku_rs_core::pipeline_writer_generic::parse_clue_range(target_clues)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
    let workers = if threads == 0 {
        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1)
    } else {
        threads
    };

    // R3.2: parse --excluded into a constant set applied to every row.
    let exc_ids: Vec<GTechniqueId> = match excluded
        .iter()
        .map(|s| parse_generic_tech(s))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(v) => v,
        Err(e) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("--excluded parse: {}", e),
            ));
        }
    };
    let tech_list: Vec<GTechniqueId> = if techniques.is_empty() {
        ALL_TECHNIQUE_IDS.to_vec()
    } else {
        match techniques
            .iter()
            .map(|s| parse_generic_tech(s))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(e) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("--required parse: {}", e),
                ))
            }
        }
    };

    fn count_size<const N: usize, const BR: usize, const BC: usize>(
        spec: &GenericReverseSpec,
        attempts_per: u32,
        threads: usize,
        seed: u64,
    ) -> (u32, f64) {
        // We re-use batch_reverse_construct with num_puzzles=u32::MAX so the
        // workers exhaust their attempt budget. Hits = collected.len().
        let t0 = std::time::Instant::now();
        let res = generic_batch_reverse::<N, BR, BC>(seed, spec, u32::MAX, threads);
        let dt = t0.elapsed().as_secs_f64();
        let hits = res.len() as u32;
        // pps = hits / wall-clock time; reject_rate computed from attempts.
        let _ = attempts_per;
        (hits, dt)
    }

    println!(
        "## Reverse-construct hit-rate ({}, tier={}, clues={}..{}, attempts={}, threads={})",
        size_str, target_tier_str, clue_min, clue_max, attempts, workers
    );
    println!("| required | hits/{} | pps | wall_s |", attempts);
    println!("|---|---|---|---|");

    #[derive(serde::Serialize)]
    struct Row {
        size: String,
        tier: String,
        required: String,
        attempts: u32,
        hits: u32,
        pps: f64,
        reject_rate: f64,
        wall_s: f64,
    }
    let mut rows: Vec<Row> = Vec::new();
    for &t in &tech_list {
        // R3.2: per-row override — silently strip the row's target `t` from
        // the global excluded set. Each row benchmarks a different technique,
        // so a global `--excluded X` together with `--required X` (or default
        // tech-list including X) is *not* a user error here: we want the row
        // for X to actually attempt X. No conflict-exit (unlike gen-text /
        // gen-dataset, where required ∩ excluded is a hard error).
        let mut row_excluded = exc_ids.clone();
        row_excluded.retain(|x| *x != t);
        let spec = GenericReverseSpec {
            target_tier: tier,
            match_mode: MatchMode::AllOf(vec![t]),
            excluded_techniques: row_excluded,
            load_bearing: false,
            clue_min,
            clue_max,
            max_attempts: attempts,
        };
        let (hits, wall) = match (n, br, bc) {
            (6, 2, 3) => count_size::<6, 2, 3>(&spec, attempts, workers, seed),
            (9, 3, 3) => count_size::<9, 3, 3>(&spec, attempts, workers, seed),
            (12, 3, 4) => count_size::<12, 3, 4>(&spec, attempts, workers, seed),
            (16, 4, 4) => count_size::<16, 4, 4>(&spec, attempts, workers, seed),
            _ => unreachable!(),
        };
        let pps = hits as f64 / wall.max(1e-9);
        let reject_rate =
            1.0 - (hits as f64 / attempts.max(1) as f64);
        let name = generic_tech_str(t);
        println!(
            "| {} | {} | {:.3} | {:.2} |",
            name, hits, pps, wall
        );
        rows.push(Row {
            size: size_str.to_string(),
            tier: target_tier_str.to_string(),
            required: name.to_string(),
            attempts,
            hits,
            pps,
            reject_rate,
            wall_s: wall,
        });
    }
    // ---- R1.5: also bench representative mix modes (size 9x3x3 only). ----
    if (n, br, bc) == (9, 3, 3) {
        println!();
        println!("### Mix modes (size=9x3x3, tier={})", target_tier_str);
        println!("| mode | required | hits | pps | wall_s |");
        println!("|---|---|---|---|---|");
        let mix_modes: Vec<(&str, MatchMode)> = vec![
            (
                "any_of",
                MatchMode::AnyOf(vec![
                    GTechniqueId::Aic,
                    GTechniqueId::XWing,
                    GTechniqueId::Swordfish,
                ]),
            ),
            (
                "at_least_k=2",
                MatchMode::AtLeastK {
                    techniques: vec![
                        GTechniqueId::Aic,
                        GTechniqueId::XWing,
                        GTechniqueId::XyWing,
                        GTechniqueId::UrType1,
                    ],
                    k: 2,
                },
            ),
            (
                "weighted",
                MatchMode::Weighted {
                    weights: vec![(GTechniqueId::Aic, 0.5), (GTechniqueId::XWing, 0.5)],
                    min_picked: 1,
                },
            ),
        ];
        for (name, mm) in &mix_modes {
            let req_str = format!("{:?}", mm.implied_required());
            let spec = GenericReverseSpec {
                target_tier: tier,
                match_mode: mm.clone(),
                excluded_techniques: Vec::new(),
                load_bearing: false,
                clue_min,
                clue_max,
                max_attempts: attempts,
            };
            let (hits, wall) = count_size::<9, 3, 3>(&spec, attempts, workers, seed);
            let pps = hits as f64 / wall.max(1e-9);
            println!(
                "| {} | {} | {} | {:.3} | {:.2} |",
                name, req_str, hits, pps, wall
            );
            rows.push(Row {
                size: "9x3x3".to_string(),
                tier: target_tier_str.to_string(),
                required: format!("MIX:{}:{}", name, req_str),
                attempts,
                hits,
                pps,
                reject_rate: 1.0 - (hits as f64 / attempts.max(1) as f64),
                wall_s: wall,
            });
        }
    }

    if let Some(p) = json_out {
        let f = std::fs::File::create(&p)?;
        serde_json::to_writer_pretty(f, &rows)?;
        eprintln!("bench-reverse-hitrate: wrote {} rows -> {}", rows.len(), p.display());
    }
    Ok(())
}

/// Run the `rate-batch` pipeline for a specific (N, BR, BC). Monomorphized
/// per supported size by the dispatch in main(). Reads puzzles from `src`,
/// rates in chunks of 1024 in parallel, writes a single JSONL line per puzzle
/// in input order.
fn run_rate_batch<const N: usize, const BR: usize, const BC: usize>(
    mut src: TextFileSource<N>,
    output: &std::path::Path,
    strict: bool,
    progress: bool,
    mode: GSolverMode,
    exclude: &[GTechniqueId],
) -> std::io::Result<(u64, u64)> {
    let sink: JsonlSink<N, std::fs::File> = JsonlSink::create(output)?;
    let mut ordered: OrderingSink<N, JsonlSink<N, std::fs::File>> = OrderingSink::new(sink);

    const CHUNK: usize = 1024;
    let mut total: u64 = 0;
    let mut errors: u64 = 0;

    // tqdm progress bar (stderr). `None` total = streaming source; tqdm
    // displays count + rate + elapsed.
    let mut pbar = if progress {
        Some(tqdm::pbar(None).desc(Some("rate-batch")))
    } else {
        None
    };

    // R3.2: in non-strict mode we use TextFileSource::next_lossy so a
    // malformed line still consumes an index and we emit a synthetic
    // `rater_error: true` row at that index — preserves input-index
    // alignment in the output JSONL.
    let mut eof = false;
    while !eof {
        let mut chunk: Vec<RawPuzzle<N>> = Vec::with_capacity(CHUNK);
        let mut bad_rows: Vec<(u64, String)> = Vec::new();
        for _ in 0..CHUNK {
            if strict {
                match src.next() {
                    Ok(Some(p)) => chunk.push(p),
                    Ok(None) => {
                        eof = true;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            } else {
                match src.next_lossy()? {
                    NextLossy::Ok(p) => chunk.push(p),
                    NextLossy::Bad { i, msg } => {
                        eprintln!("warn: parse error at i={}: {} (emitting rater_error row)", i, msg);
                        bad_rows.push((i, msg));
                    }
                    NextLossy::Eos => {
                        eof = true;
                        break;
                    }
                }
            }
        }
        if chunk.is_empty() && bad_rows.is_empty() {
            break;
        }
        // Parallel rate of valid puzzles.
        let rated: Vec<RatedPuzzle<N>> = chunk
            .par_iter()
            .map(|raw| {
                let rate = match GGrid::<N, BR, BC>::from_str(&raw.puzzle) {
                    Some(g) => {
                        // When --excluded is non-empty, switch to
                        // `rate_excluding`: the rater pipeline still runs
                        // BRT + T2/T3, but skips the listed technique ids in
                        // the cascade. `mode` is ignored on this path (no
                        // uniqueness double-solve performed by the excluding
                        // rater). When empty, preserve the original
                        // mode-aware fast path.
                        if exclude.is_empty() {
                            rate_with_uniqueness_mode(&g, mode)
                        } else {
                            use sudoku_rs_core::generic::rater::rate_excluding;
                            rate_excluding(&g, exclude)
                        }
                    }
                    None => GRateResult {
                        tier: GTier::T4Plus,
                        frontier: vec![],
                        solved: false,
                        trace: vec![],
                        rater_error: true,
                        wave_depth: 0,
                        backtrack_steps: 0,
                        unique_solution: false,
                        se_score: 0.0,
                    },
                };
                RatedPuzzle::<N>::new(raw.i, raw.puzzle.clone(), rate)
            })
            .collect();

        // Synthetic rows for malformed lines (rater_error=true, empty puzzle
        // string of length N×N filled with blanks for valid encoding).
        let blank: String = std::iter::repeat('.').take(N * N).collect();
        for (i, _msg) in bad_rows.drain(..) {
            errors += 1;
            let rate = GRateResult {
                tier: GTier::T4Plus,
                frontier: vec![],
                solved: false,
                trace: vec![],
                rater_error: true,
                wave_depth: 0,
                backtrack_steps: 0,
                unique_solution: false,
                se_score: 0.0,
            };
            ordered.submit(RatedPuzzle::<N>::new(i, blank.clone(), rate))?;
            total += 1;
        }
        for rp in rated {
            if rp.rate.rater_error {
                errors += 1;
            }
            ordered.submit(rp)?;
            total += 1;
            if let Some(ref mut pb) = pbar {
                let _ = pb.update(1);
            }
        }
    }

    if let Some(mut pb) = pbar {
        let _ = pb.close();
    }
    let _sink = ordered.finish()?;
    Ok((total, errors))
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Solve { puzzle } => {
            let g: GGrid<9, 3, 3> = GGrid::from_str(&puzzle).expect("parse");
            match g_solve_unique(&g) {
                Some(s) => println!("{}", s.to_string_grid()),
                None => eprintln!("non-unique or unsolvable"),
            }
        }
        Cmd::Rate { puzzle } => {
            let g: GGrid<9, 3, 3> = GGrid::from_str(&puzzle).expect("parse");
            let r = rate_with_uniqueness(&g);
            println!("{:?}", r);
        }
        Cmd::Gen { clues, n, seed } => {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let cfg = GGenConfig::new(clues);
            for _ in 0..n {
                let (p, c): (GGrid<9, 3, 3>, u32) = g_gen_unique_puzzle(&mut rng, &cfg);
                println!("{} {}", p.to_string_grid(), c);
            }
        }
        Cmd::GenDataset {
            size,
            num,
            threads,
            target_clues,
            output,
            n_clues,
            target_total,
            flush_every,
            worker_buffer,
            num_workers,
            seed,
            out_dir,
            drop_unresolved,
            target_per_tier,
            tdoku_bin,
            mode,
            target_tier,
            min_chain_length,
            max_clues,
            min_clues,
            max_consecutive_failures,
            required,
            excluded,
            load_bearing,
            max_attempts,
            match_mode,
            k,
            weighted,
            min_picked,
            batch_mix_spec,
            aic_chain_length,
            aic_chain_slack,
            no_load_bearing,
            max_technique,
            max_tier: max_tier_arg,
        } => {
            // R3.2: unified `--no-load-bearing` overrides `--load-bearing`.
            let load_bearing = if no_load_bearing { false } else { load_bearing };
            // ---- New generic mode: triggered by `--size`. ----
            if let Some(size_str) = size.as_deref() {
                let (n, br, bc) = match parse_size(size_str) {
                    Ok(v) => v,
                    Err(e) => { eprintln!("error: --size: {}", e); std::process::exit(2); }
                };
                let tier = match parse_tier(&target_tier) {
                    Ok(t) => t,
                    Err(e) => { eprintln!("error: --target-tier: {}", e); std::process::exit(2); }
                };
                let (clue_min, clue_max) = match target_clues.as_deref() {
                    Some(r) => match parse_clue_range(r) {
                        Ok(v) => v,
                        Err(e) => { eprintln!("error: --target-clues: {}", e); std::process::exit(2); }
                    },
                    None => {
                        // Tier-specific defaults. Without these, T1 produced a
                        // constant clue_count = n*n - 1 because the very first
                        // (uniqueness-preserving) removal already lands inside
                        // [0, n*n], locks the bucket and breaks the loop.
                        // Fractions are calibrated empirically from rater
                        // distributions and reproduce the canonical 9×9 ranges.
                        let cells = (n * n) as f64;
                        let (lo_frac, hi_frac) = match tier {
                            sudoku_rs_core::generic::techniques::Tier::T1 => (0.40, 0.55),
                            sudoku_rs_core::generic::techniques::Tier::T2 => (0.30, 0.42),
                            sudoku_rs_core::generic::techniques::Tier::T3 => (0.27, 0.36),
                            sudoku_rs_core::generic::techniques::Tier::T4Plus => (0.20, 0.32),
                        };
                        let lo = (lo_frac * cells).round() as u32;
                        let hi = (hi_frac * cells).round() as u32;
                        (lo, hi)
                    }
                };
                let target_total_g = num.unwrap_or(target_total);
                let workers = if threads == 0 {
                    std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1)
                } else {
                    threads
                };
                let out_path = match output {
                    Some(p) => p,
                    None => {
                        eprintln!("error: --output is required in generic --size mode");
                        std::process::exit(2);
                    }
                };
                // ---- Reverse-construct mode: search-and-filter for
                //      technique-targeted puzzles. ----
                if mode == "reverse" {
                    // R1.5: batch-mix-spec overrides every other reverse flag.
                    if let Some(mix_path) = batch_mix_spec.as_ref() {
                        let mix = match BatchMixSpec::load_toml(mix_path) {
                            Ok(m) => m,
                            Err(e) => {
                                eprintln!("error: --batch-mix-spec: {}", e);
                                std::process::exit(2);
                            }
                        };
                        let t0 = std::time::Instant::now();
                        let stats = run_reverse_mode_mixed(
                            n, br, bc, &mix, target_total_g as u32, workers, seed,
                            &out_path,
                        )?;
                        let dt = t0.elapsed();
                        eprintln!(
                            "gen-dataset (reverse-mixed, size={}): kept={} elapsed={:.2}s pps={:.3} -> {}",
                            size_str,
                            stats,
                            dt.as_secs_f64(),
                            stats as f64 / dt.as_secs_f64().max(1e-9),
                            out_path.display(),
                        );
                        return Ok(());
                    }
                    let req_ids: Vec<GTechniqueId> = match required
                        .iter()
                        .map(|s| parse_generic_tech(s))
                        .collect::<Result<Vec<_>, _>>()
                    {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("error: --required: {}", e);
                            std::process::exit(2);
                        }
                    };
                    let mut exc_ids: Vec<GTechniqueId> = match excluded
                        .iter()
                        .map(|s| parse_generic_tech(s))
                        .collect::<Result<Vec<_>, _>>()
                    {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("error: --excluded: {}", e);
                            std::process::exit(2);
                        }
                    };
                    // R3.2: --max-technique / --max-tier shortcuts. Mutex with
                    // explicit --excluded.
                    let have_explicit_excluded = !exc_ids.is_empty();
                    if max_technique.is_some() && max_tier_arg.is_some() {
                        eprintln!("error: --max-technique and --max-tier are mutually exclusive");
                        std::process::exit(2);
                    }
                    if (max_technique.is_some() || max_tier_arg.is_some()) && have_explicit_excluded {
                        eprintln!("error: --max-technique/--max-tier cannot be combined with explicit --excluded");
                        std::process::exit(2);
                    }
                    if let Some(mt) = max_technique.as_deref() {
                        let mt_id = match parse_generic_tech(mt) {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!("error: --max-technique: {}", e);
                                std::process::exit(2);
                            }
                        };
                        exc_ids = excluded_from_max_technique(mt_id, &req_ids);
                    } else if let Some(mt) = max_tier_arg.as_deref() {
                        let mt_tier = match parse_tier(mt) {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!("error: --max-tier: {}", e);
                                std::process::exit(2);
                            }
                        };
                        exc_ids = excluded_from_max_tier(mt_tier, &req_ids);
                    }
                    // R3.2: conflict detection on required ∩ excluded.
                    for r in &req_ids {
                        if exc_ids.contains(r) {
                            eprintln!(
                                "error: '{}' is in both --required and --excluded",
                                generic_tech_str(*r)
                            );
                            std::process::exit(2);
                        }
                    }
                    let mode_norm = match_mode.trim().to_ascii_lowercase();
                    let mm = match mode_norm.as_str() {
                        "all_of" | "allof" | "all" => MatchMode::AllOf(req_ids.clone()),
                        "any_of" | "anyof" | "any" => {
                            if req_ids.is_empty() {
                                eprintln!(
                                    "error: --match-mode any_of requires at least one --required"
                                );
                                std::process::exit(2);
                            }
                            MatchMode::AnyOf(req_ids.clone())
                        }
                        "exact_k" | "exactk" => {
                            if req_ids.is_empty() {
                                eprintln!(
                                    "error: --match-mode exact_k requires --required entries"
                                );
                                std::process::exit(2);
                            }
                            MatchMode::ExactK { techniques: req_ids.clone(), k }
                        }
                        "at_least_k" | "atleastk" | "atleast_k" => {
                            if req_ids.is_empty() {
                                eprintln!(
                                    "error: --match-mode at_least_k requires --required entries"
                                );
                                std::process::exit(2);
                            }
                            MatchMode::AtLeastK { techniques: req_ids.clone(), k }
                        }
                        "weighted" => {
                            let s = match weighted.as_deref() {
                                Some(s) => s,
                                None => {
                                    eprintln!("error: --match-mode weighted requires --weighted \"tech:p,...\"");
                                    std::process::exit(2);
                                }
                            };
                            match parse_weighted_arg(s) {
                                Ok(w) => MatchMode::Weighted { weights: w, min_picked },
                                Err(e) => {
                                    eprintln!("error: --weighted: {}", e);
                                    std::process::exit(2);
                                }
                            }
                        }
                        other => {
                            eprintln!(
                                "error: --match-mode must be all_of|any_of|exact_k|at_least_k|weighted, got '{}'",
                                other
                            );
                            std::process::exit(2);
                        }
                    };
                    // R2.1: detect "constructive AIC chain-targeted" path:
                    // single required tech == aic, all_of mode, no exclusions,
                    // and `--aic-chain-length > 0`. Dispatches to
                    // `batch_aic_reverse_construct`; falls through to plain
                    // search-and-filter otherwise.
                    let only_aic_required = matches!(&mm, MatchMode::AllOf(v)
                        if v.len() == 1 && v[0] == GTechniqueId::Aic);
                    if aic_chain_length > 0 && only_aic_required && exc_ids.is_empty() {
                        let aic_spec = AicReverseSpec {
                            target_tier: tier,
                            target_chain_length: aic_chain_length,
                            chain_length_slack: aic_chain_slack,
                            clue_min,
                            clue_max,
                            max_attempts,
                            require_load_bearing: load_bearing,
                            greedy_max_trials: 0,
                        };
                        if let Err(e) = aic_spec.validate() {
                            eprintln!("error: aic-spec validation: {}", e);
                            std::process::exit(2);
                        }
                        let t0 = std::time::Instant::now();
                        let stats = run_aic_reverse_mode(
                            n, br, bc, &aic_spec, target_total_g as u32, workers, seed,
                            &out_path,
                        )?;
                        let dt = t0.elapsed();
                        eprintln!(
                            "gen-dataset (reverse-aic-chain, size={}, target_len={}±{}): kept={} elapsed={:.2}s pps={:.3} -> {}",
                            size_str,
                            aic_chain_length,
                            aic_chain_slack,
                            stats,
                            dt.as_secs_f64(),
                            stats as f64 / dt.as_secs_f64().max(1e-9),
                            out_path.display(),
                        );
                        return Ok(());
                    }
                    let spec = GenericReverseSpec {
                        target_tier: tier,
                        match_mode: mm,
                        excluded_techniques: exc_ids,
                        load_bearing,
                        clue_min,
                        clue_max,
                        max_attempts,
                    };
                    if let Err(e) = spec.validate() {
                        eprintln!("error: spec validation: {}", e);
                        std::process::exit(2);
                    }
                    let t0 = std::time::Instant::now();
                    let stats = run_reverse_mode(
                        n, br, bc, &spec, target_total_g as u32, workers, seed,
                        &out_path,
                    )?;
                    let dt = t0.elapsed();
                    eprintln!(
                        "gen-dataset (reverse, size={}): kept={} elapsed={:.2}s pps={:.3} -> {}",
                        size_str,
                        stats,
                        dt.as_secs_f64(),
                        stats as f64 / dt.as_secs_f64().max(1e-9),
                        out_path.display(),
                    );
                    return Ok(());
                }
                let cfg = GenericPipelineConfig {
                    n, br, bc,
                    target_tier: tier,
                    clue_min, clue_max,
                    target_total: target_total_g,
                    num_workers: workers,
                    seed,
                    output: out_path.clone(),
                };
                let t0 = std::time::Instant::now();
                let stats = run_generic_pipeline(&cfg)?;
                let dt = t0.elapsed();
                eprintln!(
                    "gen-dataset (generic, size={}): kept={} elapsed={:.2}s pps={:.1} -> {}",
                    size_str,
                    stats.n_puzzles,
                    dt.as_secs_f64(),
                    stats.n_puzzles as f64 / dt.as_secs_f64().max(1e-9),
                    out_path.display(),
                );
                return Ok(());
            }
            // R3.1a: legacy 9×9-only path retired. Generic mode (--size) is now mandatory.
            let _ = (
                &n_clues, &target_total, &flush_every, &worker_buffer, &num_workers,
                &seed, &out_dir, &drop_unresolved, &target_per_tier, &tdoku_bin, &mode,
                &target_tier, &min_chain_length, &max_clues, &min_clues,
                &max_consecutive_failures, &required, &excluded, &load_bearing,
                &max_attempts, &match_mode, &k, &weighted, &min_picked, &batch_mix_spec,
                &aic_chain_length, &aic_chain_slack, &no_load_bearing, &max_technique,
                &max_tier_arg,
            );
            eprintln!("error: gen-dataset requires --size NxBRxBC (legacy 9×9-only mode retired in R3.1a)");
            std::process::exit(2);
        }
        Cmd::ReverseConstruct {
            technique,
            chain_length,
            chain_length_slack,
            target_count,
            output_dir,
            num_workers,
            max_attempts_per_puzzle,
            seed,
            clue_min,
            clue_max,
            size,
            output,
            target,
            threads,
            load_bearing,
            max_attempts,
            fish_size,
            fish_no_load_bearing,
            als_size_min,
            als_size_max,
            no_load_bearing,
        } => {
            // R3.2: deprecated subcommand — forward to gen-dataset-equivalent
            // call site, but keep all existing logic. Print one-shot stderr
            // warning so callers know to migrate.
            eprintln!(
                "warning: `reverse-construct` is deprecated; use `gen-dataset --mode reverse \
                 --required <tech>` (with optional --aic-chain-length / --max-tier / \
                 --max-technique). reverse-construct will be removed in a future revision."
            );
            // R3.2: --no-load-bearing overrides --load-bearing.
            let load_bearing = if no_load_bearing { false } else { load_bearing };
            // ---- R2.2.1 generic Fish path ---------------------------------
            if technique == "fish" {
                let size_str = match size.as_deref() {
                    Some(s) => s,
                    None => {
                        eprintln!("error: --technique fish requires --size NxBRxBC");
                        std::process::exit(2);
                    }
                };
                let (n_, br_, bc_) = match parse_size(size_str) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("error: --size: {}", e);
                        std::process::exit(2);
                    }
                };
                let out_path = match output {
                    Some(p) => p,
                    None => {
                        eprintln!("error: --technique fish requires --output FILE");
                        std::process::exit(2);
                    }
                };
                if !matches!(fish_size, 2 | 3 | 4) {
                    eprintln!("error: --fish-size must be 2, 3, or 4 (got {})", fish_size);
                    std::process::exit(2);
                }
                // R3.2: prefer --threads, fall back to --num-workers (legacy
                // alias). Both 0 → auto.
                let workers = if threads == 0 {
                    if num_workers == 0 {
                        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
                    } else {
                        num_workers
                    }
                } else {
                    threads
                };
                // R3.2: unify load-bearing semantics across --load-bearing /
                // --no-load-bearing / --fish-no-load-bearing. Any of the
                // negation flags wins.
                let lb_effective = load_bearing && !fish_no_load_bearing;
                let spec = FishReverseSpec {
                    target_size: fish_size as usize,
                    target_tier: sudoku_rs_core::generic::techniques::Tier::T3,
                    clue_min,
                    clue_max,
                    // R3.2 (item #7): wire --max-attempts through to fish spec
                    // (was hardcoded 0). 0 still means unbounded.
                    max_attempts,
                    require_load_bearing: lb_effective,
                    greedy_max_trials: 0,
                };
                if let Err(e) = spec.validate() {
                    eprintln!("error: fish-spec validation: {}", e);
                    std::process::exit(2);
                }
                let t0 = std::time::Instant::now();
                let kept =
                    run_fish_reverse_mode(n_, br_, bc_, &spec, target, workers, seed, &out_path)?;
                let dt = t0.elapsed();
                eprintln!(
                    "reverse-construct (fish K={}, size={}x{}x{}): kept={} elapsed={:.2}s pps={:.3} -> {}",
                    fish_size,
                    n_,
                    br_,
                    bc_,
                    kept,
                    dt.as_secs_f64(),
                    kept as f64 / dt.as_secs_f64().max(1e-9),
                    out_path.display(),
                );
                return Ok(());
            }
            // ---- Generic mode (R2.2.2): triggered by --size + --output. ----
            if let (Some(size_str), Some(out_path)) = (size.as_deref(), output.clone()) {
                let (n, br, bc) = match parse_size(size_str) {
                    Ok(v) => v,
                    Err(e) => { eprintln!("error: --size: {}", e); std::process::exit(2); }
                };
                let workers = if threads == 0 {
                    if num_workers == 0 {
                        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1)
                    } else {
                        num_workers
                    }
                } else {
                    threads
                };
                let num = target;
                let tech_norm = technique.trim().to_ascii_lowercase().replace('_', "-");
                let t0 = std::time::Instant::now();
                let kept = match tech_norm.as_str() {                    "naked-quad" | "nakedquad" | "naked_quad" => {
                        let spec = NakedQuadReverseSpec {
                            target_tier: sudoku_rs_core::generic::techniques::Tier::T2,
                            clue_min,
                            clue_max,
                            max_attempts,
                            require_load_bearing: load_bearing,
                            greedy_max_trials: 0,
                        };
                        if let Err(e) = spec.validate() {
                            eprintln!("error: naked-quad-spec validation: {}", e);
                            std::process::exit(2);
                        }
                        run_naked_quad_reverse_mode(n, br, bc, &spec, num, workers, seed, &out_path)?
                    }
                    "hidden-quad" | "hiddenquad" | "hidden_quad" => {
                        let spec = HiddenQuadReverseSpec {
                            target_tier: sudoku_rs_core::generic::techniques::Tier::T2,
                            clue_min,
                            clue_max,
                            max_attempts,
                            require_load_bearing: load_bearing,
                            greedy_max_trials: 0,
                        };
                        if let Err(e) = spec.validate() {
                            eprintln!("error: hidden-quad-spec validation: {}", e);
                            std::process::exit(2);
                        }
                        run_hidden_quad_reverse_mode(n, br, bc, &spec, num, workers, seed, &out_path)?
                    }
                    "als-xz" | "alsxz" | "als" => {
                        let spec = AlsXzReverseSpec {
                            target_tier: sudoku_rs_core::generic::techniques::Tier::T3,
                            clue_min,
                            clue_max,
                            als_size_min,
                            als_size_max,
                            max_attempts,
                            require_load_bearing: load_bearing,
                            greedy_max_trials: 0,
                        };
                        if let Err(e) = spec.validate() {
                            eprintln!("error: als-xz-spec validation: {}", e);
                            std::process::exit(2);
                        }
                        run_als_xz_reverse_mode(n, br, bc, &spec, num, workers, seed, &out_path)?
                    }
                    "ur-type2" | "urtype2" | "ur2" => {
                        let spec = UrType2ReverseSpec {
                            target_tier: sudoku_rs_core::generic::techniques::Tier::T3,
                            clue_min,
                            clue_max,
                            max_attempts,
                            require_load_bearing: load_bearing,
                            greedy_max_trials: 0,
                        };
                        if let Err(e) = spec.validate() {
                            eprintln!("error: ur-type2-spec validation: {}", e);
                            std::process::exit(2);
                        }
                        run_ur_type2_reverse_mode(n, br, bc, &spec, num, workers, seed, &out_path)?
                    }
                    "aic" => {
                        let spec = AicReverseSpec {
                            target_tier: sudoku_rs_core::generic::techniques::Tier::T3,
                            target_chain_length: chain_length,
                            chain_length_slack,
                            clue_min,
                            clue_max,
                            max_attempts,
                            require_load_bearing: load_bearing,
                            greedy_max_trials: 0,
                        };
                        if let Err(e) = spec.validate() {
                            eprintln!("error: aic-spec validation: {}", e);
                            std::process::exit(2);
                        }
                        run_aic_reverse_mode(n, br, bc, &spec, num, workers, seed, &out_path)?
                    }
                    other => {
                        eprintln!(
                            "error: --technique '{}' not supported in generic mode (expected: aic | ur-type2 | als-xz | naked-quad | hidden-quad)",
                            other
                        );
                        std::process::exit(2);
                    }
                };
                let dt = t0.elapsed();
                eprintln!(
                    "reverse-construct (generic, size={}, technique={}): kept={} elapsed={:.2}s pps={:.3} -> {}",
                    size_str,
                    technique,
                    kept,
                    dt.as_secs_f64(),
                    kept as f64 / dt.as_secs_f64().max(1e-9),
                    out_path.display(),
                );
                return Ok(());
            }
            // R3.1a: legacy 9×9-only path retired. --size + --output is now mandatory.
            // R3.2: also: missing --size for generic-mode techniques is now
            // a fast-fail (rather than silent no-op).
            let _ = (
                &chain_length, &chain_length_slack, &target_count, &output_dir,
                &num_workers, &max_attempts_per_puzzle, &technique, &no_load_bearing,
            );
            eprintln!(
                "error: reverse-construct requires --size NxBRxBC and --output FILE (legacy 9×9-only mode retired in R3.1a)\n\
                 hint: prefer `gen-dataset --mode reverse --size NxBRxBC --required <tech> --output FILE`"
            );
            std::process::exit(2);
        }
        Cmd::BenchReverseHitrate {
            size,
            target_tier,
            required,
            excluded,
            attempts,
            threads,
            target_clues,
            seed,
            json_out,
        } => {
            run_bench_reverse_hitrate(
                &size, &target_tier, required, excluded, attempts, threads, &target_clues, seed,
                json_out,
            )?;
        }
        Cmd::ReRate {
            input_glob,
            output_dir,
            tdoku_bin,
            num_workers,
            shard_size,
            size,
            output,
            mode,
            progress,
        } => {
            let _mode_parsed = match GSolverMode::parse(&mode) {
                Ok(m) => m,
                Err(e) => { eprintln!("error: --mode: {}", e); std::process::exit(2); }
            };
            // Expand all globs; deduplicate paths.
            let mut all_paths: Vec<PathBuf> = Vec::new();
            for g in &input_glob {
                let paths = expand_glob(g)?;
                if paths.is_empty() {
                    eprintln!("warning: glob '{}' matched zero files", g);
                }
                all_paths.extend(paths);
            }
            all_paths.sort();
            all_paths.dedup();
            if all_paths.is_empty() {
                eprintln!("error: no input shards matched any --input-glob");
                std::process::exit(2);
            }
            // Generic mode is triggered by --size + --output. Default (no
            // flags) preserves the byte-identical 9×9 legacy path.
            if let (Some(size_str), Some(out_path)) = (size.as_deref(), output.clone()) {
                let (n, br, bc) = match parse_size(size_str) {
                    Ok(v) => v,
                    Err(e) => { eprintln!("error: --size: {}", e); std::process::exit(2); }
                };
                eprintln!("re-rate (generic, size={}): {} input file(s)", size_str, all_paths.len());
                // Optional tqdm progress bar driven by an AtomicUsize counter
                // that the rerate par_iter increments. Watcher thread polls
                // the counter every 100ms and updates the bar; we don't know
                // the input row count up-front (read happens inside `run`),
                // so total stays unknown — tqdm renders count + rate + elapsed.
                let progress_counter: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>> =
                    if progress { Some(std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0))) } else { None };
                let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let watcher = if let Some(ref counter) = progress_counter {
                    let counter = counter.clone();
                    let stop = stop_flag.clone();
                    Some(std::thread::spawn(move || {
                        let mut pb = tqdm::pbar(None).desc(Some("re-rate"));
                        let mut last: usize = 0;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            let cur = counter.load(std::sync::atomic::Ordering::Relaxed);
                            if cur > last {
                                let _ = pb.update(cur - last);
                                last = cur;
                            }
                        }
                        // Drain final delta after stop.
                        let cur = counter.load(std::sync::atomic::Ordering::Relaxed);
                        if cur > last {
                            let _ = pb.update(cur - last);
                        }
                        let _ = pb.close();
                    }))
                } else {
                    None
                };
                let cfg = RerateGenericConfig {
                    n, br, bc,
                    input_paths: all_paths,
                    output: out_path.clone(),
                    num_workers,
                    mode: _mode_parsed,
                    progress: progress_counter.clone(),
                };
                let t0 = std::time::Instant::now();
                let stats = run_rerate_generic(&cfg)?;
                stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                if let Some(h) = watcher { let _ = h.join(); }
                let dt = t0.elapsed();
                eprintln!(
                    "re-rate: input_rows={} parse_errors={} rater_errors={} kept={} elapsed={:.2}s rps={:.1} -> {}",
                    stats.n_input_rows,
                    stats.n_parse_errors,
                    stats.n_rater_errors,
                    stats.n_kept,
                    dt.as_secs_f64(),
                    stats.n_input_rows as f64 / dt.as_secs_f64().max(1e-9),
                    out_path.display(),
                );
                eprintln!("re-rate counters: {:?}", stats.counters);
                return Ok(());
            }
            // R3.1a: legacy 9×9-only path retired. --size + --output now mandatory.
            let _ = (&output_dir, &tdoku_bin, &shard_size, &num_workers, &all_paths);
            eprintln!("error: re-rate requires --size NxBRxBC and --output FILE (legacy 9×9-only mode retired in R3.1a)");
            std::process::exit(2);
        }
        Cmd::GenText {
            num,
            output,
            size,
            required,
            excluded,
            match_mode,
            max_technique,
            max_tier: max_tier_arg,
            excluded_all_others,
            target_tier,
            target_clues,
            seed,
            threads,
            include_solution,
            max_attempts_per,
            load_bearing,
            no_load_bearing,
        } => {
            // --no-load-bearing overrides --load-bearing.
            let load_bearing = if no_load_bearing { false } else { load_bearing };
            let (n, br, bc) = match parse_size(&size) {
                Ok(v) => v,
                Err(e) => { eprintln!("error: --size: {}", e); std::process::exit(2); }
            };
            let tier = match parse_tier(&target_tier) {
                Ok(t) => t,
                Err(e) => { eprintln!("error: --target-tier: {}", e); std::process::exit(2); }
            };
            let (clue_min, clue_max) = match target_clues.as_deref() {
                Some(r) => match parse_clue_range(r) {
                    Ok(v) => v,
                    Err(e) => { eprintln!("error: --target-clues: {}", e); std::process::exit(2); }
                },
                None => {
                    let cells = (n * n) as f64;
                    let (lo_frac, hi_frac) = match tier {
                        GTier::T1 => (0.40, 0.55),
                        GTier::T2 => (0.30, 0.42),
                        GTier::T3 => (0.27, 0.36),
                        GTier::T4Plus => (0.20, 0.32),
                    };
                    ((lo_frac * cells).round() as u32, (hi_frac * cells).round() as u32)
                }
            };
            // R3.4 Stage 4: early intercept for nested-fc-l2/l3/l4 BEFORE
            // the generic technique-id parse. Dispatches to seed-anchored
            // guided-constructive synthesis in nested_fc_reverse.
            {
                // Recognise --required nested_fc_l2 | nested_fc_l3 | nested_fc_l4
                // (and canonical aliases: nested_forcing_chain, nested_forcing_chain_l3, etc.)
                let req_norm: Vec<String> = required
                    .iter()
                    .map(|s| s.trim().to_ascii_lowercase().replace('-', "_"))
                    .collect();
                let nested_fc_level: Option<u8> = if req_norm.len() == 1 {
                    match req_norm[0].as_str() {
                        "nested_fc_l2" | "nested_forcing_chain" | "nfc" | "nested_fc" => Some(2),
                        "nested_fc_l3" | "nested_forcing_chain_l3" | "nfc_l3" => Some(3),
                        "nested_fc_l4" | "nested_forcing_chain_l4" | "nfc_l4" => Some(4),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(level) = nested_fc_level {
                    // target_se defaults from level if not supplied via --target-se
                    // (we expose it via environment variable NESTED_FC_TARGET_SE or
                    // derive from the level).
                    let default_se = match level {
                        4 => 10.5_f64,
                        3 => 10.0,
                        _ => 9.5,
                    };
                    // Allow override via env var NESTED_FC_TARGET_SE.
                    let target_se: f64 = std::env::var("NESTED_FC_TARGET_SE")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(default_se);
                    // Seed path: default to forum_hardest_1905_11plus.txt relative
                    // to crate root, or override via NESTED_FC_SEED_PATH.
                    let seed_path: Option<std::path::PathBuf> =
                        std::env::var("NESTED_FC_SEED_PATH")
                        .ok()
                        .map(std::path::PathBuf::from)
                        .or_else(|| {
                            let default = std::path::PathBuf::from(
                                "data/seeds/public_hardest/forum_hardest_1905/forum_hardest_1905_11plus.txt"
                            );
                            if default.exists() { Some(default) } else { None }
                        });
                    let max_seeds: usize = std::env::var("NESTED_FC_MAX_SEEDS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    let attempts_per_seed: u32 = match level {
                        4 => 1000,
                        3 => 500,
                        _ => 200,
                    };
                    // Resolve worker thread count: respect `--threads`, with
                    // `0` meaning auto-detect (rayon-style: physical cores).
                    let effective_threads: usize = if threads > 0 {
                        threads as usize
                    } else {
                        std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(1)
                    };
                    let spec = NestedFcReverseSpec {
                        target_level: level,
                        target_se,
                        attempts_per_seed,
                        rng_seed: seed,
                        seed_path,
                        max_seeds,
                        relaxed_frontier: false,
                        wall_time_budget_secs: None,
                        threads: effective_threads,
                    };
                    if (n, br, bc) != (9, 3, 3) {
                        eprintln!("error: nested-fc-l{} is only supported for 9x3x3 in v1", level);
                        std::process::exit(2);
                    }
                    let t0 = std::time::Instant::now();
                    let results = batch_nested_fc_reverse_construct::<9, 3, 3>(&spec, num);
                    let lines: Vec<String> = results.iter().map(|e| {
                        if include_solution {
                            format!("{} {}", e.puzzle, e.solution)
                        } else {
                            e.puzzle.clone()
                        }
                    }).collect();
                    let out_disp = output.display().to_string();
                    if out_disp == "-" {
                        use std::io::Write as _;
                        let stdout = std::io::stdout();
                        let mut h = stdout.lock();
                        for l in &lines { writeln!(h, "{}", l)?; }
                    } else {
                        use std::io::Write as _;
                        if let Some(p) = output.parent() {
                            if !p.as_os_str().is_empty() { std::fs::create_dir_all(p)?; }
                        }
                        let f = std::fs::File::create(&output)?;
                        let mut bw = std::io::BufWriter::new(f);
                        for l in &lines { writeln!(bw, "{}", l)?; }
                        bw.flush()?;
                    }
                    let dt = t0.elapsed();
                    eprintln!(
                        "gen-text nested-fc-l{} (9x3x3): wrote={} elapsed={:.2}s pps={:.4} target_se={} -> {}",
                        level, lines.len(), dt.as_secs_f64(),
                        lines.len() as f64 / dt.as_secs_f64().max(1e-9),
                        target_se,
                        out_disp,
                    );
                    return Ok(());
                }
            }
            // R3.4 Stage 3b: early intercept for nested-aic BEFORE the
            // generic technique-id parse (nested-aic is not a TechniqueId).
            {
                let is_nested_aic = required.len() == 1
                    && matches!(
                        required[0].trim().to_ascii_lowercase().replace('-', "_").as_str(),
                        "nested_aic"
                    );
                if is_nested_aic {
                    let workers = if threads == 0 {
                        std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1)
                    } else {
                        threads
                    };
                    let _ = workers; // nested_aic_batch_construct is single-threaded in v1
                    let nested_cfg = NestedAicConfig {
                        target_se_score: 7.0,
                        fragment_len_min: 5,
                        fragment_len_max: 7,
                        clue_count_target: Some((clue_min as usize, clue_max as usize)),
                        preempt_max_iters: 10,
                        max_outer_attempts: max_attempts_per as usize,
                        seed,
                        preempt_max_se: 6.5,
                    };
                    let t0 = std::time::Instant::now();
                    if (n, br, bc) != (9, 3, 3) {
                        eprintln!("error: nested-aic is only supported for 9x3x3 in v1");
                        std::process::exit(2);
                    }
                    let results = nested_aic_batch_construct::<9, 3, 3>(&nested_cfg, num as usize);
                    let lines: Vec<String> = results.iter().map(|e| {
                        if include_solution {
                            format!("{} {}", e.puzzle, e.solution)
                        } else {
                            e.puzzle.clone()
                        }
                    }).collect();
                    let out_disp = output.display().to_string();
                    if out_disp == "-" {
                        use std::io::Write as _;
                        let stdout = std::io::stdout();
                        let mut h = stdout.lock();
                        for l in &lines { writeln!(h, "{}", l)?; }
                    } else {
                        use std::io::Write as _;
                        if let Some(p) = output.parent() {
                            if !p.as_os_str().is_empty() { std::fs::create_dir_all(p)?; }
                        }
                        let f = std::fs::File::create(&output)?;
                        let mut bw = std::io::BufWriter::new(f);
                        for l in &lines { writeln!(bw, "{}", l)?; }
                        bw.flush()?;
                    }
                    let dt = t0.elapsed();
                    eprintln!(
                        "gen-text nested-aic (9x3x3): wrote={} elapsed={:.2}s pps={:.3} -> {}",
                        lines.len(), dt.as_secs_f64(),
                        lines.len() as f64 / dt.as_secs_f64().max(1e-9),
                        out_disp,
                    );
                    return Ok(());
                }
            }

            // RFC dedicated intercept: routes to batch_rfc_reverse_construct
            // which uses the same underlying search but with RFC-specific metadata.
            {
                let is_rfc = required.len() == 1
                    && matches!(
                        required[0].trim().to_ascii_lowercase()
                            .replace('-', "_")
                            .replace(' ', "_")
                            .as_str(),
                        "region_fc" | "region_forcing_chain" | "regionforcingchain"
                            | "regionfc" | "rfc"
                    )
                    && matches!(
                        match_mode.trim().to_ascii_lowercase().as_str(),
                        "all_of" | "allof" | "all"
                    );
                if is_rfc {
                    let rfc_spec = RfcReverseSpec {
                        target_se: 7.6,
                        attempts_per_seed: max_attempts_per,
                        clue_min,
                        clue_max,
                        rng_seed: seed,
                        require_load_bearing: load_bearing,
                    };
                    let t0 = std::time::Instant::now();

                    fn run_rfc_reverse_mode<const N: usize, const BR: usize, const BC: usize>(
                        spec: &RfcReverseSpec,
                        num: u32,
                        include_solution: bool,
                    ) -> Vec<String> {
                        let results = batch_rfc_reverse_construct::<N, BR, BC>(spec, num);
                        results.iter().map(|cp| {
                            if include_solution {
                                format!("{} {}", cp.puzzle, cp.solution)
                            } else {
                                cp.puzzle.clone()
                            }
                        }).collect()
                    }

                    let lines: Vec<String> = match (n, br, bc) {
                        (6, 2, 3) => run_rfc_reverse_mode::<6, 2, 3>(&rfc_spec, num, include_solution),
                        (9, 3, 3) => run_rfc_reverse_mode::<9, 3, 3>(&rfc_spec, num, include_solution),
                        (12, 3, 4) => run_rfc_reverse_mode::<12, 3, 4>(&rfc_spec, num, include_solution),
                        (16, 4, 4) => run_rfc_reverse_mode::<16, 4, 4>(&rfc_spec, num, include_solution),
                        _ => {
                            eprintln!("error: unsupported size {}x{}x{}", n, br, bc);
                            std::process::exit(2);
                        }
                    };
                    let out_disp = output.display().to_string();
                    if out_disp == "-" {
                        use std::io::Write as _;
                        let stdout = std::io::stdout();
                        let mut h = stdout.lock();
                        for l in &lines { writeln!(h, "{}", l)?; }
                    } else {
                        use std::io::Write as _;
                        if let Some(p) = output.parent() {
                            if !p.as_os_str().is_empty() { std::fs::create_dir_all(p)?; }
                        }
                        let f = std::fs::File::create(&output)?;
                        let mut bw = std::io::BufWriter::new(f);
                        for l in &lines { writeln!(bw, "{}", l)?; }
                        bw.flush()?;
                    }
                    let dt = t0.elapsed();
                    eprintln!(
                        "gen-text rfc ({}x{}x{}): wrote={} elapsed={:.2}s pps={:.3} -> {}",
                        n, br, bc, lines.len(), dt.as_secs_f64(),
                        lines.len() as f64 / dt.as_secs_f64().max(1e-9),
                        out_disp,
                    );
                    return Ok(());
                }
            }

            // DFC dedicated intercept: routes to batch_dfc_reverse_construct
            // (guided-constructive, SE 9.0+). See src/generic/dfc_reverse.rs.
            {
                let is_dfc = required.len() == 1
                    && matches!(
                        required[0].trim().to_ascii_lowercase()
                            .replace('-', "_")
                            .replace(' ', "_")
                            .as_str(),
                        "dynamic_fc" | "dynamic_forcing_chain" | "dynamicforcingchain"
                            | "dfc" | "dfcl1"
                    )
                    && matches!(
                        match_mode.trim().to_ascii_lowercase().as_str(),
                        "all_of" | "allof" | "all"
                    );
                if is_dfc {
                    // Seed file resolution order:
                    //   1. DFC_SEED_PATH env var (if set)
                    //   2. NESTED_FC_SEED_PATH env var (shared with NestedFC reverse)
                    //   3. Default forum_hardest path on disk
                    // DFC hit rate from pure random ≈ 0%; seed-anchoring raises it.
                    let auto_seed: Option<std::path::PathBuf> = std::env::var("DFC_SEED_PATH")
                        .ok()
                        .map(std::path::PathBuf::from)
                        .or_else(|| std::env::var("NESTED_FC_SEED_PATH").ok().map(std::path::PathBuf::from))
                        .or_else(|| {
                            let candidate = std::path::PathBuf::from(
                                "data/seeds/public_hardest/forum_hardest_1905/forum_hardest_1905_11plus.txt"
                            );
                            if candidate.exists() { Some(candidate) } else { None }
                        });
                    if auto_seed.is_none() {
                        eprintln!(
                            "warning: no DFC seed file resolved (DFC_SEED_PATH / NESTED_FC_SEED_PATH unset and default not on disk); generator will return 0 puzzles"
                        );
                    }
                    let effective_threads_dfc: usize = if threads > 0 {
                        threads as usize
                    } else {
                        std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(1)
                    };
                    // DFC_EXCLUDE_NESTED_FC: set to "0"/"false"/"no" to disable the
                    // NestedFC-exclusion probe and rate with the full cascade
                    // instead. Default true (NestedFC excluded — see commit `50b6f98`).
                    let dfc_exclude_nested: bool = std::env::var("DFC_EXCLUDE_NESTED_FC")
                        .map(|v| !matches!(v.trim(), "0" | "false" | "no"))
                        .unwrap_or(true);
                    // DFC_TARGET_SE_UPPER: strict upper bound on SE score.
                    // - When NestedFC is excluded (default): default 9.5 (strict
                    //   DFC-top bucket [9.0, 9.5)).
                    // - When NestedFC is NOT excluded: full-cascade rates DFC
                    //   puzzles at ≥9.5 (NestedFC shadow); default to `None` so
                    //   the bucket gate doesn't reject everything. The user can
                    //   still set the env var explicitly. Codex review on `50b6f98`
                    //   flagged the CRITICAL footgun when both defaults applied.
                    // Treat NaN / negative as invalid → fall back to default with warning.
                    let env_se_upper = std::env::var("DFC_TARGET_SE_UPPER").ok();
                    let parsed_se_upper: Option<f64> = env_se_upper
                        .as_deref()
                        .and_then(|s| s.parse::<f64>().ok())
                        .filter(|x| x.is_finite() && *x > 0.0);
                    if let Some(raw) = &env_se_upper {
                        if parsed_se_upper.is_none() && raw.trim() != "inf" && raw.trim() != "none" {
                            eprintln!(
                                "warning: DFC_TARGET_SE_UPPER={:?} is not a positive finite float; falling back to default",
                                raw
                            );
                        }
                    }
                    let dfc_se_upper: Option<f64> = if let Some(v) = parsed_se_upper {
                        Some(v)
                    } else if env_se_upper.as_deref().map(str::trim) == Some("none") {
                        None
                    } else if dfc_exclude_nested {
                        Some(9.5)
                    } else {
                        // Full-cascade mode: no upper bound by default so DFC
                        // candidates aren't all rejected by NestedFC SE inflation.
                        None
                    };
                    let dfc_spec = DfcReverseSpec {
                        target_se: 9.0,
                        target_se_upper: dfc_se_upper,
                        excluded_nested_fc: dfc_exclude_nested,
                        attempts_per_seed: max_attempts_per,
                        rng_seed: seed,
                        seed_path: auto_seed,
                        clue_min,
                        clue_max,
                        // max_total_attempts = attempts per puzzle × requested count,
                        // capped at 10M to prevent infinite loops on pathological input.
                        max_total_attempts: max_attempts_per
                            .saturating_mul(num.max(1))
                            .min(10_000_000),
                        threads: effective_threads_dfc,
                    };
                    let t0 = std::time::Instant::now();

                    fn run_dfc_reverse_mode<const N: usize, const BR: usize, const BC: usize>(
                        spec: &DfcReverseSpec,
                        num: u32,
                        include_solution: bool,
                    ) -> Vec<String> {
                        let results = generic_batch_dfc_reverse::<N, BR, BC>(spec, num);
                        results.iter().map(|cp| {
                            if include_solution {
                                format!("{} {}", cp.puzzle, cp.solution)
                            } else {
                                cp.puzzle.clone()
                            }
                        }).collect()
                    }

                    let lines: Vec<String> = match (n, br, bc) {
                        (6, 2, 3) => run_dfc_reverse_mode::<6, 2, 3>(&dfc_spec, num, include_solution),
                        (9, 3, 3) => run_dfc_reverse_mode::<9, 3, 3>(&dfc_spec, num, include_solution),
                        (12, 3, 4) => run_dfc_reverse_mode::<12, 3, 4>(&dfc_spec, num, include_solution),
                        (16, 4, 4) => run_dfc_reverse_mode::<16, 4, 4>(&dfc_spec, num, include_solution),
                        _ => {
                            eprintln!("error: unsupported size {}x{}x{}", n, br, bc);
                            std::process::exit(2);
                        }
                    };
                    let out_disp = output.display().to_string();
                    if out_disp == "-" {
                        use std::io::Write as _;
                        let stdout = std::io::stdout();
                        let mut h = stdout.lock();
                        for l in &lines { writeln!(h, "{}", l)?; }
                    } else {
                        use std::io::Write as _;
                        if let Some(p) = output.parent() {
                            if !p.as_os_str().is_empty() { std::fs::create_dir_all(p)?; }
                        }
                        let f = std::fs::File::create(&output)?;
                        let mut bw = std::io::BufWriter::new(f);
                        for l in &lines { writeln!(bw, "{}", l)?; }
                        bw.flush()?;
                    }
                    let dt = t0.elapsed();
                    eprintln!(
                        "gen-text dfc ({}x{}x{}): wrote={} elapsed={:.2}s pps={:.3} -> {}",
                        n, br, bc, lines.len(), dt.as_secs_f64(),
                        lines.len() as f64 / dt.as_secs_f64().max(1e-9),
                        out_disp,
                    );
                    return Ok(());
                }
            }

            // R3.5: CFC intercept — routes to batch_cfc_reverse_construct
            // (guided-constructive, SE 8.0). See src/generic/cfc_reverse.rs.
            {
                let is_cfc = required.len() == 1
                    && matches!(
                        required[0]
                            .trim()
                            .to_ascii_lowercase()
                            .replace('-', "_")
                            .replace(' ', "_")
                            .as_str(),
                        "cell_fc"
                            | "cell_forcing_chain"
                            | "cellforcingchain"
                            | "cellfc"
                            | "cfc"
                    )
                    && matches!(
                        match_mode.trim().to_ascii_lowercase().as_str(),
                        "all_of" | "allof" | "all"
                    );
                if is_cfc {
                    let workers = if threads == 0 {
                        std::thread::available_parallelism()
                            .map(|x| x.get())
                            .unwrap_or(1)
                    } else {
                        threads
                    };
                    let cfc_spec = CfcReverseSpec {
                        target_se: 8.0,
                        target_k_min: 3,
                        clue_min,
                        clue_max,
                        attempts_per_seed: max_attempts_per.min(500),
                        max_seeds: max_attempts_per.max(500),
                        rng_seed: seed,
                        // CFC defaults to require_load_bearing=true. Pass
                        // --no-load-bearing to disable the probe for throughput.
                        require_load_bearing: !no_load_bearing,
                        target_se_upper: None,
                        excluded_techniques: vec![],
                    };
                    if let Err(e) = cfc_spec.validate() {
                        eprintln!("error: cfc-spec validation: {}", e);
                        std::process::exit(2);
                    }
                    let t0 = std::time::Instant::now();
                    let lines = run_cfc_reverse_mode(
                        n, br, bc, &cfc_spec, num, workers, seed, include_solution,
                    );
                    let out_disp = output.display().to_string();
                    if out_disp == "-" {
                        use std::io::Write as _;
                        let stdout = std::io::stdout();
                        let mut h = stdout.lock();
                        for l in &lines {
                            writeln!(h, "{}", l)?;
                        }
                    } else {
                        use std::io::Write as _;
                        if let Some(p) = output.parent() {
                            if !p.as_os_str().is_empty() {
                                std::fs::create_dir_all(p)?;
                            }
                        }
                        let f = std::fs::File::create(&output)?;
                        let mut bw = std::io::BufWriter::new(f);
                        for l in &lines {
                            writeln!(bw, "{}", l)?;
                        }
                        bw.flush()?;
                    }
                    let dt = t0.elapsed();
                    eprintln!(
                        "gen-text cfc ({}x{}x{}): wrote={} elapsed={:.2}s pps={:.3} -> {}",
                        n,
                        br,
                        bc,
                        lines.len(),
                        dt.as_secs_f64(),
                        lines.len() as f64 / dt.as_secs_f64().max(1e-9),
                        out_disp,
                    );
                    return Ok(());
                }
            }

            let req_ids: Vec<GTechniqueId> = match required.iter().map(|s| parse_generic_tech(s)).collect::<Result<Vec<_>, _>>() {
                Ok(v) => v,
                Err(e) => { eprintln!("error: --required: {}", e); std::process::exit(2); }
            };
            let mut exc_ids: Vec<GTechniqueId> = match excluded.iter().map(|s| parse_generic_tech(s)).collect::<Result<Vec<_>, _>>() {
                Ok(v) => v,
                Err(e) => { eprintln!("error: --excluded: {}", e); std::process::exit(2); }
            };
            // Mutex enforcement.
            let shortcuts_used = max_technique.is_some() as u8 + max_tier_arg.is_some() as u8
                + excluded_all_others as u8;
            if shortcuts_used > 1 {
                eprintln!("error: --max-technique / --max-tier / --excluded-all-others are mutually exclusive");
                std::process::exit(2);
            }
            if shortcuts_used > 0 && !exc_ids.is_empty() {
                eprintln!("error: shortcut flags cannot be combined with explicit --excluded");
                std::process::exit(2);
            }
            if let Some(mt) = max_technique.as_deref() {
                let mt_id = match parse_generic_tech(mt) {
                    Ok(v) => v,
                    Err(e) => { eprintln!("error: --max-technique: {}", e); std::process::exit(2); }
                };
                exc_ids = excluded_from_max_technique(mt_id, &req_ids);
            } else if let Some(mt) = max_tier_arg.as_deref() {
                let mt_tier = match parse_tier(mt) {
                    Ok(v) => v,
                    Err(e) => { eprintln!("error: --max-tier: {}", e); std::process::exit(2); }
                };
                exc_ids = excluded_from_max_tier(mt_tier, &req_ids);
            } else if excluded_all_others {
                if req_ids.is_empty() {
                    eprintln!("error: --excluded-all-others requires at least one --required");
                    std::process::exit(2);
                }
                exc_ids = excluded_all_others_fn(&req_ids);
            }
            for r in &req_ids {
                if exc_ids.contains(r) {
                    eprintln!("error: '{}' is in both --required and --excluded", generic_tech_str(*r));
                    std::process::exit(2);
                }
            }
            let mode_norm = match_mode.trim().to_ascii_lowercase();
            let mm = match mode_norm.as_str() {
                "all_of" | "allof" | "all" => MatchMode::AllOf(req_ids.clone()),
                "any_of" | "anyof" | "any" => {
                    if req_ids.is_empty() {
                        eprintln!("error: --match-mode any_of requires --required");
                        std::process::exit(2);
                    }
                    MatchMode::AnyOf(req_ids.clone())
                }
                other => {
                    eprintln!("error: gen-text --match-mode supports all_of|any_of (got '{}')", other);
                    std::process::exit(2);
                }
            };
            let workers = if threads == 0 {
                std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1)
            } else {
                threads
            };
            let spec = GenericReverseSpec {
                target_tier: tier,
                match_mode: mm,
                excluded_techniques: exc_ids,
                load_bearing: false,
                clue_min,
                clue_max,
                max_attempts: max_attempts_per,
            };
            if let Err(e) = spec.validate() {
                eprintln!("error: spec validation: {}", e);
                std::process::exit(2);
            }
            // Run reverse and collect.
            let t0 = std::time::Instant::now();
            let lines: Vec<String> = match (n, br, bc) {
                (6, 2, 3) => run_gen_text::<6, 2, 3>(&spec, num, workers, seed, include_solution),
                (9, 3, 3) => run_gen_text::<9, 3, 3>(&spec, num, workers, seed, include_solution),
                (12, 3, 4) => run_gen_text::<12, 3, 4>(&spec, num, workers, seed, include_solution),
                (16, 4, 4) => run_gen_text::<16, 4, 4>(&spec, num, workers, seed, include_solution),
                _ => {
                    eprintln!("error: unsupported size {}x{}x{}", n, br, bc);
                    std::process::exit(2);
                }
            };
            // Write.
            let out_disp = output.display().to_string();
            if out_disp == "-" {
                use std::io::Write as _;
                let stdout = std::io::stdout();
                let mut h = stdout.lock();
                for l in &lines {
                    writeln!(h, "{}", l)?;
                }
            } else {
                use std::io::Write as _;
                if let Some(p) = output.parent() {
                    if !p.as_os_str().is_empty() {
                        std::fs::create_dir_all(p)?;
                    }
                }
                let f = std::fs::File::create(&output)?;
                let mut bw = std::io::BufWriter::new(f);
                for l in &lines {
                    writeln!(bw, "{}", l)?;
                }
                bw.flush()?;
            }
            let dt = t0.elapsed();
            eprintln!(
                "gen-text (size={}x{}x{}): wrote={} elapsed={:.2}s pps={:.3} -> {}",
                n, br, bc, lines.len(),
                dt.as_secs_f64(),
                lines.len() as f64 / dt.as_secs_f64().max(1e-9),
                out_disp,
            );
        }
        Cmd::RateBatch {
            input,
            output,
            size,
            threads,
            strict,
            progress,
            mode,
            excluded,
        } => {
            let mode = match GSolverMode::parse(&mode) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: --mode: {}", e);
                    std::process::exit(2);
                }
            };
            // Parse excluded technique ids once (used by run_rate_batch when
            // non-empty: switches the inner rate call to `rate_excluding`).
            let exc_ids: Vec<GTechniqueId> = match excluded
                .iter()
                .map(|s| parse_generic_tech(s))
                .collect::<Result<Vec<_>, _>>()
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: --excluded: {}", e);
                    std::process::exit(2);
                }
            };
            let (n, br, bc) = match parse_size(&size) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: --size: {}", e);
                    std::process::exit(2);
                }
            };
            if threads > 0 {
                if let Err(e) = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global()
                {
                    eprintln!("warn: rayon thread-pool already initialized ({}); using existing", e);
                }
            }
            let t0 = std::time::Instant::now();
            let (total, errors) = match (n, br, bc) {
                (6, 2, 3) => {
                    let src: TextFileSource<6> = TextFileSource::open(&input)?;
                    run_rate_batch::<6, 2, 3>(src, &output, strict, progress, mode, &exc_ids)?
                }
                (9, 3, 3) => {
                    let src: TextFileSource<9> = TextFileSource::open(&input)?;
                    run_rate_batch::<9, 3, 3>(src, &output, strict, progress, mode, &exc_ids)?
                }
                (16, 4, 4) => {
                    let src: TextFileSource<16> = TextFileSource::open(&input)?;
                    run_rate_batch::<16, 4, 4>(src, &output, strict, progress, mode, &exc_ids)?
                }
                _ => {
                    eprintln!(
                        "error: --size {}x{}x{} not supported in rate-batch (R3.0a: 6x2x3, 9x3x3, 16x4x4; 12x3x4 deferred to R3.0a')",
                        n, br, bc
                    );
                    std::process::exit(2);
                }
            };
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "rate-batch: total={} errors={} elapsed={:.2}s pps={:.1}",
                total,
                errors,
                dt,
                total as f64 / dt.max(1e-9)
            );
        }
        Cmd::RateShcBatch {
            input,
            output,
            max_length,
            buffer_size,
            threads,
            classification,
        } => {
            use std::io::Write;
            use sudoku_rs_core::shc::{
                rate_b, rate_bxb, rate_bxbb, rate_te_depth, Board, RateError,
            };

            #[derive(Clone, Copy)]
            enum ClassMode {
                B,
                TeDepth,
                BxB,
                BxBB,
            }
            let mode = match classification.as_str() {
                "b" => ClassMode::B,
                "te-depth" => ClassMode::TeDepth,
                "bxb" => ClassMode::BxB,
                "bxbb" => ClassMode::BxBB,
                other => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "unknown --classification {:?} (expected b|te-depth|bxb|bxbb)",
                            other
                        ),
                    ));
                }
            };

            // Read input — keep only well-formed puzzle lines (≥81 non-';'
            // chars). Skip blank lines, comments (`#`), and SHC.jar headers
            // (`*`). Any `;…` tail is stripped.
            let raw = std::fs::read_to_string(&input)?;
            let puzzles: Vec<String> = raw
                .lines()
                .filter_map(|line| {
                    let s = line.trim();
                    if s.is_empty() || s.starts_with('#') || s.starts_with('*') {
                        return None;
                    }
                    let head = s.split(';').next().unwrap_or("").trim();
                    if head.len() < 81 {
                        return None;
                    }
                    Some(head[..81].to_string())
                })
                .collect();

            if threads > 1 {
                if let Err(e) = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global()
                {
                    eprintln!(
                        "warn: rayon thread-pool already initialized ({}); using existing",
                        e
                    );
                }
            }

            #[derive(serde::Serialize)]
            struct Row<'a> {
                puzzle: &'a str,
                #[serde(skip_serializing_if = "Option::is_none")]
                b: Option<u16>,
                #[serde(skip_serializing_if = "Option::is_none")]
                te_depth: Option<u8>,
                #[serde(skip_serializing_if = "Option::is_none")]
                bxb: Option<u16>,
                #[serde(skip_serializing_if = "Option::is_none")]
                bxbb: Option<u16>,
                #[serde(skip_serializing_if = "Option::is_none")]
                error: Option<&'static str>,
                wall_ms: u128,
            }

            #[derive(Clone, Copy, Default)]
            struct RateOut {
                b: Option<u16>,
                te_depth: Option<u8>,
                bxb: Option<u16>,
                bxbb: Option<u16>,
                err: Option<&'static str>,
                wall_ms: u128,
            }

            let err_str = |e: &RateError| -> &'static str {
                match e {
                    RateError::Malformed => "malformed",
                    RateError::Unclassifiable => "unclassifiable",
                    RateError::BufferOverflow => "buffer_overflow",
                }
            };

            let rate_one = |p: &str| -> RateOut {
                let t0 = std::time::Instant::now();
                let mut out = RateOut::default();
                let mut board = match Board::from_81_chars(p) {
                    Ok(b) => b,
                    Err(_) => {
                        out.err = Some("parse_error");
                        out.wall_ms = t0.elapsed().as_millis();
                        return out;
                    }
                };
                match mode {
                    ClassMode::B => match rate_b(&mut board, max_length, buffer_size) {
                        Ok(b) => out.b = Some(b),
                        Err(e) => out.err = Some(err_str(&e)),
                    },
                    ClassMode::TeDepth => match rate_te_depth(&mut board, max_length, buffer_size) {
                        Ok(d) => out.te_depth = Some(d),
                        Err(e) => out.err = Some(err_str(&e)),
                    },
                    ClassMode::BxB => match rate_bxb(&mut board, max_length, buffer_size) {
                        Ok(v) => out.bxb = Some(v),
                        Err(e) => out.err = Some(err_str(&e)),
                    },
                    ClassMode::BxBB => match rate_bxbb(&mut board, max_length, buffer_size) {
                        Ok(v) => out.bxbb = Some(v),
                        Err(e) => out.err = Some(err_str(&e)),
                    },
                }
                out.wall_ms = t0.elapsed().as_millis();
                out
            };

            // Streaming output: process puzzles in chunks of CHUNK, rate the
            // chunk in parallel (par_iter preserves index order on .collect),
            // then write+flush the rows for that chunk to disk before moving
            // on. A tailing reader sees rows within ~chunk_wall_time of
            // production rather than at process exit. CHUNK is small enough
            // that the first batch lands on disk in well under a second even
            // for the slowest classification mode (bxbb).
            let t0 = std::time::Instant::now();
            const CHUNK: usize = 16;
            let f = std::fs::File::create(&output)?;
            // LineWriter auto-flushes on '\n' so each writeln! hits disk.
            let mut w = std::io::LineWriter::new(f);
            let mut errors = 0usize;
            for chunk in puzzles.chunks(CHUNK) {
                let rated: Vec<RateOut> = if threads > 1 {
                    chunk.par_iter().map(|p| rate_one(p)).collect()
                } else {
                    chunk.iter().map(|p| rate_one(p)).collect()
                };
                for (p, r) in chunk.iter().zip(rated.iter()) {
                    if r.err.is_some() {
                        errors += 1;
                    }
                    let row = Row {
                        puzzle: p,
                        b: r.b,
                        te_depth: r.te_depth,
                        bxb: r.bxb,
                        bxbb: r.bxbb,
                        error: r.err,
                        wall_ms: r.wall_ms,
                    };
                    writeln!(w, "{}", serde_json::to_string(&row).unwrap())?;
                }
                // Defensive: LineWriter already flushed on '\n', but make the
                // intent explicit so future BufWriter-substitutions stay
                // streaming.
                w.flush()?;
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "rate-shc-batch: total={} errors={} elapsed={:.2}s pps={:.1} threads={}",
                puzzles.len(),
                errors,
                dt,
                puzzles.len() as f64 / dt.max(1e-9),
                threads
            );
        }
        Cmd::IngestSeeds { input, output, label, solve, max } => {
            use sudoku_rs_core::io::ingest::{IngestConfig, run as run_ingest};
            // Derive label from input stem when not provided.
            let label = if label.is_empty() {
                input
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".to_string())
            } else {
                label
            };
            let cfg = IngestConfig { input, output, label, solve, max };
            run_ingest(&cfg)?;
        }
        Cmd::GenVicinity {
            seed_in,
            out,
            target_se,
            bt_prefilter,
            mute_n,
            budget_iters,
            max_outputs,
            seed,
            max_seeds,
            label,
            fitness,
            target_bxb,
            bxb_max_length,
            bxb_buffer_size,
            threads,
            mutator,
            allow_clue_count_change,
        } => {
            use sudoku_rs_core::generic::vicinity::{
                VicinityConfig, VicinityEntry, explore,
                write_vicinity_parquet, write_vicinity_manifest,
            };
            use sudoku_rs_core::generic::grid::Grid as VGrid;
            use sudoku_rs_core::generic::rater::rate as vrate;
            use sudoku_rs_core::generic::search::{
                count_solutions_up_to as csup, solve_unique as sol_uniq,
            };

            let label = if label.is_empty() {
                seed_in
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "vicinity".to_string())
            } else {
                label
            };

            // Parse mute_n list.
            let mute_set: Vec<u8> = mute_n
                .split(',')
                .filter_map(|s| s.trim().parse::<u8>().ok())
                .collect();
            if mute_set.is_empty() {
                eprintln!("error: --mute-n must be a comma-separated list of integers, e.g. '1,2'");
                std::process::exit(2);
            }

            // Read seed file.
            let ext = seed_in.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            let raw_seeds: Vec<(String, Option<String>)> = if ext == "parquet" {
                // Parquet input: read puzzle + solution columns via row iteration.
                eprintln!("gen-vicinity: reading parquet seeds from {:?}", seed_in);
                let file = std::fs::File::open(&seed_in)?;
                use parquet::file::reader::{FileReader, SerializedFileReader};
                use parquet::record::reader::RowIter;
                use parquet::record::RowAccessor;
                let reader = SerializedFileReader::new(file)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                let schema = reader.metadata().file_metadata().schema_descr().columns()
                    .iter()
                    .map(|c| c.name().to_string())
                    .collect::<Vec<_>>();
                let puzzle_col = schema.iter().position(|n| n == "puzzle").unwrap_or(0);
                let solution_col = schema.iter().position(|n| n == "solution");
                let iter = RowIter::from_file_into(Box::new(reader));
                let mut rows: Vec<(String, Option<String>)> = Vec::new();
                for row_result in iter {
                    let row = row_result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                    let puzzle = row.get_string(puzzle_col).ok().map(|s| s.clone());
                    let solution = solution_col.and_then(|c| row.get_string(c).ok().map(|s| s.clone()));
                    if let Some(p) = puzzle {
                        rows.push((p, solution));
                    }
                    if max_seeds > 0 && rows.len() >= max_seeds {
                        break;
                    }
                }
                rows
            } else {
                // Text input: one puzzle per line.
                eprintln!("gen-vicinity: reading text seeds from {:?}", seed_in);
                use std::io::{BufRead, BufReader};
                let file = std::fs::File::open(&seed_in)?;
                let reader = BufReader::new(file);
                let mut rows: Vec<(String, Option<String>)> = Vec::new();
                for line_result in reader.lines() {
                    let line = line_result?;
                    let trimmed = line.trim_end_matches(['\r', '\n', ' ', '\t']);
                    if trimmed.is_empty() || trimmed.starts_with('#') {
                        continue;
                    }
                    // Accept "puzzle solution" or just "puzzle".
                    let mut parts = trimmed.split_whitespace();
                    let puzzle_part = match parts.next() {
                        Some(p) if p.len() == 81 => {
                            // Normalize: '0' -> '.'.
                            p.chars().map(|c| if c == '0' { '.' } else { c }).collect::<String>()
                        }
                        _ => continue,
                    };
                    let solution_part = parts.next().map(|s| s.to_string());
                    rows.push((puzzle_part, solution_part));
                    if max_seeds > 0 && rows.len() >= max_seeds {
                        break;
                    }
                }
                rows
            };

            eprintln!("gen-vicinity: {} seeds loaded", raw_seeds.len());

            if fitness == "bxb" {
                if threads > 0 {
                    if let Err(e) = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build_global()
                    {
                        eprintln!(
                            "warn: rayon thread-pool already initialized ({}); using existing",
                            e
                        );
                    }
                }
                let mutator_kind = match mutator.as_str() {
                    "v1" => BxbMutator::V1,
                    "v3" => BxbMutator::V3,
                    "v4" => BxbMutator::V4,
                    _ => BxbMutator::V2,
                };
                run_gen_vicinity_bxb(
                    raw_seeds,
                    &out,
                    &label,
                    target_bxb,
                    bxb_max_length,
                    bxb_buffer_size,
                    &mute_set,
                    budget_iters,
                    max_outputs,
                    seed,
                    mutator_kind,
                    allow_clue_count_change,
                )?;
                return Ok(());
            }

            // Build VicinityEntry for each seed: solve if needed, rate.
            let mut seeds: Vec<VicinityEntry> = Vec::new();
            for (puzzle, maybe_solution) in raw_seeds {
                let grid = match VGrid::<9, 3, 3>::from_str(&puzzle) {
                    Some(g) => g,
                    None => {
                        eprintln!("warn: skipping unparseable seed: {:.20}", puzzle);
                        continue;
                    }
                };
                // Check uniqueness + solve if solution not provided.
                let solution = match maybe_solution {
                    Some(s) if s.len() == 81 => s,
                    _ => {
                        let n_sol = csup::<9, 3, 3>(&grid, 2);
                        if n_sol != 1 {
                            eprintln!("warn: seed has {} solutions, skipping", n_sol);
                            continue;
                        }
                        match sol_uniq::<9, 3, 3>(&grid) {
                            Some(sg) => sg.to_string_grid(),
                            None => {
                                eprintln!("warn: solve_unique failed for seed, skipping");
                                continue;
                            }
                        }
                    }
                };
                let r = vrate::<9, 3, 3>(&grid);
                seeds.push(VicinityEntry {
                    puzzle,
                    solution,
                    se_score: r.se_score,
                    tier: r.tier,
                    generation: 0,
                });
            }

            eprintln!("gen-vicinity: {} valid seeds after solve/rate", seeds.len());

            let cfg_v = VicinityConfig {
                target_se,
                bt_prefilter,
                mute_set,
                budget_iters,
                max_outputs,
                seed,
            };

            let seed_count = seeds.len();
            let t0 = std::time::Instant::now();
            let outputs = explore(seeds, &cfg_v);
            let dt = t0.elapsed();

            eprintln!(
                "gen-vicinity: {} puzzles emitted in {:.2}s -> {:?}",
                outputs.len(),
                dt.as_secs_f64(),
                out
            );

            // Print se_score distribution.
            if !outputs.is_empty() {
                let mut scores: Vec<f64> = outputs.iter().map(|e| e.se_score).collect();
                scores.sort_by(|a, b| a.total_cmp(b));
                let min_s = scores[0];
                let max_s = scores[scores.len() - 1];
                let mean_s = scores.iter().sum::<f64>() / scores.len() as f64;
                eprintln!("gen-vicinity: se_score min={:.2} mean={:.2} max={:.2}", min_s, mean_s, max_s);
            }

            write_vicinity_parquet(&out, &outputs)?;
            write_vicinity_manifest(&out, &label, seed_count, outputs.len())?;
        }
        Cmd::GenUaPilot {
            attempts,
            seed,
            max_ua_size,
            target_bxb,
            threads,
            output,
        } => {
            run_gen_ua_pilot(attempts, seed, max_ua_size, target_bxb, threads, &output)?;
        }
        Cmd::UaKillSwitch {
            input_solutions,
            max_ua_size,
            max,
        } => {
            run_ua_kill_switch(&input_solutions, max_ua_size, max)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gen-ua-pilot driver
// ---------------------------------------------------------------------------

fn run_gen_ua_pilot(
    attempts: u64,
    seed_base: u64,
    max_ua_size: usize,
    target_bxb: i16,
    threads: usize,
    output: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    use sudoku_rs_core::shc::ua::{construct_one_puzzle, ConstructResult};

    if threads > 1 {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            eprintln!(
                "warn: rayon thread-pool already initialized ({}); using existing",
                e
            );
        }
    }

    #[derive(serde::Serialize)]
    struct Row {
        attempt: u64,
        seed: u64,
        grid: String,
        puzzle: String,
        n_uas: usize,
        n_clues: u8,
        n_clues_pre_repair: u8,
        is_unique_pre_repair: bool,
        is_unique: bool,
        bxb_rating: i16,
        is_target_hit: bool,
        wall_ms_total: u64,
        wall_ms_grid: u64,
        wall_ms_ua_enum: u64,
        wall_ms_min_cover: u64,
        wall_ms_uniqueness: u64,
        wall_ms_rate_bxb: u64,
    }

    let t_start = std::time::Instant::now();
    let seeds: Vec<(u64, u64)> = (0..attempts)
        .map(|i| (i, seed_base.wrapping_add(i.wrapping_mul(1_000_003))))
        .collect();

    let run_one = |(attempt, s): &(u64, u64)| -> (u64, u64, ConstructResult) {
        let res = construct_one_puzzle(*s, max_ua_size);
        (*attempt, *s, res)
    };

    // Streaming output. We chunk the seed list and parallel-rate one chunk at
    // a time, writing each row to the file immediately (LineWriter auto-flush
    // on newline). par_iter().collect() preserves input order inside a chunk.
    const CHUNK: usize = 32;
    let f = std::fs::File::create(output)?;
    let mut w = std::io::LineWriter::new(f);
    let mut bxb_hist: HashMap<i16, usize> = HashMap::new();
    let mut wall_total: Vec<u64> = Vec::with_capacity(seeds.len());
    let mut wall_grid: Vec<u64> = Vec::new();
    let mut wall_ua: Vec<u64> = Vec::new();
    let mut wall_min: Vec<u64> = Vec::new();
    let mut wall_uniq: Vec<u64> = Vec::new();
    let mut wall_rate: Vec<u64> = Vec::new();
    let mut valid_count = 0usize;
    let mut hit_count = 0usize;
    let mut pre_repair_unique_count = 0usize;
    let mut clue_counts: Vec<u8> = Vec::with_capacity(seeds.len());
    let mut clue_counts_pre: Vec<u8> = Vec::with_capacity(seeds.len());
    let mut n_results = 0usize;
    for chunk in seeds.chunks(CHUNK) {
        let chunk_results: Vec<(u64, u64, ConstructResult)> = if threads > 1 {
            chunk.par_iter().map(run_one).collect()
        } else {
            chunk.iter().map(run_one).collect()
        };
        for (attempt, s, r) in &chunk_results {
        let is_hit = r.bxb_rating >= target_bxb && r.is_unique;
        if r.is_unique {
            valid_count += 1;
        }
        if r.is_unique_pre_repair {
            pre_repair_unique_count += 1;
        }
        clue_counts.push(r.clue_count);
        clue_counts_pre.push(r.clue_count_pre_repair);
        if is_hit {
            hit_count += 1;
        }
        *bxb_hist.entry(r.bxb_rating).or_insert(0) += 1;
        let total = r.wall_ms_grid
            + r.wall_ms_ua_enum
            + r.wall_ms_min_cover
            + r.wall_ms_uniqueness
            + r.wall_ms_rate_bxb;
        wall_total.push(total);
        wall_grid.push(r.wall_ms_grid);
        wall_ua.push(r.wall_ms_ua_enum);
        wall_min.push(r.wall_ms_min_cover);
        wall_uniq.push(r.wall_ms_uniqueness);
        wall_rate.push(r.wall_ms_rate_bxb);
        let row = Row {
            attempt: *attempt,
            seed: *s,
            grid: r.grid_str.clone(),
            puzzle: r.puzzle_str.clone(),
            n_uas: r.n_uas_found,
            n_clues: r.clue_count,
            n_clues_pre_repair: r.clue_count_pre_repair,
            is_unique_pre_repair: r.is_unique_pre_repair,
            is_unique: r.is_unique,
            bxb_rating: r.bxb_rating,
            is_target_hit: is_hit,
            wall_ms_total: total,
            wall_ms_grid: r.wall_ms_grid,
            wall_ms_ua_enum: r.wall_ms_ua_enum,
            wall_ms_min_cover: r.wall_ms_min_cover,
            wall_ms_uniqueness: r.wall_ms_uniqueness,
            wall_ms_rate_bxb: r.wall_ms_rate_bxb,
        };
            writeln!(w, "{}", serde_json::to_string(&row).unwrap())?;
        }
        // Defensive explicit flush — LineWriter already flushed on '\n'.
        w.flush()?;
        n_results += chunk_results.len();
    }
    let _ = n_results;
    let dt = t_start.elapsed();

    // Summary
    let median = |v: &mut Vec<u64>| -> u64 {
        if v.is_empty() {
            return 0;
        }
        v.sort_unstable();
        v[v.len() / 2]
    };
    let p90 = |v: &mut Vec<u64>| -> u64 {
        if v.is_empty() {
            return 0;
        }
        v.sort_unstable();
        v[(v.len() * 9 / 10).min(v.len() - 1)]
    };

    let mut bxb_keys: Vec<i16> = bxb_hist.keys().copied().collect();
    bxb_keys.sort_unstable();
    let mut dist_str = String::new();
    for k in bxb_keys {
        let cnt = bxb_hist[&k];
        dist_str.push_str(&format!("BxB={}: {}, ", k, cnt));
    }

    let med_total = median(&mut wall_total.clone());
    let med_grid = median(&mut wall_grid);
    let med_ua = median(&mut wall_ua);
    let med_min = median(&mut wall_min);
    let med_uniq = median(&mut wall_uniq);
    let med_rate = median(&mut wall_rate);
    let p90_total = p90(&mut wall_total);

    eprintln!(
        "\ngen-ua-pilot summary (output={:?}):",
        output
    );
    eprintln!("  Attempts: {}", attempts);
    eprintln!(
        "  Valid (unique post-repair): {} ({:.1}%)",
        valid_count,
        100.0 * valid_count as f64 / attempts as f64
    );
    eprintln!(
        "  Pre-repair unique: {} ({:.1}%)",
        pre_repair_unique_count,
        100.0 * pre_repair_unique_count as f64 / attempts as f64
    );
    // Median / min / max clue counts (pre and post repair).
    let median_u8 = |v: &mut Vec<u8>| -> u8 {
        if v.is_empty() { 0 } else { v.sort_unstable(); v[v.len() / 2] }
    };
    let mut clue_counts_sorted = clue_counts.clone();
    let mut clue_counts_pre_sorted = clue_counts_pre.clone();
    let med_clue = median_u8(&mut clue_counts_sorted);
    let med_clue_pre = median_u8(&mut clue_counts_pre_sorted);
    let min_clue = clue_counts.iter().copied().min().unwrap_or(0);
    let max_clue = clue_counts.iter().copied().max().unwrap_or(0);
    let min_clue_pre = clue_counts_pre.iter().copied().min().unwrap_or(0);
    let max_clue_pre = clue_counts_pre.iter().copied().max().unwrap_or(0);
    eprintln!(
        "  Clue count pre-repair:  median={} min={} max={}",
        med_clue_pre, min_clue_pre, max_clue_pre
    );
    eprintln!(
        "  Clue count post-repair: median={} min={} max={}",
        med_clue, min_clue, max_clue
    );
    eprintln!("  BxB distribution: {}", dist_str);
    eprintln!(
        "  Target hits (BxB >= {} AND unique): {} ({:.2}%)",
        target_bxb,
        hit_count,
        100.0 * hit_count as f64 / attempts as f64
    );
    eprintln!(
        "  Wall: total={:.1}s median/attempt={}ms p90/attempt={}ms",
        dt.as_secs_f64(),
        med_total,
        p90_total
    );
    eprintln!(
        "  Per-stage wall (median ms): grid={} ua_enum={} min_cover={} uniqueness={} rate_bxb={}",
        med_grid, med_ua, med_min, med_uniq, med_rate
    );
    Ok(())
}

fn run_ua_kill_switch(
    input: &std::path::Path,
    max_ua_size: usize,
    max: usize,
) -> std::io::Result<()> {
    use sudoku_rs_core::shc::ua::{
        enumerate_3digit_ua_cycles, enumerate_unavoidable_sets, full_grid_from_81,
    };
    let raw = std::fs::read_to_string(input)?;
    let mut grids: Vec<String> = Vec::new();
    for line in raw.lines() {
        let s = line.trim();
        if s.len() < 81 {
            continue;
        }
        let head = &s[..81];
        if head.chars().all(|c| ('1'..='9').contains(&c)) {
            grids.push(head.to_string());
        }
        if max > 0 && grids.len() >= max {
            break;
        }
    }
    println!("ua-kill-switch: {} grids loaded (max_ua_size={})", grids.len(), max_ua_size);
    let mut n_uas_all: Vec<usize> = Vec::with_capacity(grids.len());
    let mut n_2d_all: Vec<usize> = Vec::with_capacity(grids.len());
    let mut n_3d_all: Vec<usize> = Vec::with_capacity(grids.len());
    for (i, g) in grids.iter().enumerate() {
        let fg = full_grid_from_81(g).expect("validated above");
        let t0 = std::time::Instant::now();
        let uas_2 = enumerate_unavoidable_sets(&fg, max_ua_size);
        let dt2 = t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();
        let uas_3 = enumerate_3digit_ua_cycles(&fg, max_ua_size);
        let dt3 = t1.elapsed().as_millis();
        let total = uas_2.len() + uas_3.len();
        let mut size_hist_2: HashMap<u8, usize> = HashMap::new();
        for ua in &uas_2 {
            *size_hist_2.entry(ua.size).or_insert(0) += 1;
        }
        let mut sizes_2: Vec<u8> = size_hist_2.keys().copied().collect();
        sizes_2.sort_unstable();
        let dist_2: String = sizes_2
            .iter()
            .map(|s| format!("s{}={}", s, size_hist_2[s]))
            .collect::<Vec<_>>()
            .join(",");
        let mut size_hist_3: HashMap<u8, usize> = HashMap::new();
        for ua in &uas_3 {
            *size_hist_3.entry(ua.size).or_insert(0) += 1;
        }
        let mut sizes_3: Vec<u8> = size_hist_3.keys().copied().collect();
        sizes_3.sort_unstable();
        let dist_3: String = sizes_3
            .iter()
            .map(|s| format!("s{}={}", s, size_hist_3[s]))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "  grid[{}]: total={} 2d={} ({}) wall_2d={}ms 3d={} ({}) wall_3d={}ms",
            i, total, uas_2.len(), dist_2, dt2, uas_3.len(), dist_3, dt3
        );
        n_uas_all.push(total);
        n_2d_all.push(uas_2.len());
        n_3d_all.push(uas_3.len());
    }
    if n_uas_all.is_empty() {
        println!("ua-kill-switch: no grids — nothing to report");
        return Ok(());
    }
    let stats = |mut v: Vec<usize>| -> (usize, usize, usize, f64) {
        v.sort_unstable();
        let n = v.len();
        let mean = v.iter().sum::<usize>() as f64 / n as f64;
        (v[0], v[n / 2], v[n - 1], mean)
    };
    let (tmin, tmed, tmax, tmean) = stats(n_uas_all.clone());
    let (n2min, n2med, n2max, n2mean) = stats(n_2d_all.clone());
    let (n3min, n3med, n3max, n3mean) = stats(n_3d_all.clone());
    println!(
        "\nua-kill-switch summary (n={}):",
        n_uas_all.len()
    );
    println!(
        "  2-digit UAs:  min={} median={} max={} mean={:.1}",
        n2min, n2med, n2max, n2mean
    );
    println!(
        "  3-digit UAs:  min={} median={} max={} mean={:.1}",
        n3min, n3med, n3max, n3mean
    );
    println!(
        "  Combined:     min={} median={} max={} mean={:.1}",
        tmin, tmed, tmax, tmean
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// gen-vicinity BxB-fitness path (T&E(2)-targeted)
// ---------------------------------------------------------------------------
//
// Mirrors the SE-fitness path in src/generic/vicinity.rs but rates candidates
// via the SHC TE engine (`shc::te::rate_bxb`) instead of the CLIPS cascade.
// Mutation, uniqueness and canonical-dedup plumbing is reused from
// `generic::vicinity` (private helpers are too tied to VicinityEntry; we
// re-implement the lean BxB loop here using the same primitives).
//
// Hill-climbing rule: greedy ≥ (accept lateral plateau moves). Best-so-far is
// emitted when it first reaches `target_bxb`; subsequent improvements are
// emitted too, deduped by canonical hash.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BxbMutator {
    V1,
    V2,
    V3,
    V4,
}

#[derive(Clone, Debug)]
struct BxbEntry {
    puzzle: String,
    solution: String,
    bxb: u16,
    generation: u32,
}

fn bxb_rate_puzzle(
    puzzle: &str,
    max_length: u16,
    buffer_size: usize,
) -> Option<u16> {
    use sudoku_rs_core::shc::Board;
    use sudoku_rs_core::shc::te::rate_bxb;
    let mut b = Board::from_81_chars(puzzle).ok()?;
    rate_bxb(&mut b, max_length, buffer_size).ok()
}

/// Single mutation: pick `n` distinct clue cells and replace each digit with
/// a random digit ≠ current digit. Mirrors `generic::vicinity::mutate_n` but
/// returns a single attempt (callers loop).
fn bxb_mutate_once(
    puzzle: &str,
    n: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<String> {
    use rand::seq::SliceRandom;
    let clue_idx: Vec<usize> = puzzle
        .bytes()
        .enumerate()
        .filter_map(|(i, b)| if b != b'.' && b != b'0' { Some(i) } else { None })
        .collect();
    if clue_idx.len() < n {
        return None;
    }
    let mut shuffled = clue_idx.clone();
    shuffled.shuffle(rng);
    let picked = &shuffled[..n];

    let mut bytes: Vec<u8> = puzzle.bytes().collect();
    for &cell in picked {
        let cur = bytes[cell];
        let mut cands: Vec<u8> = (b'1'..=b'9').filter(|&d| d != cur).collect();
        cands.shuffle(rng);
        bytes[cell] = cands[0];
    }
    let s = String::from_utf8(bytes).ok()?;
    if s == puzzle {
        return None;
    }
    Some(s)
}

/// Uniqueness-aware mutator v2. Performs `n` solution-aware moves in sequence.
/// Each move is one of:
///   - swap_add_remove (w=0.5): reveal a non-clue cell to its solution-value AND
///     erase a clue. Net clue count preserved.
///   - remove_only (w=0.25): erase a clue. Clue count decreases.
///   - add_only (w=0.25): reveal a non-clue cell to its solution-value. Clue
///     count increases.
///
/// All "add" moves use the provided `solution` so the new clue is guaranteed
/// consistent with the canonical solution — this lets candidates have a
/// realistic chance of retaining uniqueness (vs the v1 naive operator which
/// fails uniqueness ~99% of the time at radius ≥4).
///
/// Returns `None` if no move could be made (e.g. clue erase on an already-empty
/// grid) or if the final puzzle equals the input.
fn bxb_mutate_once_v2(
    puzzle: &str,
    solution: &str,
    n: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<String> {
    use rand::Rng;
    if puzzle.len() != 81 || solution.len() != 81 {
        return None;
    }
    let sol_bytes: &[u8] = solution.as_bytes();
    let mut bytes: Vec<u8> = puzzle.bytes().collect();

    for _ in 0..n {
        // Recompute clue / non-clue index sets each step (clue set drifts).
        let mut clue_idx: Vec<usize> = Vec::with_capacity(81);
        let mut empty_idx: Vec<usize> = Vec::with_capacity(81);
        for (i, &b) in bytes.iter().enumerate() {
            if b == b'.' || b == b'0' {
                empty_idx.push(i);
            } else {
                clue_idx.push(i);
            }
        }

        let r: f64 = rng.gen();
        // Effective weights, accounting for degenerate cases.
        let want_swap = r < 0.5;
        let want_remove = !want_swap && r < 0.75;
        // else want_add

        // Resolve to an executable move with fallback when the chosen kind is
        // not possible.
        enum Move { Swap, Remove, Add }
        let mv: Option<Move> = if want_swap {
            if !clue_idx.is_empty() && !empty_idx.is_empty() {
                Some(Move::Swap)
            } else if !clue_idx.is_empty() {
                Some(Move::Remove)
            } else if !empty_idx.is_empty() {
                Some(Move::Add)
            } else {
                None
            }
        } else if want_remove {
            if !clue_idx.is_empty() {
                Some(Move::Remove)
            } else if !empty_idx.is_empty() {
                Some(Move::Add)
            } else {
                None
            }
        } else {
            if !empty_idx.is_empty() {
                Some(Move::Add)
            } else if !clue_idx.is_empty() {
                Some(Move::Remove)
            } else {
                None
            }
        };

        match mv {
            Some(Move::Swap) => {
                let c_add = empty_idx[rng.gen_range(0..empty_idx.len())];
                let c_remove = clue_idx[rng.gen_range(0..clue_idx.len())];
                let v = sol_bytes[c_add];
                if !(b'1'..=b'9').contains(&v) {
                    return None;
                }
                bytes[c_add] = v;
                bytes[c_remove] = b'.';
            }
            Some(Move::Remove) => {
                let c_remove = clue_idx[rng.gen_range(0..clue_idx.len())];
                bytes[c_remove] = b'.';
            }
            Some(Move::Add) => {
                let c_add = empty_idx[rng.gen_range(0..empty_idx.len())];
                let v = sol_bytes[c_add];
                if !(b'1'..=b'9').contains(&v) {
                    return None;
                }
                bytes[c_add] = v;
            }
            None => return None,
        }
    }

    let s = String::from_utf8(bytes).ok()?;
    if s == puzzle {
        return None;
    }
    Some(s)
}

// ---------------------------------------------------------------------------
// BRT-correct mutator v3 (mith/Methuselah procedure).
//
// brt_vicinity_step(P, G, p, q):
//   1. P_ext  = brt_expand(P, G)       // add every cell derivable via
//                                         Naked/Hidden Singles from P.
//   2. P_mut  = go_p_q(P_ext, p, q)    // remove p clue-cells; add q non-clue
//                                         cells from G.
//   3. P_min  = reminimize(P_mut)      // iteratively remove redundant clues.
//   4. return P_min if unique else None.
//
// The expansion to a BRT-closed neighbourhood is what makes the operator
// BRT-equivalent: classification functions are continuous in the BRTinc
// topology, so a step of this shape preserves (or only slightly shifts) the
// BxB rating with high probability.
// ---------------------------------------------------------------------------

/// Iteratively apply Naked + Hidden Singles starting from `puzzle`. For each
/// cell that becomes assigned via singles, write the corresponding `solution`
/// digit into the output. Returns the extended 81-char puzzle string (same
/// solution, ≥ as many clues).
///
/// This uses `shc::tb::propagate(L0)` — TB.L0 fires NAKED + HIDDEN singles
/// only (no box/line, no higher-order eliminations) — so the closure is
/// exactly the BRT singles-closure used by mith's procedure.
fn brt_expand(puzzle: &str, solution: &str) -> String {
    use sudoku_rs_core::shc::tb::{propagate, TbLevel};
    use sudoku_rs_core::shc::Board;

    if puzzle.len() != 81 || solution.len() != 81 {
        return puzzle.to_string();
    }
    let mut board = match Board::from_81_chars(puzzle) {
        Ok(b) => b,
        Err(_) => return puzzle.to_string(),
    };
    // Singles-only cascade. Any contradiction / solved-state still leaves
    // `cell.assigned` populated for every derivation, so we just read them
    // back regardless of the outcome.
    let _ = propagate(&mut board, TbLevel::L0);

    let in_bytes = puzzle.as_bytes();
    let sol_bytes = solution.as_bytes();
    let mut out: Vec<u8> = in_bytes.to_vec();
    for i in 0..81 {
        let was_clue = in_bytes[i] != b'.' && in_bytes[i] != b'0';
        if was_clue {
            continue;
        }
        if board.cells[i].assigned.is_some() {
            // Derived via singles. Write the canonical solution digit (not
            // `assigned.unwrap()+1`) to keep this consistent with the same
            // canonical solution even in pathological inputs.
            let v = sol_bytes[i];
            if (b'1'..=b'9').contains(&v) {
                out[i] = v;
            }
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| puzzle.to_string())
}

/// Remove `p` random clue cells and add `q` random non-clue cells (revealed
/// to the solution value at each chosen index). Returns the new puzzle, or
/// `None` if the puzzle doesn't have enough clue / non-clue positions.
fn go_p_q(
    puzzle: &str,
    solution: &str,
    p: usize,
    q: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<String> {
    use rand::seq::SliceRandom;
    if puzzle.len() != 81 || solution.len() != 81 {
        return None;
    }
    let mut clue_idx: Vec<usize> = Vec::with_capacity(81);
    let mut empty_idx: Vec<usize> = Vec::with_capacity(81);
    for (i, &b) in puzzle.as_bytes().iter().enumerate() {
        if b == b'.' || b == b'0' {
            empty_idx.push(i);
        } else {
            clue_idx.push(i);
        }
    }
    if clue_idx.len() < p || empty_idx.len() < q {
        return None;
    }
    let sol_bytes = solution.as_bytes();
    let mut bytes: Vec<u8> = puzzle.bytes().collect();

    clue_idx.shuffle(rng);
    for &c in &clue_idx[..p] {
        bytes[c] = b'.';
    }
    empty_idx.shuffle(rng);
    for &c in &empty_idx[..q] {
        let v = sol_bytes[c];
        if !(b'1'..=b'9').contains(&v) {
            return None;
        }
        bytes[c] = v;
    }
    String::from_utf8(bytes).ok()
}

/// Iteratively try to remove each clue cell. If removal preserves uniqueness
/// (csup == 1), commit. Repeat passes until a full pass finds nothing
/// removable. Returns the minimal puzzle. If input itself is non-unique,
/// returns `None`.
fn reminimize(puzzle: &str, rng: &mut Xoshiro256PlusPlus) -> Option<String> {
    use rand::seq::SliceRandom;
    use sudoku_rs_core::generic::grid::Grid as VGrid;
    use sudoku_rs_core::generic::search::count_solutions_up_to as csup;
    if puzzle.len() != 81 {
        return None;
    }
    let grid0 = VGrid::<9, 3, 3>::from_str(puzzle)?;
    if csup::<9, 3, 3>(&grid0, 2) != 1 {
        return None;
    }

    let mut bytes: Vec<u8> = puzzle.bytes().collect();
    loop {
        let mut clue_idx: Vec<usize> = bytes
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b != b'.' && b != b'0' { Some(i) } else { None })
            .collect();
        clue_idx.shuffle(rng);
        let mut removed_any = false;
        for c in clue_idx {
            let saved = bytes[c];
            bytes[c] = b'.';
            let cand_str = match std::str::from_utf8(&bytes) {
                Ok(s) => s,
                Err(_) => {
                    bytes[c] = saved;
                    continue;
                }
            };
            let g = match VGrid::<9, 3, 3>::from_str(cand_str) {
                Some(g) => g,
                None => {
                    bytes[c] = saved;
                    continue;
                }
            };
            if csup::<9, 3, 3>(&g, 2) == 1 {
                removed_any = true;
                // Commit (bytes already mutated).
            } else {
                // Restore.
                bytes[c] = saved;
            }
        }
        if !removed_any {
            break;
        }
    }
    String::from_utf8(bytes).ok()
}

/// Tridagon-aware mutator v4 (Eleven's replacement). Returns both the new
/// puzzle and the new solution (digit-relabel changes both).
///
/// Procedure:
///   1. detect_tridagons(solution); if empty → None.
///   2. Pick a random detected tridagon T.
///   3. Pick a random disjoint 3-digit subset of {1..9}\T.triplet → new_triplet.
///   4. Pick a random 3! match permutation between old & new triplet.
///   5. Apply eleven_replace.
///   6. Verify the result is unique-solution; if not → None.
///   7. Verify the new puzzle is not identical to input → None on identity.
fn bxb_mutate_once_v4(
    puzzle: &str,
    solution: &str,
    _n: usize, // unused — v4 is a structural relabel, not a per-cell op
    rng: &mut Xoshiro256PlusPlus,
) -> Option<(String, String)> {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use sudoku_rs_core::generic::grid::Grid as VGrid;
    use sudoku_rs_core::generic::search::count_solutions_up_to as csup;
    use sudoku_rs_core::shc::tridagon::{detect_tridagons, eleven_replace};

    let tris = detect_tridagons(solution);
    if tris.is_empty() {
        return None;
    }
    let t = &tris[rng.gen_range(0..tris.len())];

    // Pick 3 distinct digits from {1..9} \ t.triplet.
    let mut pool: Vec<u8> =
        (1u8..=9).filter(|d| !t.triplet.contains(d)).collect();
    pool.shuffle(rng);
    let new_triplet = [pool[0], pool[1], pool[2]];

    // Random match-perm.
    let mut mp = [0usize, 1, 2];
    mp.shuffle(rng);
    let match_perm = [mp[0], mp[1], mp[2]];

    let (new_p, new_s) = eleven_replace(puzzle, solution, t, new_triplet, match_perm)?;
    if new_p == puzzle {
        return None;
    }
    // Uniqueness check.
    let grid = VGrid::<9, 3, 3>::from_str(&new_p)?;
    if csup::<9, 3, 3>(&grid, 2) != 1 {
        return None;
    }
    Some((new_p, new_s))
}

/// BRT-correct mutator v3 (mith procedure). See module docs above for the
/// pseudo-code.
fn bxb_mutate_once_v3(
    puzzle: &str,
    solution: &str,
    p: usize,
    q: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<String> {
    let ext = brt_expand(puzzle, solution);
    let mut_p = go_p_q(&ext, solution, p, q, rng)?;
    let min_p = reminimize(&mut_p, rng)?;
    if min_p == puzzle {
        return None;
    }
    Some(min_p)
}

#[allow(clippy::too_many_arguments)]
fn run_gen_vicinity_bxb(
    raw_seeds: Vec<(String, Option<String>)>,
    out: &std::path::Path,
    label: &str,
    target_bxb: u16,
    bxb_max_length: u16,
    bxb_buffer_size: usize,
    mute_set: &[u8],
    budget_iters: usize,
    max_outputs: usize,
    rng_seed: u64,
    mutator: BxbMutator,
    allow_clue_count_change: bool,
) -> std::io::Result<()> {
    use std::collections::HashSet;
    use sudoku_rs_core::generic::canonical::canonical_hash;
    use sudoku_rs_core::generic::grid::Grid as VGrid;
    use sudoku_rs_core::generic::search::{
        count_solutions_up_to as csup, solve_unique as sol_uniq,
    };

    // `outputs` is populated post-loop from the streaming writer snapshot.
    let outputs: Vec<BxbEntry>;

    // Open the streaming parquet writer eagerly — the file appears on disk
    // immediately (with parquet magic bytes), so a tailing reader can poll
    // for existence right from the start of the run rather than after the
    // seed-validation step (which can itself take tens of seconds at scale).
    let stream_writer = std::sync::Arc::new(std::sync::Mutex::new(
        StreamingBxbWriter::create(out)?,
    ));

    // Build seeds: ensure unique solution, baseline BxB rating.
    // Validation per-seed is pure & local — do it in parallel, then
    // pre-populate `seen` serially after collect.
    let seeds: Vec<BxbEntry> = raw_seeds
        .into_par_iter()
        .filter_map(|(puzzle, maybe_solution)| {
            let grid = match VGrid::<9, 3, 3>::from_str(&puzzle) {
                Some(g) => g,
                None => {
                    eprintln!("warn: skipping unparseable seed: {:.20}", puzzle);
                    return None;
                }
            };
            let solution = match maybe_solution {
                Some(s) if s.len() == 81 => s,
                _ => {
                    let n_sol = csup::<9, 3, 3>(&grid, 2);
                    if n_sol != 1 {
                        eprintln!("warn: seed has {} solutions, skipping", n_sol);
                        return None;
                    }
                    match sol_uniq::<9, 3, 3>(&grid) {
                        Some(sg) => sg.to_string_grid(),
                        None => {
                            eprintln!("warn: solve_unique failed for seed, skipping");
                            return None;
                        }
                    }
                }
            };
            let bxb = match bxb_rate_puzzle(&puzzle, bxb_max_length, bxb_buffer_size) {
                Some(v) => v,
                None => {
                    eprintln!("warn: BxB rate failed for seed, skipping");
                    return None;
                }
            };
            Some(BxbEntry { puzzle, solution, bxb, generation: 0 })
        })
        .collect();

    let mut seen: HashSet<u128> = HashSet::new();
    for entry in &seeds {
        if let Some(grid) = VGrid::<9, 3, 3>::from_str(&entry.puzzle) {
            seen.insert(canonical_hash::<9, 3, 3>(&grid));
        }
    }

    eprintln!(
        "gen-vicinity[bxb]: {} valid seeds; baseline BxB min={} max={}",
        seeds.len(),
        seeds.iter().map(|e| e.bxb).min().unwrap_or(0),
        seeds.iter().map(|e| e.bxb).max().unwrap_or(0),
    );

    let seed_count = seeds.len();
    let t0 = std::time::Instant::now();

    // Parallel per-seed climb. Each worker:
    //   - owns a deterministic RNG seeded by a splitmix-style stride of the
    //     global rng_seed by seed index;
    //   - accumulates accepted candidates into a thread-local Vec;
    //   - performs its own per-worker canonical_hash dedup over the seeds it
    //     has already visited (the global pre-seed `seen` set is read-only
    //     during the climb).
    //
    // Semantic difference vs. the serial version: `max_outputs` is no longer
    // an across-seed early-stop signal — each worker runs its full
    // `budget_iters`. We dedup + truncate to `max_outputs` after the join.
    // For the baseline workloads this is acceptable; the cap is enforced
    // post-hoc.
    // Diagnostic counters shared across workers — give us visibility into
    // *why* the vicinity space is (or isn't) producing hits.
    use std::sync::atomic::{AtomicUsize, Ordering};
    let n_mutations_proposed = std::sync::Arc::new(AtomicUsize::new(0));
    let n_clue_mismatch = std::sync::Arc::new(AtomicUsize::new(0));
    let n_canonical_dup = std::sync::Arc::new(AtomicUsize::new(0));
    let n_uniqueness_fail = std::sync::Arc::new(AtomicUsize::new(0));
    let n_rated = std::sync::Arc::new(AtomicUsize::new(0));
    let n_above_target = std::sync::Arc::new(AtomicUsize::new(0));
    // v3 mutator: clue count changes by construction (reminimize). For v1/v2
    // this counter stays at 0.
    let n_clue_count_changed = std::sync::Arc::new(AtomicUsize::new(0));
    // v4 mutator diagnostics.
    let n_tridagon_found = std::sync::Arc::new(AtomicUsize::new(0));
    let n_v4_emit = std::sync::Arc::new(AtomicUsize::new(0));

    let pre_seen = std::sync::Arc::new(seen);

    // `stream_writer` was opened at function entry. Workers flush their
    // local buffer to it every BATCH_FLUSH accepted candidates (or at end
    // of seed) so partial output is on disk well before the run completes.
    const BATCH_FLUSH: usize = 64;
    // Global dedup across workers — applied at flush time so we don't ship
    // duplicates to disk.
    let global_seen_arc: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u128>>> =
        std::sync::Arc::new(std::sync::Mutex::new((*pre_seen).clone()));
    // Global cap on emitted rows — workers stop flushing once reached.
    let global_emitted = std::sync::Arc::new(AtomicUsize::new(0));

    let per_seed_outputs: Vec<Vec<BxbEntry>> = seeds
        .par_iter()
        .enumerate()
        .map(|(seed_idx, seed)| {
            let stream_writer = std::sync::Arc::clone(&stream_writer);
            let global_seen_arc = std::sync::Arc::clone(&global_seen_arc);
            let global_emitted = std::sync::Arc::clone(&global_emitted);
            let n_mutations_proposed = std::sync::Arc::clone(&n_mutations_proposed);
            let n_clue_mismatch = std::sync::Arc::clone(&n_clue_mismatch);
            let n_canonical_dup = std::sync::Arc::clone(&n_canonical_dup);
            let n_uniqueness_fail = std::sync::Arc::clone(&n_uniqueness_fail);
            let n_rated = std::sync::Arc::clone(&n_rated);
            let n_above_target = std::sync::Arc::clone(&n_above_target);
            let n_clue_count_changed = std::sync::Arc::clone(&n_clue_count_changed);
            let n_tridagon_found = std::sync::Arc::clone(&n_tridagon_found);
            let n_v4_emit = std::sync::Arc::clone(&n_v4_emit);
            let stride = (seed_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut rng =
                Xoshiro256PlusPlus::seed_from_u64(rng_seed.wrapping_add(stride));
            let mut local_outputs: Vec<BxbEntry> = Vec::new();
            let mut local_seen: HashSet<u128> = HashSet::new();
            let seed_clue_count =
                seed.puzzle.bytes().filter(|&b| b != b'.' && b != b'0').count();
            let mut current = seed.clone();
            let mut best_bxb = seed.bxb;

            if best_bxb >= target_bxb {
                local_outputs.push(seed.clone());
            }

            if mute_set.is_empty() {
                // Bug-fix: flush the seed row (if any) before the early return.
                // Without this, the seed pushed above was returned in the
                // join Vec, but the join path no longer drains worker Vecs
                // into the writer — those rows would be silently dropped.
                if let Err(e) = flush_local_to_stream::<9, 3, 3>(
                    &mut local_outputs,
                    &stream_writer,
                    &global_seen_arc,
                    &global_emitted,
                    max_outputs,
                ) {
                    eprintln!(
                        "warn: gen-vicinity[bxb] seed flush failed (mute empty): {}",
                        e
                    );
                }
                return local_outputs;
            }

            let mut iters = 0usize;
            let mut stall_n1 = 0usize;
            let stall_threshold = 10usize;

            while iters < budget_iters {
                iters += 1;

                let mute_level: u8 = if mute_set.len() == 1
                    || stall_n1 < stall_threshold
                {
                    mute_set[0]
                } else {
                    mute_set[1]
                };
                let n = mute_level as usize;

                n_mutations_proposed.fetch_add(1, Ordering::Relaxed);
                // For v1/v2/v3 the solution is invariant (no relabel).
                // For v4 the solution changes; we receive it from the mutator.
                let (cand, cand_solution): (Option<String>, Option<String>) = match mutator {
                    BxbMutator::V1 => (bxb_mutate_once(&current.puzzle, n, &mut rng), None),
                    BxbMutator::V2 => (
                        bxb_mutate_once_v2(
                            &current.puzzle,
                            &current.solution,
                            n,
                            &mut rng,
                        ),
                        None,
                    ),
                    BxbMutator::V3 => (
                        bxb_mutate_once_v3(
                            &current.puzzle,
                            &current.solution,
                            n,
                            n,
                            &mut rng,
                        ),
                        None,
                    ),
                    BxbMutator::V4 => {
                        // Pre-detect to update tridagon counter regardless of
                        // emission success.
                        use sudoku_rs_core::shc::tridagon::detect_tridagons;
                        if !detect_tridagons(&current.solution).is_empty() {
                            n_tridagon_found.fetch_add(1, Ordering::Relaxed);
                        }
                        match bxb_mutate_once_v4(
                            &current.puzzle,
                            &current.solution,
                            n,
                            &mut rng,
                        ) {
                            Some((p, s)) => {
                                n_v4_emit.fetch_add(1, Ordering::Relaxed);
                                (Some(p), Some(s))
                            }
                            None => (None, None),
                        }
                    }
                };
                let cand = match cand {
                    Some(c) => c,
                    None => {
                        // For v3 this is the "reminimize failed → non-unique
                        // after go_p_q" path; count it as a uniqueness
                        // failure so the diagnostic is comparable to v1/v2.
                        if mutator == BxbMutator::V3 {
                            n_uniqueness_fail.fetch_add(1, Ordering::Relaxed);
                        }
                        stall_n1 += 1;
                        continue;
                    }
                };

                let cand_cc =
                    cand.bytes().filter(|&b| b != b'.' && b != b'0').count();
                if mutator == BxbMutator::V3 {
                    // v3 reminimizes — clue count change is structural, not
                    // a defect. Track but never reject.
                    if cand_cc != seed_clue_count {
                        n_clue_count_changed.fetch_add(1, Ordering::Relaxed);
                    }
                } else if !allow_clue_count_change && cand_cc != seed_clue_count {
                    n_clue_mismatch.fetch_add(1, Ordering::Relaxed);
                    stall_n1 += 1;
                    continue;
                }
                // Touch to avoid an unused-variable warning when relaxed.
                let _ = cand_cc;

                let grid = match VGrid::<9, 3, 3>::from_str(&cand) {
                    Some(g) => g,
                    None => {
                        stall_n1 += 1;
                        continue;
                    }
                };
                let hash = canonical_hash::<9, 3, 3>(&grid);
                if pre_seen.contains(&hash) || local_seen.contains(&hash) {
                    n_canonical_dup.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                local_seen.insert(hash);

                if csup::<9, 3, 3>(&grid, 2) != 1 {
                    n_uniqueness_fail.fetch_add(1, Ordering::Relaxed);
                    stall_n1 += 1;
                    continue;
                }
                let solution_str = match cand_solution {
                    Some(s) => s,
                    None => match sol_uniq::<9, 3, 3>(&grid) {
                        Some(sg) => sg.to_string_grid(),
                        None => {
                            stall_n1 += 1;
                            continue;
                        }
                    },
                };

                n_rated.fetch_add(1, Ordering::Relaxed);
                let bxb = match bxb_rate_puzzle(&cand, bxb_max_length, bxb_buffer_size)
                {
                    Some(v) => v,
                    None => {
                        stall_n1 += 1;
                        continue;
                    }
                };

                if bxb >= target_bxb {
                    n_above_target.fetch_add(1, Ordering::Relaxed);
                }

                if bxb >= best_bxb {
                    if bxb > best_bxb {
                        stall_n1 = 0;
                    } else {
                        stall_n1 += 1;
                    }
                    let entry = BxbEntry {
                        puzzle: cand,
                        solution: solution_str,
                        bxb,
                        generation: current.generation + 1,
                    };
                    best_bxb = bxb;
                    current = entry.clone();
                    if bxb >= target_bxb {
                        local_outputs.push(entry);
                        // Streaming flush: when this worker has accumulated
                        // BATCH_FLUSH candidates, push them through the global
                        // dedup → shared writer so the parquet on disk grows
                        // mid-run.
                        if local_outputs.len() >= BATCH_FLUSH {
                            if let Err(e) = flush_local_to_stream::<9, 3, 3>(
                                &mut local_outputs,
                                &stream_writer,
                                &global_seen_arc,
                                &global_emitted,
                                max_outputs,
                            ) {
                                eprintln!(
                                    "warn: gen-vicinity[bxb] mid-seed flush failed: {}",
                                    e
                                );
                            }
                        }
                    }
                } else {
                    stall_n1 += 1;
                }
            }
            // End-of-seed flush.
            if let Err(e) = flush_local_to_stream::<9, 3, 3>(
                &mut local_outputs,
                &stream_writer,
                &global_seen_arc,
                &global_emitted,
                max_outputs,
            ) {
                eprintln!(
                    "warn: gen-vicinity[bxb] end-of-seed flush failed: {}",
                    e
                );
            }
            local_outputs
        })
        .collect();

    // Streaming pipeline: workers already pushed everything to
    // `stream_writer` via `flush_local_to_stream`, applying the global
    // canonical_hash dedup + max_outputs cap. `per_seed_outputs` was kept
    // around only to drive the parallel iterator; drop it.
    drop(per_seed_outputs);
    let _ = pre_seen;
    let output_count = global_emitted.load(Ordering::Relaxed).min(max_outputs);
    // `outputs` Vec retained for the summary stats below. Pull the buffered
    // entries back from the writer so we can compute min/mean/max BxB.
    {
        let w = stream_writer.lock().expect("stream_writer mutex poisoned");
        outputs = w.snapshot_entries();
    }

    let dt = t0.elapsed();
    let diag_n_mutations_proposed = n_mutations_proposed.load(Ordering::Relaxed);
    let diag_n_clue_mismatch = n_clue_mismatch.load(Ordering::Relaxed);
    let diag_n_canonical_dup = n_canonical_dup.load(Ordering::Relaxed);
    let diag_n_uniqueness_fail = n_uniqueness_fail.load(Ordering::Relaxed);
    let diag_n_rated = n_rated.load(Ordering::Relaxed);
    let diag_n_above_target = n_above_target.load(Ordering::Relaxed);
    let diag_n_clue_count_changed = n_clue_count_changed.load(Ordering::Relaxed);
    let diag_n_tridagon_found = n_tridagon_found.load(Ordering::Relaxed);
    let diag_n_v4_emit = n_v4_emit.load(Ordering::Relaxed);
    eprintln!(
        "gen-vicinity[bxb] diagnostics: mutations_proposed={} clue_mismatch={} clue_count_changed={} canonical_dup={} uniqueness_fail={} rated={} above_target={} tridagon_found={} v4_emit={}",
        diag_n_mutations_proposed,
        diag_n_clue_mismatch,
        diag_n_clue_count_changed,
        diag_n_canonical_dup,
        diag_n_uniqueness_fail,
        diag_n_rated,
        diag_n_above_target,
        diag_n_tridagon_found,
        diag_n_v4_emit,
    );
    eprintln!(
        "gen-vicinity[bxb]: {} puzzles emitted in {:.2}s -> {:?}",
        outputs.len(),
        dt.as_secs_f64(),
        out
    );
    if !outputs.is_empty() {
        let mut bs: Vec<u16> = outputs.iter().map(|e| e.bxb).collect();
        bs.sort_unstable();
        let min_b = bs[0];
        let max_b = bs[bs.len() - 1];
        let mean_b = bs.iter().map(|&b| b as f64).sum::<f64>() / bs.len() as f64;
        eprintln!("gen-vicinity[bxb]: bxb min={} mean={:.2} max={}", min_b, mean_b, max_b);
    }

    // Close the streaming writer — writes the parquet footer so the file is
    // a valid parquet. Until this point, the file exists and contains
    // row-group data but no footer.
    {
        let arc = stream_writer;
        // Drop all remaining Arc references (workers already finished).
        let inner = std::sync::Arc::try_unwrap(arc)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "stream_writer still has outstanding references"))?
            .into_inner()
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "stream_writer mutex poisoned"))?;
        inner.close()?;
    }
    let _ = output_count;
    let diagnostics = BxbDiagnostics {
        n_mutations_proposed: diag_n_mutations_proposed,
        n_clue_mismatch: diag_n_clue_mismatch,
        n_canonical_dup: diag_n_canonical_dup,
        n_uniqueness_fail: diag_n_uniqueness_fail,
        n_rated: diag_n_rated,
        n_above_target: diag_n_above_target,
        n_clue_count_changed: diag_n_clue_count_changed,
    };
    write_bxb_manifest(
        out,
        label,
        seed_count,
        outputs.len(),
        target_bxb,
        bxb_max_length,
        bxb_buffer_size,
        &diagnostics,
    )?;
    Ok(())
}

#[derive(Default, Clone, Copy)]
struct BxbDiagnostics {
    n_mutations_proposed: usize,
    n_clue_mismatch: usize,
    n_canonical_dup: usize,
    n_uniqueness_fail: usize,
    n_rated: usize,
    n_above_target: usize,
    /// Number of v3 candidates whose post-reminimize clue count differs from
    /// the seed. Always 0 for v1/v2.
    n_clue_count_changed: usize,
}

/// Streaming parquet writer for BxB-fitness gen-vicinity outputs.
///
/// Holds an open `ArrowWriter` and a small in-memory buffer of accepted
/// `BxbEntry` rows. `append_batch` writes one `RecordBatch` to the parquet
/// row-group stream and `writer.flush()`'s the underlying file, so the
/// partial parquet on disk is readable mid-run (after the first batch).
/// `close()` writes the footer and finalises the file.
///
/// **RAII finalization (bug #2 fix).** The `ArrowWriter` is held inside an
/// `Option`; `Drop` invokes a best-effort `close()` so the parquet footer is
/// written even on panic, early `?`-return, or process kill (via SIGTERM,
/// because the process unwinds cleanly via `std::process::exit`-style
/// handlers — for SIGKILL nothing can save us). The explicit `close()`
/// method is the happy-path; `Drop` is the safety net.
struct StreamingBxbWriter {
    writer: Option<parquet::arrow::ArrowWriter<std::fs::File>>,
    schema: std::sync::Arc<arrow_schema::Schema>,
    /// Snapshot of every row that's been pushed to the parquet stream. Held
    /// in RAM so the caller can still compute summary stats (min/mean/max
    /// BxB) without re-reading the file. For the gen-vicinity workloads this
    /// is bounded by `max_outputs` ≤ a few thousand.
    snapshot: Vec<BxbEntry>,
}

impl StreamingBxbWriter {
    fn create(path: &std::path::Path) -> std::io::Result<Self> {
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use parquet::basic::Compression;
        use parquet::file::properties::{EnabledStatistics, WriterProperties};

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("puzzle", DataType::Utf8, false),
            Field::new("solution", DataType::Utf8, false),
            Field::new("clue_count", DataType::Int32, false),
            Field::new("bxb", DataType::UInt16, false),
            Field::new("generation", DataType::Int32, false),
        ]));
        let file = std::fs::File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_statistics_enabled(EnabledStatistics::None)
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(Self {
            writer: Some(writer),
            schema,
            snapshot: Vec::new(),
        })
    }

    fn append_batch(&mut self, entries: &[BxbEntry]) -> std::io::Result<()> {
        use arrow_array::builder::{Int32Builder, StringBuilder, UInt16Builder};
        use arrow_array::{ArrayRef, RecordBatch};
        if entries.is_empty() {
            return Ok(());
        }
        let writer = self.writer.as_mut().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "writer already closed")
        })?;
        let n = entries.len();
        let mut puzzle_b = StringBuilder::with_capacity(n, n * 82);
        let mut solution_b = StringBuilder::with_capacity(n, n * 82);
        let mut clue_b = Int32Builder::with_capacity(n);
        let mut bxb_b = UInt16Builder::with_capacity(n);
        let mut gen_b = Int32Builder::with_capacity(n);
        for e in entries {
            puzzle_b.append_value(&e.puzzle);
            solution_b.append_value(&e.solution);
            clue_b.append_value(
                e.puzzle.bytes().filter(|&b| b != b'.' && b != b'0').count() as i32,
            );
            bxb_b.append_value(e.bxb);
            gen_b.append_value(e.generation as i32);
        }
        let cols: Vec<ArrayRef> = vec![
            std::sync::Arc::new(puzzle_b.finish()),
            std::sync::Arc::new(solution_b.finish()),
            std::sync::Arc::new(clue_b.finish()),
            std::sync::Arc::new(bxb_b.finish()),
            std::sync::Arc::new(gen_b.finish()),
        ];
        let batch = RecordBatch::try_new(self.schema.clone(), cols)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        writer
            .write(&batch)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        writer
            .flush()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.snapshot.extend_from_slice(entries);
        Ok(())
    }

    fn snapshot_entries(&self) -> Vec<BxbEntry> {
        self.snapshot.clone()
    }

    /// Happy-path close — writes the parquet footer and consumes self.
    fn close(mut self) -> std::io::Result<()> {
        if let Some(w) = self.writer.take() {
            w.close().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })?;
        }
        Ok(())
    }
}

/// RAII safety net: if `close()` wasn't called explicitly (panic, early
/// `?`-return, drop-on-error), best-effort write the parquet footer here so
/// the file is a valid parquet rather than a footerless corrupt blob.
/// Errors are swallowed — there's no caller to receive them in `Drop`.
impl Drop for StreamingBxbWriter {
    fn drop(&mut self) {
        if let Some(w) = self.writer.take() {
            if let Err(e) = w.close() {
                eprintln!(
                    "warn: StreamingBxbWriter::drop best-effort close failed: {}",
                    e
                );
            }
        }
    }
}

/// Drain a worker's local `BxbEntry` buffer through the global canonical-hash
/// dedup + `max_outputs` cap, then append the survivors to the streaming
/// parquet writer. `local_outputs` is cleared on return.
///
/// **Bug #3 fix.** Global dedup state (`global_seen`) and the emitted counter
/// (`global_emitted`) are updated ONLY after the parquet append succeeds. On
/// append failure, freshly inserted hashes are rolled back from `global_seen`
/// (the emitted counter was never touched), so a transient I/O failure does
/// not poison dedup or burn output slots.
///
/// Returns Err iff the parquet append failed; callers in the worker closure
/// log the error but do not abort the run — partial output remains a valid
/// (footer-on-Drop) parquet, and subsequent flushes can still succeed.
fn flush_local_to_stream<const N: usize, const BR: usize, const BC: usize>(
    local_outputs: &mut Vec<BxbEntry>,
    stream_writer: &std::sync::Arc<std::sync::Mutex<StreamingBxbWriter>>,
    global_seen: &std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u128>>>,
    global_emitted: &std::sync::Arc<std::sync::atomic::AtomicUsize>,
    max_outputs: usize,
) -> std::io::Result<()> {
    use std::sync::atomic::Ordering;
    use sudoku_rs_core::generic::canonical::canonical_hash;
    use sudoku_rs_core::generic::grid::Grid as VGrid;
    if local_outputs.is_empty() {
        return Ok(());
    }
    let drained: Vec<BxbEntry> = local_outputs.drain(..).collect();
    let mut to_write: Vec<BxbEntry> = Vec::with_capacity(drained.len());
    // Hashes we speculatively inserted into `global_seen` — rolled back on
    // append failure so dedup state matches what's actually on disk.
    let mut inserted_hashes: Vec<u128> = Vec::with_capacity(drained.len());
    {
        let mut seen = global_seen.lock().expect("global_seen mutex poisoned");
        // Bound by max_outputs using a local counter against the *current*
        // observed emit count — we only commit the increment after the
        // append succeeds, so this is an upper-bound reservation; we may
        // over-reserve briefly but never over-commit.
        let already_emitted = global_emitted.load(Ordering::Relaxed);
        let mut reserved = 0usize;
        for entry in drained {
            if already_emitted + reserved >= max_outputs {
                break;
            }
            let grid = match VGrid::<N, BR, BC>::from_str(&entry.puzzle) {
                Some(g) => g,
                None => continue,
            };
            let hash = canonical_hash::<N, BR, BC>(&grid);
            if !seen.insert(hash) {
                continue;
            }
            inserted_hashes.push(hash);
            to_write.push(entry);
            reserved += 1;
        }
    }
    if to_write.is_empty() {
        return Ok(());
    }
    let mut w = stream_writer.lock().expect("stream_writer mutex poisoned");
    match w.append_batch(&to_write) {
        Ok(()) => {
            // Commit: emitted counter advances only on successful disk write.
            global_emitted.fetch_add(to_write.len(), Ordering::Relaxed);
            Ok(())
        }
        Err(e) => {
            // Rollback the speculative dedup insertions so the same puzzles
            // can be retried by future iterations / workers.
            drop(w);
            let mut seen = global_seen.lock().expect("global_seen mutex poisoned");
            for h in &inserted_hashes {
                seen.remove(h);
            }
            Err(e)
        }
    }
}

/// Parquet schema for BxB-fitness gen-vicinity outputs.
#[allow(dead_code)]
fn write_bxb_parquet(path: &std::path::Path, entries: &[BxbEntry]) -> std::io::Result<()> {
    use arrow_array::builder::{Int32Builder, StringBuilder, UInt16Builder};
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};

    let schema = std::sync::Arc::new(Schema::new(vec![
        Field::new("puzzle", DataType::Utf8, false),
        Field::new("solution", DataType::Utf8, false),
        Field::new("clue_count", DataType::Int32, false),
        Field::new("bxb", DataType::UInt16, false),
        Field::new("generation", DataType::Int32, false),
    ]));
    let n = entries.len();
    let mut puzzle_b = StringBuilder::with_capacity(n, n * 82);
    let mut solution_b = StringBuilder::with_capacity(n, n * 82);
    let mut clue_b = Int32Builder::with_capacity(n);
    let mut bxb_b = UInt16Builder::with_capacity(n);
    let mut gen_b = Int32Builder::with_capacity(n);
    for e in entries {
        puzzle_b.append_value(&e.puzzle);
        solution_b.append_value(&e.solution);
        clue_b.append_value(
            e.puzzle.bytes().filter(|&b| b != b'.' && b != b'0').count() as i32,
        );
        bxb_b.append_value(e.bxb);
        gen_b.append_value(e.generation as i32);
    }
    let cols: Vec<ArrayRef> = vec![
        std::sync::Arc::new(puzzle_b.finish()),
        std::sync::Arc::new(solution_b.finish()),
        std::sync::Arc::new(clue_b.finish()),
        std::sync::Arc::new(bxb_b.finish()),
        std::sync::Arc::new(gen_b.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), cols)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    writer
        .close()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

fn write_bxb_manifest(
    out_path: &std::path::Path,
    label: &str,
    seed_count: usize,
    output_count: usize,
    target_bxb: u16,
    bxb_max_length: u16,
    bxb_buffer_size: usize,
    diagnostics: &BxbDiagnostics,
) -> std::io::Result<()> {
    let manifest_path = {
        let mut p = out_path.to_path_buf();
        let ext = p.extension().unwrap_or_default().to_string_lossy().into_owned();
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        p.set_file_name(format!("{}.{}.manifest.json", stem, ext));
        p
    };
    let json = format!(
        r#"{{
  "label": {label},
  "fitness": "bxb",
  "target_bxb": {target_bxb},
  "bxb_max_length": {bxb_max_length},
  "bxb_buffer_size": {bxb_buffer_size},
  "seed_count": {seed_count},
  "output_count": {output_count},
  "diagnostics": {{
    "n_mutations_proposed": {diag_mp},
    "n_clue_mismatch": {diag_cm},
    "n_canonical_dup": {diag_cd},
    "n_uniqueness_fail": {diag_uf},
    "n_rated": {diag_r},
    "n_above_target": {diag_at},
    "n_clue_count_changed": {diag_ccc}
  }}
}}"#,
        label = serde_json::to_string(label).unwrap(),
        target_bxb = target_bxb,
        bxb_max_length = bxb_max_length,
        bxb_buffer_size = bxb_buffer_size,
        seed_count = seed_count,
        output_count = output_count,
        diag_mp = diagnostics.n_mutations_proposed,
        diag_cm = diagnostics.n_clue_mismatch,
        diag_cd = diagnostics.n_canonical_dup,
        diag_uf = diagnostics.n_uniqueness_fail,
        diag_r = diagnostics.n_rated,
        diag_at = diagnostics.n_above_target,
        diag_ccc = diagnostics.n_clue_count_changed,
    );
    std::fs::write(&manifest_path, json)?;
    eprintln!("vicinity[bxb] manifest: {}", manifest_path.display());
    Ok(())
}

#[cfg(test)]
mod bxb_vicinity_tests {
    use super::*;

    /// A puzzle known to be in the BxB range. We don't pin the exact value,
    /// only that rating succeeds and lands in [2, 14].
    /// (Same magictour T3 seed used in src/generic/vicinity.rs.)
    const SEED: &str =
        "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.";

    #[test]
    fn bxb_rate_returns_sensible_range() {
        let bxb = bxb_rate_puzzle(SEED, 14, 4096);
        assert!(bxb.is_some(), "BxB rating should succeed for seed");
        let v = bxb.unwrap();
        // BxB range: 0..=max_length (0 = solvable by base propagation alone).
        assert!(v <= 14, "BxB={} > max_length=14", v);
    }

    #[test]
    fn bxb_mutate_preserves_clue_count() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let original_cc = SEED.bytes().filter(|&b| b != b'.').count();
        for _ in 0..32 {
            if let Some(m) = bxb_mutate_once(SEED, 1, &mut rng) {
                let cc = m.bytes().filter(|&b| b != b'.' && b != b'0').count();
                assert_eq!(cc, original_cc, "clue count must be preserved by mutation");
                assert_ne!(m, SEED, "mutation must differ from original");
            }
        }
    }

    #[test]
    fn bxb_vicinity_one_step_smoke() {
        // 1-iter run: must not crash; BxB output of seed must be sane.
        let raw = vec![(SEED.to_string(), None)];
        let tmpdir = std::env::temp_dir();
        let out = tmpdir.join("bxb_vicinity_smoke.parquet");
        let _ = std::fs::remove_file(&out);
        run_gen_vicinity_bxb(
            raw,
            &out,
            "smoke",
            /*target_bxb=*/ 2,
            /*bxb_max_length=*/ 14,
            /*bxb_buffer_size=*/ 4096,
            &[1u8],
            /*budget_iters=*/ 1,
            /*max_outputs=*/ 4,
            /*rng_seed=*/ 1,
            BxbMutator::V1,
            /*allow_clue_count_change=*/ false,
        )
        .expect("bxb gen-vicinity should not error");
        assert!(out.exists(), "output parquet not written");
    }

    #[test]
    fn test_bxb_mutate_v2_preserves_solution_consistency() {
        use sudoku_rs_core::generic::grid::Grid as VGrid;
        use sudoku_rs_core::generic::search::solve_unique as sol_uniq;

        let grid = VGrid::<9, 3, 3>::from_str(SEED).expect("seed parses");
        let solution = sol_uniq::<9, 3, 3>(&grid)
            .expect("seed has unique solution")
            .to_string_grid();
        let sol_bytes = solution.as_bytes();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(13);
        let mut n_swap = 0usize;
        let mut n_remove = 0usize;
        let mut n_add = 0usize;
        let mut produced = 0usize;
        for _ in 0..256 {
            for &n in &[1usize, 2, 3] {
                if let Some(m) = bxb_mutate_once_v2(SEED, &solution, n, &mut rng) {
                    produced += 1;
                    assert_eq!(m.len(), 81);
                    // Every clue in m must equal the solution at that index.
                    for (i, b) in m.bytes().enumerate() {
                        if b != b'.' && b != b'0' {
                            assert_eq!(
                                b, sol_bytes[i],
                                "clue at {} ({}) inconsistent with solution ({})",
                                i, b as char, sol_bytes[i] as char,
                            );
                        }
                    }
                    let orig_cc = SEED.bytes().filter(|&b| b != b'.' && b != b'0').count();
                    let new_cc = m.bytes().filter(|&b| b != b'.' && b != b'0').count();
                    let delta = new_cc as i32 - orig_cc as i32;
                    if delta == 0 && n == 1 {
                        n_swap += 1;
                    } else if delta < 0 && n == 1 {
                        n_remove += 1;
                    } else if delta > 0 && n == 1 {
                        n_add += 1;
                    }
                }
            }
        }
        assert!(produced > 0, "mutator must produce candidates");
        // For n=1, all three move kinds should fire at least once across 256 draws.
        assert!(n_swap > 0, "swap move never fired");
        assert!(n_remove > 0, "remove move never fired");
        assert!(n_add > 0, "add move never fired");
    }

    fn solve_seed(seed: &str) -> String {
        use sudoku_rs_core::generic::grid::Grid as VGrid;
        use sudoku_rs_core::generic::search::solve_unique as sol_uniq;
        let g = VGrid::<9, 3, 3>::from_str(seed).expect("seed parses");
        sol_uniq::<9, 3, 3>(&g)
            .expect("seed has unique solution")
            .to_string_grid()
    }

    #[test]
    fn test_brt_expand_idempotent() {
        let solution = solve_seed(SEED);
        let once = brt_expand(SEED, &solution);
        let twice = brt_expand(&once, &solution);
        assert_eq!(
            once, twice,
            "brt_expand must be idempotent (singles closure)"
        );
    }

    #[test]
    fn test_brt_expand_only_adds() {
        let solution = solve_seed(SEED);
        let cc_in = SEED.bytes().filter(|&b| b != b'.' && b != b'0').count();
        let out = brt_expand(SEED, &solution);
        let cc_out = out.bytes().filter(|&b| b != b'.' && b != b'0').count();
        assert!(
            cc_out >= cc_in,
            "brt_expand must not remove clues (cc_in={}, cc_out={})",
            cc_in,
            cc_out,
        );
        // And every original clue must still match.
        for (i, (a, b)) in SEED.bytes().zip(out.bytes()).enumerate() {
            if a != b'.' && a != b'0' {
                assert_eq!(a, b, "original clue at {} altered by brt_expand", i);
            }
        }
    }

    #[test]
    fn test_reminimize_below_minimal_returns_minimal() {
        // Take the SEED (assumed minimal), add a non-clue digit at first '.'
        // position from its solution → puzzle is now non-minimal. Reminimize
        // should restore a clue count ≤ original + 0 (typically == original,
        // since the added clue is redundant given uniqueness of seed).
        let solution = solve_seed(SEED);
        let mut bytes: Vec<u8> = SEED.bytes().collect();
        let sol_bytes = solution.as_bytes();
        let first_dot = bytes.iter().position(|&b| b == b'.' || b == b'0').unwrap();
        bytes[first_dot] = sol_bytes[first_dot];
        let inflated = String::from_utf8(bytes).unwrap();
        let inflated_cc = inflated.bytes().filter(|&b| b != b'.' && b != b'0').count();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(99);
        let min_p = reminimize(&inflated, &mut rng).expect("inflated puzzle unique");
        let min_cc = min_p.bytes().filter(|&b| b != b'.' && b != b'0').count();
        assert!(
            min_cc < inflated_cc,
            "reminimize should strip at least one redundant clue ({} -> {})",
            inflated_cc,
            min_cc,
        );
        // And the result must still be unique.
        use sudoku_rs_core::generic::grid::Grid as VGrid;
        use sudoku_rs_core::generic::search::count_solutions_up_to as csup;
        let g = VGrid::<9, 3, 3>::from_str(&min_p).expect("min parses");
        assert_eq!(csup::<9, 3, 3>(&g, 2), 1, "reminimize output must be unique");
    }

    #[test]
    fn test_v4_mutator_preserves_uniqueness_when_tridagon_present() {
        // Seed #6 from the BxB=6 corpus has 1 structural tridagon
        // (triplet={1,3,4}, boxes={1,2,7,8}). Verify v4 produces unique
        // puzzles across many draws.
        const SEED: &str =
            "........1....12.3...13..4....35...4..6...78..9...2......82....4.7..9....2....65..";
        use sudoku_rs_core::generic::grid::Grid;
        use sudoku_rs_core::generic::search::solve_unique;
        let grid = Grid::<9, 3, 3>::from_str(SEED).expect("seed parses");
        let solution = solve_unique::<9, 3, 3>(&grid).unwrap().to_string_grid();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEAD_BEEF);
        let mut produced = 0;
        for _ in 0..50 {
            if let Some((p, s)) = bxb_mutate_once_v4(SEED, &solution, 1, &mut rng) {
                // Solution length sanity.
                assert_eq!(s.len(), 81);
                // Clue count preserved (relabel never adds/removes clues).
                let in_cc = SEED.bytes().filter(|&b| b != b'.' && b != b'0').count();
                let out_cc = p.bytes().filter(|&b| b != b'.' && b != b'0').count();
                assert_eq!(in_cc, out_cc, "v4 must preserve clue count");
                // Uniqueness already verified inside the mutator; double-
                // check externally.
                use sudoku_rs_core::generic::search::count_solutions_up_to as csup;
                let g = Grid::<9, 3, 3>::from_str(&p).unwrap();
                assert_eq!(csup::<9, 3, 3>(&g, 2), 1, "v4 emitted non-unique");
                produced += 1;
            }
        }
        assert!(produced > 0, "v4 must emit at least one candidate on seed with tridagon");
    }

    #[test]
    fn test_v3_mutator_preserves_uniqueness() {
        use sudoku_rs_core::generic::grid::Grid as VGrid;
        use sudoku_rs_core::generic::search::count_solutions_up_to as csup;

        let solution = solve_seed(SEED);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
        let mut produced = 0usize;
        for _ in 0..100 {
            for &n in &[1usize, 2, 3] {
                if let Some(m) =
                    bxb_mutate_once_v3(SEED, &solution, n, n, &mut rng)
                {
                    produced += 1;
                    let g = VGrid::<9, 3, 3>::from_str(&m)
                        .expect("v3 output should parse");
                    assert_eq!(
                        csup::<9, 3, 3>(&g, 2),
                        1,
                        "v3 mutator emitted non-unique puzzle: {}",
                        m,
                    );
                }
            }
        }
        assert!(
            produced > 0,
            "v3 mutator must produce at least one candidate over 300 draws",
        );
    }
}
