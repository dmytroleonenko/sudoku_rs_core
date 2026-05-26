//! Generic-substrate dataset pipeline (P2).
//!
//! Companion to `pipeline_writer.rs` for the new generic CLI surface
//! (`gen-dataset --size NxBRxBC --target-tier T --output path.parquet`).
//! Differences from the legacy pipeline writer:
//!   - operates on `crate::generic::Grid<N, BR, BC>` for arbitrary sizes;
//!   - parquet schema adds a `size` Utf8 column ("9x3x3", "12x3x4", …);
//!   - single output parquet file (no shard-rotation / manifest).
//!
//! Threading model: rayon-parallel constrained-removal workers + a single
//! collector that buffers records until `target_total` is reached, then
//! writes one parquet file. This keeps the parquet writer code path simple;
//! sharding can be re-added later without touching the generic substrate.

use crate::generic::{
    gen_constrained,
    rater::RateResult,
    techniques::Tier,
    Grid, TechniqueChainSpec,
};
use crate::schema::{
    backtracking_entropy, puzzle_id, se_rating_estimate, split_for_id,
};
use arrow_array::builder::{
    BooleanBuilder, Float64Builder, Int64Builder, ListBuilder, StringBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Output record. Mirrors the legacy `Record` schema plus an explicit `size`
/// column. Storage is `Vec<i8>` (variable length) since N varies.
#[derive(Clone, Debug)]
pub struct GRecord {
    pub puzzle: Vec<i8>,    // length N*N
    pub solution: Vec<i8>,  // length N*N
    pub size: String,       // e.g. "9x3x3"
    pub tier: i64,
    pub tier_name: String,
    pub technique_frontier: Vec<String>,
    pub longest_inference_chain: i64,
    pub backtracking_entropy: f64,
    pub trace_techniques: Vec<i64>,
    pub trace_actions: String,
    pub trace_length: i64,
    pub se_rating_estimate: f64,
    pub puzzle_id: String,
    pub clue_count: i64,
    pub split: String,
    /// SE-equivalent cascade score from RateResult.se_score (Stage 0).
    /// 0.0 for T1 puzzles (no technique fired), ≥7.5 for unsolved T4Plus.
    pub se_score: f64,
}

/// Generic-pipeline parquet schema. Strict superset of the legacy schema:
/// every legacy column is present (same name, same type) plus a new `size`
/// Utf8 column at the end. Legacy-only columns we don't populate
/// (`propagation_wave_depth`, `search_backtracks`, `search_unique`) are
/// preserved with placeholder values so downstream readers expecting the
/// legacy schema still parse cleanly; flag-renames are deferred to P4.
pub fn generic_schema() -> Arc<Schema> {
    let elem_i64 = Arc::new(Field::new("element", DataType::Int64, true));
    let elem_str = Arc::new(Field::new("element", DataType::Utf8, true));
    Arc::new(Schema::new(vec![
        Field::new("puzzle", DataType::List(elem_i64.clone()), true),
        Field::new("solution", DataType::List(elem_i64.clone()), true),
        Field::new("tier", DataType::Int64, true),
        Field::new("tier_name", DataType::Utf8, true),
        Field::new("propagation_wave_depth", DataType::Int64, true),
        Field::new("technique_frontier", DataType::List(elem_str), true),
        Field::new("longest_inference_chain", DataType::Int64, true),
        Field::new("backtracking_entropy", DataType::Float64, true),
        Field::new("trace_techniques", DataType::List(elem_i64), true),
        Field::new("trace_actions", DataType::Utf8, true),
        Field::new("trace_length", DataType::Int64, true),
        Field::new("search_backtracks", DataType::Int64, true),
        Field::new("search_unique", DataType::Boolean, true),
        Field::new("se_rating_estimate", DataType::Float64, true),
        Field::new("puzzle_id", DataType::Utf8, true),
        Field::new("clue_count", DataType::Int64, true),
        Field::new("split", DataType::Utf8, true),
        Field::new("size", DataType::Utf8, true),
        // Stage 0: SE-equivalent cascade score (nullable for backward compat
        // with parquet files written before this schema version).
        Field::new("se_score", DataType::Float64, true),
    ]))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenericPipelineConfig {
    pub n: usize,
    pub br: usize,
    pub bc: usize,
    pub target_tier: Tier,
    pub clue_min: u32,
    pub clue_max: u32,
    pub target_total: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub output: PathBuf,
}

#[derive(Clone, Debug, Serialize)]
pub struct GenericPipelineStats {
    pub n_puzzles: usize,
    pub n_attempted: u64,
}

/// Build a record from a generated puzzle.
pub fn make_record<const N: usize, const BR: usize, const BC: usize>(
    puzzle: &Grid<N, BR, BC>,
    solution: &Grid<N, BR, BC>,
    r: &RateResult,
    clue_count: u32,
    size_str: &str,
    br: usize,
    bc: usize,
) -> GRecord {
    let _ = (br, bc);
    let nn = N * N;
    let mut puz: Vec<i8> = Vec::with_capacity(nn);
    let mut sol: Vec<i8> = Vec::with_capacity(nn);
    let mut puz_u8 = [0u8; 81];
    let mut sol_u8 = [0u8; 81];
    for i in 0..nn {
        puz.push(puzzle.solved[i] as i8);
        sol.push(solution.solved[i] as i8);
    }
    // puzzle_id helper expects fixed [u8; 81]; for non-9×9 we hash a
    // size-prefixed canonical string instead.
    let pid = if N == 9 {
        for i in 0..81 { puz_u8[i] = puzzle.solved[i]; sol_u8[i] = solution.solved[i]; }
        puzzle_id(&sol_u8, &puz_u8)
    } else {
        // md5("size|solution_digits|clue_mask")[:16]
        use md5::{Digest, Md5};
        let mut s = String::with_capacity(8 + nn + 1 + nn);
        s.push_str(size_str);
        s.push('|');
        for &d in solution.solved.iter() {
            if d <= 9 { s.push((b'0' + d) as char); } else { s.push((b'A' + d - 10) as char); }
        }
        s.push('|');
        for &d in puzzle.solved.iter() { s.push(if d != 0 { '1' } else { '0' }); }
        let mut h = Md5::new();
        h.update(s.as_bytes());
        hex::encode(h.finalize())[..16].to_string()
    };
    let split = split_for_id(&pid).to_string();
    let tier_i = match r.tier {
        Tier::T1 => 1,
        Tier::T2 => 2,
        Tier::T3 => 3,
        Tier::T4Plus => 4,
    } as i64;
    let tier_name = match r.tier {
        Tier::T1 => "T1",
        Tier::T2 => "T2",
        Tier::T3 => "T3",
        Tier::T4Plus => "T4Plus",
    }
    .to_string();
    let trace_t: Vec<i64> = r.trace.iter().map(|&id| id as i64).collect();
    let chain_proxy = trace_t.len() as i64;
    let frontier: Vec<String> = r
        .frontier
        .iter()
        .map(|&id| crate::generic::reverse_construct::technique_id_str(id).to_string())
        .collect();
    let trace_actions_json = {
        let parts: Vec<String> = trace_t.iter().map(|t| format!("[{}, {{}}]", t)).collect();
        format!("[{}]", parts.join(", "))
    };
    GRecord {
        puzzle: puz,
        solution: sol,
        size: size_str.to_string(),
        tier: tier_i,
        tier_name,
        technique_frontier: frontier,
        longest_inference_chain: chain_proxy,
        backtracking_entropy: backtracking_entropy(tier_i, chain_proxy),
        trace_techniques: trace_t,
        trace_actions: trace_actions_json,
        trace_length: chain_proxy,
        se_rating_estimate: se_rating_estimate(tier_i),
        puzzle_id: pid,
        clue_count: clue_count as i64,
        split,
        se_score: r.se_score,
    }
}

/// Inner per-size driver, monomorphized.
fn run_size<const N: usize, const BR: usize, const BC: usize>(
    cfg: &GenericPipelineConfig,
) -> std::io::Result<Vec<GRecord>> {
    let size_str = format!("{}x{}x{}", N, BR, BC);
    let spec = TechniqueChainSpec {
        target_tier: cfg.target_tier,
        required_techniques: Vec::new(),
        clue_min: cfg.clue_min,
        clue_max: cfg.clue_max,
    };
    let stop = Arc::new(AtomicBool::new(false));
    let attempted = Arc::new(AtomicU64::new(0));
    let kept = Arc::new(AtomicU64::new(0));
    let target_total = cfg.target_total;
    let num_workers = cfg.num_workers.max(1);
    let seed = cfg.seed;

    let (tx, rx) = std::sync::mpsc::channel::<GRecord>();
    let mut handles = Vec::with_capacity(num_workers);
    for w in 0..num_workers {
        let tx = tx.clone();
        let stop = stop.clone();
        let attempted = attempted.clone();
        let kept = kept.clone();
        let spec = spec.clone();
        let size_str = size_str.clone();
        let br = BR;
        let bc = BC;
        handles.push(std::thread::spawn(move || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(
                seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            );
            while !stop.load(Ordering::Relaxed) {
                attempted.fetch_add(1, Ordering::Relaxed);
                let out = gen_constrained::<N, BR, BC, _>(&mut rng, &spec, 4);
                if let Some((puzzle, r, clue_count)) = out {
                    // Reconstruct the solution by solving the puzzle.
                    let solution = match crate::generic::solve_unique(&puzzle) {
                        Some(s) => s,
                        None => continue,
                    };
                    let rec = make_record::<N, BR, BC>(
                        &puzzle, &solution, &r, clue_count, &size_str, br, bc,
                    );
                    let n = kept.fetch_add(1, Ordering::Relaxed) + 1;
                    if (n as usize) >= target_total {
                        stop.store(true, Ordering::Relaxed);
                    }
                    if tx.send(rec).is_err() {
                        return;
                    }
                }
            }
        }));
    }
    drop(tx);
    let mut records: Vec<GRecord> = Vec::with_capacity(target_total);
    while let Ok(r) = rx.recv() {
        if records.len() < target_total {
            records.push(r);
        }
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(records)
}

/// Top-level entry. Dispatches to `run_size::<N,BR,BC>` per (N,BR,BC) and
/// writes a single parquet file at `cfg.output`.
pub fn run(cfg: &GenericPipelineConfig) -> std::io::Result<GenericPipelineStats> {
    let records = match (cfg.n, cfg.br, cfg.bc) {
        (6, 2, 3) => run_size::<6, 2, 3>(cfg)?,
        (9, 3, 3) => run_size::<9, 3, 3>(cfg)?,
        (12, 3, 4) => run_size::<12, 3, 4>(cfg)?,
        (16, 4, 4) => run_size::<16, 4, 4>(cfg)?,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "unsupported size {}x{}x{} (supported: 6x2x3, 9x3x3, 12x3x4, 16x4x4)",
                    cfg.n, cfg.br, cfg.bc
                ),
            ));
        }
    };
    let n_puzzles = records.len();
    if let Some(parent) = cfg.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    // R3.1b: atomic parquet write + manifest sidecar. Schema unchanged.
    crate::generic_writer_helpers::atomic_write_parquet(&cfg.output, &records)?;
    let manifest = crate::generic_writer_helpers::GenerationManifest::now(
        n_puzzles,
        "gen-dataset",
        serde_json::json!({
            "size": format!("{}x{}x{}", cfg.n, cfg.br, cfg.bc),
            "target_tier": format!("{:?}", cfg.target_tier),
            "clue_min": cfg.clue_min,
            "clue_max": cfg.clue_max,
            "target_total": cfg.target_total,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
        }),
    );
    let mpath = crate::generic_writer_helpers::manifest_path_for(&cfg.output);
    crate::generic_writer_helpers::write_manifest(&mpath, &manifest)?;
    Ok(GenericPipelineStats {
        n_puzzles,
        n_attempted: 0, // attempted is tracked inside run_size; re-plumb if needed
    })
}

pub fn write_generic_parquet(
    path: &std::path::Path,
    records: &[GRecord],
) -> std::io::Result<()> {
    let schema = generic_schema();
    let n = records.len();
    let elem_i64 = Arc::new(Field::new("element", DataType::Int64, true));
    let elem_str = Arc::new(Field::new("element", DataType::Utf8, true));

    let mut puzzle_b =
        ListBuilder::new(Int64Builder::new()).with_field(elem_i64.clone());
    let mut solution_b =
        ListBuilder::new(Int64Builder::new()).with_field(elem_i64.clone());
    let mut tier_b = Int64Builder::with_capacity(n);
    let mut tier_name_b = StringBuilder::with_capacity(n, n * 8);
    let mut wave_b = Int64Builder::with_capacity(n);
    let mut frontier_b = ListBuilder::new(StringBuilder::new()).with_field(elem_str);
    let mut chain_b = Int64Builder::with_capacity(n);
    let mut entropy_b = Float64Builder::with_capacity(n);
    let mut trace_t_b = ListBuilder::new(Int64Builder::new()).with_field(elem_i64);
    let mut trace_a_b = StringBuilder::new();
    let mut trace_l_b = Int64Builder::with_capacity(n);
    let mut bt_b = Int64Builder::with_capacity(n);
    let mut uniq_b = BooleanBuilder::with_capacity(n);
    let mut se_b = Float64Builder::with_capacity(n);
    let mut pid_b = StringBuilder::with_capacity(n, n * 16);
    let mut clues_b = Int64Builder::with_capacity(n);
    let mut split_b = StringBuilder::with_capacity(n, n * 8);
    let mut size_b = StringBuilder::with_capacity(n, n * 6);
    let mut se_score_b = Float64Builder::with_capacity(n);

    for r in records {
        for &v in r.puzzle.iter() {
            puzzle_b.values().append_value(v as i64);
        }
        puzzle_b.append(true);
        for &v in r.solution.iter() {
            solution_b.values().append_value(v as i64);
        }
        solution_b.append(true);
        tier_b.append_value(r.tier);
        tier_name_b.append_value(&r.tier_name);
        wave_b.append_value(0);
        for s in r.technique_frontier.iter() {
            frontier_b.values().append_value(s);
        }
        frontier_b.append(true);
        chain_b.append_value(r.longest_inference_chain);
        entropy_b.append_value(r.backtracking_entropy);
        for &v in r.trace_techniques.iter() {
            trace_t_b.values().append_value(v);
        }
        trace_t_b.append(true);
        trace_a_b.append_value(&r.trace_actions);
        trace_l_b.append_value(r.trace_length);
        bt_b.append_value(0);
        uniq_b.append_value(true);
        se_b.append_value(r.se_rating_estimate);
        pid_b.append_value(&r.puzzle_id);
        clues_b.append_value(r.clue_count);
        split_b.append_value(&r.split);
        size_b.append_value(&r.size);
        se_score_b.append_value(r.se_score);
    }

    let cols: Vec<ArrayRef> = vec![
        Arc::new(puzzle_b.finish()),
        Arc::new(solution_b.finish()),
        Arc::new(tier_b.finish()),
        Arc::new(tier_name_b.finish()),
        Arc::new(wave_b.finish()),
        Arc::new(frontier_b.finish()),
        Arc::new(chain_b.finish()),
        Arc::new(entropy_b.finish()),
        Arc::new(trace_t_b.finish()),
        Arc::new(trace_a_b.finish()),
        Arc::new(trace_l_b.finish()),
        Arc::new(bt_b.finish()),
        Arc::new(uniq_b.finish()),
        Arc::new(se_b.finish()),
        Arc::new(pid_b.finish()),
        Arc::new(clues_b.finish()),
        Arc::new(split_b.finish()),
        Arc::new(size_b.finish()),
        Arc::new(se_score_b.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), cols)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
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

/// Streaming generic parquet writer.
///
/// Mirrors the atomic `write_generic_parquet` but writes batches as they
/// arrive and `flush()`'s after each batch — the parquet file on disk grows
/// monotonically (after the first batch lands, downstream readers can poll
/// the file size to detect progress). The footer is written either by an
/// explicit `close()` (happy path) or by `Drop` (panic / early-?-return /
/// SIGTERM unwind safety net).
///
/// Trade-off vs `atomic_write_parquet`: streaming gives up the atomic-rename
/// contract — a reader that races a still-running writer sees a footerless
/// (and from `parquet::read`'s POV, invalid) file. After `close()` /
/// `Drop`, the file is a valid parquet. Suitable for long-running rerate
/// jobs where visibility-during-run matters more than read-during-write
/// atomicity.
pub struct StreamingGenericWriter {
    writer: Option<ArrowWriter<std::fs::File>>,
    schema: Arc<Schema>,
    n_written: usize,
}

impl StreamingGenericWriter {
    pub fn create(path: &std::path::Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let schema = generic_schema();
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
            n_written: 0,
        })
    }

    /// Append a batch of `GRecord`s to the parquet stream and flush the
    /// underlying file. After this returns Ok, `n_written` is incremented.
    pub fn append(&mut self, records: &[GRecord]) -> std::io::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let writer = self.writer.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "StreamingGenericWriter: writer already closed",
            )
        })?;
        let n = records.len();
        let elem_i64 = Arc::new(Field::new("element", DataType::Int64, true));
        let elem_str = Arc::new(Field::new("element", DataType::Utf8, true));
        let mut puzzle_b =
            ListBuilder::new(Int64Builder::new()).with_field(elem_i64.clone());
        let mut solution_b =
            ListBuilder::new(Int64Builder::new()).with_field(elem_i64.clone());
        let mut tier_b = Int64Builder::with_capacity(n);
        let mut tier_name_b = StringBuilder::with_capacity(n, n * 8);
        let mut wave_b = Int64Builder::with_capacity(n);
        let mut frontier_b =
            ListBuilder::new(StringBuilder::new()).with_field(elem_str);
        let mut chain_b = Int64Builder::with_capacity(n);
        let mut entropy_b = Float64Builder::with_capacity(n);
        let mut trace_t_b =
            ListBuilder::new(Int64Builder::new()).with_field(elem_i64);
        let mut trace_a_b = StringBuilder::new();
        let mut trace_l_b = Int64Builder::with_capacity(n);
        let mut bt_b = Int64Builder::with_capacity(n);
        let mut uniq_b = BooleanBuilder::with_capacity(n);
        let mut se_b = Float64Builder::with_capacity(n);
        let mut pid_b = StringBuilder::with_capacity(n, n * 16);
        let mut clues_b = Int64Builder::with_capacity(n);
        let mut split_b = StringBuilder::with_capacity(n, n * 8);
        let mut size_b = StringBuilder::with_capacity(n, n * 6);
        let mut se_score_b = Float64Builder::with_capacity(n);

        for r in records {
            for &v in r.puzzle.iter() {
                puzzle_b.values().append_value(v as i64);
            }
            puzzle_b.append(true);
            for &v in r.solution.iter() {
                solution_b.values().append_value(v as i64);
            }
            solution_b.append(true);
            tier_b.append_value(r.tier);
            tier_name_b.append_value(&r.tier_name);
            wave_b.append_value(0);
            for s in r.technique_frontier.iter() {
                frontier_b.values().append_value(s);
            }
            frontier_b.append(true);
            chain_b.append_value(r.longest_inference_chain);
            entropy_b.append_value(r.backtracking_entropy);
            for &v in r.trace_techniques.iter() {
                trace_t_b.values().append_value(v);
            }
            trace_t_b.append(true);
            trace_a_b.append_value(&r.trace_actions);
            trace_l_b.append_value(r.trace_length);
            bt_b.append_value(0);
            uniq_b.append_value(true);
            se_b.append_value(r.se_rating_estimate);
            pid_b.append_value(&r.puzzle_id);
            clues_b.append_value(r.clue_count);
            split_b.append_value(&r.split);
            size_b.append_value(&r.size);
            se_score_b.append_value(r.se_score);
        }

        let cols: Vec<ArrayRef> = vec![
            Arc::new(puzzle_b.finish()),
            Arc::new(solution_b.finish()),
            Arc::new(tier_b.finish()),
            Arc::new(tier_name_b.finish()),
            Arc::new(wave_b.finish()),
            Arc::new(frontier_b.finish()),
            Arc::new(chain_b.finish()),
            Arc::new(entropy_b.finish()),
            Arc::new(trace_t_b.finish()),
            Arc::new(trace_a_b.finish()),
            Arc::new(trace_l_b.finish()),
            Arc::new(bt_b.finish()),
            Arc::new(uniq_b.finish()),
            Arc::new(se_b.finish()),
            Arc::new(pid_b.finish()),
            Arc::new(clues_b.finish()),
            Arc::new(split_b.finish()),
            Arc::new(size_b.finish()),
            Arc::new(se_score_b.finish()),
        ];
        let batch = RecordBatch::try_new(self.schema.clone(), cols)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        writer
            .write(&batch)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        writer
            .flush()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        self.n_written += records.len();
        Ok(())
    }

    pub fn n_written(&self) -> usize {
        self.n_written
    }

    /// Happy-path close — writes the parquet footer and consumes self.
    pub fn close(mut self) -> std::io::Result<()> {
        if let Some(w) = self.writer.take() {
            w.close().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
            })?;
        }
        Ok(())
    }
}

/// RAII safety net: write the parquet footer if `close()` wasn't called
/// (panic, early `?`-return). On SIGTERM the Rust runtime unwinds, so this
/// runs; on SIGKILL nothing can save us.
impl Drop for StreamingGenericWriter {
    fn drop(&mut self) {
        if let Some(w) = self.writer.take() {
            if let Err(e) = w.close() {
                eprintln!(
                    "warn: StreamingGenericWriter::drop best-effort close failed: {}",
                    e
                );
            }
        }
    }
}

/// Parse a "NxBRxBC" size string into a (N, BR, BC) triple. Accepts the four
/// supported sizes; returns Err otherwise. We require all three components
/// (N, BR, BC) explicitly to disambiguate cases like "9x3x3" vs "9x9" — the
/// latter is rejected so the parser is unambiguous.
pub fn parse_size(s: &str) -> Result<(usize, usize, usize), String> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 3 {
        return Err(format!(
            "expected NxBRxBC (e.g. 9x3x3); got '{}'",
            s
        ));
    }
    let n: usize = parts[0]
        .parse()
        .map_err(|e| format!("invalid N in '{}': {}", s, e))?;
    let br: usize = parts[1]
        .parse()
        .map_err(|e| format!("invalid BR in '{}': {}", s, e))?;
    let bc: usize = parts[2]
        .parse()
        .map_err(|e| format!("invalid BC in '{}': {}", s, e))?;
    if br * bc != n {
        return Err(format!("BR*BC ({}) must equal N ({}) in '{}'", br * bc, n, s));
    }
    match (n, br, bc) {
        (6, 2, 3) | (9, 3, 3) | (12, 3, 4) | (16, 4, 4) => Ok((n, br, bc)),
        _ => Err(format!(
            "unsupported size '{}': supported are 6x2x3, 9x3x3, 12x3x4, 16x4x4",
            s
        )),
    }
}

/// Parse a "MIN..MAX" target-clues range (inclusive on both ends).
pub fn parse_clue_range(s: &str) -> Result<(u32, u32), String> {
    let (lo, hi) = s
        .split_once("..")
        .ok_or_else(|| format!("expected MIN..MAX (got '{}')", s))?;
    let lo: u32 = lo.parse().map_err(|e| format!("invalid MIN: {}", e))?;
    let hi: u32 = hi.parse().map_err(|e| format!("invalid MAX: {}", e))?;
    if lo > hi {
        return Err(format!("min ({}) > max ({})", lo, hi));
    }
    Ok((lo, hi))
}

pub fn parse_tier(s: &str) -> Result<Tier, String> {
    match s.to_ascii_uppercase().as_str() {
        "T1" => Ok(Tier::T1),
        "T2" => Ok(Tier::T2),
        "T3" => Ok(Tier::T3),
        "T4PLUS" | "T4+" | "T4" => Ok(Tier::T4Plus),
        other => Err(format!("expected T1|T2|T3|T4Plus, got '{}'", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_size_ok() {
        assert_eq!(parse_size("9x3x3").unwrap(), (9, 3, 3));
        assert_eq!(parse_size("6x2x3").unwrap(), (6, 2, 3));
        assert_eq!(parse_size("12x3x4").unwrap(), (12, 3, 4));
        assert_eq!(parse_size("16x4x4").unwrap(), (16, 4, 4));
    }

    #[test]
    fn parse_size_rejects_legacy_NxN_form() {
        assert!(parse_size("9x9").is_err());
        assert!(parse_size("16x16").is_err());
    }

    #[test]
    fn parse_size_rejects_bad_block() {
        assert!(parse_size("9x2x3").is_err()); // 2*3 != 9
        assert!(parse_size("12x4x4").is_err()); // 4*4 != 12
    }

    #[test]
    fn parse_size_rejects_unsupported_size() {
        assert!(parse_size("4x2x2").is_err());
        assert!(parse_size("25x5x5").is_err());
    }

    #[test]
    fn parse_clue_range_ok() {
        assert_eq!(parse_clue_range("22..28").unwrap(), (22, 28));
        assert_eq!(parse_clue_range("17..81").unwrap(), (17, 81));
    }

    #[test]
    fn parse_clue_range_rejects_swapped() {
        assert!(parse_clue_range("30..22").is_err());
    }

    #[test]
    fn parse_tier_ok() {
        assert_eq!(parse_tier("T1").unwrap(), Tier::T1);
        assert_eq!(parse_tier("T4Plus").unwrap(), Tier::T4Plus);
        assert_eq!(parse_tier("T4+").unwrap(), Tier::T4Plus);
        assert_eq!(parse_tier("T4").unwrap(), Tier::T4Plus);
    }

    #[test]
    fn end_to_end_9x9_t1_writes_parquet() {
        let tmpdir = std::env::temp_dir().join(format!("p2_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let out = tmpdir.join("t1_9x9.parquet");
        let cfg = GenericPipelineConfig {
            n: 9, br: 3, bc: 3,
            target_tier: Tier::T1,
            clue_min: 17, clue_max: 81,
            target_total: 5,
            num_workers: 1,
            seed: 999,
            output: out.clone(),
        };
        let stats = run(&cfg).unwrap();
        assert_eq!(stats.n_puzzles, 5);
        assert!(out.exists());
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    #[test]
    fn end_to_end_6x6_t1_writes_parquet() {
        let tmpdir = std::env::temp_dir().join(format!("p2_test6_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let out = tmpdir.join("t1_6x6.parquet");
        let cfg = GenericPipelineConfig {
            n: 6, br: 2, bc: 3,
            target_tier: Tier::T1,
            clue_min: 4, clue_max: 36,
            target_total: 5,
            num_workers: 1,
            seed: 1234,
            output: out.clone(),
        };
        let stats = run(&cfg).unwrap();
        assert_eq!(stats.n_puzzles, 5);
        assert!(out.exists());
        let _ = std::fs::remove_dir_all(&tmpdir);
    }

    /// Schema test: verify the written parquet has a `size` column containing
    /// the correct value.
    #[test]
    fn parquet_has_size_column() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use arrow_array::Array;
        let tmpdir = std::env::temp_dir().join(format!("p2_schema_{}", std::process::id()));
        std::fs::create_dir_all(&tmpdir).unwrap();
        let out = tmpdir.join("schema.parquet");
        let cfg = GenericPipelineConfig {
            n: 9, br: 3, bc: 3,
            target_tier: Tier::T1,
            clue_min: 17, clue_max: 81,
            target_total: 3,
            num_workers: 1,
            seed: 7,
            output: out.clone(),
        };
        run(&cfg).unwrap();
        let f = std::fs::File::open(&out).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(f).unwrap().build().unwrap();
        let mut size_seen: Option<String> = None;
        for batch in reader {
            let batch = batch.unwrap();
            let size_idx = batch.schema().index_of("size").expect("`size` column present");
            let arr = batch
                .column(size_idx)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap();
            for i in 0..arr.len() {
                let v = arr.value(i).to_string();
                if let Some(prev) = &size_seen {
                    assert_eq!(prev, &v);
                } else {
                    size_seen = Some(v);
                }
            }
        }
        assert_eq!(size_seen.unwrap(), "9x3x3");
        let _ = std::fs::remove_dir_all(&tmpdir);
    }
}
