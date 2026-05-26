//! Generic re-rate pipeline (P2.1).
//!
//! Reads puzzles from parquet or plaintext, recomputes classification fields
//! using the const-generic rater (`crate::generic::rate`), and writes a single
//! parquet file via `pipeline_writer_generic::write_generic_parquet`.
//!
//! Companion to `crate::rerate` — that module remains the byte-identical
//! 9×9-only implementation. This one is dispatched by CLI when `--size` is
//! present (so the legacy 9×9 default path is unchanged).
//!
//! Input formats (auto-detected by file extension):
//!   - `.parquet`: read `puzzle` List<Int64> column (legacy schema or generic
//!     schema both work — only the column is required). One record per row.
//!   - `.txt` / other: one puzzle per line, N*N chars per line, '.'/'0' empty,
//!     '1'..'9'/'A'..'G' digits (uppercase or lowercase). Reused via
//!     `Grid::from_str`.
//!
//! Output: a single parquet file at `cfg.output` matching the schema written
//! by `gen-dataset --size`.

use crate::generic::{rate_with_mode, Grid, SolverMode};
use crate::generic::techniques::Tier;
use crate::generic_writer_helpers::{
    manifest_path_for, write_manifest, GenerationManifest, OrderedGRecordBuffer,
};
use crate::pipeline_writer_generic::GRecord;
use crate::schema::{backtracking_entropy, se_rating_estimate};
use arrow_array::{cast::AsArray, types::Int64Type, Array, ListArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerateGenericConfig {
    pub n: usize,
    pub br: usize,
    pub bc: usize,
    /// Input file paths (parquet or text). Mixed lists are allowed.
    pub input_paths: Vec<PathBuf>,
    /// Output parquet path.
    pub output: PathBuf,
    /// Rayon thread-pool override. 0 = default.
    pub num_workers: usize,
    /// R3.3a perf knob. `Full` (default) preserves pre-R3.3a behaviour:
    /// cascade + `solve_unique` second-pass. `Tier` runs the cascade and
    /// reuses its solved grid when possible (no second solve). `Solve` skips
    /// the cascade entirely (T1 propagation only) and falls back to
    /// `solve_unique` for the solution column.
    #[serde(default = "default_solver_mode")]
    pub mode: SolverMode,
    /// Optional progress counter incremented once per rated row. The CLI uses
    /// it to drive a `tqdm` bar from a watcher thread. Skipped from
    /// (de)serialization — pure runtime plumbing.
    #[serde(skip, default)]
    pub progress: Option<Arc<AtomicUsize>>,
}

fn default_solver_mode() -> SolverMode { SolverMode::Full }

#[derive(Clone, Debug, Default, Serialize)]
pub struct RerateGenericStats {
    pub n_input_rows: usize,
    pub n_parse_errors: usize,
    pub n_rater_errors: usize,
    pub n_kept: usize,
    pub counters: std::collections::HashMap<String, usize>,
}

/// Read puzzles (as raw digit strings of length N*N) from a parquet file.
/// Accepts the `puzzle` List<Int64> column (legacy or generic schema).
fn read_parquet_puzzles(path: &Path, nn: usize) -> std::io::Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("parquet open {}: {e}", path.display()),
        )
    })?;
    let reader = builder.build().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("parquet build {}: {e}", path.display()),
        )
    })?;
    let mut out: Vec<String> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("parquet read {}: {e}", path.display()),
            )
        })?;
        let n = batch.num_rows();
        let puzzle_col = batch
            .column_by_name("puzzle")
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("missing 'puzzle' in {}", path.display()),
                )
            })?
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::Other, "puzzle: not List")
            })?;
        for i in 0..n {
            let arr = puzzle_col.value(i);
            let p = arr.as_primitive::<Int64Type>();
            if p.len() != nn {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "{}: row {} puzzle len {} != {}",
                        path.display(),
                        i,
                        p.len(),
                        nn
                    ),
                ));
            }
            let mut s = String::with_capacity(nn);
            for k in 0..nn {
                let d = p.value(k);
                if d == 0 {
                    s.push('.');
                } else if d <= 9 {
                    s.push((b'0' + d as u8) as char);
                } else {
                    s.push((b'A' + (d as u8) - 10) as char);
                }
            }
            out.push(s);
        }
    }
    Ok(out)
}

/// Read puzzle lines from a text file. One line = one puzzle, expected
/// length N*N (after trim). Empty / shorter lines are skipped silently;
/// longer lines are truncated to the first N*N chars (so we tolerate trailing
/// whitespace or comments). Actual digit-set validation is delegated to
/// `Grid::from_str`.
fn read_text_puzzles(path: &Path, nn: usize) -> std::io::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)?;
    let mut out: Vec<String> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.len() < nn {
            // Treat as a parse-error candidate; let downstream produce an
            // unparseable Grid for accounting.
            out.push(line.to_string());
            continue;
        }
        out.push(line[..nn].to_string());
    }
    Ok(out)
}

fn classify(p: &Path) -> &'static str {
    match p.extension().and_then(|s| s.to_str()).map(|s| s.to_ascii_lowercase()) {
        Some(ref e) if e == "parquet" => "parquet",
        _ => "text",
    }
}

fn rerate_one<const N: usize, const BR: usize, const BC: usize>(
    digits: &str,
    size_str: &str,
    mode: SolverMode,
) -> Option<GRecord> {
    let g: Grid<N, BR, BC> = Grid::from_str(digits)?;
    let r = rate_with_mode::<N, BR, BC>(&g, mode);
    if r.rater_error {
        return None;
    }
    let nn = N * N;
    // puzzle digits (i8 of length N*N).
    let mut puz: Vec<i8> = Vec::with_capacity(nn);
    for &d in g.solved.iter() {
        puz.push(d as i8);
    }
    // solution: try to solve via generic backtracker.
    let solution_grid = match crate::generic::solve_unique(&g) {
        Some(s) => s,
        None => {
            // Non-unique — keep with empty solution rather than dropping, so
            // downstream sees the rater verdict on this puzzle. Mark by zero
            // solution. (Mirrors legacy `rerate.rs` tolerance for missing
            // solution.)
            let mut empty = g.clone();
            // produce an all-zero grid of the right length.
            empty.solved = vec![0u8; nn];
            empty
        }
    };
    let mut sol: Vec<i8> = Vec::with_capacity(nn);
    for &d in solution_grid.solved.iter() {
        sol.push(d as i8);
    }

    let tier_i = match r.tier {
        Tier::T1 => 1i64,
        Tier::T2 => 2,
        Tier::T3 => 3,
        Tier::T4Plus => 4,
    };
    let tier_name = match r.tier {
        Tier::T1 => "T1",
        Tier::T2 => "T2",
        Tier::T3 => "T3",
        Tier::T4Plus => "T4Plus",
    }
    .to_string();
    let frontier: Vec<String> = r
        .frontier
        .iter()
        .map(|&id| crate::generic::reverse_construct::technique_id_str(id).to_string())
        .collect();
    let trace_t: Vec<i64> = r.trace.iter().map(|&id| id as i64).collect();
    let chain_proxy = trace_t.len() as i64;
    let trace_actions = {
        let parts: Vec<String> = trace_t.iter().map(|t| format!("[{}, {{}}]", t)).collect();
        format!("[{}]", parts.join(", "))
    };
    let clue_count = puz.iter().filter(|&&d| d != 0).count() as i64;
    // puzzle_id: md5(size|digits|clue-mask)[:16]; matches gen-dataset for
    // non-9×9 and is stable for 9×9 here too (we don't try to match the
    // legacy 9×9 puzzle_id form because the byte-identical 9×9 path is
    // routed through the legacy `rerate` module instead).
    let pid = {
        use md5::{Digest, Md5};
        let mut s = String::with_capacity(8 + nn + 1 + nn);
        s.push_str(size_str);
        s.push('|');
        for &d in solution_grid.solved.iter() {
            if d <= 9 {
                s.push((b'0' + d) as char);
            } else {
                s.push((b'A' + d - 10) as char);
            }
        }
        s.push('|');
        for &d in g.solved.iter() {
            s.push(if d != 0 { '1' } else { '0' });
        }
        let mut h = Md5::new();
        h.update(s.as_bytes());
        hex::encode(h.finalize())[..16].to_string()
    };
    let split = crate::schema::split_for_id(&pid).to_string();
    Some(GRecord {
        puzzle: puz,
        solution: sol,
        size: size_str.to_string(),
        tier: tier_i,
        tier_name,
        technique_frontier: frontier,
        longest_inference_chain: chain_proxy,
        backtracking_entropy: backtracking_entropy(tier_i, chain_proxy),
        trace_techniques: trace_t,
        trace_actions,
        trace_length: chain_proxy,
        se_rating_estimate: se_rating_estimate(tier_i),
        puzzle_id: pid,
        clue_count,
        split,
        se_score: r.se_score,
    })
}

fn run_size<const N: usize, const BR: usize, const BC: usize>(
    cfg: &RerateGenericConfig,
) -> std::io::Result<RerateGenericStats> {
    let nn = N * N;
    let size_str = format!("{}x{}x{}", N, BR, BC);
    if cfg.num_workers > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.num_workers)
            .build_global();
    }
    // Stage 1: read all inputs → flat Vec<String> of digit lines.
    let read_results: Vec<std::io::Result<Vec<String>>> = cfg
        .input_paths
        .par_iter()
        .map(|p| match classify(p) {
            "parquet" => read_parquet_puzzles(p, nn),
            _ => read_text_puzzles(p, nn),
        })
        .collect();
    let mut digits: Vec<String> = Vec::new();
    for res in read_results {
        digits.extend(res?);
    }
    let n_input_rows = digits.len();
    // Stage 2: re-rate in parallel and stream to a parquet writer as records
    // become contiguous in input order. We chunk to bound peak memory + emit
    // partial output mid-run (a tailing reader sees the file grow). The
    // streaming writer's RAII `Drop` writes the parquet footer on any early
    // exit (panic, ?-return, SIGTERM), so the on-disk file is always a valid
    // parquet (possibly truncated).
    //
    // Trade-off vs. R3.1b's `atomic_write_parquet`: we lose the
    // write-to-tmp-then-rename atomicity. User explicitly preferred
    // visibility-during-run over atomic completion (2026-05-21).
    if let Some(parent) = cfg.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut writer =
        crate::pipeline_writer_generic::StreamingGenericWriter::create(&cfg.output)?;
    let progress = cfg.progress.clone();
    let mut buf = OrderedGRecordBuffer::with_capacity(digits.len());
    let mut n_parse_errors = 0usize;
    let mut n_rater_errors = 0usize;
    let mut counters: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for k in ["T1", "T2", "T3", "T4Plus"] {
        counters.insert(k.into(), 0);
    }
    // 64-row write batches: large enough to keep parquet row-group overhead
    // reasonable, small enough that the first batch lands in well under a
    // second even on slow rater modes.
    // Small RATE_CHUNK gives prompt mid-run visibility — even at the
    // T4Plus-heavy ~30 rps rates we see on forum_hardest, a 64-row chunk
    // lands in ~2s on a 4-worker pool. Larger chunks improve par_iter
    // amortization but trade off observability.
    const RATE_CHUNK: usize = 64;
    const WRITE_BATCH: usize = 64;
    let mut pending: Vec<GRecord> = Vec::with_capacity(WRITE_BATCH);
    let mut n_kept_total = 0usize;
    let n_input = digits.len();

    let mut idx_base = 0usize;
    for chunk in digits.chunks(RATE_CHUNK) {
        let rated: Vec<Option<GRecord>> = chunk
            .par_iter()
            .map(|d| {
                let r = rerate_one::<N, BR, BC>(d, &size_str, cfg.mode);
                if let Some(ref c) = progress {
                    c.fetch_add(1, Ordering::Relaxed);
                }
                r
            })
            .collect();
        for (local_idx, (raw, rec)) in chunk.iter().zip(rated.into_iter()).enumerate() {
            let global_idx = idx_base + local_idx;
            match rec {
                Some(r) => buf.submit(global_idx, r),
                None => {
                    if Grid::<N, BR, BC>::from_str(raw).is_none() {
                        n_parse_errors += 1;
                    } else {
                        n_rater_errors += 1;
                    }
                    // Note: a gap here means subsequent rated records will
                    // stay buffered in `OrderedGRecordBuffer` until
                    // `finish()`. This matches the pre-streaming behaviour
                    // (which also skipped failed rows entirely).
                }
            }
        }
        idx_base += chunk.len();
        // Drain whatever buffer prefix is contiguous → write batch(es).
        buf.flush_to(|rec| {
            *counters.entry(rec.tier_name.clone()).or_insert(0) += 1;
            pending.push(rec);
            if pending.len() >= WRITE_BATCH {
                let batch: Vec<GRecord> = std::mem::take(&mut pending);
                n_kept_total += batch.len();
                writer.append(&batch)?;
            }
            Ok(())
        })?;
    }
    // Final drain — `finish` returns the ascending-tail (handles gaps from
    // failed rows) — write any remaining records.
    let tail: Vec<GRecord> = buf.finish();
    for r in tail {
        *counters.entry(r.tier_name.clone()).or_insert(0) += 1;
        pending.push(r);
        if pending.len() >= WRITE_BATCH {
            let batch: Vec<GRecord> = std::mem::take(&mut pending);
            n_kept_total += batch.len();
            writer.append(&batch)?;
        }
    }
    if !pending.is_empty() {
        n_kept_total += pending.len();
        writer.append(&pending)?;
        pending.clear();
    }
    // Explicit close — writes the parquet footer. If this returns Err the
    // Drop impl will not run a second close (writer.take()'d inside close).
    writer.close()?;
    let _ = n_input;
    let n_kept = n_kept_total;
    // Sidecar manifest. Written after the parquet so a successful manifest
    // implies a closed parquet — but the parquet may be valid (footer
    // written by Drop) even without a manifest if we ?-returned earlier.
    let manifest = GenerationManifest::now(
        n_kept,
        "re-rate",
        serde_json::json!({
            "size": size_str,
            "num_workers": cfg.num_workers,
            "mode": format!("{:?}", cfg.mode),
            "n_input_rows": n_input_rows,
            "n_parse_errors": n_parse_errors,
            "n_rater_errors": n_rater_errors,
            "n_input_paths": cfg.input_paths.len(),
            "streaming": true,
        }),
    );
    let mpath = manifest_path_for(&cfg.output);
    write_manifest(&mpath, &manifest)?;
    Ok(RerateGenericStats {
        n_input_rows,
        n_parse_errors,
        n_rater_errors,
        n_kept,
        counters,
    })
}

pub fn run(cfg: &RerateGenericConfig) -> std::io::Result<RerateGenericStats> {
    match (cfg.n, cfg.br, cfg.bc) {
        (6, 2, 3) => run_size::<6, 2, 3>(cfg),
        (9, 3, 3) => run_size::<9, 3, 3>(cfg),
        (12, 3, 4) => run_size::<12, 3, 4>(cfg),
        (16, 4, 4) => run_size::<16, 4, 4>(cfg),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "unsupported size {}x{}x{} (supported: 6x2x3, 9x3x3, 12x3x4, 16x4x4)",
                cfg.n, cfg.br, cfg.bc
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline_writer_generic::{run as run_gen_pipeline, GenericPipelineConfig};

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "p2_1_test_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn round_trip_9x9_t1() {
        // Generate with gen-dataset, rerate, verify tier.
        let dir = tmp("rt9");
        let in_pq = dir.join("in.parquet");
        let out_pq = dir.join("out.parquet");
        let cfg = GenericPipelineConfig {
            n: 9, br: 3, bc: 3,
            target_tier: Tier::T1,
            clue_min: 17, clue_max: 81,
            target_total: 8, num_workers: 1, seed: 41,
            output: in_pq.clone(),
        };
        run_gen_pipeline(&cfg).unwrap();
        let rcfg = RerateGenericConfig {
            n: 9, br: 3, bc: 3,
            input_paths: vec![in_pq.clone()],
            output: out_pq.clone(),
            num_workers: 1,
            mode: SolverMode::Full,
            progress: None,
        };
        let st = run(&rcfg).unwrap();
        assert_eq!(st.n_input_rows, 8);
        assert!(st.n_kept >= 1, "expected at least 1 kept, got {}", st.n_kept);
        assert!(out_pq.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_16x16_t1() {
        let dir = tmp("rt16");
        let in_pq = dir.join("in.parquet");
        let out_pq = dir.join("out.parquet");
        let cfg = GenericPipelineConfig {
            n: 16, br: 4, bc: 4,
            target_tier: Tier::T1,
            clue_min: 100, clue_max: 256,
            target_total: 2, num_workers: 1, seed: 7,
            output: in_pq.clone(),
        };
        run_gen_pipeline(&cfg).unwrap();
        let rcfg = RerateGenericConfig {
            n: 16, br: 4, bc: 4,
            input_paths: vec![in_pq.clone()],
            output: out_pq.clone(),
            num_workers: 1,
            mode: SolverMode::Full,
            progress: None,
        };
        let st = run(&rcfg).unwrap();
        assert_eq!(st.n_input_rows, 2);
        assert!(st.n_kept >= 1);
        assert!(out_pq.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_input_9x9() {
        let dir = tmp("txt9");
        // Trivial near-solved 9×9: take a known solution and remove one cell.
        // Using a hand-crafted easy puzzle (T1).
        let line = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        assert_eq!(line.len(), 81);
        let txt = dir.join("p.txt");
        std::fs::write(&txt, format!("{}\n", line)).unwrap();
        let out_pq = dir.join("out.parquet");
        let rcfg = RerateGenericConfig {
            n: 9, br: 3, bc: 3,
            input_paths: vec![txt],
            output: out_pq.clone(),
            num_workers: 1,
            mode: SolverMode::Full,
            progress: None,
        };
        let st = run(&rcfg).unwrap();
        assert_eq!(st.n_input_rows, 1);
        assert_eq!(st.n_kept, 1);
        assert!(out_pq.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_input_16x16_alphabet() {
        // Build a 16×16 puzzle: full solution with one cell blanked.
        // We construct via gen-dataset, then encode to text ourselves.
        let dir = tmp("txt16");
        let in_pq = dir.join("in.parquet");
        let cfg = GenericPipelineConfig {
            n: 16, br: 4, bc: 4,
            target_tier: Tier::T1,
            clue_min: 100, clue_max: 256,
            target_total: 1, num_workers: 1, seed: 99,
            output: in_pq.clone(),
        };
        run_gen_pipeline(&cfg).unwrap();
        // Read the puzzle column back, encode as 256-char digit string.
        let pcontent = read_parquet_puzzles(&in_pq, 256).unwrap();
        assert_eq!(pcontent.len(), 1);
        let txt = dir.join("p.txt");
        std::fs::write(&txt, format!("{}\n", pcontent[0])).unwrap();
        let out_pq = dir.join("out.parquet");
        let rcfg = RerateGenericConfig {
            n: 16, br: 4, bc: 4,
            input_paths: vec![txt],
            output: out_pq.clone(),
            num_workers: 1,
            mode: SolverMode::Full,
            progress: None,
        };
        let st = run(&rcfg).unwrap();
        assert_eq!(st.n_input_rows, 1);
        assert_eq!(st.n_kept, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// R3.3a: re-rate with `mode = Solve` produces a kept record for a
    /// singles-only puzzle without running the cascade.
    #[test]
    fn rerate_mode_solve_skips_cascade() {
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79\n";
        let txt = tmp("rerate_solve_in").with_extension("txt");
        std::fs::write(&txt, p).unwrap();
        let out_pq = tmp("rerate_solve_out").with_extension("parquet");
        let rcfg = RerateGenericConfig {
            n: 9, br: 3, bc: 3,
            input_paths: vec![txt.clone()],
            output: out_pq.clone(),
            num_workers: 1,
            mode: SolverMode::Solve,
            progress: None,
        };
        let st = run(&rcfg).unwrap();
        assert_eq!(st.n_input_rows, 1);
        assert_eq!(st.n_kept, 1);
        assert_eq!(st.n_rater_errors, 0);
        let _ = std::fs::remove_file(&txt);
        let _ = std::fs::remove_file(&out_pq);
    }
}
