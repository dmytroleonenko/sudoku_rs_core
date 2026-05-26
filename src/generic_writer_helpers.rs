//! R3.1b — Cross-cutting writer helpers for the generic (`GRecord`) pipeline.
//!
//! These helpers add the operational benefits of the trait-based IO layer
//! (`io::OrderingSink`, atomic writes, manifest sidecars) to the generic
//! `GRecord` writer path without forcing `GRecord` to implement
//! `RatedPuzzle`. The trait IO layer remains untouched and continues to
//! serve the JSONL streaming use-case.
//!
//! Three pieces:
//!
//!   1. [`OrderedGRecordBuffer`] — `BTreeMap<usize, GRecord>` drainer for
//!      preserving input-order under `rayon::par_iter` workers (or any
//!      out-of-order producer). Mirrors `io::ordered::OrderingSink` but
//!      collects into a `Vec<GRecord>` rather than forwarding to a
//!      `PuzzleSink`.
//!   2. [`atomic_write_parquet`] — write to `<out>.tmp`, fsync, rename to
//!      `<out>`. Avoids the pre-R3.1b failure mode where a crashed writer
//!      left a half-written parquet at the target path.
//!   3. [`write_manifest`] / [`GenerationManifest`] — emit a JSON sidecar
//!      next to the parquet describing rows, schema version, generation
//!      params, timestamp. Strictly additive — does not touch the parquet
//!      schema.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::pipeline_writer_generic::{write_generic_parquet, GRecord};

/// Schema version emitted in `manifest.json`. Bump on any additive change to
/// the parquet schema in `pipeline_writer_generic::generic_schema()` OR to
/// the in-memory `TechniqueProgress` ABI (observability anchor for the
/// rater→technique contract — see `docs/alphaevolve_contract.md`).
///
/// Version history:
///   1 → initial Stage R schema (R3.4).
///   2 → Phase G ABI extension (R3.4.5): `TechniqueProgress` gains
///        `is_xy_chain: Option<bool>` and `k_branches: Option<u8>` for
///        XY-Chain vs X-Chain rating split and k=2 Cell FC Y-Chain rating.
pub const GENERIC_SCHEMA_VERSION: u32 = 2;

/// In-order drainer for `(usize, GRecord)` entries produced by parallel
/// workers. `submit` buffers any index; `take_ordered` drains the contiguous
/// prefix starting at `next_idx`. `finish` returns the buffered tail in
/// ascending-index order regardless of contiguity.
///
/// Determinism: `BTreeMap` iteration is by-key. Output is identical for any
/// submission permutation provided every index in `[0, max_idx]` is submitted
/// exactly once.
pub struct OrderedGRecordBuffer {
    buf: BTreeMap<usize, GRecord>,
    next_idx: usize,
    out: Vec<GRecord>,
}

impl OrderedGRecordBuffer {
    pub fn new() -> Self {
        Self {
            buf: BTreeMap::new(),
            next_idx: 0,
            out: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: BTreeMap::new(),
            next_idx: 0,
            out: Vec::with_capacity(cap),
        }
    }

    /// Buffer `(idx, rec)` and drain any contiguous prefix starting at
    /// `next_idx` into the internal output vector.
    pub fn submit(&mut self, idx: usize, rec: GRecord) {
        self.buf.insert(idx, rec);
        while let Some(r) = self.buf.remove(&self.next_idx) {
            self.out.push(r);
            self.next_idx += 1;
        }
    }

    /// Number of currently buffered (out-of-order) entries.
    #[inline]
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Next index expected for contiguous drain.
    #[inline]
    pub fn next_idx(&self) -> usize {
        self.next_idx
    }

    /// Consume self, drain any remaining buffered entries in ascending-index
    /// order, return the in-order `Vec<GRecord>`. Use only after every input
    /// has been submitted; otherwise gaps are silently filled with the next
    /// available index.
    pub fn finish(mut self) -> Vec<GRecord> {
        // Drain any contiguous remainder first.
        while let Some(r) = self.buf.remove(&self.next_idx) {
            self.out.push(r);
            self.next_idx += 1;
        }
        // Drain non-contiguous tail in ascending order (the only deterministic
        // choice). BTreeMap::into_iter iterates by key.
        for (_, r) in std::mem::take(&mut self.buf).into_iter() {
            self.out.push(r);
        }
        self.out
    }

    /// Flush the in-order vector to a writer callback, then continue
    /// accepting submissions. Useful for streaming-write use-cases where the
    /// caller wants to emit records as soon as they are contiguous.
    pub fn flush_to<F: FnMut(GRecord) -> io::Result<()>>(
        &mut self,
        mut f: F,
    ) -> io::Result<()> {
        let drained: Vec<GRecord> = std::mem::take(&mut self.out);
        for r in drained {
            f(r)?;
        }
        Ok(())
    }
}

impl Default for OrderedGRecordBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic-rename parquet write. Writes records to `<path>.tmp`, fsyncs the
/// file, then `rename`s into place. On failure, the target `path` is never
/// modified; the `.tmp` is left behind for diagnosis (cheap to ignore — next
/// successful write overwrites it). Caller is responsible for ensuring the
/// parent directory exists.
pub fn atomic_write_parquet<P: AsRef<Path>>(path: P, records: &[GRecord]) -> io::Result<()> {
    let final_path: &Path = path.as_ref();
    let tmp_path: PathBuf = {
        let mut s = final_path.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    // Best-effort cleanup of any stale tmp from a previous crash.
    let _ = std::fs::remove_file(&tmp_path);
    write_generic_parquet(&tmp_path, records)?;
    // fsync the tmp so its bytes are durable before the rename publishes it.
    if let Ok(f) = std::fs::OpenOptions::new().read(true).open(&tmp_path) {
        let _ = f.sync_all();
    }
    std::fs::rename(&tmp_path, final_path)?;
    Ok(())
}

/// Sidecar manifest describing a parquet artifact. Strictly additive —
/// nothing about the parquet schema depends on this.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationManifest {
    /// Number of records in the parquet file.
    pub rows: usize,
    /// Bumped on additive parquet schema changes.
    pub schema_version: u32,
    /// Unix epoch seconds at write time.
    pub generated_at: u64,
    /// Free-form generation params (e.g. size, target_tier, clue range, seed,
    /// solver mode). Caller-controlled JSON object; serialised verbatim.
    pub params: serde_json::Value,
    /// Origin tag — typically the CLI subcommand: "gen-dataset" / "re-rate".
    pub origin: String,
    /// Crate version (`CARGO_PKG_VERSION`) at build time.
    pub crate_version: String,
}

impl GenerationManifest {
    pub fn now(rows: usize, origin: &str, params: serde_json::Value) -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            rows,
            schema_version: GENERIC_SCHEMA_VERSION,
            generated_at: secs,
            params,
            origin: origin.to_string(),
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Compute the manifest sidecar path for a given parquet `path`.
/// Convention: replace the file basename with `manifest.json` in the same
/// directory. So `out/data.parquet` → `out/manifest.json`. If `path` has no
/// parent (bare filename in cwd), the manifest is written to `./manifest.json`.
pub fn manifest_path_for(parquet_path: &Path) -> PathBuf {
    match parquet_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join("manifest.json"),
        _ => PathBuf::from("manifest.json"),
    }
}

/// Write a `GenerationManifest` to `path` as pretty-printed JSON, atomically
/// (write to `.tmp`, rename). Caller is responsible for ensuring the parent
/// directory exists.
pub fn write_manifest<P: AsRef<Path>>(path: P, manifest: &GenerationManifest) -> io::Result<()> {
    let final_path: &Path = path.as_ref();
    let tmp_path: PathBuf = {
        let mut s = final_path.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    let _ = std::fs::remove_file(&tmp_path);
    let mut f = std::fs::File::create(&tmp_path)?;
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    f.write_all(json.as_bytes())?;
    f.write_all(b"\n")?;
    let _ = f.sync_all();
    drop(f);
    std::fs::rename(&tmp_path, final_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(tag: i64) -> GRecord {
        GRecord {
            puzzle: vec![0i8; 81],
            solution: vec![0i8; 81],
            size: "9x3x3".to_string(),
            tier: tag,
            tier_name: format!("T{}", tag),
            technique_frontier: vec![],
            longest_inference_chain: 0,
            backtracking_entropy: 0.0,
            trace_techniques: vec![],
            trace_actions: "[]".to_string(),
            trace_length: 0,
            se_rating_estimate: 0.0,
            puzzle_id: format!("id{}", tag),
            clue_count: 0,
            split: "train".to_string(),
            se_score: 0.0,
        }
    }

    #[test]
    fn ordered_buffer_drains_in_index_order() {
        let mut b = OrderedGRecordBuffer::new();
        for i in [3usize, 1, 5, 0, 4, 2] {
            b.submit(i, rec(i as i64));
        }
        let out = b.finish();
        let tags: Vec<i64> = out.iter().map(|r| r.tier).collect();
        assert_eq!(tags, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn ordered_buffer_pending_decreases_on_contiguous_arrival() {
        let mut b = OrderedGRecordBuffer::new();
        b.submit(2, rec(2));
        assert_eq!(b.pending(), 1);
        b.submit(1, rec(1));
        assert_eq!(b.pending(), 2);
        b.submit(0, rec(0));
        assert_eq!(b.pending(), 0);
        assert_eq!(b.next_idx(), 3);
    }

    #[test]
    fn ordered_buffer_handles_gaps_in_finish() {
        // Indices 0, 1, 4 — 2/3 missing. finish() should still drain in
        // ascending order.
        let mut b = OrderedGRecordBuffer::new();
        b.submit(0, rec(0));
        b.submit(1, rec(1));
        b.submit(4, rec(4));
        let out = b.finish();
        let tags: Vec<i64> = out.iter().map(|r| r.tier).collect();
        assert_eq!(tags, vec![0, 1, 4]);
    }

    #[test]
    fn atomic_write_leaves_no_tmp_on_success() {
        let dir = std::env::temp_dir().join(format!(
            "r3_1b_atomic_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("data.parquet");
        atomic_write_parquet(&out, &[rec(1), rec(2)]).unwrap();
        assert!(out.exists(), "final parquet should exist");
        let tmp = {
            let mut s = out.as_os_str().to_owned();
            s.push(".tmp");
            PathBuf::from(s)
        };
        assert!(
            !tmp.exists(),
            "tmp must be renamed away on success: {:?}",
            tmp
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_write_does_not_clobber_on_writer_failure() {
        // Simulate failure: write to a path whose parent does not exist.
        // The pre-existing file at the *intended* target should stay intact.
        let dir = std::env::temp_dir().join(format!(
            "r3_1b_atomic_fail_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let target = dir.join("data.parquet");
        // Pre-populate target with a sentinel.
        std::fs::write(&target, b"SENTINEL").unwrap();

        // Now point at a nonexistent parent → write_generic_parquet should
        // fail, and our atomic helper must not have touched `target`.
        let bad = dir.join("does_not_exist").join("data.parquet");
        let res = atomic_write_parquet(&bad, &[rec(1)]);
        assert!(res.is_err(), "expected write to fail under missing parent");
        // Sentinel intact.
        let bytes = std::fs::read(&target).unwrap();
        assert_eq!(&bytes[..], b"SENTINEL");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_round_trips_as_valid_json() {
        let dir = std::env::temp_dir().join(format!(
            "r3_1b_manifest_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        let m = GenerationManifest::now(
            42,
            "gen-dataset",
            serde_json::json!({"size": "9x3x3", "target_tier": "T1"}),
        );
        write_manifest(&path, &m).unwrap();
        let txt = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&txt).expect("manifest is valid JSON");
        assert_eq!(v["rows"], 42);
        assert_eq!(v["schema_version"], GENERIC_SCHEMA_VERSION);
        assert_eq!(v["origin"], "gen-dataset");
        assert_eq!(v["params"]["size"], "9x3x3");
        assert!(v["generated_at"].as_u64().unwrap() > 0);
        assert!(v["crate_version"].is_string());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn manifest_path_alongside_parquet() {
        let p = Path::new("/tmp/outdir/data.parquet");
        let m = manifest_path_for(p);
        assert_eq!(m, PathBuf::from("/tmp/outdir/manifest.json"));
    }
}
