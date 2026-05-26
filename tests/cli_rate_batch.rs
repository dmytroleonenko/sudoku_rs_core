//! R3.0a — `rate-batch` CLI E2E.
//!
//! Smoke: 100 9×9 puzzles → 100 sorted JSONL rows; every row has the schema
//! fields R3.0a guarantees; T1 rows have `frontier == []` (singles only).
//!
//! Throughput probe: 1000 puzzles on 8 threads — log pps to stderr; assert ≥
//! 500 pps (slack from 1000 target). Skip the throughput assertion when
//! running under debug-mode `cargo test` (release-only behaviour).

use std::io::Write;
use std::path::PathBuf;

use serde_json::Value;
use sudoku_rs_core::generic::generator::{gen_unique_puzzle, GenConfig};
use sudoku_rs_core::generic::grid::Grid as GGrid;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn cargo_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sudoku_rs_core"))
}

fn gen_9x9_puzzles(n: usize, seed: u64) -> Vec<String> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let cfg = GenConfig::new(30);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let g: GGrid<9, 3, 3> = gen_unique_puzzle::<9, 3, 3, _>(&mut rng, &cfg).0;
        out.push(g.to_string_grid());
    }
    out
}

fn write_input(p: &std::path::Path, lines: &[String]) {
    let mut f = std::fs::File::create(p).unwrap();
    for l in lines {
        f.write_all(l.as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

#[test]
fn rate_batch_smoke_100_puzzles_jsonl_schema() {
    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("in.txt");
    let out = dir.path().join("out.jsonl");
    let puzzles = gen_9x9_puzzles(100, 42);
    write_input(&inp, &puzzles);

    let st = std::process::Command::new(cargo_bin())
        .args([
            "rate-batch",
            "--input",
            inp.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--size",
            "9x3x3",
            "--threads",
            "4",
        ])
        .status()
        .unwrap();
    assert!(st.success());

    let body = std::fs::read_to_string(&out).unwrap();
    let rows: Vec<Value> = body
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Value>(l).expect("valid JSON"))
        .collect();
    assert_eq!(rows.len(), 100, "expected 100 JSONL rows");

    // Strict input order.
    for (k, r) in rows.iter().enumerate() {
        assert_eq!(r["i"].as_u64().unwrap(), k as u64, "row {} out of order", k);
        // Schema required fields.
        for field in [
            "puzzle",
            "tier",
            "wave_depth",
            "frontier",
            "trace",
            "backtrack_steps",
            "unique_solution",
            "solved",
            "rater_error",
        ] {
            assert!(r.get(field).is_some(), "row {} missing field {}", k, field);
        }
        // T1 puzzles must have empty frontier (singles-only).
        if r["tier"].as_str() == Some("T1") {
            let f = r["frontier"].as_array().unwrap();
            assert!(f.is_empty(), "T1 row {} has non-empty frontier: {:?}", k, f);
            assert_eq!(r["wave_depth"].as_u64().unwrap(), 0,
                "T1 must have wave_depth=0 (solved by initial propagation), row {}", k);
        }
        // All gen_unique_puzzle outputs are unique.
        assert!(r["unique_solution"].as_bool().unwrap(),
            "row {}: gen_unique_puzzle output must be unique", k);
        assert!(r["solved"].as_bool().unwrap(),
            "row {}: gen_unique_puzzle T1/T2/T3 must solve", k);
    }
}

#[test]
fn rate_batch_throughput_probe_1000_puz_8_threads() {
    // Release-mode probe; under `cargo test --release`. Skip silently if the
    // binary was built in debug mode (heuristic: bin path contains /debug/).
    let bin = cargo_bin();
    let bin_str = bin.to_string_lossy();
    let release = bin_str.contains("/release/") || bin_str.contains("\\release\\");

    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("in.txt");
    let out = dir.path().join("out.jsonl");
    let puzzles = gen_9x9_puzzles(1000, 7);
    write_input(&inp, &puzzles);

    let t0 = std::time::Instant::now();
    let st = std::process::Command::new(&bin)
        .args([
            "rate-batch",
            "--input",
            inp.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--size",
            "9x3x3",
            "--threads",
            "8",
        ])
        .status()
        .unwrap();
    assert!(st.success());
    let dt = t0.elapsed().as_secs_f64();
    let pps = 1000.0 / dt.max(1e-9);
    eprintln!(
        "rate-batch throughput probe: 1000 puzzles, 8 threads, {:.2}s, {:.0} pps (release={})",
        dt, pps, release
    );
    if release {
        // Slack: target 1000, gate at 500.
        assert!(pps >= 500.0, "throughput regression: {:.0} pps < 500", pps);
    }
}
