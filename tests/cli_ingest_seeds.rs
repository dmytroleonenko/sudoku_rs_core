//! R3.4 Stage S — `ingest-seeds` CLI E2E smoke tests.
//!
//! Three puzzles from the magictour top1465 set; known to be unique 9×9.

use std::io::Write;
use std::path::PathBuf;

fn cargo_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sudoku_rs_core"))
}

/// Three known valid 9×9 puzzles (dot-encoded, unique solutions).
const PUZZLES: &[&str] = &[
    "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.",
    "..53.....8......2..7..1.5..4....53...1..7...6..32...8..6.5....9..4....3......97..",
    "12..4......5.69.1...9...5.........7.7...52.9..3......2.9.6...5.4..9..8.1..3...9.4",
];

fn write_puzzle_file(p: &std::path::Path, lines: &[&str]) {
    let mut f = std::fs::File::create(p).unwrap();
    for l in lines {
        f.write_all(l.as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

#[test]
fn ingest_seeds_no_solve_basic() {
    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("puzzles.txt");
    let out = dir.path().join("out.parquet");
    write_puzzle_file(&inp, PUZZLES);

    let status = std::process::Command::new(cargo_bin())
        .args([
            "ingest-seeds",
            "--in",
            inp.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--label",
            "test_no_solve",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "ingest-seeds exited with {:?}", status);

    // Output parquet must exist and have non-zero size.
    let meta = std::fs::metadata(&out).unwrap();
    assert!(meta.len() > 0, "parquet output is empty");

    // Manifest sidecar must exist.
    let manifest_path = dir.path().join("out.parquet.manifest.json");
    assert!(
        manifest_path.exists(),
        "manifest sidecar {:?} not found",
        manifest_path
    );
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest.contains("\"count_kept\": 3"), "manifest count_kept mismatch: {}", manifest);
    assert!(manifest.contains("\"label\": \"test_no_solve\""), "manifest label missing");
}

#[test]
fn ingest_seeds_with_solve() {
    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("puzzles.txt");
    let out = dir.path().join("solved.parquet");
    // Only use 1 puzzle to keep it fast.
    write_puzzle_file(&inp, &PUZZLES[..1]);

    let status = std::process::Command::new(cargo_bin())
        .args([
            "ingest-seeds",
            "--in",
            inp.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--solve",
            "--label",
            "test_solve",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "ingest-seeds --solve exited with {:?}", status);

    let meta = std::fs::metadata(&out).unwrap();
    assert!(meta.len() > 0, "parquet output is empty");

    let manifest_path = dir.path().join("solved.parquet.manifest.json");
    assert!(manifest_path.exists(), "manifest not found");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    // 1 puzzle, should be unique.
    assert!(manifest.contains("\"count_unique\": 1"), "expected 1 unique: {}", manifest);
}

#[test]
fn ingest_seeds_skips_comments_and_blank_lines() {
    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("mixed.txt");
    let out = dir.path().join("mixed.parquet");
    {
        let mut f = std::fs::File::create(&inp).unwrap();
        writeln!(f, "# comment line").unwrap();
        writeln!(f).unwrap(); // blank
        writeln!(f, "{}", PUZZLES[0]).unwrap();
        writeln!(f, "# another comment").unwrap();
        writeln!(f, "{}", PUZZLES[1]).unwrap();
    }

    let status = std::process::Command::new(cargo_bin())
        .args([
            "ingest-seeds",
            "--in",
            inp.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let manifest_path = dir.path().join("mixed.parquet.manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    // Only 2 real puzzle lines.
    assert!(manifest.contains("\"count_kept\": 2"), "expected 2 kept: {}", manifest);
}

#[test]
fn ingest_seeds_max_cap() {
    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("three.txt");
    let out = dir.path().join("capped.parquet");
    write_puzzle_file(&inp, PUZZLES);

    let status = std::process::Command::new(cargo_bin())
        .args([
            "ingest-seeds",
            "--in",
            inp.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--max",
            "2",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let manifest_path = dir.path().join("capped.parquet.manifest.json");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(manifest.contains("\"count_kept\": 2"), "expected 2 kept (cap): {}", manifest);
}
