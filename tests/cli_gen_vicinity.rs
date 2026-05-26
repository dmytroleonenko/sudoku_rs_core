//! R3.4 Stage 2 — `gen-vicinity` CLI E2E smoke test.

use std::io::Write;
use std::path::PathBuf;

fn cargo_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_sudoku_rs_core"))
}

/// One known valid T3 9×9 puzzle (unique solution, ~26 clues).
const SEED_PUZZLE: &str = "85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.";

fn write_seed_file(p: &std::path::Path, puzzles: &[&str]) {
    let mut f = std::fs::File::create(p).unwrap();
    for puz in puzzles {
        f.write_all(puz.as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

/// Basic smoke: single seed, low target_se, small budget.
/// Asserts parquet exists and has >= 1 row with puzzle and se_score columns.
#[test]
fn gen_vicinity_basic_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let seed_file = dir.path().join("seeds.txt");
    let out_file = dir.path().join("out.parquet");

    write_seed_file(&seed_file, &[SEED_PUZZLE]);

    let status = std::process::Command::new(cargo_bin())
        .args([
            "gen-vicinity",
            "--seed-in",
            seed_file.to_str().unwrap(),
            "--out",
            out_file.to_str().unwrap(),
            "--target-se",
            "1.0",
            "--budget-iters",
            "20",
            "--max-outputs",
            "5",
            "--seed",
            "42",
        ])
        .status()
        .unwrap();

    assert!(status.success(), "gen-vicinity exited with {:?}", status);

    // Parquet must exist and have non-zero size.
    let meta = std::fs::metadata(&out_file).unwrap();
    assert!(meta.len() > 0, "parquet output is empty");

    // Manifest sidecar must exist.
    let manifest_path = dir.path().join("out.parquet.manifest.json");
    assert!(
        manifest_path.exists(),
        "manifest sidecar {:?} not found",
        manifest_path
    );

    // Read parquet to verify columns.
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::{reader::RowIter, RowAccessor};
    let file = std::fs::File::open(&out_file).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let schema_cols: Vec<String> = reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .map(|c| c.name().to_string())
        .collect();

    assert!(schema_cols.contains(&"puzzle".to_string()), "missing 'puzzle' column");
    assert!(schema_cols.contains(&"se_score".to_string()), "missing 'se_score' column");

    // Must have >= 1 row.
    let num_rows = reader.metadata().file_metadata().num_rows();
    assert!(num_rows >= 1, "expected >= 1 row, got {}", num_rows);

    // All rows must have valid puzzle strings.
    let puzzle_col = schema_cols.iter().position(|n| n == "puzzle").unwrap();
    let iter = RowIter::from_file_into(Box::new(SerializedFileReader::new(
        std::fs::File::open(&out_file).unwrap(),
    ).unwrap()));
    for row_result in iter {
        let row = row_result.unwrap();
        let puzzle = row.get_string(puzzle_col).unwrap();
        assert_eq!(puzzle.len(), 81, "puzzle has wrong length: {}", puzzle.len());
    }
}
