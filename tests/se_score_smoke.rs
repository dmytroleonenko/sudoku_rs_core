//! Stage 0 SE score smoke test.
//!
//! Verifies:
//! 1. `RateResult.se_score` is populated correctly for T1/T2/T3 puzzles.
//! 2. T1 puzzles (singles only) have se_score == 0.0.
//! 3. T2 puzzles (locked/pair) have se_score in [2.6, 5.0).
//! 4. T3 puzzles have se_score >= 3.2.
//! 5. Written parquet has `se_score` column with finite non-null values.

use sudoku_rs_core::generic::{
    generator::{gen_unique_puzzle, GenConfig},
    grid::Grid,
    rater::rate,
    techniques::{Tier},
};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// T1 puzzle: solvable by singles alone → se_score must be 0.0.
#[test]
fn t1_puzzle_se_score_is_zero() {
    let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
    let g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
    let r = rate(&g);
    assert_eq!(r.tier, Tier::T1, "expected T1, got {:?}", r.tier);
    assert!(r.solved);
    assert_eq!(r.se_score, 0.0, "T1 puzzle must have se_score=0.0, got {}", r.se_score);
}

/// T2 puzzles should produce se_score in the T2 weight range [2.6, 5.0).
#[test]
fn t2_puzzles_se_score_in_range() {
    let cfg = GenConfig::new(30);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let mut found = false;
    for _ in 0..500 {
        let (g, _): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng, &cfg);
        let r = rate(&g);
        if r.tier != Tier::T2 || !r.solved {
            continue;
        }
        assert!(
            r.se_score >= 2.6 && r.se_score < 5.0,
            "T2 puzzle se_score out of range: {} (frontier: {:?})",
            r.se_score, r.frontier
        );
        assert!(r.se_score.is_finite(), "se_score must be finite");
        found = true;
        break;
    }
    if !found {
        eprintln!("warn: no T2 puzzle found in 500 samples — test soft-passed");
    }
}

/// T3 puzzles should produce se_score >= 3.2 (minimum T3 technique weight is
/// XWing at 3.2).
#[test]
fn t3_puzzles_se_score_at_least_xwing() {
    let cfg = GenConfig::new(30);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
    let mut found = false;
    for _ in 0..500 {
        let (g, _): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng, &cfg);
        let r = rate(&g);
        if r.tier != Tier::T3 || !r.solved {
            continue;
        }
        assert!(
            r.se_score >= 3.2,
            "T3 puzzle se_score too low: {} (frontier: {:?})",
            r.se_score, r.frontier
        );
        assert!(r.se_score.is_finite());
        found = true;
        break;
    }
    if !found {
        eprintln!("warn: no T3 puzzle found in 500 samples — test soft-passed");
    }
}

/// Empty grid (T4Plus stuck): se_score should be 7.5 (lower bound sentinel).
#[test]
fn t4plus_empty_grid_se_score_sentinel() {
    let g: Grid<9, 3, 3> = Grid::empty();
    let r = rate(&g);
    assert_eq!(r.tier, Tier::T4Plus);
    assert!(!r.solved);
    assert!(
        r.se_score >= 7.5,
        "T4Plus stuck cascade must return se_score >= 7.5, got {}",
        r.se_score
    );
}

/// Batch of 200 random puzzles: all se_score values must be finite and non-negative.
#[test]
fn bulk_se_score_all_finite() {
    let cfg = GenConfig::new(30);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
    for i in 0..200 {
        let (g, _): (Grid<9, 3, 3>, u32) = gen_unique_puzzle(&mut rng, &cfg);
        let r = rate(&g);
        assert!(
            r.se_score.is_finite() && r.se_score >= 0.0,
            "puzzle {}: se_score not valid: {}",
            i, r.se_score
        );
    }
}

/// Parquet written by gen-dataset pipeline has se_score column with finite values.
#[test]
fn parquet_has_se_score_column() {
    use sudoku_rs_core::pipeline_writer_generic::{GenericPipelineConfig, run};
    use sudoku_rs_core::generic::techniques::Tier;

    let tmpdir = std::env::temp_dir().join(format!("se_score_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmpdir).unwrap();
    let out = tmpdir.join("se_score.parquet");

    let cfg = GenericPipelineConfig {
        n: 9, br: 3, bc: 3,
        target_tier: Tier::T1,
        clue_min: 25,
        clue_max: 81,
        target_total: 10,
        num_workers: 1,
        seed: 42,
        output: out.clone(),
    };
    run(&cfg).unwrap();

    // Read back and verify se_score column exists and is finite.
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let f = std::fs::File::open(&out).unwrap();
    let reader = ParquetRecordBatchReaderBuilder::try_new(f).unwrap().build().unwrap();
    let mut found_col = false;
    let mut n_rows = 0usize;
    for batch in reader {
        let batch = batch.unwrap();
        let idx = batch.schema().index_of("se_score").expect("se_score column must exist");
        found_col = true;
        let arr = batch.column(idx)
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("se_score must be Float64");
        for i in 0..arr.len() {
            let v = arr.value(i);
            assert!(v.is_finite() && v >= 0.0,
                "se_score[{}] is not valid: {}", i, v);
            n_rows += 1;
        }
    }
    assert!(found_col, "se_score column not found");
    assert!(n_rows > 0, "no rows in parquet");

    let _ = std::fs::remove_dir_all(&tmpdir);
}
