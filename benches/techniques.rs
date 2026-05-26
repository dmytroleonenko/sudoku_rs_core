//! Bench harness for all registered techniques (Stage R, R3.4).
//!
//! Runs `all_techniques_9x9()` over a small fixture set and reports
//! median µs/grid per technique group. AlphaEvolve parses Criterion's
//! JSON output from `target/criterion/`.
//!
//! Usage:
//!   cargo bench --bench techniques           # full run
//!   cargo bench --bench techniques -- aic    # single technique filter
//!   cargo bench --bench techniques --no-run  # compile check only

use criterion::{criterion_group, criterion_main, Criterion};
use sudoku_rs_core::generic::grid::Grid;
use sudoku_rs_core::generic::techniques::all_techniques_9x9;

// ---------------------------------------------------------------------------
// Fixture grids — 5 known 9×9 puzzles (81-char strings, '.' = empty).
// These span T1/T2/T3 difficulty; more will be added per-technique in Stage 0.
// ---------------------------------------------------------------------------

const FIXTURE_PUZZLES: &[&str] = &[
    // T1/T2 — "Easy" puzzle
    "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
    // T2 — requires locked candidates
    "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..",
    // T3 — requires chains / fish
    "1.......2.9.4...5...6...7...5.9.3.......7.......85..4.7.....6...3...9.8...2.....1",
    // T3 — another chain puzzle
    "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......",
    // T3 — AIC/ALS target
    "......9.7...42.18....7.5.261..9.4...9.1.....4...5.7..513.6.8....83.15...6.8......",
];

fn parse_9x9(s: &str) -> Grid<9, 3, 3> {
    let s = s.trim();
    assert_eq!(s.len(), 81, "fixture must be 81 chars");
    let mut grid = Grid::<9, 3, 3>::empty();
    for (i, ch) in s.chars().enumerate() {
        if ch != '.' && ch != '0' {
            let d = ch as u8 - b'0';
            let _ = grid.assign(i, d);
        }
    }
    grid
}

fn bench_all_techniques(c: &mut Criterion) {
    let fixtures: Vec<Grid<9, 3, 3>> = FIXTURE_PUZZLES.iter().map(|s| parse_9x9(s)).collect();
    let techniques = all_techniques_9x9();

    c.bench_function("all_techniques_9x9_fixture_set", |b| {
        b.iter(|| {
            for fixture in &fixtures {
                let mut g = fixture.clone();
                for tech in &techniques {
                    let _ = tech.apply(&mut g);
                }
            }
        });
    });
}

criterion_group!(benches, bench_all_techniques);
criterion_main!(benches);
