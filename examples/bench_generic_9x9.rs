//! Throughput micro-bench: generic gen_unique_puzzle::<9,3,3> single-thread.
//!
//! Mirrors the workload of `sudoku_rs_core gen-dataset --num 1000
//! --target-clues 25 --threads 1 --seed 42` for an apples-to-apples
//! comparison of the const-generic generator vs the legacy 9-specialised one.

use rand::SeedableRng;
use rand::rngs::StdRng;
use sudoku_rs_core::generic::{gen_unique_puzzle, GenConfig};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let cfg = GenConfig {
        target_clues: 25,
        max_attempts: 0,
    };
    let mut rng = StdRng::seed_from_u64(42);
    let start = std::time::Instant::now();
    let mut total_clues: u64 = 0;
    for _ in 0..n {
        let (_g, clues) = gen_unique_puzzle::<9, 3, 3, _>(&mut rng, &cfg);
        total_clues += clues as u64;
    }
    let elapsed = start.elapsed();
    let pps = (n as f64) / elapsed.as_secs_f64();
    let mean_clues = (total_clues as f64) / (n as f64);
    println!(
        "generic 9x9: {} puzzles in {:.3}s, {:.1} pps, mean_clues={:.2}",
        n,
        elapsed.as_secs_f64(),
        pps,
        mean_clues
    );
}
