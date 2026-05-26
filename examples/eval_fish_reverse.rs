//! Validity + speed harness for `fish_reverse_construct`.
//!
//! Intended as the core of an OpenEvolve evaluator. Single-threaded,
//! deterministic: each puzzle's seed is derived from `base_seed + i *
//! 0x9E3779B97F4A7C15`, mirroring the splitmix-style child-seed scheme used by
//! `batch_fish_reverse_construct`.
//!
//! Usage:
//!   eval_fish_reverse [--k 2|3|4] [--n <int>] [--seed <u64>] [--json]
//!
//! Always prints one-line JSON to stdout. Exits 0 if all validity checks pass,
//! 1 otherwise.

use std::time::Instant;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use sha2::{Digest, Sha256};

use sudoku_rs_core::generic::{
    count_solutions_up_to,
    fish_reverse::{find_first_fish_size, fish_reverse_construct, FishReverseSpec},
    techniques::{TechniqueId, Tier},
};

fn fish_technique_id(k: usize) -> TechniqueId {
    match k {
        2 => TechniqueId::XWing,
        3 => TechniqueId::Swordfish,
        4 => TechniqueId::Jellyfish,
        _ => panic!("k must be 2, 3, or 4"),
    }
}

fn parse_args() -> (usize, usize, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut k: usize = 3;
    let mut n: usize = 50;
    let mut seed: u64 = 42;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--k" => {
                i += 1;
                k = args[i].parse().expect("--k must be 2, 3, or 4");
            }
            "--n" => {
                i += 1;
                n = args[i].parse().expect("--n must be an integer");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("--seed must be a u64");
            }
            "--json" => {}
            other => eprintln!("unknown arg: {}", other),
        }
        i += 1;
    }
    (k, n, seed)
}

fn main() {
    let (k, n, base_seed) = parse_args();

    let spec = FishReverseSpec {
        target_size: k,
        target_tier: Tier::T3,
        clue_min: 22,
        clue_max: 30,
        max_attempts: 5000,
        require_load_bearing: false,
        greedy_max_trials: 0,
    };

    let target_id = fish_technique_id(k);

    let mut n_ok: usize = 0;
    let mut n_total: usize = 0;
    let mut hasher = Sha256::new();

    let wall_start = Instant::now();

    for i in 0..n {
        // Per-puzzle seed: same splitmix-style derivation as batch_fish_reverse_construct
        // worker seeds. This gives deterministic, non-overlapping RNG streams per puzzle.
        let child_seed = base_seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);

        let result = fish_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);

        n_total += 1;

        let res = match result {
            Some(r) => r,
            None => {
                // construction failed → validity fails
                let line = format!("NONE|0\n");
                hasher.update(line.as_bytes());
                continue;
            }
        };

        let puzzle_str = res.puzzle.to_string_grid();
        let clue_count = res.clue_count;

        // Mirror the exact validity gates from fish_reverse_construct (lines 446-482):
        // 1. Uniqueness
        let puzzle_grid = &res.puzzle;
        let unique = count_solutions_up_to(puzzle_grid, 2) == 1;
        // 2. Target technique in rated frontier
        let frontier_ok = res.rate.frontier.contains(&target_id) && !res.rate.rater_error;
        // 3. First-firing fish is exactly K
        let first_fish_ok = find_first_fish_size::<9, 3, 3>(puzzle_grid) == Some(k);

        let valid = unique && frontier_ok && first_fish_ok;
        if valid {
            n_ok += 1;
        }

        // Fingerprint: hash puzzle_str + "|" + clue_count + "\n"
        let line = format!("{}|{}\n", puzzle_str, clue_count);
        hasher.update(line.as_bytes());
    }

    let total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let ms_per_puzzle = if n_total > 0 {
        total_ms / n_total as f64
    } else {
        0.0
    };

    let fingerprint_bytes = hasher.finalize();
    let fingerprint = format!("sha256:{}", hex::encode(fingerprint_bytes));

    let validity = n_ok == n_total;

    println!(
        "{{\"n_ok\":{},\"n_total\":{},\"fingerprint\":\"{}\",\"validity\":{},\"total_ms\":{:.1},\"ms_per_puzzle\":{:.3}}}",
        n_ok,
        n_total,
        fingerprint,
        validity,
        total_ms,
        ms_per_puzzle
    );

    std::process::exit(if validity { 0 } else { 1 });
}
