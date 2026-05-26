//! Census of structural tridagons across input puzzles. Reads 81-char
//! puzzle strings from argv files (one per line), solves each to a unique
//! solution, then runs `detect_tridagons` and prints the count + parity
//! breakdown.

use sudoku_rs_core::generic::grid::Grid;
use sudoku_rs_core::generic::search::{count_solutions_up_to, solve_unique};
use sudoku_rs_core::shc::tridagon::detect_tridagons;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: tridagon_census <file> [<file> ...]");
        std::process::exit(2);
    }
    for path in &args {
        let text = std::fs::read_to_string(path).expect("read");
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.len() != 81 {
                continue;
            }
            let grid = match Grid::<9, 3, 3>::from_str(line) {
                Some(g) => g,
                None => {
                    println!("[{}:{}] unparseable", path, lineno + 1);
                    continue;
                }
            };
            let n_sol = count_solutions_up_to::<9, 3, 3>(&grid, 2);
            if n_sol != 1 {
                println!("[{}:{}] non-unique ({} solutions)", path, lineno + 1, n_sol);
                continue;
            }
            let sol = match solve_unique::<9, 3, 3>(&grid) {
                Some(s) => s.to_string_grid(),
                None => {
                    println!("[{}:{}] solve_unique failed", path, lineno + 1);
                    continue;
                }
            };
            let tris = detect_tridagons(&sol);
            let n_odd = tris.iter().filter(|t| !t.even_total_parity()).count();
            let n_even = tris.len() - n_odd;
            println!(
                "[{}:{}] tridagons={} (odd_parity={}, even_parity={})",
                path,
                lineno + 1,
                tris.len(),
                n_odd,
                n_even
            );
            // Print up to 3 sample triplets.
            for t in tris.iter().take(3) {
                println!(
                    "   sample: triplet={:?} boxes={:?} parity={:?}",
                    t.triplet, t.boxes, t.parity
                );
            }
        }
    }
}
