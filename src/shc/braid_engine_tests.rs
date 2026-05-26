//! Phase 2 tests: braid engine + wave driver.
//!
//! B=5/6/7 fixtures are drawn from `/tmp/chain_rerate/berthier_b5_7.txt`
//! with ratings cross-checked against `/tmp/chain_rerate/b5_7_shc_out.txt`
//! (SHC.jar v6.2, max-length=9, buffer-size=1000000).

use super::board::Board;
use super::wave::{rate_b, RateError};

const BUFFER_SIZE: usize = 200_000;
const MAX_LENGTH: u16 = 9;

/// A very-easy puzzle, solved by naked+hidden singles alone — B=0.
const EASY_B0: &str =
    "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..";

/// Berthier corpus puzzle `cbg000#22`, labeled B=1 — solved by TB(L1)
/// (box/line) but not by TB(L0) alone. Before the B0/B1 off-by-one fix
/// (rate_b running TB(L1) up front and returning 0), this puzzle was
/// misclassified as B=0; the fix restores Java's separate i=0 (L0 only) /
/// i=1 (L1 only) phases.
const PUZZLE_B1: &str =
    ".23....8.......12.7.91.........97.46...3....29..6......6.9..5.457.....98.....4...";

/// Berthier corpus picks (cross-checked against b5_7_shc_out.txt).
const PUZZLE_B5: &str =
    "..34......5...912.7...2.....1.5.7..86...9...7.......34..2.............9.9...61.75";
const PUZZLE_B6: &str =
    "12..56.8..........7.8...5.6......39....97...4..5.2.....7...84..5....36..68......1";
const PUZZLE_B7: &str =
    "12.....8...6..9......2.36...1..943....4......9..1...2.3..84..97..7.....8.41.7.2..";

#[test]
fn berthier_b1_rates_b1() {
    let mut b = Board::from_81_chars(PUZZLE_B1).expect("parse");
    let r = rate_b(&mut b, MAX_LENGTH, BUFFER_SIZE).expect("rate");
    assert_eq!(r, 1, "expected B=1 (TB-L1 only), got {}", r);
}

#[test]
fn easy_rates_b0() {
    let mut b = Board::from_81_chars(EASY_B0).expect("parse");
    let r = rate_b(&mut b, MAX_LENGTH, BUFFER_SIZE).expect("rate");
    assert_eq!(r, 0, "easy puzzle should be B=0 (singles only)");
}

#[test]
fn berthier_b5_rates_b5() {
    let mut b = Board::from_81_chars(PUZZLE_B5).expect("parse");
    let r = rate_b(&mut b, MAX_LENGTH, BUFFER_SIZE).expect("rate");
    assert_eq!(r, 5, "expected B=5, got {}", r);
}

#[test]
fn berthier_b6_rates_b6() {
    let mut b = Board::from_81_chars(PUZZLE_B6).expect("parse");
    let r = rate_b(&mut b, MAX_LENGTH, BUFFER_SIZE).expect("rate");
    assert_eq!(r, 6, "expected B=6, got {}", r);
}

#[test]
fn berthier_b7_rates_b7() {
    let mut b = Board::from_81_chars(PUZZLE_B7).expect("parse");
    let r = rate_b(&mut b, MAX_LENGTH, BUFFER_SIZE).expect("rate");
    assert_eq!(r, 7, "expected B=7, got {}", r);
}

/// Full 30-puzzle corpus check. Loads from /tmp/chain_rerate. Runtime ~40s
/// release on M3 Max — kept as a regular test (no `#[ignore]`) because the
/// Phase 3 brief explicitly authorizes inclusion at <60s wall-clock.
#[test]
fn berthier_b5_7_corpus_all_30() {
    let puzzles = std::fs::read_to_string("/tmp/chain_rerate/berthier_b5_7.txt")
        .expect("read berthier_b5_7.txt");
    let labels_raw = std::fs::read_to_string("/tmp/chain_rerate/b5_7_shc_out.txt")
        .expect("read b5_7_shc_out.txt");
    let labels: Vec<u16> = labels_raw
        .lines()
        .skip(1) // header line
        .filter_map(|s| s.trim().parse::<u16>().ok())
        .collect();
    let puzzle_lines: Vec<&str> = puzzles.lines().collect();
    assert_eq!(puzzle_lines.len(), labels.len(), "puzzle/label count mismatch");
    let mut mismatches = Vec::new();
    for (i, (p, &exp)) in puzzle_lines.iter().zip(labels.iter()).enumerate() {
        let mut b = Board::from_81_chars(p).expect("parse");
        match rate_b(&mut b, MAX_LENGTH, BUFFER_SIZE) {
            Ok(r) if r == exp => {}
            Ok(r) => mismatches.push(format!("puzzle {} expected B={} got B={}", i, exp, r)),
            Err(e) => mismatches.push(format!("puzzle {} expected B={} got err={:?}", i, exp, e)),
        }
    }
    assert!(mismatches.is_empty(), "mismatches: {:#?}", mismatches);
}
