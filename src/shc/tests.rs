//! Phase 1 sanity tests for the SHC port.

use super::board::{ApplyOutcome, Board};
use super::tb::{propagate, TbLevel};
use super::uniqueness::verify_unique_solution;

/// First puzzle from `/tmp/chain_rerate/berthier_b5_7.txt` (Berthier B5/7
/// dataset). Known to have a unique solution.
const BERTHIER_FIRST: &str =
    "..34......5...912.7...2.....1.5.7..86...9...7.......34..2.............9.9...61.75";

/// A trivially singles-solvable puzzle (very-easy sudoku). Source: a generic
/// example commonly used for solver smoke tests.
const EASY_SINGLES: &str =
    "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..";

#[test]
fn roundtrip_81_chars() {
    let b = Board::from_81_chars(BERTHIER_FIRST).expect("parse");
    let s = b.to_81_chars();
    assert_eq!(s.len(), 81);
    // Clue cells must round-trip identically; empty markers normalize to '.'.
    for (i, (a, b)) in BERTHIER_FIRST.chars().zip(s.chars()).enumerate() {
        let a_norm = if a == '0' { '.' } else { a };
        if a_norm != '.' {
            assert_eq!(a_norm, b, "mismatch at {}", i);
        }
    }
}

#[test]
fn berthier_is_unique() {
    let b = Board::from_81_chars(BERTHIER_FIRST).expect("parse");
    verify_unique_solution(&b).expect("unique");
}

#[test]
fn easy_solves_with_singles() {
    let mut b = Board::from_81_chars(EASY_SINGLES).expect("parse");
    let outcome = propagate(&mut b, TbLevel::L0);
    assert_eq!(outcome, ApplyOutcome::Solved, "expected naked+hidden singles to finish easy puzzle");
    assert!(b.is_solved());
    // Sanity: each row/col/box should contain digits 0..9 exactly once.
    let regs = super::board::regions();
    for region in regs.iter() {
        let mut seen = [false; 9];
        for &cell_id in region.cells.iter() {
            let d = b.cells[cell_id as usize].assigned.expect("solved cell");
            assert!(!seen[d as usize]);
            seen[d as usize] = true;
        }
    }
}

#[test]
fn contradiction_detected_on_duplicate_in_row() {
    // Place '1' at (0,0); attempt to assign '1' at (0,1) — same row.
    let mut b = Board::empty();
    assert!(matches!(b.assign(0, 0), ApplyOutcome::Continue));
    let outcome = b.assign(1, 0);
    assert_eq!(outcome, ApplyOutcome::Contradiction);
}

#[test]
fn tb_l1_does_no_harm_on_easy() {
    // L1 (with box/line) must still solve the easy puzzle.
    let mut b = Board::from_81_chars(EASY_SINGLES).expect("parse");
    let outcome = propagate(&mut b, TbLevel::L1);
    assert_eq!(outcome, ApplyOutcome::Solved);
}

#[test]
fn bad_length_rejected() {
    let bad = "....";
    assert!(Board::from_81_chars(bad).is_err());
}

#[test]
fn bad_char_rejected() {
    let mut s = String::from(BERTHIER_FIRST);
    // Replace first char with an illegal one.
    s.replace_range(0..1, "X");
    assert!(Board::from_81_chars(&s).is_err());
}
