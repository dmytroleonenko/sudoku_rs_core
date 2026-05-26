//! Uniqueness check via simple DFS.
//!
//! Port of the `DFS.f(jeu, 2, false)` entry point in `DFS.java` plus
//! `Resolution_init.calcul_sol`. We only need a yes/no on
//! "exactly one solution"; no need to stamp `Candidat.sol`.

use super::board::{ApplyOutcome, Board};
use super::error::UniquenessError;
use super::tb::{propagate, TbLevel};

/// Returns Ok(()) iff the puzzle has exactly one solution.
pub fn verify_unique_solution(board: &Board) -> Result<(), UniquenessError> {
    let mut work = board.clone();
    let mut count: u32 = 0;
    search(&mut work, &mut count, 2);
    match count {
        0 => Err(UniquenessError::NoSolution),
        1 => Ok(()),
        _ => Err(UniquenessError::Multiple),
    }
}

/// DFS that stops as soon as `cap` solutions have been found.
fn search(board: &mut Board, count: &mut u32, cap: u32) {
    if *count >= cap {
        return;
    }
    match propagate(board, TbLevel::L0) {
        ApplyOutcome::Solved => {
            *count += 1;
            return;
        }
        ApplyOutcome::Contradiction => return,
        ApplyOutcome::Continue => {}
    }
    // Pick a branching cell: minimum-remaining-values heuristic.
    let mut best_cell: Option<u8> = None;
    let mut best_n: u32 = 10;
    for (i, c) in board.cells.iter().enumerate() {
        if c.assigned.is_some() {
            continue;
        }
        let n = c.cand_mask.count_ones();
        if n < best_n {
            best_n = n;
            best_cell = Some(i as u8);
            if best_n == 2 {
                break;
            }
        }
    }
    let Some(cell) = best_cell else {
        // No unassigned cells but not flagged solved — defensive.
        if board.is_solved() {
            *count += 1;
        }
        return;
    };
    // Iterate candidate digits via the mask: low bit -> assign -> clear -> repeat.
    let mut mask = board.cells[cell as usize].cand_mask;
    while mask != 0 {
        let d = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        let mut child = board.clone();
        match child.assign(cell, d) {
            ApplyOutcome::Contradiction => continue,
            ApplyOutcome::Solved => {
                *count += 1;
                if *count >= cap {
                    return;
                }
                continue;
            }
            ApplyOutcome::Continue => {
                search(&mut child, count, cap);
                if *count >= cap {
                    return;
                }
            }
        }
    }
}
