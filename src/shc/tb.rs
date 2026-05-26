//! Techniques de Base — basic propagator.
//!
//! Port of `TB.java`. We propagate until quiescence using:
//!   * L0: naked singles (`chercher_1`) + hidden singles (`chercher_2`)
//!   * L1: L0 + box/line (pointing + claiming) (`chercher_11`)
//!
//! Returns `Solved` / `Contradiction` / `Continue` per `ApplyOutcome`.

use super::board::{regions, ApplyOutcome, Board};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbLevel {
    L0,
    L1,
}

pub fn propagate(board: &mut Board, level: TbLevel) -> ApplyOutcome {
    propagate_counted(board, level).0
}

/// Same as [`propagate`], but also returns the number of TB rule firings
/// observed during propagation. Mirrors Java's `TB.nombre` counter (TB.java:53),
/// which increments once per outer rule firing (naked single, hidden single,
/// or box/line). Used by `chercher_cand_ctr` for priority ranking of probe
/// candidates.
pub fn propagate_counted(board: &mut Board, level: TbLevel) -> (ApplyOutcome, u32) {
    let mut nombre: u32 = 0;
    loop {
        // Initial check — already solved on entry.
        if board.is_solved() {
            return (ApplyOutcome::Solved, nombre);
        }
        // Try naked singles first (cheapest).
        match find_naked_single(board) {
            Some(outcome) => {
                nombre = nombre.saturating_add(1);
                if outcome == ApplyOutcome::Contradiction {
                    return (ApplyOutcome::Contradiction, nombre);
                }
                if outcome == ApplyOutcome::Solved {
                    return (ApplyOutcome::Solved, nombre);
                }
                continue;
            }
            None => {}
        }
        // Then hidden singles.
        match find_hidden_single(board) {
            Some(outcome) => {
                nombre = nombre.saturating_add(1);
                if outcome == ApplyOutcome::Contradiction {
                    return (ApplyOutcome::Contradiction, nombre);
                }
                if outcome == ApplyOutcome::Solved {
                    return (ApplyOutcome::Solved, nombre);
                }
                continue;
            }
            None => {}
        }
        // Then box/line, if enabled.
        if level == TbLevel::L1 {
            match find_box_line(board) {
                Some(outcome) => {
                    nombre = nombre.saturating_add(1);
                    if outcome == ApplyOutcome::Contradiction {
                        return (ApplyOutcome::Contradiction, nombre);
                    }
                    if outcome == ApplyOutcome::Solved {
                        return (ApplyOutcome::Solved, nombre);
                    }
                    continue;
                }
                None => {}
            }
        }
        // Quiescence.
        return if board.is_solved() {
            (ApplyOutcome::Solved, nombre)
        } else {
            (ApplyOutcome::Continue, nombre)
        };
    }
}

/// Find ONE naked single and apply it. Returns Some(outcome) if a single
/// was applied, None if no naked single exists.
fn find_naked_single(board: &mut Board) -> Option<ApplyOutcome> {
    for cidx in 0..81u8 {
        let cell = &board.cells[cidx as usize];
        if cell.assigned.is_some() {
            continue;
        }
        if cell.cand_mask.count_ones() == 1 {
            // Found it — digit index = trailing_zeros of the singleton mask.
            let d = cell.cand_mask.trailing_zeros() as u8;
            return Some(board.assign(cidx, d));
        }
    }
    None
}

/// Find ONE hidden single (region:digit with only one candidate cell) and apply.
fn find_hidden_single(board: &mut Board) -> Option<ApplyOutcome> {
    let regs = regions();
    for reg_idx in 0..27usize {
        for digit in 0..9u8 {
            if board.digit_in_region[reg_idx][digit as usize] {
                continue;
            }
            if board.n_cand_in_region[reg_idx][digit as usize] == 1 {
                // Locate the single cell via mask test.
                let dbit = 1u16 << digit;
                let mut target: Option<u8> = None;
                for &cell_id in regs[reg_idx].cells.iter() {
                    let c = &board.cells[cell_id as usize];
                    if c.assigned.is_none() && (c.cand_mask & dbit) != 0 {
                        target = Some(cell_id);
                        break;
                    }
                }
                if let Some(t) = target {
                    return Some(board.assign(t, digit));
                }
                // FIXME(shc-port-phase1): decompile ambiguity — counter desync;
                // verify against /tmp/shc_decompiled/SHC/SHC/Jeu.java:178
            }
        }
    }
    None
}

/// Find ONE box/line (pointing/claiming) elimination and apply it.
fn find_box_line(board: &mut Board) -> Option<ApplyOutcome> {
    let regs = regions();
    for reg_idx in 0..27usize {
        for digit in 0..9u8 {
            if board.digit_in_region[reg_idx][digit as usize] {
                continue;
            }
            let count = board.n_cand_in_region[reg_idx][digit as usize];
            if count != 2 && count != 3 {
                continue;
            }
            // Collect candidate cells in this region for `digit`.
            let dbit = 1u16 << digit;
            let mut cand_cells: [u8; 9] = [0; 9];
            let mut ncc = 0;
            for &cell_id in regs[reg_idx].cells.iter() {
                let c = &board.cells[cell_id as usize];
                if c.assigned.is_none() && (c.cand_mask & dbit) != 0 {
                    cand_cells[ncc] = cell_id;
                    ncc += 1;
                }
            }
            if ncc == 0 {
                continue;
            }
            // Decide the "other" region we might intersect.
            // Java logic: if source is row/col, look for a containing box;
            // if source is box, look for a containing row, else containing col.
            let other_region: Option<usize> = if reg_idx < 18 {
                // row or col -> box
                same_region(board, &cand_cells[..ncc], 2)
            } else {
                // box -> row, else col
                same_region(board, &cand_cells[..ncc], 0)
                    .or_else(|| same_region(board, &cand_cells[..ncc], 1))
            };
            let Some(other) = other_region else { continue };
            // If the "other" region's total candidate count for `digit`
            // equals our count, there is nothing extra to eliminate (the
            // intersection IS the other region's candidate set).
            if board.n_cand_in_region[other][digit as usize] as usize == ncc {
                continue;
            }
            // Eliminate `digit` from every cell of `other` that is NOT in cand_cells.
            for &peer in regs[other].cells.iter() {
                let in_source = cand_cells[..ncc].iter().any(|&c| c == peer);
                if in_source {
                    continue;
                }
                let pc = &board.cells[peer as usize];
                if pc.assigned.is_some() {
                    continue;
                }
                if (pc.cand_mask & dbit) == 0 {
                    continue;
                }
                match board.eliminate(peer, digit) {
                    ApplyOutcome::Contradiction => return Some(ApplyOutcome::Contradiction),
                    _ => {}
                }
            }
            return Some(if board.is_solved() {
                ApplyOutcome::Solved
            } else {
                ApplyOutcome::Continue
            });
        }
    }
    None
}

/// If every cell in `cells` shares the same region at `slot` (0=row,1=col,2=box),
/// return that region's absolute index; otherwise None.
fn same_region(board: &Board, cells: &[u8], slot: usize) -> Option<usize> {
    if cells.is_empty() {
        return None;
    }
    let r0 = board.cells[cells[0] as usize].regions[slot];
    for &c in &cells[1..] {
        if board.cells[c as usize].regions[slot] != r0 {
            return None;
        }
    }
    Some(r0 as usize)
}
