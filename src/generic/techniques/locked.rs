//! # LockedCandidates (LockedPointing + LockedClaiming)
//!
//! ## Inputs
//! Reads from `Grid<N,BR,BC>`: candidate bitmasks per cell, placed digits.
//!
//! ## Mutates
//! May eliminate candidates and/or place digits in `grid` (in place). Does
//! not touch any state outside the passed-in `Grid`.
//!
//! ## Returns
//! `Some(TechniqueProgress)` if at least one elimination/placement happened
//! on this call; `None` if the technique did not fire. Sets
//! `progress.contradiction = true` if elimination drove a cell to 0
//! candidates.
//!
//! ## Performance budget
//! Target: < 20 µs/grid on 9×9 on commodity x86_64.
//!
//! ## Algorithm reference
//! Pointing: candidates in a box confined to one line → eliminate from rest
//! of that line. Claiming: candidates on a line confined to one box →
//! eliminate from rest of that box. (Hodoku locked-candidates glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct LockedPointing;

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for LockedPointing {
    fn id(&self) -> TechniqueId { TechniqueId::LockedPointing }
    fn tier(&self) -> Tier { Tier::T2 }
    fn name(&self) -> &'static str { "locked_pointing" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let table = grid.table().clone(); // Arc<PeerTable>
        let mut prog = TechniqueProgress::default();
        // Boxes are units 2N..3N.
        for b in 0..N {
            // Snapshot the box-unit cells (Vec<u16>); we'll mutate grid below.
            let box_cells: Vec<usize> = table.units[2 * N + b].iter().map(|&x| x as usize).collect();
            for d in 1u8..=(N as u8) {
                let bit = 1u32 << (d - 1);
                // For each unsolved cell in the box where d is candidate,
                // collect its row and column. If the box already contains a
                // solved d, skip (digit is fixed within this box).
                let mut rows_seen: u32 = 0; // bit r set = row r holds a candidate
                let mut cols_seen: u32 = 0; // bit c set = col c holds a candidate
                let mut already_placed = false;
                for &c in &box_cells {
                    if grid.solved[c] == d {
                        already_placed = true;
                        break;
                    }
                    if grid.solved[c] == 0 && (grid.candidates[c] & bit) != 0 {
                        let r = table.cells[c].row as u32;
                        let cc = table.cells[c].col as u32;
                        rows_seen |= 1u32 << r;
                        cols_seen |= 1u32 << cc;
                    }
                }
                if already_placed { continue; }
                if rows_seen == 0 { continue; }
                // Confined to a single row?
                if rows_seen.count_ones() == 1 {
                    let r = rows_seen.trailing_zeros() as usize;
                    // Eliminate d from cells in row r that are outside box b.
                    let row_unit_cells: Vec<usize> = table.units[r].iter().map(|&x| x as usize).collect();
                    for c in row_unit_cells {
                        if (table.cells[c].box_id as usize) == b { continue; }
                        if grid.solved[c] != 0 { continue; }
                        if (grid.candidates[c] & bit) == 0 { continue; }
                        match grid.eliminate(c, d) {
                            Ok(true) => prog.eliminations.push((c, d)),
                            Ok(false) => {}
                            Err(_) => { prog.contradiction = true; return Some(prog); }
                        }
                    }
                }
                // Confined to a single column?
                if cols_seen.count_ones() == 1 {
                    let cc = cols_seen.trailing_zeros() as usize;
                    let col_unit_cells: Vec<usize> = table.units[N + cc].iter().map(|&x| x as usize).collect();
                    for c in col_unit_cells {
                        if (table.cells[c].box_id as usize) == b { continue; }
                        if grid.solved[c] != 0 { continue; }
                        if (grid.candidates[c] & bit) == 0 { continue; }
                        match grid.eliminate(c, d) {
                            Ok(true) => prog.eliminations.push((c, d)),
                            Ok(false) => {}
                            Err(_) => { prog.contradiction = true; return Some(prog); }
                        }
                    }
                }
            }
        }
        if prog.fired() || prog.contradiction { Some(prog) } else { None }
    }
}

pub struct LockedClaiming;

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for LockedClaiming {
    fn id(&self) -> TechniqueId { TechniqueId::LockedClaiming }
    fn tier(&self) -> Tier { Tier::T2 }
    fn name(&self) -> &'static str { "locked_claiming" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let table = grid.table().clone();
        let mut prog = TechniqueProgress::default();
        // Lines: rows (units 0..N) and columns (units N..2N).
        for line_unit in 0..(2 * N) {
            let line_cells: Vec<usize> = table.units[line_unit].iter().map(|&x| x as usize).collect();
            for d in 1u8..=(N as u8) {
                let bit = 1u32 << (d - 1);
                let mut boxes_seen: u32 = 0;
                let mut already_placed = false;
                for &c in &line_cells {
                    if grid.solved[c] == d {
                        already_placed = true;
                        break;
                    }
                    if grid.solved[c] == 0 && (grid.candidates[c] & bit) != 0 {
                        let bx = table.cells[c].box_id as u32;
                        boxes_seen |= 1u32 << bx;
                    }
                }
                if already_placed || boxes_seen == 0 { continue; }
                if boxes_seen.count_ones() == 1 {
                    let bx = boxes_seen.trailing_zeros() as usize;
                    let box_cells: Vec<usize> = table.units[2 * N + bx].iter().map(|&x| x as usize).collect();
                    // "On line" check: cell shares the same row/col-unit as
                    // line_unit. Easiest: cross-check via cell_units.
                    for c in box_cells {
                        let cu = &table.cell_units[c];
                        let on_line = (cu[0] as usize) == line_unit || (cu[1] as usize) == line_unit;
                        if on_line { continue; }
                        if grid.solved[c] != 0 { continue; }
                        if (grid.candidates[c] & bit) == 0 { continue; }
                        match grid.eliminate(c, d) {
                            Ok(true) => prog.eliminations.push((c, d)),
                            Ok(false) => {}
                            Err(_) => { prog.contradiction = true; return Some(prog); }
                        }
                    }
                }
            }
        }
        if prog.fired() || prog.contradiction { Some(prog) } else { None }
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for LockedPointing
{
    fn base_rating(&self) -> f64 { 2.6 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for LockedClaiming
{
    fn base_rating(&self) -> f64 { 2.8 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    // ---- 9×9 parity with legacy reference ----
    #[test]
    fn pointing_fires_9x9_row_confinement() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Box 0 cells: 0,1,2,9,10,11,18,19,20. Eliminate 5 from rows 1,2 of box.
        for c in [9usize, 10, 11, 18, 19, 20] {
            g.eliminate(c, 5).unwrap();
        }
        let t = LockedPointing;
        let p = <LockedPointing as Technique<9,3,3>>::apply(&t, &mut g)
            .expect("should fire");
        assert!(p.fired() && !p.contradiction);
        // 5 must be eliminated from row 0 cells 3..9 (outside box 0).
        for c in 3..9usize {
            assert_eq!(g.candidates[c] & (1 << 4), 0, "cell {} still has 5", c);
        }
        // Idempotent: second call returns None.
        let q = <LockedPointing as Technique<9,3,3>>::apply(&t, &mut g);
        assert!(q.is_none(), "expected idempotent None on second apply");
    }

    #[test]
    fn claiming_fires_9x9_box_confinement() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Restrict 7 in row 0 to cells 0,1,2 (box 0) by removing from 3..9.
        for c in 3..9usize {
            g.eliminate(c, 7).unwrap();
        }
        let t = LockedClaiming;
        let p = <LockedClaiming as Technique<9,3,3>>::apply(&t, &mut g)
            .expect("should fire");
        assert!(p.fired());
        // Cells in box 0 outside row 0: 9,10,11,18,19,20.
        for c in [9usize, 10, 11, 18, 19, 20] {
            assert_eq!(g.candidates[c] & (1 << 6), 0, "cell {} still has 7", c);
        }
        let q = <LockedClaiming as Technique<9,3,3>>::apply(&t, &mut g);
        assert!(q.is_none());
    }

    // ---- 6×6 (BR=2, BC=3): box has 2 rows × 3 cols ----
    #[test]
    fn pointing_fires_6x6() {
        // 6×6: cells per row = 6, box 0 spans rows 0..2 cols 0..3 →
        // cells 0,1,2, 6,7,8.
        let mut g: Grid<6, 2, 3> = Grid::empty();
        // Eliminate digit 4 from row 1 of box 0 (cells 6,7,8) so 4 is
        // confined to row 0 within box 0.
        for c in [6usize, 7, 8] {
            g.eliminate(c, 4).unwrap();
        }
        let t = LockedPointing;
        let p = <LockedPointing as Technique<6,2,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Row 0 cells outside box 0: 3,4,5. They should lose 4.
        for c in [3usize, 4, 5] {
            assert_eq!(g.candidates[c] & (1 << 3), 0, "cell {} still has 4", c);
        }
        let q = <LockedPointing as Technique<6,2,3>>::apply(&t, &mut g);
        assert!(q.is_none());
    }

    #[test]
    fn claiming_fires_6x6() {
        // Restrict digit 5 in row 0 to cells 0..3 (box 0 spans cols 0..3).
        let mut g: Grid<6, 2, 3> = Grid::empty();
        for c in [3usize, 4, 5] {
            g.eliminate(c, 5).unwrap();
        }
        // Row 0 still has 5 only in box 0 (cells 0,1,2).
        let t = LockedClaiming;
        let p = <LockedClaiming as Technique<6,2,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Box 0 cells outside row 0: 6,7,8. They should lose 5.
        for c in [6usize, 7, 8] {
            assert_eq!(g.candidates[c] & (1 << 4), 0, "cell {} still has 5", c);
        }
    }

    // ---- 12×12 (BR=3, BC=4): box has 3 rows × 4 cols ----
    #[test]
    fn pointing_fires_12x12() {
        // box 0 cells: rows 0..3, cols 0..4. row 0 has cells 0,1,2,3 in box 0.
        // Eliminate digit 7 from box 0 cells outside row 0:
        //   rows 1..3, cols 0..4 → cells 12..16, 24..28, 36..40.
        let mut g: Grid<12, 3, 4> = Grid::empty();
        let to_drop: Vec<usize> = (1..3).flat_map(|r: usize| (0..4).map(move |c| r * 12 + c)).collect();
        for c in to_drop {
            g.eliminate(c, 7).unwrap();
        }
        let t = LockedPointing;
        let p = <LockedPointing as Technique<12,3,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Row 0 cells outside box 0: cols 4..12.
        for c in 4..12usize {
            assert_eq!(g.candidates[c] & (1 << 6), 0, "cell {} still has 7", c);
        }
        let q = <LockedPointing as Technique<12,3,4>>::apply(&t, &mut g);
        assert!(q.is_none());
    }

    // ---- 16×16 (BR=4, BC=4) ----
    #[test]
    fn pointing_fires_16x16() {
        // Box 0: rows 0..4, cols 0..4. Drop digit 9 from box 0 rows 1..4.
        let mut g: Grid<16, 4, 4> = Grid::empty();
        for r in 1..4usize {
            for c in 0..4usize {
                g.eliminate(r * 16 + c, 9).unwrap();
            }
        }
        let t = LockedPointing;
        let p = <LockedPointing as Technique<16,4,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Row 0 cells outside box 0: cols 4..16.
        for c in 4..16usize {
            assert_eq!(g.candidates[c] & (1 << 8), 0, "cell {} still has 9", c);
        }
    }

    // Column-based pointing on 9×9 (covers cols_seen path).
    #[test]
    fn pointing_fires_9x9_col_confinement() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Box 0 cells; eliminate digit 3 from cols 1,2 of box 0 so col 0 is the only one.
        // Cells in box 0 cols 1,2: 1,2,10,11,19,20.
        for c in [1usize, 2, 10, 11, 19, 20] {
            g.eliminate(c, 3).unwrap();
        }
        let t = LockedPointing;
        let p = <LockedPointing as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Col 0 cells outside box 0: rows 3..9 → idx 27, 36, 45, 54, 63, 72.
        for c in [27usize, 36, 45, 54, 63, 72] {
            assert_eq!(g.candidates[c] & (1 << 2), 0, "cell {} still has 3", c);
        }
    }

    // Claiming on 12×12: row 0 confined to box 0 (cols 0..4).
    #[test]
    fn claiming_fires_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        for c in 4..12usize { g.eliminate(c, 5).unwrap(); }
        let t = LockedClaiming;
        let p = <LockedClaiming as Technique<12,3,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Box 0 cells outside row 0: rows 1,2 cols 0..4.
        for r in 1..3usize {
            for c in 0..4usize {
                assert_eq!(g.candidates[r * 12 + c] & (1 << 4), 0,
                    "cell r{}c{} still has 5", r, c);
            }
        }
    }

    // Claiming on 16×16.
    #[test]
    fn claiming_fires_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        for c in 4..16usize { g.eliminate(c, 6).unwrap(); }
        let t = LockedClaiming;
        let p = <LockedClaiming as Technique<16,4,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Box 0 cells outside row 0: rows 1..4 cols 0..4.
        for r in 1..4usize {
            for c in 0..4usize {
                assert_eq!(g.candidates[r * 16 + c] & (1 << 5), 0,
                    "cell r{}c{} still has 6", r, c);
            }
        }
    }

    #[test]
    fn no_fire_on_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let tp = LockedPointing;
        let tc = LockedClaiming;
        assert!(<LockedPointing as Technique<9,3,3>>::apply(&tp, &mut g).is_none());
        assert!(<LockedClaiming as Technique<9,3,3>>::apply(&tc, &mut g).is_none());
    }
}
