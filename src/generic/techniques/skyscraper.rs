//! # Skyscraper
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
//! Target: < 100 µs/grid on 9×9 on commodity x86_64.
//!
//! ## Algorithm reference
//! Skyscraper: two rows (or cols) each with exactly 2 cells for digit d
//! sharing one col (row); the two free ends' common peers lose d.
//! (Hodoku skyscraper glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct Skyscraper;

fn peer_lookup<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    cell: usize,
) -> Vec<bool> {
    let nn = N * N;
    let mut v = vec![false; nn];
    let table = grid.table();
    for &p in &table.cells[cell].peers {
        v[p as usize] = true;
    }
    v
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for Skyscraper {
    fn id(&self) -> TechniqueId { TechniqueId::Skyscraper }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "skyscraper" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let table = grid.table().clone();
        for d_bit in 0..(N as u8) {
            let bit = 1u32 << d_bit;
            let digit = d_bit + 1;

            // Per-row col list and per-col row list.
            let mut row_cols: Vec<Vec<u16>> = vec![Vec::new(); N];
            let mut col_rows: Vec<Vec<u16>> = vec![Vec::new(); N];
            let mut placed_row = vec![false; N];
            let mut placed_col = vec![false; N];
            for cell in 0..nn {
                let r = table.cells[cell].row as usize;
                let c = table.cells[cell].col as usize;
                if grid.solved[cell] != 0 {
                    if grid.solved[cell] == digit {
                        placed_row[r] = true;
                        placed_col[c] = true;
                    }
                    continue;
                }
                if grid.candidates[cell] & bit != 0 {
                    row_cols[r].push(c as u16);
                    col_rows[c].push(r as u16);
                }
            }

            // Row-paired skyscraper.
            for r1 in 0..N {
                if placed_row[r1] || row_cols[r1].len() != 2 { continue; }
                for r2 in (r1 + 1)..N {
                    if placed_row[r2] || row_cols[r2].len() != 2 { continue; }
                    let a1 = row_cols[r1][0]; let a2 = row_cols[r1][1];
                    let b1 = row_cols[r2][0]; let b2 = row_cols[r2][1];
                    let (free1, free2) = if a1 == b1 { (a2, b2) }
                        else if a1 == b2 { (a2, b1) }
                        else if a2 == b1 { (a1, b2) }
                        else if a2 == b2 { (a1, b1) }
                        else { continue; };
                    if free1 == free2 { continue; }
                    let cell_x = r1 * N + free1 as usize;
                    let cell_y = r2 * N + free2 as usize;
                    let px = peer_lookup::<N, BR, BC>(grid, cell_x);
                    let py = peer_lookup::<N, BR, BC>(grid, cell_y);
                    let mut elims: Vec<(usize, u8)> = Vec::new();
                    for cell in 0..nn {
                        if cell == cell_x || cell == cell_y { continue; }
                        if !(px[cell] && py[cell]) { continue; }
                        if grid.solved[cell] != 0 { continue; }
                        if grid.candidates[cell] & bit != 0 {
                            match grid.eliminate(cell, digit) {
                                Ok(true) => elims.push((cell, digit)),
                                Ok(false) => {}
                                Err(_) => {
                                    let mut prog = TechniqueProgress::default();
                                    prog.eliminations = elims;
                                    prog.contradiction = true;
                                    return Some(prog);
                                }
                            }
                        }
                    }
                    if !elims.is_empty() {
                        let mut prog = TechniqueProgress::default();
                        prog.eliminations = elims;
                        return Some(prog);
                    }
                }
            }
            // Column-paired skyscraper.
            for c1 in 0..N {
                if placed_col[c1] || col_rows[c1].len() != 2 { continue; }
                for c2 in (c1 + 1)..N {
                    if placed_col[c2] || col_rows[c2].len() != 2 { continue; }
                    let a1 = col_rows[c1][0]; let a2 = col_rows[c1][1];
                    let b1 = col_rows[c2][0]; let b2 = col_rows[c2][1];
                    let (free1, free2) = if a1 == b1 { (a2, b2) }
                        else if a1 == b2 { (a2, b1) }
                        else if a2 == b1 { (a1, b2) }
                        else if a2 == b2 { (a1, b1) }
                        else { continue; };
                    if free1 == free2 { continue; }
                    let cell_x = free1 as usize * N + c1;
                    let cell_y = free2 as usize * N + c2;
                    let px = peer_lookup::<N, BR, BC>(grid, cell_x);
                    let py = peer_lookup::<N, BR, BC>(grid, cell_y);
                    let mut elims: Vec<(usize, u8)> = Vec::new();
                    for cell in 0..nn {
                        if cell == cell_x || cell == cell_y { continue; }
                        if !(px[cell] && py[cell]) { continue; }
                        if grid.solved[cell] != 0 { continue; }
                        if grid.candidates[cell] & bit != 0 {
                            match grid.eliminate(cell, digit) {
                                Ok(true) => elims.push((cell, digit)),
                                Ok(false) => {}
                                Err(_) => {
                                    let mut prog = TechniqueProgress::default();
                                    prog.eliminations = elims;
                                    prog.contradiction = true;
                                    return Some(prog);
                                }
                            }
                        }
                    }
                    if !elims.is_empty() {
                        let mut prog = TechniqueProgress::default();
                        prog.eliminations = elims;
                        return Some(prog);
                    }
                }
            }
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for Skyscraper
{
    fn base_rating(&self) -> f64 { 4.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn skyscraper_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = Skyscraper;
        assert!(<Skyscraper as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    /// Canonical skyscraper on digit 5 (mirrors legacy fixture):
    ///   Row 0: 5 only in cols 0,4. Row 1: 5 only in cols 0,5.
    ///   Free ends (0,4),(1,5) share box 1; cell (2,4) sees (0,4) by col 4
    ///   and (1,5) by box 1 → loses 5.
    #[test]
    fn skyscraper_fires_canonical_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 0..9u8 { if c != 0 && c != 4 { g.eliminate(0 * 9 + c as usize, 5).unwrap(); } }
        for c in 0..9u8 { if c != 0 && c != 5 { g.eliminate(1 * 9 + c as usize, 5).unwrap(); } }
        let bit5 = 1u32 << 4;
        assert!(g.candidates[2 * 9 + 4] & bit5 != 0);
        let t = Skyscraper;
        let p = <Skyscraper as Technique<9,3,3>>::apply(&t, &mut g).expect("skyscraper fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[2 * 9 + 4] & bit5, 0, "(2,4) should lose 5");
    }

    #[test]
    fn skyscraper_smoke_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        let t = Skyscraper;
        let _ = <Skyscraper as Technique<6,2,3>>::apply(&t, &mut g);
    }

    #[test]
    fn skyscraper_smoke_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        let t = Skyscraper;
        let _ = <Skyscraper as Technique<12,3,4>>::apply(&t, &mut g);
    }

    #[test]
    fn skyscraper_smoke_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        let t = Skyscraper;
        let _ = <Skyscraper as Technique<16,4,4>>::apply(&t, &mut g);
    }
}
