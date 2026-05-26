//! # TwoStringKite
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
//! Two-String Kite: row and column each with exactly 2 cells for digit d,
//! one pair sharing a box (roof); common peers of the two free ends lose d.
//! (Hodoku two-string kite glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct TwoStringKite;

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

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for TwoStringKite {
    fn id(&self) -> TechniqueId { TechniqueId::TwoStringKite }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "two_string_kite" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let table = grid.table().clone();
        for d_bit in 0..(N as u8) {
            let bit = 1u32 << d_bit;
            let digit = d_bit + 1;

            let mut row_cells: Vec<Vec<u16>> = vec![Vec::new(); N];
            let mut col_cells: Vec<Vec<u16>> = vec![Vec::new(); N];
            let mut placed_row = vec![false; N];
            let mut placed_col = vec![false; N];
            for cell in 0..nn {
                let r = table.cells[cell].row as usize;
                let c = table.cells[cell].col as usize;
                if grid.solved[cell] != 0 {
                    if grid.solved[cell] == digit {
                        placed_row[r] = true; placed_col[c] = true;
                    }
                    continue;
                }
                if grid.candidates[cell] & bit != 0 {
                    row_cells[r].push(cell as u16);
                    col_cells[c].push(cell as u16);
                }
            }

            for r in 0..N {
                if placed_row[r] || row_cells[r].len() != 2 { continue; }
                for c in 0..N {
                    if placed_col[c] || col_cells[c].len() != 2 { continue; }
                    let rcs = [row_cells[r][0] as usize, row_cells[r][1] as usize];
                    let ccs = [col_cells[c][0] as usize, col_cells[c][1] as usize];
                    for ri in 0..2 {
                        for ci in 0..2 {
                            let roof_r = rcs[ri];
                            let roof_c = ccs[ci];
                            if roof_r == roof_c { continue; }
                            if table.cells[roof_r].box_id != table.cells[roof_c].box_id { continue; }
                            let free_r = rcs[1 - ri];
                            let free_c = ccs[1 - ci];
                            if free_r == free_c { continue; }
                            let pa = peer_lookup::<N, BR, BC>(grid, free_r);
                            let pb = peer_lookup::<N, BR, BC>(grid, free_c);
                            let mut elims: Vec<(usize, u8)> = Vec::new();
                            for cell in 0..nn {
                                if cell == roof_r || cell == roof_c
                                    || cell == free_r || cell == free_c { continue; }
                                if !(pa[cell] && pb[cell]) { continue; }
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
            }
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for TwoStringKite
{
    fn base_rating(&self) -> f64 { 4.1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn t2k_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = TwoStringKite;
        assert!(<TwoStringKite as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    /// Canonical T2K on digit 5 (mirrors legacy fixture):
    ///   Row 1: 5 in (1,3),(1,8). Col 5: 5 in (0,5),(8,5).
    ///   Roof (1,3)/(0,5) share box 1. Free ends (1,8),(8,5).
    ///   Target (8,8) sees (1,8) by col 8 and (8,5) by row 8 → loses 5.
    #[test]
    fn t2k_fires_canonical_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 0..9u8 { if c != 3 && c != 8 { g.eliminate(1 * 9 + c as usize, 5).unwrap(); } }
        for r in 0..9u8 { if r != 0 && r != 8 { g.eliminate(r as usize * 9 + 5, 5).unwrap(); } }
        let bit5 = 1u32 << 4;
        assert!(g.candidates[8 * 9 + 8] & bit5 != 0);
        let t = TwoStringKite;
        let p = <TwoStringKite as Technique<9,3,3>>::apply(&t, &mut g).expect("T2K fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[8 * 9 + 8] & bit5, 0, "(8,8) should lose 5");
    }

    #[test]
    fn t2k_smoke_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        let t = TwoStringKite;
        let _ = <TwoStringKite as Technique<6,2,3>>::apply(&t, &mut g);
    }

    #[test]
    fn t2k_smoke_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        let t = TwoStringKite;
        let _ = <TwoStringKite as Technique<12,3,4>>::apply(&t, &mut g);
    }

    #[test]
    fn t2k_smoke_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        let t = TwoStringKite;
        let _ = <TwoStringKite as Technique<16,4,4>>::apply(&t, &mut g);
    }
}
