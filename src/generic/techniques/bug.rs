//! # Bug (Bivalue Universal Grave +1)
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
//! BUG+1: all unsolved cells bivalue except one tri-cell; the tri-cell must
//! take the digit appearing 3 times in one of its units (to avoid the deadly
//! BUG pattern). (Hodoku BUG glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct Bug;

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for Bug {
    fn id(&self) -> TechniqueId { TechniqueId::Bug }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "bug" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let mut tri_cell: Option<usize> = None;
        for cell in 0..nn {
            if grid.solved[cell] != 0 { continue; }
            let pc = grid.candidates[cell].count_ones();
            if pc == 2 { continue; }
            if pc == 3 {
                if tri_cell.is_some() { return None; }
                tri_cell = Some(cell);
            } else {
                return None;
            }
        }
        let tri = tri_cell?;
        let mask = grid.candidates[tri];
        let table = grid.table().clone();
        let cell_units = table.cell_units[tri];
        let mut bits = mask;
        while bits != 0 {
            let bb = bits & bits.wrapping_neg();
            bits ^= bb;
            let digit = (bb.trailing_zeros() as u8) + 1;
            for &unit_id in cell_units.iter() {
                let unit = &table.units[unit_id as usize];
                let mut cnt = 0u32;
                for &c in unit {
                    let c = c as usize;
                    if grid.solved[c] != 0 { continue; }
                    if grid.candidates[c] & bb != 0 { cnt += 1; }
                }
                if cnt >= 3 {
                    if grid.assign(tri, digit).is_err() {
                        let mut prog = TechniqueProgress::default();
                        prog.contradiction = true;
                        return Some(prog);
                    }
                    let mut prog = TechniqueProgress::default();
                    prog.placements.push((tri, digit));
                    return Some(prog);
                }
            }
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for Bug
{
    fn base_rating(&self) -> f64 { 5.6 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn bug_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = Bug;
        assert!(<Bug as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn bug_no_fire_solved_9x9() {
        let solved = "534678912672195348198342567859761423426853791713924856961537284287419635345286179";
        let mut g: Grid<9, 3, 3> = Grid::from_str(solved).expect("parse");
        let t = Bug;
        assert!(<Bug as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn bug_smoke_6x6() {
        let g: Grid<6, 2, 3> = Grid::empty();
        let mut g2 = g.clone();
        let t = Bug;
        // No fire on empty (every cell is N-valent, not 2- or 3-).
        assert!(<Bug as Technique<6,2,3>>::apply(&t, &mut g2).is_none());
    }

    #[test]
    fn bug_smoke_12x12() {
        let g: Grid<12, 3, 4> = Grid::empty();
        let mut g2 = g.clone();
        let t = Bug;
        assert!(<Bug as Technique<12,3,4>>::apply(&t, &mut g2).is_none());
    }

    #[test]
    fn bug_smoke_16x16() {
        let g: Grid<16, 4, 4> = Grid::empty();
        let mut g2 = g.clone();
        let t = Bug;
        assert!(<Bug as Technique<16,4,4>>::apply(&t, &mut g2).is_none());
    }
}
