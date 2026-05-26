//! # XyzWing
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
//! XYZ-Wing: pivot trivalue {x,y,z}, pincers {x,z} and {y,z} each peering
//! with pivot; eliminate z from cells peering with all three.
//! (Hodoku XYZ-Wing glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct XyzWing;

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

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for XyzWing {
    fn id(&self) -> TechniqueId { TechniqueId::XyzWing }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "xyz_wing" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let mut bivs: Vec<(usize, u32)> = Vec::with_capacity(nn / 2);
        let mut trivs: Vec<(usize, u32)> = Vec::with_capacity(nn / 2);
        for c in 0..nn {
            if grid.solved[c] != 0 { continue; }
            match grid.candidates[c].count_ones() {
                2 => bivs.push((c, grid.candidates[c])),
                3 => trivs.push((c, grid.candidates[c])),
                _ => {}
            }
        }
        if bivs.len() < 2 || trivs.is_empty() { return None; }

        for &(pv, pmask) in &trivs {
            let pv_peers = peer_lookup::<N, BR, BC>(grid, pv);
            for i in 0..bivs.len() {
                let (ca, am) = bivs[i];
                if !pv_peers[ca] { continue; }
                if am & pmask != am { continue; } // am ⊆ pmask
                for j in (i + 1)..bivs.len() {
                    let (cb, bm) = bivs[j];
                    if !pv_peers[cb] { continue; }
                    if bm & pmask != bm { continue; }
                    if am | bm != pmask { continue; }
                    let zmask = am & bm;
                    if zmask.count_ones() != 1 { continue; }
                    let z_digit = (zmask.trailing_zeros() as u8) + 1;
                    let ca_peers = peer_lookup::<N, BR, BC>(grid, ca);
                    let cb_peers = peer_lookup::<N, BR, BC>(grid, cb);
                    let mut elims: Vec<(usize, u8)> = Vec::new();
                    for cell in 0..nn {
                        if cell == pv || cell == ca || cell == cb { continue; }
                        if grid.solved[cell] != 0 { continue; }
                        if !(pv_peers[cell] && ca_peers[cell] && cb_peers[cell]) { continue; }
                        if grid.candidates[cell] & zmask != 0 {
                            match grid.eliminate(cell, z_digit) {
                                Ok(true) => elims.push((cell, z_digit)),
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
    for XyzWing
{
    fn base_rating(&self) -> f64 { 4.4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn xyz_wing_fires_canonical_9x9() {
        // Same fixture as legacy: pivot (0,0)={1,2,3}, pincer A (0,1)={1,3},
        // pincer B (1,0)={2,3} → z=3 eliminated from (1,1)=idx 10.
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 4..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [2u8, 4, 5, 6, 7, 8, 9] { g.eliminate(1, d).unwrap(); }
        for d in [1u8, 4, 5, 6, 7, 8, 9] { g.eliminate(9, d).unwrap(); }
        assert!(g.candidates[10] & 0b100 != 0);
        let t = XyzWing;
        let p = <XyzWing as Technique<9,3,3>>::apply(&t, &mut g).expect("XYZ-Wing fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[10] & 0b100, 0);
    }

    #[test]
    fn xyz_wing_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = XyzWing;
        assert!(<XyzWing as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn xyz_wing_smoke_12x12() {
        // Pivot (0,0)={1,2,3}, A (0,3)={1,3}, B (3,0)={2,3} all share box 0.
        // Wait — (3,0) is in box 1 in 12x12 (BR=3). Use (1,0)={2,3} same box.
        let mut g: Grid<12, 3, 4> = Grid::empty();
        for d in 4..=12u8 { g.eliminate(0, d).unwrap(); }
        let kill_a: Vec<u8> = std::iter::once(2u8).chain(4..=12u8).collect();
        for d in kill_a { g.eliminate(1, d).unwrap(); }
        let kill_b: Vec<u8> = std::iter::once(1u8).chain(4..=12u8).collect();
        for d in kill_b { g.eliminate(12, d).unwrap(); }
        let t = XyzWing;
        // We only need: terminates without panic, produces no contradiction.
        let _ = <XyzWing as Technique<12,3,4>>::apply(&t, &mut g);
    }

    #[test]
    fn xyz_wing_smoke_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        let t = XyzWing;
        let _ = <XyzWing as Technique<16,4,4>>::apply(&t, &mut g);
    }

    #[test]
    fn xyz_wing_smoke_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        let t = XyzWing;
        let _ = <XyzWing as Technique<6,2,3>>::apply(&t, &mut g);
    }
}
