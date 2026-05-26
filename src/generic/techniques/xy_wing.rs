//! # XyWing
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
//! XY-Wing: pivot bivalue {x,y}, two pincers {x,z} and {y,z} each peering
//! with pivot; eliminate z from cells peering with both pincers.
//! (Hodoku XY-Wing glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct XyWing;

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

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for XyWing {
    fn id(&self) -> TechniqueId { TechniqueId::XyWing }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "xy_wing" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let mut bivs: Vec<(usize, u32)> = Vec::with_capacity(nn / 2);
        for c in 0..nn {
            if grid.solved[c] == 0 && grid.candidates[c].count_ones() == 2 {
                bivs.push((c, grid.candidates[c]));
            }
        }
        if bivs.len() < 3 { return None; }

        for &(pv, pmask) in &bivs {
            let pv_peers = peer_lookup::<N, BR, BC>(grid, pv);
            for i in 0..bivs.len() {
                let (ca, am) = bivs[i];
                if ca == pv { continue; }
                if !pv_peers[ca] { continue; }
                let shared_a = am & pmask;
                if shared_a.count_ones() != 1 { continue; }
                let z_a = am ^ shared_a;
                if z_a.count_ones() != 1 { continue; }
                for j in (i + 1)..bivs.len() {
                    let (cb, bm) = bivs[j];
                    if cb == pv { continue; }
                    if !pv_peers[cb] { continue; }
                    let shared_b = bm & pmask;
                    if shared_b.count_ones() != 1 { continue; }
                    if shared_b == shared_a { continue; }
                    let z_b = bm ^ shared_b;
                    if z_b != z_a { continue; }
                    // z is shared; eliminate from cells peering both ca, cb.
                    let z_digit = z_a.trailing_zeros() as u8 + 1;
                    let cb_peers = peer_lookup::<N, BR, BC>(grid, cb);
                    let table = grid.table().clone();
                    let mut elims: Vec<(usize, u8)> = Vec::new();
                    for &p in &table.cells[ca].peers {
                        let cell = p as usize;
                        if !cb_peers[cell] { continue; }
                        if cell == pv || cell == ca || cell == cb { continue; }
                        if grid.solved[cell] != 0 { continue; }
                        if grid.candidates[cell] & z_a != 0 {
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
    for XyWing
{
    fn base_rating(&self) -> f64 { 4.2 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn xy_wing_fires_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [2u8, 4, 5, 6, 7, 8, 9] { g.eliminate(3, d).unwrap(); }
        for d in [1u8, 4, 5, 6, 7, 8, 9] { g.eliminate(9, d).unwrap(); }
        assert!(g.candidates[12] & 0b100 != 0);
        let t = XyWing;
        let p = <XyWing as Technique<9,3,3>>::apply(&t, &mut g).expect("XY-Wing fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[12] & 0b100, 0);
    }

    #[test]
    fn xy_wing_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = XyWing;
        assert!(<XyWing as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn xy_wing_fires_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        // (0,0)={1,2}, (0,3)={1,3} (same row), (1,0)={2,3} (same col).
        // Common peer of (0,3) and (1,0): (1,3) idx=15. Should lose 3.
        for d in 3..=12u8 { g.eliminate(0, d).unwrap(); }
        let kill_3: Vec<u8> = std::iter::once(2u8).chain(4..=12u8).collect();
        for d in kill_3 { g.eliminate(3, d).unwrap(); }
        let kill_10: Vec<u8> = std::iter::once(1u8).chain(4..=12u8).collect();
        for d in kill_10 { g.eliminate(12, d).unwrap(); }
        assert!(g.candidates[15] & 0b100 != 0);
        let t = XyWing;
        let p = <XyWing as Technique<12,3,4>>::apply(&t, &mut g).expect("XY-Wing 12×12 fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[15] & 0b100, 0);
    }

    #[test]
    fn xy_wing_fires_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        for d in 3..=16u8 { g.eliminate(0, d).unwrap(); }
        let kill_3: Vec<u8> = std::iter::once(2u8).chain(4..=16u8).collect();
        for d in kill_3 { g.eliminate(3, d).unwrap(); }
        let kill_10: Vec<u8> = std::iter::once(1u8).chain(4..=16u8).collect();
        for d in kill_10 { g.eliminate(16, d).unwrap(); }
        // common peer (1,3) = idx 19
        assert!(g.candidates[19] & 0b100 != 0);
        let t = XyWing;
        let p = <XyWing as Technique<16,4,4>>::apply(&t, &mut g).expect("XY-Wing 16×16 fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[19] & 0b100, 0);
    }

    #[test]
    fn xy_wing_idempotent_after_fire() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [2u8, 4, 5, 6, 7, 8, 9] { g.eliminate(3, d).unwrap(); }
        for d in [1u8, 4, 5, 6, 7, 8, 9] { g.eliminate(9, d).unwrap(); }
        let t = XyWing;
        let _ = <XyWing as Technique<9,3,3>>::apply(&t, &mut g);
        assert!(<XyWing as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }
}
