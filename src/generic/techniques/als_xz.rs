//! # AlsXz (Almost Locked Sets — XZ rule)
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
//! ALS-XZ: two Almost Locked Sets sharing a restricted-common digit X and a
//! second common digit Z; Z can be eliminated from cells outside both ALSes
//! that see all Z-cells in both. (Hodoku ALS-XZ glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct AlsXz;

const MAX_ALS_SIZE: usize = 4;

#[derive(Clone)]
struct Als {
    cells: Vec<u16>,
    mask: u32,
}

fn next_combination(idx: &mut [usize], pool: usize) -> bool {
    let n = idx.len();
    let mut i = n;
    while i > 0 {
        i -= 1;
        if idx[i] < pool - (n - i) {
            idx[i] += 1;
            for j in (i + 1)..n { idx[j] = idx[j - 1] + 1; }
            return true;
        }
    }
    false
}

fn als_key(a: &Als) -> u128 {
    // Mask in low bits, then up to 4 cells * 9 bits.
    let mut k: u128 = a.mask as u128;
    for &c in &a.cells {
        k = (k << 9) | (c as u128);
    }
    k
}

fn enumerate_alses<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Vec<Als> {
    let mut out: Vec<Als> = Vec::new();
    let mut seen: std::collections::HashSet<u128> = std::collections::HashSet::new();
    let table = grid.table().clone();
    let n_units = 3 * N;

    for u_idx in 0..n_units {
        let unit = &table.units[u_idx];
        let mut cells: Vec<(u16, u32)> = Vec::with_capacity(N);
        for &c in unit {
            let cu = c as usize;
            if grid.solved[cu] == 0 {
                cells.push((c, grid.candidates[cu]));
            }
        }
        let n = cells.len();
        if n == 0 { continue; }

        // size 1: bivalue cells
        for i in 0..n {
            if cells[i].1.count_ones() == 2 {
                let als = Als { cells: vec![cells[i].0], mask: cells[i].1 };
                let key = als_key(&als);
                if seen.insert(key) { out.push(als); }
            }
        }
        // size 2..=MAX_ALS_SIZE
        let max_size = MAX_ALS_SIZE.min(n);
        let mut idx = vec![0usize; MAX_ALS_SIZE];
        for size in 2..=max_size {
            for k in 0..size { idx[k] = k; }
            loop {
                let mut mask: u32 = 0;
                for k in 0..size { mask |= cells[idx[k]].1; }
                if (mask.count_ones() as usize) == size + 1 {
                    let mut cs: Vec<u16> = (0..size).map(|k| cells[idx[k]].0).collect();
                    cs.sort_unstable();
                    let als = Als { cells: cs, mask };
                    let key = als_key(&als);
                    if seen.insert(key) { out.push(als); }
                }
                if !next_combination(&mut idx[..size], n) { break; }
            }
        }
    }
    out
}

fn shares_unit<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    a: usize,
    b: usize,
) -> bool {
    if a == b { return false; }
    let table = grid.table();
    table.cells[a].peers.iter().any(|&p| p as usize == b)
}

fn restricted_common<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    a: &Als,
    b: &Als,
    d_bit: u8,
) -> bool {
    let bit = 1u32 << d_bit;
    let a_cells: Vec<usize> = a.cells.iter()
        .map(|&c| c as usize)
        .filter(|&c| grid.candidates[c] & bit != 0)
        .collect();
    if a_cells.is_empty() { return false; }
    let b_cells: Vec<usize> = b.cells.iter()
        .map(|&c| c as usize)
        .filter(|&c| grid.candidates[c] & bit != 0)
        .collect();
    if b_cells.is_empty() { return false; }
    for &ac in &a_cells {
        for &bc in &b_cells {
            if ac == bc { return false; }
            if !shares_unit::<N, BR, BC>(grid, ac, bc) { return false; }
        }
    }
    true
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for AlsXz {
    fn id(&self) -> TechniqueId { TechniqueId::AlsXz }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "als_xz" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let alses = enumerate_alses::<N, BR, BC>(grid);
        let n = alses.len();
        let nn = N * N;
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &alses[i];
                let b = &alses[j];
                // overlap check
                let mut overlap = false;
                'ov: for &ac in &a.cells {
                    for &bc in &b.cells {
                        if ac == bc { overlap = true; break 'ov; }
                    }
                }
                if overlap { continue; }
                let common = a.mask & b.mask;
                if common.count_ones() < 2 { continue; }
                let mut bits = common;
                while bits != 0 {
                    let bb = bits & bits.wrapping_neg();
                    bits ^= bb;
                    let d_bit = bb.trailing_zeros() as u8;
                    if !restricted_common::<N, BR, BC>(grid, a, b, d_bit) { continue; }
                    let mut zbits = common & !bb;
                    while zbits != 0 {
                        let zb = zbits & zbits.wrapping_neg();
                        zbits ^= zb;
                        let z_bit = zb.trailing_zeros() as u8;
                        let z_digit = z_bit + 1;
                        // Z-cells in a ∪ b
                        let mut z_cells: Vec<usize> = Vec::new();
                        for &c in &a.cells {
                            let cu = c as usize;
                            if grid.candidates[cu] & zb != 0 { z_cells.push(cu); }
                        }
                        for &c in &b.cells {
                            let cu = c as usize;
                            if grid.candidates[cu] & zb != 0 { z_cells.push(cu); }
                        }
                        if z_cells.is_empty() { continue; }
                        // scan all cells outside (a ∪ b)
                        let mut elims: Vec<(usize, u8)> = Vec::new();
                        for cell in 0..nn {
                            if grid.solved[cell] != 0 { continue; }
                            if grid.candidates[cell] & zb == 0 { continue; }
                            if a.cells.iter().any(|&c| c as usize == cell) { continue; }
                            if b.cells.iter().any(|&c| c as usize == cell) { continue; }
                            let mut sees_all = true;
                            for &zc in &z_cells {
                                if !shares_unit::<N, BR, BC>(grid, cell, zc) {
                                    sees_all = false; break;
                                }
                            }
                            if sees_all {
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
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for AlsXz
{
    fn base_rating(&self) -> f64 { 7.5 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn als_xz_no_fire_on_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = AlsXz;
        assert!(<AlsXz as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    /// XY-Wing as an ALS-XZ over three size-1 ALSes (bivalues):
    ///   pivot (0,0)={1,2}; pincers (0,3)={1,3}, (1,0)={2,3}.
    /// One pair of bivalues (e.g. (0,0) and (0,3)) shares row 0; restricted-
    /// common digit = 1; Z = 2 or 3. ALS-XZ finds at least one elimination
    /// (the same as the XY-Wing): 3 from cell (1,3).
    #[test]
    fn als_xz_xy_wing_pattern_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [2u8, 4, 5, 6, 7, 8, 9] { g.eliminate(3, d).unwrap(); }
        for d in [1u8, 4, 5, 6, 7, 8, 9] { g.eliminate(9, d).unwrap(); }
        let t = AlsXz;
        let p = <AlsXz as Technique<9,3,3>>::apply(&t, &mut g);
        // ALS-XZ must fire (XY-Wing degenerate case) and eliminate something.
        // We don't pin a specific (cell,digit) here because the ALS-XZ search
        // visits ALSes in a different order than XY-Wing and may fire a
        // different (but equally valid) elimination first; what we DO assert
        // is no contradiction was raised.
        assert!(p.is_some(), "ALS-XZ expected to fire on XY-Wing grid");
        let prog = p.unwrap();
        assert!(!prog.eliminations.is_empty());
        assert!(!prog.contradiction);
    }

    /// 16×16 smoke: same XY-Wing-style construction.
    #[test]
    fn als_xz_xy_wing_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        // (0,0) → {1,2}
        for d in 3..=16u8 { g.eliminate(0, d).unwrap(); }
        // (0,3) → {1,3}
        let kill_3: Vec<u8> = std::iter::once(2u8).chain(4..=16u8).collect();
        for d in kill_3 { g.eliminate(3, d).unwrap(); }
        // (1,0) idx=16 → {2,3}
        let kill_10: Vec<u8> = std::iter::once(1u8).chain(4..=16u8).collect();
        for d in kill_10 { g.eliminate(16, d).unwrap(); }
        let t = AlsXz;
        let p = <AlsXz as Technique<16,4,4>>::apply(&t, &mut g);
        assert!(p.is_some(), "ALS-XZ XY-Wing should fire on 16×16");
        assert!(!p.unwrap().eliminations.is_empty());
    }

    /// Idempotent: after a fire, immediately re-running may fire again on a
    /// different ALS pair, but never produces a contradiction on this grid.
    #[test]
    fn als_xz_no_contradiction_on_constructive_smoke() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [1u8, 4, 5, 6, 7, 8, 9] { g.eliminate(1, d).unwrap(); }
        let t = AlsXz;
        if let Some(p) = <AlsXz as Technique<9,3,3>>::apply(&t, &mut g) {
            assert!(!p.contradiction);
        }
    }
}
