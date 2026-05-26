//! # NakedSet (NakedPair / NakedTriple / NakedQuad)
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
//! For each unit, find K-subset of unsolved cells whose candidate-union has
//! exactly K bits; eliminate those K digits from all other cells in the unit.
//! (Standard naked-set pattern, Sudoku.com / Hodoku glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct NakedSet<const K: usize>;

pub type NakedPair = NakedSet<2>;
pub type NakedTriple = NakedSet<3>;
pub type NakedQuad = NakedSet<4>;

impl<const K: usize, const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC>
    for NakedSet<K>
{
    fn id(&self) -> TechniqueId {
        match K {
            2 => TechniqueId::NakedPair,
            3 => TechniqueId::NakedTriple,
            4 => TechniqueId::NakedQuad,
            _ => panic!("NakedSet only supported for K in {{2,3,4}}"),
        }
    }
    fn tier(&self) -> Tier {
        match K {
            2 => Tier::T2,
            _ => Tier::T3,
        }
    }
    fn name(&self) -> &'static str {
        match K {
            2 => "naked_pairs",
            3 => "naked_triples",
            4 => "naked_quads",
            _ => "naked_set",
        }
    }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        debug_assert!((2..=4).contains(&K), "K must be 2, 3, or 4");
        let table = grid.table().clone();
        let n_units = 3 * N;
        let mut prog = TechniqueProgress::default();

        // Per-unit scratch: list of (cell, mask) for unsolved cells with
        // 2 ≤ popcount ≤ K.
        let mut cells_buf: Vec<(usize, u32)> = Vec::with_capacity(N);

        for u in 0..n_units {
            cells_buf.clear();
            // Snapshot unit cells (small) so we can mutate grid during elimination.
            // `unit_cells` is used both for collecting candidates and iterating
            // the elimination targets.
            let unit_cells: Vec<usize> = table.units[u].iter().map(|&x| x as usize).collect();
            for &c in &unit_cells {
                if grid.solved[c] != 0 { continue; }
                let m = grid.candidates[c];
                let pop = m.count_ones() as usize;
                if pop >= 2 && pop <= K {
                    cells_buf.push((c, m));
                }
            }
            if cells_buf.len() < K { continue; }

            // Iterate K-subsets in lex order.
            let mut idx = [0usize; 4]; // K ≤ 4
            for i in 0..K { idx[i] = i; }
            loop {
                let mut union: u32 = 0;
                for i in 0..K { union |= cells_buf[idx[i]].1; }
                if union.count_ones() as usize == K {
                    // Selected cells (skip these during elimination).
                    let mut combo_cells = [usize::MAX; 4];
                    for i in 0..K { combo_cells[i] = cells_buf[idx[i]].0; }
                    for &cu in &unit_cells {
                        let mut in_combo = false;
                        for i in 0..K { if combo_cells[i] == cu { in_combo = true; break; } }
                        if in_combo { continue; }
                        if grid.solved[cu] != 0 { continue; }
                        let to_remove = grid.candidates[cu] & union;
                        let mut rb = to_remove;
                        while rb != 0 {
                            let bb = rb & rb.wrapping_neg();
                            rb ^= bb;
                            let d = bb.trailing_zeros() as u8 + 1;
                            match grid.eliminate(cu, d) {
                                Ok(true) => prog.eliminations.push((cu, d)),
                                Ok(false) => {}
                                Err(_) => { prog.contradiction = true; return Some(prog); }
                            }
                        }
                    }
                }
                if !next_combo(&mut idx, K, cells_buf.len()) { break; }
            }
        }
        if prog.fired() || prog.contradiction { Some(prog) } else { None }
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for NakedSet<2>
{
    fn base_rating(&self) -> f64 { 3.0 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for NakedSet<3>
{
    fn base_rating(&self) -> f64 { 3.6 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for NakedSet<4>
{
    fn base_rating(&self) -> f64 { 5.0 }
}

fn next_combo(idx: &mut [usize; 4], k: usize, nc: usize) -> bool {
    let mut i = k;
    loop {
        if i == 0 { return false; }
        i -= 1;
        let max_i = nc - (k - i);
        if idx[i] < max_i {
            idx[i] += 1;
            for j in (i + 1)..k { idx[j] = idx[j - 1] + 1; }
            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    // 9×9: classic naked pair {1,2} in row 0, cells 0 and 1.
    #[test]
    fn naked_pair_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 {
            g.eliminate(0, d).unwrap();
            g.eliminate(1, d).unwrap();
        }
        let t = NakedSet::<2>;
        let p = <NakedPair as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Cell 2 (same row) lost {1,2}.
        assert_eq!(g.candidates[2] & 0b11, 0);
        // Idempotent.
        let q = <NakedPair as Technique<9,3,3>>::apply(&t, &mut g);
        assert!(q.is_none(), "second apply should be None");
    }

    // 9×9 naked triple: cells 0,1,2 with subsets of {1,2,3}.
    #[test]
    fn naked_triple_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Cell 0: {1,2}. Cell 1: {2,3}. Cell 2: {1,3}.
        // Eliminate everything else from each.
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [1u8].iter().copied() { g.eliminate(1, d).unwrap(); }
        for d in 4..=9u8 { g.eliminate(1, d).unwrap(); }
        for d in [2u8].iter().copied() { g.eliminate(2, d).unwrap(); }
        for d in 4..=9u8 { g.eliminate(2, d).unwrap(); }
        // Now cell 0: {1,2}, cell 1: {2,3}, cell 2: {1,3}; union = {1,2,3}.
        assert_eq!(g.candidates[0], 0b011);
        assert_eq!(g.candidates[1], 0b110);
        assert_eq!(g.candidates[2], 0b101);
        let t = NakedSet::<3>;
        let p = <NakedTriple as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Cells 3..9 lose 1, 2, 3.
        for c in 3..9usize {
            assert_eq!(g.candidates[c] & 0b111, 0, "cell {} still has 1/2/3", c);
        }
    }

    // 9×9 naked quad.
    #[test]
    fn naked_quad_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // Cell 0: {1,2}, cell 1: {2,3}, cell 2: {3,4}, cell 3: {1,4}.
        // Union = {1,2,3,4}.
        let masks = [0b0011u32, 0b0110, 0b1100, 0b1001];
        for (c, m) in masks.iter().enumerate() {
            for d in 1..=9u8 {
                if m & (1u32 << (d - 1)) == 0 {
                    g.eliminate(c, d).unwrap();
                }
            }
            assert_eq!(g.candidates[c], *m);
        }
        let t = NakedSet::<4>;
        let p = <NakedQuad as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 4..9usize {
            assert_eq!(g.candidates[c] & 0b1111, 0, "cell {} still has 1..4", c);
        }
    }

    // 6×6 naked pair.
    #[test]
    fn naked_pair_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        for d in 3..=6u8 {
            g.eliminate(0, d).unwrap();
            g.eliminate(1, d).unwrap();
        }
        let t = NakedSet::<2>;
        let p = <NakedPair as Technique<6,2,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        assert_eq!(g.candidates[2] & 0b11, 0);
    }

    // 12×12 naked triple in a row.
    #[test]
    fn naked_triple_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        // Cells 0,1,2 in row 0. masks {1,2}, {2,3}, {1,3}.
        let want = [0b011u32, 0b110, 0b101];
        for (c, m) in want.iter().enumerate() {
            for d in 1..=12u8 {
                if m & (1u32 << (d - 1)) == 0 {
                    g.eliminate(c, d).unwrap();
                }
            }
        }
        let t = NakedSet::<3>;
        let p = <NakedTriple as Technique<12,3,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 3..12usize {
            assert_eq!(g.candidates[c] & 0b111, 0);
        }
    }

    // 16×16 naked quad in a row.
    #[test]
    fn naked_quad_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        let masks = [0b0011u32, 0b0110, 0b1100, 0b1001];
        for (c, m) in masks.iter().enumerate() {
            for d in 1..=16u8 {
                if m & (1u32 << (d - 1)) == 0 {
                    g.eliminate(c, d).unwrap();
                }
            }
        }
        let t = NakedSet::<4>;
        let p = <NakedQuad as Technique<16,4,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 4..16usize {
            assert_eq!(g.candidates[c] & 0b1111, 0);
        }
    }

    #[test]
    fn no_fire_on_empty() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let p = NakedSet::<2>; let t = NakedSet::<3>; let q = NakedSet::<4>;
        assert!(<NakedPair as Technique<9,3,3>>::apply(&p, &mut g).is_none());
        assert!(<NakedTriple as Technique<9,3,3>>::apply(&t, &mut g).is_none());
        assert!(<NakedQuad as Technique<9,3,3>>::apply(&q, &mut g).is_none());
    }
}
