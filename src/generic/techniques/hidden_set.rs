//! # HiddenSet (HiddenPair / HiddenTriple / HiddenQuad)
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
//! For each unit, find K digits whose candidate positions union to exactly K
//! cells; those cells can only hold those K digits — eliminate all other
//! candidates from them. (Standard hidden-set pattern, Hodoku glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct HiddenSet<const K: usize>;

pub type HiddenPair = HiddenSet<2>;
pub type HiddenTriple = HiddenSet<3>;
pub type HiddenQuad = HiddenSet<4>;

impl<const K: usize, const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC>
    for HiddenSet<K>
{
    fn id(&self) -> TechniqueId {
        match K {
            2 => TechniqueId::HiddenPair,
            3 => TechniqueId::HiddenTriple,
            4 => TechniqueId::HiddenQuad,
            _ => panic!("HiddenSet only supported for K in {{2,3,4}}"),
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
            2 => "hidden_pairs",
            3 => "hidden_triples",
            4 => "hidden_quads",
            _ => "hidden_set",
        }
    }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        debug_assert!((2..=4).contains(&K), "K must be 2, 3, or 4");
        let table = grid.table().clone();
        let n_units = 3 * N;
        let mut prog = TechniqueProgress::default();

        // where_d: for each digit 1..=N, bitmap (u32) of positions 0..N
        // within the unit where the digit is a candidate.
        let mut where_d: Vec<u32> = vec![0u32; N + 1]; // index 1..=N
        let mut avail: Vec<u8> = Vec::with_capacity(N);

        for u in 0..n_units {
            // Reset scratch.
            for v in where_d.iter_mut() { *v = 0; }
            avail.clear();

            let unit_cells: Vec<usize> = table.units[u].iter().map(|&x| x as usize).collect();
            // Mask of digits already solved in this unit.
            let mut placed: u32 = 0;
            for k in 0..N {
                let c = unit_cells[k];
                if grid.solved[c] != 0 {
                    placed |= 1u32 << (grid.solved[c] - 1);
                    continue;
                }
                let mut bits = grid.candidates[c];
                while bits != 0 {
                    let bit = bits & bits.wrapping_neg();
                    bits ^= bit;
                    let d = (bit.trailing_zeros() as usize) + 1;
                    where_d[d] |= 1u32 << k;
                }
            }
            // Available digits: not yet placed and 2 ≤ popcount ≤ K.
            for d in 1u8..=(N as u8) {
                if placed & (1u32 << (d - 1)) != 0 { continue; }
                let pop = where_d[d as usize].count_ones() as usize;
                if pop >= 2 && pop <= K {
                    avail.push(d);
                }
            }
            if avail.len() < K { continue; }

            let na = avail.len();
            let mut idx = [0usize; 4];
            for i in 0..K { idx[i] = i; }
            loop {
                let mut union_pos: u32 = 0;
                let mut digit_mask: u32 = 0;
                for i in 0..K {
                    let d = avail[idx[i]];
                    union_pos |= where_d[d as usize];
                    digit_mask |= 1u32 << (d - 1);
                }
                if union_pos.count_ones() as usize == K {
                    // Restrict candidates of cells at union_pos positions to digit_mask.
                    let mut bits = union_pos;
                    while bits != 0 {
                        let bb = bits & bits.wrapping_neg();
                        bits ^= bb;
                        let k = bb.trailing_zeros() as usize;
                        let c = unit_cells[k];
                        let to_remove = grid.candidates[c] & !digit_mask;
                        let mut rb = to_remove;
                        while rb != 0 {
                            let rbb = rb & rb.wrapping_neg();
                            rb ^= rbb;
                            let dd = rbb.trailing_zeros() as u8 + 1;
                            match grid.eliminate(c, dd) {
                                Ok(true) => prog.eliminations.push((c, dd)),
                                Ok(false) => {}
                                Err(_) => { prog.contradiction = true; return Some(prog); }
                            }
                        }
                    }
                }
                if !next_combo(&mut idx, K, na) { break; }
            }
        }
        if prog.fired() || prog.contradiction { Some(prog) } else { None }
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for HiddenSet<2>
{
    fn base_rating(&self) -> f64 { 3.4 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for HiddenSet<3>
{
    fn base_rating(&self) -> f64 { 4.0 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for HiddenSet<4>
{
    fn base_rating(&self) -> f64 { 5.4 }
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

    // 9×9 hidden pair: digits 1,2 confined to cells 0,1 of row 0.
    #[test]
    fn hidden_pair_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 2..9usize {
            g.eliminate(c, 1).unwrap();
            g.eliminate(c, 2).unwrap();
        }
        let t = HiddenSet::<2>;
        let p = <HiddenPair as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        // Cells 0,1 should now have only {1,2}.
        assert_eq!(g.candidates[0], 0b11);
        assert_eq!(g.candidates[1], 0b11);
        let q = <HiddenPair as Technique<9,3,3>>::apply(&t, &mut g);
        assert!(q.is_none());
    }

    // 9×9 hidden triple: digits 1,2,3 confined to cells 0,1,2 of row 0.
    #[test]
    fn hidden_triple_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 3..9usize {
            for d in [1u8, 2, 3] { g.eliminate(c, d).unwrap(); }
        }
        let t = HiddenSet::<3>;
        let p = <HiddenTriple as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 0..3usize {
            assert_eq!(g.candidates[c] & !0b111u32, 0);
        }
    }

    // 9×9 hidden quad.
    #[test]
    fn hidden_quad_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 4..9usize {
            for d in [1u8, 2, 3, 4] { g.eliminate(c, d).unwrap(); }
        }
        let t = HiddenSet::<4>;
        let p = <HiddenQuad as Technique<9,3,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 0..4usize {
            assert_eq!(g.candidates[c] & !0b1111u32, 0);
        }
    }

    // 6×6 hidden pair.
    #[test]
    fn hidden_pair_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        for c in 2..6usize {
            g.eliminate(c, 1).unwrap();
            g.eliminate(c, 2).unwrap();
        }
        let t = HiddenSet::<2>;
        let p = <HiddenPair as Technique<6,2,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        assert_eq!(g.candidates[0], 0b11);
        assert_eq!(g.candidates[1], 0b11);
    }

    // 6×6 hidden triple — N=6, K=3, three digits confined to three cells.
    #[test]
    fn hidden_triple_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        for c in 3..6usize {
            for d in [1u8, 2, 3] { g.eliminate(c, d).unwrap(); }
        }
        let t = HiddenSet::<3>;
        let p = <HiddenTriple as Technique<6,2,3>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 0..3usize { assert_eq!(g.candidates[c] & !0b111u32, 0); }
    }

    // 6×6: hidden_quad construction would need N≥4 cells confined; N=6 supports it
    // but it's degenerate (4 cells with full mask of remaining 4 digits ≡ naked-pair on
    // the other 2). We just verify no fire on empty grid.
    #[test]
    fn hidden_quad_6x6_empty_no_fire() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        let t = HiddenSet::<4>;
        assert!(<HiddenQuad as Technique<6,2,3>>::apply(&t, &mut g).is_none());
    }

    // 12×12 hidden triple.
    #[test]
    fn hidden_triple_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        for c in 3..12usize {
            for d in [1u8, 2, 3] { g.eliminate(c, d).unwrap(); }
        }
        let t = HiddenSet::<3>;
        let p = <HiddenTriple as Technique<12,3,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 0..3usize { assert_eq!(g.candidates[c] & !0b111u32, 0); }
    }

    // 16×16 hidden quad.
    #[test]
    fn hidden_quad_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        for c in 4..16usize {
            for d in [1u8, 2, 3, 4] { g.eliminate(c, d).unwrap(); }
        }
        let t = HiddenSet::<4>;
        let p = <HiddenQuad as Technique<16,4,4>>::apply(&t, &mut g).expect("fire");
        assert!(p.fired());
        for c in 0..4usize { assert_eq!(g.candidates[c] & !0b1111u32, 0); }
    }

    #[test]
    fn no_fire_on_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let p = HiddenSet::<2>; let t = HiddenSet::<3>; let q = HiddenSet::<4>;
        assert!(<HiddenPair as Technique<9,3,3>>::apply(&p, &mut g).is_none());
        assert!(<HiddenTriple as Technique<9,3,3>>::apply(&t, &mut g).is_none());
        assert!(<HiddenQuad as Technique<9,3,3>>::apply(&q, &mut g).is_none());
    }
}
