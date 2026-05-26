//! # Fish (XWing K=2 / Swordfish K=3 / Jellyfish K=4)
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
//! Target: < 50 µs/grid on 9×9 on commodity x86_64.
//!
//! ## Algorithm reference
//! For digit d, find K base lines (rows or cols) whose candidate positions
//! span at most K cover lines of the transversal direction; eliminate d from
//! non-base cells in those cover lines. (Hodoku fish glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct Fish<const K: usize>;

pub type XWing = Fish<2>;
pub type Swordfish = Fish<3>;
pub type Jellyfish = Fish<4>;

impl<const K: usize, const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC>
    for Fish<K>
{
    fn id(&self) -> TechniqueId {
        match K {
            2 => TechniqueId::XWing,
            3 => TechniqueId::Swordfish,
            4 => TechniqueId::Jellyfish,
            _ => panic!("Fish only supported for K in {{2,3,4}}"),
        }
    }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str {
        match K {
            2 => "x_wing",
            3 => "swordfish",
            4 => "jellyfish",
            _ => "fish",
        }
    }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        debug_assert!((2..=4).contains(&K), "K must be 2, 3, or 4");
        if K >= N { return None; }
        let table = grid.table().clone();
        for d_bit in 0..(N as u8) {
            let bit = 1u32 << d_bit;
            let digit = d_bit + 1;
            let mut row_mask = vec![0u32; N];
            let mut col_mask = vec![0u32; N];
            let mut placed_row = vec![false; N];
            let mut placed_col = vec![false; N];
            for cell in 0..(N * N) {
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
                    row_mask[r] |= 1u32 << c;
                    col_mask[c] |= 1u32 << r;
                }
            }
            if let Some(p) = try_fish_k::<N, BR, BC>(grid, digit, K, &row_mask, &placed_row, true) {
                return Some(p);
            }
            if let Some(p) = try_fish_k::<N, BR, BC>(grid, digit, K, &col_mask, &placed_col, false) {
                return Some(p);
            }
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for Fish<2>
{
    fn base_rating(&self) -> f64 { 3.2 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for Fish<3>
{
    fn base_rating(&self) -> f64 { 3.8 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for Fish<4>
{
    fn base_rating(&self) -> f64 { 5.2 }
}

fn try_fish_k<const N: usize, const BR: usize, const BC: usize>(
    grid: &mut Grid<N, BR, BC>,
    digit: u8,
    k: usize,
    line_mask: &[u32],
    placed_line: &[bool],
    base_is_row: bool,
) -> Option<TechniqueProgress> {
    let mut bases: Vec<usize> = Vec::with_capacity(N);
    for i in 0..N {
        if placed_line[i] { continue; }
        let pc = line_mask[i].count_ones() as usize;
        if (2..=k).contains(&pc) { bases.push(i); }
    }
    if bases.len() < k { return None; }

    let mut idx = vec![0usize; k];
    for kk in 0..k { idx[kk] = kk; }
    let bit = 1u32 << (digit - 1);
    loop {
        let mut union: u32 = 0;
        for kk in 0..k { union |= line_mask[bases[idx[kk]]]; }
        if (union.count_ones() as usize) == k {
            let mut base_bits: u32 = 0;
            for kk in 0..k { base_bits |= 1u32 << bases[idx[kk]]; }
            let mut elims: Vec<(usize, u8)> = Vec::new();
            let mut cover = union;
            while cover != 0 {
                let cb = cover & cover.wrapping_neg();
                cover ^= cb;
                let cover_idx = cb.trailing_zeros() as usize;
                for line in 0..N {
                    if (base_bits >> line) & 1 != 0 { continue; }
                    let cell = if base_is_row {
                        line * N + cover_idx
                    } else {
                        cover_idx * N + line
                    };
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
            }
            if !elims.is_empty() {
                let mut prog = TechniqueProgress::default();
                prog.eliminations = elims;
                return Some(prog);
            }
        }
        if !next_combination(&mut idx, bases.len()) { break; }
    }
    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn xwing_fires_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for c in 0..9 {
            if c != 4 && c != 7 { g.eliminate(0 * 9 + c, 5).unwrap(); }
        }
        for c in 0..9 {
            if c != 4 && c != 7 { g.eliminate(4 * 9 + c, 5).unwrap(); }
        }
        let bit5 = 1u32 << 4;
        assert!(g.candidates[2 * 9 + 4] & bit5 != 0);
        let t = Fish::<2>;
        let p = <Fish<2> as Technique<9,3,3>>::apply(&t, &mut g).expect("x-wing fires");
        assert!(!p.contradiction && !p.eliminations.is_empty());
        assert_eq!(g.candidates[2 * 9 + 4] & bit5, 0);
        assert_eq!(g.candidates[2 * 9 + 7] & bit5, 0);
        // idempotent
        assert!(<Fish<2> as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn swordfish_fires_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let allowed = [(0usize, [0,1].as_slice()), (1, [1,2].as_slice()), (2, [0,2].as_slice())];
        for &(r, a) in &allowed {
            for c in 0..9 { if !a.contains(&c) { g.eliminate(r * 9 + c, 7).unwrap(); } }
        }
        let bit = 1u32 << 6;
        assert!(g.candidates[4 * 9 + 0] & bit != 0);
        let t = Fish::<3>;
        let p = <Fish<3> as Technique<9,3,3>>::apply(&t, &mut g).expect("swordfish fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[4 * 9 + 0] & bit, 0);
    }

    #[test]
    fn jellyfish_fires_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        // Rows 0..4 each restrict digit 7 to cols 0..4. Union has 4 cover cols.
        let allowed_per_row: [(usize, &[usize]); 4] = [
            (0, &[0, 1, 2, 3]),
            (1, &[0, 1, 2, 3]),
            (2, &[0, 1, 2, 3]),
            (3, &[0, 1, 2, 3]),
        ];
        for &(r, a) in &allowed_per_row {
            for c in 0..16 { if !a.contains(&c) { g.eliminate(r * 16 + c, 7).unwrap(); } }
        }
        let bit = 1u32 << 6;
        assert!(g.candidates[4 * 16 + 0] & bit != 0);
        let t = Fish::<4>;
        let p = <Fish<4> as Technique<16,4,4>>::apply(&t, &mut g).expect("jellyfish fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[4 * 16 + 0] & bit, 0);
    }

    #[test]
    fn xwing_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = Fish::<2>;
        assert!(<Fish<2> as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn jellyfish_returns_none_on_6x6_no_pattern_possible() {
        // K=4 with N=6 leaves only 2 cover lines but they'd need to be 4-wide;
        // defensive guard returns None when K >= N is too tight; for K=4, N=6
        // it's allowed by the guard but no pattern exists in an empty grid.
        let mut g: Grid<6, 2, 3> = Grid::empty();
        let t = Fish::<4>;
        assert!(<Fish<4> as Technique<6,2,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn xwing_fires_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        for c in 0..12 {
            if c != 5 && c != 8 { g.eliminate(0 * 12 + c, 5).unwrap(); }
        }
        for c in 0..12 {
            if c != 5 && c != 8 { g.eliminate(6 * 12 + c, 5).unwrap(); }
        }
        let bit = 1u32 << 4;
        assert!(g.candidates[3 * 12 + 5] & bit != 0);
        let t = Fish::<2>;
        let p = <Fish<2> as Technique<12,3,4>>::apply(&t, &mut g).expect("x-wing 12x12 fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[3 * 12 + 5] & bit, 0);
        assert_eq!(g.candidates[3 * 12 + 8] & bit, 0);
    }
}
