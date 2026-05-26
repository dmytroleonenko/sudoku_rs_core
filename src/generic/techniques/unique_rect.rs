//! # UniqueRectangle (UrType1 + UrType2)
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
//! Unique Rectangle: four cells forming a 2×2 rectangle across exactly 2
//! boxes; Type 1 eliminates the UR digits from the one "extra" corner; Type 2
//! uses a common extra digit locked into the two non-bivalue corners.
//! (Hodoku UR glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

/// Which UR branch a struct exposes.
#[derive(Clone, Copy)]
enum UrKind { Type1, Type2 }

/// Convenience type — runs Type 1 only.
pub struct UrType1;
/// Convenience type — runs Type 2 only.
pub struct UrType2;
/// Combined Type 1 OR Type 2 (for callers that don't need to distinguish);
/// reports `TechniqueId::UrType1` since it can fire either branch — prefer
/// `UrType1`/`UrType2` separately for telemetry.
pub struct UniqueRectangle;

#[inline(always)]
fn box_id<const N: usize, const BR: usize, const BC: usize>(r: usize, c: usize) -> usize {
    let n_box_cols = N / BC; // = BR
    (r / BR) * n_box_cols + (c / BC)
}

/// Build "is peer of `cell`" lookup as a length-`N*N` bitvec.
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

fn try_rectangle<const N: usize, const BR: usize, const BC: usize>(
    grid: &mut Grid<N, BR, BC>,
    cells: [usize; 4],
    kind: UrKind,
) -> Option<TechniqueProgress> {
    if cells.iter().any(|&c| grid.solved[c] != 0) {
        return None;
    }
    let masks = [
        grid.candidates[cells[0]],
        grid.candidates[cells[1]],
        grid.candidates[cells[2]],
        grid.candidates[cells[3]],
    ];
    // cells[0]=(r1,c1) cells[1]=(r1,c2) cells[2]=(r2,c1) cells[3]=(r2,c2).
    // Sides: {0,1} share row, {2,3} share row, {0,2} share col, {1,3} share col.

    if matches!(kind, UrKind::Type1) {
    // -------- Type 1
    for ext in 0..4usize {
        let others: [usize; 3] = match ext {
            0 => [1, 2, 3],
            1 => [0, 2, 3],
            2 => [0, 1, 3],
            _ => [0, 1, 2],
        };
        let m0 = masks[others[0]];
        if m0.count_ones() != 2 { continue; }
        if masks[others[1]] != m0 || masks[others[2]] != m0 { continue; }
        let m_ext = masks[ext];
        if m_ext & m0 != m0 { continue; }
        if m_ext.count_ones() <= 2 { continue; }
        let mut elims: Vec<(usize, u8)> = Vec::new();
        let mut bits = m0;
        while bits != 0 {
            let b = bits & bits.wrapping_neg();
            bits ^= b;
            let d = b.trailing_zeros() as u8 + 1;
            match grid.eliminate(cells[ext], d) {
                Ok(true) => elims.push((cells[ext], d)),
                Ok(false) => {}
                Err(_) => {
                    let mut prog = TechniqueProgress::default();
                    prog.eliminations = elims;
                    prog.contradiction = true;
                    return Some(prog);
                }
            }
        }
        if !elims.is_empty() {
            let mut prog = TechniqueProgress::default();
            prog.eliminations = elims;
            return Some(prog);
        }
    }
    return None;
    } // end Type 1 branch

    // -------- Type 2 only (same-row pair or same-col pair; diagonal=Type 5
    // deferred). Pair partitions:
    //   pi=0: bivalue pair on same row, ext pair on same row
    //   pi=1: bivalue pair on same col, ext pair on same col
    let pair_partitions: [[(usize, usize); 2]; 2] = [
        [(0, 1), (2, 3)], // by row sides
        [(0, 2), (1, 3)], // by col sides
    ];
    for parts in pair_partitions.iter() {
        for swap in 0..2 {
            let (b_a, b_b) = if swap == 0 { parts[0] } else { parts[1] };
            let (e_a, e_b) = if swap == 0 { parts[1] } else { parts[0] };
            let m0 = masks[b_a];
            if m0.count_ones() != 2 { continue; }
            if masks[b_b] != m0 { continue; }
            let me_a = masks[e_a];
            let me_b = masks[e_b];
            if me_a & m0 != m0 || me_b & m0 != m0 { continue; }
            let extra_a = me_a & !m0;
            let extra_b = me_b & !m0;
            if extra_a == 0 || extra_b == 0 { continue; }
            // Type 2: same single extra digit.
            if extra_a == extra_b && extra_a.count_ones() == 1 {
                let x_bit = extra_a;
                let x_digit = x_bit.trailing_zeros() as u8 + 1;
                // common peers of cells[e_a] and cells[e_b], excluding the 4
                // rectangle corners.
                let lookup_b = peer_lookup::<N, BR, BC>(grid, cells[e_b]);
                let table = grid.table().clone();
                let mut elims: Vec<(usize, u8)> = Vec::new();
                for &p in &table.cells[cells[e_a]].peers {
                    let cell = p as usize;
                    if !lookup_b[cell] { continue; }
                    if cell == cells[0] || cell == cells[1]
                        || cell == cells[2] || cell == cells[3] { continue; }
                    if grid.solved[cell] != 0 { continue; }
                    if grid.candidates[cell] & x_bit != 0 {
                        match grid.eliminate(cell, x_digit) {
                            Ok(true) => elims.push((cell, x_digit)),
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

/// Iterate every UR rectangle (4 cells, 2 rows × 2 cols × exactly 2 boxes)
/// and call `f(cells)`; return early with `Some(progress)` if `f` returns
/// `Some`. Two enumeration classes are visited:
///   (a) rows in same band, cols in different stacks → 2 boxes side-by-side
///   (b) cols in same stack, rows in different bands → 2 boxes top/bottom
/// Class (a) and (b) produce disjoint sets when BR ≠ BC; for BR == BC each
/// rectangle still appears in only one class because (a) requires same band
/// + different stack and (b) requires same stack + different band. A rect
/// with same band AND same stack is fully inside one box (excluded from both
/// classes). A rect with different band AND different stack spans 4 boxes
/// and is not enumerated (legitimate UR requires exactly 2 boxes).
fn iterate_rectangles<const N: usize, const BR: usize, const BC: usize, F>(
    grid: &mut Grid<N, BR, BC>,
    mut f: F,
) -> Option<TechniqueProgress>
where
    F: FnMut(&mut Grid<N, BR, BC>, [usize; 4]) -> Option<TechniqueProgress>,
{
    // Class (a): same band, different stacks.
    let n_bands = N / BR;
    for band in 0..n_bands {
        let r_base = band * BR;
        for r1 in r_base..(r_base + BR) {
            for r2 in (r1 + 1)..(r_base + BR) {
                for c1 in 0..N {
                    for c2 in (c1 + 1)..N {
                        if c1 / BC == c2 / BC { continue; }
                        let cells = [
                            r1 * N + c1,
                            r1 * N + c2,
                            r2 * N + c1,
                            r2 * N + c2,
                        ];
                        debug_assert_ne!(box_id::<N,BR,BC>(r1,c1), box_id::<N,BR,BC>(r1,c2));
                        debug_assert_eq!(box_id::<N,BR,BC>(r1,c1), box_id::<N,BR,BC>(r2,c1));
                        if let Some(p) = f(grid, cells) {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    // Class (b): same stack, different bands.
    let n_stacks = N / BC;
    for stack in 0..n_stacks {
        let c_base = stack * BC;
        for c1 in c_base..(c_base + BC) {
            for c2 in (c1 + 1)..(c_base + BC) {
                for r1 in 0..N {
                    for r2 in (r1 + 1)..N {
                        if r1 / BR == r2 / BR { continue; }
                        let cells = [
                            r1 * N + c1,
                            r1 * N + c2,
                            r2 * N + c1,
                            r2 * N + c2,
                        ];
                        debug_assert_eq!(box_id::<N,BR,BC>(r1,c1), box_id::<N,BR,BC>(r1,c2));
                        debug_assert_ne!(box_id::<N,BR,BC>(r1,c1), box_id::<N,BR,BC>(r2,c1));
                        if let Some(p) = f(grid, cells) {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    None
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for UrType1 {
    fn id(&self) -> TechniqueId { TechniqueId::UrType1 }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "unique_rectangle_t1" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        iterate_rectangles::<N, BR, BC, _>(grid, |g, cells| {
            try_rectangle::<N, BR, BC>(g, cells, UrKind::Type1)
        })
    }
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for UrType2 {
    fn id(&self) -> TechniqueId { TechniqueId::UrType2 }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "unique_rectangle_t2" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        iterate_rectangles::<N, BR, BC, _>(grid, |g, cells| {
            try_rectangle::<N, BR, BC>(g, cells, UrKind::Type2)
        })
    }
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for UniqueRectangle {
    fn id(&self) -> TechniqueId { TechniqueId::UrType1 }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "unique_rectangle" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        // Convenience: try Type 1 first, then Type 2. Note: this combined
        // entry-point reports id=UrType1 unconditionally; callers needing
        // accurate per-branch attribution should use `UrType1` and `UrType2`
        // directly.
        if let Some(p) = (UrType1).apply(grid) { return Some(p); }
        (UrType2).apply(grid)
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for UrType1
{
    fn base_rating(&self) -> f64 { 4.5 }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for UrType2
{
    fn base_rating(&self) -> f64 { 4.7 }
}

// TODO sub-stage 3.5: Types 3, 4, 5, 6, 7.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn ur_type1_fires_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for cell in &[0usize, 4, 9] {
            for d in 3..=9u8 { g.eliminate(*cell, d).unwrap(); }
        }
        for d in 4..=9u8 { g.eliminate(13, d).unwrap(); }
        assert_eq!(g.candidates[13].count_ones(), 3);
        let t = UniqueRectangle;
        let p = <UniqueRectangle as Technique<9,3,3>>::apply(&t, &mut g).expect("UR T1 fires");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[13].count_ones(), 1);
        assert_eq!(g.candidates[13], 0b100);
    }

    #[test]
    fn ur_no_fire_on_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = UniqueRectangle;
        assert!(<UniqueRectangle as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn ur_type2_fires_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        // (0,0),(0,4) bivalue {1,2}
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in 3..=9u8 { g.eliminate(4, d).unwrap(); }
        // (1,0),(1,4) {1,2,3}
        for d in 4..=9u8 { g.eliminate(9, d).unwrap(); }
        for d in 4..=9u8 { g.eliminate(13, d).unwrap(); }
        let bit3 = 0b100u32;
        assert!(g.candidates[14] & bit3 != 0);
        let t = UniqueRectangle;
        let p = <UniqueRectangle as Technique<9,3,3>>::apply(&t, &mut g).expect("T2 fires");
        assert!(!p.eliminations.is_empty());
        // 3 must be eliminated from row-1 cells outside the rectangle (cols !=0,!=4).
        for c in 1..9usize {
            if c == 4 { continue; }
            let cell = 9 + c;
            assert_eq!(g.candidates[cell] & bit3, 0, "3 still in (1,{})", c);
        }
    }

    #[test]
    fn ur_type1_fires_12x12() {
        // Rectangle: r1=0, r2=1 (same band, BR=3); c1=0, c2=4 (different box-col,
        // since BC=4: 0/4=0, 4/4=1).
        // Bivalue {1,2} at 3 corners; 4th has {1,2,5}.
        let mut g: Grid<12, 3, 4> = Grid::empty();
        let corners = [0usize, 4, 12, 16];
        // (0,0),(0,4),(1,0) → {1,2}: kill 3..=12
        for &c in &corners[0..3] {
            for d in 3..=12u8 { g.eliminate(c, d).unwrap(); }
        }
        // (1,4) → {1,2,5}: kill 3,4,6..=12
        let kill: Vec<u8> = vec![3u8, 4].into_iter().chain(6..=12u8).collect();
        for d in kill { g.eliminate(16, d).unwrap(); }
        assert_eq!(g.candidates[16].count_ones(), 3);
        let t = UniqueRectangle;
        let p = <UniqueRectangle as Technique<12,3,4>>::apply(&t, &mut g).expect("UR T1 12×12");
        assert!(!p.eliminations.is_empty());
        // After fire, 4th corner should have only digit 5 (bit 4).
        assert_eq!(g.candidates[16], 1u32 << 4);
    }

    #[test]
    fn ur_type1_fires_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        // r1=0, r2=1 same band; c1=0, c2=4 different box-col.
        let corners = [0usize, 4, 16, 20];
        for &c in &corners[0..3] {
            for d in 3..=16u8 { g.eliminate(c, d).unwrap(); }
        }
        let kill: Vec<u8> = vec![3u8, 4].into_iter().chain(6..=16u8).collect();
        for d in kill { g.eliminate(20, d).unwrap(); }
        assert_eq!(g.candidates[20].count_ones(), 3);
        let t = UniqueRectangle;
        let p = <UniqueRectangle as Technique<16,4,4>>::apply(&t, &mut g).expect("UR T1 16×16");
        assert!(!p.eliminations.is_empty());
        assert_eq!(g.candidates[20], 1u32 << 4);
    }

    #[test]
    fn ur_type2_fires_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        // Same idea as 9×9 T2: bivalue pair on row, ext pair on row.
        // Rows 0,1 (same band, BR=3); cols 0,4 (different box-col, BC=4).
        // (0,0),(0,4) bivalue {1,2}: kill 3..=12.
        for c in [0usize, 4] {
            for d in 3..=12u8 { g.eliminate(c, d).unwrap(); }
        }
        // (1,0),(1,4) {1,2,3}: kill 4..=12.
        for c in [12usize, 16] {
            for d in 4..=12u8 { g.eliminate(c, d).unwrap(); }
        }
        let bit3 = 0b100u32;
        // (1,5) = idx 17 — should still have 3.
        assert!(g.candidates[17] & bit3 != 0);
        let t = UniqueRectangle;
        let p = <UniqueRectangle as Technique<12,3,4>>::apply(&t, &mut g).expect("T2 12x12");
        assert!(!p.eliminations.is_empty());
        // 3 must be eliminated from row-1 outside rectangle.
        for c in 1..12usize {
            if c == 4 { continue; }
            let cell = 12 + c;
            assert_eq!(g.candidates[cell] & bit3, 0, "3 still in (1,{})", c);
        }
    }

    /// Class-(b) UR enumeration: rows in different bands, cols in same stack.
    /// On 12×12 (BR=3, BC=4): r1=0 (band 0), r2=3 (band 1), c1=0, c2=1
    /// (both in stack 0, since 0/4=1/4=0). The two boxes are box(0,0) and
    /// box(1,0) — exactly 2 boxes. Construct UR T1: 3 corners {1,2}, 4th
    /// {1,2,5}.
    #[test]
    fn ur_class_b_fires_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        // r1=0,r2=3, c1=0,c2=1 → cells 0, 1, 36, 37.
        let corners = [0usize, 1, 36, 37];
        for &c in &corners[0..3] {
            for d in 3..=12u8 { g.eliminate(c, d).unwrap(); }
        }
        let kill: Vec<u8> = vec![3u8, 4].into_iter().chain(6..=12u8).collect();
        for d in kill { g.eliminate(37, d).unwrap(); }
        assert_eq!(g.candidates[37].count_ones(), 3);
        let t = UrType1;
        let p = <UrType1 as Technique<12,3,4>>::apply(&t, &mut g)
            .expect("class-(b) UR T1 12×12 must fire");
        assert!(!p.eliminations.is_empty());
        // 4th corner reduced to just digit 5.
        assert_eq!(g.candidates[37], 1u32 << 4);
    }

    /// UrType1 reports `TechniqueId::UrType1`; UrType2 reports `UrType2`.
    /// (Mislabeling bug fix from review.)
    #[test]
    fn ur_type_ids_are_distinct() {
        let g: Grid<9, 3, 3> = Grid::empty();
        let _ = g; // unused
        let t1 = UrType1;
        let t2 = UrType2;
        assert_eq!(<UrType1 as Technique<9,3,3>>::id(&t1), TechniqueId::UrType1);
        assert_eq!(<UrType2 as Technique<9,3,3>>::id(&t2), TechniqueId::UrType2);
    }
}
