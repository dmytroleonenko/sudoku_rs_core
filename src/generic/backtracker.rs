//! Generic propagator: naked + hidden singles to fixed point.
//!
//! Mirrors `crate::backtracker::propagate_singles` but works on
//! `Grid<N, BR, BC>`. Returns `Ok(())` on success, `Err(AssignErr)` on
//! contradiction.

use super::grid::{AssignErr, Grid};

pub fn propagate_singles<const N: usize, const BR: usize, const BC: usize>(
    g: &mut Grid<N, BR, BC>,
) -> Result<(), AssignErr> {
    let nn = N * N;
    let all_mask: u32 = if N >= 32 { u32::MAX } else { (1u32 << N) - 1 };
    // Hold an Arc clone of the peer table so we can iterate `units[u]` while
    // also calling `&mut self` methods on `g` (e.g. `assign`). This avoids the
    // per-iteration `unit_cells.clone()` that previously allocated a Vec<u16>
    // every pass through every unit — the dominant alloc cost in the
    // generic propagator hot loop.
    let table = g.table_arc();
    loop {
        let mut changed = false;
        // Naked singles
        for c in 0..nn {
            if g.solved[c] != 0 {
                continue;
            }
            let m = g.candidates[c];
            if m == 0 {
                return Err(AssignErr::Contradiction);
            }
            if m.count_ones() == 1 {
                let d = m.trailing_zeros() as u8 + 1;
                g.assign(c, d)?;
                changed = true;
            }
        }
        // Hidden singles per unit. We borrow unit slices through `table`
        // (Arc-shared, never mutated) so `g` stays free for `assign`.
        let n_units = 3 * N;
        for u in 0..n_units {
            let unit_cells: &[u16] = &table.units[u];
            let mut once: u32 = 0;
            let mut more: u32 = 0;
            let mut placed: u32 = 0;
            for &c in unit_cells {
                let c = c as usize;
                if g.solved[c] != 0 {
                    placed |= 1u32 << (g.solved[c] - 1);
                    continue;
                }
                let m = g.candidates[c];
                more |= once & m;
                once |= m;
            }
            let unique = once & !more & !placed & all_mask;
            if unique == 0 {
                continue;
            }
            let mut bits = unique;
            while bits != 0 {
                let bit = bits & bits.wrapping_neg();
                bits ^= bit;
                let d = bit.trailing_zeros() as u8 + 1;
                for &c in unit_cells {
                    let c = c as usize;
                    if g.solved[c] == 0 && (g.candidates[c] & bit) != 0 {
                        g.assign(c, d)?;
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn solve_easy_9x9_via_singles() {
        // A puzzle that propagate_singles solves to completion (singles only).
        let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79";
        let mut g: Grid<9, 3, 3> = Grid::from_str(p).unwrap();
        propagate_singles(&mut g).expect("no contradiction");
        assert!(g.is_solved(), "easy 9×9 must solve via singles propagation");
    }

    /// 6×6 puzzle (BR=2, BC=3, digits 1..6) with a unique solution.
    /// Constructed by hand: solution grid:
    ///   1 2 3 | 4 5 6
    ///   4 5 6 | 1 2 3
    ///   ------+------
    ///   2 1 4 | 3 6 5
    ///   3 6 5 | 2 1 4
    ///   ------+------
    ///   5 3 1 | 6 4 2
    ///   6 4 2 | 5 3 1
    /// Givens: enough that singles propagation alone drives to solution.
    #[test]
    fn solve_6x6_via_singles_or_search() {
        // Most cells given, leaves a few empty that singles will fill.
        // We give the full solution minus a handful of cells.
        let p = "12345.\
                 456123\
                 21436.\
                 365214\
                 5316.2\
                 642531";
        let mut g: Grid<6, 2, 3> = Grid::from_str(p).unwrap();
        // singles should solve trivially
        propagate_singles(&mut g).expect("ok");
        assert!(g.is_solved(), "expected fully solved, got {}", g.to_string_grid());
        assert!(g.is_consistent());
        assert_eq!(
            g.to_string_grid(),
            "123456456123214365365214531642642531"
        );
    }

    /// 12×12 puzzle (BR=3, BC=4, digits 1..C). We hand-construct a valid
    /// solution and use it as a near-complete puzzle.
    #[test]
    fn solve_12x12_via_singles() {
        // Construct a valid 12×12 sudoku (BR=3, BC=4). Use a simple algebraic
        // construction: cell (r,c) = ((r * BC) + (r / BR) + c) mod N + 1, but
        // adjusted to satisfy box constraint.
        //
        // Standard construction: value(r,c) = ((BR*r + r/BC + c) mod N) + 1.
        // This produces a valid sudoku for any (BR,BC). Let's use that.
        let n = 12usize;
        let br = 3usize;
        let bc = 4usize;
        let mut sol = vec![0u8; n * n];
        for r in 0..n {
            for c in 0..n {
                let v = ((bc * r + r / br + c) % n) + 1;
                sol[r * n + c] = v as u8;
            }
        }
        // Encode full solution as string.
        let mut s = String::new();
        for &d in &sol {
            if d <= 9 {
                s.push((b'0' + d) as char);
            } else {
                s.push((b'A' + d - 10) as char);
            }
        }
        let g: Grid<12, 3, 4> = Grid::from_str(&s).unwrap();
        assert!(g.is_consistent(), "constructed solution must be consistent");
        assert!(g.is_solved());
        // Now blank out a few cells and re-solve via singles.
        let mut s2: Vec<u8> = s.bytes().collect();
        for &i in &[0usize, 13, 26, 50, 100, 130] {
            s2[i] = b'.';
        }
        let s2 = std::str::from_utf8(&s2).unwrap();
        let mut g2: Grid<12, 3, 4> = Grid::from_str(s2).unwrap();
        propagate_singles(&mut g2).expect("ok");
        assert!(g2.is_solved(), "singles should fill in 6 missing cells");
        assert_eq!(g2.to_string_grid(), s);
    }

    /// 16×16 puzzle (BR=4, BC=4). Use the same algebraic construction.
    #[test]
    fn solve_16x16_via_singles() {
        let n = 16usize;
        let br = 4usize;
        let bc = 4usize;
        let mut sol = vec![0u8; n * n];
        for r in 0..n {
            for c in 0..n {
                let v = ((bc * r + r / br + c) % n) + 1;
                sol[r * n + c] = v as u8;
            }
        }
        let mut s = String::new();
        for &d in &sol {
            if d <= 9 {
                s.push((b'0' + d) as char);
            } else {
                s.push((b'A' + d - 10) as char);
            }
        }
        let g: Grid<16, 4, 4> = Grid::from_str(&s).unwrap();
        assert!(g.is_consistent());
        assert!(g.is_solved());
        // Blank a few cells, ensure singles refill.
        let mut s2: Vec<u8> = s.bytes().collect();
        for &i in &[0usize, 17, 34, 51, 100, 200, 250] {
            s2[i] = b'.';
        }
        let s2 = std::str::from_utf8(&s2).unwrap();
        let mut g2: Grid<16, 4, 4> = Grid::from_str(s2).unwrap();
        propagate_singles(&mut g2).expect("ok");
        assert!(g2.is_solved());
        assert_eq!(g2.to_string_grid(), s);
    }
}
