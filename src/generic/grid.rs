//! Generic `Grid<N, BR, BC>`.
//!
//! Mirrors the semantics of `crate::grid::Grid` (9×9) but parameterised over
//! arbitrary block dimensions. Storage is heap-allocated `Vec<...>` of length
//! `N*N`; on stable Rust we can't write `[T; N*N]` field types without
//! `generic_const_exprs`.
//!
//! Candidate masks: `u32`, with bit `(d-1)` set when digit `d` (1-based) is
//! still possible. `ALL_DIGITS_MASK` = (1 << N) - 1.

use std::sync::Arc;

use super::peer_tables::{self, PeerTable};

#[derive(Debug, Clone, Copy)]
pub enum AssignErr {
    Contradiction,
}

/// `#[repr(C)]` guarantees a stable, language-specified field layout for all
/// const-generic instantiations of `Grid<N,BR,BC>`. This is required for the
/// `unsafe` pointer-cast in `chain_rated.rs` that transmutes `Grid<N,BR,BC>`
/// to `Grid<9,3,3>` after a runtime N==9 check. Without `repr(C)`, the compiler
/// is free to reorder fields across monomorphizations, making the cast UB.
/// (CR-FIN-M1o fix.)
#[repr(C)]
#[derive(Clone)]
pub struct Grid<const N: usize, const BR: usize, const BC: usize> {
    /// Per-cell digit mask. `candidates[i]` bit `(d-1)` set => digit d possible.
    pub candidates: Vec<u32>,
    /// Per-cell solved digit (0 = empty, else 1..=N).
    pub solved: Vec<u8>,
    pub solved_count: u32,
    /// Cached pointer to peer tables for this `(N, BR, BC)`.
    table: Arc<PeerTable>,
}

impl<const N: usize, const BR: usize, const BC: usize> Grid<N, BR, BC> {
    #[inline]
    pub fn all_digits_mask() -> u32 {
        // N<=32 for our targets (max 16). Use shift; for N==32 would overflow,
        // but we never instantiate that.
        debug_assert!(N <= 31, "N must be ≤ 31 for u32 mask storage");
        (1u32 << N) - 1
    }

    pub fn empty() -> Self {
        debug_assert_eq!(BR * BC, N, "BR*BC must equal N");
        let nn = N * N;
        let mask = Self::all_digits_mask();
        Grid {
            candidates: vec![mask; nn],
            solved: vec![0u8; nn],
            solved_count: 0,
            table: peer_tables::get(N, BR, BC),
        }
    }

    /// Parse a string of length N*N. '.' or '0' = empty. Digits use:
    ///   - `'1'..='9'` for digits 1..=9
    ///   - `'A'..` (uppercase) for 10, 11, ... (A=10, B=11, …, G=16)
    ///   - lowercase also accepted for symmetry.
    /// Returns None on parse error or contradiction.
    pub fn from_str(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        let nn = N * N;
        if bytes.len() < nn {
            return None;
        }
        let mut g = Grid::<N, BR, BC>::empty();
        for i in 0..nn {
            let b = bytes[i];
            if b == b'.' || b == b'0' {
                continue;
            }
            let d = if (b'1'..=b'9').contains(&b) {
                b - b'0'
            } else if (b'A'..=b'Z').contains(&b) {
                10 + (b - b'A')
            } else if (b'a'..=b'z').contains(&b) {
                10 + (b - b'a')
            } else {
                return None;
            };
            if d == 0 || (d as usize) > N {
                return None;
            }
            if g.assign(i, d).is_err() {
                return None;
            }
        }
        Some(g)
    }

    /// Encode current solved state to a string of length N*N. Empty -> '.'.
    /// Digits 1..=9 use '1'..='9'; 10..=N use 'A'..
    pub fn to_string_grid(&self) -> String {
        let nn = N * N;
        let mut s = String::with_capacity(nn);
        for i in 0..nn {
            let d = self.solved[i];
            if d == 0 {
                s.push('.');
            } else if d <= 9 {
                s.push((b'0' + d) as char);
            } else {
                s.push((b'A' + (d - 10)) as char);
            }
        }
        s
    }

    #[inline(always)]
    pub fn is_solved(&self) -> bool {
        self.solved_count as usize == N * N
    }

    #[inline(always)]
    pub fn pop(&self, cell: usize) -> u32 {
        self.candidates[cell].count_ones()
    }

    #[inline(always)]
    pub fn table(&self) -> &PeerTable {
        &self.table
    }

    /// Cheap Arc clone of the peer-table reference. Useful when callers need to
    /// iterate the table while also holding a `&mut Grid` (e.g. propagator and
    /// technique loops).
    #[inline(always)]
    pub fn table_arc(&self) -> Arc<PeerTable> {
        self.table.clone()
    }

    /// Place digit d (1..=N) at `cell`. Eliminates d from all peer candidate
    /// sets. Returns Err on contradiction (digit not in cell's current mask,
    /// or a peer's mask becomes empty, or a solved peer already had d).
    #[inline]
    pub fn assign(&mut self, cell: usize, d: u8) -> Result<(), AssignErr> {
        let bit = 1u32 << (d - 1);
        if self.candidates[cell] & bit == 0 {
            return Err(AssignErr::Contradiction);
        }
        self.candidates[cell] = bit;
        self.solved[cell] = d;
        self.solved_count += 1;
        let nbit = !bit;
        // Iterate peers via cached vector.
        // SAFETY: we don't mutate the peer-table; just read.
        let peers_ptr: *const u16 = self.table.cells[cell].peers.as_ptr();
        let n_peers = self.table.cells[cell].peers.len();
        // Use unsafe pointer iteration to avoid borrow conflict with self.
        for i in 0..n_peers {
            let p = unsafe { *peers_ptr.add(i) } as usize;
            if self.solved[p] == 0 {
                self.candidates[p] &= nbit;
                if self.candidates[p] == 0 {
                    return Err(AssignErr::Contradiction);
                }
            } else if self.candidates[p] == bit {
                // peer already placed this same digit — contradiction
                return Err(AssignErr::Contradiction);
            }
        }
        Ok(())
    }

    /// Eliminate digit d (1..=N) from cell's candidate set.
    /// Returns Ok(true) if changed, Ok(false) if no-op, Err on contradiction.
    #[inline]
    pub fn eliminate(&mut self, cell: usize, d: u8) -> Result<bool, AssignErr> {
        let bit = 1u32 << (d - 1);
        if self.candidates[cell] & bit == 0 {
            return Ok(false);
        }
        if self.solved[cell] != 0 {
            return Err(AssignErr::Contradiction);
        }
        self.candidates[cell] &= !bit;
        if self.candidates[cell] == 0 {
            return Err(AssignErr::Contradiction);
        }
        Ok(true)
    }

    /// Verify that no unit has the same solved digit twice.
    pub fn is_consistent(&self) -> bool {
        for u in &self.table.units {
            let mut seen: u32 = 0;
            for &c in u {
                let d = self.solved[c as usize];
                if d == 0 {
                    continue;
                }
                let bit = 1u32 << (d - 1);
                if seen & bit != 0 {
                    return false;
                }
                seen |= bit;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_9x9_has_full_masks() {
        let g: Grid<9, 3, 3> = Grid::empty();
        assert_eq!(g.candidates.len(), 81);
        assert_eq!(g.solved.len(), 81);
        for i in 0..81 {
            assert_eq!(g.candidates[i], 0x1FF);
        }
        assert_eq!(g.solved_count, 0);
    }

    #[test]
    fn assign_propagates_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        g.assign(0, 5).unwrap();
        // Same row
        assert_eq!(g.candidates[1] & 0b1_0000, 0);
        // Same col
        assert_eq!(g.candidates[9] & 0b1_0000, 0);
        // Same box
        assert_eq!(g.candidates[10] & 0b1_0000, 0);
        // Not a peer
        assert_eq!(g.candidates[40], 0x1FF);
    }

    #[test]
    fn empty_6x6_full_mask() {
        let g: Grid<6, 2, 3> = Grid::empty();
        assert_eq!(g.candidates.len(), 36);
        for i in 0..36 {
            assert_eq!(g.candidates[i], 0x3F);
        }
    }

    #[test]
    fn empty_16x16_full_mask() {
        let g: Grid<16, 4, 4> = Grid::empty();
        assert_eq!(g.candidates.len(), 256);
        assert_eq!(g.candidates[0], 0xFFFF);
    }

    #[test]
    fn empty_12x12_full_mask() {
        let g: Grid<12, 3, 4> = Grid::empty();
        assert_eq!(g.candidates.len(), 144);
        assert_eq!(g.candidates[0], 0xFFF);
    }

    #[test]
    fn assign_contradiction_in_row() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        g.assign(0, 5).unwrap();
        // Trying to put 5 in cell 1 (same row) must fail because peer
        // propagation already removed the candidate.
        assert!(g.assign(1, 5).is_err());
    }
}
