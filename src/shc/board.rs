//! Board state for the SHC clean-room port.
//!
//! Layout differences vs the Java reference (`Jeu`, `Cellule`, `Region` in
//! `/tmp/shc_decompiled/SHC/SHC/`):
//!   * everything is zero-indexed (cells 0..81, digits 0..9, regions 0..27);
//!   * `assign` / `eliminate` return `ApplyOutcome` instead of mutating a
//!     global `TB.stop_inco` string — callers must short-circuit on
//!     `Contradiction` (the Java code's failure to do so is exactly the
//!     buffer-overflow mode mentioned in the brief).
//!
//! Region indexing convention: rows 0..9, cols 9..18, boxes 18..27.
//!
//! Candidate representation: each [`Cell`] stores a 9-bit mask
//! (`cand_mask: u16`, bit `d` set ⇔ digit `d` still possible). This replaces
//! the original `[Candidate; 9]` array of booleans and unlocks popcount /
//! trailing_zeros for the hot loops in `tb.rs` and `braid_engine.rs`.

use super::error::BoardError;

/// 9-bit mask covering digits 0..=8 (bits 0..=8).
pub const ALL_DIGITS_MASK: u16 = 0x1FF;

#[derive(Debug, Clone, Copy)]
pub struct Cell {
    pub assigned: Option<u8>,
    /// Bit `d` set ⇔ digit `d` (0-indexed) is still a candidate for this cell.
    pub cand_mask: u16,
    /// (row_region, col_region, box_region) absolute indices into the 27-region table.
    pub regions: [u8; 3],
}

impl Cell {
    fn empty(regions: [u8; 3]) -> Self {
        Cell {
            assigned: None,
            cand_mask: ALL_DIGITS_MASK,
            regions,
        }
    }

    /// Number of candidates still set (popcount of `cand_mask`).
    #[inline]
    pub fn n_candidates(&self) -> u8 {
        self.cand_mask.count_ones() as u8
    }

    /// Whether digit `d` (0..9) is still a candidate.
    #[inline]
    pub fn has_candidate(&self, d: u8) -> bool {
        (self.cand_mask >> d) & 1 == 1
    }

    /// First-existing-candidate digit (0..9), or `None`.
    #[inline]
    pub fn first_candidate(&self) -> Option<u8> {
        if self.cand_mask == 0 {
            None
        } else {
            Some(self.cand_mask.trailing_zeros() as u8)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Region {
    /// Cell indices (0..81) belonging to this region.
    pub cells: [u8; 9],
}

/// Lazily-built static table of the 27 regions (9 rows, 9 cols, 9 boxes).
pub fn regions() -> &'static [Region; 27] {
    use std::sync::OnceLock;
    static R: OnceLock<[Region; 27]> = OnceLock::new();
    R.get_or_init(build_regions)
}

fn build_regions() -> [Region; 27] {
    let mut out = [Region { cells: [0; 9] }; 27];
    // Rows: region 0..9.
    for r in 0..9 {
        for c in 0..9 {
            out[r].cells[c] = (r * 9 + c) as u8;
        }
    }
    // Cols: region 9..18.
    for c in 0..9 {
        for r in 0..9 {
            out[9 + c].cells[r] = (r * 9 + c) as u8;
        }
    }
    // Boxes: region 18..27 (box index = (br*3 + bc), reading order).
    for br in 0..3 {
        for bc in 0..3 {
            let box_id = 18 + br * 3 + bc;
            let mut k = 0;
            for r in (br * 3)..(br * 3 + 3) {
                for c in (bc * 3)..(bc * 3 + 3) {
                    out[box_id].cells[k] = (r * 9 + c) as u8;
                    k += 1;
                }
            }
        }
    }
    out
}

/// 20-peer table per cell (the union of the row, column, and box minus the
/// cell itself). Used by `assign` to fan out single-digit eliminations.
pub fn peers() -> &'static [[u8; 20]; 81] {
    use std::sync::OnceLock;
    static P: OnceLock<[[u8; 20]; 81]> = OnceLock::new();
    P.get_or_init(build_peers)
}

fn build_peers() -> [[u8; 20]; 81] {
    let regs = regions();
    // For each cell we have to know which regions it belongs to. Easiest:
    // recompute the (row,col,box) triple directly.
    let mut out = [[0u8; 20]; 81];
    for cell in 0..81u8 {
        let r = (cell / 9) as usize;
        let c = (cell % 9) as usize;
        let br = r / 3;
        let bc = c / 3;
        let row_reg = r;
        let col_reg = 9 + c;
        let box_reg = 18 + br * 3 + bc;
        let mut seen = [false; 81];
        let mut n = 0usize;
        for &peer in regs[row_reg]
            .cells
            .iter()
            .chain(regs[col_reg].cells.iter())
            .chain(regs[box_reg].cells.iter())
        {
            if peer == cell {
                continue;
            }
            if seen[peer as usize] {
                continue;
            }
            seen[peer as usize] = true;
            out[cell as usize][n] = peer;
            n += 1;
        }
        debug_assert_eq!(n, 20, "cell {} expected 20 peers, got {}", cell, n);
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyOutcome {
    Continue,
    Solved,
    Contradiction,
}

#[derive(Debug, Clone)]
pub struct Board {
    pub cells: [Cell; 81],
    /// `digit_in_region[region][digit]` — set once a digit has been placed in that region.
    pub digit_in_region: [[bool; 9]; 27],
    /// Count of cells in `region` that still carry `digit` as a candidate.
    pub n_cand_in_region: [[u8; 9]; 27],
    /// Opt #3e: count of assigned cells (0..=81). Lets `is_solved` run in O(1)
    /// instead of an 81-cell scan. The Java reference avoids `est_solution` on
    /// the hot per-`valider` path; we matched that by inlining the same check
    /// inside `assign`, but at O(81) cost per call. The counter is maintained
    /// at the unique transition None→Some(digit) inside `assign` (eliminate
    /// never assigns, so it never increments). Cloned with the struct.
    pub n_assigned: u8,
}

impl Board {
    /// Construct an empty board (all candidates present, nothing assigned).
    pub fn empty() -> Self {
        let regs = regions();
        // Compute per-cell region triple from the static table.
        let mut cells = [Cell::empty([0, 0, 0]); 81];
        for (reg_idx, region) in regs.iter().enumerate() {
            // 0..9 -> row, 9..18 -> col, 18..27 -> box.
            let slot = if reg_idx < 9 {
                0
            } else if reg_idx < 18 {
                1
            } else {
                2
            };
            for &cell_id in region.cells.iter() {
                cells[cell_id as usize].regions[slot] = reg_idx as u8;
            }
        }
        Board {
            cells,
            digit_in_region: [[false; 9]; 27],
            n_cand_in_region: [[9; 9]; 27],
            n_assigned: 0,
        }
    }

    /// Parse 81 chars; `.` or `0` mean empty, `1..9` mean clue.
    pub fn from_81_chars(s: &str) -> Result<Board, BoardError> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 81 {
            return Err(BoardError::BadLength(chars.len()));
        }
        let mut board = Board::empty();
        // Stage 1: validate.
        let mut clues: Vec<(u8, u8)> = Vec::with_capacity(81);
        for (i, &c) in chars.iter().enumerate() {
            match c {
                '.' | '0' => {}
                '1'..='9' => clues.push((i as u8, (c as u8 - b'1'))),
                _ => return Err(BoardError::BadChar { index: i, ch: c }),
            }
        }
        // Stage 2: apply clues; any contradiction here = bad puzzle.
        for (cell, digit) in clues {
            match board.assign(cell, digit) {
                ApplyOutcome::Contradiction => return Err(BoardError::InitialContradiction),
                _ => {}
            }
        }
        Ok(board)
    }

    /// Serialize back to 81 chars; empty cells become `.`.
    pub fn to_81_chars(&self) -> String {
        let mut s = String::with_capacity(81);
        for cell in self.cells.iter() {
            match cell.assigned {
                Some(d) => s.push(char::from(b'1' + d)),
                None => s.push('.'),
            }
        }
        s
    }

    /// Whether every cell has been assigned. O(1) via `n_assigned` counter.
    #[inline]
    pub fn is_solved(&self) -> bool {
        self.n_assigned >= 81
    }

    /// Assign `digit` (0..9) to `cell` (0..81). Returns `Contradiction` on
    /// the first detected inconsistency; further operations are still safe
    /// (idempotent) but the caller MUST stop.
    ///
    /// Opt #3d: bounds-check elision via `get_unchecked` on the cells and
    /// region-counter arrays. The compiler can't prove `cell < 81` /
    /// `digit < 9` because both are `u8` from external callers; explicit
    /// debug_asserts + unchecked indexing recovers the perf without losing
    /// safety in debug builds.
    pub fn assign(&mut self, cell: u8, digit: u8) -> ApplyOutcome {
        debug_assert!(cell < 81, "cell {} out of range", cell);
        debug_assert!(digit < 9, "digit {} out of range", digit);
        let cidx = cell as usize;
        let didx = digit as usize;
        let dbit = 1u16 << digit;
        // SAFETY: cidx < 81 (debug_assert above); `cells` has length 81.
        let cell_ref = unsafe { self.cells.get_unchecked_mut(cidx) };
        // If already assigned, sanity-check.
        if let Some(prev) = cell_ref.assigned {
            if prev == digit {
                return ApplyOutcome::Continue;
            }
            return ApplyOutcome::Contradiction;
        }
        // Candidate must still exist.
        if (cell_ref.cand_mask & dbit) == 0 {
            return ApplyOutcome::Contradiction;
        }
        // Stamp assignment.
        cell_ref.assigned = Some(digit);
        // Opt #3e: track assigned count for O(1) is_solved.
        self.n_assigned = self.n_assigned.saturating_add(1);
        // Snapshot regions (3 region indices, each < 27) before we touch
        // `self.digit_in_region` to avoid a double-borrow.
        let regs = cell_ref.regions;
        let other_mask = cell_ref.cand_mask & !dbit;
        // Collapse the cell's candidate mask to just `dbit` up-front.
        cell_ref.cand_mask = dbit;
        for slot in 0..3 {
            // SAFETY: regs[slot] ∈ 0..27 (built by build_regions); didx < 9.
            let r = unsafe { *regs.get_unchecked(slot) } as usize;
            unsafe {
                *self
                    .digit_in_region
                    .get_unchecked_mut(r)
                    .get_unchecked_mut(didx) = true;
            }
        }
        // Walk the set bits of other_mask and decrement region counters; check
        // emptiness invariant per region:digit.
        let mut m = other_mask;
        while m != 0 {
            let d = m.trailing_zeros() as usize;
            m &= m - 1;
            // d ∈ 0..9 (trailing_zeros of a 9-bit mask with at least one bit).
            for slot in 0..3 {
                // SAFETY: regs[slot] ∈ 0..27; d ∈ 0..9.
                let r = unsafe { *regs.get_unchecked(slot) } as usize;
                let n_ref = unsafe {
                    self.n_cand_in_region
                        .get_unchecked_mut(r)
                        .get_unchecked_mut(d)
                };
                *n_ref = n_ref.saturating_sub(1);
                let placed = unsafe {
                    *self
                        .digit_in_region
                        .get_unchecked(r)
                        .get_unchecked(d)
                };
                if !placed && *n_ref == 0 {
                    return ApplyOutcome::Contradiction;
                }
            }
        }
        // Eliminate `digit` from every peer (20 cells = row ∪ col ∪ box minus
        // self) using the cached peer table.
        let p = peers();
        // SAFETY: cidx < 81; PEERS table has 81 rows.
        let peer_list = unsafe { p.get_unchecked(cidx) };
        for i in 0..20 {
            // SAFETY: 0..20 within peer_list length 20.
            let peer = unsafe { *peer_list.get_unchecked(i) };
            // SAFETY: peer ∈ 0..81 (build_peers stores only valid cell ids).
            let peer_cand = unsafe { self.cells.get_unchecked(peer as usize).cand_mask };
            if (peer_cand & dbit) != 0 {
                match self.eliminate(peer, digit) {
                    ApplyOutcome::Contradiction => return ApplyOutcome::Contradiction,
                    _ => {}
                }
            }
        }
        if self.is_solved() {
            ApplyOutcome::Solved
        } else {
            ApplyOutcome::Continue
        }
    }

    /// Remove `digit` from the candidate set of `cell`. Detects empty cell
    /// and empty region:digit. Idempotent.
    ///
    /// Opt #3d: unchecked indexing on the same invariants as `assign`.
    pub fn eliminate(&mut self, cell: u8, digit: u8) -> ApplyOutcome {
        debug_assert!(cell < 81, "cell {} out of range", cell);
        debug_assert!(digit < 9, "digit {} out of range", digit);
        let cidx = cell as usize;
        let didx = digit as usize;
        let dbit = 1u16 << digit;
        // SAFETY: cidx < 81.
        let cell_ref = unsafe { self.cells.get_unchecked_mut(cidx) };
        if (cell_ref.cand_mask & dbit) == 0 {
            return ApplyOutcome::Continue;
        }
        cell_ref.cand_mask &= !dbit;
        let cell_cand_mask = cell_ref.cand_mask;
        let cell_assigned = cell_ref.assigned;
        let regs = cell_ref.regions;
        for slot in 0..3 {
            // SAFETY: regs[slot] ∈ 0..27; didx < 9.
            let r = unsafe { *regs.get_unchecked(slot) } as usize;
            let n_ref = unsafe {
                self.n_cand_in_region
                    .get_unchecked_mut(r)
                    .get_unchecked_mut(didx)
            };
            *n_ref = n_ref.saturating_sub(1);
        }
        // Empty cell?
        if cell_assigned.is_none() && cell_cand_mask == 0 {
            return ApplyOutcome::Contradiction;
        }
        // Empty region:digit (digit not yet placed AND no candidate cell remains)?
        for slot in 0..3 {
            // SAFETY: regs[slot] ∈ 0..27; didx < 9.
            let r = unsafe { *regs.get_unchecked(slot) } as usize;
            let placed = unsafe {
                *self
                    .digit_in_region
                    .get_unchecked(r)
                    .get_unchecked(didx)
            };
            let n = unsafe {
                *self
                    .n_cand_in_region
                    .get_unchecked(r)
                    .get_unchecked(didx)
            };
            if !placed && n == 0 {
                return ApplyOutcome::Contradiction;
            }
        }
        ApplyOutcome::Continue
    }
}
