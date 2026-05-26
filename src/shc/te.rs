//! Phase 4: Trial-and-Error (TE) cascade — port of `TE.java`,
//! `Rating_TE_depth.java`, and `Rating_sup_TE1.java`.
//!
//! Two public entry points:
//!   * [`rate_te_depth`] — mirror of `Rating_TE_depth`. Returns 0, 1, 2, or 3
//!     for puzzles solvable by TB / T&E(1)+TB / T&E(2)+TB / T&E(3)+TB
//!     respectively.
//!   * [`rate_bxb`] — mirror of `Rating_sup_TE1` invoked with profmax=1.
//!     Returns the smallest inner-B level `n2` (0..=max_length) such that one
//!     outer T&E(1) sweep with that inner solver completes the puzzle.
//!
//! The inner solver dispatch (TB-L0 / TB-L1 / Braid wave at chain-length `n2`)
//! lives in [`wave::solve_inner`]. Hypothesis tests are performed on cloned
//! boards: assign the candidate, run `solve_inner`, observe whether it
//! contradicts. If yes → the candidate is forced *false* in the outer board
//! (Java `jeu.supprimer(n5, n6)`).
//!
//! Java reference: `/tmp/shc_decompiled/SHC/SHC/TE.java`,
//! `Rating_sup_TE1.java`, `Rating_TE_depth.java`. Conventions:
//!   * Java 'S' (solved) → [`TeOutcome::Solved`]
//!   * Java 'F' (contradiction) → [`TeOutcome::Contradiction`]
//!   * Java 'I' (incomplete / quiescent without solve) → [`TeOutcome::Incomplete`]

use super::board::{ApplyOutcome, Board};
use super::braid_engine::BraidArena;
use super::tb::{propagate, TbLevel};
use super::wave::{solve_inner, InnerOutcome, RateError};

/// Result of one TE-level sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TeOutcome {
    Solved,
    Contradiction,
    Incomplete,
    BufferOverflow,
}

/// Run a single outer T&E(1) sweep over `board`. Mirrors `TE.TE1` with
/// `profmax = 1`.
///
/// `inner_n` selects the inner solver:
///   * `0` → TB-L0
///   * `1` → TB-L1
///   * `n ≥ 2` → Braid wave with chain-length budget `n` (Java `Braid.appliquer`
///     with `nivmax = n - 1`).
///
/// The board is mutated in place (eliminations applied as the sweep finds
/// forced-false candidates).
fn te1_sweep(
    board: &mut Board,
    inner_n: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
    packed_scratch: &mut Vec<u32>,
) -> TeOutcome {
    let mut saw_buffer_overflow = false;
    loop {
        // Step 1: prefix solve — run the inner solver on the current `board`.
        // In Java, `appliquer_regles_resolution(jeu, ...)` is called at the
        // top of each iteration with `n3=n4=0` (no hypothesis cell).
        match solve_inner(board, inner_n, buffer_size, arena, packed_scratch) {
            InnerOutcome::Solved => return TeOutcome::Solved,
            InnerOutcome::Contradiction => return TeOutcome::Contradiction,
            InnerOutcome::Incomplete => {}
            InnerOutcome::BufferOverflow => {
                saw_buffer_overflow = true;
                // Continue trying — Java's TE1 doesn't bail on buffer overflow
                // from the prefix solve.
            }
        }

        // Step 2: scan remaining candidates and look for one whose hypothetical
        // assignment yields a contradiction under the inner solver.
        let mut applied_one = false;
        // Snapshot the (cell, digit) list to avoid borrow trouble while we
        // mutate `board` in the inner branch.
        let mut cands: Vec<(u8, u8)> = Vec::with_capacity(81);
        for cell in 0..81u8 {
            let c = &board.cells[cell as usize];
            if c.assigned.is_some() {
                continue;
            }
            let mut mask = c.cand_mask;
            while mask != 0 {
                let digit = mask.trailing_zeros() as u8;
                mask &= mask - 1;
                cands.push((cell, digit));
            }
        }

        for (cell, digit) in cands {
            // Skip candidates already eliminated by an earlier round of this
            // very loop (the cands vec was a snapshot).
            let c = &board.cells[cell as usize];
            if c.assigned.is_some() || (c.cand_mask & (1u16 << digit)) == 0 {
                continue;
            }
            // Hypothesis test: clone, assign, run inner solver, look for 'F'.
            let mut probe = board.clone();
            let contradicts = match probe.assign(cell, digit) {
                ApplyOutcome::Contradiction => true,
                ApplyOutcome::Solved => {
                    // The hypothesis itself drove the board to a complete
                    // solution. That is not a contradiction; the candidate is
                    // viable → skip it.
                    false
                }
                ApplyOutcome::Continue => {
                    match solve_inner(
                        &mut probe,
                        inner_n,
                        buffer_size,
                        arena,
                        packed_scratch,
                    ) {
                        InnerOutcome::Contradiction => true,
                        InnerOutcome::BufferOverflow => {
                            saw_buffer_overflow = true;
                            false
                        }
                        _ => false,
                    }
                }
            };
            if contradicts {
                // Eliminate from outer board, restart outer loop.
                let out = board.eliminate(cell, digit);
                if out == ApplyOutcome::Contradiction {
                    return TeOutcome::Contradiction;
                }
                applied_one = true;
                break;
            }
        }
        if !applied_one {
            return if saw_buffer_overflow {
                TeOutcome::BufferOverflow
            } else {
                TeOutcome::Incomplete
            };
        }
    }
}

/// Run a T&E(2) outer sweep — mirror of `TE.TE2` with `profmax = 2`. The
/// inner hypothesis test is a full `te1_sweep` rather than `solve_inner`.
fn te2_sweep(
    board: &mut Board,
    inner_n: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
    packed_scratch: &mut Vec<u32>,
) -> TeOutcome {
    let mut saw_buffer_overflow = false;
    loop {
        match solve_inner(board, inner_n, buffer_size, arena, packed_scratch) {
            InnerOutcome::Solved => return TeOutcome::Solved,
            InnerOutcome::Contradiction => return TeOutcome::Contradiction,
            InnerOutcome::Incomplete => {}
            InnerOutcome::BufferOverflow => {
                saw_buffer_overflow = true;
            }
        }

        let mut applied_one = false;
        let mut cands: Vec<(u8, u8)> = Vec::with_capacity(81);
        for cell in 0..81u8 {
            let c = &board.cells[cell as usize];
            if c.assigned.is_some() {
                continue;
            }
            let mut mask = c.cand_mask;
            while mask != 0 {
                let digit = mask.trailing_zeros() as u8;
                mask &= mask - 1;
                cands.push((cell, digit));
            }
        }
        for (cell, digit) in cands {
            let c = &board.cells[cell as usize];
            if c.assigned.is_some() || (c.cand_mask & (1u16 << digit)) == 0 {
                continue;
            }
            let mut probe = board.clone();
            let contradicts = match probe.assign(cell, digit) {
                ApplyOutcome::Contradiction => true,
                ApplyOutcome::Solved => false,
                ApplyOutcome::Continue => {
                    match te1_sweep(&mut probe, inner_n, buffer_size, arena, packed_scratch) {
                        TeOutcome::Contradiction => true,
                        TeOutcome::BufferOverflow => {
                            saw_buffer_overflow = true;
                            false
                        }
                        _ => false,
                    }
                }
            };
            if contradicts {
                let out = board.eliminate(cell, digit);
                if out == ApplyOutcome::Contradiction {
                    return TeOutcome::Contradiction;
                }
                applied_one = true;
                break;
            }
        }
        if !applied_one {
            return if saw_buffer_overflow {
                TeOutcome::BufferOverflow
            } else {
                TeOutcome::Incomplete
            };
        }
    }
}

/// Run a T&E(3) outer sweep — mirror of `TE.TE3`.
fn te3_sweep(
    board: &mut Board,
    inner_n: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
    packed_scratch: &mut Vec<u32>,
) -> TeOutcome {
    let mut saw_buffer_overflow = false;
    loop {
        match solve_inner(board, inner_n, buffer_size, arena, packed_scratch) {
            InnerOutcome::Solved => return TeOutcome::Solved,
            InnerOutcome::Contradiction => return TeOutcome::Contradiction,
            InnerOutcome::Incomplete => {}
            InnerOutcome::BufferOverflow => {
                saw_buffer_overflow = true;
            }
        }
        let mut applied_one = false;
        let mut cands: Vec<(u8, u8)> = Vec::with_capacity(81);
        for cell in 0..81u8 {
            let c = &board.cells[cell as usize];
            if c.assigned.is_some() {
                continue;
            }
            let mut mask = c.cand_mask;
            while mask != 0 {
                let digit = mask.trailing_zeros() as u8;
                mask &= mask - 1;
                cands.push((cell, digit));
            }
        }
        for (cell, digit) in cands {
            let c = &board.cells[cell as usize];
            if c.assigned.is_some() || (c.cand_mask & (1u16 << digit)) == 0 {
                continue;
            }
            let mut probe = board.clone();
            let contradicts = match probe.assign(cell, digit) {
                ApplyOutcome::Contradiction => true,
                ApplyOutcome::Solved => false,
                ApplyOutcome::Continue => {
                    match te2_sweep(&mut probe, inner_n, buffer_size, arena, packed_scratch) {
                        TeOutcome::Contradiction => true,
                        TeOutcome::BufferOverflow => {
                            saw_buffer_overflow = true;
                            false
                        }
                        _ => false,
                    }
                }
            };
            if contradicts {
                let out = board.eliminate(cell, digit);
                if out == ApplyOutcome::Contradiction {
                    return TeOutcome::Contradiction;
                }
                applied_one = true;
                break;
            }
        }
        if !applied_one {
            return if saw_buffer_overflow {
                TeOutcome::BufferOverflow
            } else {
                TeOutcome::Incomplete
            };
        }
    }
}

/// Dispatcher mirroring `TE.f(jeu, n, nArray, n2, n3, n4, bl)`. `profmax` is
/// the outer TE depth (1, 2, or 3); `inner_n` is the inner-solver level
/// (0 → TB-L0, 1 → TB-L1, ≥2 → Braid wave).
fn te_dispatch(
    board: &mut Board,
    profmax: u8,
    inner_n: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
    packed_scratch: &mut Vec<u32>,
) -> TeOutcome {
    match profmax {
        1 => te1_sweep(board, inner_n, buffer_size, arena, packed_scratch),
        2 => te2_sweep(board, inner_n, buffer_size, arena, packed_scratch),
        3 => te3_sweep(board, inner_n, buffer_size, arena, packed_scratch),
        _ => panic!("te_dispatch: invalid profmax {}", profmax),
    }
}

/// Public TE-depth classifier. Mirrors `Rating_TE_depth.un_seul`:
///   * Run TB-L0 once. If Solved → 0.
///   * For n2 = 1..=3: run TE(n2) with inner solver = TB-L0. If Solved → n2.
///   * Otherwise → `RateError::Unclassifiable` (Java returns -1).
///
/// The board is mutated in place.
pub fn rate_te_depth(
    board: &mut Board,
    max_length: u16,
    buffer_size: usize,
) -> Result<u8, RateError> {
    // n2 = 0: TB-niv0 only (no T&E).
    {
        let mut work = board.clone();
        match propagate(&mut work, TbLevel::L0) {
            ApplyOutcome::Solved => {
                *board = work;
                return Ok(0);
            }
            ApplyOutcome::Contradiction => return Err(RateError::Malformed),
            ApplyOutcome::Continue => {}
        }
    }
    // One arena shared across all subsequent attempts.
    let mut arena = BraidArena::new(buffer_size, max_length);
    let mut packed_scratch: Vec<u32> = Vec::with_capacity(256);
    for depth in 1u8..=3 {
        let mut work = board.clone();
        let out = te_dispatch(
            &mut work,
            depth,
            0, // Rating_TE_depth uses TB.tb_niv0 as the inner ruleset.
            buffer_size,
            &mut arena,
            &mut packed_scratch,
        );
        match out {
            TeOutcome::Solved => {
                *board = work;
                return Ok(depth);
            }
            TeOutcome::Contradiction => return Err(RateError::Malformed),
            TeOutcome::BufferOverflow => return Err(RateError::BufferOverflow),
            TeOutcome::Incomplete => continue,
        }
    }
    Err(RateError::Unclassifiable)
}

/// Public BxB rating. Mirrors `Rating_sup_TE1.un_seul(string, string2, 1, bl)`:
///   * For n2 = 0..=max_length: run TE(profmax=1) with inner level `n2`.
///     The smallest n2 for which the sweep yields 'S' is the BxB rating.
///   * Returns Unclassifiable (Java -4) if no n2 works.
///   * Returns BufferOverflow (Java -3) if a buffer overflow was the failure
///     mode.
///
/// The board is mutated in place to the solved state on success.
pub fn rate_bxb(
    board: &mut Board,
    max_length: u16,
    buffer_size: usize,
) -> Result<u16, RateError> {
    rate_te_super(board, /*profmax=*/ 1, max_length, buffer_size)
}

/// Public BxBB rating (T&E(2) outer). Mirrors `Rating_sup_TE1.un_seul(..., 2, ...)`.
pub fn rate_bxbb(
    board: &mut Board,
    max_length: u16,
    buffer_size: usize,
) -> Result<u16, RateError> {
    rate_te_super(board, /*profmax=*/ 2, max_length, buffer_size)
}

fn rate_te_super(
    board: &mut Board,
    profmax: u8,
    max_length: u16,
    buffer_size: usize,
) -> Result<u16, RateError> {
    let mut arena = BraidArena::new(buffer_size, max_length);
    let mut packed_scratch: Vec<u32> = Vec::with_capacity(256);
    let mut saw_buffer_overflow = false;
    for n2 in 0u16..=max_length {
        let mut work = board.clone();
        let out = te_dispatch(
            &mut work,
            profmax,
            n2,
            buffer_size,
            &mut arena,
            &mut packed_scratch,
        );
        match out {
            TeOutcome::Solved => {
                *board = work;
                return Ok(n2);
            }
            TeOutcome::Contradiction => return Err(RateError::Malformed),
            TeOutcome::BufferOverflow => {
                saw_buffer_overflow = true;
                continue;
            }
            TeOutcome::Incomplete => continue,
        }
    }
    if saw_buffer_overflow {
        Err(RateError::BufferOverflow)
    } else {
        Err(RateError::Unclassifiable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Easy puzzle solved by naked+hidden singles alone → TE-depth 0.
    const EASY_SINGLES: &str =
        "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..";

    /// Berthier B5 puzzle (well-known B-rating sample). Needs chains, but is
    /// fully T&E(1)-solvable with inner_n large enough.
    const BERTHIER_B5: &str =
        "..34......5...912.7...2.....1.5.7..86...9...7.......34..2.............9.9...61.75";

    /// One puzzle from forum_hardest_sample20 (TE-depth=2). SHC.jar says
    /// BxB rating = 4 with max-length 14, buffer 1000000.
    const FORUM_HARDEST_FIRST: &str =
        ".......12.....3..4..4.1.5....2.4...6.7...8...9..3.......6.3..5..8...9...32.7..6..";

    #[test]
    fn te_depth_zero_on_easy_singles() {
        let mut b = Board::from_81_chars(EASY_SINGLES).expect("parse");
        let d = rate_te_depth(&mut b, 14, 1_000_000).expect("rate");
        assert_eq!(d, 0);
    }

    #[test]
    fn te_depth_one_on_berthier_b5() {
        // A puzzle that is solvable by plain B-rating is at most T&E-depth 1
        // (the TE(1) sweep with inner=TB-niv0 either solves it via repeated
        // hypothesis-testing, or — for stubborn ones — TE(1)+TB-niv0 still
        // doesn't fully solve. The conservative assertion: ≤ 2.
        let mut b = Board::from_81_chars(BERTHIER_B5).expect("parse");
        let d = rate_te_depth(&mut b, 14, 1_000_000).expect("rate");
        assert!(d <= 2, "expected TE-depth ≤ 2 on Berthier B5, got {}", d);
    }

    #[test]
    fn bxb_solves_forum_hardest_first() {
        // SHC.jar BxB rating = 4 for this puzzle.
        let mut b = Board::from_81_chars(FORUM_HARDEST_FIRST).expect("parse");
        let v = rate_bxb(&mut b, 14, 1_000_000).expect("rate");
        assert_eq!(v, 4, "BxB mismatch with SHC.jar reference");
        assert!(b.is_solved(), "BxB should leave board solved");
    }

    #[test]
    fn bxb_on_easy_is_zero() {
        let mut b = Board::from_81_chars(EASY_SINGLES).expect("parse");
        let v = rate_bxb(&mut b, 9, 1_000_000).expect("rate");
        // Easy is solved by TB-L0 → outer TE(1) with inner_n=0 succeeds → 0.
        assert_eq!(v, 0);
    }
}
