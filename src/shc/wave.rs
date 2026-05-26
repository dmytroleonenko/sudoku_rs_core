//! Phase 2: wave driver — `rate_b`.
//!
//! Port of `Braid.appliquer` + `Jeu.chercher_cand_ctr` from
//! `/tmp/shc_decompiled/SHC/SHC/Braid.java` and `Jeu.java`. Implements the
//! Rating_inf_TE1 cascade: for i = 0,1,2,...,max_length, try to drive the
//! puzzle to a solved state using only chains of length ≤ i. The smallest
//! `i` that succeeds is the B-rating.

use super::board::{ApplyOutcome, Board};
use super::braid_engine::{tuple_ctr, BraidArena, BraidResult};
use super::tb::{propagate, propagate_counted, TbLevel};

/// Outcome of an inner solve attempt (used by TE.f hypothesis tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InnerOutcome {
    /// Inner solver drove the board to a complete solution.
    Solved,
    /// Inner solver hit a contradiction (Java 'F'). For TE, this is the
    /// signal that the hypothesis was wrong → the originating candidate
    /// must be eliminated from the outer board.
    Contradiction,
    /// Inner solver ran to quiescence without solving or contradicting
    /// (Java 'I').
    Incomplete,
    /// Buffer too small inside the braid stage at depth ≥ 2.
    BufferOverflow,
}

/// Inner solver dispatcher mirroring Java's `appliquer_regles_resolution`:
///   * `n == 0` → TB(L0) only
///   * `n == 1` → TB(L1) only
///   * `n ≥ 2`  → wave at chain-length budget `n` (= Braid.appliquer with
///     nivmax = n - 1)
///
/// `arena` and `packed_scratch` are reusable scratch passed in from the
/// caller. `board` is mutated in place to the post-propagation state.
pub fn solve_inner(
    board: &mut Board,
    inner_level: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
    packed_scratch: &mut Vec<u32>,
) -> InnerOutcome {
    match inner_level {
        0 => match propagate(board, TbLevel::L0) {
            ApplyOutcome::Solved => InnerOutcome::Solved,
            ApplyOutcome::Contradiction => InnerOutcome::Contradiction,
            ApplyOutcome::Continue => InnerOutcome::Incomplete,
        },
        1 => match propagate(board, TbLevel::L1) {
            ApplyOutcome::Solved => InnerOutcome::Solved,
            ApplyOutcome::Contradiction => InnerOutcome::Contradiction,
            ApplyOutcome::Continue => InnerOutcome::Incomplete,
        },
        n => match wave_loop(board, n, buffer_size, arena, packed_scratch) {
            Ok(WaveOutcome::Solved) => InnerOutcome::Solved,
            Ok(WaveOutcome::Stuck) => InnerOutcome::Incomplete,
            Err(RateError::Malformed) => InnerOutcome::Contradiction,
            Err(RateError::BufferOverflow) => InnerOutcome::BufferOverflow,
            // Unclassifiable cannot happen inside wave_loop (only the outer
            // cascade returns it). Treat defensively as Incomplete.
            Err(RateError::Unclassifiable) => InnerOutcome::Incomplete,
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateError {
    /// Puzzle malformed: TB itself reaches contradiction from the input.
    Malformed,
    /// Cannot classify within `max_length` (Java's `-4`).
    Unclassifiable,
    /// Buffer too small (Java's `-3`).
    BufferOverflow,
}

/// Compute the B-rating of `board`. Returns `Ok(b)` where b in 0..=max_length.
///
/// The board is mutated in place during rating (mirrors Java behaviour of
/// solving it as a side-effect). Pass a clone if you want to preserve.
pub fn rate_b(board: &mut Board, max_length: u16, buffer_size: usize) -> Result<u16, RateError> {
    // Mirror Rating_inf_TE1.un_seul (Rating_inf_TE1.java:35-62) exactly:
    //   i = 0 → run TB(L0) once; if Solved → B=0.
    //   i = 1 → run TB(L1) once; if Solved → B=1.
    //   i = 2..=max_length → wave loop with nivmax = i - 1.
    // Each iteration restarts from a fresh clone of the original board (Java
    // re-parses input each call; our `un_seul` analog clones).

    // i = 0: TB(L0) only.
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

    // i = 1: TB(L1) only (no chain search).
    {
        let mut work = board.clone();
        match propagate(&mut work, TbLevel::L1) {
            ApplyOutcome::Solved => {
                *board = work;
                return Ok(1);
            }
            ApplyOutcome::Contradiction => return Err(RateError::Malformed),
            ApplyOutcome::Continue => {}
        }
    }

    // i ≥ 2: wave loop with chain-length budget i, nivmax = i - 1.
    // Allocate ONE arena for the whole puzzle: the same `(buffer_size,
    // max_length)` stride is used for every probe at every i. This is the
    // load-bearing reuse — saves ~2 × (buffer_size * (max_length+1) * 2 B)
    // of zero-init per probe.
    let mut arena = BraidArena::new(buffer_size, max_length);
    // Reusable scratch for `chercher_cand_ctr` priority keys.
    let mut packed_scratch: Vec<u32> = Vec::with_capacity(256);
    for i in 2..=max_length {
        let mut work = board.clone();
        match wave_loop(
            &mut work,
            i,
            buffer_size,
            &mut arena,
            &mut packed_scratch,
        )? {
            WaveOutcome::Solved => {
                *board = work;
                return Ok(i);
            }
            WaveOutcome::Stuck => continue,
        }
    }
    Err(RateError::Unclassifiable)
}

pub(super) enum WaveOutcome {
    Solved,
    Stuck,
}

/// Inner wave loop at fixed chain-length budget `i`. Mirrors
/// `Braid.appliquer`: TB to quiescence; if stuck, find a candidate whose
/// elimination is provable by a chain of length ≤ i; eliminate; repeat.
pub(super) fn wave_loop(
    board: &mut Board,
    max_chain_len: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
    packed_scratch: &mut Vec<u32>,
) -> Result<WaveOutcome, RateError> {
    let nivmax = max_chain_len.saturating_sub(1); // Java: nivmax = n - 1
    let mut buffer_saturated_anywhere = false;

    loop {
        match propagate(board, TbLevel::L1) {
            ApplyOutcome::Solved => return Ok(WaveOutcome::Solved),
            ApplyOutcome::Contradiction => return Err(RateError::Malformed),
            ApplyOutcome::Continue => {}
        }
        let targets = match chercher_cand_ctr(board, packed_scratch) {
            Some(t) => t,
            None => {
                if buffer_saturated_anywhere {
                    return Err(RateError::BufferOverflow);
                }
                return Ok(WaveOutcome::Stuck);
            }
        };
        let mut applied = false;
        for (target_code, _priority) in targets {
            match tuple_ctr(board, target_code, nivmax, buffer_size, arena) {
                BraidResult::Found(proof) => {
                    // Eliminate target candidate.
                    let outcome = board.eliminate(proof.target_cell, proof.target_digit);
                    if outcome == ApplyOutcome::Contradiction {
                        return Err(RateError::Malformed);
                    }
                    applied = true;
                    break;
                }
                BraidResult::NotFound => continue,
                BraidResult::BufferOverflow => {
                    buffer_saturated_anywhere = true;
                    continue;
                }
            }
        }
        if !applied {
            if buffer_saturated_anywhere {
                return Err(RateError::BufferOverflow);
            }
            return Ok(WaveOutcome::Stuck);
        }
    }
}

/// Port of `Jeu.chercher_cand_ctr` (Jeu.java:200-230). For each remaining
/// candidate `(cell, digit)`, test: clone, ASSIGN the candidate, run
/// `TB.appliquer(tb_niv0)`; if it returns 'F' (contradiction), this candidate
/// is provably eliminable. Java packs `nombre*1000 + cell*10 + digit` and
/// sorts ascending via `X.trier`. Here `nombre` is the count of TB rule
/// firings during the probe (TB.java:53), NOT just assignments — box/line
/// eliminations count too. The wave caller iterates from the end backwards,
/// i.e. highest-`nombre` first (most informative probe = most likely to
/// admit a short proof).
///
/// We pre-reverse so the existing caller's forward iteration matches Java's
/// reverse iteration. Tie-break: cell*10+digit ascending in Java's order, so
/// in our reversed view it appears as cell*10+digit descending for equal
/// `nombre`.
///
/// `packed_scratch` is reused across calls to avoid per-call allocation of
/// the priority Vec.
///
/// Returns `(code, priority)` pairs in the order the caller should try them
/// (highest priority first).
fn chercher_cand_ctr(board: &Board, packed_scratch: &mut Vec<u32>) -> Option<Vec<(u16, u32)>> {
    // Packed key = nombre * 1000 + cell * 10 + digit (matches Jeu.java:212).
    // Java cell range 1..=81, digit 1..=9 ⇒ cell*10+digit < 1000. Our 0..=80
    // / 0..=8 fits the same envelope.
    packed_scratch.clear();
    for cell in 0..81u8 {
        let c = &board.cells[cell as usize];
        if c.assigned.is_some() {
            continue;
        }
        let mut mask = c.cand_mask;
        while mask != 0 {
            let digit = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            let mut probe = board.clone();
            // Java's TB.appliquer starts with `nombre = 0`, then increments
            // once per rule firing AFTER applying the initial valider(n, n2).
            // The initial assign is NOT counted in `nombre`. We match that.
            let assign_outcome = probe.assign(cell, digit);
            let (contradicts, nombre) = match assign_outcome {
                ApplyOutcome::Contradiction => (true, 0u32),
                // Java's TB.appliquer returns 'S' (solved) here, not 'F';
                // chercher_cand_ctr only collects 'F' cases. Skip.
                ApplyOutcome::Solved => (false, 0u32),
                ApplyOutcome::Continue => {
                    let (outcome, n) = propagate_counted(&mut probe, TbLevel::L0);
                    (outcome == ApplyOutcome::Contradiction, n)
                }
            };
            if !contradicts {
                continue;
            }
            let key: u32 = nombre
                .saturating_mul(1000)
                .saturating_add((cell as u32) * 10 + digit as u32);
            packed_scratch.push(key);
        }
    }
    if packed_scratch.is_empty() {
        return None;
    }
    // Match Java's `X.trier`: ascending sort, then iterate from end. We sort
    // ascending and then reverse so the returned Vec is already in
    // "highest-nombre first" order, matching Braid.java:41's forward loop
    // intent over Java's pre-reversed view.
    packed_scratch.sort();
    packed_scratch.reverse();
    let out: Vec<(u16, u32)> = packed_scratch
        .iter()
        .copied()
        .map(|k| {
            let nombre = k / 1000;
            let cd = (k % 1000) as u16;
            let cell = (cd / 10) as u8;
            let digit = (cd % 10) as u8;
            (super::braid_engine::encode(cell, digit), nombre)
        })
        .collect();
    Some(out)
}

