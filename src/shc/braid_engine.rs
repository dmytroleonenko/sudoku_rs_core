//! Phase 2: partial-braid search engine.
//!
//! Clean-room port of `RLC.tuple_ctr` + `RLC.creer` + `RLC.singles_possibles`
//! from `/tmp/shc_decompiled/SHC/SHC/RLC.java`.
//!
//! A "code" is `(cell, digit)` encoded as `u16 = cell * 9 + digit` (range
//! 0..729). The Java reference uses `cell * 10 + digit_1based`; we choose
//! `cell * 9 + digit_0based` for compactness and zero-indexed digits (matches
//! Phase 1 board layout).
//!
//! The dedup multiset key is `cle = Σ_{(_, d) in tuple} POW10[d]`, with
//! `POW10 = [1, 10, 100, ..., 100_000_000]` — base-10 digit-histogram hash,
//! identical to `RLC.dp` semantics. Equal `cle` ⇒ same digit multiset (which
//! is *not* the full tuple multiset, hence the bucket-level set-equality
//! check on the raw `u16` codes).
//!
//! Performance note: a [`BraidArena`] owns the two ping-pong `LevelBuf`s
//! (each `buffer_size * stride * 2 bytes` of `Vec<u16>` plus the parallel
//! `keys` vector). The wave driver allocates one per puzzle and threads
//! `&mut BraidArena` into every `tuple_ctr` call, so the per-invocation cost
//! collapses to a `count = 0` reset plus a `HashMap::clear()`.

use super::board::{ApplyOutcome, Board};
// Opt #3b: FxHashMap (rustc_hash) over std HashMap. The braid dedup table is
// keyed by u32 (digit-histogram `cle`) and is hit ~1e5-1e6 times per puzzle
// at B=8..9. FxHash for small integer keys is ~3-5× faster than SipHash and
// has no DoS-resistance overhead (irrelevant for in-process hot maps).
use rustc_hash::FxHashMap as HashMap;

/// Encode (cell, digit) -> u16 code.
#[inline]
pub fn encode(cell: u8, digit: u8) -> u16 {
    (cell as u16) * 9 + (digit as u16)
}

#[inline]
pub fn decode(code: u16) -> (u8, u8) {
    ((code / 9) as u8, (code % 9) as u8)
}

/// Base-10 digit-histogram weight for the multiset dedup key. Index by digit (0..9).
const POW10: [u32; 9] = [
    1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000,
];

#[derive(Debug, Clone)]
pub struct EliminationProof {
    pub target_cell: u8,
    pub target_digit: u8,
    pub length: u16,
    /// Full chain (length codes). For debug/audit.
    pub tuple: Vec<u16>,
}

#[derive(Debug, Clone)]
pub enum BraidResult {
    Found(EliminationProof),
    NotFound,
    BufferOverflow,
}

/// Per-level tuple buffer: each entry is the full chain (length k+1 codes)
/// plus its `cle` (digit-histogram key).
#[derive(Clone)]
struct LevelBuf {
    /// Flat storage: tuple `i` occupies `tuples[i * stride .. i * stride + len]`
    /// where stride is `max_len`. `len` is implicit per-level.
    tuples: Vec<u16>,
    keys: Vec<u32>,
    count: usize,
    stride: usize,
    /// Per-level dedup: `cle -> indices-into-this-buf`. Owned here so it can
    /// be cleared (not reallocated) on every `tuple_ctr` invocation.
    dedup: HashMap<u32, Vec<usize>>,
}

impl LevelBuf {
    fn new(capacity: usize, stride: usize) -> Self {
        LevelBuf {
            tuples: vec![0u16; capacity * stride],
            keys: vec![0u32; capacity],
            count: 0,
            stride,
            dedup: HashMap::default(),
        }
    }

    /// Reset to empty without freeing the backing allocations. The dedup
    /// `HashMap` is `.clear()`-ed (entries gone, bucket capacity retained).
    /// The inner `Vec<usize>` per entry is dropped along with the entry — we
    /// could pool those too, but clear() of a small map is already cheap.
    fn reset(&mut self) {
        self.count = 0;
        self.dedup.clear();
    }

    fn tuple_slice(&self, i: usize, len: usize) -> &[u16] {
        let off = i * self.stride;
        &self.tuples[off..off + len]
    }

    fn push(&mut self, tuple: &[u16], len: usize, key: u32) -> Option<usize> {
        if self.count >= self.keys.len() {
            return None; // buffer overflow
        }
        let i = self.count;
        let off = i * self.stride;
        self.tuples[off..off + len].copy_from_slice(&tuple[..len]);
        self.keys[i] = key;
        self.count += 1;
        Some(i)
    }
}

/// Pre-allocated scratch space for repeated [`tuple_ctr`] calls within a
/// single puzzle. Holds two `LevelBuf` instances (ping-pong) and a reusable
/// `Vec<u16>` scratch for `chercher_cas` outputs.
///
/// Hoisting these out of `tuple_ctr` saves the 2 × `buffer_size * stride * 2 B`
/// zero-init that otherwise happens once per probe (B=8..9 puzzles fire
/// thousands of probes).
pub struct BraidArena {
    buf_a: LevelBuf,
    buf_b: LevelBuf,
    /// Buffer for fresh singles surfaced by `chercher_cas` (re-used per probe).
    singles_scratch: Vec<u16>,
    /// Configured at construction; if a later call asks for a larger max_length,
    /// the arena rebuilds itself (rare — max_length is fixed per puzzle).
    capacity: usize,
    stride: usize,
}

impl BraidArena {
    /// `capacity` = max tuples per level (Java's `nbtumax`); `max_length` =
    /// chain-length cap. Allocates ~2 × capacity × (max_length + 1) × 2 B.
    pub fn new(capacity: usize, max_length: u16) -> Self {
        let stride: usize = (max_length as usize + 1).max(1);
        BraidArena {
            buf_a: LevelBuf::new(capacity, stride),
            buf_b: LevelBuf::new(capacity, stride),
            singles_scratch: Vec::with_capacity(16),
            capacity,
            stride,
        }
    }

    /// Resize backing storage if the caller wants a different stride/capacity.
    /// In the current wave driver this never fires inside a puzzle (both
    /// args are fixed for the puzzle), but it guards against external misuse.
    fn ensure(&mut self, capacity: usize, max_length: u16) {
        let stride: usize = (max_length as usize + 1).max(1);
        if stride != self.stride || capacity != self.capacity {
            self.buf_a = LevelBuf::new(capacity, stride);
            self.buf_b = LevelBuf::new(capacity, stride);
            self.capacity = capacity;
            self.stride = stride;
        }
    }
}

/// Multiset-equality on two slices of u16 codes (order-independent).
fn equiv(a: &[u16], b: &[u16]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    // Stack-buffer copies, then sort and compare. Max chain length is small
    // (≤ U.max_length = 9 in practice; we cap higher).
    let mut aa: [u16; 32] = [0; 32];
    let mut bb: [u16; 32] = [0; 32];
    let n = a.len();
    debug_assert!(n <= 32);
    aa[..n].copy_from_slice(a);
    bb[..n].copy_from_slice(b);
    aa[..n].sort_unstable();
    bb[..n].sort_unstable();
    aa[..n] == bb[..n]
}

/// Apply a sequence of codes to a board (no TB propagation between codes —
/// matches `RLC.appliquer_tuple`). Returns Contradiction on the first invalid
/// assignment; Solved if the chain completes the grid; Continue otherwise.
fn apply_tuple(board: &mut Board, codes: &[u16]) -> ApplyOutcome {
    for &c in codes {
        let (cell, digit) = decode(c);
        match board.assign(cell, digit) {
            ApplyOutcome::Contradiction => return ApplyOutcome::Contradiction,
            ApplyOutcome::Solved => return ApplyOutcome::Solved,
            ApplyOutcome::Continue => {}
        }
    }
    ApplyOutcome::Continue
}

/// Enumerate all (cell, digit) singles in `board` — both naked and hidden.
/// Ports `RLC.chercher_1` + `RLC.chercher_2`. Output codes are deduped
/// (naked-single cell is reported once even if it is also a hidden single).
///
/// Writes into `out` (cleared on entry) to avoid per-call allocation.
fn chercher_cas_into(board: &Board, out: &mut Vec<u16>) {
    use super::board::regions;
    out.clear();
    // Naked singles.
    for cidx in 0..81u8 {
        let cell = &board.cells[cidx as usize];
        if cell.assigned.is_some() {
            continue;
        }
        if cell.cand_mask.count_ones() == 1 {
            let d = cell.cand_mask.trailing_zeros() as u8;
            out.push(encode(cidx, d));
        }
    }
    // Hidden singles, skipping cells already in `out` as a naked single.
    let regs = regions();
    for reg_idx in 0..27usize {
        for digit in 0..9u8 {
            if board.digit_in_region[reg_idx][digit as usize] {
                continue;
            }
            if board.n_cand_in_region[reg_idx][digit as usize] != 1 {
                continue;
            }
            let dbit = 1u16 << digit;
            // Locate the single cell.
            for &cell_id in regs[reg_idx].cells.iter() {
                let c = &board.cells[cell_id as usize];
                if c.assigned.is_some() {
                    continue;
                }
                if (c.cand_mask & dbit) == 0 {
                    continue;
                }
                let code = encode(cell_id, digit);
                // Skip if it's already a naked single of that cell.
                if c.cand_mask.count_ones() == 1 {
                    break;
                }
                if !out.contains(&code) {
                    out.push(code);
                }
                break;
            }
        }
    }
}

/// `tuple_ctr` — main entry. Build partial braids starting at `target_code`,
/// up to `max_length` (= Java's `nivmax`), bounded by `buffer_size` per level.
///
/// `arena` MUST have been constructed for the same `(buffer_size, max_length)`
/// — the function asserts/repairs this defensively (`ensure`).
///
/// Returns `Found(proof)` if some chain of length ≤ max_length+1 ends in a
/// contradiction (proving the target candidate must be eliminated).
pub fn tuple_ctr(
    board: &Board,
    target_code: u16,
    max_length: u16,
    buffer_size: usize,
    arena: &mut BraidArena,
) -> BraidResult {
    arena.ensure(buffer_size, max_length);
    let stride: usize = arena.stride;
    debug_assert!(stride <= 32, "chain length cap exceeded — bump equiv() stack buffers");
    let (target_cell, target_digit) = decode(target_code);

    // --- Level 0: just the singleton (target_code). Check immediate contradiction. ---
    let mut work = board.clone();
    let outcome0 = work.assign(target_cell, target_digit);
    if outcome0 == ApplyOutcome::Contradiction {
        return BraidResult::Found(EliminationProof {
            target_cell,
            target_digit,
            length: 1,
            tuple: vec![target_code],
        });
    }
    if max_length == 0 {
        return BraidResult::NotFound;
    }

    // --- Levels 1..max_length: ping-pong buffers from the arena. ---
    arena.buf_a.reset();
    arena.buf_b.reset();

    // Seed buf_a with the singleton tuple (the target_code alone).
    let key0 = POW10[target_digit as usize];
    arena
        .buf_a
        .push(&[target_code], 1, key0)
        .expect("buffer_size >= 1 required");

    // `cur_len` = number of codes per tuple in `src`. Start at 1 (the seed).
    let mut cur_len: usize = 1;
    // Drive both buffers via mutable references threaded through a tag (0=A, 1=B).
    // Keeping them as field refs would force a borrow split; tag-swap is
    // simpler and equally fast (single-cycle branch on `cur_len & 1`).
    let mut use_a_as_src = true;

    while (cur_len as u16) <= max_length {
        // Snapshot src/dst as a pair of disjoint borrows.
        let (src, dst) = if use_a_as_src {
            let (a, b) = (&mut arena.buf_a, &mut arena.buf_b);
            (a as &mut LevelBuf, b as &mut LevelBuf)
        } else {
            let (a, b) = (&mut arena.buf_a, &mut arena.buf_b);
            (b as &mut LevelBuf, a as &mut LevelBuf)
        };
        dst.reset();
        let src_count = src.count;
        if src_count == 0 {
            return BraidResult::NotFound;
        }
        // src.dedup is left alone — only dst.dedup is used to dedup tuples at
        // length `cur_len + 1`.

        for i in 0..src_count {
            // Materialize the i-th source tuple onto a stack array to avoid
            // aliasing src.tuples while we mutate dst (different fields, but
            // both behind the same `arena` borrow).
            let mut codes_i_buf: [u16; 32] = [0; 32];
            {
                let s = src.tuple_slice(i, cur_len);
                codes_i_buf[..cur_len].copy_from_slice(s);
            }
            let codes_i = &codes_i_buf[..cur_len];
            let key_i = src.keys[i];

            // Build the post-tuple board: clone, apply all codes, no TB.
            let mut post = board.clone();
            let apply_outcome = apply_tuple(&mut post, codes_i);
            if apply_outcome == ApplyOutcome::Contradiction
                || apply_outcome == ApplyOutcome::Solved
            {
                continue;
            }
            // Enumerate freshly-emerged singles.
            chercher_cas_into(&post, &mut arena.singles_scratch);
            // We need to iterate singles_scratch by value because we will
            // borrow `arena.singles_scratch` via the arena later — copy out.
            let n_singles = arena.singles_scratch.len();
            let mut singles_local: [u16; 81] = [0; 81];
            // Worst-case singles count is bounded by # unassigned cells ≤ 81.
            debug_assert!(n_singles <= 81);
            singles_local[..n_singles].copy_from_slice(&arena.singles_scratch[..n_singles]);

            for k in 0..n_singles {
                let new_code = singles_local[k];
                // Skip if `new_code` is already in the existing tuple.
                if codes_i.iter().any(|&c| c == new_code) {
                    continue;
                }
                let (nc, nd) = decode(new_code);
                // Build the extended tuple.
                let mut ext: [u16; 32] = [0; 32];
                let new_len = cur_len + 1;
                ext[..cur_len].copy_from_slice(codes_i);
                ext[cur_len] = new_code;
                let new_key = key_i + POW10[nd as usize];

                // Dedup against `dst.dedup`. We need both a read of the bucket
                // (to compare against existing dst.tuples slices) and a later
                // push that mutably borrows `dst`. Borrow checker friendly:
                // do the read scan first, then push, then update dedup.
                let mut duplicate = false;
                if let Some(bucket) = dst.dedup.get(&new_key) {
                    for &j in bucket.iter() {
                        if equiv(dst.tuple_slice(j, new_len), &ext[..new_len]) {
                            duplicate = true;
                            break;
                        }
                    }
                }
                if duplicate {
                    continue;
                }

                // Novel — test the contradiction extension. Apply the new
                // single from the post-tuple board (already propagated under
                // assign-only). If contradiction → we have a proof.
                let mut probe = post.clone();
                match probe.assign(nc, nd) {
                    ApplyOutcome::Contradiction => {
                        return BraidResult::Found(EliminationProof {
                            target_cell,
                            target_digit,
                            length: new_len as u16,
                            tuple: ext[..new_len].to_vec(),
                        });
                    }
                    _ => {}
                }

                // FIXME(shc-port-fix2-deferred): Java checks nbtu[n]==nbtumax before dedup/probe (RLC.java:95); our port overflow-checks at push. Edge case near buffer_size limit only — defer to Phase 5 perf pass.
                // Otherwise, insert into dst buffer for the next iteration.
                let pushed = dst.push(&ext[..new_len], new_len, new_key);
                let Some(idx) = pushed else {
                    return BraidResult::BufferOverflow;
                };
                dst.dedup.entry(new_key).or_default().push(idx);
            }
        }
        // Swap and advance.
        use_a_as_src = !use_a_as_src;
        cur_len += 1;
    }

    BraidResult::NotFound
}
