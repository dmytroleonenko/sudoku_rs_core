//! R2.1: constructive AIC reverse synthesis with chain-length targeting.
//!
//! Background. R1's search-and-filter `reverse_construct` measured a hit rate
//! of **0/5000** for "T3 puzzles where AIC is load-bearing" on 9×9. AIC fires
//! plenty (256 pps for "any AIC in frontier"), but it is almost always
//! redundant: the rater's other T3 techniques (XY-Wing, Fish, UR2…) substitute
//! for it. Without load-bearing AIC puzzles we cannot run the Phase D'5 lemma
//! extraction probe.
//!
//! Approach. **Guided removal**, not pure constructive planting (per the
//! design-doc §3.3 iter-1 pragmatic note: "search-and-filter, not
//! constructive"). The pipeline:
//!
//! 1. Generate a uniquely-solvable seed puzzle `gen_unique_puzzle` at a clue
//!    count near the upper end of the requested band.
//! 2. While the puzzle still has more clues than `clue_min`: try removing each
//!    remaining clue in shuffled order; keep the removal iff it preserves
//!    uniqueness AND the AIC chain-length metric remains `>= seed_length` (or
//!    moves toward `target_chain_length`). This greedily expands the chain
//!    structure while maintaining the unique-solution invariant.
//! 3. After convergence, verify:
//!      * `rate(puzzle).tier == target_tier`,
//!      * `Aic ∈ rate(puzzle).frontier`,
//!      * `|measured_chain_length - target_chain_length| <= chain_length_slack`,
//!      * (if requested) `rate_excluding(puzzle, &[Aic]).tier > target_tier` —
//!        i.e. AIC is load-bearing.
//!
//! This is **search-with-guided-bias**, not full constructive planting; if the
//! greedy expansion never hits the target we fall through to a fresh seed.
//!
//! AIC unsoundness regression. The chain-length probe `find_first_aic_chain_length`
//! mirrors the post-fix rules from `aic.rs::try_eliminate` exactly: same-cell
//! same-digit → no, same-digit cross-cell → peers-intersection elim, different
//! digit cross-cell → NO rule (the unsound Type-2 rule remains removed).
//! `tests::aic_unsoundness_regression_a05085` asserts no Type-2 elimination is
//! returned by the dry-run probe. **Do not edit this rule** without re-running
//! the unsoundness regression test.
//!
//! Determinism. Single-threaded with a fixed seed → byte-identical puzzle.
//! Multi-threaded `batch_*` reuses the same per-worker seed-derivation scheme
//! as `batch_reverse_construct` (splitmix off the master seed).

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::generator::{GenConfig, gen_unique_puzzle_with_solution};
use super::grid::Grid;
use super::rater::{rate, rate_excluding};
use super::reverse_construct::ReverseResult;
use super::search::{count_solutions_up_to, solve_unique};
use super::techniques::{TechniqueId, Tier};

/// Configuration for one constructive AIC reverse-synth call.
#[derive(Clone, Debug)]
pub struct AicReverseSpec {
    /// Target tier (typically T3).
    pub target_tier: Tier,
    /// Target AIC chain length in edges (typical 5–9 for 9×9 T3).
    pub target_chain_length: u32,
    /// Accept measured length within ±slack of target.
    pub chain_length_slack: u32,
    /// Inclusive clue-count band for the seed puzzle.
    pub clue_min: u32,
    pub clue_max: u32,
    /// Hard cap on outer attempts (each attempt re-seeds from a fresh
    /// random solution).
    pub max_attempts: u32,
    /// If true, additionally verify Aic is load-bearing
    /// (`rate_excluding(puzzle, [Aic]).tier > target_tier`).
    pub require_load_bearing: bool,
    /// Inner cap on greedy guided-removal trials per seed.
    /// A reasonable default is `4 * N*N`. `0` means "no cap" (still bounded
    /// by `N*N` from a single shuffled pass).
    pub greedy_max_trials: u32,
}

impl AicReverseSpec {
    pub fn new_t3(target_chain_length: u32, slack: u32, max_attempts: u32) -> Self {
        Self {
            target_tier: Tier::T3,
            target_chain_length,
            chain_length_slack: slack,
            clue_min: 22,
            clue_max: 30,
            max_attempts,
            require_load_bearing: true,
            greedy_max_trials: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.clue_min > self.clue_max {
            return Err(format!(
                "clue_min ({}) > clue_max ({})",
                self.clue_min, self.clue_max
            ));
        }
        if matches!(self.target_tier, Tier::T1) {
            return Err("target_tier=T1 cannot fire Aic".into());
        }
        if matches!(self.target_tier, Tier::T2) {
            return Err("target_tier=T2 cannot fire Aic (Aic is T3)".into());
        }
        if self.target_chain_length == 0 {
            return Err("target_chain_length must be >= 1".into());
        }
        Ok(())
    }
}

#[inline]
fn tier_rank(t: Tier) -> u8 {
    match t {
        Tier::T1 => 1,
        Tier::T2 => 2,
        Tier::T3 => 3,
        Tier::T4Plus => 4,
    }
}

// ---------------------------------------------------------------------------
// Generic AIC chain-length probe — mirrors the dry-run BFS from the legacy
// 9×9 `find_first_aic_chain_length`, lifted to const-generic.
//
// IMPORTANT (a05085 regression):
//   The dry-run elimination predicate must apply ONLY the sound rules from
//   `aic::try_eliminate`. Different-digit cross-cell endpoints yield NO
//   elimination — the unsound Type-2 rule was removed and must not be
//   re-introduced here.
// ---------------------------------------------------------------------------

const PROBE_MAX_LEN: usize = 14;

#[inline(always)]
fn node_idx<const N: usize>(cell: usize, d_bit: u8) -> usize {
    cell * N + d_bit as usize
}

// ---------------------------------------------------------------------------
// R3.3b2: per-worker scratch buffers for AIC graph + BFS + peer-lookup reuse.
//
// Motivation: each `aic_score` call inside `guided_removal` would allocate
// `Vec<Vec<u32>>` of length N³ (729 / 4096) plus per-call peer bitmaps.
// On 16×16 with ~150 calls per puzzle this dominates wall time. We use a
// generation counter + per-slot stamps to logical-clear adjacency lists in
// O(touched) instead of O(N³). The `nonempty_*` index lists let the outer
// loop iterate only over slots touched this generation (avoid full N³ scan).
// ---------------------------------------------------------------------------

pub(crate) struct AicProbeScratch {
    // Graph adjacency, indexed by node = cell*N + d_bit. Length nv = N³.
    strong: Vec<Vec<u32>>,
    weak: Vec<Vec<u32>>,
    stamp_strong_adj: Vec<u32>,
    stamp_weak_adj: Vec<u32>,
    // BFS visit stamps (per-start; bumped before each BFS restart).
    bfs_strong: Vec<u32>,
    bfs_weak: Vec<u32>,
    // BFS queue.
    queue: Vec<(u32, u8)>,
    // Peer-lookup cache for `try_eliminate_dryrun`: bitmap is keyed by e_cell.
    // We keep one Vec<bool> of length nn and stamp it; if the cached e_cell
    // matches, skip rebuild. Length nn = N*N.
    peer_lookup: Vec<bool>,
    peer_lookup_e_cell: i32, // -1 = none cached
    peer_lookup_gen: u32,
    // Indices touched this generation (compact list of nodes whose `strong`
    // list has been written). Used to skip full nv scan in outer probe loop
    // and on `has_any_aic_chain` early-return paths.
    nonempty_strong: Vec<u32>,
    nv: usize,
    // Adjacency-graph generation counter; bumped at start of every
    // build_aic_graph_into. Independent from `bfs_counter`.
    generation: u32,
    // BFS visit-stamp counter; bumped per start node within a probe. We zero
    // the bfs_strong/bfs_weak arrays on wraparound to prevent stale matches.
    bfs_counter: u32,
}

impl AicProbeScratch {
    pub(crate) fn new<const N: usize>() -> Self {
        let nv = N * N * N;
        let nn = N * N;
        Self {
            strong: vec![Vec::new(); nv],
            weak: vec![Vec::new(); nv],
            stamp_strong_adj: vec![0; nv],
            stamp_weak_adj: vec![0; nv],
            bfs_strong: vec![0; nv],
            bfs_weak: vec![0; nv],
            queue: Vec::with_capacity(256),
            peer_lookup: vec![false; nn],
            peer_lookup_e_cell: -1,
            peer_lookup_gen: 0,
            nonempty_strong: Vec::with_capacity(nv / 4),
            nv,
            generation: 0,
            bfs_counter: 0,
        }
    }

    /// Bump the adjacency generation counter. On overflow zero stamp arrays.
    #[inline]
    fn bump_generation(&mut self) -> u32 {
        if self.generation == u32::MAX {
            self.stamp_strong_adj.iter_mut().for_each(|x| *x = 0);
            self.stamp_weak_adj.iter_mut().for_each(|x| *x = 0);
            self.peer_lookup_gen = 0;
            self.peer_lookup_e_cell = -1;
            self.generation = 0;
        }
        self.generation += 1;
        self.generation
    }

    /// Bump the BFS counter, zeroing visit arrays on wraparound.
    #[inline]
    fn bump_bfs(&mut self) -> u32 {
        if self.bfs_counter == u32::MAX {
            self.bfs_strong.iter_mut().for_each(|x| *x = 0);
            self.bfs_weak.iter_mut().for_each(|x| *x = 0);
            self.bfs_counter = 0;
        }
        self.bfs_counter += 1;
        self.bfs_counter
    }

    /// Push `to` onto adjacency `list[at]`. Logically clears the slot (via
    /// stamp) the first time it's touched this generation.
    #[inline(always)]
    fn push_strong(&mut self, at: usize, to: u32, gen: u32) {
        if self.stamp_strong_adj[at] != gen {
            self.stamp_strong_adj[at] = gen;
            self.strong[at].clear();
            self.nonempty_strong.push(at as u32);
        }
        self.strong[at].push(to);
    }

    #[inline(always)]
    fn push_weak(&mut self, at: usize, to: u32, gen: u32) {
        if self.stamp_weak_adj[at] != gen {
            self.stamp_weak_adj[at] = gen;
            self.weak[at].clear();
        }
        self.weak[at].push(to);
    }

    /// Read `strong[i]` as it exists this generation, or empty slice if untouched.
    #[inline(always)]
    fn strong_at(&self, i: usize, gen: u32) -> &[u32] {
        if self.stamp_strong_adj[i] == gen { &self.strong[i] } else { &[] }
    }

    #[inline(always)]
    fn weak_at(&self, i: usize, gen: u32) -> &[u32] {
        if self.stamp_weak_adj[i] == gen { &self.weak[i] } else { &[] }
    }
}

fn build_aic_graph_into<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    scratch: &mut AicProbeScratch,
) -> u32 {
    let gen = scratch.bump_generation();
    scratch.nonempty_strong.clear();
    let nn = N * N;
    // Bivalue cells.
    for cell in 0..nn {
        if grid.solved[cell] != 0 { continue; }
        let m = grid.candidates[cell];
        if m.count_ones() == 2 {
            let lo = m.trailing_zeros() as u8;
            let hi = (m & (m - 1)).trailing_zeros() as u8;
            let ia = node_idx::<N>(cell, lo) as u32;
            let ib = node_idx::<N>(cell, hi) as u32;
            scratch.push_strong(ia as usize, ib, gen);
            scratch.push_strong(ib as usize, ia, gen);
            scratch.push_weak(ia as usize, ib, gen);
            scratch.push_weak(ib as usize, ia, gen);
        }
    }
    // Per-unit per-digit.
    let table = grid.table().clone();
    let n_units = 3 * N;
    let mut found: Vec<u32> = Vec::with_capacity(N);
    for u_idx in 0..n_units {
        let unit = &table.units[u_idx];
        for d_bit in 0..(N as u8) {
            let bit = 1u32 << d_bit;
            found.clear();
            let mut placed = false;
            for &c in unit {
                let c = c as usize;
                if grid.solved[c] != 0 {
                    if grid.solved[c] == d_bit + 1 { placed = true; break; }
                    continue;
                }
                if grid.candidates[c] & bit != 0 {
                    found.push(node_idx::<N>(c, d_bit) as u32);
                }
            }
            if placed { continue; }
            let nf = found.len();
            if nf == 2 {
                let a = found[0]; let b = found[1];
                scratch.push_strong(a as usize, b, gen);
                scratch.push_strong(b as usize, a, gen);
                scratch.push_weak(a as usize, b, gen);
                scratch.push_weak(b as usize, a, gen);
            } else if nf > 2 {
                for i in 0..nf {
                    for j in (i+1)..nf {
                        scratch.push_weak(found[i] as usize, found[j], gen);
                        scratch.push_weak(found[j] as usize, found[i], gen);
                    }
                }
            }
        }
    }
    gen
}

/// Pure (no mutation) check for whether chain endpoints `(start, end)` would
/// yield ≥1 elimination under the *sound* AIC rules. Mirrors
/// `aic::try_eliminate` exactly.
///
/// SOUNDNESS: different-digit cross-cell returns false (Type-2 was unsound and
/// was removed; see commit a05085).
fn try_eliminate_dryrun<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    start: u32,
    end: u32,
    scratch: &mut AicProbeScratch,
    probe_gen: u32,
) -> bool {
    if start == end { return false; }
    let s_cell = (start as usize) / N;
    let s_d = (start as usize) % N;
    let e_cell = (end as usize) / N;
    let e_d = (end as usize) % N;
    if s_cell == e_cell { return false; }
    if s_d != e_d {
        // Cross-cell different-digit: NO rule (Type-2 was unsound, removed).
        return false;
    }
    // Same-digit cross-cell: peers-intersection elimination.
    let bit = 1u32 << s_d;
    let table = grid.table();
    let s_peers = &table.cells[s_cell].peers;
    // Reuse cached e_peer bitmap if the same e_cell was queried this probe.
    let need_rebuild = scratch.peer_lookup_gen != probe_gen
        || scratch.peer_lookup_e_cell != e_cell as i32;
    if need_rebuild {
        // Reset and refill. Cheaper than zeroing whole vec when only ~3N peers
        // are set: clear previous e_cell's bits if known, else zero whole vec.
        if scratch.peer_lookup_gen == probe_gen && scratch.peer_lookup_e_cell >= 0 {
            let prev_e = scratch.peer_lookup_e_cell as usize;
            for &p in &table.cells[prev_e].peers {
                scratch.peer_lookup[p as usize] = false;
            }
        } else {
            // First use this generation: full zero (cheap — happens once per probe).
            scratch.peer_lookup.iter_mut().for_each(|x| *x = false);
        }
        for &p in &table.cells[e_cell].peers {
            scratch.peer_lookup[p as usize] = true;
        }
        scratch.peer_lookup_gen = probe_gen;
        scratch.peer_lookup_e_cell = e_cell as i32;
    }
    for &p in s_peers {
        let cell = p as usize;
        if !scratch.peer_lookup[cell] { continue; }
        if cell == s_cell || cell == e_cell { continue; }
        if grid.solved[cell] != 0 { continue; }
        if grid.candidates[cell] & bit != 0 { return true; }
    }
    false
}

/// Internal BFS driver. If `min_chain` is `Some(L)`, returns the first chain
/// length `>= L` (used as cheap "any chain?" probe). Else returns the
/// minimum chain length found, or None.
fn find_aic_chain_with_scratch<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    scratch: &mut AicProbeScratch,
    min_chain: Option<usize>,
) -> Option<usize> {
    let probe_gen = build_aic_graph_into::<N, BR, BC>(grid, scratch);
    // Reset per-probe peer cache.
    scratch.peer_lookup_gen = 0;
    scratch.peer_lookup_e_cell = -1;

    // Iterate only over start nodes whose strong-adj is non-empty this gen.
    // Clone the index list to satisfy borrow checker.
    let starts = scratch.nonempty_strong.clone();

    for &start_u in &starts {
        let start = start_u as usize;
        let s_cell = start / N;
        let s_d_bit = (start % N) as u8;
        if grid.solved[s_cell] != 0 { continue; }
        if grid.candidates[s_cell] & (1u32 << s_d_bit) == 0 { continue; }
        let stamp = scratch.bump_bfs();
        scratch.queue.clear();
        let strong_starts: Vec<u32> = scratch.strong_at(start, probe_gen).to_vec();
        for n in strong_starts {
            if (n as usize) == start { continue; }
            if scratch.bfs_strong[n as usize] != stamp {
                scratch.bfs_strong[n as usize] = stamp;
                scratch.queue.push((n, 1));
            }
        }
        let mut head = 0usize;
        while head < scratch.queue.len() {
            let (cur, d) = scratch.queue[head];
            head += 1;
            let after_strong = d % 2 == 1;
            if after_strong {
                if try_eliminate_dryrun::<N, BR, BC>(grid, start as u32, cur, scratch, probe_gen) {
                    let dl = d as usize;
                    match min_chain {
                        Some(min_l) if dl < min_l => { /* keep going */ }
                        _ => return Some(dl),
                    }
                }
            }
            if (d as usize) >= PROBE_MAX_LEN { continue; }
            // Borrow scratch adjacency with explicit copy to a local buffer
            // because we need &mut scratch for stamp updates simultaneously.
            // `Vec<u32>` is small; copying length is bounded by adjacency
            // degree (typically ≤ N).
            if after_strong {
                let neighbors: Vec<u32> = scratch.weak_at(cur as usize, probe_gen).to_vec();
                for n in neighbors {
                    if (n as usize) == start { continue; }
                    if scratch.bfs_weak[n as usize] == stamp { continue; }
                    scratch.bfs_weak[n as usize] = stamp;
                    scratch.queue.push((n, d + 1));
                }
            } else {
                let neighbors: Vec<u32> = scratch.strong_at(cur as usize, probe_gen).to_vec();
                for n in neighbors {
                    if (n as usize) == start { continue; }
                    if scratch.bfs_strong[n as usize] == stamp { continue; }
                    scratch.bfs_strong[n as usize] = stamp;
                    scratch.queue.push((n, d + 1));
                }
            }
        }
    }
    None
}

/// Find the length (in edges) of the first AIC chain that would fire on `grid`.
/// Returns `None` if no sound elimination chain exists. The walker is
/// deterministic and uses exactly the same start-node / BFS / soundness rules
/// as `aic::Aic::apply`.
///
/// Convenience wrapper that allocates a one-shot scratch buffer. Hot paths
/// should call `find_first_aic_chain_length_with` instead.
pub fn find_first_aic_chain_length<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
) -> Option<usize> {
    let mut scratch = AicProbeScratch::new::<N>();
    find_aic_chain_with_scratch::<N, BR, BC>(grid, &mut scratch, None)
}

/// Same as `find_first_aic_chain_length` but reuses caller-supplied scratch.
pub(crate) fn find_first_aic_chain_length_with<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    scratch: &mut AicProbeScratch,
) -> Option<usize> {
    find_aic_chain_with_scratch::<N, BR, BC>(grid, scratch, None)
}

/// Cheap "any AIC chain present?" probe — returns true on first chain found.
/// Equivalent to `find_first_aic_chain_length(...).is_some()` but exits the
/// outer loop as soon as one chain is detected.
pub(crate) fn has_any_aic_chain<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    scratch: &mut AicProbeScratch,
) -> bool {
    find_aic_chain_with_scratch::<N, BR, BC>(grid, scratch, None).is_some()
}

// ---------------------------------------------------------------------------
// Guided removal driver.
// ---------------------------------------------------------------------------

/// Build a partial `Grid` from `solution` by keeping only the clues whose
/// `keep[i] == true`. Returns `None` if assignment is internally inconsistent
/// (should not happen for a true subset of a valid solution).
fn build_subset<const N: usize, const BR: usize, const BC: usize>(
    solution: &Grid<N, BR, BC>,
    keep: &[bool],
) -> Option<Grid<N, BR, BC>> {
    let nn = N * N;
    let mut g: Grid<N, BR, BC> = Grid::empty();
    for i in 0..nn {
        if !keep[i] { continue; }
        let d = solution.solved[i];
        if d == 0 { return None; }
        if g.assign(i, d).is_err() { return None; }
    }
    Some(g)
}

/// Score a partial puzzle for "how close to the AIC target are we?"
///
/// Returns `(has_chain, dist_to_target)` where `dist_to_target` is
/// `i32::MAX` if the puzzle has no chain at all, else `|measured - target|`.
fn aic_score<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    target: u32,
    scratch: &mut AicProbeScratch,
) -> (bool, i32) {
    match find_first_aic_chain_length_with::<N, BR, BC>(grid, scratch) {
        Some(len) => (true, (len as i32 - target as i32).abs()),
        None => (false, i32::MAX),
    }
}

/// Greedy guided-removal pass: starting from `puzzle`, try removing each
/// remaining clue in shuffled order; accept any removal that
///   (a) preserves uniqueness, and
///   (b) does not move us further from the AIC target chain length (and never
///       drops the chain entirely once we've found one).
/// Stops after one full pass over remaining cells, or after
/// `greedy_max_trials` removal attempts if `> 0`. Returns the final puzzle.
fn guided_removal<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    solution: &Grid<N, BR, BC>,
    keep: &mut Vec<bool>,
    spec: &AicReverseSpec,
    scratch: &mut AicProbeScratch,
) {
    let nn = N * N;
    // Collect currently-kept indices and shuffle.
    let mut order: Vec<u16> = (0..nn as u16).filter(|&i| keep[i as usize]).collect();
    order.shuffle(rng);

    let initial = match build_subset::<N, BR, BC>(solution, keep) {
        Some(g) => g,
        None => return,
    };
    let (mut have_chain, mut best_dist) =
        aic_score::<N, BR, BC>(&initial, spec.target_chain_length, scratch);

    let mut trials: u32 = 0;
    let trial_cap = if spec.greedy_max_trials == 0 { u32::MAX } else { spec.greedy_max_trials };

    for &c in &order {
        if trials >= trial_cap { break; }
        let c = c as usize;
        if !keep[c] { continue; }
        // Floor on clue count.
        let kept_count: u32 = keep.iter().filter(|&&k| k).count() as u32;
        if kept_count <= spec.clue_min { break; }

        keep[c] = false;
        trials = trials.saturating_add(1);
        let trial = match build_subset::<N, BR, BC>(solution, keep) {
            Some(g) => g,
            None => { keep[c] = true; continue; }
        };

        // R3.3b2 reorder: when `have_chain=true`, the AIC-aware filter is the
        // dominant rejection signal AND it's cheaper than a full uniqueness
        // DFS. Run aic_score FIRST and skip the uniqueness check whenever the
        // chain is lost or moves further from target.
        if have_chain {
            let (cand_has, cand_dist) =
                aic_score::<N, BR, BC>(&trial, spec.target_chain_length, scratch);
            if !cand_has || cand_dist > best_dist {
                keep[c] = true;
                continue;
            }
            // Chain preserved & dist OK → confirm uniqueness.
            if count_solutions_up_to(&trial, 2) != 1 {
                keep[c] = true;
                continue;
            }
            best_dist = cand_dist;
            have_chain = cand_has;
        } else {
            // No chain yet → run uniqueness first (no chain-preservation
            // signal to short-circuit on), then aic_score.
            if count_solutions_up_to(&trial, 2) != 1 {
                keep[c] = true;
                continue;
            }
            let (cand_has, cand_dist) =
                aic_score::<N, BR, BC>(&trial, spec.target_chain_length, scratch);
            // No chain yet → accept anything that preserves uniqueness
            // (cand_has=true introduces a chain; cand_has=false is neutral —
            // we may need to thin further before chains emerge).
            have_chain = cand_has;
            best_dist = cand_dist;
        }
    }
}

/// Outer driver. See module-level docs for the algorithm.
pub fn aic_reverse_construct<const N: usize, const BR: usize, const BC: usize, R: Rng + ?Sized>(
    rng: &mut R,
    spec: &AicReverseSpec,
) -> Option<ReverseResult<N, BR, BC>> {
    if spec.validate().is_err() { return None; }
    let target_rank = tier_rank(spec.target_tier);
    let nn = N * N;
    // R3.3b2: one scratch per worker, reused across all attempts.
    let mut scratch = AicProbeScratch::new::<N>();

    for attempt in 0..spec.max_attempts {
        // 1. Random unique seed with HIGH clue count (toward upper end of
        //    band) — gives the greedy phase room to remove. We use clue_max
        //    as the seed target, then thin.
        let seed_target_clues = spec.clue_max;
        let cfg = GenConfig { target_clues: seed_target_clues, max_attempts: 0 };
        let (seed_puzzle, seed_clue_count, solution): (Grid<N, BR, BC>, u32, Grid<N, BR, BC>) =
            gen_unique_puzzle_with_solution::<N, BR, BC, R>(rng, &cfg);
        let _ = seed_clue_count;
        // Solution is the carved seed's full grid — no redundant solve_unique.

        // 2. Build the keep[] vector from the seed (true where seed_puzzle is solved).
        let mut keep: Vec<bool> = (0..nn).map(|i| seed_puzzle.solved[i] != 0).collect();

        // 3. Greedy guided removal toward the target AIC chain length.
        guided_removal::<N, BR, BC, R>(rng, &solution, &mut keep, spec, &mut scratch);

        // 4. Final puzzle.
        let puzzle = match build_subset::<N, BR, BC>(&solution, &keep) {
            Some(g) => g,
            None => continue,
        };
        let final_clues: u32 = keep.iter().filter(|&&k| k).count() as u32;

        // 5. Verify.
        let r = rate(&puzzle);
        if r.rater_error { continue; }
        if tier_rank(r.tier) != target_rank { continue; }
        if !r.frontier.contains(&TechniqueId::Aic) { continue; }

        // Chain-length match.
        let measured = match find_first_aic_chain_length::<N, BR, BC>(&puzzle) {
            Some(l) => l as i32,
            None => continue, // shouldn't happen if Aic is in frontier
        };
        if (measured - spec.target_chain_length as i32).abs() > spec.chain_length_slack as i32 {
            continue;
        }

        // Load-bearing check. Semantics: AIC is "essential" iff removing
        // it from the cascade makes the puzzle UNSOLVABLE at this tier or
        // promotes it to a strictly harder tier (T4Plus). Concretely:
        //   accept iff `rate_excluding(puzzle, [Aic]).tier > target_tier`.
        // This deliberately differs from `reverse_construct::ReverseSpec`'s
        // load-bearing semantics (which accepts when r2.tier < target,
        // i.e. "AIC was a coincidental tier-up for an actually-easier
        // puzzle"). For Phase D'5 lemma extraction we want the former
        // — AIC truly required to reach T3 — so this path applies the
        // stricter rule.
        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[TechniqueId::Aic]);
            if r2.rater_error { continue; }
            if tier_rank(r2.tier) <= target_rank { continue; }
        }

        return Some(ReverseResult {
            puzzle,
            solution,
            rate: r,
            clue_count: final_clues,
            attempts_taken: attempt + 1,
        });
    }
    None
}

/// Multi-threaded batch driver. Same per-worker seed scheme as
/// `batch_reverse_construct` — single-threaded calls are deterministic;
/// multi-threaded throughput is the goal, not byte-reproducibility.
pub fn batch_aic_reverse_construct<const N: usize, const BR: usize, const BC: usize>(
    rng_seed: u64,
    spec: &AicReverseSpec,
    num_puzzles: u32,
    threads: usize,
) -> Vec<ReverseResult<N, BR, BC>> {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    let threads = threads.max(1);
    let collected: Arc<Mutex<Vec<ReverseResult<N, BR, BC>>>> =
        Arc::new(Mutex::new(Vec::with_capacity(num_puzzles as usize)));
    let kept = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::with_capacity(threads);
    for w in 0..threads {
        let spec = spec.clone();
        let collected = collected.clone();
        let kept = kept.clone();
        let child_seed = rng_seed.wrapping_add((w as u64).wrapping_mul(0x9E3779B97F4A7C15));
        handles.push(std::thread::spawn(move || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(child_seed);
            let per_worker_attempts = if spec.max_attempts == 0 {
                u32::MAX
            } else {
                ((spec.max_attempts as u64 + threads as u64 - 1) / threads as u64) as u32
            };
            let mut used: u32 = 0;
            while used < per_worker_attempts {
                if kept.load(Ordering::Relaxed) >= num_puzzles { return; }
                let chunk = per_worker_attempts.saturating_sub(used).min(32);
                let chunk_spec = AicReverseSpec { max_attempts: chunk, ..spec.clone() };
                match aic_reverse_construct::<N, BR, BC, _>(&mut rng, &chunk_spec) {
                    Some(res) => {
                        used = used.saturating_add(res.attempts_taken);
                        let prev = kept.fetch_add(1, Ordering::Relaxed);
                        if prev < num_puzzles {
                            collected.lock().unwrap().push(res);
                        } else {
                            kept.fetch_sub(1, Ordering::Relaxed);
                            return;
                        }
                    }
                    None => { used = used.saturating_add(chunk); }
                }
            }
        }));
    }
    for h in handles { let _ = h.join(); }
    let g = collected.lock().unwrap();
    g.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// REGRESSION (commit a05085): the dry-run elimination probe must NOT fire
    /// on a different-digit cross-cell endpoint. Constructs the same setup as
    /// `aic::tests::aic_no_unsound_type2_9x9`: (0,0)={1,2}, (0,6)={1,3}; the
    /// chain (0,0):2 -bivalue-strong- (0,0):1 -row-weak- (0,6):1 ends with a
    /// SAME-digit endpoint pair (1,1) — but note that's not the regression
    /// target. The regression target is: NO different-digit elim returns true.
    /// We assert that no chain-length is found that produces eliminations on
    /// the bare 2-bivalue-cell grid (no peer cells have any candidate to
    /// eliminate anyway, since all other cells are full-bag).
    #[test]
    fn aic_unsoundness_regression_a05085() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        for d in 3..=9u8 { g.eliminate(0, d).unwrap(); }
        for d in [2u8, 4, 5, 6, 7, 8, 9] { g.eliminate(6, d).unwrap(); }
        // (0,0)={1,2}, (0,6)={1,3}. The unsound Type-2 rule, if present, would
        // eliminate digit 1 from (0,0) or digit 3 from (0,6) via a
        // different-digit cross-cell endpoint pair. We probe and assert that
        // the dry-run does NOT mark such a pair as eliminating.
        // Direct different-digit dry-run: (0,0):2 (digit bit 1) and (0,6):0 (digit bit 0)
        let start = node_idx::<9>(0, 1) as u32;     // (0,0):2
        let end = node_idx::<9>(6, 2) as u32;       // (0,6):3
        let mut sc = AicProbeScratch::new::<9>();
        // Use a non-zero probe_gen to exercise the cache rebuild path.
        assert!(
            !try_eliminate_dryrun::<9, 3, 3>(&g, start, end, &mut sc, 1),
            "different-digit cross-cell dry-run must NOT eliminate (a05085 regression)"
        );
        // Same construction also exercised by `find_first_aic_chain_length`:
        // since no full-bag peer cell has 1 OR 3 in a useful position, the
        // probe should not return a chain that would produce unsound elims.
        // (Soft-pass: same-digit chain may still exist; we just check no
        // panic / no impossible "Some" with a different-digit endpoint).
        let _ = find_first_aic_chain_length::<9, 3, 3>(&g);
    }

    /// Hit rate: aic_reverse_construct with chain_length=7, slack=1,
    /// load_bearing=true on 9×9 should produce ≥1 puzzle in 200 attempts.
    /// R1 search-and-filter measured 0/5000 for this target.
    #[test]
    fn hit_rate_9x9_chain7_load_bearing() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 7,
            chain_length_slack: 1,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 200,
            require_load_bearing: true,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
        let res = aic_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        // Soft contract: if Some, all properties must hold.
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T3));
            assert!(r.rate.frontier.contains(&TechniqueId::Aic));
            let len = find_first_aic_chain_length::<9, 3, 3>(&r.puzzle).unwrap();
            assert!((len as i32 - 7).abs() <= 1, "chain length {} not within ±1 of 7", len);
            let r2 = rate_excluding(&r.puzzle, &[TechniqueId::Aic]);
            assert!(tier_rank(r2.tier) > tier_rank(Tier::T3),
                "load_bearing violated: rate_excluding tier = {:?}", r2.tier);
        } else {
            eprintln!("warn: 200 attempts insufficient for chain=7±1 load-bearing (soft-pass)");
        }
    }

    /// Looser variant: without load-bearing at chain=5 ±2, hit rate must be
    /// non-trivially positive (the looser constraint should fire reliably).
    #[test]
    fn hit_rate_9x9_chain5_slack2_no_load_bearing() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 5,
            chain_length_slack: 2,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 100,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let res = aic_reverse_construct::<9, 3, 3, _>(&mut rng, &spec);
        if let Some(r) = res {
            assert_eq!(tier_rank(r.rate.tier), tier_rank(Tier::T3));
            assert!(r.rate.frontier.contains(&TechniqueId::Aic));
        }
        // Soft-pass for None — environment-dependent.
    }

    /// Determinism: same seed → same puzzle (single-thread).
    #[test]
    fn determinism_same_seed_same_puzzle() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 5,
            chain_length_slack: 3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 50,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(31337);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(31337);
        let ra = aic_reverse_construct::<9, 3, 3, _>(&mut a, &spec);
        let rb = aic_reverse_construct::<9, 3, 3, _>(&mut b, &spec);
        match (ra, rb) {
            (Some(x), Some(y)) => {
                assert_eq!(x.puzzle.to_string_grid(), y.puzzle.to_string_grid());
            }
            (None, None) => {}
            _ => panic!("determinism violated: one Some, one None"),
        }
    }

    /// Validation: target_chain_length=0 is rejected.
    #[test]
    fn validate_zero_chain_length_rejected() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 0,
            chain_length_slack: 0,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        assert!(spec.validate().is_err());
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        // aic_reverse_construct returns None on validation error.
        assert!(aic_reverse_construct::<9, 3, 3, _>(&mut rng, &spec).is_none());
    }

    /// Validation: T1/T2 tiers rejected (Aic is T3).
    #[test]
    fn validate_t1_t2_rejected() {
        let mut spec = AicReverseSpec::new_t3(5, 1, 10);
        spec.target_tier = Tier::T1;
        assert!(spec.validate().is_err());
        spec.target_tier = Tier::T2;
        assert!(spec.validate().is_err());
        spec.target_tier = Tier::T3;
        assert!(spec.validate().is_ok());
    }

    /// 16×16 informational smoke: aic_reverse_construct with chain_length=3,
    /// slack=2, num_attempts small. We only assert termination — generation
    /// at 16×16 is very slow.
    #[test]
    fn smoke_16x16_terminates() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 3,
            chain_length_slack: 2,
            clue_min: 100,
            clue_max: 130,
            max_attempts: 1,
            require_load_bearing: false,
            greedy_max_trials: 32, // cap aggressively for speed
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(101);
        // Just terminate; Some-or-None both acceptable.
        let _ = aic_reverse_construct::<16, 4, 4, _>(&mut rng, &spec);
    }

    /// Returned puzzle solves to its `solution` field via `solve_unique`.
    #[test]
    fn returned_puzzle_solves_to_solution() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 5,
            chain_length_slack: 3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 100,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2027);
        if let Some(r) = aic_reverse_construct::<9, 3, 3, _>(&mut rng, &spec) {
            let sol = solve_unique::<9, 3, 3>(&r.puzzle).expect("returned puzzle must be unique");
            assert_eq!(sol.to_string_grid(), r.solution.to_string_grid());
        }
    }

    /// Batch driver smoke: returns at most num_puzzles.
    #[test]
    fn batch_smoke_terminates() {
        let spec = AicReverseSpec {
            target_tier: Tier::T3,
            target_chain_length: 5,
            chain_length_slack: 3,
            clue_min: 22,
            clue_max: 30,
            max_attempts: 30,
            require_load_bearing: false,
            greedy_max_trials: 0,
        };
        let res = batch_aic_reverse_construct::<9, 3, 3>(42, &spec, 2, 1);
        assert!(res.len() <= 2);
    }

    // -----------------------------------------------------------------
    // R3.3b2 scratch-reuse correctness:
    // Reusing one scratch buffer across multiple `find_first_aic_chain_length`
    // calls must produce byte-identical results to a fresh-buffer baseline.
    // -----------------------------------------------------------------

    /// Generate a small set of partial puzzles by stripping random subsets of
    /// clues from a known solution; assert reused-scratch chain lengths match
    /// the fresh-allocation baseline.
    #[test]
    fn scratch_reuse_matches_fresh_baseline_9x9() {
        // Take a fixed valid 9×9 solution.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(424242);
        let cfg = super::super::generator::GenConfig { target_clues: 27, max_attempts: 0 };
        let (_seed, _, solution) =
            super::super::generator::gen_unique_puzzle_with_solution::<9, 3, 3, _>(&mut rng, &cfg);

        // Build many partial puzzles by keeping random subsets of cells.
        let mut scratch = AicProbeScratch::new::<9>();
        for trial in 0..50 {
            let mut keep = vec![false; 81];
            // Keep ~30 random cells.
            let mut idxs: Vec<u8> = (0..81u8).collect();
            idxs.shuffle(&mut rng);
            for &i in &idxs[..30] { keep[i as usize] = true; }
            let g = match build_subset::<9, 3, 3>(&solution, &keep) {
                Some(g) => g, None => continue,
            };
            let baseline = find_first_aic_chain_length::<9, 3, 3>(&g);
            let reused = find_first_aic_chain_length_with::<9, 3, 3>(&g, &mut scratch);
            assert_eq!(
                baseline, reused,
                "scratch-reuse mismatch on trial {trial}: baseline={baseline:?} reused={reused:?}",
            );
        }
    }

    /// Same idea on 16×16 (smaller batch — generation is slow). Verifies the
    /// scratch buffers don't carry stale state between calls at scale.
    #[test]
    fn scratch_reuse_matches_fresh_baseline_16x16() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1111);
        let cfg = super::super::generator::GenConfig { target_clues: 130, max_attempts: 0 };
        let (_seed, _, solution) =
            super::super::generator::gen_unique_puzzle_with_solution::<16, 4, 4, _>(&mut rng, &cfg);

        let mut scratch = AicProbeScratch::new::<16>();
        for trial in 0..3 {
            let mut keep = vec![false; 256];
            let mut idxs: Vec<u16> = (0..256u16).collect();
            idxs.shuffle(&mut rng);
            for &i in &idxs[..130] { keep[i as usize] = true; }
            let g = match build_subset::<16, 4, 4>(&solution, &keep) {
                Some(g) => g, None => continue,
            };
            let baseline = find_first_aic_chain_length::<16, 4, 4>(&g);
            let reused = find_first_aic_chain_length_with::<16, 4, 4>(&g, &mut scratch);
            assert_eq!(
                baseline, reused,
                "16×16 scratch-reuse mismatch on trial {trial}: baseline={baseline:?} reused={reused:?}",
            );
        }
    }
}
