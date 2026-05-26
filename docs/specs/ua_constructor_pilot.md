# UA-Constructor Pilot — Specification

**Status:** Spec, not yet implemented.
**Goal:** validate the UA-based construction approach against AlphaZero, ES, and GFlowNet alternatives on the BxB ≥ 6 generation problem.
**Owner:** TBD (recommend Opus subagent for implementation).
**Estimated effort:** 800-1200 LOC + 1-3 days execution.

## 1. Problem statement

Generate a Sudoku puzzle P satisfying:
- `shc::te::rate_bxb(P, max_length=14, buffer_size=1M) >= 6`
- Uniqueness verified (single solution).
- Without using any pre-existing hard puzzle as input.

Pilot's role: an MVP measuring whether the UA-based construction approach can hit this target at the empirical hit-rate / cost claimed by literature (10-30% per attempt at 2-10 min/attempt). The pilot decides whether to commit to the full ~2000 LOC version or pivot.

## 2. Algorithmic procedure

The constructor takes a random full solution grid and constructs a minimal-clue puzzle by minimum-cover over Unavoidable Sets.

```
PROCEDURE construct_one_puzzle(seed: u64) -> Option<(Puzzle, u16)>:
    1. G = random_full_solution_grid(seed)           # ~10ms
    2. UAs = enumerate_unavoidable_sets(G, max_size=12)
       # Returns Vec<HashSet<CellIndex>>
       # ~1-30s depending on grid
    3. S = greedy_min_hitting_set(UAs)
       # Returns Vec<CellIndex>
       # ~1ms
    4. P = build_puzzle_from(G, S)
       # P has clues only at positions in S, with values G[s]
    5. if not is_unique(P): return None  # rare with proper UA cover
    6. rating = rate_bxb(P, 14, 1_000_000)
       # 0.5-15s
    7. if rating >= 6: return Some((P, rating))
       else: return None
```

The full ~2000 LOC version would later replace step 3 with ILP-based exact min cover and extend step 2 to UAs of size 14+. The pilot uses cheaper greedy + smaller UA bound.

## 3. Acceptance criteria (pass / fail)

**Pass — proceed to full version:**
- ≥ 1 BxB ≥ 6 puzzle produced in 100 attempts (so hit rate ≥ 1%, vs 10⁻⁶ for uniform random — 10⁴× improvement).
- 95%+ of attempts produce VALID puzzles (uniqueness verified, fully decided).
- Median wall per attempt < 60s on M3 Max single thread.
- Output BxB distribution shows non-trivial mass at BxB ≥ 3 (sanity check that the procedure produces meaningful hardness, not all BxB=0 like uniform random).

**Marginal — investigate before committing:**
- BxB ≥ 6 hit rate ∈ [0.1%, 1%]. UAs work but worse than literature claims. Likely need ILP + larger UA enumeration.
- Median wall ∈ [60s, 300s] per attempt. Plausible to speed up.

**Fail — pivot to GFlowNet or other alternative:**
- 0 BxB ≥ 6 puzzles in 100 attempts.
- > 50% of attempts produce invalid puzzles (non-unique or unsolvable).
- Median wall > 300s per attempt.

## 4. Data structures

```rust
// src/shc/ua/mod.rs (new module)

pub type CellIndex = u8;  // 0..81

pub struct FullGrid {
    pub cells: [u8; 81],  // each ∈ 1..=9
}

pub struct UnavoidableSet {
    pub cells: Vec<CellIndex>,  // sorted ascending for stable hashing
    pub size: u8,
}

pub struct HittingSet {
    pub cells: Vec<CellIndex>,  // sorted
    pub n_uas_hit: usize,  // for diagnostics
}

pub struct ConstructResult {
    pub puzzle: shc::Board,           // for downstream rate_bxb
    pub clue_count: u8,
    pub bxb_rating: i16,              // -3/-4 mapped, else u16
    pub n_uas_found: usize,
    pub wall_ms_grid: u64,
    pub wall_ms_ua_enum: u64,
    pub wall_ms_min_cover: u64,
    pub wall_ms_uniqueness: u64,
    pub wall_ms_rate_bxb: u64,
}
```

## 5. Component specifications

### 5.1 Random full grid generator

```rust
pub fn random_full_grid(rng: &mut impl Rng) -> FullGrid
```

Use the existing `src/shc/uniqueness::search` or `src/generic/generator::gen_unique_puzzle_with_solution` machinery — extract the "full solution" portion only. Should NOT do clue removal — we want a complete 81-cell grid.

Expected: ≤ 10ms per call.

### 5.2 Unavoidable Sets enumeration

```rust
pub fn enumerate_unavoidable_sets(
    grid: &FullGrid,
    max_size: usize,
) -> Vec<UnavoidableSet>
```

**Definition.** A set of cells U ⊂ {0..81} is an Unavoidable Set if there exists another valid full grid G' agreeing with G outside U but differing on every cell in U. Equivalently: the digit values on U can be permuted to yield a different valid grid.

**Pilot enumeration (size 4-12 only):**
- Size 4 (deadly rectangles): for each pair of rows (r1, r2), for each pair of cols (c1, c2): check if {G[r1,c1], G[r1,c2], G[r2,c1], G[r2,c2]} forms a 2×2 with cells in 2 boxes that admits the swap. ~36² = 1296 candidates, filter to ~100-300 actual UAs.
- Size 6, 8, 10, 12: extend by adding cells one row/col at a time and verifying the cycle property. Branch-and-bound on the cell graph.

Literature reference: see Berthier HCCS Ch. 4 (sec. on UA enumeration) and the enjoysudoku forum thread "Finding unavoidable sets" (linked in `research/hardest_puzzle_construction_notes.md`).

**Algorithm sketch (pilot — size-bounded BFS):**
```
function enumerate_unavoidable_sets(grid, max_size):
    UAs = []
    # Seed: all size-4 UAs (deadly rectangles)
    for each 2×2 cell subset where same digits in same row/col pattern:
        if valid_UA(subset, grid): UAs.push(subset)
    # Extend by 2 cells at a time (UAs are even-sized for swap cycle)
    for size in 6..=max_size step 2:
        for each existing UA of size < target:
            for each cell extension:
                if forms_valid_UA(extension): add to UAs
                dedup
    return UAs
```

Validity check `is_unavoidable(U, G)`: construct G' = G with values on U cyclically permuted. Check G' is also a valid Sudoku (all rows/cols/boxes contain 1-9 exactly once). If yes, U is unavoidable.

**Performance budget:** < 30s per grid. Limit `max_size=12` for pilot (covers ~99% of practically relevant UAs per Berthier).

### 5.3 Greedy minimum hitting set

```rust
pub fn greedy_min_hitting_set(uas: &[UnavoidableSet]) -> HittingSet
```

Classic greedy set cover: repeatedly pick the cell that hits the most uncovered UAs.

```
S = {}
remaining = uas.clone()
while remaining is non-empty:
    cell* = argmax_{cell ∈ 0..81} |{ua ∈ remaining : cell ∈ ua}|
    S.add(cell*)
    remaining = filter(ua ∈ remaining : cell* ∉ ua)
return S
```

Performance: O(|UAs| × 81) per iteration × O(|UAs|) iterations ≈ ~ms for |UAs| ≤ 10⁴.

Greedy gives an O(log n)-approximation to true minimum. For the pilot this is acceptable — full version may use ILP for exact min.

### 5.4 Puzzle construction & verification

```rust
pub fn build_puzzle_from(grid: &FullGrid, clues: &HittingSet) -> shc::Board
pub fn is_unique(board: &shc::Board) -> bool  // existing: shc::uniqueness::verify_unique_solution
pub fn rate_bxb(board: &shc::Board) -> i16    // existing: shc::te::rate_bxb wrapper
```

## 6. CLI

Add `gen-ua-pilot` subcommand to `src/cli.rs`:

```bash
sudoku_rs_core gen-ua-pilot \
  --attempts 100 \
  --seed 42 \
  --max-ua-size 12 \
  --target-bxb 6 \
  --threads 8 \
  --output /tmp/chain_rerate/ua_pilot_results.jsonl
```

Output JSONL per attempt:
```json
{
  "attempt": 0,
  "seed": 42000,
  "grid_hash": "0xabc...",
  "n_uas": 423,
  "n_clues": 22,
  "is_unique": true,
  "bxb_rating": 4,
  "is_target_hit": false,
  "wall_ms_total": 24536,
  "wall_ms_grid": 8,
  "wall_ms_ua_enum": 19234,
  "wall_ms_min_cover": 12,
  "wall_ms_uniqueness": 1,
  "wall_ms_rate_bxb": 5281
}
```

After all attempts complete, print summary:
```
Attempts: 100
Valid (unique + complete): 98 (98%)
BxB distribution: BxB=0: 12, BxB=2: 38, BxB=3: 31, BxB=4: 12, BxB=5: 4, BxB=6: 1, BxB=7: 0
Target hits (BxB >= 6): 1 (1.0%)
Wall: median per-attempt 24.5s, total 41 min
Per-stage wall (median): grid=8ms ua_enum=19s min_cover=12ms uniqueness=1ms rate_bxb=5.3s
```

## 7. Testing

Inline unit tests (`#[cfg(test)] mod tests` in `src/shc/ua/`):

- `test_deadly_rectangle_is_ua`: construct a known 4-cell UA, verify `is_unavoidable` returns true.
- `test_non_ua_returns_false`: cell set that's NOT a UA must return false.
- `test_greedy_covers_all_uas`: synthetic UA list, verify greedy output hits every UA.
- `test_construct_from_known_grid`: hard-code a grid where UA construction is expected to yield ≥1 unique puzzle. Verify uniqueness.
- `test_construct_full_pipeline_smoke`: run 1 attempt end-to-end, verify all stages return without panic and BxB ∈ {0..14}.

Integration test: full pilot script with 5 attempts (small N) producing the JSONL.

## 8. Risks (descending probability)

1. **UA enumeration too slow / explodes for some grids.** Mitigation: hard cap on wall (`max_ua_enum_ms = 30000`); abort grid if exceeded. Move on.
2. **Greedy min hitting set is too lossy** vs ILP. May produce fewer high-BxB puzzles. Mitigation: report greedy_size / lower_bound_LP ratio as diagnostic.
3. **Random grid distribution is skewed.** Some grids may have very few UAs of relevant size; constructor produces ~empty clue sets. Mitigation: pre-screen grids by UA count (require ≥ 50 UAs found, else skip).
4. **rate_bxb timeouts on some constructed puzzles.** Mitigation: wrap rate_bxb call with timeout; on timeout return -3 (buffer overflow indicator) and continue.
5. **Min-clue threshold disagreement.** Pilot may produce 17-clue puzzles which we already know are BxB=0. Mitigation: track clue count; if `n_clues < 20`, expect low BxB. This is informative, not a failure.

## 9. Stop / continue decision

After 100 attempts complete:
- **Target hit ≥ 1%** AND median wall < 60s: PASS → proceed to full implementation (~2000 LOC including ILP, larger UA, parallel attempt batching).
- **Target hit 0.1-1%** OR median wall 60-300s: MARGINAL → investigate per-stage walls, consider mitigations from §8, run 200 more attempts with tweaks.
- **Target hit < 0.1%** OR median wall > 300s: FAIL → publish negative finding, pivot to GFlowNet pilot or accept that BxB ≥ 6 generation is community-curation-bound.

## 10. Out of scope (pilot does NOT include)

- ILP-based exact min hitting set (uses greedy instead).
- UAs of size > 12 (will limit hit rate but pilot is about validating the approach, not maximizing yield).
- Parallel grid attempts within a single MCTS-style search (just embarrassingly parallel over attempts).
- Backward iterative refinement (e.g. try removing clues from the constructed puzzle to find smaller minimal version that retains BxB ≥ 6).
- Caching of UAs across grids (each attempt is independent).
- GUI / visualization of UAs.

## 11. Deliverables

1. `src/shc/ua/mod.rs` — UA enumeration + min hitting set + pilot driver. ~800 LOC.
2. CLI subcommand `gen-ua-pilot` in `src/cli.rs`. ~150 LOC.
3. Unit tests (5 listed in §7). ~100 LOC.
4. Pilot execution: 100 attempts with `--seed 42 --max-ua-size 12 --target-bxb 6 --threads 8`. Output JSONL + summary printout.
5. Pilot report (~300 words) appended to `docs/research/hardest_puzzle_construction_notes.md` §6 (NEW): attempt count, hit rate, BxB distribution, per-stage walls, stop/continue verdict.

## 12. Verification before pilot launch

- `cargo test --release --lib` — no new failures (existing 17 shc:: + 461/0/12 generic).
- `cargo build --release` clean.
- `gen-ua-pilot --attempts 1 --seed 0` runs to completion within 60s wall (smoke).
- Sample output verified against `shc::te::rate_bxb` on a hand-crafted puzzle (e.g. forum_hardest #1).

## 13. Out-of-band kill switch (cheap pre-test)

OPTIONAL — can be skipped if running on schedule:

Before launching the 100-attempt pilot, run a 10-attempt smoke pilot on the existing forum_hardest BxB=5 corpus (1166 puzzles). For each: run UA enumeration on the puzzle's grid, count UAs. Check that grids underlying known BxB=5+ puzzles have ≥ 100 UAs of size ≤ 12. If they don't, our UA enumeration is undercounting and the approach is broken before we even start.

Wall: ~ 5 min total. Defends against a UA-enumeration bug going undetected through the full pilot.
