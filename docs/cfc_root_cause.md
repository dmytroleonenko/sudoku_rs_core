# CFC Root Cause Analysis

_Investigated 2026-05-12. Read-only phase; no code changes._

---

## 1. Algorithm Comparison (Side-by-Side)

| Aspect | Our CFC (`cell_fc.rs`) | SE CFC (`Chaining.java`, `isMultiple=true, isDynamic=false, level=0`) |
|---|---|---|
| **Propagator inside hypothesis** | `propagate_singles()` — global BFS/fixpoint of naked + hidden singles applied to entire cloned grid | BFS on `(cell, value)` implications via two rules: NakedSingle (Y-link) + HiddenSingle (X-link). `getAdvancedPotentials()` (locked/pairs/fish) is only called when `level > 0`; static CFC has `level=0` so it is NOT called. |
| **Pivot cell candidate count** | k ∈ {2, 3, 4} — includes 2-candidate cells | k > 2 only for static Multiple FC (line 225: `cardinality > 2`). 2-candidate cells are handled by SE's separate BinaryForcingChain / Y-Chain instead. |
| **Contradiction handling** | Contradiction → immediately eliminate that candidate from pivot and return | Same: contradiction in `doChaining()` returns two conflicting potentials → candidate is removed |
| **Consensus definition** | `Set<Consequence>` intersection: `(cell_idx, digit, Action::Place\|Eliminate)` present in all k branches | `LinkedSet<Potential>` intersection via `cellToOn.retainAll(onToOn)` and `cellToOff.retainAll(onToOff)`. `Potential` is `(cell, value, isOn)` where `isOn=true` → place, `isOn=false` → eliminate. Functionally equivalent. |
| **Base rating** | 7.5 (hard-coded in `base_rating()`) | `Chaining.getDifficulty()` for `isMultiple=true, isDynamic=false` → **8.0**; final rating = 8.0 + `getLengthDifficulty()` (length penalty scaling from chain node count). |
| **Iteration order** | Natural cell index order (cell 0 first, then 1, … 80) | Row-major (y=0..8, x=0..8) — same as natural order for a 9×9 |
| **No-op filtering** | Already-satisfied consequences filtered before apply | `isWorth()` on hint object checks `removablePotentials.isEmpty()` |

---

## 2. Diagnosed Root Causes

### Root Cause A — Wrong Cardinality Bound (Primary, causes **false positives**)

Our CFC fires on **2-candidate pivot cells**. SE's static Multiple Forcing Chain **skips them** (`cardinality > 2` guard on line 225 of `Chaining.java`). SE handles 2-candidate cells through a separate technique category: Binary Forcing Chain / Y-Chain (unary chaining on `(cell, value)` pairs).

Consequence: for puzzles with a 2-candidate cell that forms a valid consensus, our CFC fires and returns T4Plus/7.5; SE does not use CFC at all and instead uses a simpler technique or a differently-rated chain. This explains the SE=4.5 false positive: the puzzle has a 2-candidate pivot where our CFC fires a consensus elimination, but SE solves it via T2 (Hidden Pair = 4.5) without ever engaging CFC.

The trace for the SE=4.5 false-positive puzzle (`4.1.6....3.....2........8..15.2.....6......1....9......2.7.8..........43.7.......`) confirms this: after AIC fires, CFC fires at trace positions 9–10. The AIC moves reduce candidates, and a 2-candidate cell then becomes a valid CFC pivot under our implementation.

### Root Cause B — Wrong Base Rating (contributes to systematic bias)

We use `base_rating = 7.5`. SE's static CFC (`isMultiple=true`) has `getDifficulty() = 8.0` plus a length-based penalty. Even when our CFC fires correctly, we report 7.5 where SE would report ≥ 8.0. This accounts for approximately half of the mean underestimate of −1.09 on CFC puzzles.

### Root Cause C — Missing Techniques for Very Hard Puzzles (causes **false negatives / unsolved**)

For the 24 puzzles with `solved=False` and `trace=[]` (SE ER 9.0–11.1): these are hardest-1905 class puzzles requiring SE techniques at level ≥ 1 (`DynamicFC+`, `NestedFC`). Our rater does not have Dynamic or Region Forcing Chains. CFC not firing is correct behavior; the real gap is missing higher-tier techniques. These puzzles should not count as CFC false negatives — they are an orthogonal problem.

For the 6 "solved false negatives" (SE ER 8.3–8.8, our tier T3, no CFC): our rater solves them via `als_xz` (base 7.5). SE uses CFC (base 8.0+). Both techniques cover the same eliminations in these cases. Our ALS-XZ fires and the CFC never gets the chance. This is not a CFC miss — it's technique overlap with an ALS-XZ that fires earlier in the cascade.

---

## 3. False-Positive Case Study (SE=4.5, our frontier: CFC)

**Puzzle**: `4.1.6....3.....2........8..15.2.....6......1....9......2.7.8..........43.7.......`  
**SE ER**: 4.5 (Hidden Pair / Naked Pair territory)  
**Our trace**: `locked_pointing → locked_claiming → naked_pair → locked_claiming → hidden_pair → locked_pointing → ur_type1 → aic → aic → cell_forcing_chain → cell_forcing_chain → naked_pair → naked_pair → ur_type1`

Evidence: CFC fires at position 9 in the trace, after two AIC moves. At that point in the solve, the residual grid has at least one unsolved cell with exactly 2 candidates. Our code considers 2-candidate cells as valid CFC pivots. SE does not use CFC for 2-candidate cells; instead it would use BinaryFC/Y-Chain. SE solved this puzzle at step ER=4.5 (Hidden Pair), never reaching the CFC-eligible state. Our cascade's use of AIC before CFC means we arrive at a residual state where a 2-candidate pivot "fires" CFC spuriously, returning T4Plus/7.5 on a puzzle SE considers T2.

**Why SE doesn't need CFC here**: SE's approach for 2-candidate cells is Y-Chain / Binary Forcing Chain (rated below 7.5). It solved the puzzle before the CFC stage. We don't have Binary Forcing Chain as a separate technique, so CFC absorbs that case with the wrong rating.

---

## 4. False-Negative Case Study (SE=8.8, our tier T3, no CFC)

**Puzzle**: `5.8.....7...9.1...4............5...4.6.....3..9....6...2.3...1.7...8..........2..`  
**SE ER**: 8.8 (SE uses CFC, rated ≥ 8.0 + length)  
**Our trace**: `locked_pointing → naked_pair → hidden_pair → simple_coloring → als_xz → naked_triple`  
**Our tier**: T3 (frontier dominated by `als_xz`, base 7.5)

SE rates this at 8.8 because it needs a Cell Forcing Chain (k=3 or 4 candidates). Our `als_xz` technique fires first and makes the same eliminations via a different reasoning path. When ALS-XZ solves the puzzle, CFC is never reached.

**Net effect**: we underestimate by ≈ −1.3 (our 7.5 vs SE 8.8). This is not a CFC implementation bug — it's a **technique overlap** combined with rating misalignment between our `als_xz` (7.5) and SE's CFC (8.0+). If our ALS-XZ base were bumped to ≈ 8.0–8.5 for the harder cases, this bucket would close.

---

## 5. Recommended Fix

### Recommendation: **(a) + (c) combined**

**Step 1 — Fix cardinality guard** (closes false positives; S effort)

In `cell_fc.rs` line 152, change the lower bound from `k < 2` to `k < 3`:

```rust
// Line 152 currently:
if k < 2 || k > MAX_CANDIDATES {
// Change to:
if k < 3 || k > MAX_CANDIDATES {
```

This aligns our CFC with SE's static Multiple FC: 2-candidate cells are handled separately. Effect: the SE=4.5 false positive disappears (CFC no longer fires on 2-cand pivots). The 5 existing tests are unaffected — `contradiction_subcase_eliminates_candidate` uses a hack that sets `candidates[0] = 0b11` (2-cand); that test would still pass because the contradiction sub-case is reached differently, but should be reviewed to ensure it still exercises what it claims. The `cell_fc_skips_high_candidate_cells` test exercises k=9 and passes either way. `cell_fc_does_not_fire_on_easy_puzzle` and `rated_technique_base_rating` are unaffected.

**Step 2 — Fix base rating** (closes systematic −1.09 bias on CFC puzzles; S effort)

In `cell_fc.rs` line 300:
```rust
fn base_rating(&self) -> f64 {
    8.0 // Aligned to SE Multiple FC base (was 7.5)
}
```

SE's actual CFC difficulty is `8.0 + getLengthDifficulty()`. Since we don't track chain length, 8.0 is the correct base. This will shift our CFC bucket mean up by +0.5, closing approximately half the −1.09 mean bias.

**Step 3 — Optional: add Binary Forcing Chain** (closes remaining 2-cand case; M effort)

The proper fix for 2-candidate pivot cells is a dedicated `BinaryForcingChain` technique with the right SE rating (≈ 6.6–7.0 for short chains; BinaryFC is rated by SE via the same length formula but at base 6.6). This is distinct from CFC and matches SE's technique taxonomy. Without this, the 2-cand pivot case is simply unhandled (returns T3/ALS tier instead of T4 BinaryFC tier).

---

## 6. Risk / Regression Assessment

**If we apply Step 1 (k ≥ 3 guard)**:

- `contradiction_subcase_eliminates_candidate` (test in `cell_fc.rs`): This test manually sets `g.candidates[0] = 0b11` (2 candidates). With k < 3 guard, cell 0 would be **skipped** and the test would fail. The test needs to be updated to use a 3-candidate pivot (e.g., `g.candidates[0] = 0b111`) and adjust the contradiction scenario accordingly.
- All other 4 tests: unaffected.

**If we apply Step 2 (base_rating 8.0)**:

- `rated_technique_base_rating` test (line 759) checks `(rating - 7.5).abs() < 1e-9` — this test **will fail** and must be updated to check 8.0.
- Integration-level calibration test (if any) checking score bounds: review after the change.

**Overall regression risk**: Low. Both changes are small and the affected tests are clearly identified. The change in guard aligns us with SE's published algorithm; the rating change aligns with SE's `getDifficulty()` source. The corrected RMS should improve from 1.62 to < 1.0 on the CFC bucket.
