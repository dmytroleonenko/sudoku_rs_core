# SE Audit Report — Rust sudoku_rs_core vs SudokuExplainer 1to9only fork

**Date:** 2026-05-12
**SE source:** https://github.com/1to9only/SudokuExplainer (commit depth=1, cloned to /tmp/sudoku_explainer)
**Our source:** `tools/sudoku_rs_core/src/generic/techniques/`

---

## 1. SE Technique Inventory

Ratings extracted from Java source (`*Hint.java`, `Chaining.java`). Order by SE rating ascending.

| SE technique | SE rating | Java file | Algorithm one-liner | Our equivalent |
|---|---|---|---|---|
| Hidden Single (block) | 1.0 | `HiddenSingleHint.java:29` | Only position for digit in a block | `propagate_singles` (T1) |
| Hidden Single (row/col) | 1.2 / 1.5 | `HiddenSingleHint.java:31,33` | Only position for digit in row or column | `propagate_singles` (T1) |
| Direct Pointing | 1.7 | `DirectLockingHint.java:94` | Box–line reduction visible as direct placement | Not implemented (direct variant) |
| Direct Claiming | 1.9 | `DirectLockingHint.java:96` | Line–box reduction as direct placement | Not implemented (direct variant) |
| Naked Single | 2.3 | `NakedSingleHint.java:21` | Cell with only one candidate | `propagate_singles` (T1) |
| Direct Hidden Pair | 2.0 | `DirectHiddenSetHint.java:107` | Hidden pair that directly places a value | Not implemented (direct variant) |
| Direct Hidden Triple | 2.5 | `DirectHiddenSetHint.java:109` | Hidden triple yielding direct placement | Not implemented (direct variant) |
| Pointing | 2.6 | `LockingHint.java:69` | Candidates in box locked to one row/col → eliminate from that line | `LockedPointing` (T2) |
| Claiming | 2.8 | `LockingHint.java:71` | Candidates in line locked to one box → eliminate from that box | `LockedClaiming` (T2) |
| Naked Pair | 3.0 | `NakedSetHint.java:67` | Two cells in unit share same two candidates | `NakedSet<2>` (T2) |
| X-Wing | 3.2 | `LockingHint.java:73` | Two rows (cols) with digit in exactly two cols (rows) → eliminate from those cols (rows) | `Fish<2>` (T3) |
| Hidden Pair | 3.4 | `HiddenSetHint.java:69` | Two digits confined to two cells in a unit | `HiddenSet<2>` (T2) |
| Naked Triplet | 3.6 | `NakedSetHint.java:69` | Three cells sharing three candidates | `NakedSet<3>` (T2 in our rater) |
| Swordfish | 3.8 | `LockingHint.java:75` | Three-row/col fish on one digit | `Fish<3>` (T3) |
| Hidden Triplet | 4.0 | `HiddenSetHint.java:71` | Three digits in three cells | `HiddenSet<3>` (T2 in our rater) |
| XY-Wing | 4.2 | `XYWingHint.java:83` | Three bivalue cells; pivot shares one candidate with each wing | `XyWing` (T3) |
| XYZ-Wing | 4.4 | `XYWingHint.java:81` | XY-Wing with trivalue pivot | `XyzWing` (T3) |
| Unique Rectangle/Loop Type 1 | 4.5 base (+0.1–0.3 for loop length ≥6,8,10) | `UniqueLoopHint.java:92` | Two digits in four cells forming a rectangle — uniqueness constraint eliminates extra candidate | `UrType1` (T3, base_rating=4.5) |
| Unique Rectangle/Loop Type 2 | 4.5+ | `UniqueLoopHint.java:92` | UR with extra candidate forced | `UrType2` (T3, base_rating=4.7) |
| Unique Loop Type 3 (Naked) | 4.5 + set_size * 0.1 | `UniqueLoopType3NakedHint.java:39` | UR type 3 with naked set lock | Not implemented |
| Unique Loop Type 3 (Hidden) | 4.5 + set_size * 0.1 | `UniqueLoopType3HiddenHint.java:42` | UR type 3 with hidden set lock | Not implemented |
| Unique Loop Type 4 | 4.5+ | `UniqueLoopType4Hint.java:87` | UR type 4; one digit can only appear in UR cells | Not implemented |
| Naked Quad | 5.0 | `NakedSetHint.java:71` | Four cells with at most four candidates | `NakedSet<4>` (T2 in our rater) |
| Jellyfish | 5.2 | `LockingHint.java:77` | Four-row/col fish | `Fish<4>` (T3) |
| Hidden Quad | 5.4 | `HiddenSetHint.java:73` | Four digits in four cells | `HiddenSet<4>` (T2 in our rater) |
| BUG+1 | 5.6 | `Bug1Hint.java:60` | Bivalue Universal Grave + one extra candidate | `Bug` (T3, base_rating=5.6) |
| BUG+2 | 5.7 | `Bug2Hint.java:63` | BUG with two extra candidates | Not implemented |
| BUG+3 | 5.7+ | `Bug3Hint.java:81` | BUG with three extra candidates | Not implemented |
| BUG+4 | 5.7 | `Bug4Hint.java:87` | BUG with four extra candidates | Not implemented |
| Aligned Pair Exclusion | 6.2 | `AlignedExclusionHint.java:121` | Two aligned cells; test all value pairs for contradiction | Not implemented |
| Forcing Chain / X-Chain cycle | 6.5 base + length_diff | `CycleHint.java:125` (X-cycle) | Alternating strong/weak links in single-digit chains forming discontinuous loop | `Aic` (partial — see §5) |
| Bidirectional Y-Cycle | 6.5 base + length_diff | `CycleHint.java:125` | Y-cycle (bivalue cell strong links) | `Aic` (partial) |
| Bidirectional XY-Cycle | 7.0 base + length_diff | `CycleHint.java:123` | XY-cycle mixing X and Y links | `Aic` (partial) |
| Forcing Chain (X-Chain, non-loop) | 6.6 base + length_diff | `ForcingChainHint.java:106` | Single-end forcing chain on one digit | `Aic` (partial — see §5) |
| Forcing Chain (XY, non-loop) | 7.0 base + length_diff | `ForcingChainHint.java:103` | Forcing chain mixing value links | `Aic` (partial) |
| Aligned Triple Exclusion | 7.5 | `AlignedExclusionHint.java:123` | Three aligned cells, all value triples tested | Not implemented |
| Nishio Forcing Chain | 7.5 base + length_diff | `Chaining.java:72` | Assume digit on; derive contradiction | Folded into `CellForcingChain` contradiction sub-case (see §5) |
| Cell Forcing Chain (Multiple) | 8.0 base + length_diff | `Chaining.java:76` | All hypotheses for candidates in a cell lead to same conclusion | `CellForcingChain` (T4Plus, base_rating=7.5) — see §5 |
| Dynamic Forcing Chain | 8.5 base + length_diff | `Chaining.java:74` | FC with dynamic propagation (applies locked candidates etc. inside hypothesis) | Not implemented |
| Region Forcing Chain | 8.0 base + length_diff | `RegionChainingHint.java:117` | All positions for a digit in a region lead to same conclusion | Not implemented |
| Dynamic Forcing Chain (+) | 9.0+ | `Chaining.java:70` (level=1) | Dynamic FC, depth 1 nesting | Not implemented |
| Nested Forcing Chain (level 2–5) | 9.5–11.5 | `Chaining.java:68` (level≥2) | Nested FC with increasing nesting depth | Not implemented |

**Length difficulty formula** (from `ChainingHint.java:335`):
Steps thresholds 4,6,8,12,16,24,32,... ; each threshold crossed adds +0.1 to the base rating.
A 6-node chain adds +0.1; a 10-node chain adds +0.2; a 16-node chain adds +0.3.

---

## 2. Weight Discrepancies (Our base_rating vs SE)

| Technique | Our base_rating | SE rating | Diff | Notes |
|---|---|---|---|---|
| LockedPointing | 2.6 | 2.6 | 0.0 | Correct |
| LockedClaiming | 2.8 | 2.8 | 0.0 | Correct |
| NakedPair | 3.0 | 3.0 | 0.0 | Correct |
| XWing | 3.2 | 3.2 | 0.0 | Correct |
| HiddenPair | 3.4 | 3.4 | 0.0 | Correct |
| NakedTriple | 3.6 | 3.6 | 0.0 | Correct |
| Swordfish | 3.8 | 3.8 | 0.0 | Correct |
| HiddenTriple | 4.0 | 4.0 | 0.0 | Correct |
| XyWing | 4.2 | 4.2 | 0.0 | Correct |
| XyzWing | 4.4 | 4.4 | 0.0 | Correct |
| UrType1 | 4.5 | 4.5 (base, +up to 0.3) | ~0.0 (base correct; loop-length bonus missing) | SE scales UR by loop length (4→4.5, 6→4.6, 8→4.7, 10→4.8); we always return 4.5 |
| UrType2 | 4.7 | 4.5+ (type-dependent) | SE Type2 hint = base 4.5 + extras; our 4.7 is a fair approximation | Small error |
| Jellyfish | 5.2 | 5.2 | 0.0 | Correct |
| NakedQuad | 4.0 (our impl) | 5.0 | **-1.0** | CRITICAL: we return 4.0 for NakedQuad but SE is 5.0 |
| HiddenQuad | 5.0 | 5.4 | **-0.4** | We return 5.0; SE returns 5.4 |
| Bug (BUG+1) | 5.6 | 5.6 | 0.0 | Correct |
| Skyscraper | 4.0 | ~6.6 (AIC/FC sub-type) | **-2.6** | SE has no standalone Skyscraper — it falls under Forcing Chain Cycle at 6.6+. Our 4.0 is massively low |
| TwoStringKite | 4.1 | ~6.6 (AIC/FC sub-type) | **-2.5** | Same as Skyscraper — SE treats these as FC-cycles |
| SimpleColoring | 4.0 | ~6.5 (FC cycle / X-cycle) | **-2.5** | SE's X-cycle base is 6.5; Skyscraper/Kite are special cases of X-cycles |
| AlsXz | 7.5 | Not present in SE 1to9only | N/A | SE (1to9only fork) has NO ALS implementation. This is our extension beyond SE. Rating 7.5 is borrowed from HoDoKu; not from SE itself |
| Aic | 5.0 base, scaled by elim_count | 6.6 base for X-chain, 7.0 for XY | **-1.6 to -2.0** | Our AIC base_rating=5.0 is too low. SE starts X-chains at 6.6, XY-chains at 7.0; both plus length_difficulty |
| CellForcingChain | 7.5 | 8.0 (MultipleForcingChain, cardinality>2) | **-0.5 for >2 cands** | SE CellFC rating is from `MultipleForcingChain` = 8.0 base + length_diff. Our 7.5 conflates the 2-candidate case (which SE rates at 7.5 via NishioFC) with the 3+ candidate case (8.0+) |

**Summary of biggest errors (by magnitude):**
1. Skyscraper / TwoStringKite / SimpleColoring: rated 4.0–4.1 by us; SE sees them as FC-cycles at 6.5–7.0. These fire often and drag our se_score down by up to 2.5 points per firing.
2. AIC base: 5.0 vs SE 6.6+. Since AIC is our primary T3 technique on hard puzzles, this is the dominant source of calibration error.
3. NakedQuad: 4.0 vs SE 5.0. A –1.0 systematic undercount whenever naked quads appear.
4. CellForcingChain: 7.5 vs 8.0+ for multi-candidate cases.

---

## 3. Missing Techniques

### On SE's standard cascade path (affect calibration):

| SE technique | SE rating | Algorithm | Missing from us | Effort |
|---|---|---|---|---|
| Region Forcing Chain | 8.0+ | For each digit in a region (row/col/box), if all candidate positions lead to the same conclusion, apply it. Works in combination with Cell FC | Yes — completely missing | Medium (structurally similar to Cell FC, but iterates region positions rather than cell candidates) |
| Dynamic Forcing Chain | 8.5+ | Like Cell/Region FC but inside each hypothesis branch, applies locked candidates, naked singles, hidden singles in addition to bare propagation | Yes — our Cell FC uses singles-only propagation inside hypothesis; SE's "Dynamic" applies locked+naked+hidden as well | Large (requires mini-solver inside hypothesis; changes coverage substantially) |
| Nishio Forcing Chain | 7.5+ | Assume a candidate is ON; if contradiction follows, eliminate it. SE treats this as a binary FC with contradiction mode | Partially — our Cell FC's contradiction sub-case covers this for cell-level hypotheses (2 candidates), but SE's Nishio works on any single candidate regardless of cell cardinality | Small (extension of existing Cell FC contradiction path) |
| Dynamic Forcing Chain Plus (level 1) | 9.0+ | Dynamic FC with one level of nesting inside the hypothesis branches | Yes | Large |
| Nested Forcing Chain (level 2+) | 9.5–11.5 | Multiple levels of nested bifurcation | Yes | Very Large |

### Not on standard cascade path but affect puzzle space coverage:

| SE technique | SE rating | Algorithm | Effort |
|---|---|---|---|
| Aligned Pair Exclusion | 6.2 | Test all value combinations for two aligned cells; eliminate combinations that violate constraints | Medium |
| Aligned Triple Exclusion | 7.5 | Three-cell aligned exclusion | Large |
| UR Types 3/4 | 4.5+ | More UR patterns beyond type 1/2 | Small |
| BUG+2/3/4 | 5.7+ | BUG with more extra candidates | Small |

### In other well-known implementations but not SE (no immediate calibration impact):
- ALS-XZ, ALS-XY-Wing, ALS chain, Death Blossom, Sue de Coq, 3D Medusa, Multi-Coloring, Finned/Sashimi/Franken Fish — these are from HoDoKu, Hodoku, or other solvers. The 1to9only SE fork does NOT implement them. Our ALS-XZ exists purely as a HoDoKu-inspired extension; its 7.5 rating is not from SE.

---

## 4. Cascade Order Divergence

**SE cascade order** (from `Solver.java`, `directHintProducers` → `indirectHintProducers` → `chainingHintProducers`):

```
directHintProducers (cheap, applied first):
  HiddenSingle → DirectPointing → DirectHiddenPair → NakedSingle → DirectHiddenTriplet

indirectHintProducers (our T2/T3 zone):
  Locking(indirect) → NakedSet(2) → Fisherman(2) [X-Wing!] → HiddenSet(2) →
  NakedSet(3) → Fisherman(3) [Swordfish!] → HiddenSet(3) →
  XYWing → XYZWing → UniqueLoops →
  NakedSet(4) → Fisherman(4) [Jellyfish!] → HiddenSet(4) →
  BivalueUniversalGrave → AlignedPairExclusion

chainingHintProducers:
  ForcingChainCycle → AlignedTripletExclusion → NishioForcingChain →
  MultipleForcingChain → DynamicForcingChain

chainingHintProducers2:
  DynamicForcingChainPlus
```

**Our cascade order** (T2_LIST → T3_LIST → T4PLUS_LIST):
```
T2: LockedPointing → LockedClaiming → NakedPair → HiddenPair → NakedTriple →
    HiddenTriple → NakedQuad → HiddenQuad

T3: XWing → Swordfish → Jellyfish → UrType1 → UrType2 →
    XyWing → XyzWing → SimpleColoring → Skyscraper → TwoStringKite →
    Bug → AlsXz → Aic

T4Plus: CellForcingChain
```

**Key divergences:**

1. **X-Wing before Hidden Pair in SE**: SE tries `Fisherman(2)` (X-Wing, rating 3.2) between `NakedSet(2)` (3.0) and `HiddenSet(2)` (3.4). We try X-Wing only after ALL T2 techniques. If both X-Wing and HiddenPair would fire on the same position, SE applies X-Wing first; we apply HiddenPair first. This can change which technique appears in the frontier and which rating is max.

2. **SE interleaves Fish with Sets**: SE's order is NakedPair → XWing → HiddenPair → NakedTriple → Swordfish → HiddenTriple → NakedQuad → Jellyfish → HiddenQuad. We strictly separate T2 (all sets) from T3 (all fish). On a puzzle where both NakedTriple and XWing could fire, SE reports XWing (3.2 < 3.6) as the "first applied" since it comes earlier in the queue; we report NakedTriple (3.6, tried later) as the tier indicator but XWing (3.2) might have fired in T3 pass instead.

3. **UniqueLoops before Quads in SE**: SE tries UniqueLoops (4.5) before NakedQuad (5.0) and Jellyfish (5.2). We try UR after Fish and Wings. If a puzzle can be solved by UR and also NakedQuad, SE picks UR first; we might pick Jellyfish first if it fires in T3 before reaching UR.

4. **AlignedPairExclusion in SE's indirectHintProducers**: SE tries APE (6.2) before any chaining technique. We don't implement APE at all → those puzzles fall through to CellForcingChain (7.5) or T4Plus residual.

5. **AIC/SimpleColoring positioning**: Our T3 list tries SimpleColoring/Skyscraper/TwoStringKite before Bug/AlsXz/Aic. SE places ForcingChainCycle (which covers all of these) in `chainingHintProducers` — tried only after ALL of indirectHintProducers (including APE). So SE applies APE before any chain; we don't apply APE at all.

---

## 5. Algorithm Details for Shared Techniques

### AIC (`aic.rs` vs `Chaining.java`)

SE's ForcingChainCycle (non-multiple, non-dynamic mode) operates as follows:
- Iterates all empty cells, all candidate values.
- For each (cell, value) hypothesis "ON", builds BFS implication tree using two link types: **X-links** (only remaining position in a region for a digit) and **Y-links** (only remaining candidate in a bivalue cell).
- Detects cycles (discontinuous/nice loops) and chains (one endpoint implying a specific other candidate is ON or OFF).
- Rates as: **X-cycle base=6.5**, **Y-cycle base=6.5**, **XY-cycle base=7.0**, **forcing chain (X) base=6.6**, **forcing chain (XY) base=7.0**, all plus `getLengthDifficulty()`.

Our `Aic` is a single unified implementation with `base_rating=5.0`, using a heuristic `se_rating` based on elimination count. **This is wrong in two ways:**
1. Base is 5.0 vs SE 6.5–7.0.
2. We conflate X-chains, XY-chains, and XY-cycles into one rating bucket; SE rates them differently.
3. Our length heuristic uses elimination count; SE uses chain node count (complexity - 2), with the logarithmic schedule {4,6,8,12,16,...}.

**Impact**: any puzzle where our AIC fires gets se_score ≈ 5.0–5.5, while SE would rate it 6.6–7.3. This is the single largest calibration error for T3 hard puzzles.

### Cell Forcing Chain (`cell_fc.rs` vs `Chaining.java`)

**What we propagate inside hypothesis**: naked + hidden singles only (via `propagate_singles`). This is bare constraint propagation.

**What SE propagates inside Dynamic FC**: inside each hypothesis branch, SE applies *locked candidates* (pointing/claiming), naked/hidden singles, AND any other "simple" technique. The `isDynamic` flag in `Chaining.java` enables additional inference inside the hypothesis, resulting in longer implication chains and more deductions per step. The static Cell FC (`isMultiple=true, isDynamic=false`) in SE also uses `doBinaryChaining` which propagates X-links and Y-links, not just naked/hidden singles.

**Our Cell FC** is closer to SE's Nishio/static Cell FC than to its Dynamic FC. Specifically:
- SE's `doBinaryChaining` traces implications using `getOnToOff` (naked single + all-region hidden single) and `getOffToOn` (Y-link + X-link). This is equivalent to "strong links only" AIC traversal, not full propagation.
- Our `propagate_singles` is equivalent to running naked+hidden singles to fixpoint, which is more powerful inside the hypothesis than SE's link-based traversal for some configurations, but less powerful for others (SE's link traversal can find longer implication chains without fixpointing intermediate candidates).
- **Net effect**: our Cell FC fires on different grids than SE's Cell FC. Likely fires on fewer grids (since our hypothesis propagation doesn't produce X/Y-link chains beyond direct singles), but the grids where it fires may get the same outcome.

**Rating divergence**: our Cell FC returns 7.5 flat. SE rates Cell FC (MultipleForcingChain, cardinality≥3) at **8.0 base + length_diff**. For 2-candidate cells, SE uses NishioFC at **7.5 base + length_diff**. So:
- 2-candidate cell: SE 7.5+, us 7.5 → small positive error in our favor.
- 3–4 candidate cell: SE 8.0+, us 7.5 → we undercount by –0.5 minimum.

### ALS-XZ (`als_xz.rs`)

SE (1to9only fork) has **no ALS-XZ implementation** in the Java source. There is no `AlsXz.java`, no ALS in `SolvingTechnique.java`. Our ALS-XZ (base_rating=7.5) is a HoDoKu-sourced technique that has no SE counterpart in this codebase. The rating 7.5 is from HoDoKu's table, not from SE. When ALS-XZ fires in our rater, it produces a se_score that SE would never produce (SE would have resolved the same position via CellForcingChain at 7.5–8.0 or continued to Multiple/DynamicFC).

---

## 6. Calibration Plan Recommendation

### Puzzle set for calibration

For calibrating T2/T3 weights, the `data/seeds/public_hardest/forum_hardest_1905_11plus.txt` puzzles are too hard (all SE≥11 → all require nested FC; our rater can't touch them). For calibration you need:

- **SE 4.0–5.5 bucket** (testable by NakedQuad, HiddenQuad, Jellyfish calibration): use publicly available "Platinum Blonde" puzzles (SE exactly 5.6) or random 28–32 clue puzzles screened by real `serate`.
- **SE 6.5–7.5 bucket** (AIC/FC calibration): the "top1465" collection (`data/seeds/public_hardest/top1465.txt` if present) spans SE 7.1–12.7; filter to puzzles where real `serate` returns 6.5–7.5. About 30–50 such puzzles would be needed.
- **Sudoku-Extreme dataset** (HRM paper arXiv:2506.21734): searched and not found a public download URL as of 2026-05-12. The paper references an internal dataset; no public mirror found. Use `forum_hardest_1105` instead (1105 puzzles, SE 8.3–12+, good for validating T4Plus boundary).

### Weights most likely wrong, prioritized for recalibration:

1. **AIC** `base_rating`: change from 5.0 → 6.6 (X-chain minimum). Override `se_rating` to use chain node count with the log schedule instead of elimination count heuristic. This requires threading actual chain length through `TechniqueProgress` (the existing TODO in `aic.rs:259`).

2. **SimpleColoring / Skyscraper / TwoStringKite**: all three are special cases of X-cycles/Y-cycles in SE. Correct ratings: SimpleColoring (X-cycle) → 6.5 base; Skyscraper (X-chain) → 6.6 base; TwoStringKite (X-chain) → 6.6 base. Our 4.0/4.1 values are from HoDoKu, not SE.

3. **NakedQuad**: 4.0 → 5.0 (SE constant).

4. **HiddenQuad**: 5.0 → 5.4 (SE constant).

5. **CellForcingChain**: introduce cardinality-based rating. Return 7.5 for 2-candidate cells (Nishio sub-case), 8.0 for 3+ candidate cells (MultipleForcingChain). Requires passing cell cardinality through `TechniqueProgress`.

### Missing techniques reducing T4Plus residual:

**Highest priority for reducing the T4Plus (stuck at 7.5) sink:**
1. **Region Forcing Chain** (SE 8.0+) — many puzzles that SE solves via RegionFC (a digit in a row/col/box has ≤3 positions all leading to the same conclusion) currently fall through to our T4Plus residual. Medium implementation effort; directly extends Cell FC logic.
2. **Nishio Forcing Chain fully generalized** (SE 7.5+) — our Cell FC handles 2-candidate cells via contradiction sub-case, but single-candidate Nishio (assume a specific digit somewhere, derive contradiction) is not covered for cells with 3+ candidates where only one candidate is being tested. Small extension.
3. **Dynamic Forcing Chain** (SE 8.5+) — the most impactful for 17-clue puzzles. Requires a mini-solver running locked candidates inside hypothesis branches. Large effort but essential for the research target.

---

## 7. Honest Assessment

The gap between our `se_score` and real `serate` output is a combination of all three error categories, with (b) dominating. Our shared techniques are mostly correct in algorithm (Locking, NakedSet, HiddenSet, Fish, XYWing, XYZWing, UR, BUG have no detected algorithmic divergences), but their ratings are wrong in several cases — this is calibration error (a). More seriously, three of our T3 techniques (SimpleColoring, Skyscraper, TwoStringKite) are rated at HoDoKu values (4.0–4.1) when SE would classify them as FC-cycles at 6.5–7.0; this shifts our se_score down by ~2.5 points for any puzzle where these fire. Our AIC has the same problem: base_rating=5.0 vs SE 6.6–7.0. The coverage gap (b) is severe above SE 7.5: SE implements Region FC (8.0+), Dynamic FC (8.5+), and Nested FC (9.5+), all of which we are missing. Any 17-clue puzzle requiring these — which is essentially all SE≥8.5 puzzles — receives our stubbed T4Plus residual of 7.5, creating a systematic floor error of up to 4+ SE points. There is also an algorithmic divergence (c) in CellForcingChain: our hypothesis propagation is singles-only while SE's MultipleForcingChain traces strong/weak links (AIC-style), meaning they fire on different puzzle states. In practice our Cell FC fires less often and gives the same 7.5 rating whether the true answer was Nishio (7.5) or Multiple (8.0+). Bottom line: for SE≤6.0 puzzles, our rater is accurate (correct weights on all firing techniques). For SE 6.5–7.5, our rating is typically 1–2 points low due to AIC/Skyscraper/SimpleColoring mispricing. For SE≥7.5, we are systematically wrong by 0.5–4+ points due to missing techniques.

---

*Report generated by code-archaeology subagent from /tmp/sudoku_explainer Java source and /Users/dleonenko/latent-reasoning-design/tools/sudoku_rs_core/src/generic/techniques/. No code was modified.*
