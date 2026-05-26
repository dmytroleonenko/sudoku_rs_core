# Research note — algorithmic construction of BxB ≥ 6 puzzles

**Question.** What is the minimum-compute algorithm to generate a Sudoku puzzle with `shc::te::rate_bxb ≥ 6` *without using a known hard-puzzle seed*?

**Bottom line.** No proven from-scratch algorithm is known that beats search. Community state-of-the-art is mining + perturbation. This document records the methods we surveyed, why none of them is a closed-form constructive engine, and which path we are taking in practice.

## 1. Surveyed approaches

### Replica-exchange Monte Carlo (Watanabe 2013)

Watanabe (arxiv:1303.1886) proposed defining puzzle difficulty as an Ising spin-glass-like Hamiltonian and minimizing it via replica-exchange MC. He generated a puzzle claimed at the time to be "the world hardest." Two days after submission he added a v2 remark: **"the created puzzle can be solved easily by hand. Our definition of the difficulty is inappropriate."**

What this means for us: the Hamiltonian he chose did not align with Berthier B/BxB. The replica-exchange machinery is reusable, but only if the energy function is the actual BxB rating — and computing BxB is the expensive bit, so the MC approach degenerates into vicinity-search dressed up as MC.

**Verdict.** Not a free lunch. No useful algorithmic content beyond "do vicinity-search with simulated annealing."

### Expert-guided UA-based construction (community)

The hardest puzzles in the enjoysudoku community catalogue (Easter Monster, Golden Nugget, AI Escargot, Hanson-Marans, tarx0134, ...) were discovered by **expert hand-construction**, not by automated algorithm. The expert procedure roughly:

1. Identify a **Deadly Pattern / Unavoidable Set (UA) structure** in a candidate full-grid: a minimal set of cells whose values can be permuted without changing the solution count, requiring at least one clue inside.
2. Construct a clue set that places the **minimum cover** over the UA hypergraph, such that the remaining propagation is maximally entangled.
3. Place clues that induce structures like **SK Loops** — a circular pattern of two-candidate cells discovered during the solving of Easter Monster (Steve Kurzhals, 2008).
4. Verify uniqueness, rate, iterate.

The procedure is *not* a closed-form algorithm. Each new "hardest" puzzle is typically:
- A derivative of an earlier hardest (single-cell perturbation, digit swap, transposition), or
- A novel construction by someone with deep insight, requiring hours of human work plus rating compute.

**Verdict.** Theoretically the most compute-efficient method (avoids random-search waste), but the algorithmic component is not automatable today. The deadly-pattern + minimum-cover combinatorial search is itself an NP-hard problem.

### Replica-exchange / MCMC with BxB energy

If we use `shc::te::rate_bxb` as the energy function for replica-exchange MC (instead of Watanabe's Hamiltonian), each MC step costs one rating (~1.5 s). To converge to a local minimum in a basin containing BxB=6+, we'd need O(10³-10⁴) steps. Cost: similar to vicinity hill-climb but with the parallel-tempering escape from local minima — could find harder puzzles than plain hill-climb when basins are shallow.

**Verdict.** Reasonable enhancement to vicinity-search. Not implemented; consider for Phase 2 if vicinity alone stalls.

### Hard-coded "schema" generation

Some sudoku-puzzle-generator software (e.g. Sudoku Snake's generator) constructs puzzles starting from a **predetermined pattern of clue positions** (e.g. symmetric patterns) and randomized digit assignments, then filters by rating. The pattern bias drastically reduces the search space.

The hardest-known forum_hardest puzzles do **not** correlate with classical symmetric patterns — many have asymmetric clue placements with diagonal bands. The pattern that correlates is "clue density along certain bands," not visual symmetry.

**Verdict.** Could be useful as a heuristic for biased random sampling: sample clue positions from the empirical distribution of forum_hardest clue positions, fill digits, rate. Not implemented.

### Brute-force uniform random

Generate random minimal unique-solution puzzles (Royle-style backtracking), rate, keep if BxB ≥ 6. Hit rate ~10⁻⁶ based on forum_hardest density. **Cost: ~13 hours/puzzle on 32 threads.** Intractable but unbiased.

**Verdict.** Last resort. Useful only as a theoretical baseline.

## 2. The 17-clue cliff — and why it does NOT lead to BxB ≥ 6

17-clue puzzles are special: this is the proven minimum number of clues for a uniquely-solvable Sudoku (McGuire, Tugemann, Civario 2012, arxiv:1201.0749). The complete enumeration of all *essentially different* 17-clue minimal puzzles ran for ~7.1 CPU-years of compute (2012). The result is the Royle catalogue: 49,151 puzzles published, plus another ~7 added in subsequent re-checks.

**Empirical finding (2026-05-20).** We rated all 49,151 Royle puzzles via `shc::te::rate_bxb` (max-length=14, threads=10, total wall ~3 min). Result:
- BxB distribution: 100% at BxB=0 (i.e. solvable without T&E(2)).
- TE-depth distribution (100-sample): 52% TE-depth=0, 48% TE-depth=1.
- Zero puzzles at TE-depth ≥ 2.

**The 17-clue catalogue does not contain T&E(2) puzzles at all.** 17 clues is the uniqueness minimum, not the difficulty maximum. The community-curated hardest puzzles are 20-25 clues, not 17:

| Puzzle | Clues | Era |
|---|---|---|
| AI Escargot (Inkala 2006) | 24 | "World's hardest" 2006 |
| Easter Monster (community 2010) | 21 | T&E(2) prototype |
| Golden Nugget | 23 | T&E(2) |
| Hanson-Marans | 21 | T&E(2) |
| Tarx0134 | ~22 | T&E(2) |
| forum_hardest_1905_11plus median | ~22-24 | T&E(2) curated |

**Why hardest puzzles cluster at 20-25 clues, not 17:**
- 17-clue puzzles have so little starting information that their *uniqueness* is fragile. The constraints are tight enough that propagation usually solves quickly — solving is brittle but linear.
- 20-25 clue puzzles allow specific **entanglement structures** (SK loops, multi-cell deadly patterns, BUG configurations). These require enough digits placed to construct interlocking ambiguity sources but few enough to leave the candidate domain rich. The "sweet spot" for forcing T&E(2) is empirically 20-25.

**Consequence for our research question.** Royle catalogue mining (Path C in §1) is NOT a viable from-scratch path to BxB ≥ 6. Wrong corpus. The mining cost was cheap (~3 min) but the yield was zero.

The remaining from-scratch options:
- (B) vicinity hill-climb on random seed — basin-of-attraction issue, but at least addresses 20-25 clue space if seeded there.
- (D) UA-based construction — open algorithmic problem; described in §1.
- (E) NEW — vicinity climb on forum_hardest BxB=5 seeds — pragmatic but not "from scratch" in user's strict sense.

## 3. What we are doing

Given the above (especially the 2026-05-20 Royle empirical finding), our pipeline is:

1. **Done**: rated all 49,151 Royle puzzles. **Yield: zero BxB ≥ 6.** Wrong corpus — they're all TE-depth ≤ 1.

2. **Done**: added `--fitness bxb --target-bxb N` to `gen-vicinity` (commit `2dca117`). Hill-climbs from any seed pool using `shc::te::rate_bxb` as fitness.

3. **Next**: hill-climb from forum_hardest_1905_11plus BxB=5 seeds (1166 puzzles) toward BxB=6+. Strict-sense "seeded" — uses pre-existing curated corpus — but the seeds are themselves T&E(2), so basins should contain BxB=6+ local optima with reasonable frequency. Expected hit rate: 1-10% per seed at budget ≥ 500 iters.

4. **Open**: from-scratch (no community corpus) generation of BxB ≥ 6. Two surviving paths:
   - **UA-based construction** (path D from §1) — automatable but requires implementing UA enumeration + minimum hitting set ILP. Estimated 2000 LOC project. Expected hit rate 10-30% per attempt at ~30-130s per attempt → **~2-10 min per BxB=6 puzzle from scratch.**
   - **Vicinity from random 20-25 clue seed** (extending path B to the right clue band) — cheap to try, basins likely shallow, may stall at BxB=3-4.

5. **Deferred**: replica-exchange MC with BxB energy. Same compute order as vicinity, with parallel-tempering escape from local minima.

## 4. What the network-training pipeline needs

For curriculum / hardest-evaluation:

- Tier 0..7 (B=0..7 via gbraid_reverse): generate as many as needed, fast.
- Tier 8-9 (B=8-9): generate via gbraid_reverse at high k; budget compute.
- Tier BxB=2..5 (T&E(2) easy-to-moderate): currently only available via forum_hardest_1905_11plus filtering (5189+32117+10215+1166 = 48687 puzzles) and Royle mining. Will grow with vicinity-search.
- Tier BxB=6..7: currently ~79 known instances in forum_hardest + (expected) ~80-150 in Royle. Total ≤ ~250 puzzles. Vicinity-search may grow this 2-10×; new from-scratch constructions are an open research problem.

## 5. Citations

- McGuire G., Tugemann B., Civario G. (2012). "There is no 16-Clue Sudoku: Solving the Sudoku Minimum Number of Clues Problem." [arXiv:1201.0749](https://arxiv.org/abs/1201.0749).
- Watanabe H. (2013). "Difficult Sudoku Puzzles Created by Replica Exchange Monte Carlo Method." [arXiv:1303.1886](https://arxiv.org/abs/1303.1886). (Difficulty definition retracted in v2.)
- Berthier D. (2024). "Hierarchical Classifications in Constraint Satisfaction" (HCCS). Ch.4 on construction and equivalence classes.
- Royle G. "Minimum Sudoku" (2010 catalogue; mirror at [shadaj/sudoku](https://github.com/shadaj/sudoku/blob/master/sudoku17.txt)).
- Forum threads:
  - [enjoysudoku — "The hardest sudokus"](http://forum.enjoysudoku.com/the-hardest-sudokus-t4212.html)
  - [enjoysudoku — "Easter Monster Derivatives"](http://forum.enjoysudoku.com/easter-monster-derivatives-t34331.html)
  - [Sudopedia — "Deadly Pattern"](http://sudopedia.enjoysudoku.com/Deadly_Pattern.html)
- Kurzhals S. (2008). [SK Loops solution to Easter Monster](https://sudoku.allanbarker.com/sweb/extra/steaster/steaster.htm).

## 6. UA Pilot Results (2026-05-20)

Pilot implementation per `docs/specs/ua_constructor_pilot.md`. Code:
`src/shc/ua/mod.rs` (~880 LOC including tests) + CLI subcommands
`gen-ua-pilot` and `ua-kill-switch` in `src/cli.rs`.

**Implementation summary.** The UA enumerator is restricted to **two-digit
swap cycles** of even size 4..=12 (the standard pair-swap construction:
for every digit pair {d1,d2} of the 36 pairs, the 18 (d1,d2)-cells form a
3-regular bipartite graph by row/col/box partners, and every simple
alternating cycle is a UA). Multi-digit cycles (3+ digits) are **NOT**
enumerated — that's the standard pilot-vs-production gap. Minimum
hitting set is by greedy max-coverage. Uniqueness verified via
`shc::uniqueness::verify_unique_solution`. Because pair-only enumeration
is provably incomplete, a "repair pass" was added that iteratively adds
disambiguating cells when the greedy clue set leaves the puzzle
non-unique.

**Kill-switch (§13).** On 10 BxB=5 puzzles from `forum_hardest_sample1000`
(solved via `solve_unique` to get full grids), the enumerator returned
n_uas ∈ [421, 939] (median 688, mean 636). All grids had n_uas ≥ 100,
the threshold for proceeding. Per-size breakdown was healthy: size-4
counts 36–75, size-12 counts 91–354. Wall ≈ 2 ms per grid. **PASS.**
The enumerator is not under-counting in the relevant size range.

**Pilot (§9), 100 attempts, seed=42, max_ua_size=12, target_bxb=6,
threads=8:**

| Metric | Value |
|---|---|
| Attempts | 100 |
| Valid (unique after repair) | 100 (100%) |
| BxB distribution | BxB=0: 100 |
| Target hits (BxB ≥ 6) | 0 (0.00%) |
| Median clue count | 31 |
| Min / max clue count | 25 / 36 |
| Median wall / attempt | 2 ms |
| p90 wall / attempt | 6 ms |
| Per-stage median walls | grid=0 ms, ua_enum=2 ms, min_cover=0 ms, uniqueness=0 ms, rate_bxb=0 ms |

**Verdict: FAIL.** 0 BxB ≥ 6 hits in 100 attempts (spec §3 / §9 fail
threshold is < 0.1%, met decisively). Walls are far under the 60 s
budget — the limiting factor is not throughput; it's that the
constructed puzzles are systematically *too easy* (every single one
solvable by naked / hidden singles, BxB=0).

**Diagnosis.** Two compounding issues, each independently fatal at this
size cap:

1. **Missing UA classes.** Pair-only enumeration up to size 12 found
   ~640 UAs per grid, but on the *constructed* puzzles, greedy left
   *every* one of 100 attempts non-unique — meaning every constructed
   clue set fails to hit at least one UA outside the enumerated set.
   The remaining undetected UAs are almost certainly the 3-digit cycle
   classes (and 2-digit cycles of length 14+), which the literature
   (Berthier HCCS Ch. 4; enjoysudoku forum "Finding unavoidable sets")
   notes are needed for puzzles at the 17–22 clue floor. Bumping
   `max_ua_size` to 18, 24 did NOT fix uniqueness (still 0/5 unique),
   confirming the missing class is multi-digit, not larger 2-digit
   cycles.
2. **Repair pass over-saturates.** With pair-only UAs, the repair
   loop kept adding cells until unique — landing at median 31 clues,
   which is well above the BxB≥6 region (empirically 22-25 clues).
   The greedy hitting set + repair cannot reach the 17–25 clue range
   where BxB≥6 puzzles live. ILP min cover *might* shrink the hitting
   set by 2–4 clues, but the bigger blocker is enumeration completeness:
   without 3-digit UAs the optimal pair-only cover is itself non-minimal
   w.r.t. true puzzle uniqueness.

**Recommended remediation if revisited.**
- Add 3-digit cycle enumeration (size 6–12). For each unordered triple
  of digits, enumerate even-length alternating walks in the digit-triple
  graph. Berthier estimates this captures another ~200–500 UAs per
  grid.
- Replace greedy with ILP exact min cover.
- Pre-screen grids: only accept grids with ≥ 1000 enumerated UAs
  (across all digit-tuple classes) — these correlate with hard
  underlying minimal puzzles per Berthier.
- Consider hybridizing with vicinity-search: use UA-cover output as
  an initial clue set, then run vicinity/SA to find adjacent puzzles
  with BxB ≥ 6.

**Pivot recommendation.** Per spec §9 FAIL clause, do not commit to
the full ~2000 LOC UA-construct ILP version on the back of this pilot.
The 100% non-unique baseline (before repair) is the load-bearing
signal: pair-only UA enumeration is genuinely insufficient, and the
full version's main upgrade (ILP) does not fix that. Either invest in
the 3-digit-cycle extension first (smaller, targeted experiment) or
pivot to GFlowNet / community-curation augmentation as the production
path to BxB ≥ 6 puzzle generation.

## 7. UA Pilot Round 2 Results (2026-05-20) — 3-digit cyclic-rotation UAs

Round 2 extends the round-1 enumerator with **3-digit cyclic-rotation UAs**
(`enumerate_3digit_ua_cycles` + combined `enumerate_all_uas`). 2-digit
pair-swap enumeration retained verbatim alongside.

**Algorithm.** A 3-digit UA on digits {A,B,C} is a set S where every unit
(row/col/box) contains either 0 or all 3 of its (A,B,C) cells. Equivalently
S is a union of connected components in the 27-cell graph where two
cells holding {A,B,C}-digits are linked iff they share a row/col/box.
Rotating digits A→B→C→A inside any such S preserves Sudoku validity.
Enumeration: 84 unordered triples × union-find over 27 cells. ~3 ms / grid.

**Kill-switch results** (10 BxB=5 puzzles from `forum_hardest_sample1000`):

| Metric | 2-digit only | 3-digit only | Combined |
|---|---|---|---|
| min | 421 | 0 | 422 |
| median | 688 | **0** | 688 |
| max | 939 | 2 | 939 |
| mean | 636.2 | 0.1 | 636.3 |

At `max_ua_size=12`: median of 0 three-digit UAs per grid. At
`max_ua_size=24`: still median 0, max 2. **Critical empirical finding:**
on essentially every random or BxB=5 grid, every digit triple's 27-cell
unit-triple-closure graph is a single connected component of size 27.
Components of size ≤ 24 occurred in 1/10 grids only. The "compact 3-digit
UAs" the literature alludes to are rare in practice for random grids.

**Pilot (§9), 100 attempts, seed=42, max_ua_size=12, target_bxb=6,
threads=8:**

| Metric | Round 1 (2-digit) | Round 2 (2+3-digit) |
|---|---|---|
| Attempts | 100 | 100 |
| Pre-repair unique | 0% | **0%** (no change) |
| Valid (post-repair) | 100% | 100% |
| Clue count pre-repair median | n/a | 23 |
| Clue count post-repair median | 31 | **31** (no change) |
| Min / max clue count post | 25 / 36 | 25 / 36 |
| BxB distribution | BxB=0: 100 | BxB=0: 100 |
| Target hits (BxB ≥ 6) | 0 (0.00%) | **0 (0.00%)** |
| Wall total | 0.2 s | 0.1 s |

At `max_ua_size=24` (separate run, `/tmp/chain_rerate/ua_pilot_round2_max24.jsonl`):
pre-repair unique rate creeps from 0% to 1/100, post-repair median clue
count still 31, still 0 hits. So even larger 3-digit components do not
fix the saturation.

**Verdict: FAIL (unchanged from round 1).** Per spec §9, < 0.1% hits.

**Comparison to round 1.**
- Pre-repair uniqueness: 0% → **0% (unchanged)**. 3-digit cycles are *not* the missing UA class.
- Median clue count: 31 → 31 (unchanged).
- Median pre-repair clue count: 23 (this is in the BxB≥6 sweet spot, but every such cover is non-unique).
- All 8 unit tests + 25 `shc::` tests pass.

**Diagnosis update.** The hypothesis "3-digit cycles fix the
uniqueness problem" is empirically FALSIFIED. The 27-cell closure graph
collapsing to a single component means structurally that 3-digit UA
classes contribute essentially zero small UAs on random grids, despite
existing in principle. What IS missing must therefore be:

1. **4-digit and higher cyclic UAs** — same construction generalizes:
   for an ordered cycle (A→B→C→D→A) on 4 digits, the 36-cell unit
   closure graph may have smaller components. But by the same argument
   used here, the union-find of 36 cells across 27 units is *even more*
   likely to be a single giant component. Speculatively, 4-digit cycles
   contribute even fewer small UAs than 3-digit.
2. **Non-cyclic (non-rotational) UAs** — e.g. the so-called
   "Templates" / "SK loops" / "multi-cycle compound UAs" described in
   the enjoysudoku forum thread "Finding unavoidable sets". These are
   NOT closure components of single-permutation rotations; they are
   formed by *combinations* of pair-swaps + cycle-swaps interacting
   across digit groups. Enumerating them is combinatorially harder
   (requires searching over the orbit structure of all valid grid
   automorphisms agreeing on the complement of S).
3. **The pair-swap-cycle DFS may itself be incomplete at size ≤ 12.**
   Our 2-digit enumeration up to size 12 captures even cycles 4..12.
   Cycles of length 14, 16, 18 also exist and were partially measured
   at `max_ua_size=24` in round 1 (each of the 10 grids had additional
   200-400 UAs at sizes 14-18). But round 1 already showed bumping to
   18/24 did not help uniqueness — confirming that the missing class is
   structural, not just bigger 2-digit cycles.

**Pivot recommendation (strengthened from round 1).** The UA-cover
approach with closed-form enumerable UA classes (2-digit, 3-digit, and
plausibly k-digit cyclic rotations) does not produce unique puzzles
in the BxB≥6 clue range. The pre-repair median of 23 clues with 0%
uniqueness is the load-bearing signal: the missing UAs require either
(a) full SAT-based UA discovery (enumerate via difference between
multiple solutions, expensive per-grid), or (b) abandoning
construction-from-UAs entirely.

Recommended next steps, ordered by expected value:
- (best) Pivot to **multi-solution-enumeration UA discovery**: for each
  candidate grid, run a bounded all-solutions solver on its "minimal
  remaining clue set", extract differing cells as UAs. This finds the
  true UAs of the grid without enumerating any structural family.
  Combined with greedy hitting set, this should achieve pre-repair
  uniqueness > 80% — directly testable.
- (second) Hybridize UA-cover with vicinity-search/SA: use round-2's
  median-23-clue pre-repair non-unique cover as the seed; run mutation
  + BxB rating to find adjacent unique BxB≥6 puzzles. The seed is
  *much* closer to the manifold than random clue sets.
- (last resort) ILP min cover on the enumerated 2+3-digit UAs. Will
  shrink post-repair clue counts by 2-4, still won't reach BxB≥6 because
  the underlying enumeration is structurally incomplete.

Do NOT iterate further on cyclic-rotation UA classes (4-digit, etc.).
The empirical 27-cell single-component pattern repeats at higher k,
and the marginal lift is bounded.

**Artifacts.**
- `tools/sudoku_rs_core/src/shc/ua/mod.rs` (+221 LOC: 3-digit enumerator + tests).
- `tools/sudoku_rs_core/src/cli.rs` (kill-switch now reports 2d/3d split; pilot tracks pre-repair uniqueness).
- `/tmp/chain_rerate/ua_pilot_round2_results.jsonl` (max_ua_size=12).
- `/tmp/chain_rerate/ua_pilot_round2_max24.jsonl` (max_ua_size=24).


## 8. Vicinity Baseline Results (2026-05-20) — pre-GFlowNet calibration

After UA pilot rounds 1+2 FAIL and GFlowNet v1 spec PAUSED on FATAL double-review, ran vicinity-climb empirical baselines on the forum_hardest BxB=5 seed pool (27 puzzles extracted from sample1000 rating) to (a) calibrate whether *any* local-search approach can lift BxB=5 → BxB≥6 and (b) collect diagnostic counters to inform GFlowNet v2 redesign.

### Vicinity sweeps

| Run | Mutator | n | Budget | Mutations | Uniq-fail | Rated | BxB≥6 hits |
|-----|---------|---|--------|-----------|-----------|-------|-----------|
| v1  | digit-swap (cli.rs old) | 1,2 | 300   | 16 200  | ~16 050 (99%) |   1  | 0 |
| v3  | digit-swap (cli.rs old) | 1,2,3 | 5000 | 135 000 | 29 897 (22%) | 154 | 0 |
| v4  | digit-swap (cli.rs old) | 4,5,6 | 5000 | 135 000 |  6 669       |   1  | 0 |
| v5  | digit-swap (cli.rs old) | 1,2,3 | 20 000 | 540 000 | 78 288 (15%)| 340 | 0 |
| v6  | **v2 solution-aware**   | 1,2,3 | 5000 | 135 000 | 113 303 (84%) | **9704** | 0 |
| v7  | v2 solution-aware       | 5,10,20 | 3000 | 81 000 | 80 832 (99.8%) | 168 | 0 |

### Verdicts

1. **The naive digit-swap mutator (cli.rs:3919, original) was wrong-by-construction.** Changing a clue digit to a random non-self value rarely yields a puzzle whose unique solution agrees with the new digit — 99%+ uniqueness rejection. Confirmed empirically across radii.

2. **The v2 solution-aware mutator works on throughput axis** (63× more candidates reach `rate_bxb` than v1) but **does NOT find BxB≥6.** 9704 valid unique puzzles produced by single-step add/remove moves from 27 BxB=5 seeds; **zero** reached BxB≥6. This is the strongest empirical signal of the round.

3. **Multi-step moves don't bridge the gap either.** v7 with n=5,10,20 hit 99.8% uniqueness rejection (large moves destroy puzzle structure); the 168 surviving candidates also failed to reach BxB=6.

4. **BxB=5 puzzles in forum_hardest are not single-move adjacent to BxB=6.** Either (a) BxB=6 puzzles live on different underlying full grids G, or (b) reaching BxB=6 requires coordinated multi-cell moves that simultaneously preserve uniqueness and increase difficulty — which random + uniqueness-filter cannot find.

### Consequences for GFlowNet v2 redesign

- **Sequential clue-revelation on a single fixed G is structurally empty for BxB≥6** (confirms UA pilot + this baseline). The GFlowNet v1 spec MDP is dead.
- **Single-G search is dead.** The action space must include changing G (the underlying full solution grid). State should be (G, partial-clue-mask), action space includes G-modification moves (digit swaps that preserve full-grid validity).
- **BxB=6 seeds needed for connectedness test.** Rating forum_hardest full corpus (5k subsample running now) will yield ~8 BxB=6 seeds. Vicinity FROM BxB=6 seeds will tell us if BxB=6 puzzles cluster (curatable family) or are isolated points (require full grid-space search).

### Files
- Code: `src/cli.rs` `bxb_mutate_once_v2`, `BxbMutator` enum, `--mutator` / `--allow-clue-count-change` CLI flags, manifest `diagnostics` block. Commits `1f0218a` (parallelization+counters), `0b484ad` (v2 mutator + 5 sweeps).
- Raw runs: `/tmp/chain_rerate/vicinity_baseline_v{3,4,5,6,7}/{bxb6_out.parquet,manifest.json,console.log}`.

## 9. Connectedness test — BxB=6 puzzles are isolated (2026-05-20)

After §8 vicinity baseline from BxB=5 seeds returned 0 hits, ran a critical follow-up: vicinity FROM the BxB=6 seeds directly to test whether BxB=6 puzzles cluster (curatable family) or are isolated points in mutation space.

### Pre-requisite: corpus rating

Rated a random 5000-puzzle subsample of `forum_hardest_1905_11plus` (48 766 total) via `rate-shc-batch --classification bxb`. Wall: ~30 min at 4 threads. BxB distribution:

| BxB | Count | Fraction |
|-----|-------|----------|
| 2   |  537  | 10.74%   |
| 3   | 3312  | 66.24%   |
| 4   | 1019  | 20.38%   |
| 5   |  124  |  2.48%   |
| 6   |    8  |  0.16%   |
| 7+  |    0  |  0.00%   |

Extrapolated to full corpus: ~78 BxB=6 puzzles (matches community references). No BxB≥7 found in subsample.

### Vicinity from BxB=6 seeds

8 BxB=6 seeds, v2 mutator, n∈{1,2,3}, budget 10k iters, threads=6, 80 000 mutations:

| Counter | Value | % of mutations |
|---------|-------|----------------|
| `mutations_proposed` | 80 000 | 100.0% |
| `canonical_dup`      | 11 410 | 14.3%  |
| `uniqueness_fail`    | 63 174 | 79.0%  |
| `rated`              |  5 102 |  6.4%  |
| `above_target` (≥6)  |      **0** | 0.00%  |

**Every one of the 5102 rated candidates dropped to BxB<6.** Single-step mutation from a BxB=6 puzzle NEVER lands on another BxB=6 puzzle in this sample.

### Vicinity from 124 BxB=5 seeds (extended pool)

| Counter | Value |
|---------|-------|
| seeds            | 124    |
| mutations        | 372 000 |
| uniqueness_fail  | 321 836 (86.5%) |
| rated            | 28 249  (7.6%)  |
| above_target (≥6) | **0**           |

124 seeds × ~228 rated each = 28 249 valid puzzle candidates around BxB=5 grids; zero reached BxB=6.

### Verdict

**BxB≥6 puzzles are isolated extrema in puzzle-mutation space.** Two complementary observations:
1. The neighborhood of BxB=5 puzzles contains NO BxB=6 puzzles (28k rated, 0 hit).
2. The neighborhood of BxB=6 puzzles contains NO BxB=6 puzzles (5k rated, all dropped to ≤5).

This is a *much* stronger structural claim than "vicinity is empty". The BxB=6 puzzles in `forum_hardest` are point-extrema: any single-step solution-aware perturbation (add/remove/swap with clue-count change allowed) drops them to BxB≤5. They are unreachable AND unconnected.

### Consequences

**For puzzle generation.** Algorithmic generation of BxB≥6 via local search is empirically dead. The remaining options are:
- (a) Random sampling of full grids + UA-construct (UA pilot rounds 1+2 already showed this is empty at the structural radius we explored).
- (b) Action space that includes GRID-LEVEL mutations (swap two cells with same row/col/box pattern in the underlying full solution G), combined with clue-mask mutations. This is the GFlowNet v2 redesign axis.
- (c) Brute-force enumeration over canonical grid classes (Watanabe-style Monte Carlo). Compute-bound, no learnable structure.
- (d) **Stop generating, use existing forum_hardest corpus directly.** The ~78 BxB=6 + 1200 BxB=5 puzzles in the curated corpus are sufficient for training-set / eval-suite purposes in the upstream latent-reasoning project. We don't need to generate new ones if we have enough for training.

**For the latent-reasoning project's actual goal.** The meta-goal (per `CLAUDE.md`) is inducing multi-step reasoning on sudoku, not maximizing puzzle hardness for its own sake. forum_hardest already gives us:
- ~80% of puzzles at BxB=3 (easy, T&E(0)-solvable training bulk)
- ~20% at BxB=4 (T&E(1)-solvable, mid difficulty)
- ~2.5% at BxB=5, ~0.16% at BxB=6 — sufficient hard-bucket eval slices

Generating MORE BxB=6 puzzles would not change the substrate's discriminative power. The substrate is already stratified enough for the OOD/depth questions in §0 of `CLAUDE.md`.

### Decision

**Pause puzzle generation. Pivot to using existing forum_hardest corpus.** Specifically:
1. Rate the FULL 48 766-puzzle corpus (extrapolated ~10 GPU-hours at 4 threads). Output: a parquet with `puzzle, bxb, wall_ms` columns.
2. Stratify into difficulty buckets (T0..T5+) per `difficulty_scale.md` §3 tier mapping.
3. Hand the stratified corpus to `utm-jax` as the training+eval substrate.
4. The GFlowNet v2 spec remains paused. Reopen if the upstream training discovers it needs more BxB≥6 puzzles than the natural corpus contains (~78). Until then, do not pursue.

This closes the puzzle-construction research thread for the latent-reasoning v6 work. Generation can be re-opened later if there's a concrete demand signal.

### Files
- Code: `bxb_mutate_once_v2` (commit `0b484ad`), this analysis (commit upcoming).
- Raw runs: `/tmp/chain_rerate/forum_5k_bxb/rated.jsonl`, `/tmp/chain_rerate/vicinity_from_bxb{5_124,6}/*`.
- Seed files: `/tmp/chain_rerate/seeds_bxb{5,6}.txt` (124, 8 puzzles).

## 10. §9 verdict OVERTURNED — naive vicinity ≠ BRT-correct vicinity (2026-05-20)

External methodology input (user-provided expert synthesis) reveals the §8–§9 vicinity FAIL was based on a fundamentally **wrong mutation operator**. The empirical result ("BxB≥6 isolated") holds only for the naive operator we used; it does NOT hold for the canonical mith / Methuselah vicinity-search procedure used by the expert community.

### What we did wrong

Our `bxb_mutate_once_v2` operates as: `puzzle → solution-aware single-cell add/remove → uniqueness check → rate_bxb`. This skips the two critical steps that make vicinity search converge in the expert literature:

1. **BRT-expansion via Singles** (missing). Before mutation, the puzzle P should be expanded by adding ALL clues at cells derivable via Naked/Hidden Singles. This temporarily moves P into a simpler coordinate of the poset, but it stays inside the same BRT-equivalence class — i.e. the same topological neighbourhood under BRTinc.
2. **Re-minimization** (missing). After the `go{-p+q}` step (remove p clues, add q clues from the target solution), the result must be re-minimized: iteratively remove every redundant clue. This snaps back to the minimal-clue poset surface; without it, the search wanders off the difficulty manifold.

Without (1)+(2), single-cell perturbation drops out of the BRT neighbourhood almost surely, hence the 100% drop below BxB=5 we observed.

### What the canonical procedure does

Per mith's modification of Methuselah's vicinity search (expert sudokuwiki / enjoysudoku discussions):

```
fn brt_vicinity_step(P: MinimalPuzzle, solution: FullGrid, p: usize, q: usize):
    P_ext = brt_expand(P, solution)              // add Singles-derivable clues
    P_mut = go_p_q(P_ext, p, q, solution)        // remove p, add q from solution
    P_min = reminimize(P_mut)                    // remove redundant clues until minimal
    if is_unique(P_min) and bxb(P_min) >= target:
        emit P_min
```

The continuity theorem (BRTinc topology + classification functions are continuous) GUARANTEES this stays inside the difficulty manifold with high probability. Our naive procedure had no such guarantee.

### Algebraic-construction track (Tridagons / Eleven's replacement)

A complementary deterministic generation route: **tridagon-based construction**. Tridagons ("Thor's Hammer") are algebraic obstructions in 4 boxes forming a rectangle across 2 bands × 2 stacks, where 3 cells in each box hold a triplet {a,b,c}. Cyclic parity in 3 boxes vs 1 forces a non-3-colorable graph → unique solution requires "guardian" cells with extra candidates → solving requires deep OR-chains → automatic BxB≥6 / T&E(3).

**Eleven's replacement technique** (in SudoRules / enjoysudoku):
1. Identify cyclic triplet group in 3 target cells of a block.
2. Pick a substitute candidate (even one currently excluded).
3. Permute target digits within the triplet.
4. Apply global isomorphism (relabeling) to restore consistency.

This yields new non-isomorphic BxB≥6 puzzles in O(1) per sample, FROM A SINGLE PARENT TEMPLATE. Pure algebraic, no rating cascade needed (difficulty guaranteed by construction).

### SHC cascade filter — we already have the primitive

The expert pipeline uses SHC in "fast filter mode": `max-length = 6`. If `shc::te::rate_bxb` returns the buffer-overflow sentinel (our `-3` equivalent), the puzzle is provably NOT solvable by braids of length ≤ 6, hence BxB ≥ 7. Our existing `rate_bxb` already supports this: pass `max_length=6` and treat the -3 return as a HIT.

Expected speedup: full rate_bxb at max_length=14 is 1-5 s/puzzle; at max_length=6 it's sub-100ms because the search either terminates fast or overflows fast.

### Plan revision

Vicinity-via-naive-mutator track: closed (verdict valid for that operator).

New tracks to implement:
- **(A) BRT-correct vicinity**: implement `brt_expand` + `go_p_q` + `reminimize` mutator (mutator v3). Re-run vicinity from 8 BxB=6 + 124 BxB=5 seeds. Expected hit rate per literature: 10-30% non-isomorphic BxB≥6 emit per attempt.
- **(B) Tridagon constructor**: implement tridagon-pattern detection in solution grid + Eleven's replacement permutation engine. Test on the 8 BxB=6 seeds first (do any of them contain tridagons?). If yes, generate variants. If no, attempt synthesis from random full grids that admit tridagons.
- **(C) SHC fast-filter (max_length=6)**: confirm sentinel -3 contract; use as the cascade prefilter for both (A) and (B) outputs before full BxB rating.

Code requirements:
- `brt_expand(puzzle, solution) -> puzzle_with_more_clues`: iterative Singles propagation.
- `reminimize(puzzle) -> minimal_puzzle`: iteratively remove any clue whose absence still leaves the puzzle unique-solution.
- `go_p_q(puzzle, p, q, solution, rng) -> Option<puzzle>`: combined remove-p / add-q operator.
- `tridagon_detect(solution) -> Vec<TridagonInstance>`: find 4-box rectangles with cyclic triplets.
- `eleven_substitute(puzzle, tridagon, rng) -> Vec<puzzle>`: triplet permutation + relabel.

These replace `bxb_mutate_once_v2` as the canonical operator. Mutator v3 = BRT-correct; mutator v4 = tridagon-Eleven.

The §9 "puzzle-construction thread closed" decision is REVERSED. Reopen Task #12 with redesigned spec.

## 11. v3 (mith BRT-correct) implemented + FAIL — root cause + SHC cascade benchmark (2026-05-21)

### Mutator v3 result

Implemented `bxb_mutate_once_v3` = `brt_expand` (Singles closure via shc::tb::propagate L0) → `go_p_q(p,q)` → `reminimize` per §10 spec. Benchmark on 8 BxB=6 seeds, p=q∈{2,3,4}, budget 1000, threads 8: **0/24 000 emitted; 8000/8000 uniqueness_fail.**

**Root cause (verified).** L0 singles-closure on the 8 BxB=6 seeds is EMPTY: every seed has clue_count=22 and ext_clue_count=22. The 22-clue forum_hardest BxB=6 puzzles are at maximal information content; no Singles available. `brt_expand` becomes a no-op → v3 collapses to "remove 2 add 2" on an already-tight minimal puzzle, which destroys uniqueness 100% of the time.

**Mith vicinity preserves difficulty class; it does not climb the manifold.** This is consistent with the continuity-of-classification theorem (continuous ≠ monotone-strict-increasing). The procedure produces non-isomorphic CLONES of similar-difficulty seeds, not new harder puzzles. For boundary-mining (BxB=5→6) it's the wrong tool.

### SHC cascade prefilter benchmark

Tested `rate_bxb(max_length=N)` as cheap prefilter for "BxB ≥ N+1". Sentinel is `Err(RateError::Unclassifiable)` when no buffer_overflow occurred (src/shc/te.rs:365-419). Zero false-positives across 27 BxB=5 and 8 BxB=6 seeds.

| config (ml, buf)      | seeds   | n  | median wall | real BxB returned        | sentinel |
|-----------------------|---------|----|-------------|--------------------------|----------|
| 14, 1048576           | BxB=6   |  8 | 13 578 ms   | 8/8 BxB=6                | 0        |
| 6,  1048576           | BxB=6   |  8 | 13 917 ms   | 8/8 BxB=6                | 0        |
| 14, 1048576           | BxB=5   | 27 | 10 056 ms   | 27/27 BxB=5              | 0        |
| 6,  1048576           | BxB=5   | 27 | 10 042 ms   | 27/27 BxB=5              | 0        |
| 6,  1048576           | B(SHC)=8/9 | 12 | 4 ms     | 12/12 BxB=0              | 0        |
| 6,  1048576           | forum easy | 8 | 484 ms    | 8/8 BxB∈{2,3}            | 0        |

**Key finding.** The "sub-100ms at max_length=6" claim in §10 is FALSE on boundary BxB=5/6 puzzles. Cost is in the *successful* n2-sweep (n2=BxB), which runs identically at max_length=6 or 14 for BxB≤6 puzzles. max_length=6 only saves doomed n2=7..14 sweeps — relevant only when target really is BxB≥7.

**Recommended cascade** (verified):
- `max_length=2, buf=4096`: <1 ms, rejects BxB≤2.
- `max_length=4, buf=4096`: ~0.5 ms on easy, ~seconds at boundary; sentinel on BxB≥5.
- `max_length=6, buf=4096`: 10s on boundary BxB=5/6; sentinel on BxB≥7 (zero FP observed).
- `max_length=14, buf=1M`: same regime, full rating.

Speedup between BxB≤4 and BxB=5 is 3 orders of magnitude; between max_length=6 and 14 at fixed boundary BxB it's negligible. Cascade useful only when input is easy-dominated.

### Decision

- v3 (mith BRT-correct) is implemented and correct per literature, but EMPIRICALLY INAPPLICABLE to forum_hardest BxB=6 seeds (no Singles slack). Keep code for future use on non-minimal seeds.
- SHC cascade prefilter: API works (Unclassifiable sentinel), but not cheap on boundary. Document in code, don't wire as automatic filter.
- Next: implement mutator v4 = tridagon detection + Eleven's replacement (deterministic algebraic construction; one parent template → many BxB≥6 variants in O(1) per sample). This is the remaining unblocked path.
- Forum_hardest full corpus rating (Task #13) continues in background — produces stratified eval substrate regardless of generation outcome.

## 12. v4 tridagon + Eleven's replacement — structural FAIL (2026-05-21)

Implemented `shc::tridagon::{detect_tridagons, eleven_replace}` + `BxbMutator::V4`. Two empirical findings:

### Census on 8 BxB=6 seeds

| Seed | # detected tridagons | Notes |
|------|---------------------|-------|
| #1–#5, #8 | 0 | No transversal-of-triplet across any 2×2 box rectangle |
| #6 | 1 | triplet={1,3,4}, boxes={1,2,7,8}, parity=[T,T,F,F] (even total) |
| #7 | 1 | triplet={1,6,7}, boxes={3,4,6,7}, parity=[T,T,F,F] (even total) |

**Only 2/8 = 25% of forum_hardest BxB=6 seeds contain ANY tridagon**, and both have EVEN parity (not the canonical odd-parity "true" Thor's Hammer obstruction the literature describes as forcing BxB≥6). The tridagon structure is NOT a universal property of BxB=6 puzzles — at best it explains a minority sub-family.

### Eleven's replacement = isomorphism

Benchmark on 8 seeds, mutator v4: 200 successful `eleven_replace` calls → 200 canonical_dup → **0 emitted puzzles**. Reason: `generic::canonical::canonical_hash` already canonicalizes the 9! digit-relabel orbit. Eleven's relabel-only operator produces puzzles that are canonically identical to their parents.

The literature's "non-isomorphic variant" claim must rely on an additional **structural permutation INSIDE the triplet** (changing which cell of the tridagon holds which of the 3 digits — a permutation that's NOT a global relabel). The original §10 spec elided this step; the implementation only did the global relabel. Even with the correct triplet-permutation implemented:
- Applicable only to 2/8 seeds (coverage ceiling).
- Each parent yields O(triplet_perms × box_choices) ≈ tens of variants — bounded, not "millions".

### Three-path verdict

| Path | Operator | Hits on BxB=6 seeds | Root cause of FAIL |
|------|----------|---------------------|-------------------|
| v1   | digit-swap (cli.rs old) | 0/154 rated | Operator never lands on solution |
| v2   | solution-aware add/remove | 0/9704 rated | Single-step doesn't reach harder puzzles |
| v3   | mith BRT-correct | 0/24 000 emitted | brt_expand no-op on tight 22-clue seeds |
| v4   | tridagon + Eleven | 0/200 v4_emit (all dup) | Eleven = relabel-isomorphism; tridagon present in 2/8 seeds only |

Across 4 mutator generations and ~6 vicinity sweeps, **0 new BxB ≥ 6 puzzles synthesized** beyond what we already had.

### Final decision

**Pause puzzle-generation research.** The original §9 decision (use forum_hardest corpus as substrate) is reaffirmed. The §10 reversal based on external methodology, while methodologically sound in the abstract, did NOT yield generation in practice on this corpus.

Forward direction:
- Task #13 (forum_hardest 48 766 full corpus rating) continues in background; produces stratified eval substrate.
- For latent-reasoning training: use the ~78 BxB=6 + 1166 BxB=5 + 16 000 BxB=4 + 32 000 BxB≤3 (extrapolated from §9 sample) as stratified buckets per `difficulty_scale.md` tiers T0..T5.
- Generation research can resume IF utm-jax training discovers concrete need for more BxB≥6 puzzles than the natural corpus contains. Until that signal, the cost-benefit favors using what we have.

This closes Tasks #7, #14, #15, #16. Task #13 remains in_progress.

## 13. Full corpus rated — substrate ready (2026-05-21)

Rated all 48 766 puzzles of forum_hardest_1905_11plus via `rate-shc-batch --classification bxb --max-length 14 --buffer-size 1048576` on remote host minis (16 threads, x86_64, Ubuntu 24.04). Total wall 2h56m, CPU 46.5 h, 0 errors.

| BxB | Count | Fraction   | Median wall (ms) | Max wall (ms) |
|-----|-------|------------|------------------|---------------|
| 2   |  5189 | 10.641%    |    967           |   8 693       |
| 3   | 32117 | 65.859%    |  1 914           |  18 427       |
| 4   | 10215 | 20.947%    |  5 492           |  34 282       |
| 5   |  1166 |  2.391%    | 13 981           |  47 048       |
| 6   |    78 |  0.160%    | 26 858           |  80 677       |
| 7   |     1 |  0.002%    | 32 447           |  32 447       |

5k subsample (§9) projected accurately to full corpus (BxB=6 fraction 0.160% identical at both N). One BxB=7 puzzle confirmed in the corpus.

### Substrate artifacts

- `/tmp/chain_rerate/forum_48k_bxb/rated.jsonl` — raw per-puzzle (puzzle, bxb, wall_ms).
- `/tmp/chain_rerate/forum_48k_bxb/seeds_by_bxb/bxb{2..7}.txt` — stratified seed files, 81-char puzzles, one per line.
- `/tmp/chain_rerate/forum_48k_bxb/stratified.jsonl` — combined JSONL with `{puzzle, bxb}`.

### Eval-bucket sizing for utm-jax substrate

Per `difficulty_scale.md` tier mapping (T0..T5+), the stratified corpus gives:
- T0 (BxB=0..2 sentinel; forum_hardest is hard-curated so no T0 here)
- T2 (BxB=2): 5189 — comfortable easy-bucket evals.
- T3 (BxB=3): 32117 — bulk training / mid-difficulty evals.
- T4 (BxB=4): 10215 — main hard-bucket training.
- T5 (BxB=5): 1166 — frontier eval (statistically valid).
- T6 (BxB=6): 78 — extreme tail eval (small but viable).
- T7 (BxB=7): 1 — single anchor point.

For utm-jax training: this substrate covers the difficulty gradient required for OOD/depth questions in CLAUDE.md §0. The hard bucket (BxB≥5: 1245 puzzles) is curated and stable; we don't need to generate more for substrate purposes.

This completes Task #13 and closes the substrate-preparation thread for forum_hardest. Subsequent generation work (if needed) gates on concrete training signal from utm-jax.
