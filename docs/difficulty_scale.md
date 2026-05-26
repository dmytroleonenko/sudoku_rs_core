# Difficulty Scale and Generation Capability

**Purpose.** This document maps Berthier's classification hierarchy onto our two ports (`src/generic/` CLIPS-based, `src/shc/` SHC-decompile-based), describes the rate-cost and construct-cost of each tier, identifies where our generation capability ends, and frames the open research problem of generating the hardest puzzles (BxB ≥ 6).

The downstream consumer is the latent-reasoning network training pipeline, which needs (a) puzzles labeled by algorithmic difficulty for curriculum design, and (b) targeted-difficulty synthesis for evaluation set construction.

## 1. Scale (lowest → highest)

| Tier | Berthier label | Algorithm | Rate engine | Rate cost | Construct engine | Construct cost |
|---|---|---|---|---|---|---|
| 0 | B=0 | Naked single + Hidden single | `shc::tb::propagate(L0)` | < 1 µs | clue-removal from solved grid | trivial |
| 1 | B=1 | + Box/line interaction | `shc::tb::propagate(L1)` | < 1 µs | clue-removal | trivial |
| 2-3 | B=2..3 | GBraid[2..3] partial chains | `shc::wave::rate_b` | ~ 1 ms | `gbraid_reverse` (CLIPS) | ~ 10 ms/puzzle |
| 4-5 | B=4..5 | GBraid[4..5] | `shc::wave::rate_b` | ~ 10-50 ms | `gbraid_reverse` | ~ 100 ms/puzzle |
| 6-9 | B=6..9 | GBraid deeper | `shc::wave::rate_b` | 0.1-10 s | `gbraid_reverse` (theoretical) | ⚠️ untested at high k |
| 10+ | (T&E(1) ceiling exceeded) | T&E(2) outer + GBraid inner | `shc::te::rate_bxb` | 0.5-15 s | **NONE** | research problem |
| 11+ | BxB=2..7 | T&E(2) variable depth | `shc::te::rate_bxb` | 1-60 s | **NONE** | research problem |
| 13+ | BxBB | T&E(3) | `shc::te::rate_bxbb` | 30s-? | **NONE** | research problem |
| 14+ | T&E(>3) | beyond classified | `shc::te::rate_te_depth` returns >3 | extreme | n/a | n/a |

**Triangulation status.** All rate engines verified against SHC.jar reference on the Berthier 92-puzzle corpus (B=0..9, 92/92) and on the 20-puzzle forum_hardest_1905 sample for BxB/TE-depth (20/20). 48k forum_hardest_1905_11plus rated end-to-end via `shc::te::rate_bxb` with 0 errors.

## 2. Real-world distribution

### forum_hardest_1905_11plus (48766 puzzles, "hardest known T&E(2)" corpus)

| BxB | Count | % |
|---|---|---|
| 2 | 5189 | 10.64% |
| 3 | 32117 | 65.86% |
| 4 | 10215 | 20.95% |
| 5 | 1166 | 2.39% |
| 6 | 78 | 0.16% |
| 7 | 1 | 0.00% (= 1 puzzle) |

All 48766 are T&E-depth = 2. The "11plus" name refers to SER (Sudoku Explainer Rating) ≥ 11.x, NOT BxB; SER and BxB are correlated but not identical.

The single BxB=7 puzzle is the empirical hardest in the catalogue:
```
5.......9.2.1...7...8...3...4.6.........5.......2.7.1...3...8...6...4.2.9.......5
```

### Berthier reference corpus (92 puzzles, T&E(1))

Stratified by Berthier label: 10 puzzles each at B=0..7, 10 at B=8, 2 at B=9.

## 3. Generation capability — where we are

### What works (T&E(1) range)

`src/generic/gbraid_reverse.rs` synthesizes puzzles where GBraid[k] is the load-bearing technique at user-specified k. Mechanism: generate seed grid, remove clues guided by re-rating, accept iff rated frontier matches the GBraid[target_k] spec.

This covers tiers 2-9 in our scale. In practice, the per-clue probe overhead means generation cost grows with target k.

### What's missing (T&E(2) and up)

There is no constructor that targets BxB ratings. Specifically:
- No outer-elimination synthesis: we cannot construct a puzzle that REQUIRES T&E branching at depth k.
- `gbraid_reverse` operates entirely within T&E(1) — its rejection criterion is "GBraid[k] does/doesn't fire", and it can't probe "what would T&E(2) say".

This is the gap blocking generation of forum_hardest-grade puzzles.

## 4. Open research question — generate BxB ≥ 6 without seed

**Restated.** Produce a puzzle with `shc::te::rate_bxb >= 6` from a state that is not derived from any pre-existing hard puzzle. Minimize total compute.

### Approach taxonomy by cost (most-expensive → cheapest)

#### (A) Naive uniform random search-and-filter — UPPER BOUND
1. Sample a random unique-solution minimal puzzle (Royle-style backtracking).
2. Compute `rate_bxb`. Accept iff ≥ 6.
3. Repeat.

Hit rate estimate: forum_hardest_1905_11plus is a *curated* corpus of hard puzzles; uniform-random minimal sudoku is dramatically easier on average. Rough estimate based on community surveys: P(BxB≥6 | random minimal) ≈ 10⁻⁷ to 10⁻⁶.

Cost: 10⁶ × 1.5 s/rate (median) ≈ 17 days per puzzle on 1 thread, or ~13 hours per puzzle on 32 threads. Intractable for systematic generation.

#### (B) Vicinity hill-climb on uniform random seed
1. Generate random minimal puzzle (cheap).
2. Apply local mutations (single clue swap, delete-and-readd, value flip).
3. Rate; accept mutation iff BxB ≥ current.
4. Continue until stuck or BxB target reached.

Existing infra: `gen-vicinity` CLI command (`src/generic/reverse_construct.rs`, with `--target-se` fitness). To support BxB, fitness needs to swap to `shc::te::rate_bxb`.

Expected hit rate: vicinity climbs typically reach BxB=3-4 from random seed; reaching BxB=5+ requires lucky basin. Empirical question.

Cost estimate: 1000-10000 iterations × 1.5 s/rate = 0.5-4 hours per puzzle, *if* the basin contains a BxB≥6 local optimum. Often it doesn't.

#### (C) 17-clue catalogue mining — TESTED 2026-05-20, NEGATIVE RESULT
Royle's catalogue of all known minimal 17-clue Sudokus (49,151 puzzles, public, mirrored at github.com/shadaj/sudoku). The 17-clue floor is the absolute minimum: a 16-clue puzzle has multiple solutions.

**Empirical result (rated all 49,151 via `shc::te::rate_bxb`):**
- BxB distribution: 100% BxB=0.
- TE-depth: 52% depth-0, 48% depth-1. Zero at depth ≥ 2.
- Wall: ~3 min on 10 threads.

**The 17-clue catalogue does NOT contain T&E(2) puzzles.** 17 clues is the uniqueness minimum, not the difficulty maximum. The famous hardest puzzles are 20-25 clues (AI Escargot 24, Easter Monster 21, Golden Nugget 23). 17-clue puzzles have so little starting information that their constraint structure forces fast linear propagation; the 20-25 clue band is where entanglement (SK loops, multi-cell deadly patterns) can be constructed.

**Path C is dead.** No BxB ≥ 6 from Royle mining.

#### (D) Targeted construction via unavoidable sets — THEORETICAL CHEAPEST
Berthier (PBCS / HCCS, Chapter 4) and the enjoysudoku community have explored constructive methods that explicitly build puzzles requiring deep proof trees:

1. **Unavoidable Sets (UAs)**: minimal sets of cells that, if all uncleared, leave the solution ambiguous. The clue set must "hit" every UA at minimum once.
2. **Bivalue Universal Grave (BUG)** and similar deadlock structures: configurations where standard logic stalls and T&E is forced.
3. **MUG (Minimal UnAvoidable Graph)** analysis: identifies puzzles where the clue set is the minimum-vertex cover of the UA hypergraph.

These approaches require:
- Computing the UA structure of a candidate full-grid (NP-hard in general; tractable for small UA sizes).
- Selecting clue placements that maximize T&E depth.
- Algorithmic insight beyond what `sudoku_rs_core` currently implements.

In principle this is the optimal-compute path (avoids the random-search waste), but the algorithmic gap to close is substantial. The enjoysudoku community has produced hardest puzzles via this route, but the construction procedures are largely *expert-guided*, not fully automated.

### Recommended next steps

For the project:
1. **Implement T&E(2)-fitness extension to `gen-vicinity`** (path B + seeded from forum_hardest BxB=5 puzzles). This is the immediate engineering deliverable.
2. **Download and rate Royle's 17-clue catalogue** (path C). Adds ~50-100 new BxB≥6 seeds. One-off, cheap.
3. **Research read on UA / BUG construction methods** (path D). Targeted reading on the enjoysudoku forum and Berthier HCCS Chapter 4 to inform a possible future constructive engine.

For research:
- Open question: are there compact "T&E(2) certificates" (proof skeletons) that can be inverse-constructed without enumerating the full puzzle search space? Berthier's `rate-shc-batch BxBB` output gives proof tuples but not in a form that's directly invertible.
- Open question: does adversarial training of a small generator network against `shc::te::rate_bxb` as a frozen critic find new BxB≥6 puzzles? Cost would be dominated by rating.

## 5. Map onto the network training problem

The network learns to produce solutions. Difficulty is the metric of "how hard is this for the algorithm." Mapping:

- **Network solves it easily** → Tier 0-1 (singles + box/line). Training data: cheap, generate millions.
- **Network solves with effort** → Tier 2-7 (GBraid range). Training data: `gbraid_reverse` synthesizes targeted at each k. Available.
- **Network struggles** → Tier 8-9 (T&E(1) ceiling). Training data: `gbraid_reverse` at high k, partial coverage.
- **Network fails** → Tier 10+ (T&E(2)+). Training data: **no synthesizer**; we use forum_hardest_1905_11plus + Royle 17-clue mining as the only sources. Limited to ~48k + ~49k ≈ 100k puzzles total, with the curated hardest subset being ~80 puzzles.

The lack of a T&E(2) constructor is the bottleneck for evaluating network generalization at the hardest difficulty levels. The constructor is a research problem; mining + vicinity-search is the practical interim solution.

---

## Appendix A — Quick reference

Rate a puzzle:
```bash
echo "<81-char puzzle>" | sudoku_rs_core rate-shc-batch \
  --input /dev/stdin --output /dev/stdout \
  --classification bxb --max-length 14 --threads 1
```

Construct GBraid[k] puzzle:
```bash
sudoku_rs_core reverse-construct --target-tier T4Plus \
  --required gbraid --num 100 --threads 8
```

Vicinity-search from seed pool (SE fitness, existing):
```bash
sudoku_rs_core gen-vicinity --seed-in seeds.parquet \
  --out hard.parquet --target-se 11.0 --budget-iters 200
```

Vicinity-search with BxB fitness (TO BE IMPLEMENTED):
```bash
sudoku_rs_core gen-vicinity --seed-in seeds.parquet \
  --out hard.parquet --target-bxb 6 --budget-iters 500
```

## §6. gen-vicinity with BxB fitness (T&E(2)-targeted, implemented)

The `gen-vicinity` CLI subcommand supports two fitness functions for
hill-climbing, selected via `--fitness`:

- `--fitness se` (default, backward-compatible): rate candidates via the
  CLIPS-based cascade in `src/generic/rater.rs`; accept iff
  `se_score >= --target-se`. Output parquet columns:
  `puzzle, solution, clue_count, se_score, tier, frontier_json, generation`.

- `--fitness bxb`: rate candidates via `shc::te::rate_bxb` (T&E(2) outer);
  accept iff returned BxB rating `>= --target-bxb`. Output parquet columns:
  `puzzle, solution, clue_count, bxb, generation`.

The BxB path uses a greedy ≥ hill-climb (lateral plateau moves accepted) and
the same mutation/uniqueness/canonical-dedup primitives as the SE path
(`generic::canonical::canonical_hash`, `generic::search::count_solutions_up_to`,
`generic::search::solve_unique`). Mutation: pick `n` distinct clue cells and
replace each digit with a random digit ≠ current (n controlled by `--mute-n`).

BxB-specific knobs:

| Flag | Default | Meaning |
|------|---------|---------|
| `--fitness bxb` | `se` | Switch to BxB fitness |
| `--target-bxb` | `6` | Emit when `rate_bxb(...) >= target` |
| `--bxb-max-length` | `14` | `max_length` passed to `rate_bxb` (mirrors SHC.jar BxB default) |
| `--bxb-buffer-size` | `4096` | `BraidArena` buffer size |

Example — drive generation of BxB ≥ 6 from a BxB=5 seed pool:

```bash
sudoku_rs_core gen-vicinity \
  --seed-in seeds_bxb5.txt --out bxb6_pool.parquet \
  --fitness bxb --target-bxb 6 \
  --budget-iters 500 --max-outputs 1000 --mute-n "1,2" --seed 0
```

A sidecar manifest (`<out>.parquet.manifest.json`) records
`fitness`, `target_bxb`, `bxb_max_length`, `bxb_buffer_size`,
`seed_count`, `output_count`.

Smoke run (20 BxB=5 forum_hardest seeds, `--budget-iters 100`, `--mute-n 1`,
`--target-bxb 6`): ~27 s wall, 0 BxB ≥ 6 puzzles emitted from 100 mutations
per seed. BxB=6 is rare (79 of 48766 in forum_hardest); larger
`--budget-iters` and `--mute-n "1,2"` are likely needed for non-zero yield.
