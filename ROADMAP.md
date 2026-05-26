# sudoku_rs_core Roadmap

Active: R3.2 + R2.2.4 reviews → merge → R3.1b → R3.3 perf.

## Completed

- R1, R1.5, R2.1 AIC, R2.2.1 Fish, R2.2.2 UR Type 2, R2.2.3 ALS-XZ, R2.2.4 NakedQuad/HiddenQuad
- R3.0a io module + rate-batch CLI
- R3.1a port + delete legacy
- R3.2 CLI consolidation + gen-text + bug fixes (in review)
- T1 generator fix
- Catalog + manifest system

## Pending

### R7 — anti-memorization eval suite (separate reasoning from pattern matching)

Phase 5 на 9×9 AIC chain7 достигла EM=0.891. Codex strategic review предупредил: это **сильный signal но не proof of reasoning**. Главный risk — модель запомнила библиотеку AIC pattern families вместо learning underlying reasoning.

**Eval suite** (Python script load Phase-5 ckpt + battery of tests):

**Cheap** (no new datasets, augment existing eval):
1. **Digit relabel OOD** — random permutation of {1..9} per puzzle (solution invariant); pattern memorizers drop, reasoners preserve.
2. **Symmetry tests** — transpose, band/stack permutations, digit relabel; preserves AIC structure; drop = pattern.
3. **Forced compute-budget curve** — eval at ponder=4/8/16/32/48; harder bucket should require more steps.
4. **Causal ablations** — zero/shuffle memory at inference, truncate ponder, freeze halt; if no EM drop → memory/ponder декоративны.

**Medium-cost** (new generated puzzles via Rust generator):
5. **Length extrapolation** — generate AIC chain 8-10 (train was ≤7); reasoning generalizes if EM holds.
6. **Topology OOD** — held-out chain graph families (label puzzles by chain topology, then split).
7. **Cross-generator** — different generator (constraint-satisfaction direct vs reverse-construct).

**High-cost**:
8. **Probe intermediate states** — train linear probes on hidden states to predict per-cell candidate elimination progress.

**Decision rules**:
- Если все cheap tests show EM preservation → reasoning likely real
- Если digit relabel / symmetry drop EM significantly → pattern memorization
- Если forced ponder = 4 ≈ 48 на hard buckets → compute не используется
- Если causal ablations не drop'ит EM → memory/ponder декоративны

**Workflow**:
1. Code: `tools/utm-jax/eval_anti_memorization.py` (cuda-host2)
2. OOD datasets: generated locally в `data/generic_v1/9x9_aic_chain*_lb_v2/` (chain length sweeps)
3. Run battery on Phase-5 ckpt 29670
4. Если signal weak — invest в better-falsifying data (length extrapolation harder splits)

### R8 — universality (deferred, only if needed)

Если main goal Phase 6 success → architecture covers 9-16 range. True size-agnosticity (>16) — отдельный milestone требующий **new model family**:

- **Constraint bipartite graph ↔ K recurrent latent slots** (codex recommendation): cell+constraint nodes, latents cross-attend каждый ponder step, halt из latent state. Лучшая комбинация size scaling (graph tokens, no quadratic attention) + reasoning workspace (fixed K bank + ponder/halt machinery).
- **Bucketed training by size** (incremental): factorized cell features `(digit, row, col, box, board_size)`, relative position features, separate compile buckets.
- True universality = новая model family + новое обучение. Old model используется как teacher/distillation target для 9-16 range.

Defer until: (a) Phase 6 fail на 16T1, или (b) R7 anti-memorization showed narrow pattern family, или (c) project explicitly extends scope к arbitrary N×N.

### R3.1b — refactor gen-dataset/re-rate onto io traits

After R3.2 / R2.2.4 merge.

### R3.3 — performance pass (target: close ~80% of 85× gap to schoku)

- ✅ **R3.3a** (landed 398ce2e): `--mode {solve, tier, full}` on rate-batch + re-rate. mode=solve gives 12-110× speedup on rate hotpath.
- **R3.3b** (HIGH/M, ~600 LOC): in-place search + undo log (kill `Grid::clone` в search.rs/backtracker.rs/aic_reverse.rs). Codex анализ 2026-05-09 — biggest all-case win. Expected 2-4× на 16×16 T1/AIC, 1.5-2.5× chain≥9.
- **R3.3b2** (HIGH/M, ~250 LOC): AIC scratch buffers (reuse `strong/weak Vec<Vec<u32>>` через thread-local scratch struct, `stamp` counter вместо realloc) + reorder pre-filter (run `aic_score` before uniqueness DFS, reject early if chain lost). Plan agent 2026-05-09 — 2-3× chain≥9, 4-6× 16×16 AIC probe.
- **R3.3b3** (HIGH/S, ~50 LOC): remove redundant solution resolves (`aic_reverse_construct` re-solves seed, `reverse_construct` re-solves accepted, `count_solutions_up_to` clones unused first solution). Codex 2026-05-09 — 1.2-1.6× T1, 1.1-1.4× AIC.
- **R3.3c** (HIGH/M, ~150 LOC): peer bitset (`Vec<bool>` → `[u64; (NN+63)/64]` stack-allocated). Helps every T3 technique on 16×16. Plan 2026-05-09 — 1.5-2× T3 cascade.
- **R3.3d** (MED/L, ~250 LOC): bidirectional / iterative-deepening BFS for AIC chain probe. Only candidate с asymptotic improvement (5-10× chain 9-13). Plan 2026-05-09.
- **R3.3e** (MED/L, ~500 LOC): SIMD u128 candidate masks (9×9), AVX2 propagation. Profile-driven.
- **R3.3d** (LOW/M): AVX-512VBMI2 (variable byte-shuffle) + BITALG (popcount) на Ice Lake+ — modest **10–25% over AVX2 на hard puzzles** per tdoku numbers. Worth only after R3.3a–c land.
- **R3.3e** (MED/M): NEON path для ARM / Apple Silicon — straightforward port of `pshufb` → `vqtbl1q_u8`. Engineering project, not research; unlocks M-series local benchmarking parity.
- **R3.3f** (LOW/S): PGO (profile-guided optimization) for production builds. tdoku reports significant gains; cheap to add to CI release builds.
- RISC-V V — no published Sudoku solver. Defer indefinitely.

Reference perf class: tdoku, sudokusse, schoku ~few µs/puzzle на 9×9. Наш target — close ~80% of gap, не paper-match. **Always report (dataset, CPU, compiler, flags) tuple** with any perf number — single µs/puzzle figures без этих координат are benchmark gaming (см. synthesis 2026-05-08 "Hardest puzzle datasets").

### R4.0 — Berthier W/B/gB_n rating axis

Adopt Berthier resolution-theoretic ratings as secondary difficulty label alongside our T1/T3/AIC tags.

- **W[k]** — whip rating (chain length k).
- **B_n / B** — braid rating (chains-with-subchains, strictly stronger than whips).
- **gB_n** — g-braid rating (grouped candidates).
- **S_p** variants (typed, restricted to single CSP-Variable space rc/rn/cn/bn — cheaper to compute).
- **T&E(k)** orthogonal axis: Trial & Error depth.

Confluence property (proven in PBCS 3rd ed.) обеспечивает intrinsic rating = single pass.

Map to current tagging: AIC chain=5±1 ≈ W[5] typed.

Cross-reference SUDCL/CBGC corpora (companion repo `denis-berthier/Sudoku-classif`) для standard difficulty calibration.

Implementation: extend manifest.json schema с `berthier_rating: { w: int?, b: int?, gb: int?, te_depth: int }`. Run rating через CSP-Rules CLIPS reference solver или port whip/braid evaluation.

### R4.1 — Additional reasoning techniques (REVISED priority)

Per Berthier CSP-Rules-V2.1 catalogue, "few universal rules > zoo of patterns". Targets ranked by yield × generalization:

**HIGH** (subsume multiple named patterns):
- **braid[k]** — chains with embedded subchains as conditions. Strictly stronger than whips. Subsumes Death Blossom + Sue de Coq + many forcing patterns. Currently MISSING (we cover whip-equivalent via AIC).
- **g-whip[k]** / **g-braid[k]** — grouped candidates. Required for AHS-style grouped fish. Generalizes naked/hidden subset chains.
- **3D Medusa** — extends our SimpleColoring to 3D lattice (row × col × digit). Multi-digit clusters, several eliminations per pass.

**MEDIUM** (zero-rank logic, finds patterns no AIC reproduces):
- **MSLS / Multi-Sector Locked Sets** — Bird 2013 formal definition. Partition digits Home/Away (4+5 для 9×9), balance NS/HS/DC algebraic across houses. C(9,4)=126 partitions × 2^h house assignments — feasible search. Reference: `denis-berthier/CSP-Rules-V2.1`, YZF_Sudoku (closed-source). Decide upfront: MSNS-only OR MSHS-only OR both. SK-Loop is strict subtype — covered automatically.

**LOW** (rare, named-pattern, defer):
- **Junior Exocet** — only essential for t-series/h-series/Unsolvable corpus.
- **AMSLS** (almost-MSLS) — eliminations rarely unique vs base MSLS per forum consensus.

**DROPPED** (subsumed):
- **Death Blossom** — covered by braid[k] + g-whip[k].
- **Sue de Coq** — covered by braid[k].
- **Forcing Nets** — covered by braid + forcing-whip variants.
- **Senior Exocet** — defer permanently (computational cost > branch guess cost).
- **SK-Loop** standalone — strict MSLS subtype.

Cross-ref `docs/v45_latent_reasoning_synthesis.md` 2026-05-08 block (Architectural references / Hardest puzzle datasets) for the literature backing this re-rank.

### R4.2 — Sudoku-Extreme benchmark ingestion

- Download dataset (HRM paper, arxiv 2506.21734) — ~250K very hard 9×9, mean 22 backtracks.
- Add ingest path в `data/generic_v1/9x9_extreme/`.
- Add OOD eval bucket для curriculum validation. Comparable point: BDH 97.4%, LLMs ≈0%.
- Также `denis-berthier/Sudoku-classif` corpora (CBGC/SUDCL) для difficulty calibration cross-reference (Berthier W/B/gB_n labelled puzzles).

### R4.3 — Difficulty metric extensions

Per Thangamani 2025 (CSP-based, technique-independent):

- **CDM** — Constraint Density.
- **LID** — Logical Inference Depth.
- **GC** — Guessing Complexity.
- **CGT** — Constraint Graph Tightness.
- **Backtrack-count** (HRM-style empirical metric).

Add as secondary columns в parquet manifest рядом с tier.

### R4.4 — SAT/SMT escape hatch (uniqueness oracle for variable grid sizes)

- **Z3 / CVC5 backend** для 16×16 / 25×25 difficult instances где наш domain solver stalls. Not a perf path — used as a *correctness oracle*.
- Use SAT/SMT как **uniqueness oracle** для variable grid sizes (correctness check on new techniques and constructive generators when no faster oracle exists).
- Reference: **Reeves 2025 PhD thesis (CMU)** — XLC encoding + SBVA для cardinality constraints (relevant для 16×16+ where naive at-most-one encodings blow up).
- Reuse Jain 2010 clue-aware clause elimination (79× CNF size reduction) from existing references — known not competitive end-to-end vs bit-parallel solvers, but cheap CNF size matters for SAT-as-oracle latency.

### R5 — GPU batch solver (16×16 throughput)

- Use case: batch throughput (16×16+), не single-puzzle latency. Single-puzzle CPU AVX2 уже в µs class.
- Architecture: warp-per-puzzle bit-parallel masks; port schoku CPU AVX2 design на CUDA warps.
- Reference: Gomez 2025 CUDA Sudoku — ~7× over naive backtracking.
- Defer until 16×16 dataset bottleneck blocks training.

### Notes / non-goals

- **SAT/CDCL solvers** — best as oracle / baseline на extreme puzzles, не speed path. Sudoku-specific CNF encodings (Jain 2010): clue-aware clause elimination даёт 79× CNF size reduction, но end-to-end solve time всё равно не догоняет dedicated bit-parallel.
- **ARM SVE / RISC-V V** — нет published Sudoku-specific work. Likely просто ports of x86 designs. Не worth deep dive.

## References

Solvers / perf:

- **Schoku** (C++, AVX2): `/Users/dleonenko/Schoku` — 4.2 µs/puzzle на big5000 9×9.
- **tdoku** (`github.com/t-dillon/tdoku`) — baseline benchmark harness + `simd_vectors.h` reference. Adopt их dataset suite (`puzzles0_kaggle` / `puzzles2_17_clue` / `puzzles3_magictour_top1465` / `puzzles5_forum_hardest_1905_11+` / `puzzles6_serg_benchmark`) для PR benchmarks. Includes log(guesses) honest difficulty metric.
- **sudokusse** (`github.com/zettsu-t/sudokusse`, zettsu-t) — SSE4.2/AVX inline assembly. Includes Sudoku-X (diagonal) variant — relevant если variant rules ever matter.
- **rust_sudoku** (`crates.io/crates/sudoku`) — pure Rust port of JCZSolve approach. **Direct competitor / reference** для R3.3 perf goals (~2.5 µs/puzzle на 17-clue Coffee Lake AVX2).

Technique catalogues / solving theory (PRIMARY):

- **CSP-Rules-V2.1** (Berthier, GPL-3.0, CLIPS, last commit 2025-11-27, DOI: 10.5281/zenodo.17491941). https://github.com/denis-berthier/CSP-Rules-V2.1 — open-source pattern-based solver, reference for whip/braid/g-whip taxonomy + W/B/gB_n ratings. **Primary technique reference вместо mixed Hodoku/Sudokuwiki** going forward.
  - **PBCS** (3rd ed., 2021) — *Pattern-Based Constraint Satisfaction and Logic Puzzles* — formal proofs (confluence, resolution theory).
  - **HCCS** (2025) — *Hidden Coverings and Coverings Systems* — current rating definitions.
  - **arXiv:1304.1628** — chains/braids formal proofs.
- **MSLS canonical thread** — http://forum.enjoysudoku.com/using-multi-sector-locked-sets-t31222.html (David P. Bird 2013).
- **MSLS ⊂ SK-Loops hierarchy** — http://forum.enjoysudoku.com/sk-loops-and-msls-s-t36887.html
- **YZF_Sudoku** (closed-source, freebasic, Windows-only) — reference impl но DO NOT depend on, no algorithm spec published. http://forum.enjoysudoku.com/yzf-sudoku-t36846.html
- **HoDoKu / Sudokuwiki** — kept as tutorial references but note label-leakage failure mode when used as ML training labels (см. synthesis 2026-05-08 SATNet caveat).

Datasets / benchmarks:

- **Sudoku-Extreme** — HRM paper, arXiv:2506.21734, ~250K hard 9×9 benchmark (mean 22 backtracks). Reproducibility caveat: HigherOrderCO replication shows 10–15pp drop on same code (см. synthesis 2026-05-08). Always cite caveat.
- **Sudoku-Bench** (arXiv:2505.16135) — additional curated hard variants (constraint-variant axis); Gemini 2.5 Pro <15%.

Difficulty metrics:

- **Pelánek 2014** (arXiv:1403.7373) — propagation-depth + phase-transition; Pearson r=0.88 with human times alone, r=0.95 combined. Reference benchmark для difficulty metrics.
- **Berthier B/Bn/gBn** — resolution-theoretic, reproducible (CSP-Rules-V2.1).
- **log(guesses) by tdoku** — honest baseline secondary metric.
- **Thangamani 2025** — CSP-based metrics (CDM / LID / GC / CGT).

ML-as-solver references:

- **HRM** (Wang et al. 2025, arXiv:2506.21734) — hierarchical recurrent module, 27M params. Reproducibility caveat (см. synthesis).
- **TRM** (Jolicoeur-Martineau 2025, arXiv:2510.04871) — 2-layer, 5M, claims higher generalisation than HRM. Awaiting independent replication.
- **NASR** (Cornelio 2023, ICLR) — Mask-Predictor + symbolic solver hybrid; pattern для interpretability.
- **ConsFormer** (Xu 2025, arXiv:2502.15794) — self-supervised CSP transformer; OOD-via-iteration claim, *not yet externally replicated*.
- **SATNet** (Chang et al. 2020, NeurIPS) — failure caveat: 98.3% textual → 0% visual without label leakage.
- **Pathway BDH** — 97.4% Sudoku-Extreme без CoT.
- **RRN** (Palm 2018, NIPS) — 96.6% hard 9×9, 32 message-passing rounds.
- **DRNets** (ICLR 2020) — ~100% MNIST-Sudoku.
- **REM** (2023) — technique-sequence extraction probe template.

SAT / GPU / encoding:

- **Jain 2010** — Sudoku CNF encoding optimization (79× clause reduction).
- **Reeves 2025 PhD thesis** (CMU) — XLC encoding + SBVA для cardinality constraints; relevant для R4.4 на 16×16+.
- **Gomez 2025** — CUDA Sudoku reference architecture.
