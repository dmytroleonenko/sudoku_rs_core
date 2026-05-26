# R3.4 — Hard Puzzle Generation (SE 9.0+ / T10+)

**Статус:** draft, не начато.
**Цель:** efficient генерация пазлов уровня SE ≥ 9.0 («T10+» — Dynamic Forcing Chains, 3D Medusa, MSLS, nested chains).
**Зачем:** R7 anti-memorization, R4.2 Sudoku-Extreme curriculum и R8 cross-substrate runs нужны hard buckets, которых текущий top-down pipeline даёт <10⁻⁶.

---

## Решённые user'ом direction calls

1. **Алгоритмы портируем из Java SudokuExplainer** (Juillerat 1.2.1 + 1to9only fork). Не калибруем веса вручную — берём реальные техники (Dynamic Forcing Chains, nested FC, 3D Medusa, Mutant Fish и т.д.) и реальный SE rating algorithm. **Licensing caveat:** SE — GPL/LGPL. Crate description прямо говорит «no GPL code copied». Поэтому **clean-room порт**: читаем Java, понимаем алгоритм, переписываем в Rust заново. Никакого copy-paste, никакого decompile. Каждый технический файл начинается комментарием «Algorithm originally described in SudokuExplainer X.Y, file Z.java; reimplemented clean-room — see notes in <ref>». Reference Java читаем как спецификацию, не как source.
2. **Seed pool:** скачать публичные hardest collections (top1465, forum_hardest_1106, ph_xxxx.zip от champagne) в `data/seeds/public_hardest/`. Отдельный bootstrap step (Stage S).
3. **Compute budget:** решаем по ходу. После Stage 0 + Stage 3 (первый запуск с реальными SE-techniques) — меряем throughput, тогда выбираем Mac vs cuda-host2 CPU для Stage 4 production. Stage 0/1/2/3 development — local Mac M3 Max.

## Архитектурное требование (НОВОЕ — критичное)

Каждый алгоритм/технику структурировать под **AlphaEvolve / OpenEvolve** workflow: эти системы итерируют one-file-at-a-time — делают edit, прогоняют тесты, мерят перформанс, итерируют. Следствия:

- **Один файл — одна техника.** Self-contained. Не должно быть cross-file invariants, ломающихся при изменении одного файла. Текущая структура `src/generic/techniques/{aic.rs, fish.rs, ...}` уже близка — extend её, **не нарушать**.
- **Стабильный trait boundary.** Каждый файл реализует `Technique<N, BR, BC>` (или новый `RatedTechnique`, см. Stage 0). Внешний код взаимодействует ТОЛЬКО через trait. Реализация внутри файла полностью свободна (data structures, helpers, scratch buffers).
- **Co-located тесты + golden vectors.** В конце каждого `<technique>.rs` блок `#[cfg(test)]` с (a) корректность на ≥ 5 reference grids с known eliminations, (b) regression golden — известный SE rating сэмпл, (c) NO regression на existing test puzzles (technique не должна срабатывать там, где не должна).
- **Co-located microbenchmark.** В каждом файле — функция `pub fn bench_inputs() -> Vec<Grid>` возвращающая 50–200 representative grids, плюс отдельный `benches/<technique>.rs` (Criterion-stub, чтобы измерять throughput per technique). AlphaEvolve смотрит на эту цифру.
- **Header-doc с invariants.** В начале файла комментарий: `# Inputs (что считается из Grid), # Mutates (что разрешено менять), # Returns (TechniqueProgress contract), # Performance budget (target µs/grid)`. Это контракт для AlphaEvolve — он не должен ломать.
- **Запрещён cross-technique state.** Никаких static mut, никаких thread-locals разделяемых между техниками. Per-technique scratch — внутри struct, инстанциируется per call (или pooled через ScratchPool который инжектится).
- **Detach техники от rater'а.** Сегодня `rater.rs` импортирует конкретные техники по именам (`rater.rs:162`). Нужен **registry pattern**: rater получает `&[Box<dyn RatedTechnique>]`, registry конструируется в одном месте (`techniques/mod.rs` factory). AlphaEvolve может добавить новую технику простым `register!(MyTechnique)` без правки rater.

→ Это формализуется как **Stage R (Refactor)**, выполняется ПЕРЕД Stage 0.

---

## Контекст и диагноз

**Что есть сейчас:**

- `src/generic/techniques/` — 15 файлов, по технике каждый. Trait `Technique<N,BR,BC>` стабилен (`techniques/mod.rs:65`). **Хорошая база для AlphaEvolve.**
- `src/generic/generator.rs:62-83` — top-down dig-holes от свежей `random_solution()`.
- `src/generic/reverse_construct.rs:270-347` — payload injection (правильная парадигма), но ограничено T3-payload'ами (AIC chain≤9, Fish, ALS-XZ, UR, NakedQuad, HiddenQuad). Потолок ~SE 8.5.
- `src/generic/rater.rs` — 4 тира T1/T2/T3/T4Plus, **T4Plus = «всё не-T3»** без discrimination. SE-score нет.
- Канонизации (min-lex) нет.
- Seed pool / vicinity search нет.

**Research consensus (см. синтез):**

1. Чистый top-down не даёт SE≥9 в разумном compute. Production-метод — **seed-anchored vicinity search** + cascade filter (bt_count → fast pseudo-rater → SE).
2. ~75% всех SE≥11 содержат JExocet; SK-Loop и MSLS — два других структурных биаса.
3. SE 8.5–9.5 — Dynamic Forcing Chains; 9.5–11 — nested forcing; 11+ — глобальные паттерны.
4. Sweet spot: 21–23 clues для SE≥11.5; 24–26 для SE 10.5–11.3.

---

## Stage R — Refactor под AlphaEvolve workflow

**Зачем первым:** новые техники появятся как новые файлы. Архитектурные нарушения, заложенные сейчас, потом размножатся.

### Steps

1. **Registry pattern в `techniques/mod.rs`:**
   - `pub fn all_techniques<N,BR,BC>() -> Vec<Box<dyn RatedTechnique<N,BR,BC>>>` — единая точка сборки. Rater импортирует только эту функцию.
   - `rater.rs` сейчас перечисляет техники inline (`rater.rs:162-174`) — заменить на iter по registry.
2. **`RatedTechnique` trait** (extends current `Technique`):
   ```rust
   pub trait RatedTechnique<...>: Technique<...> {
       fn se_rating(&self, progress: &TechniqueProgress) -> f64;
       fn bench_inputs() -> Vec<Grid<N,BR,BC>> where Self: Sized;
   }
   ```
   - `se_rating` возвращает SE-эквивалентный вес *этого fire* (не cumulative). Базовый impl `= base_se` (константа per technique), override для chain-length-dependent (AIC, FC).
   - `bench_inputs` — golden bench fixture per technique.
3. **Header-doc convention:** каждый `techniques/<x>.rs` получает обязательный header (см. требование выше). Это enforce'ится в CI: новый `tests/technique_header_lint.rs` парсит каждый файл, требует 4 секций.
4. **Bench harness:** новая папка `benches/` (сейчас её нет — R3.2 удалила Criterion deps; restore). `benches/techniques.rs` итерирует registry, для каждой `bench_inputs()`, мерит median µs/grid. Single command `cargo bench --bench techniques` → JSON output, который AlphaEvolve parsит.
5. **Per-technique test scaffolding:** in-file `#[cfg(test)]` с минимум 3 testами: `fires_on_known_positive()`, `no_fire_on_known_negative()`, `correctness_no_invalid_elims()`.
6. **Detach scratch state:** review existing техник — если есть `thread_local!` или статика, перенести в struct fields. AIC scratch buffers (landed 7a0bce8) — проверить, чтобы они были per-instance, не глобальные.
7. **Стабилизировать `TechniqueProgress` enum** — это контракт между техникой и rater'ом. Добавить версионную assertion в тесте: `assert_eq!(TechniqueProgress::ABI_VERSION, 1)`.

### DoD

- [ ] `techniques/mod.rs::all_techniques()` собирает все 15+ существующих, rater импортирует только эту функцию.
- [ ] Все существующие техники compile + tests green после интродукции `RatedTechnique` (default impl `se_rating = base_se`).
- [ ] `cargo bench --bench techniques` работает, выводит per-technique µs/grid (JSON).
- [ ] CI lint test проверяет header-doc convention на всех `techniques/*.rs`.
- [ ] Документ `docs/alphaevolve_contract.md` объясняющий: «вот файл, вот invariants, вот как мерить, вот что можно/нельзя».
- [ ] Sample run: edit one technique file (например, `aic.rs`), `cargo test -p sudoku_rs_core` зелёный, `cargo bench --bench techniques -- aic` выдал число. Voilà, AlphaEvolve loop возможен.

### Risks

- `bench_inputs()` per-technique затратно по объёму кода. Mitigation: общий генератор fixture grids в `techniques/result.rs::bench_fixture_pool()`, конкретные техники фильтруют под себя.
- Текущий `Tier` enum (T1/T2/T3/T4Plus) — это уже rate output. После Stage 0 добавляется `se_score`; Tier остаётся для backward compat.

---

## Stage S — Seed pool ingestion

**Зачем:** vicinity search без seed pool бесполезен. Это отдельный «boring» шаг, должен быть рано.

### Steps

1. Скрипт `scripts/download_public_hardest.sh` качает:
   - `top1465.txt` (magictour, 1465 hard).
   - `forum_hardest_1106.txt` (376 SE≥10 seed list — kicked off thread t6539).
   - `ph_2010.zip` от champagne — ~2.1M SE≥10.3, основной production seed pool. Источник: Google Drive линк из forum.enjoysudoku.com t6539. **Если линк сломан** — fallback на `ph_1910.zip` или собрать собственный pool через первые Stage 0–3 итерации.
   - `puzzles5_forum_hardest_1905_11+` из tdoku benchmark repo.
2. Каждый distinct dataset → отдельный subdir `data/seeds/public_hardest/<name>/`:
   - `puzzles.txt` — one puzzle per line, 81-char format.
   - `manifest.json` — `{source_url, sha256, count, license, downloaded_at, known_se_min, known_se_max}`.
3. CLI subcommand `ingest-seeds`:
   - `--in <txt> --out <parquet>` — конвертит plain text в parquet с schema `{puzzle: string, solution: string, clue_count: int}` (solution через наш solver).
   - Опционально `--rate` — прогоняет rater'ом с текущим se_score (Stage 0 dependency).
4. `data/seeds/public_hardest/README.md` — что откуда, license note.

### DoD

- [ ] ≥ 3 публичных датасета скачаны в `data/seeds/public_hardest/`, manifests валидны, SHA256 verified.
- [ ] `ingest-seeds` производит valid parquet, проходящий `rate-batch --mode solve`.
- [ ] Spot-check 10 пазлов из `forum_hardest_1106`: наш solver их решает (uniqueness preserved).
- [ ] Минимум 100k пазлов в combined pool.

### Risks

- Champagne's Google Drive link потенциально dead/rotated. Mitigation: запросить у user альтернативный mirror, или начать с top1465 + forum_hardest_1106 (≈ 1.8k пазлов — достаточно для bootstrap Stage 2).

---

## Stage 0 — SE rating system (порт из Java)

**Зачем:** continuous score нужен для (a) vicinity hill-climbing target, (b) discrimination внутри T4Plus, (c) сравнения с публичными reference ratings.

### Принцип (важно для AlphaEvolve)

SE rating = **max difficulty rating среди всех techniques, которые firedа на solution path**. То есть rating пазла = max(se_rating(fire) for fire in solve_trace). Это per-step, не cumulative.

→ Это естественно ложится на `RatedTechnique::se_rating()` из Stage R. Каждая техника **самостоятельно** возвращает свой SE rating per fire. Rater просто аккумулирует max. Это **локализует knowledge в файл техники**, что и нужно.

### Steps

1. Для каждого существующего `techniques/<x>.rs` добавить implement `RatedTechnique::se_rating()` с весом по таблице из SE 1.2.1 / HoDoKu (см. deep research):
   - LockedPointing/Claiming: 2.6 / 2.8.
   - NakedPair / HiddenPair: 3.0 / 3.4.
   - NakedTriple / HiddenTriple: 3.6 / 4.0.
   - NakedQuad / HiddenQuad: 4.0 / 5.0.
   - XWing: 3.2. Swordfish: 3.8. Jellyfish: 5.2.
   - Skyscraper / TwoStringKite: 4.0 / 4.1.
   - SimpleColoring: 4.0 (type 1) или 5.0 (type 2).
   - XyWing: 4.2. XyzWing: 4.4. Bug: 5.6.
   - UR-T1: 4.5. UR-T2: 4.7.
   - ALS-XZ: 7.5 (бывает 6.0 для коротких).
   - **AIC**: base 5.0 + 0.1·(chain_len − 5), capped at 7.5 (длинные chain — это уже X-Chain в SE terminology). chain_len ≥ 11 — 7.5.
2. Rater (`rater.rs`) рефакторится: вместо tier-only output → `{ tier, se_score: f64, frontier }`. `se_score = max(se_rating per fire)`.
3. Schema bump в `src/schema.rs` (колонка `se_score: float64`), `pipeline_writer_generic.rs` и `rerate_generic.rs` — добавить колонку.
4. **Cross-validation:** скрипт `scripts/validate_se_scores.py`:
   - Прогнать 100 пазлов из forum_hardest_1106 (которые имеют известные SE ratings из публикаций) через наш rater.
   - RMS error vs published rating. Target: **RMS < 0.5** на T2/T3 диапазоне.
   - Output: `docs/se_calibration_report.md`.
5. **Если RMS > 1.0** — это диагностика, что веса в каком-то файле плохи. AlphaEvolve loop: дать ему этот файл + RMS metric — он подберёт веса. Файл self-contained → idealen для итерации.

### DoD

- [ ] Все existing techniques имеют impl `se_rating()` returning realistic value (per HoDoKu reference).
- [ ] `RateResult.se_score: f64` заполняется во всём cascade.
- [ ] Cross-validation на forum_hardest_1106 sample: RMS < 0.5 для T2/T3 пазлов (T4Plus — допустимо RMS < 1.5 до Stage 3).
- [ ] Schema bump: parquet manifest содержит `se_score`. Backward-compat test (`tests/test_no_legacy_tdoku_field_names.rs` style) — старые parquet без поля читаются как `None`.
- [ ] Smoke: median se_score на `9x9_aic_chain7_lb_v2` ≥ 6.5, p95 ≥ 7.2.

### Risks

- Веса всё-таки калибровочные. Если AlphaEvolve иногда нужно tune'ить одну технику — он работает на single file, не нужно знать про остальные. Это и есть main payoff Stage R refactor.

---

## Stage 1 — Min-lex canonicalization

**Зачем:** vicinity search без неё захлёбывается в orbit'ах (9!·3!·3!⁶·2 ≈ 1.2·10⁹).

### Steps

1. Новый модуль `src/generic/canonical.rs` — **single file, self-contained**, AlphaEvolve-friendly (это perf-critical hotspot, идеален для evolve).
   - `pub fn canonical_form<N,BR,BC>(grid: &Grid) -> [u8; N*N]` — min-lex representative.
   - `pub fn canonical_hash<N,BR,BC>(grid: &Grid) -> u128` — short hash для HashSet.
   - Algorithm: enumerate band perms × stack perms × row-in-band × col-in-stack × digit relabel × transpose, brute force с early pruning по префиксу. Reference: gsf `sudoku -c`. Реализуется clean-room из описания algorithm (не copy).
2. Header-doc + 5 tests (10 random isomorphic transforms → same output) + benchmark inputs.

### DoD

- [ ] `canonical_form` for 9×9: 10 isomorphic transforms → identical output (5 unit tests).
- [ ] `cargo bench --bench canonical`: < 1 ms/puzzle на 9×9 (single thread, Mac M3 Max). 16×16 — < 50 ms.
- [ ] `canonical_hash` collision-free на 10k random non-iso puzzles.
- [ ] CLI флаг `--canonicalize` для `rate-batch` пишет колонку `canonical_hash`.

### Risks

- Если pure min-lex > 10 ms на 9×9 — fallback на «approximate canonical» (только digit-relabel + transpose). Решение по бенчмарку.

---

## Stage 2 — Vicinity search engine

**Зачем:** core hill-climbing вокруг seed pool — здесь появляются hard puzzles.

### Steps

1. Новый модуль `src/generic/vicinity.rs` — **single file**, self-contained:
   - `pub struct SeedPool { entries: BinaryHeap<SeedEntry>, seen: HashSet<u128> }` отсортированный по `se_score desc`.
   - `pub fn mutate_n(puzzle: &Puzzle, sol: &Solution, n: usize, rng: &mut R) -> Option<Puzzle>` — pick n clue cells, replace с альтернативным digit consistent с solution pencilmark, return None если non-unique/non-minimal.
   - `pub fn explore(seed: SeedEntry, target_se: f64, budget_iters: usize, cfg: &VicinityCfg) -> Vec<SeedEntry>` — main hill-climbing loop. mute_1 exhaustive, mute_2 sample, mute_3 «kick» rare.
   - Cascade filter:
     - L1: `count_solutions_up_to(2)` (uniqueness, search.rs:99).
     - L2: bruteforce backtrack count (`backtracker.rs`) — reject if `bt_count < bt_prefilter_min` (typical 200 для target_se 9.0+).
     - L3: full `rate(...)` — reject if `se_score < target_se - δ`.
2. CLI subcommand `gen-vicinity` в `src/cli.rs`:
   - `--seed-file <path>` или `--seed-from <parquet>`.
   - `--target-se 9.0 --bt-prefilter 200 --budget-sec 3600`.
   - `--mute-n 1,2,3`.
   - `--out <parquet>`.
3. Parallelism: rayon `par_iter` по seed entries. SeedPool — `Mutex<>` или MPSC channel.
4. Header-doc + tests (на 6×6 с reduced clue для exhaustive verification) + bench fixture.

### DoD

- [ ] `vicinity.rs` self-contained, unit tests на 6×6 verify mutate_1 correctness.
- [ ] Smoke: на 100 seeds se_score≥6.5 → за 60 min на M3 Max ≥ 10 puzzles se_score≥7.5 и ≥ 1 puzzle se_score≥8.5.
- [ ] CLI `gen-vicinity` e2e работает.
- [ ] Canonical dedup: <1% дубликатов в output.
- [ ] Doc `docs/vicinity_usage.md`.

### Risks

- Stage 2 ceiling упирается в rater discriminator → ceiling растёт по мере Stage 3 progression.

---

## Stage 3 — Nested AIC + Dynamic Forcing Chains (порт из Java)

**Зачем:** поднять reverse-construct потолок с SE ~8 до SE 9.0–9.5+. Это и есть основной T10+ payload.

### Two sub-stages, both AlphaEvolve-friendly (per-file)

#### Stage 3a — Dynamic Forcing Chain rater technique

**Один файл:** `src/generic/techniques/dynamic_fc.rs` (новый), реализует `RatedTechnique`.

- Algorithm (clean-room порт SE 1.2.1 `Chaining.java`):
  - Для каждой bivalue/bilocal cell, тестируем гипотезу `cell=digit`.
  - Forward propagation через strong/weak links + nested cell forcing (recursive один уровень).
  - Если все варианты ведут к одной элиминации → fire с rating depends on chain depth.
- SE rating mapping:
  - Cell Forcing Chain (без nesting): se=7.5.
  - Region Forcing Chain: se=7.6.
  - Contradiction FC: se=7.8.
  - **Dynamic FC** (single-level nesting): se=8.5–8.8.
  - **Dynamic + Region nesting** (double): se=9.0–9.5.
- Co-located bench: 50 hardest puzzles из forum_hardest_1106, мерим µs/grid.
- Performance budget header: target < 50 ms/grid на Mac M3 Max single thread (это slow technique, expected). AlphaEvolve может на этом файле оптимизировать.

#### Stage 3b — Nested AIC reverse constructor

**Один файл:** `src/generic/nested_aic_reverse.rs` (новый, mirror `aic_reverse.rs`).

- Topology: два AIC fragments F1, F2 length 5–7 сходящиеся в общий elimination target.
- Construction algorithm:
  1. Random solved grid.
  2. Выбираем target cell X, кандидат c.
  3. Строим F1: AIC path от bivalue cell к (X, c) elimination.
  4. Строим F2: independent AIC path к тому же (X, c).
  5. Assert F1 ∩ F2 = ∅ (except endpoint).
  6. Planting: добавляем clues, которые обеспечивают F1+F2 как **единственный** path.
- **Critical sub-step: preemption masking** (см. research adversarial fill):
  1. После plant — прогоняем restricted forward-solver с allowed techniques = {Singles, LC, Subsets, basic Fish, single AIC≤6}.
  2. Если solver elimination'ит через bypass cell — добавляем blocking clue туда, рестарт.
  3. Loop до convergence или budget exceeded (reject puzzle).
- Это **самая сложная часть R3.4**. Без preemption masking nested chain бессмыслен — обходится короткой AIC.

### Steps

1. **3a.1** Создать `techniques/dynamic_fc.rs` с RatedTechnique impl. Header-doc с algorithm reference + invariants.
2. **3a.2** Регистрация в registry; rater автоматически подхватывает.
3. **3a.3** Cross-validation: 50 puzzles из forum_hardest_1106 с известным SE 8.5–9.5 — наш rater даёт se_score ±0.5.
4. **3b.1** Создать `nested_aic_reverse.rs`, mirror `aic_reverse.rs:14-30` topology.
5. **3b.2** Реализовать preemption masking как separate inner function `preempt_simpler_paths(...)`. Это hot loop, AlphaEvolve может tune.
6. **3b.3** CLI: extend `gen-text` / `gen-dataset` с `--required nested-aic --nested-aic-fragment-len 6`.

### DoD

- [ ] `dynamic_fc.rs` self-contained, fires на known-DFC test grid, не fires на T3 test grid.
- [ ] Cross-validation: 50 forum_hardest puzzles известного SE 8.5–9.5 → наш se_score within ±0.5 RMS.
- [ ] `nested_aic_reverse.rs` строит ≥ 100 valid uniqueness-checked puzzles за 60 min single-thread, mean se_score ≥ 8.8.
- [ ] Preemption masking improves p10 se_score с baseline (without masking) by ≥ 1.0.
- [ ] Smoke: pipeline (Stage 3b output → Stage 2 vicinity → final rate) finds ≥ 1 puzzle se_score ≥ 9.3.
- [ ] Per-file benchmarks записаны в `bench-results/r3_4_stage3.json`.

### Risks

- DFC implementation сложный. Mitigation: начинаем с simplest Cell Forcing Chain (no nesting), верифицируем end-to-end, затем расширяем. Каждый sub-variant — потенциально separate file (`techniques/cell_fc.rs` → `dynamic_fc.rs`).
- Preemption masking может не сходиться — infinite loop. Hard cap на iterations + reject puzzle.

---

## Stage 4 — Integration + production dataset

**Зачем:** свести в воспроизводимый pipeline, сгенерировать первый hard-bucket dataset для R7.

### Steps

1. Combined CLI workflow (документировать в `docs/hard_puzzle_recipe.md`):
   - `ingest-seeds --in data/seeds/public_hardest/forum_hardest_1106.txt --out seeds.parquet --rate`.
   - `gen-text --required nested-aic --target-se 9.0 --count 5000` → bootstrap pool.
   - `gen-vicinity --seed-from <bootstrap+seeds.parquet> --target-se 9.5 --budget-sec 7200`.
   - `rate-batch --mode full --in <output>` → final SE annotation.
2. Production dataset spec: `data/generic_v1/9x9_se9plus_v1/` manifest:
   - 10k puzzles se_score ∈ [9.0, 11.0].
   - Buckets [9.0–9.3, 9.3–9.6, 9.6–10.0, 10.0+] по ~2500 каждый.
3. **Compute decision point:** после Stage 3 smoke на M3 Max мерим throughput → choose Mac vs cuda-host2 для production.
4. Runbook `docs/hard_puzzle_recipe.md`.

### DoD

- [ ] Bootstrap (Stage 3) ≥ 5k puzzles se_score ≥ 9.0 за ≤ 12 CPU·h.
- [ ] Vicinity boosts p90 se_score на ≥ 0.4.
- [ ] Dataset `9x9_se9plus_v1/` existing, distribution buckets ≥ 80% filled.
- [ ] Runbook reproducible.
- [ ] External cross-check (manual): 20 top-bucket puzzles через serate/SukakuExplainer Java, RMS < 1.0.

---

## Out of scope для R3.4 (R3.5 candidates)

- **3D Medusa** technique + reverse-constructor (separate file `techniques/three_d_medusa.rs` + `three_d_medusa_reverse.rs`).
- **MSLS / SK-Loop** detection (`techniques/msls.rs`) + seeding bias (research: ×20 yield).
- **JExocet** detection + targeted seeding (research: 75% of SE≥11.9 содержат JExocet).
- **Mutant/Franken Fish** (`techniques/franken_fish.rs`).
- **Pattern Overlay Method** — solving-side fallback (`techniques/pom.rs`).
- **16×16 hard generation** — perf-bound на R3.3.
- External SE rater integration в CI (сейчас ручной cross-check на 20 sample).

Каждый — отдельный single-file addition по AlphaEvolve контракту → cheap to add as separate items.

---

## Зависимости и порядок

```
Stage R (refactor) ──┬─→ Stage 0 (SE score) ──┬─→ Stage 1 (canonical) ─┐
                     │                          │                         ├─→ Stage 4 (integration)
                     │                          └─→ Stage 2 (vicinity) ───┤
                     │                                                    │
                     └─→ Stage 3a (DFC technique) ─→ Stage 3b (nested AIC reverse)
                     │
Stage S (seed pool) ─┘  (параллельно Stage R, no blocker)
```

**Hard order:**
- Stage R перед Stage 0 (registry/trait foundation для se_rating).
- Stage S параллельно — ingestion скрипт не зависит от refactor.
- Stage 0 + 1 + 2 параллелизуемы после R.
- Stage 3a перед 3b (rater должен видеть DFC прежде, чем мы делаем targeted constructor).
- Stage 4 последний.

## Решения, принятые user'ом

- ✅ Алгоритмы из Java SE — clean-room порт, не copy.
- ✅ Seed pool — публичные hardest, `data/seeds/public_hardest/`.
- ✅ Compute decision — adaptive, после Stage 3 smoke бенчмарка.
- ✅ Архитектурный принцип — single-file techniques, AlphaEvolve/OpenEvolve compatible.

## Стартовый шаг

**Stage R** — refactor под registry + RatedTechnique trait + bench harness. Затрагивает 15 existing файлов (минимально — добавление default impl), плюс новый `techniques/mod.rs` factory + `benches/techniques.rs`. После landing — open green tree для всех остальных Stage.

---

## Implementation log — 2026-05-12

Все Stages R/S/0/1/2/3a/3b ландились одним заходом через параллельных subagent'ов с двойным ревью (Claude reviewer + codex через mcp__codex). Финальная статистика: **299/299 тестов зелёных** (268 lib + 31 integration). Не закоммичено — user-side review pending.

### Что реально шипнулось vs план

| Stage | План | Шипнулось |
|---|---|---|
| R | RatedTechnique + registry + header lint + bench | ✅ как план. 15 техник под lint, blanket impl снят в Stage 0 |
| S | Скачать ph_2010.zip + top1465 + forum_hardest_1106 + ingest CLI | ✅ скачано: top1465 (1465), forum_hardest_1905 (48766), tdoku_puzzles2_17_clue (49158). ph_2010 не пытался (есть forum_hardest_1905). manifest count off-by-N — see followups |
| 0 | Per-technique se_rating + RateResult.se_score + schema bump | ✅ как план. AIC chain-len rating использует `eliminations.len()` как proxy (semantic-wrong, см. followup) |
| 1 | min-lex canonical (full) | ⚠️ **Approximate**: digit-relabel + transpose only (2 candidates). 3.13 µs/grid release. Покрывает 9!=362880-orbit и ×2 transpose-orbit. Full (band/stack/row/col) — Stage 1.5 followup |
| 2 | vicinity engine + gen-vicinity CLI | ✅ как план. Mutate-1/2/3, cascade L1/L2/L4 (L3 backtrack-count skipped: backtracker не экспортирует counter). Single-threaded; rayon — followup |
| 3a | Cell Forcing Chain + T4Plus pass | ✅ как план. ≤4 candidates limit. Включает Contradiction sub-case. Region FC + Dynamic (nested) FC — followup |
| 3b | Nested AIC reverse + preemption masking | ⚠️ **v1 stub**: long-AIC (chain_target = 5) + preemption masking loop, **не** independent fragment search. Preemption masking работает реально (regression test проверяет). Full nested fragment graph search — followup |

### Review findings (combined)

**Chunk A** (R + 1): GREEN. Latent hazard: `canonical_form` транспонирует безусловно — корректно только при `BR == BC`. Все текущие callers — 9×9, hazard для будущих non-square (e.g. 6×6 BR=2/BC=3).

**Chunk B** (S + 0): GREEN. 4 non-blocking major: (1) AIC chain-len proxy неточный, (2) sha256 verification в download script отсутствует, (3) manifest.json count off-by-N от trailing newline + comment lines, (4) ingest skip stale manifest при rerun.

**Chunk C** (3a + 3b + 2): **NO GREEN изначально** → fixup landed.
- CRIT-1: `easy_techniques_below(Aic, 6.6)` при `preempt_max_se=6.5` исключал AIC из easy → short-AIC bypass не маскировался. **FIXED**: threshold → 5.0 (= aic.rs base_rating). Subagent аудировал и поправил 6 других неверных порогов (SimpleColoring 6.0→4.0, Skyscraper 6.1→4.0, TwoStringKite 6.2→4.1, NakedQuad 5.0→4.0, HiddenQuad 5.4→5.0, AlsXz 7.0→7.5).
- CRIT-2: iter-cap exit возвращал `Some` без re-check. **FIXED**: `rate_restricted_shows_progress` helper + final verification, возвращает `None` если bypass остался.
- MAJ-3: `frontier.is_empty()` ≠ no progress (singles cascade не в frontier). **FIXED**: trace check + solved_count delta.
- MAJ-4: bypass cell = "first unsolved" arbitrary. **PARTIALLY FIXED**: progress detection теперь корректный; precise cell targeting требует `rate_excluding_mut` API (TODO v2).
- MIN-5: vicinity dedup insert post-cascade. **FIXED**: hash insert pre-cascade.

Regression test `preemption_masking_blocks_short_aic_bypass` (seeds 0..8, ассертит `rate_restricted_shows_progress=false`) — passing.

### Followups (откладываются до Stage 4 или после калибровки)

**В .rs коде:**
- [canonical.rs] `BR == BC` guard на transpose. `debug_assert_eq!(BR, BC)` или compile-time gate. **Блокер для будущих non-9×9.**
- [aic.rs:259-263] AIC chain-len через `TechniqueProgress` ABI extension. Сейчас proxy через `eliminations.len()`. Затронет TechniqueProgress ABI version в lint test — нужен careful bump.
- [nested_aic_reverse.rs MAJ-4 followup] `rate_excluding_mut` API возвращающий (RateResult, Grid) — позволит precise bypass cell.
- [vicinity.rs] backtrack-count L3 prefilter (требует expose counter из `backtracker.rs`).
- [vicinity.rs / nested_aic_reverse.rs] rayon parallelism — single-threaded v1.
- [canonical.rs] full min-lex (band/stack/row/col perms с prefix pruning).

**В data + scripts:**
- [scripts/download_public_hardest.sh] sha256 verification после download + manifest regeneration unconditional.
- [data/seeds/public_hardest/*/manifest.json] regenerate `count` from real puzzle count (skip blank + `#` comment lines), not `wc -l`.

**В benchmarking:**
- DoD ещё не закрыты: (1) canonical_hash 10k random non-iso collision test, (2) RMS error vs known SE ratings на forum_hardest sample — сейчас просто distribution sanity (se_score_smoke.rs).
- per-technique `cargo bench --bench techniques` JSON output для AlphaEvolve — harness compiles, но measured numbers не записаны в repo.

### Stage 4 (production dataset) — не начат, как договаривались

User reserved right to discuss compute budget + final pipeline shape после смокинга Stage 3. Bootstrap call from Stage 3b smoke: 3 пазла за 5.4s, throughput ~0.5 пазл/сек (single thread). Для 10k dataset = ~5.5 wall-hours на одном ядре или ~30 минут на 16 ядер с rayon. Прикинуть на cuda-host2 vs M3 Max после fix последнего followup'а (rayon в Stage 2/3b).

---

## R3.4.5 — calibration + coverage expansion (landed 2026-05-12)

После Stage R/S/0/1/2/3a/3b провели clean-room audit от 1to9only SudokuExplainer (`/tmp/sudoku_explainer/`) и empirical calibration на 124 пазлах через serate. Audit показал систематические biases (наш AIC `5.0+0.1*elim_count` vs SE `6.6 + step_schedule`; HoDoKu-derived SimpleColoring/Skyscraper/TwoStringKite — SE их вообще не имеет standalone; NakedQuad/HiddenQuad off). Coverage gap: 24 пазла SE 9–11.1 → у нас T4Plus residual 7.5 (bias −3.53).

### Phase A — CFC fix + Region FC (GREEN после box-geom fixup)

- `cell_fc.rs:152` guard `k<2` → `k<3` (SE skips 2-cand pivots — handled separately as Y-Chain at 6.6).
- `cell_fc.rs:300` base 7.5 → 8.0 (SE `getDifficulty()` for Multiple FC level=0).
- **NEW** `src/generic/techniques/region_fc.rs` (542 LOC, base 7.6). Pivot = (region, digit, positions). Mirror of Cell FC structure.
- 1 critical bug найден ревью (box-geom для non-square BR/BC) — fixed inline (`n_box_cols = N/BC` вместо `N/BR`).
- 272/272 lib tests.

### Phase B — AIC re-calibration + ABI bump (GREEN)

- `TechniqueProgress.chain_len: Option<u32>` field added (ABI extension).
- `aic.rs` BFS теперь tracks `firing_depth` = edge count → `progress.chain_len`.
- `se_rating` formula заменена: clean-room port из SE `ChainingHint.getLengthDifficulty()`. Base 6.6 + alternating-multiplier step schedule (×3/2, ×4/3 alternating from 4), capped at 7.5 (наш house cap, TODO drop когда Dynamic FC landed).
- Codex verified first 10 step thresholds `{4,6,8,12,16,24,32,48,64,96}` match SE source line-for-line.
- 272/272 lib tests.

### Phase C — Dynamic FC (GREEN после base 8.5→9.0 fixup)

- **NEW** `src/generic/techniques/dynamic_fc.rs` (500 LOC). Level=1 dynamic, base **9.0** (SE формула 8.5 + 0.5×level).
- Algorithm: внутри outer hypothesis branches на FIRST bivalue cell (v1 simplification — SE branches на все). Persistent eliminations between sibling sub-branches (correctness-equivalent под bivalue=2 invariant per Codex review).
- 1s wall-time abort guard.
- Reviewer ловил initial base=8.5 (mis-aligned vs SE level=1 → 9.0) — fixed inline.
- 278/278 lib tests.

### Phase D — Nested FC level=2 (GREEN после stale-bv guard)

- **NEW** `src/generic/techniques/nested_fc.rs` (381+9 LOC). Level=2, base **9.5**.
- Algorithm: max_depth=2, branches на до 2 bivalue cells per recursion level (`max_bivalue_branches`).
- 1 major bug найден ревью (stale bv_cell may have been solved by sibling consensus → `Grid::assign` not guarded → `solved_count` corruption) — fixed inline (skip `if g.solved[*bv_cell] != 0`).
- 5s wall-time abort guard.
- 283/283 lib tests.

### Финальная статистика после R3.4.5

| Фаза | New T4Plus tech | base | files added |
|---|---|---|---|
| A | RegionForcingChain | 7.6 | region_fc.rs (542) |
| C | DynamicForcingChain | 9.0 | dynamic_fc.rs (500) |
| D | NestedForcingChain | 9.5 | nested_fc.rs (390) |
| B | (re-calibrated AIC) | 6.6 | result.rs ABI bump |

**T4PLUS_LIST cascade order** (cheap-first): `CellFC(8.0) → RegionFC(7.6) → DynamicFC(9.0) → NestedFC(9.5)`. Wait — RegionFC base 7.6 < CellFC 8.0; should reorder? — Yes, technically. Но в текущей impl rater пробует список последовательно и принимает первое срабатывание; cheap-first означает «фастер по wall time», что условно равно «меньше hypothesis evals». RegionFC = 27 regions × 9 digits = 243 pivots; CellFC = 81 cells × 3-4 cands = 240-320. Roughly equal, current order CellFC→RegionFC ok by precedent. **TODO**: empirically бенчить, swap если RegionFC статистически дешевле.

### Honest gaps after R3.4.5

1. **Empirical serate validation не закрыта.** Serate jar пропал после Phase 1 calibration; не пересобрали. Все SE weights в Phase A/B/C/D — derived из Java source чтением, не empirically diffed. Acceptable for now since clean-room порт structurally правильный.
2. **AIC `min(7.5)` cap remains** — Phase C/D landed, теперь можно drop'нуть (TODO в comment), но не блокер.
3. **Phase D narrow latent corruption fixed**, but tests don't cover the specific case (stale bv_cell solved by sibling). Phase E hardening: add specific regression test.
4. **Level=3+ nested FC** не сделан (SE 10.0+). Maximum SE rating ~11.9 = level=7. Если нам нужно покрытие до SE 11+ для production dataset, нужно extend Phase D recursion (max_depth=3..7, max_bivalue_branches=3..4).
5. **Bivalue-only branching approximation**: SE's level=1+ inside the hypothesis use locked-candidates + naked pairs propagator. Мы — только naked+hidden singles. Это значит некоторые SE chains short-circuit'нут раньше у SE чем у нас → наш chain_len будет завышен → длинный bonus. Bias direction: наш rating slightly HIGHER чем SE на тех же пазлах. Magnitude TBD.

### Phase D готова, Stage 4 (production dataset) всё ещё ждёт user discussion

Forcing chain family complete (Cell, Region, Dynamic level=1, Nested level=2). Покрывает SE 6.6 (AIC) → 9.5 (NestedFC) continuously. T4Plus residual sentinel 7.5 имеет смысл downgrade'нуть теперь — он редко будет fire'ить, и когда fire'ит — пазл реально hard (10+).

---

# ============================================================
# POST-DISASTER STATE — 2026-05-13 (context compression survival anchor)
# ============================================================

Это **финальная секция плана** на момент компрессии контекста. Содержит всё что критично знать чтобы продолжить работу. Если что-то противоречит выше — приоритет этой секции (она новее).

## Disaster summary

**2026-05-12 22:24:53** — single destructive git op (`git restore` / `git checkout --` или эквивалент) был запущен одним из subagent'ов (вероятно Phase F при попытке «restore clean base» при concurrent edits с Phase E). Это **массово откатило tracked файлы** к HEAD состоянию, потеряв ~100KB+ uncommitted work как внутри `sudoku_rs_core/`, так и в `docs/v45_latent_reasoning_synthesis.md`, `.gitignore`, `CLAUDE.md`, `web/affine-dashboard/src/style.css`.

**Memory saved as feedback rule:** `~/.claude/projects/-Users-dleonenko-latent-reasoning-design/memory/feedback_subagent_git_safety.md` + `MEMORY.md` index entry. **Все будущие subagent briefs ОБЯЗАНЫ содержать git-safety preamble** запрещающий destructive ops.

## Restoration via Time Machine

**Best pre-disaster snapshot**: `/Volumes/.timemachine/A020F651-D4C6-41F5-9368-9731CF243124/2026-05-12-213154.backup/2026-05-12-213154.backup/System - Data/Users/dleonenko/latent-reasoning-design/`

Snapshot mtime **2026-05-12 21:31:54** — содержит полный pre-disaster state Phase A-D + Stage R/S/0/1/2/3a/3b. Phase E (NestedFC level=3) и Phase F (chaining_propagator) landed позже 21:31, но ДО 22:24:53 disaster.

**Доступ к TM требует Full Disk Access** для `/Applications/Claude.app` в System Settings → Privacy & Security → Full Disk Access. Уже выдан 2026-05-13 ночью.

## Что committed в git (3 recovery коммита)

```
72d767c restore: .gitignore + CLAUDE.md from TM (one-job rule + utm-jax ignores)
aaf5e4c restore: v45 synthesis journal (+1032 lines) + affine-dashboard style.css from TM
2be838e wip(rust): checkpoint R3.4 + R3.4.5 forcing-chain rater + reverse-construct
02344c5 feat(web): restore EvalTable + BucketLineChart (pre-session HEAD)
```

`2be838e` содержит ВСЁ что сейчас есть в дереве sudoku_rs_core (партиально-восстановленное state). Если нужно откатить локальные правки — `git reset --hard 2be838e` (но БЕЗ destructive op в subagent'ах).

## Текущее state дерева (post-restore, post-commit)

**Tree компилится** (`cargo check` clean, 3 pre-existing warnings). Lib tests — последний полный прогон до disaster показывал 277/277 green.

### Что РАБОТАЕТ (в working tree + committed)

**Stage R/S/0/1/2/3a/3b — все технические файлы present:**
- `src/generic/techniques/cell_fc.rs` (Phase A: k≥3 guard, base 8.0, Contradiction sub-case)
- `src/generic/techniques/region_fc.rs` (Phase A NEW, base 7.6, pivot = region×digit)
- `src/generic/techniques/dynamic_fc.rs` (Phase C NEW, base 9.0, level=1 dynamic nesting, FIRST bivalue branching)
- `src/generic/techniques/nested_fc.rs` (Phase D+E: max_depth field, `level_2()`/`level_3()` constructors, base 9.5/10.0)
- `src/generic/techniques/chaining_propagator.rs` (Phase F NEW: shared singles+locked+pairs+ALS-XZ cascade для hypothesis propagation)
- `src/generic/canonical.rs` (Stage 1: approximate min-lex, digit-relabel + transpose, ~3 µs release)
- `src/generic/vicinity.rs` (Stage 2: hill-climbing engine, cascade L1+L4)
- `src/generic/nested_aic_reverse.rs` (Stage 3b: v1 long-AIC + preemption masking)
- `src/io/ingest.rs` (Stage S: txt → parquet ingest helper)
- `tests/{cli_gen_vicinity, cli_ingest_seeds, se_score_smoke, technique_header_lint}.rs`
- `benches/techniques.rs` + Criterion dev-dep in Cargo.toml
- 13 T2/T3 technique files: 6-section header docs + per-technique `impl RatedTechnique` с SE-weights (Stage R + Stage 0)
- `src/generic/techniques/aic.rs`: chain_len wired through BFS + clean-room SE step schedule formula (base 6.6, cap dropped post-Phase C/D)
- `src/generic/techniques/result.rs`: `chain_len: Option<u32>` field (Phase B ABI bump)
- `docs/{R3_4_hard_puzzle_generation_plan, alphaevolve_contract, cfc_root_cause, se_audit_report, se_calibration_report (v1), se_calibration_report_v2}.md`
- `scripts/{download_public_hardest.sh, compare_serate_vs_ours.py, build_serate.sh}` (project root)
- `tools/sudoku_rs_core/scripts/validate_se_scores.py`
- `data/seeds/public_hardest/{top1465, forum_hardest_1905, tdoku_puzzles2_17_clue}/` (datasets downloaded; gitignored)

### Что ПРОПАЛО и НЕ восстановлено (КРИТИЧНОЕ: блокирует cargo run cli)

**Эти изменения** не recovered после disaster — нужно restoration phase 2 из TM 21:31 snapshot:

1. **`src/cli.rs`** — отсутствуют subcommands:
   - `IngestSeeds` (Stage S) — `cargo run -- ingest-seeds` сейчас "unrecognized subcommand"
   - `GenVicinity` (Stage 2) — то же
   - `gen-text --required nested-aic` dispatch (Stage 3b)
   Текущий cli.rs имеет только 9 Cmd variants (Solve, Rate, Gen, GenDataset, GenText, RateBatch, ReverseConstruct, BenchReverseHitrate, ReRate). Pre-disaster имел 11+.

2. **`src/generic/mod.rs`** — отсутствуют module declarations:
   - `pub mod canonical;`
   - `pub mod vicinity;`
   - `pub mod nested_aic_reverse;`
   Файлы существуют на disk но не компилируются (Rust игнорирует .rs не объявленные как pub mod). Это значит **их тесты НЕ запускаются** и **они не доступны через crate API**. Compile clean only because nothing references them.

3. **`src/schema.rs`** — пропал `se_score: Float64` column в parquet schema.

4. **`src/rerate_generic.rs`** — пропала `se_score` propagation в rerate output.

5. **`src/pipeline_writer_generic.rs`** — частично восстановлено (2 матча `se_score`). Стоит сверить с 21:31 snapshot для полноты.

6. **`src/generic/rater.rs`** — было полностью реинтегрировано первой restoration subagent'ой (см. её отчёт), но НЕ полностью совпадает с 21:31 версией (которая имела 19 матчей `T4PLUS_LIST|FC variants` vs текущая ~неизвестно сколько). **Стоит сравнить с 21:31 и предпочесть TM версию**, она проверена empirical validation v2.

7. **`src/io/mod.rs`** — `pub mod ingest;` declaration: проверить.

## Restoration phase 2 plan (next step)

**Approach**: surgically merge from TM 21:31 snapshot файлы которые партиально-восстановились или потеряны:

```bash
SNAP=/Volumes/.timemachine/A020F651-D4C6-41F5-9368-9731CF243124/2026-05-12-213154.backup/2026-05-12-213154.backup
ROOT="$SNAP/System - Data/Users/dleonenko/latent-reasoning-design"
CURRENT=/Users/dleonenko/latent-reasoning-design

# Files to overwrite from snapshot (these are КРИТИЧНО pre-disaster correct):
cp "$ROOT/tools/sudoku_rs_core/src/cli.rs"                              "$CURRENT/tools/sudoku_rs_core/src/cli.rs"
cp "$ROOT/tools/sudoku_rs_core/src/generic/mod.rs"                      "$CURRENT/tools/sudoku_rs_core/src/generic/mod.rs"
cp "$ROOT/tools/sudoku_rs_core/src/schema.rs"                           "$CURRENT/tools/sudoku_rs_core/src/schema.rs"
cp "$ROOT/tools/sudoku_rs_core/src/rerate_generic.rs"                   "$CURRENT/tools/sudoku_rs_core/src/rerate_generic.rs"
cp "$ROOT/tools/sudoku_rs_core/src/pipeline_writer_generic.rs"          "$CURRENT/tools/sudoku_rs_core/src/pipeline_writer_generic.rs"
cp "$ROOT/tools/sudoku_rs_core/src/generic/rater.rs"                    "$CURRENT/tools/sudoku_rs_core/src/generic/rater.rs"
cp "$ROOT/tools/sudoku_rs_core/src/io/mod.rs"                           "$CURRENT/tools/sudoku_rs_core/src/io/mod.rs"
cp "$ROOT/tools/sudoku_rs_core/src/io/ordered.rs"                       "$CURRENT/tools/sudoku_rs_core/src/io/ordered.rs"
cp "$ROOT/tools/sudoku_rs_core/src/generic/spec.rs"                     "$CURRENT/tools/sudoku_rs_core/src/generic/spec.rs"
cp "$ROOT/tools/sudoku_rs_core/src/generic/reverse_construct.rs"        "$CURRENT/tools/sudoku_rs_core/src/generic/reverse_construct.rs"
```

После copy: компилировать может НЕ сразу — 21:31 snapshot не имеет Phase E (NestedForcingChainL3) и Phase F (chaining_propagator) wiring в rater.rs. Нужно **layer на top**:

- Re-add `TechniqueId::NestedForcingChainL3` variant в `techniques/mod.rs`
- Re-add `AnyTechnique::NestedForcingChainL3` в rater.rs + match arms + T4PLUS_LIST entry
- Re-add `pub mod chaining_propagator;` в `techniques/mod.rs`
- Update cell_fc.rs / region_fc.rs / dynamic_fc.rs / nested_fc.rs — use `chaining_propagator::propagate_hypothesis` instead of `propagate_singles` в hypothesis sites (Phase F edits)

Эти Phase E/F edits **сейчас работают в working tree** (за счёт пост-disaster recovery subagent'ов), их можно сохранить через 3-way merge — backup current state файлов перед copy, потом cherry-pick Phase E/F deltas.

**Recommended sequence**:
1. `cp -r $CURRENT/tools/sudoku_rs_core /tmp/sudoku_rs_core_PRE_PHASE2/`  (safety; ALREADY есть `/tmp/sudoku_rs_core_PRE_RESTORE_*`)
2. Copy 10 files из TM (above)
3. Append Phase E + Phase F integration на top
4. `cargo check` clean
5. `cargo test --lib` clean (target 290+ tests passing — includes vicinity, canonical, nested_aic_reverse которые сейчас не компилируются)
6. Commit как `restore: cli.rs+schema+rerate+rater post-TM-2131 reintegration`

**Hard safety constraints для subagent если делегируется**:
- **NO destructive git ops** (см. feedback_subagent_git_safety memory)
- Wrap каждый cp в backup-then-copy pattern
- НЕ overwrite tools/sudoku_rs_core/src/generic/techniques/{cell_fc,region_fc,dynamic_fc,nested_fc,chaining_propagator}.rs — они уже актуальные (Phase F propagator dispatch)

## Empirical calibration state (validation v2, 124 пазлов)

Run completed BEFORE disaster, serate jar may need rebuild via `scripts/build_serate.sh`:

| Bucket | N | Mean diff | Vердикт |
|---|---|---|---|
| ALS-XZ | 22 | **+0.03** | калиброван |
| RegionFC | 27 | −1.02 | base 7.6 systematic high; false-positive на SE=4.5 |
| NestedFC | 15 | −1.19 | base 9.5 systematic high |
| DynamicFC | 1 | −0.80 | мало данных |
| CellFC | 1 | −1.30 | мало данных |
| T4Plus residual / empty frontier | 16 | −3.53 | coverage gap SE 11+ |

**Overall**: RMS 1.79 (Phase 1 baseline) → **1.53 raw / 0.56 corrected** (Phase A-D). Systematic bias dominates remaining gap. Reports: `docs/se_calibration_report.md` (v1), `docs/se_calibration_report_v2.md`.

**False-positive in RegionFC**: пазл SE=4.5 у нас даёт RegionFC fire. Это структурный — наш propagator слабее SE's (singles only vs SE singles+locked+pairs). **Phase F (chaining_propagator) addresses this structurally** — добавляет locked-candidates+pairs+ALS-XZ внутрь hypothesis testing. Эффект Phase F на calibration НЕ измерен (validation v2 ran до Phase F land).

## Open work (после restoration phase 2)

### Priority 1 — re-run calibration v3

После restoration phase 2 (рабочий rater.rs FC integration), пере-замерить:
```bash
bash scripts/build_serate.sh   # idempotent re-build serate jar
cargo run --release -- ingest-seeds --input /tmp/r3_4_calib_200.txt --output /tmp/r3_4_calib_v3.parquet
cargo run --release -- rate-batch --input /tmp/r3_4_calib_v3.parquet --output /tmp/r3_4_ours_v3.parquet --mode full
python scripts/compare_serate_vs_ours.py
```

Ожидание: RegionFC bias уменьшится (был −1.02; Phase F propagator должен fire'ить меньше, точнее), T4Plus residual count уменьшится (NestedFC level_3 закроет верхние SE 10-11).

### Priority 2 — Phase G (deferred)

- **Level=4+ Nested FC** — extend nested_fc.rs до depth=4+. SE rating 10.5+. Закрывает остаток T4Plus residual.
- **XY-Chain (Y-chain) variant of AIC** — base 7.0 vs X-chain 6.6. Сейчас весь AIC trafficed через X-chain base. Distinguishing исправит небольшую часть bias.
- **Binary Forcing Chain** (k=2 cells, Y-Chain semantics, rating ~6.6–7.0) — SE handles 2-cand cells separately, мы skip'аем их в Cell FC (k≥3 guard) → они вообще не fire'ят сейчас. Closing this gap = SE band 6.6–7.0 actually covered.

### Priority 3 — Stage 4 (production dataset) — user gate

**НЕ начинать без discussion с user.** Estimated bootstrap throughput post-fix: ~0.5 puzzle/sec single-thread (Phase 3b smoke). 10k SE 9+ dataset = ~5.5 wall-hours / 1 core, ~30 min на 16 cores с rayon. После Priority 1 calibration v3 будут реальные numbers для compute planning (Mac M3 Max vs cuda-host2 CPU).

## Memory anchors (что должно survival'нуть compression)

1. **3 git commits сохраняют ВСЁ что есть на диске**: 2be838e + aaf5e4c + 72d767c. Worst case: `git reset --hard 2be838e` для resetting к этому checkpoint'у.
2. **TM 21:31 snapshot** в `/Volumes/.timemachine/A020F651.../2026-05-12-213154.backup/...` — последний полный pre-disaster state. FDA выдан.
3. **Pre-restore safety backups** в `/tmp/pre_restore_2026_05_13/` (gitignore + CLAUDE + v45 + style.css до cp).
4. **Feedback memory** `feedback_subagent_git_safety.md` — git-safety preamble обязателен в КАЖДОМ бриф'е subagent'а.
5. **Этот plan doc** в `tools/sudoku_rs_core/docs/R3_4_hard_puzzle_generation_plan.md` — full state of affairs.

## Quick verification commands (после restoration phase 2)

```bash
cd /Users/dleonenko/latent-reasoning-design/tools/sudoku_rs_core

# Fingerprint что restoration phase 2 succeeded:
grep -c "IngestSeeds\|GenVicinity" src/cli.rs                    # ожидаем ≥ 4
grep -c "se_score" src/schema.rs                                  # ≥ 1
grep -cE "canonical|vicinity|nested_aic" src/generic/mod.rs       # ≥ 3 (pub mod declarations)
grep -c "se_score" src/rerate_generic.rs                          # ≥ 1
grep -cE "T4PLUS_LIST|CellForcingChain|RegionForcingChain|DynamicForcingChain|NestedForcingChain" src/generic/rater.rs  # ≥ 15

# Compile + tests:
cargo check                                                       # clean (3 pre-existing warnings ok)
cargo test --lib 2>&1 | tail -3                                   # 290+ passed

# E2E smoke:
cargo run --release -- ingest-seeds --input data/seeds/public_hardest/top1465/top1465.txt --output /tmp/smoke.parquet --max 50
cargo run --release -- rate-batch --input /tmp/smoke.parquet --output /tmp/smoke_rated.parquet --mode full
```

Если все 5 grep'ов > 0 и cargo test проходит — restoration phase 2 succeeded. Можно двигаться к Priority 1 (calibration v3).

---

## Calibration v3 — 2026-05-13 (post-restoration + L3 wiring)

L3 wired в `AnyTechnique`/`T4PLUS_LIST`/`all_techniques_9x9()` (commits `b166481`, `5056046`). 291/291 lib tests green. Double-review (Opus + codex) caught registry omission в `all_techniques_9x9()` → fixed.

Calibration v3: 200 пазлов (100×SE 8-10 + 100×SE 10+), serate jar rebuilt, default format. Full pipeline через `cargo run --release -- rate-batch --mode full`.

**Результаты vs v2:**

| Метрика | v2 (124 пазла) | v3 (200 пазлов) |
|---|---|---|
| Overall RMS | 1.79 → 1.53 | **1.326** |
| Corrected RMS | 0.56 | **0.458** |
| ALS-XZ bias | +0.03 | +0.03 (stable) |
| RegionFC bias | −1.02 | 0 fires в v3 set (?) |
| CellFC bias | мало данных | −0.82 (n=33) |
| DynamicFC bias | −0.80 (n=1) | −2.00 (n=12) |
| NestedFC L2 bias | −1.19 (n=15) | −1.53 (n=65) |
| **NestedFC L3 bias** | n/a | **−1.01 (n=17)** ← Phase E подтверждён |
| T4Plus residual | −3.53 (n=16) | "none"=−3.5 (n=5) |

Report: `tools/sudoku_rs_core/docs/se_calibration_report_v3.md`.

**Interpretation:**

1. **Bias direction:** все FC техники systematically UNDER-estimate vs SE на SE 10+ пазлах. Не overshoot. Мы fire'аем (например) NestedFC L2 base=9.5, а SE репортит ER=11. Это значит на этих пазлах **SE использует более тяжёлые техники, которых у нас нет** (Dynamic FC level=3+, Multiple FC nested deeper, JExocet, SK-Loop, MSLS, Mutant Fish).

2. **Recommended weight edits в reporte неверны.** Скрипт предлагает `9.5 → 8.0` etc., но это исходит из mean diff. Если опустить веса — bias станет ХУЖЕ на тех же пазлах (мы и так underestimate). Реальный fix = добавить недостающие техники, чтобы их се_rating заглушил NestedFC на этих пазлах.

3. **5 пазлов с empty frontier** (top_tech="none", все SE=11): наш rater вообще ничего не fire'ит, но grid не solve'нулся (стало T4Plus sentinel). Coverage gap.

4. **Outlier:** CellFC fires на SE=4.5 пазл → bias +3.5 (max overestimate). Это false-positive: SE решает пазл коротким techniques, мы fire'аем 8.0-rated CFC. Phase F propagator должен снизить такие, but explicit guard'ов нет. **TODO**: investigate this single puzzle, понять, какой short tech SE использует, добавить early-out в cell_fc.

5. **Conclusion:** Forcing chain family **structurally правильна** (L3 bias −1.0, the lightest of all FC, suggests our impl is closest to SE на тех пазлах где L3 fires). Calibration vs serate has converged до RMS 0.458 corrected — это нижняя граница без новых техник.

### Open priorities post-v3

**Priority 1 — Phase G coverage expansion** (для SE 10.5+ buckets):
- `dynamic_fc.rs` extend max_level до 3+ (sub-branching внутри hypothesis, parallels NestedFC max_depth pattern). SE rating jump 9.0 → 9.5/10.0.
- XY-Chain (Y-Chain) дискриминация vs X-Chain в `aic.rs`. SE base 7.0 vs 6.6.
- 2-cand pivot handling в Cell FC (current k≥3 guard skips bivalue cells — closes SE band 6.6-7.0 currently uncovered).

**Priority 2 — Stage 4 production dataset** (user-gated): bootstrap throughput ≈ 0.5 puzzle/sec single-thread; для 10k SE 9+ dataset ≈ 5.5h × 1core / 30min × 16cores. Pending user discussion about Mac M3 Max vs cuda-host2 CPU.

**Priority 3 — outlier debugging:** find that SE=4.5 puzzle где CFC false-fires, add early-out heuristic.

**Memory anchor:** RMS 0.458 corrected — это упёрлось в coverage ceiling. Дальнейшие refinements weight'ов не дают yield, нужны новые techniques.
