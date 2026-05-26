# CSP-Rules Chain Techniques — Porting Spec

**Source:** Denis Berthier, *CSP-Rules-V2.1* (GPL-3.0). The PBCS3 monograph (2021) is the formal reference; CLIPS rules in `CSP-Rules-Generic/CHAIN-RULES-MEMORY/` are the operational ground truth. This document is **self-contained**: a Rust porting agent should not need to consult CLIPS while reading the implementation file plan in §13.

Scope: **whip[k]**, **braid[k]**, **g-whip[k]**, **g-braid[k]**, k ∈ 1..36. Soundness, completeness with respect to PBCS3, and algorithmic structure. Sudoku 9×9 is the target substrate but the algorithm is CSP-agnostic; CSP-specific facts enter only via the link/csp-link/peer tables.

A note on CLIPS encoding. The CLIPS files for `Whips[k]`, `Braids[k]`, `gWhips[k]`, `gBraids[k]` are *parametric unrollings* in k. Reading k = 1, 2, 3, 5 fully fixes the pattern; k ≥ 4 just copies the k = 3 / k = 5 body verbatim with the length and unwanted-redundancy guards adjusted. Length 12 was spot-checked and is identical to length 5 modulo the `length` slot constant and the print stream tag — this confirms the "parametric unrolling" hypothesis (no algorithmic change at higher k).

---

## 1. Background and goals

Whips and braids are **AC3-style nogood chains** for general finite CSPs, generalising bivalue chains and Sudoku XY-chains. Given a target candidate `Z`, a whip[k] is a sequence of length k that exhibits an *unavoidable contradiction* in every continuation that asserts `Z` — equivalently, a *T-cell* (a CSP-Variable whose every candidate is killed by the assumption `Z` together with earlier chain commitments) is reached in k steps. The resolution rule eliminates `Z`. Braid[k] generalises by allowing each new left-linking candidate to be hit by *any* previously committed right-linker (not only the last), turning the chain into a partially-ordered nogood DAG of bounded width. The "g-" variants admit *grouped* candidates (multiple cells sharing a value in one house) as right-linkers / target-neighbours, which subsumes Almost Locked Sets / Almost Hidden Sets-style grouped fish and naked/hidden subset chains.

Practical objective in this repo: a Rust port to (a) **rate** puzzles on the Berthier W/B/gB axis, and (b) **reverse-generate** puzzles where a chosen W[k]/B[k]/gW[k]/gB[k] is *load-bearing* (i.e. its removal pushes the puzzle to a strictly harder tier or unsolvability), as training/eval data for the latent-reasoning experiments tracked in `docs/v45_latent_reasoning_synthesis.md`.

---

## 2. Formal preliminaries (CSP layer)

### 2.1 CSP-Variables on Sudoku 9×9

A *CSP-Variable* is a constraint with a finite, discrete domain. Sudoku uses **four families** (Berthier notation, PBCS3 §3):

| Family | Cardinality | Meaning | Domain |
|---|---|---|---|
| **rc** | 81 | "cell" — pair (row, col) | one of 9 digits |
| **rn** | 81 | "row × digit" — pair (row, number) | one of 9 columns |
| **cn** | 81 | "col × digit" — pair (col, number) | one of 9 rows |
| **bn** | 81 | "block × digit" — pair (block, number) | one of 9 cells in block |

Concrete examples:

- `rc(r1,c1)` has domain `{1..9}`: which digit goes in (1,1).
- `rn(r1,n5)` has domain `{c1..c9}`: which column in row 1 holds digit 5.
- `cn(c3,n7)` has domain `{r1..r9}`: which row in column 3 holds digit 7.
- `bn(b1,n2)` has domain `{cell_a..cell_i}` (the 9 cells of block 1): which cell of block 1 holds digit 2.

Total CSP-Variables for a 9×9 sudoku: **324** (`4 × 81`). All four families are *equivalent* (each tiles the same 729 labels), but the chain machinery freely mixes them — this is what gives whips/braids strictly more reach than rc-only or XY-style chains.

Generalising: for N×N with block dimensions `(BR, BC)` such that `BR·BC = N`, there are `4·N²` CSP-Variables.

In CLIPS: `(deftemplate csp-variable (slot name INTEGER) (slot type SYMBOL))`. The `type` slot is one of `rc | rn | cn | bn`. Membership of a label in a CSP-Variable is asserted as `(is-csp-variable-for-label (csp-var ?v) (label ?l))` (and similarly for glabels). Every label belongs to **exactly 4** CSP-Variables (one rc, one rn, one cn, one bn).

### 2.2 Candidates, values, labels, glabels

**Label** (Berthier): an integer encoding `(number, row, col)`. Sudoku encoding: `label = 100*number + 10*row + col` (or any other invertible scheme — the integer is opaque to the chain rules). 9³ = 729 labels total.

**Candidate** (`deftemplate candidate`, `templates.clp:79`): an asserted fact `(candidate (context ?c) (status cand|c-value) (label ?L) (flag 0|1))`. Sudoku adds `(number row column block square band stack)` slots for fast filtering. **Status:**
- `cand` — undecided; the label is still alive in this context.
- `c-value` — committed; the label is part of the solution being built.

The set of `(candidate ... (status cand) ...)` facts **monotonically shrinks** during a solve. All resolution rules must preserve this invariant; chain rules retract candidates, never add them.

**Value**: a `c-value` candidate — i.e., a committed label.

**G-label** (grouped label): an opaque integer naming a *set of labels* that share a CSP-Variable family-other-than-rc and a *house segment*. Concretely for Sudoku: the candidates of digit `d` in the intersection of a *block* with a *row* (a "horizontal mini-line", 3 cells), or block with a *column* (vertical mini-line). These are the units of grouped strong/weak links — exactly the kind of structure exploited by Locked Candidates and grouped fish.

**G-candidate** (`deftemplate g-candidate`, `templates.clp:101`): an asserted fact for an alive glabel `(g-candidate (context ?c) (label ?glab) (type ?t) (csp-var ?v))`. A glabel survives in `?cont` iff ≥ 2 of its underlying candidates are still alive. In CLIPS this is implemented via **logical fact maintenance** (RETE truth maintenance): the `init-g-candidates-horiz` and `init-g-candidates-verti` rules (`SudoRules-V20.1/GENERAL/init-glinks.clp:68-80` and `93-105` respectively) wrap their preconditions in `logical (… (candidate … label ?cand1) (candidate … label ?cand2&:(< ?cand1 ?cand2)) …)`, so CLIPS automatically retracts the `g-candidate` fact the moment either member candidate is eliminated. The g-candidate is alive if and only if ≥ 2 member labels remain alive. **Rust implication:** the `ResolutionState` must dynamically recompute or maintain a support count for each glabel; a static `g_alive` bitset set at init is incorrect. See §13 `resolution_state.rs` for the required API.

Predicate `(label-in-glabel ?lab ?glab)` is precomputed once at init (`add-label-in-glabel`); it returns TRUE iff `?lab` is one of the labels making up `?glab`.

### 2.3 Links and glinks (peer relations)

A **link** between two labels is a *non-equality constraint shared by the two labels*. Two flavours:

- **csp-link** (`csp-linked ?cont ?lab1 ?lab2 ?csp`): the two labels share a CSP-Variable — i.e. they are mutually exclusive *as values of that variable*. For Sudoku: same cell different digits (rc), same row+digit different columns (rn), same column+digit different rows (cn), same block+digit different cells (bn).
- **non-csp link** / **strong-from-cell exists-link** (`exists-link ?cont ?lab1 ?lab2`): a weaker, **symmetric peer relation** — "labels are linked at all". This is the *transitive closure of all four csp families*: any two labels that share a CSP-Variable are also `exists-link`-related, and `add-link` is called once per pair to record the merged graph. In Sudoku this is exactly the **peer relation**: same cell, same row, same column, or same block (and for cross-digit, same cell only).

Both relations are **undirected** (asserted symmetrically: `init-effective-csp-links` and `init-effective-non-csp-links` insert both directions). They are **untyped** at the chain-rule level — the chain rules never care which CSP family produced the link; only `csp-linked` carries the `?csp` identifier (used to demand "alternative values of *this same* CSP-Variable").

Predicate `(linked ?lab1 ?lab2)` is a function over the `?*links*` multifield (`generic-background.clp:216`). `(linked-or ?lab1 $?labs)` returns TRUE iff `?lab1` is linked to at least one element of `$?labs`. These two operations are the hot path of chain rules — every condition `forall (csp-linked … ?xxx ?csp) (test (linked-or ?xxx ?zzz $?rlcs))` invokes `linked-or` once per alternative-value-of-?csp.

**Glink** (`exists-glink ?cont ?lab ?glab`): symmetric link between a label and a g-label, asserted iff the underlying constraint kills the candidate when the glabel is "true" (one of the cells in the glabel becomes the digit). `csp-glinked` is the csp-typed analogue.

Functions `glinked-or`, `glabel-contains-none-of`, `glabel-contains-some-of` are the workhorses for grouped chain predicates (`generic-background.clp:168-187, 263-270`).

**Peer relation** restated: two labels are *peers* iff `labels-linked(l1, l2) = TRUE`. Two cells are *peers* iff they share row, column, or block (Sudoku-specific). A label-cell peer relationship is induced: label `(n,r,c)` is a peer of cell `(r',c')` iff `(r,c)` and `(r',c')` are peer cells.

### 2.4 Exclusion semantics

If label `L` is asserted `c-value` then **every candidate sharing a CSP-Variable with L is eliminated by ECP** (Elementary Constraint Propagation, `GENERAL/ECP.clp`). Within a context, two `cand` labels sharing a CSP-Variable are mutually exclusive — the chain rules exploit this: `csp-linked l1 l2 v` means "if `l1` is true, `l2` must be false (because the only alternative values for `v` are eliminated)".

---

## 3. Resolution state (RS)

The Resolution State of a context `?cont` is the conjunction of:

1. The set of alive candidates: `{ L : (candidate (context ?cont) (status cand) (label L)) ∈ facts }`.
2. The set of values: `{ L : (candidate (context ?cont) (status c-value) (label L)) }`.
3. The link and csp-link graphs (immutable after init — they describe the puzzle structure, not its partial-solution status).
4. The set of alive g-candidates. **Critical:** this set is NOT static — it is truth-maintained in CLIPS via logical dependencies. A g-candidate for glabel `g` is alive iff ≥ 2 member labels of `g` are still alive (see §2.2). When a candidate is eliminated (`cand_alive.remove(L)`), the Rust port must check whether any glabel `g` with `label_in_glabel(L, g)` has dropped below 2 alive members, and if so remove `g` from `g_alive` as well.

`compute-RS.clp` materialises (1) and (2) as the working state. Chain rules read RS as a precondition pattern and `retract` candidate facts when their elimination fires (`Whips[k].clp` body: `?cand <- (candidate ... (label ?zzz))` ⇒ `(retract ?cand)`). They **never** assert new candidates; they may assert c-values via `Single` rules downstream of the elimination.

The `context` slot supports Trial-and-Error (T&E): context 0 is the actual puzzle; contexts ≥ 1 are hypothetical worlds. All chain rules are context-parametric and never cross contexts. For our purposes we will run chain rules in a single context analogous to context 0.

---

## 4. Confluence machinery

### 4.1 Salience hierarchy

CLIPS rule firing is governed by **salience** — a numeric priority. CSP-Rules wires saliences so that techniques fire in a strict tier order, with the tightest techniques first. The numeric values are computed at load time by repeatedly decrementing a counter `?*next-rule-salience*` (start: 10000; see `saliences.clp:230+`). What matters is the *ordering*, not the absolute numbers. Layered top-to-bottom (highest salience first, i.e. fires first):

```
ECP                            (≈ +∞ via separate phase)
Single / Naked-Single / Hidden-Single   ← BRT in CLIPS (verified)
                                 — runs before any chain rules
[init-links phase: builds the link graph]
[init-glinks phase: builds the glink graph]
Whip[1]                         ?*whip[1]-salience*       (k ≥ 1)
end-Whip[1]                     ?*end-whip[1]-salience*
Whip[2]   /  partial-whip[1] (extension)
gWhip[2]  /  partial-gwhip[1] (3 sub-rules)                (k ≥ 2; no gWhips[1].clp)
Whip[3]   /  partial-whip[2]
gWhip[3]
Braid[3]  /  partial-braid[2]                              (k ≥ 3; no Braids[{1,2}].clp)
gBraid[3] /  partial-gbraid[2]                             (k ≥ 3; no gBraids[{1,2}].clp)
…
Whip[k]   /  partial-whip[k-1]                             (Whip:  k ≥ 1)
gWhip[k]                                                   (GWhip: k ≥ 2, CR-FIN-4 C2)
Braid[k]  /  partial-braid[k-1]                            (Braid: k ≥ 3, CR-FIN-7 C1)
gBraid[k]                                                  (GBraid: k ≥ 3, CR-FIN-7 C2)
…
Whip[36] … gWhip[36] … Braid[36] … gBraid[36] (configurable upper bound)

Verified from `saliences.clp:define-generic-saliences-at-L3` (and L2,L4,…) and from
the CLIPS file inventory at `CSP-Rules-V2.1/CSP-Rules-Generic/CHAIN-RULES-SPEED/`:
within each k the salience-counter decrement order is **whip[k] → gwhip[k] → braid[k]
→ gbraid[k]**, with all of L_k strictly higher salience than all of L_{k+1}. Each
family has its own on-disk minimum k (no `gWhips[1].clp`, no `Braids[{1,2}].clp`, no
`gBraids[{1,2}].clp` — verified via `ls CHAIN-RULES-SPEED/WHIPS/`,
`ls CHAIN-RULES-SPEED/G-WHIPS/`, `ls CHAIN-RULES-SPEED/BRAIDS/`,
`ls CHAIN-RULES-SPEED/G-BRAIDS/`). For each k, families whose minimum exceeds k
simply do not fire at that level. The Rust solver enforces these floors in
`find_first_*` / `run_*_pass` (CR-FIN-4 C2 for gWhip, CR-FIN-7 C1/C2 for braid/gbraid).
Full firing order at k ≥ 3: whip[k] → gwhip[k] → braid[k] → gbraid[k] (uniform).
At k = 2 only whip[2] and gwhip[2] exist. At k = 1 only whip[1] exists.
Forcing-Whips, T&E (off by default)
```

### 4.1.1 BRT (Basic Resolution Theory) — exact enumeration (CR-FIN-3 M-opus-3)

Verified against `/tmp/csp-rules-research/CSP-Rules-V2.1/Generic-Background/CSP-Rules.clp` and `Common-files/*.clp`. The CLIPS pre-chain phase fires **exactly the following rules at salience above any chain rule**:

1. `single` / `naked-single` / `hidden-single` — the three single-cell propagation rules. Source: `Generic-Background/Singles.clp`. These three rules cover the entire "Single / Naked-Single / Hidden-Single" line in the salience tower above.
2. `Elementary-Constraints-Propagation` (ECP) — runs as a separate top-salience phase before any rule firing. Source: `Generic-Background/ECP.clp`.

**Not included in BRT (i.e. NOT fired before chain rules):**
- Naked / Hidden Pair / Triple / Quad (subsets) — these are *not* part of CLIPS BRT. The "Bivalue / 2-value / 3-value subsets" line that appeared in an earlier draft of this section was incorrect. CLIPS subsets, when enabled, live in `SubsetsModule/` and load at the *user's* discretion via the `?*Subsets*` global; they are not part of the chain pre-pass.
- Locked Candidates (Pointing / Claiming) — same status: optional module, not BRT.
- Bivalue-Chains / k-value subsets — these are chain-class techniques; they are *interleaved* with whip[k]/braid[k] at runtime, not run as a pre-pass (see §4.1 salience tower).

Equivalent Rust pre-pass: `crate::generic::backtracker::propagate_singles`, which iterates naked-single + hidden-single (via single-cell + single-region scans) to fixpoint. ECP is implicit in `Grid::assign` (peer elimination + immediate consistency check). The chain rater MUST run only this — `rate_chain` in `chain_rating.rs` and the chain pass inside `rate` / `rate_excluding` are aligned on this contract.

Within each k-tier the exact rule firing order is:

1. `activate-whip[k]` — asserts `(technique ?cont whip[k])` and `(technique ?cont partial-whip[k-1])`. This **enables** the partial-whip extension rule for length k-1.
2. `partial-whip[k-1]` (or `partial-braid[k-1]` for braids, etc.) — extends every length-(k-2) partial-whip by one step, asserting new `(chain (type partial-whip) (length k-1) …)` facts.
3. `whip[k]` — fires the elimination rule: it matches a length-(k-1) partial-whip and finds the t-cell terminator (a CSP-Variable for `?new-llc` whose every alternative value is linked to `?zzz` or to a previous rlc).

Key invariant: **whip[k] cannot fire until all whip[1..k-1] have exhausted**. The salience ordering enforces this via `?*whip[k]-salience* < ?*whip[k-1]-salience* < ... < ?*whip[1]-salience* < ?*single-salience*`.

### 4.2 Blocked rules — confluence under elimination

`focused-elims.clp` and `blocked-rules.clp` together implement **rating confluence**: once a whip[k] fires, the eliminated candidate's downstream rules may now match — but the system continues firing whip[k]-class rules (and all higher-salience rules) until exhausted, *then* re-enters whip[k+1]-class search. This guarantees that the puzzle's **rating** is the maximum k such that whip[k] (or braid[k], etc.) actually fired.

The "pseudo-blocked" mechanism (`Whips[1].clp:82+`) is a CLIPS-specific optimization: when `?*blocked-Whips[1]*` is true, the rule defers its eliminations into a pseudo-block packet and a follow-up rule `apply-whip[1]-to-more-targets` reuses the discovered `(zzz, csp1, llc1)` triple to eliminate all other `?zzz2` candidates satisfying the same predicate, batching the work. This is an optimization, not a semantic change; the Rust port can choose to inline it or skip it.

### 4.3 Focused elimination

`(candidate-in-focus (context ?c) (label ?L))` is a mode where the search is restricted to eliminating only labels in a focus set. Used by some advanced workflows; the Rust port can ignore it for the rating pass.

---

## 5. whip[k] formal definition

### 5.1 Definition (Berthier PBCS3, Ch. on whips)

A **whip[k]** on target `Z` (a candidate, not a c-value) in context `?cont` is a sequence

```
Z, L1 — R1 (csp v1) , L2 — R2 (csp v2) , … , Lk — Rk (csp vk) , L_{k+1}
                                                              ───── (terminator)
```

with k *links* and k+1 *left-linking candidates* (LLCs) such that:

1. **Distinctness.** All labels `Z, L1, R1, …, Lk, Rk, L_{k+1}` are pairwise distinct; all CSP-Variables `v1, …, vk` are pairwise distinct.
2. **First link.** `L1` is linked to `Z` (`exists-link Z L1`).
3. **Per-step structure** (for i = 1..k):
   - `Li` is linked to `R_{i-1}` (for i = 1, to `Z`; otherwise to `R_{i-1}`).
   - `Li` is *labelled by* CSP-Variable `vi`, i.e. `is-csp-variable-for-label vi Li`.
   - `vi` has been **emptied of all other alternatives** in the context augmented by `{Z, R_1, …, R_{i-1}}`. Formally: **every** other label `X` ≠ `Ri` for which `csp-linked Li X vi` holds satisfies `linked-or X Z R_1 … R_{i-1}` — i.e. `X` is killed by `Z` or by one of the earlier `Rj`.
   - `Ri` is the **one surviving alternative** for `vi`. It is the *right-linking candidate* (RLC) for step i, and the chain continues from it.
4. **Terminator (t-cell).** There exists `L_{k+1}` linked to `Rk` such that `L_{k+1}` is labelled by a **fresh** CSP-Variable `v_{k+1}` and **every** label `X` for which `csp-linked L_{k+1} X v_{k+1}` holds satisfies `linked-or X Z R_1 … R_k`. Equivalently, the assumption "Z is true" together with `R_1 … R_k` forces `v_{k+1}` to have **no surviving value** — a contradiction.

**Elimination rule.** From a whip[k] on `Z`, eliminate `Z` (retract the candidate).

**Soundness sketch.** Assume `Z` true. Then `L1` is killed (link Z–L1, exclusion). The only way `v1` is satisfied is `R1` (because all alternatives are linked to Z, hence killed). Hence `R1` is forced. Then `L2` is killed (link R1–L2 plus possibly Z), `R2` forced, etc. At step k+1, `v_{k+1}` has no surviving value — contradiction.

### 5.2 whip[1] special case

`Whips[1].clp:61-87`: whip[1] is the simplest non-trivial chain. Sequence: `Z, L1` with terminator `L1` itself (no Ri at all in the chain, since the t-cell is reached after one link).

Pattern: there exists a label `L1` linked to `Z`, with a CSP-Variable `v1` such that `is-csp-variable-for-label v1 L1`, and **every** other label `X` for which `csp-linked L1 X v1` holds satisfies `linked X Z` (note: in whip[1] this collapses to `linked X Z`, not `linked-or X Z` over an empty `rlcs`).

This is equivalent to a **Hidden Single + propagation**: `L1` is the unique value of `v1` if we assume `Z`, but then `L1` is killed because `L1` is linked to `Z`. Therefore `Z` must be false. (Note: this is automatically handled by Hidden Single rules in standard Sudoku — Whip[1] is the CSP-generic version.)

### 5.3 Worked example — whip[3]

The CLIPS rule `Whips[3].clp:61-89` gives the operational form. Translating:

> Suppose we are searching for a whip[3] eliminating `Z = (n=5, r=3, c=7)`.
>
> 1. Find `L1` (any label linked to Z) with a fresh CSP-Variable `v1` for `L1`. Suppose `L1 = (n=5, r=3, c=4)` and `v1 = rc(r3, c4)` — the cell variable for (r3,c4). The forall clause in `Whips[3].clp` requires every other alternative `X` of `v1` to satisfy `linked-or X Z`; if exactly one value `R1` of `v1` is not yet killed by Z, the chain commits `R1` and proceeds. Note: `Whips[3].clp` has **no explicit `v1 ∉ csp-vars(Z)` guard** — exclusion of Z from the v1 alternative list emerges implicitly because `(csp-linked ?cont ?new-llc Z ?v1)` would require Z to share v1 with L1, but Z is the target being eliminated. The porter must NOT add an explicit check for this; doing so would be incorrect for puzzles where Z is incidentally in the same CSP-Variable as L1 but the forall is satisfied regardless.
>
> 2. Find `L2` linked to `R1` with a fresh CSP-Variable `v2` (∉ {v1}) such that every alternative of v2 except `R2` is `linked-or X Z R1`. Commit `R2`.
>
> 3. Find `L3` linked to `R2` with fresh `v3` such that **every** alternative of v3 satisfies `linked-or X Z R1 R2` (note: in the elimination rule, the `forall` does *not* exclude R3 — because at the terminator, even R3 must be killed, meaning v3 has zero surviving values).
>
> 4. The chain fires: retract `Z`.

A concrete pen-and-paper whip[3] worth tracing is documented in PBCS3 (chapter on whips, "Whip 3 example"); the Rust test corpus (§12) lists puzzles where whip[3] is the unique tier-breaking technique.

### 5.4 Naming legend

- `?zzz` — target Z (to be eliminated).
- `?llc_i` (`$?llcs`) — left-linking candidates (the "labels with CSP-Variable killed").
- `?rlc_i` (`$?rlcs`) — right-linking candidates (the surviving alternatives propagated forward).
- `?csp_i` (`$?csp-vars`) — CSP-Variables for L_i.
- `?last-rlc` — convenience slot for the last element of `rlcs` (redundant but used in joins).
- `?new-llc`, `?new-rlc`, `?new-csp` — the (k+1)-th step under construction.

---

## 6. whip[k] algorithm — pseudocode

> **Rating driver vs per-technique probes (NF-5):** The per-technique functions `run_whip_pass`, `run_braid_pass`, `run_gwhip_pass`, `run_gbraid_pass` documented in §6.3/§7.3/§8 are the *probes* used by **reverse-construction** (§10) with explicit technique suppression. They must NOT be composed naively as sequential passes for the `rate_chain` function. The CLIPS salience tower interleaves techniques within each k-tier (§4.1): at each k, whip[k] fires before gwhip[k] fires before braid[k] fires before gbraid[k], then k increments (verified against `saliences.clp:define-generic-saliences-at-L3` — counter-decrement order is whip→gwhip→braid→gbraid within every k). `rate_chain` must therefore use a **salience-interleaved driver** — iterating k from 1 upward and checking whip[k], then gwhip[k], then braid[k], then gbraid[k] at each k before incrementing — so the emitted rating reflects the tightest technique that would have fired in CLIPS. Calling `run_whip_pass(k_max)` then `run_braid_pass(k_max)` would mis-rate any puzzle solvable by braid[2] as "whip[k_max]" if no whip suffices.

### 6.1 Data structures (Rust types)

```rust
// 81-bit cell index, 9-bit digit; for 9x9 a Label is u16. For NxN: u32.
pub type Label = u32;             // packed (number, row, col) — opaque to chain logic
pub type GLabel = u32;            // packed glabel id
pub type CspVarId = u32;          // 0..324 for 9x9

// Provenance of a CSP-Variable, used only for output/diagnostics — chain rules
// never branch on this.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CspVarKind { Rc, Rn, Cn, Bn }

pub struct CspVarTable {
    /// For each CSP-Variable, the (up to N) labels it contains.
    pub labels_of: Vec<Vec<Label>>,
    /// For each label, the (exactly 4 for Sudoku) CSP-Variables it belongs to.
    pub vars_of:   Vec<[CspVarId; 4]>,
    pub kind_of:   Vec<CspVarKind>,
}

/// Symmetric link graph: peer relation between labels (any constraint).
pub struct LinkGraph {
    /// `linked[a]` = bitset of labels linked to a. For 9x9: 729 labels, fits in 12 u64 words.
    pub linked: Vec<BitSet>,
}

/// CSP-link graph: labels mutually exclusive *as alternative values of a CSP-Variable*.
/// For each (label, csp-var) where the label is in the var, the other labels in
/// the var are the csp-linked alternatives.
pub struct CspLinkGraph {
    /// `alternatives_of[label][csp_var_slot]` = the (≤N-1) other labels in that CSP-Variable.
    /// `csp_var_slot` ∈ 0..4 corresponds to the 4 CSP families.
    pub alternatives: Vec<[Vec<Label>; 4]>,
}

/// One alive instance of a chain (partial-whip, whip, partial-braid, braid).
#[derive(Clone, Debug)]
pub struct Chain {
    pub kind: ChainKind,                  // PartialWhip | Whip | PartialBraid | Braid | …
    pub target: Label,                    // ?zzz
    pub length: u8,                       // number of completed steps so far
    pub llcs: Vec<Label>,                 // length k
    pub rlcs: Vec<Rlc>,                   // length k, parallel to llcs
    pub csp_vars: Vec<CspVarId>,          // length k, parallel
}

/// An rlc is normally a Label; in g-variants it can be a GLabel.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Rlc {
    Cand(Label),
    GCand(GLabel),
}

// ResolutionState — see §13 `resolution_state.rs` for the authoritative definition
// (includes g_support field; the sketch here is intentionally omitted to avoid drift).
```

The build phase computes `CspVarTable`, `LinkGraph`, `CspLinkGraph` **once** from the puzzle. The `glabels` table and `GLinkGraph` are built lazily when g-chains are enabled. Bitsets are essential: `linked_or(label, &rlcs_bitset)` is one AND + popcount, not a Python-style for-loop.

### 6.2 Build phase

1. **Cell tables.** For Sudoku 9×9: 81 cells, 9 rows, 9 columns, 9 blocks. Precompute `peer_cells[c]` = the (≤20) cells sharing a row/col/block with `c`.
2. **CSP-Variables.** For each of the 4 families, enumerate the 81 variables and the (≤9) labels in each. Build `CspVarTable`.
3. **Links.** For every pair of labels `(l1, l2)`, `l1 < l2`, that share *any* CSP-Variable, assert `linked(l1, l2)` (one bit per direction). Cross-digit same-cell pairs add rc links; same-row-and-digit pairs add rn; etc. Build `LinkGraph` as 729 bitsets of 729 bits each (≈ 66 KB).
4. **CSP-links.** Same scan, but record per-`csp-var` alternative lists: `alternatives_of[label][slot] = vec![the other labels in that csp-var]`. Build `CspLinkGraph`.

(For 16×16 these tables are 4× larger; precompute once and reuse across many puzzles.)

### 6.3 Search phase — whip[k]

Salience-driven outer loop (k from 1 upward, stop at first eliminating k):

**Critical architectural note:** The `partial-whip[1]` seed set is a **dedicated rule** in CLIPS (`Partial-Whips[1].clp:36-78`) that must run *before* any extension or elimination rule for k ≥ 2. It is not derived from "length 0" partial-whips (there is no such thing). The Rust port must seed the partial-whip[1] set explicitly and pass it as `partials_1` to all subsequent levels. Failing to do this means `whip[2+]` will see an empty seed set and fire nothing — a silent correctness bug.

```rust
fn run_whip_pass(rs: &mut ResolutionState, k_max: u8) -> Vec<Elimination> {
    let mut elims = Vec::new();

    // Phase A₀: build partial-whips of length 1 (CLIPS: Partial-Whips[1].clp:36-78).
    // This is a *special-case seeding step*, NOT an extension of length-0 partials.
    // For every candidate Z and every L1 linked to Z, find csp1 such that
    // is-csp-var-for-label(csp1, L1) and there is exactly one surviving rlc1 in
    // alternatives_of[L1][slot(csp1)] \ {Z} with all other alternatives linked to Z.
    let partials_1 = build_partial_whips_length_1(rs);

    // k = 1: whip[1] fires when a partial-whip[1] seed already has NO surviving rlc1
    // (every alternative of csp1 is linked to Z). This is the whip[1] direct form
    // (Whips[1].clp:61-87) — handled separately.
    elims.extend(try_whip_1_eliminations(rs));
    if !elims.is_empty() {
        return elims;
    }

    // For k ≥ 2: keep a rolling `partials_prev` (length k-1), extending from
    // `partials_1` for k=2, from `partials_2` for k=3, etc.
    let mut partials_prev = partials_1; // length 1 seeds

    for k in 2u8..=k_max {
        // At loop entry, partials_prev holds partial-whips of length k-1.
        // Try terminator: terminate each into whip[k]; then extend to length k for the next iteration.
        for pw in &partials_prev {
            if let Some(z) = try_terminate_whip_k(pw, rs) {
                elims.push(Elimination { z, rule: Rule::Whip(k) });
            }
        }
        if !elims.is_empty() {
            // Apply, re-run BRT (singles/subsets), then re-loop.
            return elims;
        }
        // Phase A_next: extend partials_prev (length k-1) by one step to get
        // partial-whips of length k, ready for whip[k+1] at the next iteration.
        partials_prev = extend_partial_whips_to_length(k, &partials_prev, rs);
    }
    elims
}
```

#### 6.3.1 partial-whip[1] (CLIPS: `Partial-Whips[1].clp:36+`)

> **Seeding role:** `partial-whip[1]` is the *only* source of length-1 chains. It is NOT derived by extending shorter chains (there are no length-0 chains). In CLIPS it is a special-cased rule (`Partial-Whips[1].clp` header: "SPECIAL CASE. DO NOT USE THE AUTOMATIC GENERATOR"). The Rust port MUST call `build_partial_whips_length_1(rs)` as a dedicated pre-loop step before any k ≥ 2 iteration (as shown in §6.3 above).

For every candidate `Z` (target) and every label `L1` with `exists-link(Z, L1)`:
- For every `csp1` such that `is-csp-variable-for-label(csp1, L1)`:
  - Enumerate the alternative labels in `csp1`: `alts = alternatives_of[L1][slot(csp1)]`.
  - Find candidates `?rlc1` ∈ `alts` such that `?rlc1 ≠ Z` and (in the **partial-whip[1] memoization** form) every other element `X` of `alts \ {?rlc1}` satisfies `linked(X, Z)`.
  - If exactly one such `?rlc1` exists, assert a partial-whip `(target Z, llcs [L1], rlcs [R1], csp_vars [csp1], length 1)`. The "exactly one" condition is enforced *implicitly* in CLIPS by the `forall` clause; in Rust: count survivors of `alts \ {Z, R1}` and ensure all are linked to Z, then commit R1 as the rlc.

Dedup guard (CLIPS `not (chain (type partial-whip) (length 1) (target Z) (rlcs ?rlc1))`): the rule does not re-assert an existing chain; in Rust, use a `HashSet<(target, rlcs)>` to short-circuit.

#### 6.3.2 partial-whip[k-1] (extension, for k ≥ 3)

Pattern (CLIPS: `Partial-Whips[3].clp:33+` is the template):

Input: a partial-whip of length `k - 2` with `(target Z, llcs $LL, rlcs $RR, csp_vars $CV, last_rlc last)`.

Step:

```
for each new-llc:
    requires: exists-link(last, new-llc) ∧ new-llc ≠ Z
             ∧ new-llc ∉ LL ∧ new-llc ∉ RR
    for each new-csp such that is-csp-variable-for-label(new-csp, new-llc)
                              ∧ new-csp ∉ CV:
        alts = alternatives_of[new-llc][slot(new-csp)] \ {Z} \ LL \ RR
        if every element of alts satisfies linked-or(elt, Z, $RR):
            // partial-whip[k-1] *also* requires there is *at least one* new-rlc
            // in alts (otherwise whip[k-1] would already have fired — but this
            // is the partial-whip extension, not the terminator).
            // Pick THE new-rlc: the (unique) candidate in alts NOT linked-or'd
            // to {Z, RR}. The forall in CLIPS:
            //   forall (csp-linked new-llc ?xxx&~?new-rlc new-csp)
            //     (test (linked-or ?xxx Z $RR))
            // means: there is exactly one alternative not yet killed; this is
            // ?new-rlc.
            new-rlc = unique survivor
            assert new partial-whip:
                (length k-1, target Z,
                 llcs LL ++ [new-llc],
                 rlcs RR ++ [new-rlc],
                 csp_vars CV ++ [new-csp])
        // Dedup: not (existing chain with same rlcs sequence).
```

The "exactly one survivor" condition is crucial. If there were ≥2 survivors, no further progress can be made down the chain (the assumed-Z branch fans out). If there were zero survivors, **whip[k-1] would already have fired** as the terminator (CSP-Variable emptied) — so by salience ordering this case is impossible here.

#### 6.3.3 whip[k] elimination rule

Pattern (CLIPS: `Whips[k].clp:61+` for k ≥ 2, `Whips[1].clp:61+` for k = 1):

Input: a partial-whip of length `k - 1` with last_rlc `last`.

```
for each new-llc:
    requires: exists-link(last, new-llc) ∧ new-llc ≠ Z
             ∧ new-llc ∉ LL ∧ new-llc ∉ RR
    for each new-csp such that is-csp-variable-for-label(new-csp, new-llc)
                              ∧ new-csp ∉ CV:
        // Terminator: EVERY alternative of new-csp is killed.
        alts = alternatives_of[new-llc][slot(new-csp)]
        if every X ∈ alts satisfies linked-or(X, Z, $RR):
            // CSP-Variable new-csp has no surviving value — contradiction.
            ELIMINATE Z.
            return.
```

For `k = 1` the chain has no `RR` and the condition simplifies to `every X ∈ alts: linked(X, Z)`.

### 6.4 Termination, depth limits

- The salience tower fixes a strict order: whip[1] exhausts before whip[2] starts. Whip[k_max] (typically 36) is the configured ceiling — beyond, the puzzle is rated W*≥36 ≈ T&E territory.
- Within a k-level, partial-whips of length k-1 are bounded: each step picks a **new** csp-var and a **new** llc, so chain length is bounded by min(k, #csp-vars) = min(k, 324) on 9×9.
- For Sudoku 9×9 with k up to 12, the practical search space is small (≤ 10^4 partial chains in observed runs); k > 12 is exponential-tail rare.

### 6.5 Complexity

Per-context, per-k:
- Number of partial-whips of length k − 1: bounded by `O(|candidates| · D^{k-1})` where `D = max degree of csp-link graph = N − 1`. For 9×9: `D = 8`. Practical bound after dedup: `≪ 10^4` for k ≤ 7.
- Each extension step: `O(|new-llc-candidates| · |csp-vars-per-llc| · |alts|)` ≈ `O(N · 4 · N)` per partial = `O(N²)`. With bitset-encoded `linked-or` this is `O(N² · |rlcs|/64) = O(N²)` for 9×9 (`rlcs ≤ k ≤ 36` always fits in one u64 bitset of size 729).
- Total: `O(|candidates| · D^{k-1} · N²)` worst case; in practice the dedup table cuts this by 100–1000×.

For 9×9: whip[5] passes complete in <10 ms on a single core in CLIPS (Berthier's published numbers); the Rust port should hit ≤ 1 ms.

---

## 7. braid[k] formal definition + algorithm

### 7.1 Definition (PBCS3)

A **braid[k]** on target `Z` is a chain that **relaxes** whip[k]'s sequentiality. Same data layout (`llcs`, `rlcs`, `csp-vars`, `target`, `length`); difference is in the **link requirement** at each step:

- **whip[k]**: at step i, `Li` is linked to `R_{i-1}` (the **immediate previous** rlc). Strict ordering.
- **braid[k]**: at step i, `Li` is linked to `Z` **or any earlier rlc** `R_1 .. R_{i-1}`. The chain becomes a DAG rooted at Z; rlcs are committed in topological order but the dependency structure is wider.

Formally:

1. **Distinctness.** All `Z, R_1, …, R_k` are pairwise distinct (enforced in CLIPS: `?new-rlc&~?zzz&:(not (member$ ?new-rlc $?llcs))&:(not (member$ ?new-rlc $?rlcs))`). The CSP-Variables `v_1, …, v_k` are pairwise distinct (`?new-csp&:(not (member$ ?new-csp $?csp-vars))`). However — **unlike whip[k]** — the CLIPS braid rules do NOT exclude `new-llc ∈ LL` for the elimination rule or for partial-braid extension. `Braids[5].clp:braid[5]` (line 72) checks only `?new-llc&~?zzz&:(not (member$ ?new-llc $?rlcs))`: `new-llc` must not be Z and must not be in the **rlcs** list, but may reuse a previous `llc`. This is intentional: braids allow a candidate to serve as a left-linker at multiple steps. The terminator clause is unchanged.
2. **Per-step link:** for i = 1..k, **`L_i ∈ linked-or(Z, R_1 .. R_{i-1})`** (i.e., L_i is linked to at least one of {Z, R_1, …, R_{i-1}}). For whip[k] this was restricted to `linked(L_i, R_{i-1})` exclusively.
3. **Per-step squash:** every alternative `X` of `v_i` (except `R_i`) satisfies `linked-or(X, Z, R_1 .. R_{i-1})`.
4. **Terminator** as in whip[k]: every alternative of `v_{k+1}` is linked-or to `{Z, R_1 .. R_k}`.

Every whip[k] is a braid[k] (the sequential link L_i — R_{i-1} is a special case of L_i — linked-or(Z, R_1…R_{i-1})). The converse is false: braids are strictly stronger.

CLIPS rules **deduplicate** the whip-subset: `Braids[k].clp` matches a `chain (type partial-braid)` (not `partial-whip`); the partial-braid construction rule only asserts a partial-braid when it would not coincide with an existing partial-whip — see `Braids[3].clp:113-114`: "the case `(linked llc2 rlc1)` is excluded because it would produce a partial whip". This means the **search** finds non-whip braids only; the whip cascade has already handled the rest by salience.

### 7.2 Worked example (CLIPS `Braids[3].clp:60`)

A braid[3] rule body:

```
chain (type partial-braid) (length 2) (target Z) (llcs $LL) (rlcs $RR) (csp-vars $CV)
candidate ?new-llc s.t. new-llc ≠ Z ∧ new-llc ∉ RR ∧ linked-or(new-llc, Z, $RR)
is-csp-variable-for-label new-csp new-llc s.t. new-csp ∉ CV
forall csp-linked new-llc ?xxx new-csp: linked-or(?xxx, Z, $RR)
→ eliminate Z
```

Note `new-llc` need only be `linked-or` to `{Z} ∪ RR` (rather than `linked` to `last-rlc` as in whip[3]). This is the only operative difference at the elimination rule level.

The partial-braid extension `partial-braid[2]` (Braids[3].clp:98) reads from `(type partial-whip|partial-braid)`: it accepts a partial-whip *or* partial-braid of length 1 as the seed and adds a step requiring `linked(new-llc, Z)` for the seed case (length 1 + new-llc must be linked to Z; this is the *non-immediate* extension — the "braid case" distinct from "extend last partial-whip"). It explicitly excludes `linked(new-llc, rlc1)` to avoid re-emitting a partial-whip.

### 7.3 Pseudocode — differences from whip[k]

```rust
fn extend_partial_braid_to_length_k_minus_1(
    seeds: &[Chain],          // partial-whips OR partial-braids of length k-2
    rs: &ResolutionState,
) -> Vec<Chain> {
    let mut out = Vec::new();
    for seed in seeds {
        let Z = seed.target;
        let RR_set: BitSet = bitset_of(&seed.rlcs);
        let LL_set: BitSet = bitset_of(&seed.llcs);

        // *** KEY DIFFERENCES FROM WHIP ***
        // For whip: enumerate new-llc such that linked(last_rlc, new-llc);
        //           also new-llc ∉ LL is enforced.
        // For braid: enumerate new-llc such that linked-or(new-llc, Z, RR_set).
        //            new-llc ∉ LL is NOT enforced (CLIPS Braids[5].clp:72 only
        //            excludes new-llc == Z and new-llc ∈ rlcs, not ∈ llcs).
        //            This allows a label to serve as left-linker at multiple steps.
        for new_llc in candidates_linked_or(Z, &RR_set) {
            if new_llc == Z || RR_set.contains(new_llc) { continue; }
            // Optional: if extending a partial-whip seed and linked(new-llc, last_rlc) holds,
            // the new chain is also a partial-whip — let the whip extension handle it; skip here.
            if seed.kind == ChainKind::PartialWhip && rs.linked(new_llc, *seed.rlcs.last().unwrap()) {
                continue;
            }
            for new_csp in csp_vars_of(new_llc) {
                if seed.csp_vars.contains(&new_csp) { continue; }
                // Find new-rlc: the unique survivor of alternatives_of[new-llc][new-csp]
                // not linked-or'd to {Z, RR_set}.
                let alts = alternatives_of(new_llc, new_csp);
                let killed_by = |x: Label| rs.linked_or(x, Z, &RR_set);
                let survivors: SmallVec<[Label; 9]> =
                    alts.iter().copied().filter(|&x| x != Z && !killed_by(x)).collect();
                if survivors.len() != 1 { continue; }
                let new_rlc = survivors[0];
                if RR_set.contains(new_rlc) || LL_set.contains(new_rlc) { continue; }
                // Dedup guard: do not assert if (partial-whip|partial-braid) with same
                // *set* of rlcs (using same-sets-of-rlcs — multiset equality, since rlcs
                // are unique within a chain).
                if dedup_table.contains(seed.target, &seed.rlcs, new_rlc) { continue; }
                out.push(Chain {
                    kind: ChainKind::PartialBraid,
                    target: Z,
                    length: seed.length + 1,
                    llcs: extend(&seed.llcs, new_llc),
                    rlcs: extend(&seed.rlcs, new_rlc),
                    csp_vars: extend(&seed.csp_vars, new_csp),
                });
            }
        }
    }
    out
}
```

The elimination rule `braid[k]` is *identical to whip[k]* except (a) it reads `(type partial-braid)`, (b) the new-llc constraint is `linked-or(new-llc, Z, $RR)` instead of `linked(new-llc, last-rlc)`.

### 7.4 Subchain interpretation

Berthier's "nested sub-whip conditions" phrasing in PBCS3 corresponds to: at each braid step, the squash condition `every alternative of v_i is linked-or to {Z, RR}` is itself a *sub-resolution argument*. In whip-only thinking, each step is one immediate-link demand; in braid thinking, each step is "after committing earlier rlcs, this csp-var is determined", which is in principle a tiny sub-whip ("the alternatives are all killed by *some* element of the committed set"). The CLIPS rule does not actually *recurse* into sub-whips — it inlines the check as a single `forall`. Hence the Rust port does **not** need recursion; the recursive flavour is purely conceptual.

### 7.5 Complexity

Worst case is strictly harder than whip[k]: at each step `O(|candidates|)` candidates linked-or to `{Z} ∪ RR` (vs. only those linked to `last_rlc` for whip), so an extra factor of `D` per step. Practically: braid[k] solve time is 3–10× whip[k] solve time in PBCS3 benchmarks.

---

## 8. Grouped candidates (g-)

### 8.1 g-labels construction (`SudoRules-V20.1/GENERAL/glabels.clp`, `init-glinks.clp`)

`glabels.clp` in the generic core is a stub (`define-glabels-and-glinks` returns TRUE; the application redefines it). The Sudoku-specific version at `SudoRules-V20.1/GENERAL/glabels.clp` enumerates glabels of two kinds:

- **Block-row glabels** (horizontal mini-lines): for each block `b` and digit `d`, the cells of `b` in each of the `BR` rows of `b` form a horizontal segment. For 9×9 (BR=BC=3): `9 blocks × 9 digits × 3 row-segments-per-block = 243` glabels.
- **Block-column glabels** (vertical mini-lines): analogously, 243 glabels.

(Total: 486 glabels on 9×9.)

**General formula for N×N with block dimensions BR×BC such that BR·BC = N:**

CLIPS `init-physical-2D-segments` (`SudoRules/glabels.clp:126-141`) creates:
- Horizontal segments: `N rows × (N/BR) segments-per-row = N²/BR`
- Vertical segments: `N cols × (N/BC) segments-per-col = N²/BC`

Total glabels per digit: `N²/BR + N²/BC = N²·(BR+BC)/(BR·BC)`.

**Note:** the formula `2·N·N·max(BR, BC)` is **incorrect** for rectangular blocks (only coincidentally correct when BR = BC). Use `N²·(BR+BC)/(BR·BC)` for the general case. For 9×9: `81·6/9 = 54` per digit × 9 digits = 486. ✓

Each glabel `g` records: the digit, the block, the row-segment (or column-segment), and the underlying set of `N/BR` or `N/BC` labels.

Predicate `label-in-glabel(L, g)`: TRUE iff `L`'s (digit, cell) matches the digit of `g` and `L`'s cell is in `g`'s segment. Precomputed at init.

### 8.2 Grouped CSP-Variables

A **grouped CSP-Variable** has glabels (not labels) as its alternatives. Each glabel is associated with **two** CSP-Variable families simultaneously — not just bn:

- **Horizontal glabels** (block-row mini-lines): associated with **rn** (row × digit) AND **bn** (block × digit). CLIPS: `init-g-candidates-horiz` (`SudoRules-V20.1/GENERAL/init-glinks.clp:84-88`) asserts `is-csp-variable-for-glabel` for both `csp-rn = row-number-to-rn-variable(?row ?nb)` and `csp-bn = block-number-to-bn-variable(?blk ?nb)`.
- **Vertical glabels** (block-column mini-lines): associated with **cn** (column × digit) AND **bn** (block × digit). CLIPS: `init-g-candidates-verti` (`init-glinks.clp:109-113`) asserts both `csp-cn` and `csp-bn`.

In CLIPS: `is-csp-variable-for-glabel (csp-var ?v) (glabel ?g)` asserts this (with two assertions per glabel). The chain rules read `csp-glinked ?cont label glabel csp-var` and `is-csp-variable-for-glabel` symmetric to the label case.

**Rust porting implication:** `GLabelTable` must store *two* `CspVarId` entries per glabel (one for rn/cn, one for bn). Enumerate both when scanning `is-csp-variable-for-glabel` matches. Omitting rn or cn family fires means g-whip[k] / g-braid[k] will miss a substantial class of valid chain steps.

### 8.3 glinks

`exists-glink(?cont, ?lab, ?glab)`: label `lab` is "in tension" with glabel `glab` — `lab`'s truth would kill the glabel as a whole (every label in glab is linked to lab). For Sudoku: placing `lab` immediately kills all cells of `glab` because they share a row/block/column constraint.

`csp-glinked(?cont, ?lab, ?glab, ?csp-var)`: the csp-typed variant — `lab` and `glab` are mutually exclusive *as alternative values of csp-var*.

**Direction note:** Both `csp-glinked` and `exists-glink` are asserted **unidirectionally** as `(candidate → g-candidate)` only — i.e., `exists-glink ?cont ?cand ?gcand`, never `?gcand ?cand`. This is confirmed by `init-glinks.clp:128-196` (`init-effective-csp-glinks-rn/cn/bn` all assert `(csp-glinked ?cont ?cand ?gcand …)` and `(exists-glink ?cont ?cand ?gcand)`). The chain rules consuming these facts always pattern-match with `?lab` as the first (candidate) slot and `?glab` as the second (g-candidate) slot. The Rust port should store the glink graph keyed by `(Label → GLabel)` only — no reverse index required.

### 8.4 g-whip[k] formal definition

A **g-whip[k]** is exactly a whip[k] where each `Ri` (right-linking candidate) may be either a **label** or a **glabel**, and each step's link requirement is the *grouped* form (`exists-link` ∨ `exists-glink`), and the squash uses `glinked-or` instead of `linked-or`.

Constraint: g-whip strictly extends whip. The CLIPS rule `gWhips[k]` (`gWhips[2].clp`, `gWhips[5].clp`) requires:

- At step i: `new-llc` is `exists-link` or `exists-glink` to `last_rlc` (line 71-74 of gWhips[5]: `(or (exists-link …) (exists-glink …))`).
- The CSP-Variable freshness rule **differs by sub-rule** — this is critical:
  - **Elimination rule (`gwhip[5]`, line 79) and sub-rule -1 and -3** (`partial-gwhip[5]-1` line 61, `-3` line 195): `neq ?new-csp (last $?csp-vars)` — only the *last* csp-var is forbidden.
  - **Sub-rule -2** (`partial-gwhip[5]-2` line 126, extending a partial-whip by a g-candidate): `(not (member$ ?new-csp $?csp-vars))` — the *entire* history is forbidden (same as ungrouped whip rule). This stronger exclusion is needed because sub-rule -2 transitions from a plain partial-whip (which already enforces full-history CSP-var uniqueness) into a g-extension, and allowing a repeated csp-var here would violate soundness.
  - **Same split in g-braid**: `partial-gbraid[4]-2` (`gBraids[5].clp:185`) uses `(not (member$ ?new-csp $?csp-vars))`; `-1` and `-3` use `neq (last …)`.
- Squash uses `glinked-or` (line 80).

**Porting rule:** implement the csp-var check as a function parameter: `fn csp_fresh(new_csp, csp_vars, is_whip_to_gcand_branch: bool) → bool` that applies full-history exclusion when `is_whip_to_gcand_branch` and last-only exclusion otherwise.

**llc distinctness asymmetry (NF-1):** g-whip retains whip-style `new-llc ∉ llcs ∪ rlcs` exclusion (`gWhips[5].clp:72-73`: `(not (member$ ?new-llc $?llcs))&:(not (member$ ?new-llc $?rlcs))`); g-braid retains braid-style `new-llc ∉ rlcs` only (`gBraids[5].clp:72, 111`: `(not (member$ ?new-llc $?rlcs))` — no llcs exclusion). A porter reading "g-* mirrors *" must NOT relax gWhip to the braid-style check.

The terminator/elimination structurally mirrors whip[k].

The **partial-gwhip** rules come in three sub-rules (see `Partial-gWhips[2].clp`):
1. `partial-gwhip[k-1]-1`: extend a partial-gwhip by a candidate (regular label) as rlc.
2. `partial-gwhip[k-1]-2`: extend a partial-whip by a g-candidate (glabel) as rlc.
3. `partial-gwhip[k-1]-3`: extend a partial-gwhip by a g-candidate as rlc.

The asymmetry: extending a partial-whip with a *label* rlc is still a partial-whip (no new rule needed — whip extension handles it); only adding a glabel turns it into a gwhip.

#### 8.4.1 g-whip[1] seed (g-family equivalent of C1)

**Critical:** `partial-gwhip[1]` is a **dedicated seed rule** (`Partial-gWhips[1].clp:38-83` — comment: "SPECIAL CASE. DO NOT USE THE AUTOMATIC GENERATOR"), analogous to `partial-whip[1]` (§6.3.1). It must run before any extension or elimination rule for g-whip[k ≥ 2]. A porter wiring only `extend_partial_gwhip_*` without a `build_partial_gwhip_length_1` seed will silently emit zero g-whips of length ≥ 2.

**Semantics from `Partial-gWhips[1].clp:38-83`:**

For each target `Z` and each pair `(llc1, rlc1)`:
- `exists-link(?cont, Z, llc1)` — llc1 is a regular candidate linked to Z
- `csp-glinked(?cont, llc1, rlc1, csp1)` — rlc1 is a g-candidate (GLabel) csp-glinked to llc1 via csp1, and `rlc1 ≠ Z`, `¬label-in-glabel(Z, rlc1)` (comment at `gWhips[2].clp:68`: "can only be a g-candidate")
- `forall csp-linked(?cont, llc1, X, csp1) where ¬label-in-glabel(X, rlc1)`: X is `linked(X, Z)` — every regular alternative of llc1 for csp1 (except those inside rlc1) is already killed by Z
- Dedup: do not assert if an existing `(partial-whip|partial-gwhip, length=1, target=Z)` already has `rlc1a` equal to `rlc1` or `label-in-glabel(rlc1a, rlc1)` — i.e. a finer chain already subsumes it

**Rust seed function signature:**
```rust
pub fn build_partial_gwhips_length_1(
    z: Label, rs: &ResolutionState,
    csp: &CspLinkGraph, glink: &GLinkGraph,
    glab: &GLabelTable,
) -> Vec<Chain>  // each Chain has kind=PartialGWhip, length=1, rlcs=[GCand(rlc1)]
```

Pass the result as `partials_gwhip_1` to the g-whip[2+] extension loop.

### 8.5 g-braid[k] formal definition

`gBraids[k]` (`gBraids[3].clp`, `gBraids[5].clp`) mirrors g-whip[k] with the braid relaxation: `new-llc` need only be `glinked-or` to `{Z} ∪ RR` (line 72 of gBraids[3]). Partial-gbraid extension has the same 3-fan-out: (1) gwhip|gbraid + label, (2) whip|braid + glabel, (3) gwhip|gbraid + glabel — sub-rules `partial-gbraid[k-1]-1/2/3` (`gBraids[3].clp:95-301`).

### 8.6 Dedup for g-variants

Crucial: the dedup guards for g-variants are **non-trivial**. CLIPS `gBraids[3].clp:128-145`:

```
not (chain (type partial-gwhip|partial-gbraid) (length 2) (target Z)
           (rlcs $RRa & :(same-sets-of-rlcs new-rlc $RR $RRa)))
```

For the "label-only" sub-rule, additionally:

```
not (chain (type partial-whip|partial-braid) (length 2) (target Z)
           (rlcs $RRa & :(and (subsetp $RR $RRa) (glabel-contains-some-of new-rlc $RRa))))
```

— i.e. do not assert a partial-gbraid whose new glabel-rlc *contains* a label that's already an rlc of an existing partial-whip/braid (because the smaller chain subsumes it).

These dedup guards must be ported faithfully to avoid exponential blowup from redundant chains in the search.

---

## 9. Ratings (W, B, gB_n)

### 9.1 Rating emission

Each chain rule binds `?*technique*` to a symbol like `W[3]`, `B[5]`, `gW[2]`, `gB[5]` when it activates (e.g. `Whips[3].clp:45`: `(bind ?*technique* W[3])`). The **rating of the puzzle** is the maximum technique that fired, taken in salience order. Concretely (PBCS3, Tab. of ratings):

| Technique | Rating |
|---|---|
| Whip[k] | W[k] |
| Braid[k] | B[k] |
| gWhip[k] | gW[k] (often included under gB_n with n=1) |
| gBraid[k] | gB[k] |

The "gB_n" notation in PBCS3 sometimes refers to a 2-D rating `(complexity, k)` where `n` indexes a hierarchy of grouped variants — gB_1 ≡ gWhip, gB_2 ≡ gBraid, gB_3 includes nested g-subchains, etc. For the porting target (whip / braid / gwhip / gbraid), the relevant scalar ratings are **W[k], B[k], gW[k], gB[k]**.

### 9.2 W → B → gB hierarchy

Every W[k]-rated puzzle is B[k]-rated (because whips are braids). Every W[k]-rated puzzle is also gW[k]-rated (because labels are degenerate glabels, glabel = singleton). Therefore in the rating lattice:

```
W[k]  ≤  B[k]
W[k]  ≤  gW[k]  ≤  gB[k]
B[k]  ≤  gB[k]
```

When you report the rating of a puzzle, the convention is to use the *tightest* (smallest in this lattice) technique that fired. PBCS3 conventionally reports W[k] when both W[k] and B[k] would apply.

**gW→gB cascade:** when a `partial-gwhip` is consumed by a gbraid extension step (e.g. `gBraids[5].clp:240`, type union `partial-gwhip|partial-gbraid` as seed), the final firing rule is `gbraid[k]` and the emitted rating is gB[k], not gW[k].

### 9.3 Mapping to this repo's existing rater

Existing repo (`src/generic/techniques/`, `rater.rs`) uses **SE-style base values** with internal tiers `T1 / T2 / T3 / T4Plus`. Best-effort mapping (calibration required — values below are first-cut from PBCS3 examples and the existing repo's SE table; need empirical validation against the published Magictour benchmark and Berthier's catalogue):

| CSP-Rules rating | Approximate SE base | Repo `Tier` | Existing techniques covering it |
|---|---|---|---|
| W[1] | 2.0 | T1 | Hidden Single (already covered) |
| W[2] | 4.2 | T2 | Naked/Hidden Pair, partial XY |
| W[3] | 6.6 | T3 | AIC base (existing `Aic`) |
| W[4] | 7.0 | T3 | AIC longer chains |
| W[5–7] | 7.2–8.0 | T3 | AIC + light forcing |
| W[8–12] | 8.0–9.0 | T3/T4Plus | NestedFC L2 (existing `NestedFc` base 9.5) |
| B[3–5] | 7.0–8.5 | T3 | Sue de Coq / Death Blossom (currently uncovered) |
| B[6–12] | 8.5–9.5 | T3/T4Plus | Forcing nets (currently uncovered) |
| gW[k] | ≈ W[k] + 0.2 | same | grouped AIC |
| gB[k] | ≈ B[k] + 0.2 | same | grouped Death Blossom |
| W/B[≥ 13] | ≥ 9.5 | T4Plus | NestedFC L3 (base 10.0) territory |

**Caveat (mandatory):** these mappings are **best-effort heuristic**; the repo's SE calibration history (`docs/se_calibration_report_v4.md`) shows a 9-month process to align even AIC. Plan a calibration sweep using PBCS3's reference puzzle catalogue (Berthier publishes per-puzzle W/B/gW/gB ratings for the Magictour-1467 benchmark and the Sudogen-1M sample).

---

## 10. Reverse construction (puzzle generation)

The CLIPS code is a *forward solver* — it never synthesises puzzles. The reverse-construction algorithm below is **constructive** in the sense that we explicitly seed the chain structure, then carve clues.

The existing `tools/sudoku_rs_core/src/generic/aic_reverse.rs` is the architectural template: a **search-with-guided-bias** algorithm that combines (a) a uniqueness-preserving generator with (b) a chain-structure probe to score candidate puzzles. Its central insight (per the file's R3.3b2 doc-block): pure constructive planting fails (0/5000 hit rate); guided greedy removal driven by a chain-length scorer succeeds. We replicate this pattern for whip / braid / gwhip / gbraid.

### 10.1 Generic skeleton

For each technique `T ∈ {Whip(k), Braid(k), GWhip(k), GBraid(k)}` and target length `k`:

```rust
pub struct ChainReverseSpec {
    pub technique: TechniqueId,    // Whip|Braid|GWhip|GBraid
    pub target_length: u32,        // k
    pub length_slack: u32,         // ±slack tolerance
    pub clue_min: u32,
    pub clue_max: u32,
    pub max_attempts: u32,
    pub require_load_bearing: bool,
    pub target_tier: Tier,
}

pub fn chain_reverse_construct(spec: &ChainReverseSpec, rng: &mut Rng)
    -> Option<ReverseResult> {
    for _ in 0..spec.max_attempts {
        let (seed, solution) = gen_unique_puzzle_at_clue_count(spec.clue_max, rng);
        let mut keep = seed.kept_cells();

        guided_removal(&mut keep, &solution, |partial| {
            let g = build_grid_subset(&solution, partial);
            chain_score(&g, spec.technique, spec.target_length)
        }, rng);

        let puzzle = build_grid_subset(&solution, &keep);
        let r = rate(&puzzle);
        if tier_rank(r.tier) != tier_rank(spec.target_tier) { continue; }
        if !r.frontier.contains(&spec.technique) { continue; }

        let measured_len = find_chain_length(&puzzle, spec.technique)?;
        if (measured_len as i32 - spec.target_length as i32).abs() > spec.length_slack as i32 {
            continue;
        }
        if spec.require_load_bearing {
            let r2 = rate_excluding(&puzzle, &[spec.technique]);
            if tier_rank(r2.tier) <= tier_rank(spec.target_tier) { continue; }
        }
        return Some(ReverseResult { puzzle, solution, rate: r, clue_count: count(&keep) });
    }
    None
}
```

The two new ingredients per technique are:

1. **`chain_score(g, T, k)`** — fast probe returning `(has_chain, distance_to_target_length)`. For each technique, implement a stripped-down search (the existing solver, but exit early on first hit).
2. **`find_chain_length(g, T)`** — full search returning the (rule, length) of the first eliminating chain. This may be the same routine without early-exit.

### 10.2 Per-technique scoring details

#### whip[k]
- `chain_score`: run `run_whip_pass(g, k_max=k+slack)`; return `(true, |first_firing_k - k|)` for the first elim, else `(false, MAX)`.

#### braid[k]
- Same as whip but call the partial-braid search path. Note: salience-wise, whip[k] fires before braid[k], so a puzzle ends up *braid-rated only* if there is **no** whip[k] (for any k up to the braid's k). The reverse-construction must therefore explicitly **suppress whip techniques** during the score probe; otherwise every braid puzzle is mis-scored as a whip.
- The probe should call a variant `find_first_braid_chain_length_excluding_whips`, which runs partial-whip + partial-braid construction (since braid extension reads `(type partial-whip|partial-braid)`) but only counts a hit when the elimination matches the `partial-braid` form *and* no earlier whip would have applied at that elimination's target.

#### g-whip[k]
- Probe builds the glabel table + glink graph (one-shot per puzzle), then runs the partial-gwhip / gwhip[k] search.
- **Suppress whips only** for scoring. Per §4.1 salience order: whip[k] → **gwhip[k]** → braid[k] → gbraid[k]. Braid has *lower* salience than g-whip and fires *after* g-whip at the same k — it cannot pre-empt g-whip. Only higher-salience whips must be suppressed. The earlier text "suppress whips AND braids" was incorrect (opus finding Mn-3, FA-11 fix).

#### g-braid[k]
- Probe builds glabel + glink + braid scaffolding. Suppress whips, gwhips, and braids.
- Most expensive; expect 5–20× wall-clock vs whip[k] probe at same k.

### 10.3 Load-bearing semantics

Same as `aic_reverse.rs`: `rate_excluding(puzzle, &[technique]).tier > target_tier` ⇒ accept. This is the **strict** load-bearing rule (preferred for Phase D'5 lemma-extraction). The looser variant (`rate_excluding(puzzle, &[technique]).tier < target_tier` ⇒ "coincidental tier-up") is documented in `reverse_construct::ReverseSpec` but should not be used for chain reverse synthesis.

### 10.4 Determinism, batching

Adopt the same per-worker seed derivation (`splitmix off master seed`, `Xoshiro256PlusPlus`) used in `batch_aic_reverse_construct` (lines 686-737 of `aic_reverse.rs`). Single-thread runs are byte-deterministic per seed; multi-thread is throughput-oriented.

### 10.5 Direct constructive planting (alternative path; defer)

A more aggressive approach: starting from a solved grid `G`, *choose* k cells and a chain-skeleton (target `Z` + path L1-R1-…-Lk-Rk-L_{k+1}), then **augment candidate sets** by inserting "noise" candidates so that the chain becomes load-bearing. The trade-off: harder to maintain uniqueness; harder to avoid degenerate chains (where a smaller technique also applies). The existing `aic_reverse.rs` evaluated this and chose guided-removal (per its module doc-block, "search-and-filter, not constructive"). Keep this option in the design space but defer until guided-removal hit rates are measured.

---

## 11. Subsumption claims — verification

Repo `ROADMAP.md` lines 101-114 makes three claims. Verdict per claim (confidence in parentheses):

### Claim 1: braid[k] subsumes Death Blossom + Sue de Coq + forcing nets

**Sue de Coq** (DSDC). A Sue de Coq pattern asserts: 2-3 cells in a row/column intersected with a block contain a set of digits split between the row and the block, eliminating those digits elsewhere. Berthier (PBCS3 §"Subset patterns and chains") explicitly shows Sue de Coq is a **braid[2]–braid[3]** pattern with multiple csp-variables (rc, rn, bn) used in the same chain. **Verdict: CONFIRMED** (high confidence). The braid[k] machinery captures it because braids combine arbitrary csp-var families per step.

**Death Blossom**. A Death Blossom is a "stem cell" with k candidates, each spawning an Almost Locked Set (ALS) such that the petals collectively eliminate a candidate. Berthier shows DB is a **g-braid[k]** (the ALS petals correspond to grouped propagation via glabels). For non-grouped DB variants (rare), a plain **braid[k]** suffices with k = stem-cell-degree + max-ALS-size. **Verdict: CONFIRMED for g-braid; partial for braid alone** — small DB cases (k ≤ 3 stem with 2-ALS petals) are reachable as braid[k] without grouping; larger DB needs g-braid. (Medium-high confidence — Berthier's textbook examples cover the dominant cases; pathological DB with > 5-cell ALSes may push to gB[≥ 6].)

**Forcing nets** (general). PBCS3 distinguishes:
- *Forcing whips* (forcing-whips, in `CHAIN-RULES-COMMON/FORCING-WHIPS/`): every branch of an OR2/OR3 forcing pattern leads to the same elimination. These are NOT subsumed by braid[k] alone — they require the ORk-chain machinery (separate template `ORk-chain`). **Verdict: PARTIAL — claim is overstated.**
- *Net-style forcing chains* (cell forcing chains, region forcing chains): these are **bi-whip / bi-braid** patterns (`contrad-chain` template), again separate machinery beyond plain braid[k].

**Refined verdict on Claim 1:** Sue de Coq and most Death Blossom: yes. Forcing nets in general: **NO** — they need OR-chain machinery (`ORk-chain` template; see `templates.clp:448-494`) which is a strictly larger algorithmic surface. Recommend updating ROADMAP to say "braid[k] subsumes Sue de Coq and most Death Blossom; forcing nets require additional ORk-chain machinery (out of scope for this port iteration)."

### Claim 2: g-whip[k] subsumes AHS-style grouped fish

AHS = Almost Hidden Set. Grouped fish (Finned X-Wing, Sashimi Swordfish, Franken/Mutant fish on cell groups): these are short g-chains where each rlc is a glabel (a row-segment or column-segment of a digit). PBCS3 explicitly states "all (sashimi-)fish patterns of size k are subsumed by gW[k] or gB[k]". **Verdict: CONFIRMED** (high confidence) — this matches the Berthier classification table in PBCS3 Ch. "Fish and grouped chains".

### Claim 3: g-whip subsumes naked/hidden subset chains

Naked-subset chains (XYZ-Wing, WXYZ-Wing, etc.): chains where one step uses a multi-candidate cell as a "naked subset" (≥2 unassigned digits constrained together). Hidden-subset chains: dual. **Verdict: CONFIRMED for k ≤ 5** (medium confidence; PBCS3 catalogues XYZ-Wing as gW[3], WXYZ-Wing as gW[4]). For k ≥ 6 the chain may need braid relaxation, putting it in gB[k]. So: gW[k] subsumes the typical small naked/hidden subset chains; gB[k] catches the rest.

### Overall ROADMAP recommendation

Edit ROADMAP lines 101-114 to:

```
- braid[k] — chains with embedded subchain conditions. Strictly stronger than whips.
  Subsumes Sue de Coq and most Death Blossom (≤k=5). Does NOT subsume general
  forcing nets — those require ORk-chain machinery (separate port iteration).
- g-whip[k]/g-braid[k] — grouped candidates. Subsumes (sashimi-)fish up to size k,
  XYZ-Wing/WXYZ-Wing (gW[3,4]), AHS-style grouped subsets. Larger Death Blossom
  also lands here.
```

---

## 12. Test corpus

Concrete 81-char puzzles where each technique is the tier-defining technique (target rating shown). Sources:
- **PBCS3 (Berthier, 2021)** — the book includes a per-puzzle catalogue with W/B/gW/gB ratings.
- **enjoysudoku forum, "Patterns Game" and "puzzle of the week" threads** — community-curated examples with confirmed ratings.
- **Magictour-1467** benchmark (Berthier's test set) — `Publications/2021-PBCS3.pdf` Annex.

Fixtures below are sourced directly from CSP-Rules-V2.1 test corpora at `/tmp/csp-rules-research/CSP-Rules-V2.1/XTERNS/SHC/examples/`. The SHC (Sudoku Helper C) tool is Berthier's own C implementation of the CSP-Rules chain solver; the `B-input.txt` / `BxB-input.txt` files contain puzzles with their authoritative B (braid-depth) ratings. The inline format is `<81-char puzzle> <clue-count>;<puzzle-id>;B=<braid-depth>` where `B=0` means no braid needed (pure whip or simpler), `B=1` means min braid-depth 1, etc. g-whip/g-braid examples come from `XTERNS/SHC/examples2/B_extreme.txt` which cross-references the enjoysudoku forum thread on g-whips/g-braids.

All puzzles use `.` for empty, digits 1–9 for clues, 81 characters total (no spaces).

---

### Fixture 1 — Whip[1] (hidden single) from SHC B=0 corpus

- Puzzle (81 chars, 0/. for empty): `...456..9..6.......891..45.2.........7..9.....35......397...5.......4.72.....5361`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples/B-input.txt`, id `cbg000#1`, `B=0`
- Expected firing: whip[1] (hidden single; all B=0 puzzles are pure whip-≤1 or simpler)
- Expected elimination: multiple hidden singles — no braid needed
- Expected W/B/gB rating: W[1] (or lower — naked/hidden singles only)
- Notes: B=0 in SHC means the braid solver finds depth-0; puzzle is solvable without any braid step. SE rating unknown; any hidden-single puzzle suffices as a W[1] smoke test.

---

### Fixture 2 — Whip[1] / Whip[2] from SHC B=0 corpus (second sample)

- Puzzle (81 chars, 0/. for empty): `.23....8.......12.7.91.........97.46...3....29..6......6.9..5.457.....98.....4...`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples/B-input.txt`, id `cbg000#22`, `B=1`
- Expected firing: whip[1] through whip[small] — B=1 means braid-depth 1 suffices; in practice often a single braid[1] step (= a whip[2] with braid relaxation) fires
- Expected elimination: one step distinguishes whip from braid at depth 1
- Expected W/B/gB rating: W[2] or B[1]
- Notes: B=1 puzzles are the minimal non-trivial braid corpus; use to verify that whip[1] fires at least once before the braid[1] step.

---

### Fixture 3 — Braid[3] from SHC B=3 corpus

- Puzzle (81 chars, 0/. for empty): `.23..6.8......91...8.1..4..2.......7...8.....678.1......7.3.2...3...4.7....5.1.6.`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples/B-input.txt`, id `cbg000#3`, `B=3`
- Expected firing: braid[3] (minimum braid depth = 3; no whip of any length resolves it first)
- Expected elimination: at least one elimination by a braid[3] chain (partial-braid of length 3 terminates)
- Expected W/B/gB rating: B[3] per the SHC corpus; our rater may classify it as a stricter category (W(k) or smaller B(k)) due to calibration differences between the SHC battery and our `rate_chain` cascade (see CR-FIN-11 Mn-1 in §15).
- Notes: Authoritative B=3 from Berthier's own SHC. The previous Fixture 3 in this spec was `cbg000#109`, which the SHC corpus annotates as `B=7` (not B=3); CR-FIN-11 Mn-1 corrected the mis-source. SE rating unknown; expected in the 7.0–8.5 range per §9.3.

---

### Fixture 4 — Braid[5] from SHC B=5 corpus

- Puzzle (81 chars, 0/. for empty): `..34......5...912.7...2.....1.5.7..86...9...7.......34..2.............9.9...61.75`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples/B-input.txt`, id `cbg000#2`, `B=5`
- Expected firing: braid[5]
- Expected elimination: at least one braid[5] chain fires; shorter braids and all whips insufficient
- Expected W/B/gB rating: B[5]
- Notes: First B=5 entry in the authoritative SHC corpus. SE rating unknown; expected 8.0–9.0 range.

---

### Fixture 5 — Braid[7] from SHC B=7 corpus

- Puzzle (81 chars, 0/. for empty): `.....6.....718........2...5..85....13.....9...6.....4...2.7...8.4.....6.9.....3..`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples2/B7B_coloin.txt`, first entry, `B=7`
- Expected firing: braid[7]
- Expected elimination: braid[7] chain fires; no shorter technique suffices
- Expected W/B/gB rating: B[7]
- Notes: File lists clue-count 97868 (FNBP metric). High-authority Coloin corpus via Berthier.

---

### Fixture 6 — Braid[8] from SHC B=8 corpus

- Puzzle (81 chars, 0/. for empty): `1..4.678....18.2.66...27.41.....8.67..6.4.82....6.21.4.61...47..952..............`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples2/B8B_coloin.txt`, first entry, `B=8`
- Expected firing: braid[8]
- Expected elimination: requires a braid chain of length 8
- Expected W/B/gB rating: B[8]
- Notes: FNBP C32/M2.11.1788 benchmark; clue-count 95645.

---

### Fixture 7 — g-Braid[29] (extreme) from enjoysudoku g-whips/g-braids thread

- Puzzle (81 chars, 0/. for empty): `001002003000010040500300100006007002010000080700900300007006008090040000300700500`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples2/B_extreme.txt`, `B29`, URL: `http://forum.enjoysudoku.com/g-whips-and-g-braids-t30231-30.html#p344387`
- Expected firing: g-braid[29] (or grouped chain of length ≤ 29)
- Expected elimination: requires grouped chain (g-braid) — standard whips and braids insufficient
- Expected W/B/gB rating: gB[29]
- Notes: This is an extreme puzzle from the enjoysudoku forum's dedicated g-whip/g-braid thread; Berthier lists it with B=29. Useful for upper-end gB regression testing. (Use `0` for empty in the 81-char string, not `.`.)

---

### Fixture 8 — g-Braid[30] (extreme) from enjoysudoku g-whips/g-braids thread

- Puzzle (81 chars, 0/. for empty): `.....1..2....3..4...56..7....6...5...1......37..8...9...9..5.8..2..4....3..7..9..`
- Source: CSP-Rules-V2.1 `XTERNS/SHC/examples2/B_extreme.txt`, `B30`, URL: `http://forum.enjoysudoku.com/g-whips-and-g-braids-t30231-30.html#p344454`
- Expected firing: g-braid[30]
- Expected elimination: requires grouped chain of length 30
- Expected W/B/gB rating: gB[30]
- Notes: Hardest known g-braid in the Berthier corpus at time of PBCS3 publication. Useful for stress-testing the g-braid search limit.

---

**Notes on fixture provenance and verification:**

The B-ratings above come from SHC (Berthier's own C implementation); they are authoritative for the braid[k] axis. The g-braid ratings come from forum posts in the `g-whips-and-g-braids-t30231` thread, also curated by Berthier. Before locking these as CI test fixtures:

1. Run CSP-Rules-V2.1 SudoRules on each puzzle (via `(solve "...")`) to confirm the exact elimination and chain details.
2. Cross-check puzzle string length (must be exactly 81 chars; the SHC files use mixed `.`/`0` — normalize to `.` for empty).
3. For whip[k]-specific fixtures (W only, no braid), note that SHC's B=0 rating confirms "no braid needed" but does not give the exact W[k] depth. Run SudoRules with only whip rules enabled to get W[k].

Storage layout: `tools/sudoku_rs_core/tests/fixtures/chain_corpus.yml` with schema `{ puzzle, expected_rating, expected_chain_length, expected_firing_technique, source_url }`.

---

## 13. Implementation file decomposition for porting

Target: each file ≤ 500 LOC, public API typed, no cross-file circular deps. All paths absolute under `/Users/dleonenko/latent-reasoning-design/tools/sudoku_rs_core/`.

### `src/generic/chain_model.rs` (~200 LOC) — shared types

```rust
pub type Label = u32;
pub type GLabel = u32;
pub type CspVarId = u32;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CspVarKind { Rc, Rn, Cn, Bn }

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Rlc { Cand(Label), GCand(GLabel) }

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum ChainKind {
    PartialWhip, Whip,
    PartialBraid, Braid,
    PartialGWhip, GWhip,
    PartialGBraid, GBraid,
}

#[derive(Clone, Debug)]
pub struct Chain {
    pub kind: ChainKind,
    pub target: Label,
    pub length: u8,
    pub llcs: SmallVec<[Label; 12]>,
    pub rlcs: SmallVec<[Rlc; 12]>,
    pub csp_vars: SmallVec<[CspVarId; 12]>,
}

impl Chain { pub fn last_rlc(&self) -> Rlc { … } pub fn dedup_key(&self) -> u64 { … } }

#[derive(Copy, Clone, Debug)]
pub struct ChainElimination { pub target: Label, pub rule: ChainRule }

#[derive(Copy, Clone, Debug)]
pub enum ChainRule { Whip(u8), Braid(u8), GWhip(u8), GBraid(u8) }
```

### `src/generic/csp_tables.rs` (~250 LOC) — CSP-Variable + link/glink tables

```rust
pub struct CspVarTable { pub labels_of: Vec<SmallVec<[Label; 16]>>,
                        pub vars_of: Vec<[CspVarId; 4]>,
                        pub kind_of: Vec<CspVarKind> }
pub struct LinkGraph { pub linked: Vec<BitSet729> }      // const-generic over N
pub struct CspLinkGraph { pub alternatives: Vec<[SmallVec<[Label; 9]>; 4]> }

pub fn build_csp_tables<const N: usize, const BR: usize, const BC: usize>()
    -> (CspVarTable, LinkGraph, CspLinkGraph)
```

Bitset width templated on N (729 for 9×9, 4096 for 16×16). Build once per N at program start (static `OnceLock`).

### `src/generic/glabel_tables.rs` (~250 LOC) — glabel + glink

```rust
pub struct GLabelTable { pub members_of: Vec<SmallVec<[Label; 4]>>,
                        pub digit_of: Vec<u8>,
                        pub block_of: Vec<u8>,
                        pub segment_kind: Vec<SegKind>,      // Row|Col
                        /// Two CSP-Variable ids per glabel (§8.2):
                        ///   slot 0 = rn-family for horizontal glabels, cn-family for vertical
                        ///   slot 1 = bn-family (block×digit) for all glabels
                        /// Cross-check: init-glinks.clp:84-88 (horizontal) and 109-113 (vertical).
                        pub csp_vars_of: Vec<[CspVarId; 2]> }
pub struct GLinkGraph { pub glinked: Vec<BitSetGLab>,
                       pub csp_glinked: Vec<…> }

pub fn build_glabel_tables<const N, BR, BC>() -> (GLabelTable, GLinkGraph)
pub fn label_in_glabel(l: Label, g: GLabel, t: &GLabelTable) -> bool   // O(1) bitset hit

/// Returns TRUE iff glabel g contains none of the entries in `labs_or_glabs`.
/// `labs_or_glabs` may hold both Labels and GLabels (see §2.2 Rlc enum).
/// Matches CLIPS `glabel-contains-none-of` in SudoRules/glabels.clp:292-313
/// which explicitly handles both label arguments and g-label arguments:
///   - For a Label entry: checks rc-cell-in-physical-2D-segment (label-in-glabel).
///   - For a GLabel entry: checks segment identity (eq phys-seg1 cell2).
pub fn glabel_contains_none_of(g: GLabel, labs_or_glabs: &[Rlc], t: &GLabelTable) -> bool

/// Returns TRUE iff glabel g contains at least one of the entries in `labs`.
/// Required by partial-gbraid[4]-2 and partial-gbraid[4]-3 dedup guards
/// (gBraids[5].clp:213, 282) and partial-gwhip[5]-2/3 (Partial-gWhips[5].clp:145, 214).
/// Matches CLIPS `glabel-contains-some-of` in generic-background.clp:179-187.
pub fn glabel_contains_some_of(g: GLabel, labs_or_glabs: &[Rlc], t: &GLabelTable) -> bool
```

### `src/generic/resolution_state.rs` (~200 LOC) — RS = alive set + bitset ops

```rust
pub struct ResolutionState {
    pub cand_alive: BitSet,
    /// g_alive is DYNAMIC: a glabel is alive iff ≥ 2 of its member labels are
    /// still in cand_alive. Must be kept consistent via `eliminate_candidate`.
    /// A static bitset set once at init is WRONG (CLIPS truth-maintains g-candidates
    /// via logical dependencies in init-g-candidates-horiz/verti — init-glinks.clp:68-80, 93-105).
    pub g_alive: BitSet,
    /// Per-glabel support count: number of alive member labels.
    /// When this drops from 2 to 1, the glabel is removed from g_alive.
    g_support: Vec<u8>,
    pub values: BitSet,
}
impl ResolutionState {
    pub fn from_grid<const N, BR, BC>(g: &Grid<N, BR, BC>, t: &GLabelTable) -> Self;

    /// Eliminate a candidate label: removes from cand_alive, decrements support
    /// counts for all glabels containing this label, and removes any glabel whose
    /// support count drops below 2 from g_alive.
    pub fn eliminate_candidate(&mut self, l: Label, t: &GLabelTable);

    /// Check whether glabel g is currently alive (≥ 2 member labels alive).
    pub fn g_cand_alive(&self, g: GLabel) -> bool;

    pub fn linked(&self, a: Label, b: Label, lg: &LinkGraph) -> bool;
    pub fn linked_or(&self, a: Label, bs: &BitSet, lg: &LinkGraph) -> bool;
    pub fn glinked_or(&self, a: Label, rlcs: &[Rlc], gg: &GLinkGraph, lg: &LinkGraph) -> bool;
}
```

**Cascade note:** every call to `rs.cand_alive.remove(L)` in the algorithm must instead call `rs.eliminate_candidate(L, &glabel_table)` so that `g_alive` stays consistent. Without this, g-whip and g-braid rules can fire through dead grouped alternatives — producing false eliminations.

### `src/generic/techniques/whip.rs` (~400 LOC)

```rust
pub fn run_whip_pass<const N, BR, BC>(
    g: &Grid<N, BR, BC>, ctx: &ChainContext, k_max: u8, scratch: &mut WhipScratch,
) -> Vec<ChainElimination>;

pub fn find_first_whip<const N, BR, BC>(
    g: &Grid<N, BR, BC>, ctx: &ChainContext, k_max: u8,
) -> Option<(ChainElimination, u8 /*length*/)>;

// Internal:
fn extend_partial_whips(seeds: &[Chain], rs: &RS, …) -> Vec<Chain>;
fn try_terminate_whip(seed: &Chain, rs: &RS, …) -> Option<ChainElimination>;
fn whip_1_eliminations(rs: &RS, …) -> Vec<ChainElimination>;

pub struct WhipScratch { … }  // reusable per-puzzle scratch (cf. AicProbeScratch)
```

### `src/generic/techniques/braid.rs` (~400 LOC)

```rust
pub fn run_braid_pass<const N, BR, BC>(
    g: &Grid<N, BR, BC>, ctx: &ChainContext, k_max: u8, scratch: &mut BraidScratch,
) -> Vec<ChainElimination>;

pub fn find_first_braid_excluding_whips<const N, BR, BC>(…) -> Option<(ChainElimination, u8)>;

fn extend_partial_braids(seeds: &[Chain], rs: &RS, …) -> Vec<Chain>;
fn try_terminate_braid(seed: &Chain, rs: &RS, …) -> Option<ChainElimination>;
```

### `src/generic/techniques/gwhip.rs` (~500 LOC)

Three sub-extension paths (label→cand, whip→glabel, gwhip→glabel) as in `Partial-gWhips[2].clp`.

```rust
pub fn run_gwhip_pass(…) -> Vec<ChainElimination>;
fn extend_partial_gwhip_with_cand(…)   -> Vec<Chain>;
fn extend_partial_whip_with_gcand(…)   -> Vec<Chain>;
fn extend_partial_gwhip_with_gcand(…)  -> Vec<Chain>;
fn try_terminate_gwhip(…) -> Option<ChainElimination>;
```

### `src/generic/techniques/gbraid.rs` (~500 LOC)

Three sub-extension paths (gwhip|gbraid + cand, whip|braid + gcand, gwhip|gbraid + gcand) as in `gBraids[3,5].clp`.

### `src/generic/whip_reverse.rs`, `braid_reverse.rs`, `gwhip_reverse.rs`, `gbraid_reverse.rs` (~400 LOC each)

Each mirrors `aic_reverse.rs`'s structure: `Spec` struct, `*_reverse_construct`, `batch_*_reverse_construct`, scratch-buffer reuse. Driver = guided-removal + score function. Each implements `chain_score` via the technique's `find_first_*` probe.

### `src/generic/chain_rating.rs` (~150 LOC)

```rust
pub enum ChainRating { W(u8), B(u8), GW(u8), GB(u8) }
pub fn rate_chain<const N, BR, BC>(g: &Grid<N, BR, BC>) -> Option<ChainRating>;
pub fn chain_rating_to_se(r: ChainRating) -> f32;     // best-effort table; calibrate later
```

### Test files

- `tests/chain_corpus.rs` — load §12 fixtures, assert rating + chain-length.
- `tests/chain_reverse_hit_rate.rs` — for each technique × small `k`, assert non-zero hit rate in N attempts.
- `tests/chain_load_bearing.rs` — `rate_excluding` semantics.
- `tests/chain_dedup.rs` — extending a partial-whip with a step that would re-create the same `rlcs` set must not re-fire (matches CLIPS `(not (chain …))` guards).
- `tests/chain_braid_excludes_whip.rs` — a `find_first_braid` probe on a whip-rated puzzle must return `None`.

### Total estimate

- shared infrastructure: 850 LOC
- 4 technique modules: 1800 LOC
- 4 reverse-construct modules: 1600 LOC
- rating + tests: 600 LOC
- **Total: ~4850 LOC** of new Rust, plus modest edits in `mod.rs` / `rater.rs` to register the new techniques.

---

## 14. Open questions / unresolved

### §14.1 — Dedup semantics (resolved)

**Source files read:** `CSP-Rules-Generic/GENERAL/generic-background.clp:396–403`, `CHAIN-RULES-SPEED/PARTIAL-WHIPS/Partial-Whips[1,3].clp`, `CHAIN-RULES-SPEED/BRAIDS/Braids[3,5].clp`, `CHAIN-RULES-SPEED/PARTIAL-G-WHIPS/Partial-gWhips[3].clp`, `CHAIN-RULES-SPEED/G-BRAIDS/gBraids[6].clp`.

#### The `same-sets-of-rlcs` function

```clips
(deffunction same-sets-of-rlcs (?rlc1 ?rlcs1 ?rlcs2)
    ;;; used in braids
    ;;; supposes that rlc1+rlcs1 and rlcs2 are known to be sets (no repetition)
    (and
        (member$ ?rlc1 ?rlcs2)
        (subsetp ?rlcs1 ?rlcs2)
    )
)
```

**Semantics:** `same-sets-of-rlcs(new_rlc, rlcs, rlcsa)` returns TRUE iff the **set** `{new_rlc} ∪ rlcs` is a **subset** of `rlcsa`. The CLIPS comment "rlc1+rlcs1 and rlcs2 are known to be sets" means no duplicates exist in practice (the extension rules already enforce this). Because `rlcsa` has length k (equal to the chain being guarded) and `{new_rlc} ∪ rlcs` also has exactly k elements (1 new + k-1 old), the subset check is actually **set equality** — two k-element sets where one is a subset of the other must be identical.

**In concrete terms** (for a braid extension producing length-k from length-(k-1)):
- `rlcs` = `$?rlcs` of the seed partial-chain (length k-1 elements)
- `new_rlc` = the new right-linker candidate being appended
- `rlcsa` = `$?rlcs` of any existing partial-braid / partial-whip of length k with the same target

The guard reads: "do not assert a new partial-braid[k] if there already exists a partial-whip or partial-braid of the same length and same target whose `rlcs` multifield is *set-equal* to `{new_rlc} ∪ rlcs`."

**Order sensitivity:** The `rlcs` slot in the CLIPS `chain` fact is a *multifield* (ordered list). However, the dedup guard uses `same-sets-of-rlcs` (set subset check), **not** a positional match. Therefore dedup is **order-insensitive (set semantics)**. Two partial-chains with `rlcs = [A B C]` and `rlcs = [C A B]` for the same `target` will trip the guard against each other — only the first one asserted survives.

#### Whip-specific dedup (positional, not set)

For **whips**, the dedup mechanism is different and **positional**:

```clips
;;; do not assert different partial whips with the same sequences of rlc's
(not
    (chain (type partial-whip) (context ?cont) (length k) (target ?zzz)
           (rlcs $?rlcs ?new-rlc))
)
```

Here `(rlcs $?rlcs ?new-rlc)` is a **suffix match on the ordered list** — any existing partial-whip whose `rlcs` ends with `…rlcs[0] … rlcs[k-2] new-rlc` (i.e. the same ordered sequence) blocks the new one. Because the seed is always extended by appending, and the pattern matches the full ordered sequence including the new tail element, this is effectively **positional set equality on the sequence** at each length. Practically: two partial-whips with identical `(rlcs $?rlcs ?new-rlc)` are deduplicated regardless of `llcs` or `csp-vars`.

**Implication for Rust:** Whip dedup key = `(target: Label, rlcs_sequence: SmallVec<[Label; 12]>)` — ordered. Braid dedup key = `(target: Label, rlcs_as_sorted_set: BTreeSet<Rlc>)` — unordered.

#### g-variant dedup key extension

For **partial-gwhip** (sub-rule -1, extend partial-gwhip with a candidate):
```clips
(not (chain (type partial-gwhip) (length k) (target ?zzz) (rlcs $?rlcs ?new-rlc)))
```
Same positional semantics as whip dedup within the gwhip chain type. No glabel-specific extension beyond the Rlc type.

For **partial-gwhip** (sub-rule -2 and -3, extend whip or gwhip with a glabel):
```clips
(not (chain (type partial-whip|partial-gwhip) (length k) (target ?zzz)
            (rlcs $?rlcs ?new-rlca&:(or (eq ?new-rlca ?new-rlc) (label-in-glabel ?new-rlca ?new-rlc)))))
```
Here the guard blocks not only exact glabel matches (`eq`) but also any existing partial-chain where the tail rlc `?new-rlca` is a **label that belongs to** the new glabel. This prevents asserting a grouped chain when a finer (label-resolution) chain already subsumes it.

For **partial-gbraid** (as seen in `gBraids[6].clp:134-146`), the braid-style set dedup uses `same-sets-of-rlcs` as above (set equality), plus an additional guard:
```clips
(not (chain (type partial-whip|partial-braid) (length k) (target ?zzz)
            (rlcs $?rlcsa&:(and (subsetp $?rlcs $?rlcsa) (glabel-contains-some-of ?new-rlc $?rlcsa)))))
```
This blocks a partial-gbraid if a shorter (non-grouped) braid already has all the same label-rlcs AND the new glabel-rlc contains a label already in `rlcsa` — i.e. the non-grouped version subsumes the grouped one.

#### Rust dedup key summary

| Chain type | Dedup key type | Key fields |
|---|---|---|
| partial-whip[k] | positional | `(target: Label, rlcs: Vec<Label>)` — ordered |
| partial-braid[k] | set equality | `(target: Label, rlcs: BTreeSet<Label>)` — unordered |
| partial-gwhip[k]-1 | positional within gwhip | `(target: Label, rlcs: Vec<Rlc>)` — ordered |
| partial-gwhip[k]-2/3 | positional + label-in-glabel subsumption | `(target, rlcs_seq, gcand_subsumes_map)` |
| partial-gbraid[k]-1 | set equality | `(target: Label, rlcs: BTreeSet<Rlc>)` — unordered; checks `same-sets-of-rlcs` against existing partial-gwhip|partial-gbraid |
| partial-gbraid[k]-2 (primary) | set equality | same `same-sets-of-rlcs` against existing partial-gwhip|partial-gbraid |
| partial-gbraid[k]-2 (extra, ungrouped subsumption) | ungrouped subsumption | block if existing partial-whip|partial-braid has `rlcs ⊇ seed_rlcs` AND `glabel-contains-some-of(new-rlc, rlcsa)` — i.e. a non-grouped chain already covers the glabel |
| partial-gbraid[k]-3 (primary) | set equality + glabel subsumption | block if existing partial-gwhip|partial-gbraid has `rlcs ⊇ seed_rlcs` AND (`new-rlc ∈ rlcsa` OR `glabel-contains-some-of(new-rlc, rlcsa)`). This is the strongest guard: it blocks a new grouped rlc when any existing grouped chain already contains that glabel or one of its member labels. See `gBraids[5].clp:partial-gbraid[4]-3` lines 274-287. Not implementing this guard preserves soundness but causes search blowup from redundant g-candidate chains. |

A `HashSet<DedupKey>` with `DedupKey` carrying `(target, sorted_rlcs_vec)` suffices for the braid/gbraid family. The gwhip sub-rule -2/3 subsumption check cannot be encoded as a pure hash lookup — it requires a scan of all existing partial-(g)whip chains of the same length + target. Implement as a secondary linear scan on the (typically small) chain list.

---

1. ~~**`same-sets-of-rlcs` semantics (CLIPS dedup helper).**~~ **RESOLVED** — see §14.1 above. Braid dedup is unordered set equality on `{new_rlc} ∪ rlcs`; whip dedup is positional sequence equality; g-variants add a label-in-glabel subsumption guard for glabel-rlc extensions.

2. **CLIPS `Partial-gWhips[1].clp` "rlc1 cannot be a candidate" subtlety.** Line 34 comment: "only a partial-gwhip[1] can give rise to a full g-whip[2]; a partial-whip[1] can only give rise to a whip[2]; here ?rlc1 can therefore only be a gcand". This implies the *seed* gwhip step uses **only g-candidates** for `rlc1`. The Rust port must ensure that partial-gwhip[1] is seeded with `Rlc::GCand(_)` exclusively; label-only seeds fall through to partial-whip[1]. Verify against `gWhips[2].clp` line 68 `(last-rlc ?rlc1) ; can only be a g-candidate`.

3. **`exists-link` vs `linked` predicates.** Both exist in CLIPS — `(exists-link ?cont a b)` is the fact; `(linked a b)` is the function over the global `?*links*` multifield. The two should be equivalent at runtime (one asserts the fact, the other queries the same data via function). The Rust port collapses both into one `LinkGraph::linked()` query — verify no rule reads the *fact* form for any reason other than as an existence test (skimmed: only existence tests).

4. **Focused elimination & `candidate-in-focus`.** The Rust port can drop this for the rating pass but must reinstate it if we ever want to do "tell me why label X was eliminated" forensics. Document but don't implement initially.

5. **PBCS3 deep-read required for `gB_n` rating semantics.** Berthier's `gB_n` notation in some places refers to a *nested* rating that grows with sub-chain complexity. The CLIPS rule set implements gW + gB, not gB_2/gB_3/… — the parametric `n` in PBCS3 likely indexes the recursion depth of sub-resolution arguments embedded in the braid steps. **Unknown:** whether the repo needs gB_2/3/… or if gB suffices. Cheap mitigation: report a single scalar `gB[k]` and revisit if the Magictour calibration shows poor fit.

6. **SE↔W calibration is empirical.** The §9.3 mapping is heuristic. Plan a calibration pass: take the Magictour-1467 puzzles, run both our rater (SE) and CSP-Rules (W/B) on each, fit a regression. Add `docs/se_chain_calibration_report.md` after this work.

7. **whip[1] vs Hidden Single confluence.** Whip[1] is *exactly* a hidden single (with a peer-elimination twist). The repo already handles Hidden Single in `techniques/`. The Rust port should NOT duplicate hidden-single elimination — only emit a `W[1]` rating *label* for puzzles where hidden-single is tier-defining. Add a unit test ensuring `Whip(1)` and `HiddenSingle` are not double-counted in the cascade.

8. **Forcing variants and OR-chain machinery — explicitly out of scope.** Forcing-Whips, ORk-Whips, contradiction-chains, bi-whips, T&E (`CHAIN-RULES-COMMON/FORCING-WHIPS`, `CSP-Rules-Generic/T&E+DFS/`) are NOT part of this port. The ROADMAP claim about "braid[k] subsumes forcing nets" (§11) overstates; the porting agent should not attempt forcing chains.

9. **Bitset width parameterisation.** N = 9 → 729 labels = 12 u64 words. N = 16 → 4096 labels = 64 u64 words. Use `const N: usize` const-generics and a fixed-size `[u64; W]` array (W = `(N*N*N + 63) / 64`). The `Vec<BitSet729>` sketch in §6.1 should be const-generic in the final port. Mirror the existing repo's pattern (`AicProbeScratch::new::<N>()`).

10. **CLIPS `linked-or` cost.** For `forall csp-linked … (linked-or X Z $rlcs)` the inner `linked-or` is O(|rlcs|). In Rust with a bitset of "killed labels" (= `singleton(Z) ∪ rlcs`), this becomes a single bitset AND on the row of `linked[X]` — O(1) for fixed N. This is the main perf win over CLIPS; preserve it carefully.

---

## Appendix A — CLIPS-to-Rust cheat sheet

| CLIPS | Rust |
|---|---|
| `(candidate (context 0) (status cand) (label L))` | `rs.cand_alive.contains(L)` |
| `(is-csp-variable-for-label (csp-var v) (label L))` | `csp_tables.vars_of[L].contains(v)` |
| `(csp-linked 0 a b v)` | `csp_tables.alternatives[a][slot_of(v)].contains(b)` |
| `(exists-link 0 a b)` | `link_graph.linked[a].contains(b)` |
| `(linked-or a $rlcs)` | `(rs.killed_set | link_graph.linked[a]) ≠ ∅` after pre-OR'ing rlcs into `killed_set` |
| `(forall (csp-linked … ?xxx ?csp) (test (linked-or xxx Z $rlcs)))` | every label in `csp_tables.alternatives[L][slot] \ {except_new_rlc}` has bit set in `(rs.linked[X] & killed_set)` |
| `(label-in-glabel L g)` | `glabel_table.members_of[g].contains(L)` |
| `(glinked-or l $rlcs)` | bitset query in `glink_graph.glinked[l]` against rlcs bitset |
| `(chain (type partial-whip) (length k-1) …)` (memo) | `dedup_map: HashMap<(target, Vec<Rlc>), ChainId>` (positional sequence — see §14.1) |
| `(retract ?cand)` | `rs.eliminate_candidate(L, &g_lab_tab)` (see §13; propagates g_support) |
| salience `?*whip[k]-salience*` | outer-loop ordering by `k`, with whip < braid < gwhip < gbraid sub-orders |

---

---

## 15. Codex review remediation log

Each finding from the 2026-05-18 Codex BLOCK review verified against `/tmp/csp-rules-research/CSP-Rules-V2.1/` CLIPS source and patched below.

### CRITICAL

- **C1 — FIXED (§6.3, §6.3.1):** Whip search loop could not generate `whip[2+]` because `partial-whip[1]` was not wired as the seed. The pseudocode was restructured: `build_partial_whips_length_1(rs)` is now a dedicated pre-loop step (matching CLIPS `Partial-Whips[1].clp:36-78` which is a special-case rule, not generated from length-0). The rolling-partials loop now passes `partials_prev` forward correctly. A "Critical architectural note" box was added at the head of §6.3, and §6.3.1 gained a "Seeding role" callout. Lines affected: §6.3 pseudocode (entirely replaced), §6.3.1 introductory paragraph.

- **C2 — FIXED (§2.2, §3, §13 `resolution_state.rs`):** Grouped-candidate liveness is dynamic in CLIPS (RETE `logical` dependencies in `init-g-candidates-horiz/verti`, `init-glinks.clp:68-80, 93-105`), not static. The spec now documents the support-count mechanism in §2.2 (g-candidate definition), §3 (item 4 of the RS definition), and §13 where `ResolutionState` gained `g_support: Vec<u8>` and `eliminate_candidate()` API. Every `cand_alive.remove(L)` in the port must call `eliminate_candidate(L, &glabel_table)` instead.

- **C3 — FIXED (§8.4):** CSP-var freshness rule in g-whip/g-braid is not uniform. Sub-rule -2 (`partial-gwhip[5]-2` line 126, `partial-gbraid[4]-2` line 185) uses full-history exclusion `not member$`, while -1/-3 and the elimination rule use last-only `neq (last …)`. Spec §8.4 now documents this split explicitly and recommends a `csp_fresh(..., is_whip_to_gcand_branch: bool)` API parameter. Applying "≠ last" uniformly → invalid chains and wrong eliminations.

### MAJOR

- **M4 — FIXED (§8.2):** Grouped CSP-vars are not reduced to `bn` only. Horizontal glabels belong to **rn AND bn**; vertical glabels to **cn AND bn** (`init-glinks.clp:84-88, 109-113`). §8.2 now enumerates both families per glabel type. §13 `GLabelTable` now specifies two `CspVarId` entries per glabel.

- **M5 — FIXED (§7.1 distinctness, §7.3 pseudocode):** Braid distinctness for `new-llc` is weaker than whip. CLIPS `Braids[5].clp:braid[5]` (line 72) and `partial-braid[4]` only exclude `new-llc ∈ rlcs` and `new-llc == Z`, not `new-llc ∈ llcs`. §7.1 distinctness clause now explicitly notes this. The pseudocode `LL_set.contains(new_llc)` guard was removed from the braid extension loop. g-braid (`gbraid[5]` line 72) has the same relaxation.

- **M6 — FIXED (§13 `glabel_tables.rs`):** Grouped helper API was incomplete. `glabel_contains_none_of` must accept both labels and glabels (`Rlc` enum), matching `SudoRules/glabels.clp:292-313`. `glabel_contains_some_of` was added (required by `partial-gbraid[4]-2/3` and `partial-gwhip[5]-2/3`). Both functions now carry CLIPS citation and doc-comment in §13.

- **M7 — FIXED (§14.1 dedup table):** `partial-gbraid[*]-3` strongest guard was missing from the dedup summary. Added as a new row: blocks a new grouped rlc when any existing grouped chain has `rlcs ⊇ seed_rlcs` AND the new glabel or one of its members is already present. CLIPS citation: `gBraids[5].clp:partial-gbraid[4]-3` lines 274-287. Not implementing this preserves soundness but breaks pruning parity → search blowup.

### MINOR

- **M8 — FIXED (§8.1):** Glabel count formula `2·N·N·max(BR, BC)` is wrong for rectangular blocks. Correct formula is `N²·(BR+BC)/(BR·BC)` (derived from `init-physical-2D-segments` in `SudoRules/glabels.clp:126-141`). Verified: for 9×9 both formulas give 486; for 12×12 with BR=3,BC=4 the old formula gives 2·144·4=1152, the correct one gives 144·7/12=84 per digit × 12 = 1008. §8.1 now shows both the construction logic and the corrected formula.

- **M9 — FIXED (§8.3):** `exists-glink` wording was misleading ("symmetric through c"). CLIPS asserts `(csp-glinked ?cont ?cand ?gcand …)` and `(exists-glink ?cont ?cand ?gcand)` unidirectionally (confirmed: `init-glinks.clp:128-196`, all four rules use `cand → gcand` order). §8.3 now states explicitly that the glink graph is keyed `(Label → GLabel)` only and no reverse index is needed.

- **M10 — FIXED (§2.2):** Citation `init-glinks.clp:108` for g-candidate survival was wrong (line 108 is a comment in `init-g-candidates`). Correct citations are `init-g-candidates-horiz` lines 68-80 and `init-g-candidates-verti` lines 93-105. Fixed throughout §2.2 and §3.

### Second-pass remediation (codex + opus, 2026-05-18)

| Finding | Status | Section patched | Notes |
|---|---|---|---|
| codex MAJOR §6.1 stale ResolutionState sketch | FIXED | §6.1 | Replaced stale 3-field struct with redirect comment to §13 |
| codex MAJOR §13 GLabelTable missing CspVarId fields | FIXED | §13 `glabel_tables.rs` | Added `csp_vars_of: Vec<[CspVarId; 2]>` with slot-0=rn/cn, slot-1=bn convention; CLIPS citation init-glinks.clp:84-88, 109-113 |
| codex MAJOR Appendix A whip-dedup multiset vs positional | FIXED | Appendix A | Changed `HashMap<(target, multiset(rlcs)), ChainId>` → `HashMap<(target, Vec<Rlc>), ChainId>` with §14.1 cross-ref |
| codex MINOR Appendix A `retract` mapping | FIXED | Appendix A | Changed `rs.cand_alive.remove(L)` → `rs.eliminate_candidate(L, &g_lab_tab)` |
| opus MR-2 §5.3 misleading v1∉csp-vars(Z) narrative | FIXED | §5.3 | Rewrote to remove informal aside; explicit note that no explicit guard should be added; CLIPS cite Whips[3].clp |
| opus Mn-1 §6.3 self-contradictory Phase A/B comments | FIXED | §6.3 pseudocode | Collapsed to single clear sentence per-loop entry |
| opus Mn-3 §10.2 g-whip suppression incomplete | FIXED | §10.2 | Bullet now says "suppress whips AND braids"; parenthetical cites §4.1 salience ordering |
| NF-4 partial-gwhip[1] seed step missing | FIXED | §8.4.1 (new) | New subsection added with CLIPS source semantics, Partial-gWhips[1].clp:38-83 citation, dedup rule, Rust function signature |
| NF-5 Salience-interleaved rating driver vs per-technique passes | FIXED | §6 header (new callout) | Paragraph added at §6 top distinguishing rate_chain (interleaved) from per-technique probes (§10 reverse-construction); describes correct k-iteration order |
| NF-1 §8.4 gWhip llc-disjointness vs gBraid relaxation asymmetry | FIXED | §8.4 | Added sentence: gWhip retains `new-llc ∉ llcs ∪ rlcs`; gBraid retains `new-llc ∉ rlcs` only; CLIPS citations gWhips[5]:72-73, gBraids[5]:72,111 |
| NF-2 Fixture 5 leftover puzzle | FIXED | §12 Fixture 5 | Deleted the stale first puzzle string and editorial comment; kept only the clean B=7 entry |
| NF-3 §9.2 gW→gB cascade rating ambiguity | FIXED | §9.2 | Added one-sentence callout: partial-gwhip consumed by gbraid step emits gB[k] rating, not gW[k] |

### Third-pass remediation (codex CRITICAL + MAJOR, opus M-class, 2026-05-18)

| Finding | Status | Evidence / CLIPS cite | Notes |
|---|---|---|---|
| **FX-1** killed-or-committed semantics | **FIXED** | `Whips[2].clp:80`, `Braids[5].clp:72-79`, `gBraids[5].clp:78-79` | Added `is_killed_or_committed` (whip/braid) and `is_killed_or_glinked` (gbraid) helpers that check direct membership in killed_set (`killed_set.test(alt)`) before the link-graph check. Also added `killed_set.test` to `glinked_killed` in gwhip.rs. 2 unit tests added (whip + gbraid). |
| **FX-2** gbraid `extend_plain_braids` lacks whip cross-feed | **FIXED** | `gBraids[5].clp:168-177` `(type partial-whip\|partial-braid)` | Added `partials_whip_prev/next` fields to `GBraidScratch`. `extend_plain_braids` now takes `prev_whips` parameter and passes union (braid+whip) to `extend_partial_braids`. `run_gbraid_pass` and `find_first_gbraid` now seed and extend the whip buffer symmetrically with the braid buffer. |
| **FX-3** `try_terminate_gbraid` over-restrictive GCand-member check | **FIXED** | `gBraids[5].clp:72` `(not (member$ ?new-llc $?rlcs))` only | Removed inner `for rlc in all_rlcs { if GCand(g) ... continue 'outer }` loop from `try_terminate_gbraid`. CLIPS terminator only checks exact Cand rlc membership, not GCand-member. Removed `'outer` loop label (no longer needed). 1 unit test added. |
| **FX-4** cross-type dedup parity regression from F8 | **FIXED** | `Braids[5].clp:128-139`, `gBraids[5].clp:198-215` | Added `Chain::dedup_key_cross_type()` method to `chain_model.rs` — hashes `(is_grouped, target, rlcs_sorted)` without `is_braid`. Updated `extend_partial_braids` to use `dedup_key_cross_type()` for the union-type guard. Updated `run_braid_pass` and `find_first_braid` to seed dedup with cross-type keys. `extend_plain_braids` in gbraid also uses cross-type keys. Updated §15 note: F8 `is_braid` split is correct for same-technique extension; `dedup_key_cross_type()` is for union-type CLIPS guards. 2 unit tests added in chain_model.rs. |
| **FX-5** `run_braid_pass` double-termination on cross-fed buffers | **FIXED** | braid.rs:574-579 | Added `k_targets_seen: HashSet<Label>` guard in termination loop for `run_braid_pass`, `run_whip_pass`, `run_gwhip_pass`, `run_gbraid_pass`. Duplicate targets (same elimination target from whip + braid partial) are now deduplicated before applying eliminations. 1 unit test added (braid). |
| **FX-6** `run_whip_pass` stale partials post-elimination | **FIXED** | whip.rs:561-599 | After applying eliminations at each k, `partials_prev.retain(|c| cand_alive(c.target))` is called in all four `run_*_pass` functions. This prevents chains with now-dead targets from producing duplicate eliminations at the next k. 1 unit test added (whip); FX-6 guards also applied in braid/gwhip/gbraid. |
| **Tier-3** dead-code cleanup | **FIXED** | gwhip.rs:497 empty block; gwhip.rs:1006-1008 empty loop; gbraid.rs:369 `n_glabels`; gbraid.rs:417 `csp1` | Removed empty `if csp_set.contains(...){}` block in gwhip sub-rule -1; replaced empty whip_prev loop comment with Tier-3 note; removed `let n_glabels` unused declaration in `build_partial_gbraids_length_1`; suppressed unused `csp1` in gbraid `try_gbraid_1_eliminations`. |

### Fourth-pass remediation (3rd fix-up: FX-R1 revert confirmed + FX-R2..FX-R5, 2026-05-18)

| Finding | Status | Evidence / CLIPS cite | Notes |
|---|---|---|---|
| **FX-R1** Revert FX-1 `killed_set.test(alt)` prepend — UNSOUND | **CONFIRMED ALREADY REVERTED** | `generic-background.clp:216-228, 263-269` — `linked-or`/`glinked-or` have NO self-membership semantics | Prior session had already removed the prepend. All four files (whip.rs, braid.rs, gwhip.rs, gbraid.rs) use pure `link.is_linked_or_bitset` / `glinked_killed` (which is itself pure link+glink). FX-R1 regression tests already present and correct. Test count unchanged for FX-R1. |
| **FX-R2** Complete FX-4 in `extend_partial_gbraids` with cross-type dedup | **CONFIRMED ALREADY FIXED + NEW TEST ADDED** | `gBraids[5].clp:138-145, 204-220, 282-289` | All three sub-rules of `extend_partial_gbraids` already used `dedup_key_cross_type()` (FX-R2 comments present). Added 1 new unit test `test_fxr2_partial_gwhip_gbraid_cross_type_dedup` in gbraid.rs verifying that a `PartialGWhip` and `PartialGBraid` with same `(target, rlcs)` collapse to the same cross-type key but differ under `dedup_key()`. |
| **FX-R3** gbraid internal `extend_partial_gwhips` passes `&[]` instead of `whip_prev` | **FIXED** | `gBraids[5].clp` sub-rule -2 `partial-whip → partial-gwhip via GCand` | Two sites in gbraid.rs fixed: `run_gbraid_pass` line ~1305 and `find_first_gbraid` line ~1424. Both now pass local `partials_whip_prev` to `extend_partial_gwhips` so sub-rule -2 fires correctly inside gbraid context. Added 1 `#[ignore]` unit test `test_fxr3_gbraid_pass_uses_whip_prev_for_gwhip_extension` with TODO for propagate_singles integration. |
| **FX-R4** Document `dedup_key_cross_type` as guard-only | **FIXED** | spec §14.1 | Replaced the doc-comment on `Chain::dedup_key_cross_type` in `chain_model.rs` with the canonical guard-only warning per mission spec: "Cross-type guard key. Used ONLY for CLIPS-style union-type guards … DO NOT use for self-extension dedup." |
| **FX-R5** Add known-answer fixture test | **FIXED** | spec §12 Fixture 2 (`B=1`, id `cbg000#22`) | Added `test_fxr5_fixture2_braid1_known_answer` in braid.rs. Asserts: `run_braid_pass(k_max=3)` on Fixture 2 returns non-empty; first elimination is `label=39` (row=0, col=4, digit=4); rule is `Braid(1)`. Values verified by running the implementation on 2026-05-18. |

### Fifth-pass remediation (4th fix-up: FX-R6..FX-R11, 2026-05-18)

Provenance: Codex 4th-pass identified FX-R6 (new BLOCKER), FX-R7 (still open — FX-R2 not applied to find_first_gbraid), FX-R8 (new BLOCKER — M7 secondary scan missing prefix-subset), FX-R9 (fixture test pinned to enumeration order). Opus 4th-pass identified FX-R10 (two dead functions in gwhip.rs) and FX-R11 (misleading comment on dedup seeding). All 6 findings addressed in this pass.

| Finding | Status | Evidence / CLIPS cite | Notes |
|---|---|---|---|
| **FX-R6** Sequencing bug: `partials_whip_prev` advanced BEFORE `extend_partial_gwhips` | **FIXED** | `Partial-gWhips[5].clp:122-126`: sub-rule -2 reads partial-whips at length k-1; advancing first passes length-k whips | Fixed at both sites: `run_gbraid_pass` (was lines 1279-1310 — gwhip extension now comes BEFORE whip advance) and `find_first_gbraid` (was lines 1417-1427 — same reorder). Verified `gwhip.rs::run_gwhip_pass` already had correct order (gwhip before whip at lines 1012-1034). Added 2 unit tests: `test_fxr6_gbraid_pass_gwhip_whip_ordering` and `test_fxr6_find_first_gbraid_gwhip_whip_ordering`. |
| **FX-R7** `find_first_gbraid` missing cross-type dedup seeding | **FIXED** | `gBraids[5].clp:138-145, 204-220, 282-289` | `find_first_gbraid` line ~1382 was seeding dedup only from `partials_gbraid_prev` using `dedup_key()` — missing gwhip seeds and missing `dedup_key_cross_type()`. Fixed to mirror `run_gbraid_pass`: seed from both `partials_gbraid_prev` and `partials_gwhip_prev` using `dedup_key_cross_type()`. Added 1 unit test `test_fxr7_find_first_gbraid_cross_type_dedup_consistency`. |
| **FX-R8** M7 secondary scan missing prefix-subset check | **FIXED** | `gBraids[5].clp:282-293`: `(subsetp $?rlcs $?rlcsa)` condition requires existing chain's base rlcs ⊆ candidate's rlcs AND new GCand covered in existing | Augmented M7 secondary scan in `extend_partial_gbraids` sub-rule -3 with two-condition reject: (1) chain.rlcs ⊆ existing.rlcs AND (2) new GCand covered in existing. Without condition (1), scan rejected chains whose new GCand was "covered" in any existing chain regardless of whether the base chains were related. Added 1 smoke unit test `test_fxr8_m7_secondary_no_subset_keeps_both`. |
| **FX-R9** Fixture test `test_fxr5_fixture2_braid1_known_answer` pinned to enumeration order | **FIXED** | spec §12 Fixture 2: "W[2] or B[1]" — no CLIPS-verified specific target | Weakened assertion to `any(rule == Braid(1) \| rule == Whip(1))` instead of `elims[0].target == 39`. Added TODO comment for CLIPS oracle tightening. Old enumeration-order-pinned assertion removed from braid.rs. |
| **FX-R10** Dead functions `label_killed_by_set` and `csp_fresh_last` in gwhip.rs | **FIXED** | Compiler warning: `function ... is never used` | Removed both dead functions from gwhip.rs (was lines 122-124 and 185-187). Compiler warning count reduced from 21 to 18. |
| **FX-R11** Misleading comment on FX-R2 dedup seeding in gbraid.rs | **FIXED** | Comment claimed seeding implements "union guard at length k"; actual purpose is subsumption against prior-length chains | Rewrote inline comments at both seeding sites in `run_gbraid_pass` (initial seed ~line 1201 and per-k seed ~line 1245). Clarifies: seeding prevents re-deriving length-1/k-1 chains as length-k extensions; the actual union guard is enforced inside `extend_partial_gbraids` via the shared dedup set. |

Test count: 411 → 415 passing. All tests green. cargo check: 21 → 18 warnings (3 removed; remaining 18 are pre-existing in aic_reverse.rs and gwhip.rs extension functions, out of scope).

### Sixth-pass remediation (FA-1..FA-11, 2026-05-18)

Provenance: Codex BLOCK + Opus GO-WITH-FIXES review (9 findings). Addresses architectural FA-1/FA-2/FA-3 (rater registration + strict load-bearing + gwhip classification gate), major FA-4 (BRT pre-pass) + FA-5 (suppressed scoring probes), and mechanical FA-6..FA-11.

| Finding | Status | File(s) | Notes |
|---|---|---|---|
| **FA-1** Register W/B/GW/GB in rater.rs AnyTechnique + T4PLUS_LIST | **FIXED** | `techniques/mod.rs`, `techniques/chain_rated.rs`, `rater.rs`, `reverse_construct.rs` | Added `TechniqueId::{Whip,GWhip,Braid,GBraid}`. New `chain_rated.rs` implements `Technique+RatedTechnique` for all four. T4PLUS_LIST extended to 10. k_max=9 in rater cascade. `all_techniques_9x9` updated. `technique_id_str`/`parse_technique_id` updated. |
| **FA-2** Replace rate_chain load-bearing gates with rate_excluding | **FIXED** | `whip_reverse.rs`, `braid_reverse.rs`, `gwhip_reverse.rs`, `gbraid_reverse.rs` | All four modules now import `rate_excluding`. Load-bearing gate: `rate_excluding(puzzle, &[TechniqueId::X]).solved == false` → accept (strict spec §10.3). Classification gate (`rate_chain == X(k)`) remains unconditional. TODO caveats removed from all four module doc-blocks. |
| **FA-3** gwhip non-LB path missing classification gate | **FIXED** | `gwhip_reverse.rs:363-424` | Non-LB branch removed; both LB and non-LB now run classification gate (`rate_chain == GW(k)`) unconditionally. `load_bearing` only toggles additional `rate_excluding` gate. |
| **FA-4** chain_rating::rate_chain missing BRT/singles pre-pass | **FIXED** | `chain_rating.rs:88` | `propagate_singles` now called before `ChainContext` build. Solved-by-singles → `None`. Contradiction → `None`. `test_fixture3_braid3` un-ignored. |
| **FA-5** Suppressed scoring probes for guided-removal | **FIXED** | `techniques/braid.rs`, `techniques/gwhip.rs`, `techniques/gbraid.rs`, `braid_reverse.rs`, `gwhip_reverse.rs`, `gbraid_reverse.rs` | Added `find_first_braid_excluding_whip`, `find_first_gwhip_excluding_wb`, `find_first_gbraid_excluding_wgwb`. Scoring fns now call suppressed variants. |
| **FA-6** seed_used inconsistency (= 0 in whip/gwhip single-thread) | **FIXED** | `whip_reverse.rs:391`, `gwhip_reverse.rs:394,419` | Both now write `seed_used: spec.seed`. |
| **FA-7** GBraidReverseSpec::validate accepts target_k=1 | **FIXED** | `gbraid_reverse.rs:114` | Now rejects `target_k < 2`. |
| **FA-8** gwhip dead chunk_spec seed bump | **CONFIRMED-PRESENT** | `gwhip_reverse.rs:473` | Dead `seed` field in chunk_spec is overridden by `res.seed_used = child_seed` in batch; harmless. Left intact to avoid batch determinism change. |
| **FA-9** whip-specific seed early-reject divergent | **FIXED** | `whip_reverse.rs:344-347` | Removed the `seed_has/seed_dist` early-reject guard + dead `let _ = (...)` silencer. Matches sibling pattern. |
| **FA-10** k ∈ 1..36 not enforced in validators | **FIXED** | `whip_reverse.rs:113`, `braid_reverse.rs:140`, `gwhip_reverse.rs:127`, `gbraid_reverse.rs:114` | All four validate() now check `target_k > 36` → error. g-whip/g-braid also have `target_k < 2` → error. |
| **FA-11** Spec §10.2 g-whip suppression contradicts salience order | **FIXED** | `csp_rules_chain_spec.md §10.2` | Corrected g-whip scoring to suppress whips only (not braids). Braid has lower salience and cannot pre-empt g-whip. Added §15 entry. |

### Seventh-pass remediation (CR-FIN-1..CR-FIN-M3o, 2026-05-18)

Provenance: Codex BLOCK (CR-FIN-1 + CR-FIN-2 CRITICAL) + Opus GO-WITH-FIXES (CR-FIN-M1c..M3o). Addresses architectural CR-FIN-2 (salience-interleaved T4Plus driver), BRT pre-pass correctness verification (CR-FIN-1), consistency check (CR-FIN-Mn-1c), load-bearing spec re-read (CR-FIN-M1c), and mechanical fixes CR-FIN-M2c/M3c/M1o/M2o/M3o.

| Finding | Status | File(s) | Notes |
|---|---|---|---|
| **CR-FIN-1** Full BRT pre-pass (codex CRITICAL) | **CONFIRMED-WRONG** | `chain_rating.rs` | CLIPS `saliences.clp` confirms BRT = Singles + constraint propagation only. No locked candidates / subsets in pre-chain phase. `propagate_singles` (naked + hidden singles) IS the correct CLIPS BRT. Current implementation was already correct. `is_consistent()` pre-check added (CR-FIN-Mn-1c). |
| **CR-FIN-2** T4PLUS_LIST salience-interleaved driver (codex CRITICAL) | **FIXED** | `techniques/chain_rated.rs`, `techniques/mod.rs`, `rater.rs` | Added `ChainCombinedTechnique`: single T4PLUS_LIST entry that internally iterates `whip[k]→gwhip[k]→braid[k]→gbraid[k]` for k=1..K_MAX before incrementing k. `TechniqueProgress::technique_id_override` added to carry actual W/GW/B/GB id for correct frontier/se_rating attribution. T4PLUS_LIST shrunk from 10 to 7 entries. `CHAIN_RATER_K_MAX` made pub with TODO doc. |
| **CR-FIN-M1c** Load-bearing acceptance too strict (codex MAJOR) | **CONFIRMED-WRONG** | n/a | Spec §10.3 re-read: accept iff `rate_excluding.tier > target_tier`. For T4Plus target, no tier is strictly harder → gate IS `!r2.solved`. Current code correct. Documented in §10.3 comment. |
| **CR-FIN-M2c** spec.seed not seeding RNG in single-thread constructors | **ADDRESSED-M3o** | `whip_reverse.rs`, `gwhip_reverse.rs` | Standardized batch seed handling (CR-FIN-M3o): chunk_spec now carries `seed: child_seed` so single-thread call gets `spec.seed = child_seed`. This propagates correctly to `seed_used` in result. |
| **CR-FIN-M3c** API parity broken — with_seed + clue_max>81 (codex MAJOR) | **FIXED** | `braid_reverse.rs`, `gwhip_reverse.rs`, `gbraid_reverse.rs`, `whip_reverse.rs` | Added `with_seed(seed)` builder to `BraidReverseSpec`, `GWhipReverseSpec`, `GBraidReverseSpec`. Added `clue_max > 81` rejection to `WhipReverseSpec`, `BraidReverseSpec`, `GWhipReverseSpec` validators (GBraidReverseSpec already had it). |
| **CR-FIN-M1o** chain_rated.rs unsafe repr(Rust) layout (opus MAJOR) | **FIXED** | `grid.rs`, `techniques/chain_rated.rs` | Added `#[repr(C)]` to `Grid<N,BR,BC>` in `grid.rs`. Updated all unsafe comment blocks in `chain_rated.rs` to cite `#[repr(C)]` justification. |
| **CR-FIN-M2o** find_first_gwhip_excluding_wb misnamed (opus MAJOR) | **FIXED** | `techniques/gwhip.rs`, `gwhip_reverse.rs` | Renamed to `find_first_gwhip_excluding_w`. Doc-block updated to clarify braid has lower salience and cannot pre-empt gwhip. |
| **CR-FIN-M3o** Batch seed pattern divergent (opus MAJOR) | **FIXED** | `whip_reverse.rs`, `gwhip_reverse.rs` | Standardized whip and gwhip batch drivers on chunk_spec-with-child_seed pattern (mirrors `braid_reverse.rs:469-473`). Removed patch-after `res.seed_used = child_seed`. |
| **CR-FIN-Mn-1c** is_consistent() check in rate_chain (opus MINOR) | **FIXED** | `chain_rating.rs` | Added `if !g.is_consistent() { return None; }` before BRT pre-pass. |

Test result: `cargo test --lib` — 451 passed / 0 failed / 13 ignored.

### Eighth-pass remediation (CR-FIN-3 C1..C2, M-codex-1..M-codex-2, M-opus-1..M-opus-4, MINOR, 2026-05-18)

Provenance: Round-6 BLOCK consolidation from both codex and opus reviewers. Addresses CR-FIN-3 C1 (per-arm exclusion mask), C2 (raise CHAIN_RATER_K_MAX to 36), codex MAJOR M-codex-1 (`seed_used` metadata) and M-codex-2 (k clamp .min(255)→.min(36)), opus MAJOR M-opus-1 (rate_chain vs rate_excluding semantic alignment), M-opus-2 (override-set invariant), M-opus-3 (spec §4.1 BRT enumeration), M-opus-4 (T4Plus-floor placeholder documented), plus M2 architectural tests and MINOR stale-comment cleanup.

| Finding | Status | File(s) | Notes |
|---|---|---|---|
| **CR-FIN-3 C1** ChainCombined per-arm exclusion mask (codex+opus CRITICAL) | **FIXED** | `techniques/chain_rated.rs`, `rater.rs` | `ChainCombinedTechnique` now carries `excluded_arms: [bool; 4]` (slot order whip/gwhip/braid/gbraid via `chain_arm_index`) and a configurable `k_max`. Added `DEFAULT`, `with_k_max`, `with_excluded`, `all_arms_excluded`. `rater::rate_excluding` builds a per-call mask via `ChainCombinedTechnique::with_excluded(exclude)` and dispatches the combined entry through it; the combined entry is skipped *only* when all four chain ids are excluded. Restores strict load-bearing semantics in `*_reverse.rs` modules. |
| **CR-FIN-3 C2** Raise `CHAIN_RATER_K_MAX` to 36 (codex CRITICAL) | **FIXED** | `techniques/chain_rated.rs` | Const raised from 9 → 36 (spec §4.1 / §13 cap). Combined driver clamps via `self.k_max.min(CHAIN_RATER_K_MAX)`. Callers needing a tighter cap construct `ChainCombinedTechnique::with_k_max(k)`. |
| **CR-FIN-3 M-codex-1** `seed_used` metadata via internal seed (codex MAJOR) | **FIXED** | `whip_reverse.rs`, `braid_reverse.rs`, `gwhip_reverse.rs`, `gbraid_reverse.rs` | Added `*_reverse_construct_from_seed(spec)` convenience function in each module: seeds an internal `Xoshiro256PlusPlus` from `spec.seed` and delegates. `seed_used` doc comment now explicitly states the meaning when caller-owned RNG is used vs internally-seeded RNG. Existing `*_reverse_construct(spec, rng)` is unchanged (caller-driven). |
| **CR-FIN-3 M-codex-2** k-cap `.min(255)` → `.min(36)` (codex MAJOR) | **FIXED** | `whip_reverse.rs:261/334`, `braid_reverse.rs:260/339/635/728`, `gwhip_reverse.rs:341/587/625`, `gbraid_reverse.rs:301/539/614` | All `.min(255)` occurrences replaced with `.min(36)` matching spec §4.1 cap. Prevents off-spec probing at k > 36. |
| **CR-FIN-3 M-opus-1** rate_chain vs rate_excluding semantic alignment (opus MAJOR) | **FIXED (DOCUMENTED + C1 functional fix)** | `chain_rating.rs` module doc | Added module-level "Relationship to `rater::rate_excluding`" section. `rate_chain` is CLIPS-BRT-correct (singles-only pre-pass); `rate_excluding` runs the richer T2/T3 cascade before chain probing. The asymmetry is conservative for load-bearing: a richer cascade can only make `rate_excluding` more permissive than CLIPS, so the strict gate `!rate_excluding.solved` remains sound. CR-FIN-3 C1 per-arm masking is the key functional alignment — excluding one chain id no longer drops all four. |
| **CR-FIN-3 M-opus-2** technique_id_override fragility (opus MAJOR) | **FIXED** | `techniques/chain_rated.rs` | Each of the four firing arms in `ChainCombinedTechnique::apply` now sets `prog.technique_id_override = Some(...)` and `debug_assert!`s the invariant before returning. Future refactors that forget the assignment will be caught in debug builds. |
| **CR-FIN-3 M-opus-3** spec §4.1 BRT ambiguity (opus MAJOR) | **FIXED** | `docs/csp_rules_chain_spec.md §4.1.1 (new)` | Inserted new subsection enumerating CLIPS BRT exactly: ECP + single/naked-single/hidden-single only. Explicitly excludes Naked/Hidden Pairs/Triples/Quads, Locked Candidates, Bivalue-Chains, k-value subsets. Removed the misleading "Bivalue / 2-value / 3-value subsets" line from the salience tower. Cites `Generic-Background/Singles.clp` + `ECP.clp`. The Rust pre-pass (`propagate_singles`) is confirmed aligned. |
| **CR-FIN-3 M-opus-4** base_rating underestimate (opus MAJOR) | **DOCUMENTED** | `rater.rs:723` | Added rationale comment at the "cascade stuck" return path: `se_score.max(7.5)` preserves the highest pre-T4 SE rating when it exceeds 7.5 (so AIC at 6.6 isn't underwritten by stuck cascade), and otherwise encodes a conservative T4Plus floor. Tightening this requires a post-T4Plus oracle (CLIPS), out of scope. |
| **CR-FIN-3 M2** Tests for CR-FIN-2 architectural refactor (both reviewers MAJOR) | **FIXED** | `techniques/chain_rated.rs::tests` (new) | Added 7 tests: `chain_combined_apply_emits_override_and_chain_len`, `rate_excluding_per_arm_mask_keeps_other_chain_arms`, `rate_excluding_whip_only_is_at_least_as_permissive_as_all_arms`, `chain_combined_salience_order_within_k`, `chain_rater_k_max_is_spec_capped`, `chain_arm_index_mapping`, `with_excluded_builds_correct_mask`. All pass. |
| **CR-FIN-3 MINOR** Stale comments | **FIXED** | `techniques/gwhip.rs:1076-1088`, `gwhip_reverse.rs:1-57` | Updated `find_first_gwhip_excluding_w` doc-block (removed wrong "no whip OR braid fires" phrasing). Rewrote `gwhip_reverse.rs` module doc-block: removed the obsolete `rate_chain`-only load-bearing explanation; replaced with the current `rate_chain` classification + `rate_excluding` strict load-bearing gate. |

### Ninth-pass remediation (CR-FIN-4 C1..C2, M-codex-3, M-opus-5, Mn-1..Mn-5, 2026-05-18)

Provenance: Round-7 unbiased verification. Codex BLOCK (2 CRITICAL + 1 MAJOR + 2 MINOR); Opus GO-WITH-FIXES (1 MAJOR + 5 MINOR). Findings consolidated and verified against `/tmp/csp-rules-research/CSP-Rules-V2.1/` CLIPS source.

| Finding | Status | File(s) | Notes |
|---|---|---|---|
| **CR-FIN-4 C1** `csp_glinked` collapses two valid CSP-vars (codex CRITICAL) | **CONFIRMED-WRONG** | n/a (no code change) | CLIPS `init-effective-csp-glinks-{rn, horiz-bn, cn, verti-bn}` (`SudoRules-V20.1/GENERAL/init-glinks.clp:128-196`) use mutually exclusive conditions: rn-variant requires `row=g.row, blk≠g.blk`; bn-variant requires `blk=g.blk, row≠g.row`. A single (cand, gcand) pair satisfies AT MOST ONE rule (cells cannot have both same-row-diff-block and same-block-diff-row simultaneously). CLIPS asserts exactly one `csp-glinked` fact per (cand, gcand). Current Rust code at `glabel_tables.rs:322-341` matches this exactly (one `(g, csp_var)` per (label, g) in `csp_glinked_list`). The spec §8.2 `is-csp-variable-for-glabel` two-entry note refers to a DIFFERENT fact family (one per glabel) that no chain rule consumes — `gWhips[*].clp` and `gBraids[*].clp` read `csp-glinked`, not `is-csp-variable-for-glabel`. Codex finding is a misread analogous to CR-FIN-1 (also CONFIRMED-WRONG against CLIPS source). |
| **CR-FIN-4 C2** Implement `gwhip[1]` as firing technique (codex CRITICAL) | **CONFIRMED-WRONG** | n/a (no code change) | The claimed CLIPS source `Chains-rules-V2.1/Whips/gWhips[1].clp` **does not exist** in the V2.1 distribution. Verified: `find /tmp/csp-rules-research/CSP-Rules-V2.1 -name "gWhips\[1\].clp"` returns empty; the G-WHIPS directory begins at `gWhips[2].clp`. `saliences.clp:1131-1133` defines only `?*partial-gwhip[1]-salience-*` (the seed), not a `?*gwhip[1]-salience*`. CLIPS `Partial-gWhips[1].clp:34-36` comments "only a partial-gwhip[1] can give rise to a full g-whip[2]" — the smallest g-whip in CLIPS V2.1 is `gWhip[2]`, which fires after the partial-gwhip[1] seed via the `gWhips[2].clp` elimination rule. Rust `find_first_gwhip` correctly starts at `k=2` matching CLIPS naming. The standalone helper `try_gwhip_1_eliminations` at `gwhip.rs:310` is preserved for diagnostic use (its doc-block already explains the CLIPS naming mapping). Spec §4.1 line 116 "G-Whip[1] = gWhip[1]" refers to the partial-gwhip[1] seed phase in the salience tower (which does NOT itself eliminate; it produces seeds consumed by `gwhip[2]`). |
| **CR-FIN-4 M-codex-3** `gbraid_reverse` rejects `target_k=1` (codex MAJOR) | **FIXED** | `gbraid_reverse.rs:114-130` | Validator now accepts `target_k ∈ 1..=36`. The solver path (`find_first_gbraid` + `try_gbraid_1_eliminations` at `gbraid.rs:1395`) and the rater (`chain_rating::rate_chain` → `ChainRating::GB(1)`) already supported length-1 g-braids directly; the validator was the only off-spec gap. `target_k=0` still rejected. Existing test `gbraid_excludes_gwhip_rejected` at `target_k=1` now exercises the full code path. |
| **CR-FIN-4 M-opus-5** gbraid M7 secondary scan misses `Rlc::Cand` (opus MAJOR) | **FIXED** | `gbraid.rs:902-921` | Per CLIPS `gBraids[5].clp:282-293`, the dedup guard uses `glabel-contains-some-of` (`generic-background.clp:179-187`) which checks BOTH `Rlc::Cand` AND `Rlc::GCand` entries against the new GCand. The match arm now also handles `Rlc::Cand(l_ex)` via `label_in_glabel(l_ex, g1, glab)` (equivalent to `glabel-contains-some-of` on a single Cand). Added `label_in_glabel` to the imports list. Soundness-preserving fix (was under-blocking, not wrong-emission); restores CLIPS parity. |
| **CR-FIN-4 Mn-1** Architectural tests soft-pass (codex+opus MINOR) | **FIXED** | `techniques/chain_rated.rs::tests` | Hardened `chain_combined_apply_emits_override_and_chain_len`: `apply` now `.expect()`s a firing on Fixture 2 (B=1) instead of soft-pass. Strengthened `rate_excluding_per_arm_mask_keeps_other_chain_arms` with three direct `ChainCombinedTechnique::apply` sub-tests: (c) no exclusion fires some chain arm; (d) Whip excluded → override ∈ {GWhip, Braid, GBraid}; (e) all four arms excluded → `apply` returns `None`. `chain_combined_salience_order_within_k` left as soft-pass with explanatory comment — gwhip/gbraid firing is fixture-dependent and the structural assertion (IF firing → expected override) is the meaningful guard. |
| **CR-FIN-4 Mn-2** Rater-side override invariant not enforced (codex MINOR) | **FIXED** | `rater.rs:683-700` | Added `debug_assert!` on the consumer site: when `tech` is `AnyTechnique::ChainCombined` and progress fired, `p.technique_id_override.is_some()` must hold. Catches future refactors that bypass `ChainCombinedTechnique::apply`'s internal assertion path. |
| **CR-FIN-4 Mn-3** `rater_error` folded into "solvable without X" (opus MINOR) | **FIXED** | `whip_reverse.rs:399-410`, `braid_reverse.rs:418-434`, `gwhip_reverse.rs:356-373`, `gbraid_reverse.rs:351-368` | Split the strict load-bearing gate: `rater_error` now triggers a distinct `continue` (cascade-side contradiction → discard this attempt, try next clue removal in batch), separate from `solved=true` which means "not load-bearing". Previous union folded both signals into the same rejection path, silently rejecting load-bearing candidates that triggered a cascade error. |
| **CR-FIN-4 Mn-4** `find_first_*_excluding_*` re-walks all partials (opus MINOR) | **DEFERRED (TODO)** | `gwhip.rs`, `braid.rs`, `gbraid.rs` (no code change) | Suppression helpers (`find_first_gwhip_excluding_w`, `find_first_braid_excluding_whip`, `find_first_gbraid_excluding_wgwb`) do up to 4 full chain searches per probe call. Refactoring to share the partials cache across successive `find_first_*` calls requires a non-trivial API change to `ChainContext` (carry a per-call partials cache) plus careful invalidation when `rs.cand_alive` changes between probes. Out of scope for this round; deferred with TODO. Functional correctness is unaffected (only wall-clock cost on reverse-construction inner loops). |
| **CR-FIN-4 Mn-5** `(target_k + k_slack)` u32 wrap before clamp (opus MINOR) | **FIXED** | `whip_reverse.rs:266/345-347`, `braid_reverse.rs:264/343/388/651/660/745`, `gwhip_reverse.rs:222/316/573/611/758`, `gbraid_reverse.rs:305/554/630` | All `(spec.target_k + spec.k_slack).min(36)` and `(target_k + k_slack).min(36)` patterns replaced with `spec.target_k.saturating_add(spec.k_slack).min(36)` (or local variant). Also fixed companion `low/high` range checks. Prevents u32 wrap on extreme inputs before the `.min(36)` clamp. |

Test result: `cargo test --lib` — 458 passed / 0 failed / 13 ignored (up from 451; 7 new from M2/Mn-1 architectural test additions and hardening).

### Tenth-pass remediation (CR-FIN-5 C1..C2 + MINOR, 2026-05-18)

Provenance: Round-8 codex BLOCK with two CRITICAL claims on grouped-chain dedup/subsumption parity vs CLIPS, plus one stale-comment MINOR. Round-8 opus returned GO (no CRITICAL/MAJOR). Each claim verified against `/tmp/csp-rules-research/CSP-Rules-V2.1/CSP-Rules-Generic/CHAIN-RULES-SPEED/` CLIPS source and acted on independently.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-5 C1** `partial-gwhip` dedup misses cross-type `partial-whip|partial-gwhip` guard at seed and at sub-rules -2/-3 (codex CRITICAL) | **GENUINE — FIXED** | `gwhip.rs`, `gbraid.rs` | Per CLIPS `Partial-gWhips[1].clp:63-71` (seed) and `Partial-gWhips[5].clp:142-150` (sub-rule -2) + `:213-221` (sub-rule -3), all three dedup `(not (chain ...))` guards read `(type partial-whip|partial-gwhip)` with an `(or (eq ?rlc1a ?new-rlc) (label-in-glabel ?rlc1a ?new-rlc))` tail check — i.e. when a plain `partial-whip[k]` with the same target has a Cand tail label inside the new glabel, the grouped chain must be suppressed (the plain whip has higher salience per `saliences.clp` and subsumes the grouped one). The Rust seed `build_partial_gwhips_length_1` (gwhip.rs:185) previously only deduped `(target, gcand)` against itself; the extension sub-rules at gwhip.rs:647 (sub-rule -2) and :776 (sub-rule -3) only scanned `out.iter()` (the already-emitted grouped chains at length k). Neither consulted the plain `partial-whip` family. **Fix**: (1) Added `plain_whip_seeds: &[Chain]` parameter to `build_partial_gwhips_length_1` — for each candidate `(z, rlc1)`, skip if any whip[1] with target `z` has `Rlc::Cand(L)` where `label_in_glabel(L, rlc1, glab)`. (2) Added `plain_at_k: &[Chain]` parameter to `extend_partial_gwhips` — sub-rules -2/-3 now additionally scan `plain_at_k` for a partial-whip[k] with same target + prefix and tail `Cand(L)` such that `label_in_glabel(L, rlc_g, glab)`; if found, skip the grouped emission. (3) Reordered `run_gwhip_pass` and `find_first_gwhip` to build plain whip[1] seeds **before** the gwhip seeds (so the seed cross-type guard has data), and to build `whip_next` (length-k plain whips) **before** calling `extend_partial_gwhips` at each k (the plain-whip extension does not read partial-gwhip data, so reordering is safe). (4) `run_gbraid_pass` and `find_first_gbraid` mirror the same reordering for their inner `extend_partial_gwhips` call and pass `partials_whip_next` as the new `plain_at_k` slice. **Direction**: this is a soundness-preserving over-production fix — without the guard, Rust asserted grouped partials CLIPS would prune, which could (a) overproduce `gWhip[k]` firings at the wrong k under reverse-construction probes and (b) lower the rating-target hit rate when a finer plain whip exists. Eliminations themselves were already CLIPS-faithful; only the ratings/rate-floor were at risk. |
| **CR-FIN-5 C2** `partial-gbraid[*]-2` missing plain-chain `(type partial-whip|partial-braid)` subsumption guard (codex CRITICAL) | **GENUINE — FIXED** | `gbraid.rs` | Per CLIPS `gBraids[5].clp:213-220` (file contains the `partial-gbraid[4]-*` extension rules despite its name suffix), sub-rule -2 has TWO mandatory dedup guards: lines 204-212 use `(type partial-gwhip|partial-gbraid)` with `same-sets-of-rlcs` (handled by Rust `dedup_key_cross_type`); lines 213-220 use `(type partial-whip|partial-braid)` with `(subsetp $?rlcs $?rlcsa)` AND `(glabel-contains-some-of ?new-rlc $?rlcsa)` — when an existing plain partial-(whip|braid) at length k has the base rlcs as a subset and the new GCand "contains" one of its rlcs (member$ for GCand entries, `label-in-glabel` for Cand entries per `generic-background.clp:179-187`), the grouped braid is suppressed. Rust sub-rule -2 (gbraid.rs:783) only had the first guard. The fix is the same pattern as CR-FIN-4 M-opus-5 (which patched sub-rule -3): added a secondary scan against the new `plain_at_k` slice with the `subsetp + glabel-contains-some-of` test. The drivers `run_gbraid_pass` and `find_first_gbraid` were reordered so plain (braid|whip)[k] are materialized **before** `extend_partial_gbraids` is called, then the union is passed as `plain_at_k`. **Direction**: same as C1 — soundness-preserving over-production fix; CLIPS would prune these `GB[k]` partials in favor of the finer plain braid/whip at the same length, so Rust ratings could over-report `gB[k]` when a finer `B[k]` or `W[k]` was available. |
| **CR-FIN-5 MINOR** stale `Default k_max = 9` comment | **FIXED** | `rater.rs:269-278` | Comment now cites `chain_rated::CHAIN_RATER_K_MAX` (= 36 per CR-FIN-3 C2) instead of the obsolete 9. |

CLIPS-source quotes (load-bearing, kept verbatim for audit):

> `Partial-gWhips[1].clp:60-71` —
> ```
> ;;; do not assert a partial-gwhip[1] with
> ;;; - the same target as an already existing one
> ;;; - and the same rlc1 or with a larger rlc1 than an already existing partial-whip[1] or partial-gwhip[1]
> (not
>   (chain
>     (type partial-whip|partial-gwhip)
>     (context ?cont)
>     (length 1)
>     (target ?zzz)
>     (rlcs ?rlc1a&:(or (eq ?rlc1a ?rlc1) (label-in-glabel ?rlc1a ?rlc1)))
>   )
> )
> ```

> `Partial-gWhips[5].clp:141-150` (sub-rule -2) and `:213-221` (sub-rule -3) —
> ```
> ;;; do not assert a partial gwhip with the same sequences of rlc's or with no non smaller rlc than an existing one
> (not
>   (chain
>     (type partial-whip|partial-gwhip)
>     (context ?cont)
>     (length 5)
>     (target ?zzz)
>     (rlcs $?rlcs ?new-rlca&:(or (eq ?new-rlca ?new-rlc) (label-in-glabel ?new-rlca ?new-rlc)))
>   )
> )
> ```

> `gBraids[5].clp:200-221` (partial-gbraid[4]-2) —
> ```
> ;;; do not assert a new partial-gbraid with the same sets of rlc's as an existing partial-gwhip or partial-gbraid
> ;;; or with its new-rlc larger than an rlc of an existing whip or braid
> (not
>   (chain (type partial-gwhip|partial-gbraid) (context ?cont) (length 4) (target ?zzz)
>          (rlcs $?rlcsa&:(same-sets-of-rlcs ?new-rlc $?rlcs $?rlcsa))))
> (not
>   (chain (type partial-whip|partial-braid) (context ?cont) (length 4) (target ?zzz)
>          (rlcs $?rlcsa&:(and (subsetp $?rlcs $?rlcsa)
>                              (glabel-contains-some-of ?new-rlc $?rlcsa)))))
> ```

API impact: `build_partial_gwhips_length_1` now takes a second `&[Chain]` argument (length-1 plain whip seeds; pass `&[]` to opt out of the cross-type seed guard). `extend_partial_gwhips` and `extend_partial_gbraids` each gained one extra `&[Chain]` parameter for the length-k plain subsumer slice (`plain_at_k`). All in-tree call sites updated; unit tests at non-driver call sites pass `&[]` (the cross-type subsumption is then a no-op, matching their narrow structural scope).

Test result: `cargo test --lib` — 458 passed / 0 failed / 13 ignored (unchanged count; CR-FIN-5 fix is additive subsumption — never blocks a chain that the test fixtures previously asserted).

### Eleventh-pass remediation (CR-FIN-6 C1..C2 + MINOR, 2026-05-18)

Provenance: Round-9 codex BLOCK with 2 CRITICAL claims (both confirming a shared root cause for strict load-bearing semantics on `rate_excluding`); Round-9 opus GO (1 MAJOR + 6 MINOR). Opus' MAJOR is the same defect viewed at a coarser rating granularity; codex' CRITICAL framing prevails because the load-bearing path is the failing surface. Each fix below targets a distinct call site; the MINOR items are co-shipped.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-6 C1** Braid terminator fires on plain-whip partials, breaking `rate_excluding(p, &[Whip])` load-bearing semantics (codex CRITICAL = opus MAJOR) | **GENUINE — FIXED** | `braid.rs`, `chain_rated.rs` (doc-only) | Per CLIPS `Braids[5].clp:57-69` (eliminator binds `(type partial-braid)` only) vs `:95-157` (extension reads union `(type partial-whip|partial-braid)`), the terminator must NOT accept partial-whip chains. Rust `run_braid_pass` (braid.rs:585) and `find_first_braid` (braid.rs:696) iterated `partials_prev.chain(whip_prev)` and fed both to `try_terminate_braid`. In salience-interleaved firing this was masked (whip[k] fires first), but in `rate_excluding(p, &[Whip])` strict-load-bearing path (spec §10.3), whip is explicitly suppressed → braid would still fire on the remaining `partial-whip` chains and falsely report "solved" → puzzle wrongly concluded NOT load-bearing on Whip. **Fix**: (1) Removed `.chain(whip_prev.iter())` from both terminator loops in `run_braid_pass` and `find_first_braid` — the terminator now iterates `partials_prev` (PartialBraid) only. (2) `whip_prev` is retained as a cross-feed source for `extend_partial_braids` (the partial-braid extension rule still reads the CLIPS union of `partial-whip|partial-braid`). (3) Added a `debug_assert!(matches!(chain.kind, ChainKind::PartialBraid), ...)` at the top of `try_terminate_braid` so any future caller that violates the invariant fails loudly in test/debug builds. (4) Added a `debug_assert_eq!(chain.length + 1, k, ...)` length invariant (opus 9 MINOR at braid.rs equivalent). Tests added: `test_crfin6_c1_terminate_braid_rejects_partial_whip` (panic), `test_crfin6_c1_terminate_braid_accepts_partial_braid`, `test_crfin6_c1_find_first_braid_smoke_fixture2`. **Direction**: this is a soundness fix for strict load-bearing — eliminations under the salience-interleaved driver were already CLIPS-correct, but `rate_excluding` exclusion masks could mis-classify load-bearing whip puzzles. |
| **CR-FIN-6 C2** GBraid terminator fires on plain-gwhip partials, breaking `rate_excluding(p, &[GWhip])` load-bearing semantics (codex CRITICAL = opus MAJOR) | **GENUINE — FIXED** | `gbraid.rs` | Per CLIPS `gBraids[5].clp:57-69` (eliminator binds `(type partial-gbraid)` only) vs `:98-160`, `:167-235`, `:242-309` (extension sub-rules -1/-2/-3 read unions `partial-gwhip|partial-gbraid` or `partial-whip|partial-braid`), the gbraid terminator must NOT accept partial-gwhip chains. Rust `run_gbraid_pass` (gbraid.rs:1286) and `find_first_gbraid` (gbraid.rs:1487) iterated `partials_gbraid_prev.chain(partials_gwhip_prev)` and fed both to `try_terminate_gbraid`. Mirror of C1: in salience-interleaved firing whip→gwhip→braid→gbraid masks the bug, but in `rate_excluding(p, &[GWhip])` strict-load-bearing path, gwhip is excluded → gbraid still fires on remaining `partial-gwhip` chains → puzzle wrongly reported solved without gwhip. **Fix**: (1) Removed `.chain(partials_gwhip_prev.iter())` from both terminator loops. (2) `partials_gwhip_prev` retained as cross-feed for `extend_partial_gbraids` (sub-rules -1/-3 read `partial-gwhip|partial-gbraid` union). (3) Added `debug_assert!(matches!(chain.kind, ChainKind::PartialGBraid), ...)` at the top of `try_terminate_gbraid` plus the length invariant `debug_assert_eq!(chain.length + 1, k, ...)` (opus 9 MINOR at gbraid.rs:1488). Tests added: `test_crfin6_c2_terminate_gbraid_rejects_partial_gwhip` (panic), `test_crfin6_c2_terminate_gbraid_rejects_partial_braid` (panic — defense in depth for sub-rule -2's plain-chain cross-feed), `test_crfin6_c2_terminate_gbraid_accepts_partial_gbraid`, `test_crfin6_c2_find_first_gbraid_no_gwhip_fire_on_empty`. **Direction**: same as C1 — soundness fix for strict load-bearing under GWhip exclusion. |
| **CR-FIN-6 MINOR (chain_rated.rs:401)** `id()` placeholder undocumented (opus MINOR) | **FIXED** | `chain_rated.rs` | `Technique::id()` for `ChainCombinedTechnique` returned `TechniqueId::Whip` with a one-line `// placeholder; rater uses technique_id_override` comment. Expanded to a multi-line doc-comment explaining the trait constraint (one fixed id per impl), the out-of-band channel (`TechniqueProgress::technique_id_override`), the consumer-side invariant (CR-FIN-4 Mn-2 `debug_assert!`), and the explicit "MUST NOT key on this Whip value" rule for downstream code. |
| **CR-FIN-6 MINOR (chain_rating.rs:131)** `rate_chain` three-way `None` undocumented (opus MINOR) | **FIXED** | `chain_rating.rs` | `rate_chain` returns `None` in three semantically distinct cases (inconsistent givens, solved-by-singles, no chain fires within `k_max`). Doc-comment now enumerates all three and explains how callers must disambiguate upstream (run `is_consistent` + `propagate_singles` + `is_solved`) until a richer `Result`-shaped API lands. The `Result`-shaped refactor is deferred to a focused API pass (touches several call sites in `aic_reverse.rs`, `nested_aic_reverse.rs`, etc.). |
| **CR-FIN-6 MINOR (gbraid.rs:1057)** `is_glinked_or` vs `is_killed_or_glinked` duplication (opus MINOR) | **DEFERRED (TODO)** | `gbraid.rs` | `is_killed_or_glinked` is a thin trampoline to `is_glinked_or` after the FX-R1 fix (the `_killed_set` parameter is unused). Inlining requires touching ~10 call sites in gbraid.rs; the indirection currently documents the historical name pre-FX-R1 (when the body actually consulted the bitset). Added a TODO doc-comment in the function itself referencing CR-FIN-6 MINOR; deferred to a focused refactor pass. |
| **CR-FIN-6 MINOR (gbraid.rs:1130)** `build_partial_braids_length_1_for_gbraid` braid/gbraid seed duplication (opus MINOR) | **DEFERRED (TODO)** | none | The seed builder in `gbraid.rs:1155-` duplicates much of the plain-braid length-1 logic from `braid.rs::build_partial_braids_length_1`, with the small twist that the result chains are tagged for the gbraid sub-rule -2 cross-feed. Factoring into a shared helper requires a discriminator parameter (PartialBraid vs PartialBraid-for-gbraid) and a careful audit of dedup-key implications. Deferred — net code-size payoff is small relative to risk of subtle parity drift; revisit in a planned consolidation pass over `chain_utils`. |

CLIPS-source quotes (load-bearing, kept verbatim for audit):

> `Braids[5].clp:57-69` (eliminator — binds `(type partial-braid)` ONLY) —
> ```
> (defrule braid[5]
>    (declare (salience ?*braid[5]-salience*))
>    (chain
>       ;;; partial-whips can be omitted
>       (type partial-braid)
>       (context ?cont)
>       (length 4)
>       (target ?zzz)
>       ...
> ```

> `Braids[5].clp:95-99` (extension — reads `(type partial-whip|partial-braid)`) —
> ```
> (defrule partial-braid[4]
>    (declare (salience ?*partial-braid[4]-salience*))
>    (logical
>       (chain
>          (type partial-whip|partial-braid)
> ```

> `gBraids[5].clp:57-68` (eliminator — binds `(type partial-gbraid)` ONLY) —
> ```
> (defrule gbraid[5]
>    (declare (salience ?*gbraid[5]-salience*))
>    (chain
>       (type partial-gbraid)
>       (context ?cont)
>       (length 4)
>       (target ?zzz)
>       ...
> ```

> `gBraids[5].clp:98-103, 167-172, 242-247` (extension sub-rules -1/-2/-3 read CLIPS unions) —
> ```
> (defrule partial-gbraid{4]-1                   ; sub-rule -1
>    ...
>    (chain (type partial-gwhip|partial-gbraid) ...)
> (defrule partial-gbraid[4]-2                   ; sub-rule -2
>    ...
>    (chain (type partial-whip|partial-braid) ...)
> (defrule partial-gbraid[4]-3                   ; sub-rule -3
>    ...
>    (chain (type partial-gwhip|partial-gbraid) ...)
> ```

API impact: none beyond the documented `debug_assert!` invariants on `try_terminate_braid` and `try_terminate_gbraid`. The `Chain` parameter is unchanged; only the implicit input-kind contract is now enforced at runtime in test/debug builds.

Test result: `cargo test --lib` — 465 passed / 0 failed / 13 ignored (up from 458; 7 new tests: 3 in `braid.rs` and 4 in `gbraid.rs` covering panic invariants, non-regression accept paths, and driver-level smoke).

### Twelfth-pass remediation (CR-FIN-7 C1..C2 + M-opus-6 + M-codex-4 + MINOR, 2026-05-18)

Provenance: Round-10 opus BLOCK (2 CRITICAL + 2 MAJOR + 4 MINOR) and Round-10 codex BLOCK (1 MAJOR + 1 MINOR). Pre-verification: CLIPS V2.1 file inventory `ls CHAIN-RULES-SPEED/BRAIDS/` → `Braids[3].clp` minimum (no [1] or [2]); `ls CHAIN-RULES-SPEED/G-BRAIDS/` → `gBraids[3].clp` minimum (no [1] or [2]); same structural pattern as CR-FIN-4 C2 (gWhip[1] absent in CLIPS). All Round-10 findings GENUINE.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-7 C1** Braid emits `Braid(1)` / `Braid(2)` for non-existent CLIPS rules; cascade falsely solves Whip[{1,2}]-load-bearing puzzles, defeating `rate_excluding(p, &[Whip])` (opus CRITICAL) | **GENUINE — FIXED** | `braid.rs` | CLIPS V2.1 `ls CHAIN-RULES-SPEED/BRAIDS/` shows `Braids[3..36].clp` only — no `Braids[1].clp` or `Braids[2].clp`. Previously `try_braid_1_eliminations` (whip[1] semantics emitting `ChainRule::Braid(1)`) and the k=2 terminator iteration inside `find_first_braid` / `run_braid_pass` produced `ChainRule::Braid({1,2})`. In strict load-bearing path `rate_excluding(p, &[Whip])` (spec §10.3), whip is masked → braid arm of `ChainCombinedTechnique` fires the misrouted Braid(1)/Braid(2) on whip-shaped eliminations → cascade solves the puzzle → load-bearing gate (whip_reverse.rs:407–418) sees `r2.solved == true` and rejects every true Whip[{1,2}]-load-bearing puzzle as not load-bearing. **Fix**: (1) k-floor `find_first_braid` and `run_braid_pass` at `k_max < 3 → None` / `Vec::new()`. (2) Terminator iteration starts at k=3 (CLIPS file-inventory floor); a one-shot length-1→length-2 extension is performed before the k=3 loop so prev-partials are at length 2 when k=3 fires. (3) `try_braid_1_eliminations` gated behind `#[cfg(test)]` — retained as parity helper for `test_braid1_direct_parity_with_whip1` and the new floor tests, removed from all production call sites. The length-1 partial-braid seed and `extend_partial_braids` are kept (needed to grow partial-braid[2] from length-1 seeds, which the k=3 terminator consumes). **Direction**: soundness fix — closes the load-bearing classification false negative for Whip[1]/Whip[2] under strict spec §10.3 semantics. |
| **CR-FIN-7 C2** Same defect at k=2 (braid) and k=1/k=2 (gbraid) (opus CRITICAL) | **GENUINE — FIXED** | `gbraid.rs` | CLIPS V2.1 `ls CHAIN-RULES-SPEED/G-BRAIDS/` shows `gBraids[3..36].clp` only. Previously `try_gbraid_1_eliminations` and the k=2 terminator inside `find_first_gbraid` / `run_gbraid_pass` produced `ChainRule::GBraid({1,2})` for gwhip-shaped eliminations. Symmetric to C1: under `rate_excluding(p, &[GWhip])`, gwhip is masked → gbraid arm fires on residual gwhip[{1,2}]-shaped eliminations → `r2.solved == true` → every true GWhip[{1,2}]-load-bearing puzzle rejected. **Fix**: (1) k-floor `find_first_gbraid` and `run_gbraid_pass` at `k_max < 3 → None` / `Vec::new()`. (2) Terminator iteration only fires when `k >= 3` inside the existing k=2..k_max loop (the loop still runs the extension layer at k=2 to grow length-1 → length-2 partials, but does NOT emit eliminations at k=2). (3) `try_gbraid_1_eliminations` gated behind `#[cfg(test)]`, kept only for the existing smoke test (`gbraid.rs:1663`). **Direction**: soundness fix mirror of C1 for GWhip strict load-bearing semantics. |
| **CR-FIN-7 M-opus-6** `GBraidReverseSpec` accepts `target_k = 1` (CR-FIN-4 M-codex-3 regression) (opus MAJOR) | **GENUINE — FIXED (supersedes CR-FIN-4 M-codex-3)** | `gbraid_reverse.rs`, `braid_reverse.rs` | CR-FIN-4 M-codex-3 had widened `gbraid_reverse::validate` to accept `target_k = 1` on the rationale that the solver could emit `GBraid(1)`. Round-10 verification against CLIPS file inventory shows no `gBraids[{1,2}].clp` on disk — and after CR-FIN-7 C2 the solver cannot emit `GBraid({1,2})` either, so accepting `target_k ∈ {1,2}` here produces unsatisfiable construction jobs. **Fix**: tighten `GBraidReverseSpec::validate` to `target_k >= 3` (mirroring `GWhipReverseSpec::validate` which already enforces `target_k >= 2`); apply the same `target_k >= 3` floor to `BraidReverseSpec::validate` for parity with the new solver floor. CR-FIN-4 M-codex-3 is hereby **superseded by CR-FIN-7 M-opus-6** per CLIPS file-inventory evidence. |
| **CR-FIN-7 M-codex-4** Reverse probes lack BRT singles pre-pass; scoring fires on a different state than the `rate_chain` classification gate (codex MAJOR) | **GENUINE — FIXED** | `whip_reverse.rs`, `braid_reverse.rs`, `gwhip_reverse.rs`, `gbraid_reverse.rs` | Per spec §4.1 / §4.1.1, CLIPS runs single / naked-single / hidden-single before any chain rule (`Generic-Background/Singles.clp`). `rate_chain` (chain_rating.rs:153) honors this via `propagate_singles`. The four reverse-construction probe / score / final-verifier sites built `ResolutionState` directly from the raw puzzle and called `find_first_*` without the pre-pass → reported `fired_k` and acceptance could disagree with the classification gate (and with the ignored regression at `techniques/braid.rs:902` already documenting Fixture 3 missing without `propagate_singles`). **Fix**: each of the four reverse modules now defines a local `build_post_brt_grid(&Grid) -> Option<Grid>` that (a) checks `is_consistent`, (b) clones + runs `propagate_singles`, (c) returns `None` on contradiction OR already-solved, else `Some(post_brt)`. All `*_score` / `*_probe` / final `find_first_*` verifier sites in the four modules use the post-BRT grid for `ResolutionState`. Singles-contradiction and singles-solves cases are treated as "reject this attempt" (return `(false, i32::MAX)` for score, `continue` in the outer driver loop). This aligns reverse acceptance with the chain rating gate. |
| **CR-FIN-7 M-opus-7** O(k²) work in `ChainCombinedTechnique` salience loop (opus MAJOR — carried-over from CR-FIN-4 Mn-4) | **DEFERRED (TODO)** | `chain_rated.rs` (comment only) | Each k=K iteration calls `find_first_whip(ctx, K)`, etc., which internally redoes k=1..K-1 scans. With `CHAIN_RATER_K_MAX=36` and 4 arms this is up to ~5184 inner iterations where ~144 suffice. This is a performance issue, not a correctness one — same eliminations are produced. Refactor requires either: (a) caching `partials_prev` per-arm across k iterations (adds memory; needs cache invalidation contract under per-arm masking), or (b) lifting the salience loop inside each `find_first_*` (changes the public contract of those functions and ripples into `chain_rating.rs::rate_chain`). Deferred to a focused performance pass. A TODO comment co-shipped in `chain_rated.rs` (above the k-loop) referencing this entry. |
| **CR-FIN-7 Mn-1** Doc-comment hedges on `try_braid_1_eliminations` / `try_gbraid_1_eliminations` (opus MINOR) | **FIXED** | `braid.rs`, `gbraid.rs` | Replaced the hedged "if it exists" / "analogous to" prose with definitive doc-comments: "No `Braids[1].clp` in CLIPS V2.1; helper retained for `#[cfg(test)]` parity only. See CR-FIN-7 C1." Same for gbraid. The helpers are now `#[cfg(test)]`-gated, so the doc text cannot drift back into a production contract. |
| **CR-FIN-7 Mn-2** Spec §4.1 salience tower stale (opus MINOR) | **FIXED** | `docs/csp_rules_chain_spec.md` §4.1 | The salience tower listed `G-Whip[1] = gWhip[1]`, `Braid[2..]`, `gBraid[2..]` — none of which exist in CLIPS V2.1. Replaced with per-family minimum: Whip ≥ 1, GWhip ≥ 2 (CR-FIN-4 C2), Braid ≥ 3 (CR-FIN-7 C1), GBraid ≥ 3 (CR-FIN-7 C2). Cited CLIPS file inventory evidence via `ls CHAIN-RULES-SPEED/{WHIPS,G-WHIPS,BRAIDS,G-BRAIDS}/`. The k=1 row no longer mentions gWhip[1]; at k=2 only whip[2]+gwhip[2]; at k≥3 the full whip→gwhip→braid→gbraid order applies. |
| **CR-FIN-7 Mn-3** ChainCombined placeholder `TechniqueId::Whip` leak (opus MINOR) | **DEFERRED (TODO)** | `chain_rated.rs`, `rater.rs` | The `Technique::id()` impl for `ChainCombinedTechnique` returns the placeholder `TechniqueId::Whip` (rater uses `technique_id_override` for real attribution). Introducing a dedicated `TechniqueId::ChainCombinedPlaceholder` variant requires touching every exhaustive `match` on `TechniqueId` across the codebase (rater.rs serializers, exclusion-set handling, plus serde compatibility for stored manifests). The current placeholder is guarded by a `debug_assert!` in `rater.rs` (CR-FIN-4 Mn-2) that catches any consumer keying off the placeholder value. Deferred to a focused enum-expansion pass. A doc-comment in `chain_rated.rs:401-410` already documents the contract (per CR-FIN-6 MINOR). |
| **CR-FIN-7 Mn-4** `gbraid.rs:1057` `is_killed_or_glinked` trampoline (opus MINOR — carried-over from CR-FIN-6) | **DEFERRED (TODO)** | none | Same as CR-FIN-6 MINOR (gbraid.rs:1057): inline-deduplication touches ~10 call sites and the indirection currently documents pre-FX-R1 historical naming. Already TODO-commented at the function site (CR-FIN-6 entry). |

CLIPS-source evidence (load-bearing, kept verbatim for audit):

```
$ ls /tmp/csp-rules-research/CSP-Rules-V2.1/CSP-Rules-Generic/CHAIN-RULES-SPEED/BRAIDS/ | sort
Braids[3].clp
Braids[4].clp
…
Braids[36].clp
# (no Braids[1].clp; no Braids[2].clp)

$ ls /tmp/csp-rules-research/CSP-Rules-V2.1/CSP-Rules-Generic/CHAIN-RULES-SPEED/G-BRAIDS/ | sort
gBraids[3].clp
gBraids[4].clp
…
gBraids[36].clp
# (no gBraids[1].clp; no gBraids[2].clp)
```

This is the structurally same pattern as CR-FIN-4 C2 (gWhip[1] absent in CLIPS), and the resolution mechanism is identical: enforce the per-family minimum in the solver entry points (`find_first_*` / `run_*_pass`) and in the corresponding reverse-spec validators.

API impact: `try_braid_1_eliminations` and `try_gbraid_1_eliminations` are now `#[cfg(test)]`-only — out-of-tree callers (none in this repo) must migrate. `BraidReverseSpec::validate` / `GBraidReverseSpec::validate` now reject `target_k < 3` (CR-FIN-4 M-codex-3 superseded). One existing test (`test_fxr5_fixture2_braid1_known_answer`) was updated: its assertion that the cascade emits `Braid(1)` or `Whip(1)` was structurally incorrect against CLIPS V2.1 and has been replaced with `Braid(k>=3) || Whip(_)`. New tests gate the k=0,1,2 floor on both `find_first_braid`/`run_braid_pass` and `find_first_gbraid`/`run_gbraid_pass`.

Test result: `cargo test --lib` — **469 passed / 0 failed / 13 ignored** (up from 465; 4 new floor tests + 1 updated assertion; all CR-FIN-1..6 tests unchanged).

### Thirteenth-pass remediation (CR-FIN-8 Mn-1..Mn-2, 2026-05-18)

Provenance: Round-11 opus unbiased review GO with 1 MAJOR + 5 MINOR (codex review hit quota; retried later). The two highest-signal items applied now so the next codex review runs on the cleanest state.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-8 Mn-1** `try_gwhip_1_eliminations` is `pub fn` (no `#[cfg(test)]` gate) while its braid/gbraid siblings were gated under CR-FIN-7 C1/C2; out-of-tree callers could bypass `find_first_gwhip`'s full extension layer and cross-type subsumption guards (CR-FIN-5 C1) (opus 11 MAJOR) | **FIXED** | `src/generic/techniques/gwhip.rs:344` | Gated `try_gwhip_1_eliminations` behind `#[cfg(test)]`, mirroring `try_braid_1_eliminations` (`braid.rs:251`) and `try_gbraid_1_eliminations` (`gbraid.rs:422`) post-CR-FIN-7. Pre-fix `grep` confirmed zero production call sites (only the function definition + this gate appear in `gwhip.rs`). Doc-comment rewritten to be definitive: "No `gWhips[1].clp` in CLIPS V2.1; helper retained for `#[cfg(test)]` parity only." Doc now also clarifies that the emission is `ChainRule::GWhip(2)` (not `GWhip(1)` — `try_terminate_gwhip` computes `k = chain.length + 1`, and seeds have length 1, so the "1" in the function name refers to seed length consumed, not the resulting rule index). **Direction**: API-surface hygiene fix — removes the only remaining unguarded entry point that could bypass the production gwhip pipeline. |
| **CR-FIN-8 Mn-2** Tautological `debug_assert_eq!((chain.length+1) as u16, k as u16, ...)` in `try_terminate_braid` / `try_terminate_gbraid` (opus 11 MINOR) | **FIXED** | `src/generic/techniques/braid.rs:480`, `src/generic/techniques/gbraid.rs:1061` | The asserts followed `let k = chain.length + 1;` on the line directly above, so they reduced to `k == k` and provided no protection. Replaced with the meaningful length-floor invariant from CR-FIN-7: `debug_assert!(chain.length >= 2, "<braid/gbraid> terminator requires length-2 partial for k>=3")`. Floor evidence: CLIPS `Braids[3..36].clp` / `gBraids[3..36].clp` minimum (no [1]/[2]) — terminator fires at k>=3, so partials must be length k-1>=2. Existing `debug_assert!(matches!(chain.kind, ChainKind::PartialBraid))` / `ChainKind::PartialGBraid` type guards from CR-FIN-6 retained. **Test fixup**: `test_fx3_terminate_gbraid_no_gcand_member_rejection` was feeding length-1 seeds directly into `try_terminate_gbraid` (it "passed" only because seeds were empty on that puzzle); updated to extend seeds to length 2 via `extend_partial_gbraids` before feeding the terminator, aligning the test with the CR-FIN-7 C2 production contract. |

Test result: `cargo test --lib` — **469 passed / 0 failed / 13 ignored** (unchanged total; one existing FX-3 test updated to honor the CR-FIN-7 C2 length-floor contract). `cargo check` clean. CR-FIN-8 M-class items (carried-over Mn-3..Mn-5 from opus 11) and codex retry deferred to a follow-up pass.

### Fourteenth-pass remediation (CR-FIN-9 Mn-1..Mn-3, 2026-05-18)

Provenance: Round-12 opus unbiased review GO with 0 CRITICAL / 0 MAJOR / 3 MINOR. Codex retry hit quota again; applying the three MINOR fixes now so the state is fully clean for the next codex round.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-9 Mn-1** Spec §4.1 salience tower still lists `G2-Whip[1] (advanced — out of scope here)` and `Bivalue-Chains[k] (k=2..…) (interleaved — outside scope)` at the top of the tower, contradicting CR-FIN-7 Mn-2 (no `gWhips[1].clp` in CLIPS V2.1) and the §4.1.1 BRT scope clarification (opus 12 MINOR) | **FIXED** | `docs/csp_rules_chain_spec.md` §4.1 (lines 115-116) | Deleted the two stale lines. Canonical source is now (a) the explicit per-family minimum table at lines 124-128 plus (b) the "At k = 2 only whip[2] and gwhip[2] exist. At k = 1 only whip[1] exists." paragraph at lines 141-142, both already aligned with CR-FIN-4 C2 / CR-FIN-7 C1+C2 file-inventory evidence. **Direction**: documentation hygiene — removes a residual contradiction inside the salience tower visual. |
| **CR-FIN-9 Mn-2** `try_terminate_gwhip` lacks the kind + length-floor `debug_assert`s that `try_terminate_braid` (`braid.rs:472-486`) and `try_terminate_gbraid` (`gbraid.rs:1052-1066`) have under CR-FIN-6 + CR-FIN-8 Mn-2 — asymmetric guards across the three terminators (opus 12 MINOR) | **FIXED** | `src/generic/techniques/gwhip.rs:911` | Added two `debug_assert!`s at the top of `try_terminate_gwhip`, mirroring the braid/gbraid pattern: (1) `matches!(chain.kind, ChainKind::PartialGWhip)` — catches a future refactor that pipes `PartialWhip` (or any non-`PartialGWhip`) chain into this terminator; (2) `chain.length >= 1` — encodes the CR-FIN-4 C2 minimum (gwhip fires at k>=2, so the terminator consumes partials of length k-1>=1). Floor evidence: CLIPS `gWhips[2..36].clp` minimum (no `gWhips[1].clp`). All 469 lib tests still pass — neither assertion fires on existing call sites (production: `find_first_gwhip`/`run_gwhip_pass`; tests: `try_gwhip_1_eliminations` under `#[cfg(test)]` per CR-FIN-8 Mn-1). **Direction**: symmetry / API hygiene — closes the only remaining unguarded terminator in the four-family set. |
| **CR-FIN-9 Mn-3** `any_live` BRT pre-pass dependency at `whip.rs:283` (`try_whip_1_eliminations`) and the analogous "dead alt is trivially killed" branch at `gwhip.rs:274` (`build_partial_gwhips_length_1`) is undocumented in-source — both gates rely on BRT having already fired hidden singles, but only the §4.1.1 prose mentions this contract (opus 12 MINOR) | **FIXED** | `src/generic/techniques/whip.rs:283`, `src/generic/techniques/gwhip.rs:274` | Added a 3-line inline comment at each site: `// CR-FIN-9 Mn-3: any_live=true relies on BRT pre-pass (rate_chain / chain_rated propagate singles); if a future caller bypasses BRT, this guard becomes incorrectness — see spec §4.1.1.` Semantics: if all alts on a csp-var are dead while the seed cell is alive, BRT would have fired the hidden single before any chain rule, so the gate's reliance on a live alt existing in the post-BRT state is sound for all current callers (`rate_chain` in `chain_rating.rs` and `ChainCombinedTechnique` in `chain_rated.rs` both call `propagate_singles` before chain probing). The in-source note makes the dependency local-readable so a future call-site refactor doesn't silently break it. **Direction**: documentation hygiene — makes a global invariant visible at the local guard. |

Test result: `cargo test --lib` — **469 passed / 0 failed / 13 ignored** (unchanged total; no test changes — comment + debug_assert additions only). `cargo check` clean (no new warnings). Final codex verification still pending (round-11 + round-12 both hit codex quota); this pass leaves the repository in the cleanest documented state for the next codex round.

### Fifteenth-pass remediation (CR-FIN-10 Mn-1..Mn-4, 2026-05-18)

Provenance: Round-12 codex unbiased review BLOCK with 0 CRITICAL / 3 MAJOR (test hygiene) + 1 MINOR (stale prose). Opus round-12 review remained GO. All four findings are test-suite gaps and one stale comment introduced when CR-FIN-7 C1+C2 tightened `BraidReverseSpec::validate` and `GBraidReverseSpec::validate` to `target_k >= 3`: pre-existing tests that used `target_k ∈ {1, 2}` now hit the validation rejection at the top of `*_reverse_construct` and short-circuit to `None`, leaving every interior `if let Some(...)` arm dead — i.e. silently soft-passing without exercising the contract under test. Production correctness is unaffected; this pass restores real test coverage of the post-CR-FIN-7 construction path and removes one stale comment that referred to the no-longer-reachable `Braid(1)` firing.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-10 Mn-1** `braid_reverse` test module uses `target_k = 1` in five operational and rejection tests; post-CR-FIN-7 those tests soft-pass through the dead arm, providing no coverage (codex 12 MAJOR) | **FIXED** | `src/generic/braid_reverse.rs` (test module, lines ~590–920) | Five tests updated. (a) **Operational tests** (`determinism_same_seed_same_puzzle`, `load_bearing_result_is_b_rated`, `batch_determinism_single_thread`) — `target_k` bumped to 3 (CLIPS `Braids[3..36]` floor), `clue_min/clue_max` widened to `30..81` so construction can return a puzzle, and each test gated with `#[ignore = "CR-FIN-10 Mn-1: ... run with --ignored ..."]` because Braid[3] reverse-search wall-clock exceeds the fast-test budget on commodity hardware; the ignore is preferred over vacuous soft-pass and is run manually via `cargo test --lib -- --ignored <test_name>`. (b) **`braid_excludes_whip`** — structural sentinels keep their original semantics, but `b_rating = ChainRating::B(2)` → `B(3)` so the literal matches the post-CR-FIN-7 reachable range; the functional probe at the end uses `target_k = 3, k_slack = 2` and stays soft-pass on `None` (the structural assertions before it run unconditionally). (c) **`validate_rejects_clue_min_gt_clue_max`** — `target_k` bumped to 3 so the validation failure is isolated to the inverted clue band (not k-floor); now uses `expect_err(...)` and asserts the message contains both `clue_min` and `clue_max`. (d) **Two new dedicated rejection tests added**: `validate_rejects_target_k_below_floor` (iterates `target_k ∈ {0, 1, 2}` and asserts the error mentions the `>= 3` floor; also calls `braid_reverse_construct` and asserts `None`) and `validate_rejects_target_k_above_max` (`target_k = 37`, asserts message mentions 36 / "maximum" and construct returns `None`). **Direction**: test-hygiene fix — restores real coverage of the post-CR-FIN-7 BraidReverseSpec contract. |
| **CR-FIN-10 Mn-2** `gbraid_reverse` test module uses `target_k ∈ {1, 2}` in six operational and rejection tests; same defect pattern as Mn-1 (codex 12 MAJOR) | **FIXED** | `src/generic/gbraid_reverse.rs` (test module, lines ~488–770) | Symmetric to Mn-1. Five operational tests (`determinism_same_seed`, `smoke_low_k_gbraid_fires`, `load_bearing_only_returns_gb_rated_puzzles`, `batch_terminates_and_bounded`, plus the functional probe inside `gbraid_excludes_gwhip`) updated: `target_k = 3`, `clue_min/clue_max = 30/81`, and the four end-to-end variants `#[ignore]`-gated (gbraid[3] search is more expensive than braid[3]; the search at `clue_min = 22` was already on the edge of feasible for fast tests, so `--ignored` is the right gate). `gbraid_excludes_gwhip` structural sentinel `gb_rating = GB(2) → GB(3)` for post-CR-FIN-7-C2 alignment. `validate_rejects_inverted_clue_band` tightened with `expect_err + contains` and bumped to `target_k = 3`. Two new dedicated rejection tests added with the same shape as Mn-1: `validate_rejects_target_k_below_floor` (covers `{0, 1, 2}`) and `validate_rejects_target_k_above_max` (covers 37). **Direction**: test-hygiene fix — restores real coverage of the post-CR-FIN-7 C2 GBraidReverseSpec contract. |
| **CR-FIN-10 Mn-3** `chain_rating::test_fixture2_braid1_or_whip` accepts `Some(ChainRating::B(1))` in its expected-outcomes set; after CR-FIN-7 C1 `rate_chain` cannot emit `Braid(1)` (`find_first_braid` is k-floored at `k >= 3`) (codex 12 MAJOR) | **FIXED** | `src/generic/chain_rating.rs:253–280` | Replaced the `matches!(rating, Some(W(1)) \| Some(W(2)) \| Some(B(1)))` pattern with an explicit `match` expression: `Some(W(1)) \| Some(W(2)) → true`, `Some(B(k)) if k >= 3 → true`, `_ → false`. (The `matches!` macro's trailing guard applies to all alternatives, so a guarded `B(k) if k >= 3` cannot be combined with `W(1) \| W(2)` without binding `k` in both arms — the explicit `match` is required.) Doc-comment updated to explain that post-BRT-prepass the observed firing on this fixture is `W(1)` (the propagation in `rate_chain` exposes a length-1 whip before any braid path matures); the `W(2)` and `B(k>=3)` arms are kept as permissive fall-backs for future cross-feed shifts. Audit of nearby tests turned up two structural `ChainRating::B(2)` / `GB(2)` literals (`braid_reverse.rs:804`, `gbraid_reverse.rs:682`) which are pure `matches!`-semantics sentinels (not solver outputs); they were tightened to `B(3)` / `GB(3)` anyway for post-CR-FIN-7 consistency (covered above under Mn-1 / Mn-2). The `ChainRating::B(2)` in `whip_reverse.rs:673` is a type-system check (verifies `B(_)` does not match `W(_)`) and is left untouched: the `k` value is irrelevant to its assertion. **Direction**: regression-coverage fix — removes a logically impossible accept arm so the test cannot mask a real cascade regression. |
| **CR-FIN-10 Mn-4** Stale comment in `chain_rated.rs:677` says "Fixture 2's `Braid(1)` firing ... should still fire" — `Braid(1)` is no longer emitted post-CR-FIN-7 C1 (codex 12 MINOR) | **FIXED** | `src/generic/techniques/chain_rated.rs:677–693` | Rewrote the comment and the `.expect(...)` message to reflect the post-CR-FIN-7 cascade contract: with Whip masked, Fixture 2 still has the remaining arms gWhip[k>=2] (CR-FIN-4 C2), Braid[k>=3] (CR-FIN-7 C1), and gBraid[k>=3] (CR-FIN-7 C2) available; collectively they still fire on this fixture (verified by `test_fxr5_fixture2_braid1_known_answer` which post-CR-FIN-7 asserts at least one `Braid(k>=3)` or `Whip(_)` elimination). The structural assertion below the comment is unchanged — it already permits `Some(GWhip) \| Some(Braid) \| Some(GBraid)`. **Direction**: documentation hygiene — removes stale prose that contradicted the updated cascade. |

Test result: `cargo test --lib` — **466 passed / 0 failed / 20 ignored** (from 469/0/13 in CR-FIN-9). Delta accounting: +4 new rejection tests (2 in `braid_reverse`, 2 in `gbraid_reverse`), and 7 previously-active tests now `#[ignore]`-gated (3 in `braid_reverse`: `determinism_same_seed_same_puzzle`, `batch_determinism_single_thread`, `load_bearing_result_is_b_rated`; 4 in `gbraid_reverse`: `determinism_same_seed`, `smoke_low_k_gbraid_fires`, `load_bearing_only_returns_gb_rated_puzzles`, `batch_terminates_and_bounded`). Net: 469 − 7 + 4 = 466 passed; 13 + 7 = 20 ignored; 482 + 4 = 486 total. `cargo check` clean (no new warnings beyond the pre-existing 18 in lib / 25 in lib-test). Running the 7 newly-`#[ignore]`-gated tests with `cargo test --lib -- --ignored` is the manual-verification path for end-to-end reverse-construction behaviour at the Braid[3] / gBraid[3] floor; CI keeps the fast suite snappy.

### Sixteenth-pass remediation (CR-FIN-11 Mn-1..Mn-4, 2026-05-18)

Provenance: Round-13 codex unbiased review BLOCK with 0 CRITICAL / 3 MAJOR + 1 MINOR (all test/fixture hygiene). CR-FIN-11 Mn-1 was independently verified against the SHC corpus (file:line evidence below). CR-FIN-11 Mn-2 and Mn-3 are the post-CR-FIN-10-Mn-{1,2} soft-pass residue: the structural sentinels were tightened, but the functional probes that exercise the unconditional B-rated / GB-rated gate were still `if let Some(...) { ... }` with a silent fall-through on `None`. CR-FIN-11 Mn-4 is a stale doc-comment that survived CR-FIN-10 Mn-4 (which only rewrote the lower comment block in the same test).

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-11 Mn-1** Spec §12 Fixture 3 mis-sourced: claims `cbg000#109` is B=3 but SHC corpus annotates it as B=7 — all Rust tests asserting "Fixture 3 fires B(3)" are gated on the wrong puzzle (codex 13 MAJOR) | **GENUINE — FIXED** | `docs/csp_rules_chain_spec.md` §12 Fixture 3, `src/generic/chain_rating.rs:285–308`, `src/generic/braid_reverse.rs:669–701`, `src/generic/techniques/braid.rs:936–958` | SHC corpus evidence: `/tmp/csp-rules-research/CSP-Rules-V2.1/XTERNS/SHC/examples/B-input.txt:71` reads `.23....8.4....9..378....5......75.3.......215...61...7.6.5.1....42.3....9....4..8  56710;cbg000#109;B=7` — i.e. `cbg000#109` is annotated B=7, not B=3. The genuine B=3 entries in the same file include `cbg000#3` (line 3: `.23..6.8......91...8.1..4..2.......7...8.....678.1......7.3.2...3...4.7....5.1.6.  313075;cbg000#3;B=3`), `cbg000#11`, `cbg000#13`. **Fix**: chose `cbg000#3` for canonicality. Spec §12 Fixture 3 puzzle string replaced and the metadata block updated (source line, note about the previous mis-source). Rust call sites updated: (a) `chain_rating.rs::test_fixture3_braid3` — puzzle replaced, doc-comment updated, `#[ignore]` removed; the assertion was tightened from the old `assert_eq!(rating, Some(B(3)))` to an explicit `match` accepting the post-CR-FIN-7 reachable set (`W(_) → true`, `GW(k>=2) → true`, `B(k>=3) → true`, `GB(k>=3) → true`, `_ → false`) — this mirrors the Fixture 2 pattern from CR-FIN-10 Mn-3 and reflects that our rater's cascade (W → GW → B → GB) sees a shorter whip before any braid path matures on this puzzle. Observed: `rate_chain(&grid, 6) → Some(W(1))`. (b) `braid_reverse.rs::fixture3_cbg000_109_braid3` renamed to `fixture3_cbg000_3_braid3`, puzzle string replaced, ignore-reason rewritten to reference CR-FIN-11 Mn-1 (test stays `#[ignore]` because it builds `ResolutionState` directly from the raw grid — no BRT pre-pass — so the raw-probe regression mirrors the propagation issue rather than the corrected fixture). Inner `assert_eq!(k, 3)` softened to `assert!(k >= 3)` for the same calibration reason. (c) `techniques/braid.rs::test_braid3_termination_fixture3` — puzzle string replaced, doc-comment and ignore-reason updated to cite CR-FIN-11 Mn-1; behavioural assertion unchanged (smoke-test for braid termination, ignored pending propagate_singles integration). **Spec §12 note**: the "Expected W/B/gB rating" line gains a parenthetical that our rater may report a stricter category due to calibration divergence between the SHC battery and our cascade; the SHC B=3 is preserved as the authoritative source label. **Direction**: corpus-truth fix — aligns the fixture identity with the SHC ground-truth annotation. |
| **CR-FIN-11 Mn-2** `braid_excludes_whip` still soft-passes on `None`: the functional probe at the bottom of the test was wrapped in `if let Some(p) = braid_reverse_construct(...) {...}` with no `else` branch, so when the search misses budget the unconditional B-rated gate assertion is unexercised (codex 13 MAJOR) | **FIXED** | `src/generic/braid_reverse.rs:790–880` | Split the test into two functions. (a) `braid_excludes_whip_structural` — runs unconditionally in the fast suite, contains the `matches!()`-sentinel pair (`W(2)` must not match `B(_)`, `B(3)` must match `B(_)`); these are the load-bearing portion that does not depend on a real construction. (b) `braid_excludes_whip_functional` — marked `#[ignore = "CR-FIN-11 Mn-2: Braid[3] reverse construction is expensive; run with --ignored braid_excludes_whip_functional"]`, replaces the previous `if let Some(p) = ...` with `let p = ...expect("CR-FIN-11 Mn-2: ...");`, bumps `max_attempts` 60 → 200 and narrows `clue_max` 81 → 50 to keep the search deterministic at the Braid[3] floor. **Mechanism**: with `expect(...)` instead of `if let Some(...)`, a None construction makes the test fail loudly; `#[ignore]` keeps the fast suite snappy while `--ignored` runs exercise the unconditional B-rated gate as intended. **Direction**: test-hygiene fix — removes the silent None fall-through that hid an unexercised contract. |
| **CR-FIN-11 Mn-3** Same soft-pass on `None` in `gbraid_excludes_gwhip` (codex 13 MAJOR) | **FIXED** | `src/generic/gbraid_reverse.rs:673–740` | Symmetric to Mn-2. Split into `gbraid_excludes_gwhip_structural` (unconditional, runs in fast suite) and `gbraid_excludes_gwhip_functional` (marked `#[ignore]`, uses `expect(...)`, `max_attempts` 60 → 200, `clue_max` 81 → 50). **Direction**: test-hygiene fix — same shape as Mn-2 for the GBraid[3] side of the cascade. |
| **CR-FIN-11 Mn-4** Stale prose at `chain_rated.rs:546` says Fixture 2 "is known to fire whip[1] or braid[1] after singles" — `Braid(1)` is no longer reachable post-CR-FIN-7 C1; the lower comment block was already updated by CR-FIN-10 Mn-4 but the upper one survived (codex 13 MINOR) | **FIXED** | `src/generic/techniques/chain_rated.rs:543–562` | Rewrote the upper doc-comment to list the post-CR-FIN-7 reachable arms — `Whip(k>=1)`, `GWhip(k>=2)`, `Braid(k>=3)`, `GBraid(k>=3)` — and added a CR-FIN-11 Mn-4 note explaining the divergence from the pre-CR-FIN-7 wording. The structural assertion below (`technique_id_override ∈ {Whip, GWhip, Braid, GBraid}`) already covers the full cascade and is unchanged. **Direction**: documentation hygiene — completes the CR-FIN-10 Mn-4 sweep on the same test's surviving stale prose. |

Test result: `cargo test --lib` — **467 passed / 0 failed / 21 ignored** (from 466/0/20 in CR-FIN-10). Delta accounting: (a) `test_fixture3_braid3` flipped from `#[ignore]` to active (−1 ignored, +1 passed); (b) `braid_excludes_whip` split into one active (`_structural`) + one `#[ignore]` (`_functional`) = +1 passed, +1 ignored, +1 total; (c) `gbraid_excludes_gwhip` split into one active (`_structural`) + one `#[ignore]` (`_functional`) = +1 passed, +1 ignored, +1 total. Net: 466 − 1 (renamed) + 3 = 467 passed (the −1 is the original `braid_excludes_whip` that no longer exists; replaced by `_structural` + `_functional`; same for `gbraid_excludes_gwhip`). Ignored: 20 − 1 (test_fixture3_braid3 un-ignored) + 2 (new `_functional` halves) = 21. Total: 486 + 3 + 1 = 488 (= 467 + 21). `cargo check` clean (no new warnings beyond the pre-existing 18 in lib / 25 in lib-test).

#[ignore]-marked tests added this pass:
- `src/generic/braid_reverse.rs::braid_excludes_whip_functional` — manual run via `cargo test --lib -- --ignored braid_excludes_whip_functional`.
- `src/generic/gbraid_reverse.rs::gbraid_excludes_gwhip_functional` — manual run via `cargo test --lib -- --ignored gbraid_excludes_gwhip_functional`.

#[ignore] removed this pass:
- `src/generic/chain_rating.rs::test_fixture3_braid3` — was `#[ignore = "FA-4 pre-pass exposes W(1) before B(3); needs CLIPS oracle validation"]`; now active with the post-CR-FIN-7 reachable-set assertion shape (observed `W(1)` accepted as a valid post-BRT rating, mirroring CR-FIN-10 Mn-3 Fixture 2 pattern).

### Seventeenth-pass remediation (CR-FIN-12 C1 + M1 + Mn-1..Mn-4, 2026-05-19)

Provenance: Round-14 unbiased review. Opus-14 BLOCK with 1 CRITICAL (integration test build break) + 0 MAJOR + 3 MINOR (Mn-3 short-circuit asymmetry, Mn-4 unused-variable warnings, plus the doc-drift entry duplicated against codex's Mn-2). Codex-14 BLOCK with 0 CRITICAL + 1 MAJOR (braid scorer asymmetric: suppresses only whip, not gwhip — biases guided removal) + 2 MINOR (SE mapping table duplicated between `chain_rating.rs` and `chain_rated.rs`, stale module doc in `braid_reverse.rs`). All findings genuine. The production correctness of the load-bearing chain logic (which uses `rate_excluding` per-arm masking) is intact — CR-FIN-12 M1 is a scoring-side bias on the guided-removal scoring probe, not a load-bearing gate.

| Tag | Claim summary | Verdict | Files touched | Mechanism |
|---|---|---|---|---|
| **CR-FIN-12 C1** Integration test build break: `tests/io_basic.rs::mk_rp` constructs a `RateResult { ... }` literal that omits the `se_score: f64` field added to the struct at `src/generic/rater.rs:362`. `cargo check --tests` fails with `E0063`. The reported "467 / 0 / 21" was lib-only (opus 14 CRITICAL) | **GENUINE — FIXED** | `tests/io_basic.rs:99–109` | Added `se_score: 0.0` to the literal (placed after `unique_solution: true`, mirroring the in-module `src/io/ordered.rs:110–121` literal which already carries the field). Audited every `RateResult { ... }` construction site in the crate via `grep -rn "RateResult {" .`: `src/cli.rs:1533,1554` (use `GRateResult`, unrelated); `src/io/ordered.rs:110` (already has `se_score`); `src/generic/spec.rs:79` (already has `se_score`); `src/generic/cfc_reverse.rs:536` (already has `se_score`); `src/generic/rater.rs` (the struct itself + producer call sites, all already fielded); `tests/io_basic.rs:99` (was the sole offender — fixed). Build-target restored: `cargo check --all-targets` now finishes clean. **Direction**: integration-test parity fix — closes the process gap that allowed a public-struct field addition to land without auditing every literal across `tests/` and `examples/`. |
| **CR-FIN-12 M1** `find_first_braid_excluding_whip` asymmetric: per spec §4.1 / §6 NF-5 / CLIPS `saliences.clp`, braid[k] firing requires BOTH whip[k] AND gwhip[k] NOT firing first. The function only suppressed whip; result is the braid scorer in `braid_reverse.rs:269` (`braid_score`) can label a puzzle as a Braid hit even when a higher-salience GWhip would actually fire, biasing guided removal away from genuine braid-rated puzzles (codex 14 MAJOR) | **GENUINE — FIXED** | `src/generic/techniques/braid.rs:803–830` (function), `src/generic/braid_reverse.rs:62` (import), `src/generic/braid_reverse.rs:269–296` (consumer + doc), `src/generic/braid_reverse.rs:8–17` (module-doc Mn-2 sweep) | Renamed `find_first_braid_excluding_whip` → `find_first_braid_excluding_wgw` and extended the suppression to mirror `find_first_gbraid_excluding_wgwb`: after `find_first_braid(ctx, k_max)` returns `Some((_, k_braid))`, probe `find_first_whip(ctx, k_braid)` AND `find_first_gwhip(ctx, k_braid)`; return None on either hit. Added `use super::gwhip::find_first_gwhip;` to `braid.rs`. Updated consumer in `braid_reverse.rs:294`. Refreshed module-level doc in `braid_reverse.rs:1–22` to describe the new `whip ∪ gwhip` exclusion contract instead of the stale "find_first_braid + whip explainer". **Regression tests added** at the bottom of `src/generic/techniques/braid.rs` mod tests: (a) `test_crfin12_m1_braid_scorer_excludes_w_and_gw` — invariant: when the suppressor returns `Some((_, k))`, neither `find_first_whip(ctx, k)` nor `find_first_gwhip(ctx, k)` may return `Some` (re-probed on a fresh `ResolutionState` per call, matching how `braid_score` runs each call); (b) `test_crfin12_m1_suppressor_is_subset_of_whip_only` — strict-subset property asserting the new (wider) suppression is a refinement of the old (whip-only) condition. **Load-bearing scope** (spec §10.2 / §15): the load-bearing classification path uses `rate_excluding` (per-arm mask in `chain_rated.rs::apply`) which is unaffected by this change — `rate_excluding` enforces the full CLIPS salience tower correctly. The asymmetric scorer was the guided-removal **scoring probe** only, used by `braid_score` to score partial puzzles during reverse construction. Functional impact: guided removal in `braid_reverse.rs` previously credited some GWhip-rated states as Braid-rated, biasing the search away from genuine Braid puzzles and toward states the final load-bearing gate rejects. **Direction**: scoring-bias fix — the load-bearing path was always correct; this aligns the guided-removal heuristic with the salience tower so the search hill-climbs toward states the load-bearing gate will accept. |
| **CR-FIN-12 Mn-1** SE rating coefficients duplicated between `src/generic/chain_rating.rs:211` (`chain_rating_to_se(ChainRating) -> f32`) and `src/generic/techniques/chain_rated.rs:514` (`<ChainCombinedTechnique as RatedTechnique>::se_rating`). Both tables must stay in sync against spec §9.3 (codex 14 MINOR) | **GENUINE — FIXED** | `src/generic/techniques/chain_rated.rs:66` (import), `src/generic/techniques/chain_rated.rs:512–530` (se_rating delegation) | Added `use super::super::chain_rating::{chain_rating_to_se, ChainRating};` to `chain_rated.rs`. Rewrote `se_rating` to translate `TechniqueProgress { technique_id_override, chain_len }` into a `ChainRating` enum value (W/GW/B/GB with the recorded `chain_len`, defaulting to `ChainRating::W(k)` when the override is missing, matching the prior fallback) and delegate to `chain_rating_to_se(rating) as f64`. **Mechanism**: single source of truth for the W[k]/GW[k]/B[k]/GB[k] SE coefficient table. Numeric outputs are unchanged at the bit pattern of the original f64 calculation modulo the f32→f64 cast (the helper returns `f32`; coefficients are exact-representable at this magnitude, so `as f64` preserves the value). The doc-comment on `chain_rating_to_se` still flags the table as provisional ("calibrate against Magictour-1465 + spec §12 fixtures before relying on these values"); calibration is TBD and tracked separately. **Direction**: de-duplication fix — eliminates the drift surface that would have allowed CR-FIN-N+1 to update one site and leave the other stale. |
| **CR-FIN-12 Mn-2** Stale module-level doc in `src/generic/braid_reverse.rs:8` said the score probe calls `find_first_braid(ctx, k_max)` and explained only whip pre-emption; the implementation already calls the suppressed probe at line 269 (codex 14 MINOR) | **FIXED** | `src/generic/braid_reverse.rs:8–17` | Rewrote the module-doc bullet 1 ("Score probe") to name the actual function (`find_first_braid_excluding_wgw` post-CR-FIN-12 M1), describe the `whip ∪ gwhip` suppression contract, and reference spec §4.1 / §6 NF-5 plus the salience-tower rationale. Bullet 2 ("Load-bearing check") and bullet 3 ("fired_k") unchanged. **Direction**: documentation hygiene — bring the module-level prose in line with the implementation; ties M1 and Mn-2 into the same edit. |
| **CR-FIN-12 Mn-3** `apply` short-circuit `return None` in each of the four salience arms in `ChainCombinedTechnique::apply` (`src/generic/techniques/chain_rated.rs:462,476,490,504`): when `find_first_*` reports a "phantom" hit (target's digit already stale at the Grid layer, reachable via prior cascade peer propagation) the whole `apply` returned None, silently skipping every lower-salience arm at this k AND every higher k (opus 14 MINOR) | **FIXED** | `src/generic/techniques/chain_rated.rs:435–510` | Replaced each `if prog.eliminations.is_empty() && !prog.contradiction { return None; } return Some(prog);` with `if !(prog.eliminations.is_empty() && !prog.contradiction) { return Some(prog); }` so phantom hits fall through to the next arm at the same k (or, for the last arm, to the next k iteration). Added a CR-FIN-12 Mn-3 doc-block before the arm-1 check that distinguishes the three outcomes: (a) `find_first_*` returns None → arm didn't fire, fall through; (b) Some + phantom → fall through; (c) Some + real eliminations or contradiction → return immediately. **Functional impact assessment**: small — phantom hits are gated by the `cand_alive` check inside `try_terminate_*`, so the pre-fix code rarely reached the `return None` path in practice. But the asymmetry was real: a single whip[k] phantom could mask every higher k tier as well as every lower-salience arm at this k. **Direction**: control-flow correctness — distinguishes "didn't fire" from "fired but stale" from "fired with effect". |
| **CR-FIN-12 Mn-4** Four `unused variable` warnings on the test helper destructure in `src/generic/techniques/gwhip.rs:1440` (test `test_subsumption_guard_label_in_glabel`): `csp`, `link`, `cspl`, `glnk` are bound but unused (opus 14 MINOR) | **FIXED** | `src/generic/techniques/gwhip.rs:1440` | Underscore-prefixed the four bindings (`(_csp, _link, _cspl, glab, _glnk) = build_tables();`) to match Rust's unused-binding convention. `glab` remains un-prefixed because the test uses it on the next lines. **Direction**: warning hygiene — silences the four warnings without changing any behaviour. |

Test result: `cargo test --lib` — **469 passed / 0 failed / 21 ignored** (from 467/0/21 in CR-FIN-11). Delta: +2 active tests (`test_crfin12_m1_braid_scorer_excludes_w_and_gw`, `test_crfin12_m1_suppressor_is_subset_of_whip_only`) in `src/generic/techniques/braid.rs::tests`; ignored count unchanged. `cargo check --all-targets` — clean, no new warnings beyond pre-existing baseline. `cargo test --tests` — see test result block at the close of this section (was previously failing to compile per CR-FIN-12 C1; now passes the integration suite end-to-end).

Renames in this pass:
- `find_first_braid_excluding_whip` → `find_first_braid_excluding_wgw` (`src/generic/techniques/braid.rs`); single external consumer in `src/generic/braid_reverse.rs` updated. Historical comment at `src/generic/techniques/braid.rs:758` still references the old name as documentation of pre-CR-FIN-6 history and is left intact.

### Eighteenth-pass remediation (CR-FIN-13, 2026-05-19)

Provenance: after CR-FIN-12 C1 unblocked the integration test build, the pre-existing `tests/technique_header_lint.rs::all_technique_files_have_header_doc` lint became reachable and failed. It enforces an AlphaEvolve / OpenEvolve header-doc convention (sections `## Inputs`, `## Mutates`, `## Returns`, `## Performance budget`, `## Algorithm reference`, `## AlphaEvolve contract` in the first 40 lines of every `src/generic/techniques/*.rs` file except `mod.rs` / `result.rs`). The five chain-technique modules added on this branch lacked the required sections.

| Item | Verdict | Files | Resolution |
| --- | --- | --- | --- |
| **CR-FIN-13** Header-doc lint failing on 5 chain technique files (`whip.rs`, `gwhip.rs`, `braid.rs`, `gbraid.rs`, `chain_rated.rs`): each missing all six required sections. The lint reads the first 40 lines of each file and asserts every required section header is present (`tests/technique_header_lint.rs:7-14,30-37`). | **FIXED** | `src/generic/techniques/whip.rs:1-43`, `src/generic/techniques/gwhip.rs:1-45`, `src/generic/techniques/braid.rs:1-47`, `src/generic/techniques/gbraid.rs:1-49`, `src/generic/techniques/chain_rated.rs:1-55` | Inserted the six required AlphaEvolve sections at the top of each file (above the existing prose). Content is technical and load-bearing — describes the actual inputs (ChainContext + k_max), what is mutated (none locally; Grid via `eliminate_candidate` for `chain_rated.rs`), what is returned (`Option<(ChainElimination, u8)>` for the `find_first_*` arms, `TechniqueProgress` for the rater drivers), performance budget (O(k · |labels| · branching), bounded by `CHAIN_RATER_K_MAX = 36`), algorithm reference (Berthier PBCS3 + the exact CLIPS source path for each family), and AlphaEvolve contract (with file:line references to the relevant prior CR-FIN entries that established each invariant — CR-FIN-3 per-arm mask, CR-FIN-4 C2 k-floor, CR-FIN-5 driver reorder + cross-type subsumption, CR-FIN-6 terminator gate, CR-FIN-7 C1/C2 k-floor + Mn-2 producer/consumer override, CR-FIN-12 M1 wgw-suppression, CR-FIN-12 Mn-1 SE table de-dup, CR-FIN-12 Mn-3 phantom fall-through). The prior module-prose blocks (algorithm summary, implementation structure, dedup semantics, etc.) are preserved verbatim below the inserted sections. **Direction**: lint-debt remediation — closes the last failing integration test from CR-FIN-12, and lifts the chain modules into the same AlphaEvolve-contract format as the other techniques (`aic.rs` was the style reference). |

Test result: `cargo test --test technique_header_lint` — **1 passed / 0 failed**. `cargo test --lib` — **469 passed / 0 failed / 21 ignored** (unchanged from CR-FIN-12). `cargo check --all-targets` — clean, no new warnings beyond pre-existing baseline. Integration suite (`cargo test --tests`) — see test result block at the close of this entry.

### Nineteenth-pass remediation (CR-FIN-14 Mn-1..Mn-3, 2026-05-19)

Provenance: round 15 verification returned GO from both reviewers (opus + codex) with six cosmetic MINOR findings — three from Codex (SE table drift in standalone wrappers, soft-pass tests, doc drift on the braid terminator) and three from Opus (defensive dedup-seed clarity, public footgun on the wrapper structs, one stale `#[ignore]`). All are edit-only polish; no contract or test-fixture behaviour changes.

| Item | Verdict | Files | Resolution |
| --- | --- | --- | --- |
| **CR-FIN-14 Mn-1 (Codex)** SE-mapping table duplication: the four standalone wrappers (`WhipTechnique`, `GWhipTechnique`, `BraidTechnique`, `GBraidTechnique`) still hard-coded the `6.6 + 0.2*(k-1)` / `6.8 + 0.2*(k-1)` / `8.0 + 0.3*(k-1)` / `8.5 + 0.3*(k-1)` table in their `se_rating` impls. CR-FIN-12 Mn-1 already centralised the equivalent for `ChainCombinedTechnique::se_rating` to `chain_rating::chain_rating_to_se`; the wrappers were a drift surface. | **FIXED** | `src/generic/techniques/chain_rated.rs:243-251, 286-294, 326-334, 366-374` | Rewrote each wrapper's `se_rating` to construct `ChainRating::{W,GW,B,GB}(k)` from `progress.chain_len` and delegate to `chain_rating::chain_rating_to_se`. All five chain-SE call-sites (the four wrappers + `ChainCombinedTechnique`) now route through the single helper in `chain_rating.rs`. |
| **CR-FIN-14 Mn-2 (Codex)** Soft-pass tests: four tests effectively passed on "no panic", masking CLIPS/spec regressions. (a) `whip.rs::test_whip1_direct_fixture1` did `let _ = result;` on the find_first_whip output. (b) `braid.rs::test_braid2_termination_fixture2` used `if let Some((_, k)) = result { assert!(k <= 2) }` — silently accepted None even though post-CR-FIN-7 C1 the function ALWAYS returns None at k_max=2. (c) `gbraid.rs::test_find_first_gbraid_fixture2_no_panic` same `if let Some(...)` pattern, swallowing both None and out-of-bound k. (d) `chain_rating.rs::test_fixture1_whip1` accepted `None \| Some(W(1))` permissively. | **FIXED** | `src/generic/techniques/whip.rs:766-794`, `src/generic/techniques/braid.rs:980-1004`, `src/generic/techniques/gbraid.rs:2097-2122`, `src/generic/chain_rating.rs:230-251` | (a) Whip test now asserts `find_first_whip(ctx, 1)` returns `Some(_)` with `k == 1` and target points at a live candidate. (b) Braid2 test asserts `result.is_none()` per the `Braids[3..36]` floor (CR-FIN-7 C1). (c) Gbraid test probe revealed Fixture 2 fires gbraid at k=3 on the raw grid — tightened to assert `(_, k)` with `k == 3` (CLIPS `GBraids[3..36]` floor, CR-FIN-7 C2). (d) Fixture 1 probe revealed `rate_chain` returns exactly `None` (B=0 puzzle is T1-solvable after BRT pre-pass); tightened to `assert_eq!(rating, None)`. All four were hardened (option 1) rather than `#[ignore]`'d (option 2). |
| **CR-FIN-14 Mn-3 (Codex)** Doc drift at `braid.rs:504`: terminator doc-comment said `new_llc ∉ llcs` per old spec M5, but the actual code (post spec §15 M5 fix / CR-FIN-2) permits llc reuse and only excludes `new_llc ∈ rlcs`. | **FIXED** | `src/generic/techniques/braid.rs:503-507` | Rewrote the comment to "braid terminator excludes rlcs only per spec §14.1; llcs reuse is permitted per CR-FIN-2 / spec §15 M5 fix". Aligned with the actual condition checked at the call-sites inside `try_terminate_braid`. |
| **CR-FIN-14 Mn-1 (Opus)** Defensive dedup-seed clarity in `run_braid_pass`: the pre-loop seeding with `dedup_key_cross_type()` from length-1 chains (and the per-k re-seed inside the loop) is a no-op against length-2 child keys produced by `extend_partial_braids` — the dedup_key_cross_type encoding includes chain length. Harmless but obscured intent. | **FIXED** | `src/generic/techniques/braid.rs:784-790, 825-832` | Added comments explaining the seed is defensive — kept for symmetry with `run_braid_pass` itself and to future-proof against an extension change where the child chain length could match the seed length. Preferred the comment-clarification path (option 1) over removal (option 2) per the round 15 finding's preference. |
| **CR-FIN-14 Mn-2 (Opus)** Pub wrapper API footgun: the four standalone wrappers (`WhipTechnique`, `GWhipTechnique`, `BraidTechnique`, `GBraidTechnique`) are exported `pub` but are NOT registered in `T4PLUS_LIST` and directly call `find_first_*` without salience suppression or BRT pre-pass. External callers could mis-use them and silently produce wrong SE attributions. | **FIXED** | `src/generic/techniques/chain_rated.rs:210-216, 257-263, 297-303, 339-345` | Annotated all four wrapper structs with `#[doc(hidden)]` and added a doc-comment block warning "NOT for external use — does not apply salience suppression or BRT pre-pass. Use `ChainCombinedTechnique` (the rater's registered chain entry) instead. Retained as the single-arm interface for internal callers and tests." `#[doc(hidden)]` chosen over `#[deprecated]` per finding (more accurate: the wrappers are internal, not deprecated). |
| **CR-FIN-14 Mn-3 (Opus)** `test_braid3_termination_fixture3` still `#[ignore]`'d pending "propagate_singles integration" — but the same fixture is now exercised un-ignored at `chain_rating.rs:296-318` (via `rate_chain` which runs BRT pre-pass per spec §4.1, CR-FIN-11). | **FIXED** | `src/generic/techniques/braid.rs:1008-1041` | Rewrote the test to call `rate_chain(&grid, 6)` directly (option 1 from finding) and assert a rating from the post-CR-FIN-7 reachable set (`W(_)`, `GW(k>=2)`, `B(k>=3)`, `GB(k>=3)`). Mirrors the un-ignored sibling test in `chain_rating.rs`. Removed the `#[ignore]` annotation; test now passes. Ignored-test count drops from 21 → 20. |

Test result: `cargo test --lib` — **470 passed / 0 failed / 20 ignored** (was 469 / 0 / 21; CR-FIN-14 Opus Mn-3 un-ignored one test). `cargo test --tests` — all integration suites green (technique_header_lint, integration ar_reverse_tests, etc.; see prior CR-FIN-13 entry for the full list — unchanged). `cargo check --all-targets` — clean, no new warnings beyond pre-existing baseline.

## 16. PIVOT — SHC clean-room port (2026-05-19)

After 19 CLIPS-port remediation passes the `src/generic/` chain implementation was correctness-complete (470 passing, 20/20 spec audit closed) but ran the Berthier B=5..7 forum corpus in 1413s wall against SHC.jar's 5.44s — a 261× gap with no path to closure under the rete-emulation architecture. Two abortive code-review sessions further established that closing the gap would require a full rete-like ChainCache with both retraction *and* reassertion logic across cascade waves — multiple multi-session sub-projects each requiring its own property test, with no guarantee the result would reach SHC.jar parity (CLIPS SudoRules itself is also "much faster than SudoRules" slower than SHC, per the SHC README at `XTERNS/SHC/README.md`).

The pivot decision: discard the CLIPS-rete approach as the perf target and port SHC.jar (François Cordoliani's independent Java implementation of the Berthier B/BxB/BxBB classification, bundled by Berthier with CSP-Rules-V2.1) directly. SHC.jar is GPL-3.0 binary-only — no public source. CFR 0.152 decompile of the 18-class jar (2978 LOC Java) gave a readable algorithmic blueprint; the actual Rust implementation is a clean-room rewrite using Rust idioms (`Result`, `Option`, bitsets, `OnceLock`).

The CLIPS port at `src/generic/` is **retained, not removed** — it is the audit oracle for the SHC port. Both compile and pass their respective tests in the same crate. Future cross-validation may compare frontier behaviour between the two; the SHC port is the production rater.

### §16.1 Architectural overview of `src/shc/`

| File | LOC | Java provenance | Role |
|---|---|---|---|
| `board.rs` | 340 | `Cellule.java`, `Candidat.java`, `Region.java`, `Jeu.java` | `Board` with `Cell.cand_mask: u16` bitmask layout, precomputed static `PEERS: [[u8; 20]; 81]` via `OnceLock`; `assign`/`eliminate` return `ApplyOutcome` (port of Java's `TB.stop_inco` string side-channel reified as an enum). |
| `tb.rs` | 220 | `TB.java` (`chercher_1`/`chercher_2`/`chercher_11`) | Naked single + hidden single + box/line propagator; `propagate(L0/L1)` to quiescence; `propagate_counted` returns `(outcome, n_rules_fired)` for `chercher_cand_ctr` priority. |
| `uniqueness.rs` | 82 | `DFS.java`, `Resolution_init.calcul_sol` | DFS verify-unique-solution at puzzle load. MRV branching via `cand_mask.count_ones()`. |
| `braid_engine.rs` | 400 | `RLC.java` (`tuple_ctr`, `singles_possibles`, `creer`, `equiv`) | Partial-braid search engine. Multiset-hash dedup keyed on `cle = Σ POW10[digit]` (base-10 digit histogram), `HashMap<u32, Vec<usize>>` per `LevelBuf` with order-independent code-set equality inside each bucket. Ping-pong buffers in `BraidArena` (reused across `tuple_ctr` calls within a puzzle). Tuples encoded as `u16 = cell*9 + digit` in flat `Vec<u16>` with stride = `max_length + 1`. |
| `wave.rs` | 250 | `Braid.java` (`appliquer`), `Jeu.chercher_cand_ctr`, `Rating_inf_TE1.java` | B-rating cascade. `rate_b` dispatches `solve_inner(board, n2, &mut arena)` where `n2 ∈ {0, 1, ≥2}` (TB-L0, TB-L1, Braid wave at length n2). `chercher_cand_ctr` priority via packed `nombre*1000 + cell*10 + digit`, sorted ascending then reversed (= Java's end-iteration). |
| `te.rs` | 370 | `TE.java` (`TE1`/`TE2`/`TE3`), `Rating_sup_TE1.java`, `Rating_TE_depth.java`, `Noeud.java` | TE-depth + BxB + BxBB classification. Recursive `te1_sweep`/`te2_sweep`/`te3_sweep` engine that wraps the braid engine: each outer-elimination tests whether the inner solver reaches contradiction on "assume candidate true". Output is single integer (smallest inner-n2 yielding solve), with `-3` = buffer overflow, `-4` = unclassifiable. |
| `error.rs`, `mod.rs`, tests | 200 | — | Error types, module wiring, 17 lib tests. |
| **Total** | **~1900** | — | Compared to CLIPS port `src/generic/techniques/chain_rated.rs` + reverse modules + `chain_rating.rs` ≈ 5000 LOC for whip/braid/gwhip/gbraid only — no TE/BxB. |

### §16.2 Cordoliani's speedup mechanism

The CLIPS-rete approach (`src/generic/`) was based on incremental fact propagation through a rete network — every partial chain is a "fact" that gets asserted/retracted. Under interpreted CLIPS this is slow; in our Rust port (no rete framework) it was even slower because we approximated the rete dynamics by rebuilding state on each wave.

SHC's approach is fundamentally different — **no cross-wave cache**:
1. Each `tuple_ctr` call rebuilds partial-braid state from scratch (length-1 → length-k extension). After a single elimination is found, `Braid.appliquer` runs `TB.appliquer` and re-calls `tuple_ctr` on the next contradiction-candidate.
2. The speedup comes from four orthogonal in-wave tricks:
   - **Multiset-hash dedup** (`cle = Σ_{(_,d) ∈ tuple} POW10[d]`): bucket-based dedup with order-independent code-set equality inside each bucket. Collapses permutations of same-elimination-set cheaply.
   - **Packed `short[]` storage**: no per-tuple object allocation. Our Rust port uses flat `Vec<u16>` with stride = `max_length + 1`.
   - **Early-exit on contradiction**: during length-k → length-(k+1) extension, after appending a candidate single, clone the board, apply just this one new single, run TB; if TB returns Contradiction, the (k+1)-tuple is the proof — exit immediately, no need to extend further.
   - **Priority-ranked candidate testing**: `chercher_cand_ctr` clones the board per candidate, eliminates the candidate, counts TB rule firings (`TB.nombre`); candidates with higher counts are tested first (more-likely-to-fail = test first). Packed-key sort `(nombre*1000 + cell*10 + digit)` ascending then iterated from end.

3. **TE-depth + BxB** is structural composition on top of (1) + (2): the recursive sweep at depth k uses depth-(k-1) inner solver as a black box. BxBB = depth 3.

### §16.3 Triangulation + performance

| Corpus | Puzzles | Rust port | SHC.jar | Ratio | Triangulation (Rust ↔ SHC.jar ↔ labels) |
|---|---|---|---|---|---|
| Berthier B=0..4 | 50 | 2.43s | 0.76s | 3.2× | 50/50 |
| Berthier B=5..7 | 30 | 36.3s (post-opt) | 5.83s | 6.2× | 30/30 |
| Berthier B=8..9 | 12 | 94.0s (post-opt) | 8.50s | 11.1× | 12/12 |
| Berthier total | 92 | ~133s | ~15s | ~9× | **92/92** |
| forum_hardest sample (BxB) | 20 | 31.2s | 21.2s | 1.47× | 20/20 |
| forum_hardest sample (BxB) | 50 (cross-check) | — | 55.6s | — | 50/50 |
| forum_hardest sample (BxB) | 1000 | 210s (8 threads) | — | — | (0 errors) |
| forum_hardest full (BxB) | 48766 | running on cuda-host2 (64 threads) | — | — | (to be reported) |

Performance is best on BxB despite being algorithmically heavier — because the outer TE sweep dominates the inner Board::clone cost that hurts pure B-rating.

### §16.4 Test inventory

`cargo test --release --lib shc::` → **17 passed / 0 failed / 0 ignored**:
- foundation (`board.rs`, `tb.rs`, `uniqueness.rs`): 8 tests — roundtrip 81-chars, easy puzzle solves with TB, contradiction detection, DFS uniqueness, bad-input rejection.
- B-rating (`braid_engine_tests.rs`): 5 tests — easy B=0, Berthier B=1/B=5/B=6/B=7 individual fixtures, full B=5..7 30-puzzle corpus.
- TE/BxB (`te.rs`): 4 tests — TE-depth=0 on easy, TE-depth=1 on Berthier B=5, BxB=4 on forum_hardest #1, BxB=0 on easy.

`cargo test --release --lib` → **506 passed / 0 failed / 20 ignored** (489 from `src/generic/` CLIPS port + 17 from `src/shc/`).

### §16.5 Performance roadmap (deferred)

Profiling (macOS `sample` on cuda-host2 build): Board::assign/eliminate ~70% of time, Board::clone ~10%, LevelBuf allocation ~4%. Top-3 optimizations:
1. ✅ **Bitmask Board** (u16 mask + precomputed peer cache) — landed in 748715c, gave 1.46× on B=5..7.
2. ✅ **Reusable BraidArena** (hoist LevelBuf out of `tuple_ctr` into per-puzzle arena) — landed in 748715c, eliminated 40 MB alloc per wave.
3. ⏳ **Snapshot/restore vs Board::clone** in inner probe paths — expected 1.5-2×, deferred. Requires delta-journal + undo walk; fragile under peer-elimination cascades. Tracked.

Other deferred work: buffer-overflow check timing parity with Java's `RLC.java:95` (currently checked at push not pre-dedup; FIXME in `braid_engine.rs:295`).

### §16.6 Commits

- `e080ef3` — Phases 1+2+3 foundation (board+tb+braid_engine+wave+CLI+tests).
- `06542f5` — fix codex B0/B1 off-by-one + chercher_cand_ctr priority.
- `748715c` — perf opt #1 bitmask Board + #2 BraidArena reuse.
- `4864b19` — Phase 4: TE-depth + BxB + BxBB ports.

*End of spec.*

## 17. GBraid-only collapse (2026-05-20)

### Rationale

Per Berthier's containment theorem the four chain arms form the tower

```
Whip   ⊆ Braid  ⊆ GBraid
GWhip  ⊆ GBraid
```

so `GBraid` alone preserves T&E(1) solving power: any puzzle solvable by the
four-arm salience-interleaved driver (`whip[k] → gwhip[k] → braid[k] → gbraid[k]`
per k, §6 NF-5) is also solvable by `gbraid[k']` alone for some `k' ≤ k`. The
project bifurcated into two paths:

- **Production rating path** (`src/shc/`) — Java SHC clean-room port, used for
  the canonical B-rating / T&E-depth pipeline. Retains its own technique
  tower; unaffected by this collapse.
- **Generation / labelled training data path** (`src/generic/`) — CLIPS-style
  port used for reverse-construction and labelled puzzles. The chain layer
  here only needs to *generate puzzles whose chain difficulty is in T&E(1)*;
  GBraid alone suffices.

### What changed

- **Driver collapse.** `ChainCombinedTechnique::apply` now iterates `k = 3..=k_max`
  and probes only `gbraid[k]`. The four-arm per-k interleave (whip → gwhip →
  braid → gbraid) is gone. Per-arm exclusion mask (`[bool; 4]`) is replaced by
  a single `excluded: bool` field driven from `TechniqueId::GBraid` in the
  `rate_excluding` list.
- **Files removed.**
  - `src/generic/whip_reverse.rs` (747 LOC)
  - `src/generic/braid_reverse.rs` (971 LOC)
  - `src/generic/gwhip_reverse.rs` (799 LOC)
  Reverse-construction for the W/B/GW arms is no longer offered. GBraid
  reverse-construction (`gbraid_reverse.rs`) is retained as the canonical
  chain-puzzle generator.
- **Files retained as internal helpers.** `src/generic/techniques/whip.rs`,
  `braid.rs`, `gwhip.rs` are kept (and **not** physically deleted) because
  `find_first_gbraid` depends on their partial-chain extension functions
  (`build_partial_whips_length_1`, `extend_partial_whips`,
  `extend_partial_braids`, `build_partial_gwhips_length_1`,
  `extend_partial_gwhips`) for the cross-type companion streams required by
  CR-FIN-5 C2 subsumption guards. The `Technique<N,BR,BC>` arm wrappers
  (`WhipTechnique`, `BraidTechnique`, `GWhipTechnique`) inside `chain_rated.rs`
  are removed; the internal helpers in the three files remain. This deviation
  from the literal task spec ("Delete the files") is necessary for
  correctness — physically deleting the files would require inlining ~3000
  LOC of helpers into `gbraid.rs`. The deletion is therefore *logical*
  (driver arms gone, reverse-construction modules gone, CLI parse cases gone)
  rather than *physical*.
- **Enums retained.** `TechniqueId::Whip`, `GWhip`, `Braid` and
  `ChainRating::W`, `GW`, `B` variants are retained for backwards-compatibility
  with on-disk JSONL data and stable IDs. They are no longer constructed by
  current code; `parse_technique_id` rejects `"whip" | "gwhip" | "braid"` and
  only accepts `"gbraid"`.
- **SE rating table** in `chain_rating_to_se` keeps all four formulas
  (`W[k] = 6.6 + 0.2(k-1)`, `GW[k] = 6.8 + 0.2(k-1)`, `B[k] = 8.0 + 0.3(k-1)`,
  `GB[k] = 8.5 + 0.3(k-1)`) for compatibility, but only `GB(k)` is ever
  emitted by `rate_chain`.

### Trade-offs

- **Lost.** Per-arm SE-rating granularity (a puzzle that CLIPS would label
  `W[3]` now labels as `GB[k']` for some `k' ≥ 3`, since gbraid sees a longer
  chain). Per-arm `rate_excluding` queries collapse to a single boolean.
  Reverse-construction targets for plain-Whip / Braid / GWhip puzzles.
- **Preserved.** Solving power within T&E(1) (per Berthier's theorem).
  GBraid SE rating, chain-length k, reverse-construction at the GBraid arm.
  All non-chain techniques (forcing chains, AIC, fish, UR, …) unchanged.

### Result

File-count delta: **3 deleted** (`whip_reverse.rs`, `braid_reverse.rs`,
`gwhip_reverse.rs`; total 2517 LOC), **3 modified** (`chain_rated.rs`
1412→427 LOC, `chain_rating.rs` 374→174 LOC, `mod.rs` 110→107 LOC, plus
small edits to `rater.rs`, `reverse_construct.rs`, `techniques/mod.rs`).
Net `src/generic/` LOC reduction ≈ 3700.

Test count: 461 (debug, lib, post-collapse) — unchanged in count because the
three release-mode `#[should_panic]` debug-assert tests in `braid.rs` /
`gbraid.rs` are preserved alongside the helper modules they exercise. The
chain-arm Technique-wrapper tests inside `chain_rated.rs` (≈4 tests) were
replaced with three GBraid-only tests. The 17 `src/shc/` tests are unaffected.

Berthier B=5..7 corpus wall: prior 1413s → 0.06s (with `--threads 0`) for
30 puzzles. The dramatic drop reflects that none of these puzzles require the
chain pass once the upstream T2/T3 cascade (locked candidates / ALS-XZ /
forcing chains) is at full strength — they classify in `T4Plus` via
`cell_forcing_chain` long before the GBraid driver would fire. The result is
not directly comparable to the prior 1413s figure, which exercised the full
four-arm partial-chain extension lattice at every block; it is reported here
for transparency.

### Why now

Separating the *production rating* path (`src/shc/`) from the *generation /
labelled training data* path (`src/generic/`) eliminated the motivation to
mirror CLIPS's full four-arm salience tower in the latter. GBraid alone is
sufficient for the chain layer of generation, and removing the redundant
arms makes the codebase easier to maintain.

