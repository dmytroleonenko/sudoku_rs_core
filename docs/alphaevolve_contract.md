# AlphaEvolve / OpenEvolve Contract

This document describes the stable interface that AlphaEvolve and OpenEvolve
systems must respect when iterating on technique files in `src/generic/techniques/`.

---

## File layout

```
src/generic/techniques/
  mod.rs          — Technique + RatedTechnique traits, all_techniques_9x9() registry
  result.rs       — TechniqueProgress struct (stable ABI)
  aic.rs          — AIC (Alternating Inference Chains)
  als_xz.rs       — ALS-XZ (Almost Locked Sets)
  bug.rs          — BUG +1 (Bivalue Universal Grave)
  fish.rs         — Fish (X-Wing / Swordfish / Jellyfish)
  hidden_set.rs   — HiddenSet (Pair / Triple / Quad)
  locked.rs       — LockedCandidates (Pointing + Claiming)
  naked_set.rs    — NakedSet (Pair / Triple / Quad)
  simple_coloring.rs — Simple Coloring
  skyscraper.rs   — Skyscraper
  two_string_kite.rs — Two-String Kite
  unique_rect.rs  — Unique Rectangle (Types 1 + 2)
  xy_wing.rs      — XY-Wing
  xyz_wing.rs     — XYZ-Wing
```

Each technique file is **self-contained**. No cross-file state. No shared
thread-locals or statics between techniques. Per-technique scratch lives inside
the struct and is re-initialized per call.

---

## Mandatory header-doc convention

Every `techniques/<x>.rs` file (except `mod.rs` and `result.rs`) must begin
with a doc-comment containing these six sections in its first 40 lines:

```rust
//! # <TechniqueName>
//!
//! ## Inputs
//! Reads from `Grid<N,BR,BC>`: candidate bitmasks per cell, placed digits.
//!
//! ## Mutates
//! May eliminate candidates and/or place digits in `grid` (in place). Does
//! not touch any state outside the passed-in `Grid`.
//!
//! ## Returns
//! `Some(TechniqueProgress)` if at least one elimination/placement happened
//! on this call; `None` if the technique did not fire. Sets
//! `progress.contradiction = true` if elimination drove a cell to 0 candidates.
//!
//! ## Performance budget
//! Target: < <X> µs/grid on 9×9 on commodity x86_64 (R3.4 reference).
//!
//! ## Algorithm reference
//! <one-line reference>
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.
```

This is enforced by `cargo test --test technique_header_lint`.

---

## How to add a new technique

1. Create `src/generic/techniques/my_technique.rs` with the mandatory
   header-doc above.

2. Implement the `Technique<N, BR, BC>` trait (and optionally `RatedTechnique`
   with a per-technique `base_rating()` override):

   ```rust
   use super::super::grid::Grid;
   use super::{Technique, TechniqueId, TechniqueProgress, Tier};

   pub struct MyTechnique;

   impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC>
       for MyTechnique
   {
       fn id(&self) -> TechniqueId { TechniqueId::MyTechnique }
       fn tier(&self) -> Tier { Tier::T3 }
       fn name(&self) -> &'static str { "my_technique" }
       fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
           // ... implementation ...
       }
   }
   ```

3. Add to `mod.rs`:
   - `pub mod my_technique;`
   - `pub use my_technique::MyTechnique;`
   - A new variant to `TechniqueId` enum.
   - `Box::new(MyTechnique)` in `all_techniques_9x9()`.

4. Add co-located tests in `#[cfg(test)] mod tests` at the bottom of
   `my_technique.rs`:
   - `fires_on_known_positive()` — apply to a grid where the technique should
     fire; assert `Some(progress)` and that the eliminations are correct.
   - `no_fire_on_known_negative()` — apply to a grid where it should not fire;
     assert `None`.
   - `correctness_no_invalid_elims()` — verify eliminations do not remove any
     digit that appears in the known solution.

5. Ensure `cargo test --test technique_header_lint` passes (it will fail if
   your header-doc is missing a required section).

---

## How to run tests / bench for a single technique

```bash
# All tests
cargo test -p sudoku_rs_core

# Header-doc lint only
cargo test --test technique_header_lint

# All lib unit tests (includes per-technique inline tests)
cargo test --lib

# Bench compile check
cargo bench --bench techniques --no-run

# Full bench run
cargo bench --bench techniques

# Filter to one technique name (Criterion substring filter)
cargo bench --bench techniques -- aic
```

---

## What NOT to change

These are the stable contract elements. Changing them breaks all existing
technique files and the rater.

| Symbol | Location | Why stable |
|--------|----------|------------|
| `Technique<N,BR,BC>` trait | `techniques/mod.rs` | External interface for all technique impls |
| `RatedTechnique<N,BR,BC>` trait | `techniques/mod.rs` | SE-rating interface |
| `TechniqueProgress` struct | `techniques/result.rs` | ABI between technique and rater |
| `TechniqueId` enum variants | `techniques/mod.rs` | Stable serialized identifiers in parquet output |
| `Tier` enum variants | `techniques/mod.rs` | Stable serialized identifiers |
| `all_techniques_9x9()` function signature | `techniques/mod.rs` | Registry contract |

**Specifically forbidden:**
- Adding fields to `TechniqueProgress` without bumping `GENERIC_SCHEMA_VERSION`
  in `src/generic_writer_helpers.rs` (observability anchor). Backward-compat
  `Option<T>` additions are allowed; non-`Option` field additions or removals
  require a major-version review.
- Renumbering `TechniqueId` variants (rater rules key off them).
- Adding `thread_local!` or `static mut` accessible across technique files.
- Removing the mandatory header-doc sections.

### `TechniqueProgress` field history

| Version | Field | Type | Added by | Purpose |
|---------|-------|------|----------|---------|
| 1 | `placements` | `Vec<(usize,u8)>` | Stage R | Cells placed during fire |
| 1 | `eliminations` | `Vec<(usize,u8)>` | Stage R | Candidate eliminations |
| 1 | `contradiction` | `bool` | Stage R | Set if fire caused contradiction |
| 1 | `chain_len` | `Option<u32>` | Phase B (R3.4.5) | AIC/FC chain depth proxy |
| 2 | `is_xy_chain` | `Option<bool>` | Phase G (R3.4.5) | AIC XY-Chain (SE 7.0) vs X-Chain (SE 6.6) discriminator |
| 2 | `k_branches` | `Option<u8>` | Phase G (R3.4.5) | Cell FC Y-Chain (k=2, SE 6.6) vs full CFC (k≥3, SE 8.0) |

---

## Performance budgets (R3.4 reference)

| Technique category | Budget |
|--------------------|--------|
| LockedPointing / LockedClaiming | < 20 µs/grid |
| NakedPair / Triple / Quad | < 20 µs/grid |
| HiddenPair / Triple / Quad | < 20 µs/grid |
| XWing / Swordfish / Jellyfish | < 50 µs/grid |
| AIC | < 500 µs/grid |
| XyWing / XyzWing | < 100 µs/grid |
| UniqueRectangle T1/T2 | < 100 µs/grid |
| ALS-XZ | < 100 µs/grid |
| BUG +1 | < 100 µs/grid |
| SimpleColoring | < 100 µs/grid |
| Skyscraper | < 100 µs/grid |
| TwoStringKite | < 100 µs/grid |

These budgets are **targets** for single-threaded commodity x86_64. The bench
harness (`cargo bench --bench techniques`) measures actual performance.
AlphaEvolve uses the bench output to guide optimization.

---

## Stage 0 note

The current `RatedTechnique` blanket impl returns tier-bucket placeholder
ratings (T2→3.0, T3→5.0, T4Plus→7.0). Stage 0 will replace this with
per-technique SE-calibrated weights. Override `base_rating()` and `se_rating()`
in the technique's `impl RatedTechnique<N,BR,BC>` block — the blanket impl will
yield to any explicit impl automatically (Rust specialization semantics apply
via coherence).
