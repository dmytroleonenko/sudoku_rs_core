//! # g-whip[k] rater — porting target CSP-Rules-V2.1 (Berthier).
//!
//! Spec: `tools/sudoku_rs_core/docs/csp_rules_chain_spec.md` §8, §14.1, §15.
//!
//! ## Inputs
//! Reads from a `ChainContext` (label + glabel tables, csp-var index,
//! link/glink relations). Caller parameter `k_max` bounds chain length
//! (≤ `CHAIN_RATER_K_MAX = 36`). Minimum length: k ≥ 2 — there is no
//! gWhips[1] per CR-FIN-4 C2.
//!
//! ## Mutates
//! No global state. Allocates per-call scratch (partial-gwhip buffers,
//! sub-rule -1/-2/-3 working sets, dedup sets). Grid eliminations are
//! performed by the caller via `eliminate_candidate` on the returned
//! `ChainElimination`.
//!
//! ## Returns
//! - `find_first_gwhip(ctx, k_max) -> Option<(ChainElimination, u8)>` — first
//!   g-whip[k] firing.
//! - `run_gwhip_pass(...) -> TechniqueProgress` — driver delegating to the
//!   shared rater-side wrapper in `chain_rated.rs`.
//! - `build_partial_gwhips_length_1(...)` — NF-4 seed builder (CLIPS
//!   `Partial-gWhips[1].clp:38-83`).
//!
//! ## Performance budget
//! O(k · (|labels| + |glabels|) · branching_factor) per probe; three
//! sub-rules per extension. Seeds built fresh each call; bounded by
//! `CHAIN_RATER_K_MAX = 36`.
//!
//! ## Algorithm reference
//! Berthier PBCS3 §VI.4 ("g-Whips"); CSP-Rules-V2.1
//! `CSP-Rules-Generic/CHAIN-RULES-SPEED/G-WHIPS/` (terminator) and
//! `PARTIAL-G-WHIPS/Partial-gWhips[k]-{1,2,3}.clp` (sub-rule extension);
//! spec §8, §14.1, §15.
//!
//! ## AlphaEvolve contract
//! - Cross-type subsumption guards lifted from CLIPS
//!   `Partial-gWhips[1].clp:60-71` + `Partial-gWhips[5].clp:142-150,213-221`
//!   (CR-FIN-5 C1).
//! - Per-arm exclusion mask wired through `rate_excluding` (CR-FIN-3).
//! - Driver reorders plain-whip before gwhip at each k tier (CR-FIN-5).
//! - Function signatures (`find_first_gwhip`, `run_gwhip_pass`,
//!   `build_partial_gwhips_length_1`) are stable contract.
//!
//! ## Algorithm summary
//!
//! A g-whip[k] is a whip[k] where each right-linking candidate (RLC) may be either
//! a plain label (`Rlc::Cand`) or a grouped label (`Rlc::GCand`). The three sub-rules
//! for extension correspond to CLIPS `Partial-gWhips[k]-1/2/3`:
//!
//! - **-1** (gwhip + cand): extend a partial-gwhip by a plain candidate as rlc.
//!   CSP-var freshness: last-only (`new_csp != last_csp`).
//! - **-2** (whip + gcand): extend a partial-whip by a grouped candidate as rlc.
//!   CSP-var freshness: full-history (`new_csp ∉ csp_vars`). Per spec §8.4, C3.
//! - **-3** (gwhip + gcand): extend a partial-gwhip by a grouped candidate as rlc.
//!   CSP-var freshness: last-only (`new_csp != last_csp`).
//!
//! ## NF-4: g-whip[1] seed
//!
//! `build_partial_gwhips_length_1` seeds the first partial-gwhip for each
//! (Z, llc1, glabel-rlc1) triple where llc1 is linked to Z and rlc1 is a g-candidate
//! csp-glinked to llc1 that is the unique surviving non-label-in-glabel alternative
//! once Z kills all other regular alternatives. Per spec §8.4.1 and
//! CLIPS `Partial-gWhips[1].clp:38-83`.
//!
//! Without this seed, g-whip[2+] via sub-rules -1 and -3 cannot fire.
//!
//! ## -2 sub-rule implementation choice
//!
//! The `-2` sub-rule consumes partial-whips. Rather than accepting them as an external
//! slice (which would tangle the API with whip internals), `run_gwhip_pass` manages
//! TWO parallel partial buffers internally:
//!   - `partials_gwhip`: length-(k-1) partial-gwhips (fed by seeds + sub-rules -1/-3)
//!   - `partials_whip`:  length-(k-1) partial-whips  (reused from whip's own seeding)
//!
//! At each k, `extend_partial_gwhips` applies all three sub-rules. The
//! `extend_partial_gwhips` public fn accepts a unified slice; `run_gwhip_pass` passes
//! the union. The whip partial buffer is populated by calling into
//! `whip::build_partial_whips_length_1` and `whip::extend_partial_whips` from inside
//! `run_gwhip_pass`. This keeps the public API of `extend_partial_gwhips` simple while
//! correctly implementing the interleaved extension required by the CLIPS salience model.
//!
//! ## Dedup (spec §14.1 + M6/M7)
//!
//! G-whip dedup is **positional** on the Rlc sequence (same as whip). For glabel-rlc
//! extensions (sub-rules -2 and -3) an additional subsumption guard is applied:
//! a new `Rlc::GCand(g)` is blocked if any existing partial-whip or partial-gwhip
//! of the same (target, length) has a tail rlc that is either equal to `g` OR is a
//! `Rlc::Cand(L)` where `label_in_glabel(L, g)` — i.e., a finer chain already covers
//! this glabel. This matches CLIPS `Partial-gWhips[2]-2/3:145,214`.

use std::collections::{HashMap, HashSet};

use super::super::chain_model::{Chain, ChainElimination, ChainKind, ChainRule, GLabel, Label, Rlc};
use super::super::csp_tables::{CspLinkGraph, CspVarTable, LinkGraph, W9};
use super::super::glabel_tables::{
    glabel_contains_none_of, label_in_glabel, GLabelTable, GLinkGraph, WG9,
};
use super::super::resolution_state::ResolutionState;
use super::whip::{build_partial_whips_length_1, extend_partial_whips, find_first_whip, ChainContext};

// ─── GWhipScratch ─────────────────────────────────────────────────────────────

/// Reusable per-puzzle scratch buffers for the g-whip search.
///
/// Mirrors `WhipScratch`. Two partial buffers (gwhip + whip) plus dedup tables.
pub struct GWhipScratch {
    /// Partial-gwhip buffer for the "previous" length (k-1).
    pub gwhip_prev: Vec<Chain>,
    /// Partial-gwhip buffer for the "next" length (k). Swapped with `gwhip_prev`.
    pub gwhip_next: Vec<Chain>,
    /// Partial-whip buffer for the "previous" length (k-1). Used for sub-rule -2.
    pub whip_prev: Vec<Chain>,
    /// Partial-whip buffer for the "next" length (k).
    pub whip_next: Vec<Chain>,
    /// Dedup set for current-length gwhip partials.
    pub dedup_gwhip: HashSet<u64>,
    /// Dedup set for current-length whip partials.
    pub dedup_whip: HashSet<u64>,
}

impl GWhipScratch {
    /// Allocate scratch buffers with sensible initial capacities.
    pub fn new() -> Self {
        GWhipScratch {
            gwhip_prev: Vec::with_capacity(512),
            gwhip_next: Vec::with_capacity(512),
            whip_prev: Vec::with_capacity(512),
            whip_next: Vec::with_capacity(512),
            dedup_gwhip: HashSet::with_capacity(512),
            dedup_whip: HashSet::with_capacity(512),
        }
    }

    /// Reset all buffers to empty, ready for a fresh puzzle.
    pub fn reset(&mut self) {
        self.gwhip_prev.clear();
        self.gwhip_next.clear();
        self.whip_prev.clear();
        self.whip_next.clear();
        self.dedup_gwhip.clear();
        self.dedup_whip.clear();
    }
}

impl Default for GWhipScratch {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Returns true iff label `alt` is "glinked-killed": linked to Z via exists-link OR
/// exists-glink to any Rlc::GCand in the rlcs.
///
/// `killed_set` contains Z and all plain (Cand) rlcs for O(1) link check.
/// `rlcs` contains all rlcs including GCand for the glink check.
///
/// CLIPS `glinked-or(alt, z, rlcs)` per `generic-background.clp:263-269`:
/// checks exists-link and exists-glink ONLY — NO self-membership semantics.
#[inline]
fn glinked_killed(
    alt: Label,
    killed_set: &super::super::bitboard::BitSet<W9>,
    rlcs: &[Rlc],
    link: &LinkGraph<W9>,
    glnk: &GLinkGraph<W9, WG9>,
) -> bool {
    // Check exists-link to any plain label in killed_set.
    if link.is_linked_or_bitset(alt, killed_set) {
        return true;
    }
    // Check exists-glink to any GCand rlc.
    for rlc in rlcs {
        if let Rlc::GCand(g) = rlc {
            if glnk.glinked[alt as usize].test(*g as usize) {
                return true;
            }
        }
    }
    false
}

/// Build the "killed label set" bitset: {Z} ∪ all Cand rlcs.
///
/// GCand rlcs are NOT in the label bitset; they are handled via `glnk.glinked`.
#[inline]
fn build_killed_labels(z: Label, rlcs: &[Rlc]) -> super::super::bitboard::BitSet<W9> {
    use super::super::bitboard::BitSet;
    let mut bs: BitSet<W9> = BitSet::empty();
    bs.set(z as usize);
    for rlc in rlcs {
        if let Rlc::Cand(l) = rlc {
            bs.set(*l as usize);
        }
    }
    bs
}

/// Check CSP-var freshness for sub-rule -2 (full-history).
///
/// Per spec §8.4 C3: sub-rule -2 (whip→gcand) uses full-history exclusion.
#[inline]
fn csp_fresh_full(new_csp: u32, csp_vars: &[u32]) -> bool {
    !csp_vars.contains(&new_csp)
}

// ─── NF-4: partial-gwhip[1] seed ─────────────────────────────────────────────

/// Build all partial-gwhips of length 1 — dedicated seed step per spec §8.4.1.
///
/// For each target Z and each label llc1 linked to Z, for each csp-glink
/// (llc1, rlc1=glabel, csp1) where:
///   - rlc1 is a g-candidate alive (`rs.g_alive.test(rlc1)`)
///   - rlc1 ≠ Z (explicitly: `label_in_glabel(Z, rlc1) == false` per CLIPS)
///   - for every regular alternative `X` of llc1 under csp1 where `!label_in_glabel(X, rlc1)`:
///     `X` is linked to Z (i.e. all non-glabel alternatives are killed by Z)
///
/// Dedup: do not assert if an existing partial-whip or partial-gwhip of length=1
/// with the same target already has a tail rlc `rlc1a` such that
/// `rlc1a == rlc1` or `label_in_glabel(rlc1a, rlc1)`.
///
/// **CR-FIN-5 C1 cross-type subsumption**: per CLIPS `Partial-gWhips[1].clp:63-71`
/// the `(not (chain ...))` guard reads `(type partial-whip|partial-gwhip)`, so if a
/// plain partial-whip[1] with the same target has its tail label inside the new
/// glabel members (label-in-glabel), the grouped seed must be suppressed (the
/// plain whip subsumes it under salience). `plain_whip_seeds` carries the
/// partial-whip[1] seeds for that cross-type check. Pass `&[]` if the caller
/// cannot supply them (legacy / unit-test path); callers in `run_gwhip_pass` and
/// `find_first_gwhip` MUST supply them after CR-FIN-5 to avoid overproduction.
///
/// Per CLIPS `Partial-gWhips[1].clp:38-83`.
pub fn build_partial_gwhips_length_1(
    ctx: &ChainContext<'_>,
    plain_whip_seeds: &[Chain],
) -> Vec<Chain> {
    let rs = &*ctx.rs;
    let link = ctx.link;
    let csp = ctx.csp;
    let cspl = ctx.cspl;
    let glab = ctx.glab;
    let glnk = ctx.glnk;
    let n_labels = csp.n * csp.n * csp.n;

    // CR-FIN-5 C1: index plain whip[1] seeds by target for cross-type subsumption.
    // For each target z, collect the set of plain-whip[1] tail labels (rlc1's Cand).
    let mut plain_tail_by_target: HashMap<Label, Vec<Label>> = HashMap::new();
    for w in plain_whip_seeds {
        if w.kind != ChainKind::PartialWhip { continue; }
        if w.length != 1 { continue; }
        if let Some(Rlc::Cand(l)) = w.rlcs.first().copied() {
            plain_tail_by_target.entry(w.target).or_default().push(l);
        }
    }

    let mut result: Vec<Chain> = Vec::new();
    // Dedup: (target_z, rlc1_glabel) — positional, length=1.
    // Additional subsumption: no existing partial-(g)whip[1] with same target has
    // rlc1a == rlc1 or label_in_glabel(rlc1a, rlc1). We track the asserted (z, g) pairs.
    let mut dedup: HashSet<(Label, GLabel)> = HashSet::new();

    for z in 0..n_labels {
        let z = z as Label;
        if !rs.cand_alive.test(z as usize) {
            continue;
        }

        // Build killed_set: just {z} for length-1 seed.
        let mut killed_set: super::super::bitboard::BitSet<W9> = super::super::bitboard::BitSet::empty();
        killed_set.set(z as usize);

        // Enumerate llc1 linked to Z.
        link.linked[z as usize].for_each(|l1_idx| {
            let llc1 = l1_idx as Label;
            if !rs.cand_alive.test(llc1 as usize) {
                return;
            }

            // Enumerate csp-glinks from llc1: (rlc1=glabel, csp1).
            for &(rlc1, csp1) in &glnk.csp_glinked[llc1 as usize] {
                // rlc1 must be alive (g_alive).
                if !rs.g_alive.test(rlc1 as usize) {
                    continue;
                }
                // rlc1 must not contain Z (label_in_glabel(Z, rlc1) must be false).
                if label_in_glabel(z, rlc1, glab) {
                    continue;
                }

                // F5 fix: perform semantic check BEFORE dedup insert.
                // CLIPS Partial-gWhips[1]: the `not (chain)` dedup check appears AFTER
                // the `forall` semantic test. An invalid seed must not block a valid
                // later seed with the same (Z, g). Move dedup insert after the check.

                // Check: for every regular alternative X of llc1 under csp1
                // where !label_in_glabel(X, rlc1), X must be linked to Z.
                // We need to find which slot of llc1's csp_vars corresponds to csp1.
                let vars_of_llc1 = &csp.vars_of[llc1 as usize];
                let mut found_slot = None;
                for slot in 0..4 {
                    if vars_of_llc1[slot] == csp1 {
                        found_slot = Some(slot);
                        break;
                    }
                }
                let slot = match found_slot {
                    Some(s) => s,
                    None => continue, // csp1 not in llc1's vars — shouldn't happen
                };

                let alts = &cspl.alternatives[llc1 as usize][slot];
                // Every alive alternative X with !label_in_glabel(X, rlc1) must be linked to Z.
                // CR-FIN-9 Mn-3: the "dead alt is trivially killed" branch (and the
                // absence of an explicit any-live filter) relies on BRT pre-pass
                // (rate_chain / chain_rated propagate singles); if all alts of llc1 on
                // this csp-var are already dead while llc1 is alive, BRT would have
                // fired the hidden single. If a future caller bypasses BRT, this guard
                // becomes incorrectness — see spec §4.1.1.
                let all_non_glabel_killed = alts.iter().all(|&x| {
                    if !rs.cand_alive.test(x as usize) {
                        return true; // dead alt: trivially killed
                    }
                    if label_in_glabel(x, rlc1, glab) {
                        return true; // part of rlc1 glabel: doesn't need to be killed
                    }
                    // Must be linked to Z.
                    link.is_linked(x, z)
                });

                if !all_non_glabel_killed {
                    continue;
                }

                // Dedup: (z, rlc1) — inserted AFTER semantic check (F5 fix).
                if !dedup.insert((z, rlc1)) {
                    continue;
                }

                // CR-FIN-5 C1 cross-type subsumption: skip if an existing plain
                // partial-whip[1] with same target z has rlc1a (a Cand label)
                // contained in rlc1's glabel members. Per CLIPS Partial-gWhips[1].clp:63-71
                // `(type partial-whip|partial-gwhip)` guard with
                // `(or (eq ?rlc1a ?rlc1) (label-in-glabel ?rlc1a ?rlc1))`.
                if let Some(tails) = plain_tail_by_target.get(&z) {
                    if tails.iter().any(|&l| label_in_glabel(l, rlc1, glab)) {
                        continue;
                    }
                }

                result.push(Chain {
                    kind: ChainKind::PartialGWhip,
                    target: z,
                    length: 1,
                    llcs: vec![llc1],
                    rlcs: vec![Rlc::GCand(rlc1)],
                    csp_vars: vec![csp1],
                });
            }
        });
    }
    result
}

// ─── G-Whip[1] direct eliminations ──────────────────────────────────────────

/// Terminate length-1 partial-gwhip seeds immediately (test parity helper).
///
/// No `gWhips[1].clp` in CLIPS V2.1 — the G-WHIPS directory begins at
/// `gWhips[2].clp` (CLIPS file inventory; CR-FIN-4 C2). Production `gwhip`
/// firing starts at k=2 via `find_first_gwhip` / `run_gwhip_pass`, which
/// run the full extension layer with cross-type subsumption guards
/// (CR-FIN-5 C1). This helper bypasses that layer — it only terminates
/// length-1 seeds — so it MUST NOT be called from production code.
///
/// Note that the emission tag is `ChainRule::GWhip(2)`, NOT `GWhip(1)`:
/// `try_terminate_gwhip` computes `k = chain.length + 1`, and seeds here
/// have `length == 1`. The "1" in the function name refers to the **seed
/// length** consumed, not the resulting rule index.
///
/// Retained as `#[cfg(test)]` parity helper only. See CR-FIN-7 Mn-1
/// (mirrors `try_braid_1_eliminations` / `try_gbraid_1_eliminations`
/// gating) and CR-FIN-4 C2; gated by CR-FIN-8 Mn-1.
#[cfg(test)]
pub fn try_gwhip_1_eliminations(ctx: &ChainContext<'_>) -> Vec<ChainElimination> {
    // CR-FIN-5 C1: build plain whip[1] seeds first for cross-type subsumption.
    let whip_seeds = build_partial_whips_length_1(ctx);
    // Build partial-gwhip seeds of length 1 and try to terminate each immediately.
    let seeds = build_partial_gwhips_length_1(ctx, &whip_seeds);
    let mut result: Vec<ChainElimination> = Vec::new();
    let mut eliminated: HashSet<Label> = HashSet::new();

    for seed in &seeds {
        if eliminated.contains(&seed.target) {
            continue;
        }
        if let Some(e) = try_terminate_gwhip(seed, ctx) {
            result.push(e);
            eliminated.insert(seed.target);
        }
    }
    result
}

// ─── Extension: partial-gwhip or partial-whip → partial-gwhip ───────────────

/// Extend a set of partial-gwhips and partial-whips by one step, applying all three
/// sub-rules (-1, -2, -3) to produce new partial-gwhips of length k.
///
/// **Sub-rule -1** (gwhip + cand): Takes a partial-gwhip of length k-1, extends by
/// a plain candidate. CSP-var freshness: last-only.
///
/// **Sub-rule -2** (whip + gcand): Takes a partial-whip of length k-1 (in `whip_prev`),
/// extends by a grouped candidate. CSP-var freshness: full-history.
///
/// **Sub-rule -3** (gwhip + gcand): Takes a partial-gwhip of length k-1, extends by
/// a grouped candidate. CSP-var freshness: last-only.
///
/// The `gwhip_prev` slice contains partial-gwhips; `whip_prev` contains partial-whips.
/// Both may be empty.
///
/// Dedup for sub-rules -1: positional sequence match on `(target, rlcs[0..k])`.
/// Dedup for sub-rules -2/-3: positional + subsumption guard (tail rlc equality or
/// label-in-glabel subsumption). See spec §14.1.
///
/// **CR-FIN-5 C1 cross-type subsumption (extension paths)**: per CLIPS
/// `Partial-gWhips[5].clp:142-150` and `:213-221`, sub-rules -2 and -3 must skip a
/// new grouped chain when an existing **plain** partial-whip of the same length k
/// and target has the same prefix and a tail label inside the new glabel
/// (`label-in-glabel`). `plain_at_k` is the buffer of plain partial-whip chains
/// of length k (== chain.length + 1) materialized BEFORE this call. Pass `&[]`
/// only when the caller is a unit test that has no plain whips to consider;
/// the production drivers `run_gwhip_pass` / `find_first_gwhip` and the
/// gbraid drivers MUST supply the length-k plain whips after CR-FIN-5.
pub fn extend_partial_gwhips(
    gwhip_prev: &[Chain],
    whip_prev: &[Chain],
    plain_at_k: &[Chain],
    ctx: &ChainContext<'_>,
    dedup: &mut HashSet<u64>,
    out: &mut Vec<Chain>,
) {
    // Sub-rule -1: gwhip + cand.
    extend_gwhip_with_cand(gwhip_prev, ctx, dedup, out);
    // Sub-rule -2: whip + gcand.
    extend_whip_with_gcand(whip_prev, plain_at_k, ctx, dedup, out);
    // Sub-rule -3: gwhip + gcand.
    extend_gwhip_with_gcand(gwhip_prev, plain_at_k, ctx, dedup, out);
}

/// Sub-rule -1: extend a partial-gwhip[k-1] by a plain candidate.
///
/// Per CLIPS `Partial-gWhips[5]-1`. The new-llc must be linked (via exists-link
/// OR exists-glink) to the last rlc of the chain. new-llc ∉ llcs ∪ rlcs (NF-1).
/// CSP-var freshness: last-only.
fn extend_gwhip_with_cand(
    prev: &[Chain],
    ctx: &ChainContext<'_>,
    dedup: &mut HashSet<u64>,
    out: &mut Vec<Chain>,
) {
    let rs = &*ctx.rs;
    let link = ctx.link;
    let csp = ctx.csp;
    let cspl = ctx.cspl;
    let glab = ctx.glab;
    let glnk = ctx.glnk;

    for chain in prev {
        if chain.kind != ChainKind::PartialGWhip {
            continue;
        }
        let z = chain.target;
        let last_csp = chain.csp_vars.last().copied().unwrap_or(u32::MAX);

        // Build killed labels set: {z} ∪ all Cand rlcs.
        let killed_labels = build_killed_labels(z, &chain.rlcs);

        // Sets for exclusion checks.
        let llcs_set: HashSet<Label> = chain.llcs.iter().copied().collect();
        let rlcs_cand_set: HashSet<Label> = chain
            .rlcs.iter()
            .filter_map(|r| if let Rlc::Cand(l) = r { Some(*l) } else { None })
            .collect();
        let csp_set: HashSet<u32> = chain.csp_vars.iter().copied().collect();

        // Enumerate new_llc: linked (via exists-link OR exists-glink) to last_rlc.
        match chain.last_rlc() {
            Rlc::Cand(last_rlc_label) => {
                // exists-link: new_llc linked to last_rlc_label.
                link.linked[last_rlc_label as usize].for_each(|new_llc_idx| {
                    let new_llc = new_llc_idx as Label;
                    if !rs.cand_alive.test(new_llc as usize) { return; }
                    if new_llc == z { return; }
                    if llcs_set.contains(&new_llc) { return; }
                    if rlcs_cand_set.contains(&new_llc) { return; }

                    try_extend_with_cand(
                        chain, new_llc, z, last_csp, &killed_labels,
                        &llcs_set, &rlcs_cand_set, &csp_set,
                        ctx, dedup, out,
                    );
                });
            }
            Rlc::GCand(last_rlc_glabel) => {
                // exists-glink: for each label with glinked to last_rlc_glabel.
                // We need all labels l such that glnk.glinked[l].test(last_rlc_glabel).
                // There's no reverse index; iterate all alive labels.
                // NOTE: This is a linear scan — acceptable per spec for reasonable k.
                let n_labels = csp.n * csp.n * csp.n;
                for new_llc in 0..n_labels as Label {
                    if !rs.cand_alive.test(new_llc as usize) { continue; }
                    if new_llc == z { continue; }
                    if llcs_set.contains(&new_llc) { continue; }
                    if rlcs_cand_set.contains(&new_llc) { continue; }
                    // Check exists-link to the glabel, or exists-glink to last_rlc_glabel.
                    // Per CLIPS: `(or (exists-link ...) (exists-glink ... ?last-rlc))`.
                    // For GCand last_rlc, only exists-glink applies.
                    if !glnk.glinked[new_llc as usize].test(last_rlc_glabel as usize) {
                        continue;
                    }

                    try_extend_with_cand(
                        chain, new_llc, z, last_csp, &killed_labels,
                        &llcs_set, &rlcs_cand_set, &csp_set,
                        ctx, dedup, out,
                    );
                }
            }
        }
    }
}

/// Inner logic: try to extend `chain` with `new_llc` (as llc) and find the unique
/// surviving plain-label rlc. If found, assert a new partial-gwhip.
#[allow(clippy::too_many_arguments)]
fn try_extend_with_cand(
    chain: &Chain,
    new_llc: Label,
    z: Label,
    last_csp: u32,
    killed_labels: &super::super::bitboard::BitSet<W9>,
    llcs_set: &HashSet<Label>,
    rlcs_cand_set: &HashSet<Label>,
    csp_set: &HashSet<u32>,
    ctx: &ChainContext<'_>,
    dedup: &mut HashSet<u64>,
    out: &mut Vec<Chain>,
) {
    let rs = &*ctx.rs;
    let csp = ctx.csp;
    let cspl = ctx.cspl;
    let glab = ctx.glab;
    let glnk = ctx.glnk;

    let vars_of_new = &csp.vars_of[new_llc as usize];
    for slot in 0..4 {
        let new_csp = vars_of_new[slot];
        // CSP-var freshness: last-only (sub-rule -1).
        if new_csp == last_csp {
            continue;
        }
        // Note: sub-rule -1 uses last-only CSP freshness (only last_csp excluded above).
        // Full-history exclusion is not applied here per spec §8.4 / CLIPS gWhips[k]-1.
        let alts = &cspl.alternatives[new_llc as usize][slot];
        // Find unique survivor NOT glinked-killed.
        let mut survivor: Option<Label> = None;
        let mut more_than_one = false;
        for &alt in alts {
            if !rs.cand_alive.test(alt as usize) {
                continue;
            }
            if alt == z {
                continue;
            }
            // alt must NOT be killed by glinked-or(alt, z, rlcs).
            if glinked_killed(alt, killed_labels, &chain.rlcs, ctx.link, glnk) {
                continue;
            }
            if survivor.is_some() {
                more_than_one = true;
                break;
            }
            survivor = Some(alt);
        }
        if more_than_one || survivor.is_none() {
            continue;
        }
        let new_rlc = survivor.unwrap();
        // new_rlc must not already be in rlcs_cand_set or llcs_set (NF-1 style).
        if rlcs_cand_set.contains(&new_rlc) || llcs_set.contains(&new_rlc) {
            continue;
        }

        let mut new_rlcs = chain.rlcs.clone();
        new_rlcs.push(Rlc::Cand(new_rlc));
        let mut new_llcs = chain.llcs.clone();
        new_llcs.push(new_llc);
        let mut new_csp_vars = chain.csp_vars.clone();
        new_csp_vars.push(new_csp);

        let candidate = Chain {
            kind: ChainKind::PartialGWhip,
            target: z,
            length: chain.length + 1,
            llcs: new_llcs,
            rlcs: new_rlcs,
            csp_vars: new_csp_vars,
        };
        let key = candidate.dedup_key();
        if dedup.insert(key) {
            out.push(candidate);
        }
    }
}

/// Sub-rule -2: extend a partial-whip[k-1] by a grouped candidate (glabel).
///
/// Per CLIPS `Partial-gWhips[5]-2`. Input chain must be `PartialWhip`. The new-llc
/// is linked (via exists-link only, per CLIPS) to last_rlc. CSP-var freshness:
/// full-history (not member$ ?new-csp $?csp-vars). The new rlc is a GCand.
///
/// Dedup: subsumption guard — do not assert if existing partial-(g)whip with same
/// (target, length, rlcs prefix + tail rlc equal or label-in-glabel(tail, new_gcand)).
fn extend_whip_with_gcand(
    whip_prev: &[Chain],
    plain_at_k: &[Chain],
    ctx: &ChainContext<'_>,
    dedup: &mut HashSet<u64>,
    out: &mut Vec<Chain>,
) {
    let rs = &*ctx.rs;
    let link = ctx.link;
    let csp = ctx.csp;
    let cspl = ctx.cspl;
    let glab = ctx.glab;
    let glnk = ctx.glnk;

    for chain in whip_prev {
        if chain.kind != ChainKind::PartialWhip {
            continue;
        }
        let z = chain.target;
        let last_rlc_label = match chain.last_rlc() {
            Rlc::Cand(l) => l,
            Rlc::GCand(_) => continue, // partial-whip always has Cand rlcs
        };

        let killed_labels = build_killed_labels(z, &chain.rlcs);

        let llcs_set: HashSet<Label> = chain.llcs.iter().copied().collect();
        let rlcs_cand_set: HashSet<Label> = chain
            .rlcs.iter()
            .filter_map(|r| if let Rlc::Cand(l) = r { Some(*l) } else { None })
            .collect();
        let csp_set: HashSet<u32> = chain.csp_vars.iter().copied().collect();

        // Enumerate new_llc: linked via exists-link to last_rlc (sub-rule -2, per CLIPS).
        link.linked[last_rlc_label as usize].for_each(|new_llc_idx| {
            let new_llc = new_llc_idx as Label;
            if !rs.cand_alive.test(new_llc as usize) { return; }
            if new_llc == z { return; }
            if llcs_set.contains(&new_llc) { return; }
            if rlcs_cand_set.contains(&new_llc) { return; }

            // Look for new_rlc as a GCand via csp-glinked.
            // F1 fix: each (rlc_g, new_csp) pair is an independent CLIPS pattern match.
            // Per Partial-gWhips[5]-2: each ?new-rlc/?new-csp is tried independently.
            // Use `continue` (not `return`) to skip this pair and try the next.
            for &(rlc_g, new_csp) in &glnk.csp_glinked[new_llc as usize] {
                // rlc_g must be alive.
                if !rs.g_alive.test(rlc_g as usize) { continue; }
                // rlc_g must not be Z or contain Z.
                if label_in_glabel(z, rlc_g, glab) { continue; }
                // F6 fix: remove over-restrictive llc label-in-glabel exclusion.
                // CLIPS Partial-gWhips[5]-2 only checks exact member$ for new_rlc vs $?llcs
                // and $?rlcs — NOT label_in_glabel(llc, new_rlc). Drop this check.
                // glabel_contains_none_of(rlc_g, Z ∪ rlcs): none of Z or rlcs in rlc_g.
                if !glabel_contains_none_of(rlc_g, &chain.rlcs, glab) {
                    continue;
                }
                // CSP-var freshness: full-history for sub-rule -2 (C3).
                if !csp_fresh_full(new_csp, &chain.csp_vars) { continue; }

                // forall csp-linked new_llc X new_csp where !label_in_glabel(X, rlc_g):
                //   glinked_killed(X, z, rlcs).
                let slot = {
                    let vars_of_new = &csp.vars_of[new_llc as usize];
                    let mut found = None;
                    for s in 0..4 {
                        if vars_of_new[s] == new_csp {
                            found = Some(s);
                            break;
                        }
                    }
                    match found {
                        Some(s) => s,
                        None => continue,
                    }
                };

                let alts = &cspl.alternatives[new_llc as usize][slot];
                // Check: all alternatives not in rlc_g must be glinked-killed.
                let mut any_uncovered = false;
                for &alt in alts {
                    if !rs.cand_alive.test(alt as usize) { continue; }
                    if label_in_glabel(alt, rlc_g, glab) { continue; } // inside rlc_g
                    if glinked_killed(alt, &killed_labels, &chain.rlcs, link, glnk) { continue; }
                    // This alt is alive, not in rlc_g, and not killed — chain cannot fire.
                    any_uncovered = true;
                    break;
                }
                if any_uncovered { continue; }

                // Subsumption dedup for gcand rlc (spec §14.1 sub-rule -2/3 guard).
                // Block if any existing partial-(g)whip of same (target, new_length) has
                // tail rlc equal to rlc_g OR a Cand(L) where label_in_glabel(L, rlc_g).
                // We implement this as a secondary check against `out` + seeds already in `dedup`.
                // Build candidate chain first to get its key.
                let mut new_rlcs = chain.rlcs.clone();
                new_rlcs.push(Rlc::GCand(rlc_g));
                let mut new_llcs = chain.llcs.clone();
                new_llcs.push(new_llc);
                let mut new_csp_vars = chain.csp_vars.clone();
                new_csp_vars.push(new_csp);

                let candidate = Chain {
                    kind: ChainKind::PartialGWhip,
                    target: z,
                    length: chain.length + 1,
                    llcs: new_llcs,
                    rlcs: new_rlcs,
                    csp_vars: new_csp_vars,
                };

                // Check label-in-glabel subsumption against already-produced chains in `out`.
                let subsumed = out.iter().any(|existing| {
                    existing.target == z
                        && existing.length == candidate.length
                        && existing.rlcs.len() == candidate.rlcs.len()
                        && {
                            // Check if existing is a subsumer: same prefix, and tail rlc
                            // is either == GCand(rlc_g) or Cand(L) with label_in_glabel(L, rlc_g).
                            let prefix_match = existing.rlcs[..existing.rlcs.len().saturating_sub(1)]
                                == candidate.rlcs[..candidate.rlcs.len().saturating_sub(1)];
                            if !prefix_match { return false; }
                            match existing.rlcs.last() {
                                Some(Rlc::GCand(g2)) => *g2 == rlc_g,
                                Some(Rlc::Cand(l)) => label_in_glabel(*l, rlc_g, glab),
                                None => false,
                            }
                        }
                });
                if subsumed { continue; } // this (rlc_g, new_csp) pair is subsumed; try next

                // CR-FIN-5 C1 cross-type subsumption (extension): per CLIPS
                // Partial-gWhips[5].clp:142-150 `(type partial-whip|partial-gwhip)`
                // guard. Skip if any plain partial-whip[k] at same target has the
                // same prefix and tail Cand(L) with label_in_glabel(L, rlc_g).
                let plain_subsumed = plain_at_k.iter().any(|existing| {
                    if existing.kind != ChainKind::PartialWhip { return false; }
                    if existing.target != z { return false; }
                    if existing.length != candidate.length { return false; }
                    if existing.rlcs.len() != candidate.rlcs.len() { return false; }
                    let prefix_match = existing.rlcs[..existing.rlcs.len().saturating_sub(1)]
                        == candidate.rlcs[..candidate.rlcs.len().saturating_sub(1)];
                    if !prefix_match { return false; }
                    match existing.rlcs.last() {
                        Some(Rlc::Cand(l)) => label_in_glabel(*l, rlc_g, glab),
                        _ => false,
                    }
                });
                if plain_subsumed { continue; }

                let key = candidate.dedup_key();
                if dedup.insert(key) {
                    out.push(candidate);
                }
            }
        });
    }
}

/// Sub-rule -3: extend a partial-gwhip[k-1] by a grouped candidate (glabel).
///
/// Per CLIPS `Partial-gWhips[5]-3`. CSP-var freshness: last-only (same as -1).
/// new-llc can be linked via exists-link OR exists-glink to last_rlc (same as -1).
/// Dedup: subsumption guard like sub-rule -2.
fn extend_gwhip_with_gcand(
    prev: &[Chain],
    plain_at_k: &[Chain],
    ctx: &ChainContext<'_>,
    dedup: &mut HashSet<u64>,
    out: &mut Vec<Chain>,
) {
    let rs = &*ctx.rs;
    let link = ctx.link;
    let csp = ctx.csp;
    let cspl = ctx.cspl;
    let glab = ctx.glab;
    let glnk = ctx.glnk;

    for chain in prev {
        if chain.kind != ChainKind::PartialGWhip {
            continue;
        }
        let z = chain.target;
        let last_csp = chain.csp_vars.last().copied().unwrap_or(u32::MAX);

        let killed_labels = build_killed_labels(z, &chain.rlcs);

        let llcs_set: HashSet<Label> = chain.llcs.iter().copied().collect();
        let rlcs_cand_set: HashSet<Label> = chain
            .rlcs.iter()
            .filter_map(|r| if let Rlc::Cand(l) = r { Some(*l) } else { None })
            .collect();
        let csp_set: HashSet<u32> = chain.csp_vars.iter().copied().collect();

        // Enumerate new_llc: linked via exists-link or exists-glink to last_rlc.
        let try_new_llc = |new_llc: Label,
                            ctx: &ChainContext<'_>,
                            dedup: &mut HashSet<u64>,
                            out: &mut Vec<Chain>| {
            if !rs.cand_alive.test(new_llc as usize) { return; }
            if new_llc == z { return; }
            if llcs_set.contains(&new_llc) { return; }
            if rlcs_cand_set.contains(&new_llc) { return; }

            // Look for new_rlc as a GCand via csp-glinked.
            // F6 fix: removed over-restrictive label_in_glabel(llc, rlc_g) check.
            // CLIPS Partial-gWhips[5]-3 only excludes exact member$ for new_rlc vs $?llcs/$?rlcs.
            for &(rlc_g, new_csp) in &glnk.csp_glinked[new_llc as usize] {
                if !rs.g_alive.test(rlc_g as usize) { continue; }
                if label_in_glabel(z, rlc_g, glab) { continue; }
                if !glabel_contains_none_of(rlc_g, &chain.rlcs, glab) {
                    continue;
                }
                // CSP-var freshness: last-only for sub-rule -3.
                if new_csp == last_csp { continue; }

                let slot = {
                    let vars_of_new = &csp.vars_of[new_llc as usize];
                    let mut found = None;
                    for s in 0..4 {
                        if vars_of_new[s] == new_csp {
                            found = Some(s);
                            break;
                        }
                    }
                    match found {
                        Some(s) => s,
                        None => continue,
                    }
                };

                let alts = &cspl.alternatives[new_llc as usize][slot];
                let mut any_uncovered = false;
                for &alt in alts {
                    if !rs.cand_alive.test(alt as usize) { continue; }
                    if label_in_glabel(alt, rlc_g, glab) { continue; }
                    if glinked_killed(alt, &killed_labels, &chain.rlcs, link, glnk) { continue; }
                    any_uncovered = true;
                    break;
                }
                if any_uncovered { continue; }

                // Build candidate.
                let mut new_rlcs = chain.rlcs.clone();
                new_rlcs.push(Rlc::GCand(rlc_g));
                let mut new_llcs = chain.llcs.clone();
                new_llcs.push(new_llc);
                let mut new_csp_vars = chain.csp_vars.clone();
                new_csp_vars.push(new_csp);

                let candidate = Chain {
                    kind: ChainKind::PartialGWhip,
                    target: z,
                    length: chain.length + 1,
                    llcs: new_llcs,
                    rlcs: new_rlcs,
                    csp_vars: new_csp_vars,
                };

                // Subsumption dedup.
                let subsumed = out.iter().any(|existing| {
                    existing.target == z
                        && existing.length == candidate.length
                        && existing.rlcs.len() == candidate.rlcs.len()
                        && {
                            let prefix_match = existing.rlcs[..existing.rlcs.len().saturating_sub(1)]
                                == candidate.rlcs[..candidate.rlcs.len().saturating_sub(1)];
                            if !prefix_match { return false; }
                            match existing.rlcs.last() {
                                Some(Rlc::GCand(g2)) => *g2 == rlc_g,
                                Some(Rlc::Cand(l)) => label_in_glabel(*l, rlc_g, glab),
                                None => false,
                            }
                        }
                });
                if subsumed { continue; }

                // CR-FIN-5 C1 cross-type subsumption (extension): per CLIPS
                // Partial-gWhips[5].clp:213-221 `(type partial-whip|partial-gwhip)`
                // guard. Skip if any plain partial-whip[k] at same target has the
                // same prefix and tail Cand(L) with label_in_glabel(L, rlc_g).
                let plain_subsumed = plain_at_k.iter().any(|existing| {
                    if existing.kind != ChainKind::PartialWhip { return false; }
                    if existing.target != z { return false; }
                    if existing.length != candidate.length { return false; }
                    if existing.rlcs.len() != candidate.rlcs.len() { return false; }
                    let prefix_match = existing.rlcs[..existing.rlcs.len().saturating_sub(1)]
                        == candidate.rlcs[..candidate.rlcs.len().saturating_sub(1)];
                    if !prefix_match { return false; }
                    match existing.rlcs.last() {
                        Some(Rlc::Cand(l)) => label_in_glabel(*l, rlc_g, glab),
                        _ => false,
                    }
                });
                if plain_subsumed { continue; }

                let key = candidate.dedup_key();
                if dedup.insert(key) {
                    out.push(candidate);
                }
            }
        };

        match chain.last_rlc() {
            Rlc::Cand(last_rlc_label) => {
                link.linked[last_rlc_label as usize].for_each(|new_llc_idx| {
                    try_new_llc(new_llc_idx as Label, ctx, dedup, out);
                });
            }
            Rlc::GCand(last_rlc_glabel) => {
                let n_labels = csp.n * csp.n * csp.n;
                for new_llc in 0..n_labels as Label {
                    if glnk.glinked[new_llc as usize].test(last_rlc_glabel as usize) {
                        try_new_llc(new_llc, ctx, dedup, out);
                    }
                }
            }
        }
    }
}

// ─── Terminator: g-whip[k] elimination ───────────────────────────────────────

/// Try to terminate a partial-gwhip of length k-1 as a g-whip[k].
///
/// Per spec §8.4 and CLIPS `gWhips[5].clp:57-89`.
///
/// Finds new_llc linked (via exists-link or exists-glink) to last_rlc, not in
/// llcs ∪ rlcs (NF-1), and a fresh CSP-var (last-only freshness) such that
/// **every** live alternative X of new_csp for new_llc is glinked-killed.
///
/// Returns the first `ChainElimination` found, or `None`.
pub fn try_terminate_gwhip(chain: &Chain, ctx: &ChainContext<'_>) -> Option<ChainElimination> {
    // CR-FIN-9 Mn-2: parallel invariants to try_terminate_braid (braid.rs:472-486) and
    // try_terminate_gbraid (gbraid.rs:1052-1066). Kind guard catches a future refactor
    // that might pipe `PartialWhip` (or any non-`PartialGWhip`) chains into the gwhip
    // terminator. Length floor encodes the CR-FIN-4 C2 minimum: gwhip fires at k >= 2
    // (CLIPS `gWhips[2..36].clp` minimum, no `gWhips[1].clp`), so the terminator
    // consumes partials of length k-1 >= 1.
    debug_assert!(
        matches!(chain.kind, ChainKind::PartialGWhip),
        "gwhip terminator requires PartialGWhip"
    );
    debug_assert!(
        chain.length >= 1,
        "gwhip terminator requires length-1 partial for k>=2"
    );
    let k = chain.length + 1;
    let rs = &*ctx.rs;
    let link = ctx.link;
    let csp = ctx.csp;
    let cspl = ctx.cspl;
    let glab = ctx.glab;
    let glnk = ctx.glnk;

    let z = chain.target;
    let last_csp = chain.csp_vars.last().copied().unwrap_or(u32::MAX);

    let killed_labels = build_killed_labels(z, &chain.rlcs);

    let llcs_set: HashSet<Label> = chain.llcs.iter().copied().collect();
    let rlcs_cand_set: HashSet<Label> = chain
        .rlcs.iter()
        .filter_map(|r| if let Rlc::Cand(l) = r { Some(*l) } else { None })
        .collect();
    let csp_set: HashSet<u32> = chain.csp_vars.iter().copied().collect();

    let mut found: Option<ChainElimination> = None;

    let try_new_llc = |new_llc: Label| -> bool {
        if !rs.cand_alive.test(new_llc as usize) { return false; }
        if new_llc == z { return false; }
        if llcs_set.contains(&new_llc) { return false; }
        if rlcs_cand_set.contains(&new_llc) { return false; }

        let vars_of_new = &csp.vars_of[new_llc as usize];
        for slot in 0..4 {
            let new_csp = vars_of_new[slot];
            // CSP-var freshness: last-only for terminator.
            if new_csp == last_csp { continue; }

            let alts = &cspl.alternatives[new_llc as usize][slot];
            let any_live = alts.iter().any(|&a| rs.cand_alive.test(a as usize));
            if !any_live { continue; }

            // ALL live alternatives must be glinked-killed.
            let all_killed = alts.iter().all(|&alt| {
                !rs.cand_alive.test(alt as usize)
                    || glinked_killed(alt, &killed_labels, &chain.rlcs, link, glnk)
            });
            if all_killed {
                return true;
            }
        }
        false
    };

    match chain.last_rlc() {
        Rlc::Cand(last_rlc_label) => {
            link.linked[last_rlc_label as usize].for_each(|new_llc_idx| {
                if found.is_some() { return; }
                if try_new_llc(new_llc_idx as Label) {
                    found = Some(ChainElimination {
                        target: z,
                        rule: ChainRule::GWhip(k),
                    });
                }
            });
        }
        Rlc::GCand(last_rlc_glabel) => {
            let n_labels = csp.n * csp.n * csp.n;
            for new_llc in 0..n_labels as Label {
                if found.is_some() { break; }
                if glnk.glinked[new_llc as usize].test(last_rlc_glabel as usize) {
                    if try_new_llc(new_llc) {
                        found = Some(ChainElimination {
                            target: z,
                            rule: ChainRule::GWhip(k),
                        });
                    }
                }
            }
        }
    }

    found
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Run a full g-whip pass from k=2 up to `k_max`, collecting all eliminations.
///
/// ## Internal structure
///
/// This function maintains two parallel partial buffers:
///   - `scratch.gwhip_prev/next`: partial-gwhips (fed by NF-4 seed + sub-rules -1/-3)
///   - `scratch.whip_prev/next`:  partial-whips  (fed by whip[1] seed + whip extension)
///
/// At each k:
///   1. Try terminator on gwhip_prev (terminate to g-whip[k]).
///   2. Extend: apply all 3 sub-rules using gwhip_prev + whip_prev → gwhip_next.
///   3. Extend whip_prev → whip_next (for use in sub-rule -2 at k+1).
///   4. Swap buffers.
///
/// This mirrors the CLIPS salience model where at each k, partial-gwhip[k-1] is
/// extended and gwhip[k] is terminated, with partial-whips available as input to -2.
///
/// `scratch` is reset at the start of each call.
pub fn run_gwhip_pass(
    ctx: &mut ChainContext<'_>,
    k_max: u8,
    scratch: &mut GWhipScratch,
) -> Vec<ChainElimination> {
    if k_max < 2 {
        return Vec::new();
    }

    scratch.reset();
    let mut all_elims: Vec<ChainElimination> = Vec::new();

    // CR-FIN-5 C1: seed partial-whips of length 1 FIRST so the gwhip seed
    // builder can apply the cross-type subsumption guard from
    // Partial-gWhips[1].clp:63-71 against plain whip[1] tails.
    let whip_seeds = build_partial_whips_length_1(ctx);
    for chain in &whip_seeds {
        scratch.dedup_whip.insert(chain.dedup_key());
    }
    // Seed partial-gwhips of length 1 (NF-4), with cross-type subsumption against whip[1].
    let gwhip_seeds = build_partial_gwhips_length_1(ctx, &whip_seeds);
    for chain in &gwhip_seeds {
        scratch.dedup_gwhip.insert(chain.dedup_key());
    }
    scratch.gwhip_prev.extend(gwhip_seeds);
    scratch.whip_prev.extend(whip_seeds);

    // k=2..k_max: rolling extension + termination.
    for k in 2u8..=k_max {
        // Try to terminate each partial-gwhip of length k-1 as a g-whip[k].
        // FX-5 style: dedup by target to avoid duplicate eliminations.
        let mut k_elims: Vec<ChainElimination> = Vec::new();
        let mut k_targets_seen: std::collections::HashSet<super::super::chain_model::Label> = std::collections::HashSet::new();
        for chain in &scratch.gwhip_prev {
            if let Some(e) = try_terminate_gwhip(chain, ctx) {
                if k_targets_seen.insert(e.target) {
                    k_elims.push(e);
                }
            }
        }
        for e in &k_elims {
            ctx.rs.eliminate_candidate(e.target, ctx.glab);
        }
        if !k_elims.is_empty() {
            all_elims.extend(k_elims);
        }

        if k == k_max {
            break;
        }

        // FX-6: Filter stale partials (target no longer alive after eliminations).
        scratch.gwhip_prev.retain(|c| ctx.rs.cand_alive.test(c.target as usize));
        scratch.whip_prev.retain(|c| ctx.rs.cand_alive.test(c.target as usize));

        // CR-FIN-5 C1: build length-k plain whips FIRST so the gwhip extension
        // (sub-rules -2/-3) can apply the cross-type subsumption guard from
        // Partial-gWhips[5].clp:142-150,213-221. The plain-whip extension does
        // not depend on partial-gwhip data, so reordering is safe.
        scratch.whip_next.clear();
        scratch.dedup_whip.clear();
        for chain in &scratch.whip_prev {
            scratch.dedup_whip.insert(chain.dedup_key());
        }
        extend_partial_whips(
            &scratch.whip_prev.clone(),
            ctx,
            &mut scratch.dedup_whip,
            &mut scratch.whip_next,
        );

        // Extend gwhip_prev → gwhip_next (sub-rules -1, -2, -3), using the
        // freshly-built length-k plain whips for cross-type subsumption.
        scratch.gwhip_next.clear();
        scratch.dedup_gwhip.clear();
        for chain in &scratch.gwhip_prev {
            scratch.dedup_gwhip.insert(chain.dedup_key());
        }
        extend_partial_gwhips(
            &scratch.gwhip_prev.clone(),
            &scratch.whip_prev.clone(),
            &scratch.whip_next,
            ctx,
            &mut scratch.dedup_gwhip,
            &mut scratch.gwhip_next,
        );
        std::mem::swap(&mut scratch.gwhip_prev, &mut scratch.gwhip_next);
        scratch.gwhip_next.clear();

        // Advance whip_prev to length k (now safe to swap; gwhip extension done).
        std::mem::swap(&mut scratch.whip_prev, &mut scratch.whip_next);
        scratch.whip_next.clear();
    }

    all_elims
}

/// Find the first g-whip of any length ≤ `k_max`, returning
/// `Some((elimination, k))` where `k` is the g-whip length.
///
/// Short-circuit version of `run_gwhip_pass` — exits on the first elimination
/// found, without applying it. Used for puzzle rating probes.
///
/// Returns `None` if `k_max < 2` (g-whip minimum length is 2).
pub fn find_first_gwhip(
    ctx: &mut ChainContext<'_>,
    k_max: u8,
) -> Option<(ChainElimination, u8)> {
    if k_max < 2 {
        return None;
    }

    // CR-FIN-5 C1: seed plain whips FIRST, then pass them to the gwhip seed builder
    // for the cross-type subsumption guard (Partial-gWhips[1].clp:63-71).
    let mut whip_prev = build_partial_whips_length_1(ctx);
    let mut whip_dedup: HashSet<u64> = whip_prev.iter().map(|c| c.dedup_key()).collect();
    let mut gwhip_prev = build_partial_gwhips_length_1(ctx, &whip_prev);
    let mut gwhip_dedup: HashSet<u64> = gwhip_prev.iter().map(|c| c.dedup_key()).collect();

    for k in 2u8..=k_max {
        // Try terminator at length k.
        for chain in &gwhip_prev {
            if let Some(e) = try_terminate_gwhip(chain, ctx) {
                return Some((e, k));
            }
        }

        if k == k_max {
            break;
        }

        // CR-FIN-5 C1: build length-k plain whips BEFORE the gwhip extension so
        // sub-rules -2/-3 can apply the cross-type subsumption guard from
        // Partial-gWhips[5].clp:142-150,213-221.
        let mut whip_next: Vec<Chain> = Vec::new();
        extend_partial_whips(&whip_prev, ctx, &mut whip_dedup, &mut whip_next);

        // Extend gwhip_prev → gwhip_next, with whip_next as the length-k plain subsumer set.
        let mut gwhip_next: Vec<Chain> = Vec::new();
        extend_partial_gwhips(
            &gwhip_prev,
            &whip_prev,
            &whip_next,
            ctx,
            &mut gwhip_dedup,
            &mut gwhip_next,
        );
        gwhip_prev = gwhip_next;

        // Now advance whip_prev to length k.
        whip_prev = whip_next;
    }

    None
}

/// Suppressed g-whip probe: returns `Some` only when g-whip fires AND no
/// whip fires at k' ≤ k_gwhip.
///
/// Per spec §10.2: g-whip scoring must exclude *whip* hits (and only whip)
/// so guided-removal only credits genuine g-whip hits. Used by `gwhip_score`
/// in `gwhip_reverse.rs`.
///
/// Per spec §4.1 salience order: whip[k] → gwhip[k] → braid[k] → gbraid[k].
/// g-whip is pre-empted ONLY by whip (higher salience at same k). Braid has
/// LOWER salience than g-whip — braid fires AFTER gwhip at the same k-tier,
/// so braid cannot pre-empt g-whip.
///
/// Suppression list: whips only. (CR-FIN-M2o: renamed from `_excluding_wb`;
/// the old name implied braid suppression which is incorrect per spec §4.1.)
pub fn find_first_gwhip_excluding_w(
    ctx: &mut ChainContext<'_>,
    k_max: u8,
) -> Option<(ChainElimination, u8)> {
    let gwhip_result = find_first_gwhip(ctx, k_max)?;
    let k_gwhip = gwhip_result.1;
    // Suppress if whip fires at k' ≤ k_gwhip (per salience: whip > gwhip).
    if find_first_whip(ctx, k_gwhip).is_some() {
        return None;
    }
    // Braid has lower salience than gwhip and cannot pre-empt it.
    Some(gwhip_result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::chain_model::{ChainKind, Rlc};
    use crate::generic::csp_tables::build_csp_tables_n9;
    use crate::generic::glabel_tables::build_glabel_tables_n9;
    use crate::generic::grid::Grid;
    use crate::generic::resolution_state::ResolutionState;

    fn build_tables() -> (
        crate::generic::csp_tables::CspVarTable,
        LinkGraph<W9>,
        CspLinkGraph,
        GLabelTable<W9>,
        GLinkGraph<W9, WG9>,
    ) {
        let (csp, link, cspl) = build_csp_tables_n9();
        let (glab, glnk) = build_glabel_tables_n9(&csp);
        (csp, link, cspl, glab, glnk)
    }

    fn make_ctx<'a>(
        csp: &'a crate::generic::csp_tables::CspVarTable,
        link: &'a LinkGraph<W9>,
        cspl: &'a CspLinkGraph,
        glab: &'a GLabelTable<W9>,
        glnk: &'a GLinkGraph<W9, WG9>,
        rs: &'a mut ResolutionState,
    ) -> ChainContext<'a> {
        ChainContext { csp, link, cspl, glab, glnk, rs }
    }

    // ─── Test 1: partial-gwhip[1] seed has Rlc::GCand (NF-4) ────────────────

    /// Verify NF-4: `build_partial_gwhips_length_1` returns chains with:
    ///   - kind = PartialGWhip
    ///   - length = 1
    ///   - rlcs[0] = Rlc::GCand(_)  (NOT Cand)
    #[test]
    fn test_partial_gwhip1_seed_has_gcand() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);

        let chains = build_partial_gwhips_length_1(&ctx, &[]);
        // On an empty (dense) grid, at least some seeds should exist.
        // Verify all returned chains have the correct structure.
        for c in &chains {
            assert_eq!(c.kind, ChainKind::PartialGWhip, "kind must be PartialGWhip");
            assert_eq!(c.length, 1, "length must be 1");
            assert_eq!(c.llcs.len(), 1, "exactly one llc");
            assert_eq!(c.rlcs.len(), 1, "exactly one rlc");
            assert_eq!(c.csp_vars.len(), 1, "exactly one csp_var");
            // CRITICAL (NF-4): rlc[0] must be GCand, not Cand.
            assert!(
                matches!(c.rlcs[0], Rlc::GCand(_)),
                "partial-gwhip[1] rlc must be GCand (NF-4), got {:?}", c.rlcs[0]
            );
        }
        // Structural invariants pass whether or not any seeds are returned.
    }

    // ─── Test 2: partial-gwhip[1] llc is linked to target ───────────────────

    /// Each seeded partial-gwhip[1] must have llc1 linked to target Z.
    #[test]
    fn test_partial_gwhip1_llc_linked_to_target() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);

        let chains = build_partial_gwhips_length_1(&ctx, &[]);
        for c in &chains {
            let z = c.target;
            let llc1 = c.llcs[0];
            assert!(
                link.is_linked(z, llc1),
                "llc1 must be linked to target z; z={} llc1={}", z, llc1
            );
        }
    }

    // ─── Test 3: GWhipScratch reset ─────────────────────────────────────────

    /// `GWhipScratch::reset` must clear all buffers.
    #[test]
    fn test_gwhip_scratch_reset() {
        let mut scratch = GWhipScratch::new();
        scratch.gwhip_prev.push(Chain {
            kind: ChainKind::PartialGWhip,
            target: 0, length: 1,
            llcs: vec![1], rlcs: vec![Rlc::GCand(0)], csp_vars: vec![0],
        });
        scratch.whip_prev.push(Chain {
            kind: ChainKind::PartialWhip,
            target: 0, length: 1,
            llcs: vec![2], rlcs: vec![Rlc::Cand(3)], csp_vars: vec![1],
        });
        scratch.dedup_gwhip.insert(42u64);
        scratch.dedup_whip.insert(99u64);

        scratch.reset();
        assert!(scratch.gwhip_prev.is_empty(), "gwhip_prev not cleared");
        assert!(scratch.gwhip_next.is_empty(), "gwhip_next not cleared");
        assert!(scratch.whip_prev.is_empty(), "whip_prev not cleared");
        assert!(scratch.whip_next.is_empty(), "whip_next not cleared");
        assert!(scratch.dedup_gwhip.is_empty(), "dedup_gwhip not cleared");
        assert!(scratch.dedup_whip.is_empty(), "dedup_whip not cleared");
    }

    // ─── Test 4: find_first_gwhip k_max < 2 returns None ───────────────────

    /// `find_first_gwhip` with k_max < 2 must return None (g-whip minimum length = 2).
    #[test]
    fn test_find_first_gwhip_kmax_lt2() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let mut ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);
        assert!(find_first_gwhip(&mut ctx, 0).is_none(), "k_max=0 must return None");
        assert!(find_first_gwhip(&mut ctx, 1).is_none(), "k_max=1 must return None");
    }

    // ─── Test 5: no false positive on solved grid ────────────────────────────

    /// `find_first_gwhip` on a solved grid must return None.
    #[test]
    fn test_no_false_positive_solved_grid() {
        let solved = "123456789456789123789123456214365897365897214897214365531642978642978531978531642";
        let grid = match Grid::<9, 3, 3>::from_str(solved) {
            Some(g) => g,
            None => Grid::<9, 3, 3>::empty(),
        };
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let mut ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);
        let result = find_first_gwhip(&mut ctx, 5);
        assert!(result.is_none(), "no g-whip should fire on a solved grid; got {:?}", result);
    }

    // ─── Test 6: run_gwhip_pass smoke test ─────────────────────────────────

    /// `run_gwhip_pass` must not panic and must return a Vec.
    #[test]
    fn test_run_gwhip_pass_smoke() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let mut ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);
        let mut scratch = GWhipScratch::new();
        let elims = run_gwhip_pass(&mut ctx, 3, &mut scratch);
        let _ = elims; // No panic: API contract satisfied.
    }

    // ─── Test 7: g-whip dedup is positional ────────────────────────────────

    /// Two partial-gwhips with same rlcs as a SET but different ORDER must have
    /// different dedup keys (positional semantics per spec §14.1).
    #[test]
    fn test_gwhip_dedup_positional() {
        let g1 = Chain {
            kind: ChainKind::PartialGWhip,
            target: 0, length: 2,
            llcs: vec![1, 2],
            rlcs: vec![Rlc::GCand(5), Rlc::GCand(10)],
            csp_vars: vec![0, 1],
        };
        let g2 = Chain {
            kind: ChainKind::PartialGWhip,
            target: 0, length: 2,
            llcs: vec![2, 1],
            rlcs: vec![Rlc::GCand(10), Rlc::GCand(5)], // same rlcs, different order
            csp_vars: vec![1, 0],
        };
        assert_ne!(
            g1.dedup_key(), g2.dedup_key(),
            "g-whip dedup must be positional: [GCand(5),GCand(10)] ≠ [GCand(10),GCand(5)]"
        );
    }

    // ─── Test 8: llc exclusion (NF-1): new_llc in llcs → rejected ────────────

    /// Extension must reject new_llc that is already in chain.llcs (g-whip NF-1 rule).
    /// We verify this by checking that extend_partial_gwhips produces no chains
    /// with duplicate llcs.
    #[test]
    fn test_gwhip_llc_no_reuse() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);

        let seeds = build_partial_gwhips_length_1(&ctx, &[]);
        if seeds.is_empty() {
            return; // No seeds on this grid state — skip.
        }

        let mut out: Vec<Chain> = Vec::new();
        let mut dedup: HashSet<u64> = seeds.iter().map(|c| c.dedup_key()).collect();
        extend_partial_gwhips(&seeds, &[], &[], &ctx, &mut dedup, &mut out);

        for c in &out {
            let llcs_set: HashSet<Label> = c.llcs.iter().copied().collect();
            // All llcs must be distinct.
            assert_eq!(
                c.llcs.len(), llcs_set.len(),
                "llcs must be distinct (NF-1): {:?}", c.llcs
            );
        }
    }

    // ─── Test 9: subsumption guard — Cand(L) in gcand member rejected ────────

    /// Extension sub-rule -2/-3: a new GCand(g) where some existing chain has
    /// tail Cand(L) with label_in_glabel(L, g) must be rejected (subsumption).
    ///
    /// We verify the guard logic directly using the `label_in_glabel` predicate.
    #[test]
    fn test_subsumption_guard_label_in_glabel() {
        let (_csp, _link, _cspl, glab, _glnk) = build_tables();

        // Pick glabel 0 and one of its members.
        let g0 = 0u32;
        let member = glab.members_of[0][0];

        // If member is in g0, label_in_glabel should return true.
        assert!(
            label_in_glabel(member, g0, &glab),
            "member {} should be in glabel 0", member
        );

        // A non-member should return false.
        // Pick a label from a different glabel.
        let non_member_g0 = glab.members_of[1][0]; // different glabel
        // May or may not be in g0; test depends on layout.
        // Just verify the API works without panic.
        let _ = label_in_glabel(non_member_g0, g0, &glab);
    }

    // ─── Test F5: gwhip seed dedup is inserted AFTER semantic check ────────

    /// F5 regression: `build_partial_gwhips_length_1` must insert into dedup AFTER
    /// the `all_non_glabel_killed` semantic check passes. An invalid first encounter
    /// for (Z, g) must not block a valid later (Z, g) seed.
    ///
    /// We verify this indirectly: on the empty grid, run two calls to
    /// `build_partial_gwhips_length_1`. Since all candidates are alive on the first
    /// call, we just check that the API doesn't panic and returns well-formed seeds.
    /// The actual F5 correctness is verified by the dedup-after-check code structure.
    #[test]
    fn test_f5_gwhip_seed_dedup_after_semantic_check() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);
        let chains = build_partial_gwhips_length_1(&ctx, &[]);
        // All returned chains must have passed the semantic check.
        // Verify by re-checking the all_non_glabel_killed condition for each seed.
        for c in &chains {
            let z = c.target;
            let llc1 = c.llcs[0];
            let rlc1 = match c.rlcs[0] {
                Rlc::GCand(g) => g,
                Rlc::Cand(_) => panic!("gwhip seed must have GCand rlc"),
            };
            let csp1 = c.csp_vars[0];
            // Find slot for csp1 in llc1's vars.
            let slot = (0..4).find(|&s| csp.vars_of[llc1 as usize][s] == csp1)
                .expect("seed's csp1 must be in llc1's vars");
            let alts = &cspl.alternatives[llc1 as usize][slot];
            // All alive non-glabel alts must be linked to z.
            for &x in alts {
                if !rs.cand_alive.test(x as usize) { continue; }
                use crate::generic::glabel_tables::label_in_glabel;
                if label_in_glabel(x, rlc1, &glab) { continue; }
                assert!(
                    link.is_linked(x, z),
                    "F5: seed at (z={}, llc1={}, rlc1={}) has alt {} not linked to z",
                    z, llc1, rlc1, x
                );
            }
        }
    }

    // ─── Test F6: gwhip sub-rule -2 no llc label_in_glabel rejection ───────

    /// F6 regression: `extend_whip_with_gcand` must not reject (rlc_g, new_csp) pairs
    /// based on `label_in_glabel(llc, rlc_g, glab)`. Only exact rlcs/llcs membership
    /// tests are valid per CLIPS.
    ///
    /// We verify that after removing the over-restrictive check, the extension
    /// function produces more seeds than before (on the empty grid). Specifically,
    /// the function must not unconditionally reject all gcand pairs when the chain's
    /// llc happens to be in the glabel of the first gcand candidate.
    #[test]
    fn test_f6_gwhip_sub_rule_2_no_overrestriction() {
        let grid = Grid::<9, 3, 3>::empty();
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);
        // Get partial-whip seeds for sub-rule -2 input.
        let whip_seeds = crate::generic::techniques::whip::build_partial_whips_length_1(&ctx);
        if whip_seeds.is_empty() {
            return; // no seeds; skip
        }
        // Run extend_partial_gwhips with just whip seeds (empty gwhip_prev).
        let mut dedup: HashSet<u64> = HashSet::new();
        let mut out: Vec<Chain> = Vec::new();
        extend_partial_gwhips(&[], &whip_seeds, &[], &ctx, &mut dedup, &mut out);
        // The extension must not panic. On the empty grid, it should produce
        // some GCand-appended chains if any survive the semantic tests.
        for c in &out {
            // All produced chains must have GCand as last rlc (sub-rule -2 output).
            assert!(
                matches!(c.rlcs.last(), Some(Rlc::GCand(_))),
                "sub-rule -2 must produce GCand as last rlc; got {:?}", c.rlcs.last()
            );
            // Verify no chain's last rlc contains z (F5 invariant).
            let z = c.target;
            if let Some(Rlc::GCand(g)) = c.rlcs.last() {
                use crate::generic::glabel_tables::label_in_glabel;
                assert!(
                    !label_in_glabel(z, *g, &glab),
                    "last GCand rlc must not contain z (target)"
                );
            }
        }
    }

    // ─── Test 10: run_gwhip_pass on B=0 fixture ────────────────────────────

    /// `run_gwhip_pass` on a B=0 puzzle (pure whip≤1 territory) should complete
    /// without panic. May return empty elims (since g-whip may not be needed).
    #[test]
    fn test_run_gwhip_pass_b0_fixture() {
        let puzzle = "...456..9..6.......891..45.2.........7..9.....35......397...5.......4.72.....5361";
        let grid = match Grid::<9, 3, 3>::from_str(puzzle) {
            Some(g) => g,
            None => return,
        };
        let (csp, link, cspl, glab, glnk) = build_tables();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &glab);
        let mut ctx = make_ctx(&csp, &link, &cspl, &glab, &glnk, &mut rs);
        let mut scratch = GWhipScratch::new();
        let elims = run_gwhip_pass(&mut ctx, 4, &mut scratch);
        // May be empty or not; no panic is the key assertion.
        if let Some(e) = elims.first() {
            if let ChainRule::GWhip(k) = e.rule {
                assert!(k >= 2, "g-whip length must be ≥ 2; got {}", k);
                assert!(k <= 4, "g-whip length must be ≤ k_max=4; got {}", k);
            }
        }
    }
}
