//! Grouped-label (glabel) tables and glink graphs for arbitrary `(N, BR, BC)` grids.
//!
//! Per spec §8.1–§8.3, §13 (`glabel_tables.rs`), and the remediation log.
//!
//! ## Glabel construction (per spec §8.1, §13)
//! A glabel names a *block-row* or *block-column segment*: the N/BR (or N/BC) cells
//! of one block that lie in one particular row (or column), all holding the same digit.
//!
//! **Count per digit:**
//! - Horizontal (block-row) segments per digit: N × (N/BC) = N²/BC
//!   (N rows, each row intersects BR blocks once → N/BC = BC/BC = 1 … wait:
//!    each row has N/BC = BR block-cols, so N rows × N/BC segments-per-row = N²/BC)
//!   Hmm — from spec §8.1: `N²/BR + N²/BC`. Let me be explicit:
//!   Horizontal segments: per block there are BR rows, each forming a segment → BR segments/block.
//!   There are (N/BR)×(N/BC) = BC×BR = N blocks.
//!   Total horizontal segments per digit = N_blocks × BR = N × BR = (per spec) N²/BC.
//!   Wait: spec says `N²/BR` horizontal. Let me re-read.
//!   spec §8.1: "Horizontal segments: N rows × (N/BR) segments-per-row = N²/BR".
//!   So N rows, each row has N/BR = BC segments (one per block-column) → N × BC = N²/BC?
//!   No: each row intersects BC block-columns, and each intersection with a block is one segment.
//!   So N rows × BC block-cols-per-row = N × BC segments.
//!   But spec explicitly says N²/BR. For 9×9: N²/BR = 81/3 = 27 per digit × 9 digits = 243. ✓
//!   This means "N rows × (N/BR) = N × BC" and N/BR = BC for square blocks. OK.
//!   Summary: horizontal = N²/BR per digit, vertical = N²/BC per digit.
//!   Total glabels = N × (N²/BR + N²/BC) = N³ × (BR+BC)/(BR×BC).
//!   For 9×9 (N=9, BR=BC=3): 9³ × 6/9 = 729 × 2/3 = 486. ✓
//!
//! ## GLabel id layout
//! glabels are laid out as:
//!   [horizontal_digit_0, horizontal_digit_1, …, horizontal_digit_N-1,
//!    vertical_digit_0, vertical_digit_1, …, vertical_digit_N-1]
//! within each digit group, ordered by (block_id, segment_within_block).
//!
//! For 9×9: 54 per digit × 9 digits = 486, but we index them as 0..485.
//! Per-digit count: 27 horizontal + 27 vertical = 54.
//!
//! ## Bitset width for glabels
//! For 9×9: 486 glabels → W = (486+63)/64 = 549/64 = 8 words (rounds up).
//! For 16×16 (N=16, BR=BC=4): 16³×8/16 = 2048 glabels → W = 32.
//! We export `WG9 = 8` as the canonical width for N=9.
//!
//! ## CSP-Variable assignment per glabel (per spec §8.2, M4 fix)
//! - Horizontal glabels: csp_vars_of = [rn_var, bn_var]
//! - Vertical glabels: csp_vars_of = [cn_var, bn_var]
//! Backed by CLIPS `init-glinks.clp:84-88` (horiz) and `:109-113` (verti).
//!
//! ## GLink direction (per spec §8.3, M9 fix)
//! `exists-glink` is asserted ONLY as `(candidate → g-candidate)`.
//! The glink graph is keyed `(Label → GLabel)` only — no reverse index.

use super::bitboard::BitSet;
use super::chain_model::{CspVarId, GLabel, Label, Rlc};
use super::csp_tables::CspVarTable;

/// Width in u64 words for N=9 glabel bitset (486 glabels).
pub const WG9: usize = 8; // (486 + 63) / 64 = 549 / 64 = 8

/// Discriminates horizontal (block-row) vs vertical (block-column) glabels.
/// Per spec §8.1.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SegKind {
    /// Block-row (horizontal mini-line). Per spec §8.1.
    Row,
    /// Block-column (vertical mini-line). Per spec §8.1.
    Col,
}

/// Glabel membership and CSP-Variable tables.
/// Per spec §13 (`glabel_tables.rs`) and §8.1–§8.2.
///
/// `WL` = number of u64 words for the label-indexed `member_bits` bitsets.
/// For N=9: WL=12 (729 labels → 12 words). For N=16: WL=64.
/// This is the LABEL bitset width, not the glabel bitset width (WG).
#[derive(Clone, Debug)]
pub struct GLabelTable<const WL: usize> {
    /// `members_of[g]` = the labels that belong to glabel `g`.
    /// Size = N/BR (horizontal) or N/BC (vertical) per glabel.
    /// Per spec §8.1.
    pub members_of: Vec<Vec<Label>>,
    /// `digit_of[g]` = the digit (0-based digit_bit) for glabel `g`. Per spec §8.1.
    pub digit_of: Vec<u8>,
    /// `block_of[g]` = the block id (0..N) for glabel `g`. Per spec §8.1.
    pub block_of: Vec<u8>,
    /// `row_or_col_of[g]` = the row index (for horizontal) or column index (for vertical).
    /// Per spec §8.1.
    pub row_or_col_of: Vec<u8>,
    /// `segment_kind[g]` = Row or Col. Per spec §8.1.
    pub segment_kind: Vec<SegKind>,
    /// `csp_vars_of[g]` = [rn_var_or_cn_var, bn_var].
    /// Slot 0: rn (for horizontal) or cn (for vertical). Slot 1: bn (for all).
    /// Per spec §8.2, M4 fix, backed by init-glinks.clp:84-88,109-113.
    pub csp_vars_of: Vec<[CspVarId; 2]>,
    /// `member_bits[g]` = label-indexed bitset of members, for O(1) `label_in_glabel`.
    /// Indexed by LABEL (0..N³), so WL=12 for N=9 (not WG9=8 which is for glabels).
    pub member_bits: Vec<BitSet<WL>>,
    /// Grid size N this table was built for.
    pub n: usize,
}

/// Glink graph: for each label, the set of glabels it g-links to.
/// Per spec §8.3, M9 fix: direction is (Label → GLabel) only.
///
/// `glinked[label].test(g)` = true iff label `label` g-links to glabel `g`.
/// This encodes `exists-glink(?cont, label, g)` from CLIPS.
///
/// Width parameter WG is the number of u64 words needed for the glabel bitset.
#[derive(Clone, Debug)]
pub struct GLinkGraph<const WL: usize, const WG: usize> {
    /// Per-label bitset of glabels it g-links to. Per spec §8.3.
    pub glinked: Vec<BitSet<WG>>,
    /// For each (label, glabel) pair that is glinked, which CSP-Variable var_id
    /// caused the glink? Stored as a flat list per label for rare iteration needs.
    /// Most chain predicates only need the bitset; this is for diagnostic / dedup use.
    pub csp_glinked: Vec<Vec<(GLabel, CspVarId)>>,
}

/// Build glabel tables for N=9 (3×3 blocks).
///
/// TODO: const-generic on N for 16×16 support.
/// For N=16, instantiate with WL=64 (4096/64 labels) and WG=32 (2048/64 glabels).
pub fn build_glabel_tables_n9(
    _csp: &CspVarTable,
) -> (GLabelTable<{ super::csp_tables::W9 }>, GLinkGraph<{ super::csp_tables::W9 }, WG9>) {
    build_glabel_tables_generic::<9, 3, 3, { super::csp_tables::W9 }, WG9>(_csp)
}

/// Generic builder for arbitrary `(N, BR, BC)`.
/// Panics if `BR * BC != N`.
pub fn build_glabel_tables_generic<
    const N: usize,
    const BR: usize,
    const BC: usize,
    const WL: usize,
    const WG: usize,
>(
    _csp: &CspVarTable,
) -> (GLabelTable<WL>, GLinkGraph<WL, WG>) {
    assert_eq!(BR * BC, N, "BR*BC must equal N");

    let n_box_cols = N / BC; // = BR (number of block-columns in a row of blocks)
    // horiz: N rows × BC block-cols per row = N×BC = N²/BR (since BC = N/BR). For 9×9: 27. ✓
    let horiz_per_digit = N * BC;
    // vert:  N cols × BR block-rows per col = N×BR = N²/BC (since BR = N/BC). For 9×9: 27. ✓
    let vert_per_digit = N * BR;
    let total_per_digit = horiz_per_digit + vert_per_digit;
    let total_glabels = total_per_digit * N;

    // Sanity: spec formula N³*(BR+BC)/(BR*BC)
    let expected = N * N * N * (BR + BC) / (BR * BC);
    assert_eq!(
        total_glabels, expected,
        "glabel count mismatch: got {} expected {}", total_glabels, expected
    );

    let mut members_of: Vec<Vec<Label>> = Vec::with_capacity(total_glabels);
    let mut digit_of: Vec<u8> = Vec::with_capacity(total_glabels);
    let mut block_of: Vec<u8> = Vec::with_capacity(total_glabels);
    let mut row_or_col_of: Vec<u8> = Vec::with_capacity(total_glabels);
    let mut segment_kind: Vec<SegKind> = Vec::with_capacity(total_glabels);
    let mut csp_vars_of: Vec<[CspVarId; 2]> = Vec::with_capacity(total_glabels);
    // member_bits is indexed by LABEL (0..N³) — use WL (label bitset width), NOT WG.
    let mut member_bits: Vec<BitSet<WL>> = Vec::with_capacity(total_glabels);

    // Layout: for each digit, horizontal segments then vertical segments.
    for dbit in 0..N {
        // Horizontal (block-row) segments.
        // Each segment: one (block, row-within-block) pair.
        // block_id = block_row * n_box_cols + block_col
        // For each block, its rows are block_row*BR .. block_row*BR + BR-1.
        for block_row in 0..(N / BR) {
            // blocks in this block-row: block_ids = block_row*n_box_cols .. (block_row+1)*n_box_cols-1
            for block_col in 0..n_box_cols {
                let block_id = block_row * n_box_cols + block_col;
                // Segments within this block: one per row of the block.
                for local_row in 0..BR {
                    let global_row = block_row * BR + local_row;
                    // Members: the BC cells of this block that are in global_row.
                    let mut members: Vec<Label> = Vec::with_capacity(BC);
                    let mut bits: BitSet<WL> = BitSet::empty();
                    for local_col in 0..BC {
                        let global_col = block_col * BC + local_col;
                        let label = (global_row * N * N + global_col * N + dbit) as u32;
                        members.push(label);
                        bits.set(label as usize);
                    }
                    // CSP-Variable slot 0: rn (row×digit). Per spec §8.2.
                    let rn_var = (N * N + global_row * N + dbit) as u32;
                    // CSP-Variable slot 1: bn (block×digit).
                    let bn_var = (3 * N * N + block_id * N + dbit) as u32;

                    members_of.push(members);
                    digit_of.push(dbit as u8);
                    block_of.push(block_id as u8);
                    row_or_col_of.push(global_row as u8);
                    segment_kind.push(SegKind::Row);
                    csp_vars_of.push([rn_var, bn_var]);
                    member_bits.push(bits);
                }
            }
        }

        // Vertical (block-column) segments.
        for block_col in 0..n_box_cols {
            for block_row in 0..(N / BR) {
                let block_id = block_row * n_box_cols + block_col;
                for local_col in 0..BC {
                    let global_col = block_col * BC + local_col;
                    let mut members: Vec<Label> = Vec::with_capacity(BR);
                    let mut bits: BitSet<WL> = BitSet::empty();
                    for local_row in 0..BR {
                        let global_row = block_row * BR + local_row;
                        let label = (global_row * N * N + global_col * N + dbit) as u32;
                        members.push(label);
                        bits.set(label as usize);
                    }
                    // CSP-Variable slot 0: cn (col×digit). Per spec §8.2.
                    let cn_var = (2 * N * N + global_col * N + dbit) as u32;
                    // CSP-Variable slot 1: bn (block×digit).
                    let bn_var = (3 * N * N + block_id * N + dbit) as u32;

                    members_of.push(members);
                    digit_of.push(dbit as u8);
                    block_of.push(block_id as u8);
                    row_or_col_of.push(global_col as u8);
                    segment_kind.push(SegKind::Col);
                    csp_vars_of.push([cn_var, bn_var]);
                    member_bits.push(bits);
                }
            }
        }
    }

    let n_actual = members_of.len();
    assert_eq!(n_actual, total_glabels, "built {} glabels, expected {}", n_actual, total_glabels);

    let glab_table = GLabelTable {
        members_of,
        digit_of,
        block_of,
        row_or_col_of,
        segment_kind,
        csp_vars_of,
        member_bits,
        n: N,
    };

    // ── Build GLinkGraph ─────────────────────────────────────────────────────
    // `exists-glink(label, g)` = label is linked to glabel g, meaning label's
    // placement would kill every member of g (all members of g are peers of label
    // sharing some constraint). Per spec §8.3.
    //
    // Concrete Sudoku semantics: label l = (row_l, col_l, dbit) g-links to glabel g
    // iff g's digit = dbit AND every member of g is a peer of l (same row, col, or block).
    //
    // From CLIPS init-glinks.clp: csp-glinked and exists-glink are asserted iff
    // there is a shared CSP-Variable between label l and glabel g (concretely:
    // rn-variable for horiz glabels, cn-variable for verti glabels, or bn-variable
    // for either). We use a simpler characterization: l g-links to g iff
    // label_in_glabel gives FALSE for l (l is not a member) AND all members of g
    // share a constraint with l.
    //
    // For sudoku: label (row_l, col_l, dbit) g-links to glabel g with members
    // {(row_g, col_g_i, dbit) for i=0..BC-1} (horiz, same digit) iff
    // - same digit AND
    // - l and all members share a peer relation (same row, col, or block).
    // The shared CSP-Variable is: rn (same row+digit), or bn (same block+digit).
    //
    // Simplest correct check: l g-links to g iff l is NOT in g AND
    //   for all m in g.members: csp_tables.linked[l].test(m).
    // We do this efficiently using the member_bits bitset.

    let n_labels = N * N * N;
    let mut glinked_bits: Vec<BitSet<WG>> = vec![BitSet::empty(); n_labels];
    let mut csp_glinked_list: Vec<Vec<(GLabel, CspVarId)>> = vec![Vec::new(); n_labels];

    // Build label→glink using csp-variable shared membership.
    // A label l (row_l, col_l, dbit_l) g-links to glabel g iff:
    //   1. g.digit = dbit_l
    //   2. l ∉ g.members
    //   3. l shares a CSP-Variable with every member of g.
    //      = l is in the same rn/bn (for horiz) or cn/bn (for vert) as g's group CSP-var.
    //
    // More concretely: l shares rn_var_g iff row_l == g.row_or_col AND dbit_l == g.digit.
    //                  l shares bn_var_g iff block(row_l,col_l) == g.block AND dbit_l == g.digit.
    //                  l shares cn_var_g iff col_l == g.row_or_col AND dbit_l == g.digit.
    //
    // For horizontal glabel g: rn_var = (N*N + row_g * N + dbit), bn_var = (3*N*N + block_g * N + dbit).
    // l shares rn_var iff row_l == row_g AND dbit_l == dbit.
    // l shares bn_var iff block(l) == block_g AND dbit_l == dbit.
    // Either is sufficient for g-link (we match CLIPS which asserts for BOTH and has one exist-glink per csp-var).

    let n_box_cols_val = N / BC;

    for label in 0..n_labels {
        let dbit_l = label % N;
        let cell_l = label / N;
        let row_l = cell_l / N;
        let col_l = cell_l % N;
        let block_l = (row_l / BR) * n_box_cols_val + (col_l / BC);

        for (g, g_kind) in glab_table.segment_kind.iter().enumerate() {
            if glab_table.digit_of[g] as usize != dbit_l {
                continue;
            }
            if glab_table.member_bits[g].test(label) {
                // label is a member of g, no g-link.
                continue;
            }
            let row_or_col_g = glab_table.row_or_col_of[g] as usize;
            let block_g = glab_table.block_of[g] as usize;

            let linked = match g_kind {
                SegKind::Row => {
                    // horiz glabel: l shares rn (same row+digit) or bn (same block+digit)
                    (row_l == row_or_col_g) || (block_l == block_g)
                }
                SegKind::Col => {
                    // vert glabel: l shares cn (same col+digit) or bn (same block+digit)
                    (col_l == row_or_col_g) || (block_l == block_g)
                }
            };
            if linked {
                // Compute which CSP-Variable caused this glink.
                let csp_var = match g_kind {
                    SegKind::Row => {
                        if row_l == row_or_col_g {
                            glab_table.csp_vars_of[g][0] // rn_var
                        } else {
                            glab_table.csp_vars_of[g][1] // bn_var
                        }
                    }
                    SegKind::Col => {
                        if col_l == row_or_col_g {
                            glab_table.csp_vars_of[g][0] // cn_var
                        } else {
                            glab_table.csp_vars_of[g][1] // bn_var
                        }
                    }
                };
                glinked_bits[label].set(g);
                csp_glinked_list[label].push((g as GLabel, csp_var));
            }
        }
    }

    let glink_graph = GLinkGraph {
        glinked: glinked_bits,
        csp_glinked: csp_glinked_list,
    };

    (glab_table, glink_graph)
}

// ─── Predicate functions ─────────────────────────────────────────────────────

/// True iff label `l` is a member of glabel `g`. O(1) bitset test.
/// Per spec §8.1, CLIPS `label-in-glabel`.
#[inline(always)]
pub fn label_in_glabel<const WL: usize>(l: Label, g: GLabel, t: &GLabelTable<WL>) -> bool {
    t.member_bits[g as usize].test(l as usize)
}

/// Returns true iff glabel `g` contains NONE of the entries in `rlcs`.
///
/// For each `Rlc` in `rlcs`:
///   - `Rlc::Cand(l)`: check `l ∉ members_of[g]`.
///   - `Rlc::GCand(g2)`: check that `g` and `g2` share NO member label.
///
/// Per spec §13, M6 fix, backed by `SudoRules/glabels.clp:292-313`.
pub fn glabel_contains_none_of<const WL: usize>(g: GLabel, rlcs: &[Rlc], t: &GLabelTable<WL>) -> bool {
    let gb = &t.member_bits[g as usize];
    for rlc in rlcs {
        match rlc {
            Rlc::Cand(l) => {
                if gb.test(*l as usize) {
                    return false;
                }
            }
            Rlc::GCand(g2) => {
                // Check if g and g2 share any member label.
                let g2b = &t.member_bits[*g2 as usize];
                let mut shared = false;
                for w in 0..WL {
                    if gb.words[w] & g2b.words[w] != 0 {
                        shared = true;
                        break;
                    }
                }
                if shared {
                    return false;
                }
            }
        }
    }
    true
}

/// Returns true iff glabel `g` contains AT LEAST ONE of the entries in `rlcs`.
///
/// For each `Rlc` in `rlcs`:
///   - `Rlc::Cand(l)`: check `l ∈ members_of[g]`.
///   - `Rlc::GCand(g2)`: check that `g` and `g2` share at least one member.
///
/// Per spec §13, M6+M7 fix, backed by `generic-background.clp:179-187`.
pub fn glabel_contains_some_of<const WL: usize>(g: GLabel, rlcs: &[Rlc], t: &GLabelTable<WL>) -> bool {
    let gb = &t.member_bits[g as usize];
    for rlc in rlcs {
        match rlc {
            Rlc::Cand(l) => {
                if gb.test(*l as usize) {
                    return true;
                }
            }
            Rlc::GCand(g2) => {
                let g2b = &t.member_bits[*g2 as usize];
                for w in 0..WL {
                    if gb.words[w] & g2b.words[w] != 0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::csp_tables::build_csp_tables_n9;

    fn make_tables() -> (GLabelTable<{ super::super::csp_tables::W9 }>, GLinkGraph<{ super::super::csp_tables::W9 }, WG9>) {
        let (csp, _, _) = build_csp_tables_n9();
        build_glabel_tables_n9(&csp)
    }

    /// For N=9 (BR=BC=3), total glabel count must be 486. Per spec §8.1.
    #[test]
    fn n9_glabel_count_is_486() {
        let (t, _) = make_tables();
        assert_eq!(t.members_of.len(), 486,
            "expected 486 glabels for N=9, got {}", t.members_of.len());
    }

    /// Every glabel for N=9 must have exactly BC=3 members (horizontal) or BR=3 (vertical).
    /// Both happen to be 3 for 9×9.
    #[test]
    fn n9_glabel_member_count() {
        let (t, _) = make_tables();
        for (g, members) in t.members_of.iter().enumerate() {
            assert_eq!(members.len(), 3,
                "glabel {} has {} members, expected 3", g, members.len());
        }
    }

    /// `label_in_glabel` must be true for members and false for non-members.
    #[test]
    fn label_in_glabel_basic() {
        let (t, _) = make_tables();
        // Glabel 0 should be the first horizontal segment (digit=0, block-row=0, block-col=0, local-row=0)
        // Members: cells (row=0, col=0..2) with dbit=0 → labels 0, 9, 18
        let g = 0u32;
        let members = &t.members_of[0];
        for &m in members {
            assert!(label_in_glabel(m, g, &t), "member {} should be in glabel 0", m);
        }
        // A label with a different digit should NOT be in this glabel.
        let non_member = members[0] + 1; // same cell, different digit
        assert!(!label_in_glabel(non_member, g, &t), "non-member should not be in glabel");
    }

    /// `glabel_contains_none_of` with a member label → false.
    #[test]
    fn glabel_contains_none_of_basic() {
        let (t, _) = make_tables();
        let g = 0u32;
        let member = t.members_of[0][0];
        let non_member = member + 1; // different digit

        // Contains a member → should NOT be "none of" → returns false.
        assert!(!glabel_contains_none_of(g, &[Rlc::Cand(member)], &t));

        // Contains only non-members → returns true.
        assert!(glabel_contains_none_of(g, &[Rlc::Cand(non_member)], &t));
    }

    /// `glabel_contains_some_of` with a member label → true.
    #[test]
    fn glabel_contains_some_of_basic() {
        let (t, _) = make_tables();
        let g = 0u32;
        let member = t.members_of[0][0];
        let non_member = member + 1;

        assert!(glabel_contains_some_of(g, &[Rlc::Cand(member)], &t));
        assert!(!glabel_contains_some_of(g, &[Rlc::Cand(non_member)], &t));
    }

    /// `glabel_contains_some_of` with a GCand that overlaps → true.
    #[test]
    fn glabel_contains_some_of_gcand_overlap() {
        let (t, _) = make_tables();
        // Glabels 0 and 1 are both horizontal, digit=0. They should NOT overlap
        // if they are in different rows within the same block (different global rows).
        // But let's just verify the API works correctly with GCand.
        let g0 = 0u32;
        let g1 = 1u32; // next glabel (same digit, different row-within-block or different segment)

        let members0 = &t.members_of[0];
        let members1 = &t.members_of[1];

        // If they share any member, some_of should be true; if not, false.
        let shared = members0.iter().any(|m| members1.contains(m));
        let result = glabel_contains_some_of(g0, &[Rlc::GCand(g1)], &t);
        assert_eq!(result, shared);
    }

    /// Horizontal and vertical glabels for same block/digit should have different segment kinds.
    #[test]
    fn segment_kind_correctness() {
        let (t, _) = make_tables();
        // First 27*9 = 243 glabels (for N=9) should be Row (horizontal).
        assert_eq!(t.segment_kind[0], SegKind::Row);
        assert_eq!(t.segment_kind[243], SegKind::Col);
    }

    /// CSP-Variable slots for horizontal glabels must be (rn_var, bn_var). Per spec §8.2.
    #[test]
    fn horiz_glabel_csp_vars() {
        let (t, _) = make_tables();
        // Glabel 0: horizontal, digit=0, row=0, block=0
        // rn_var = N*N + row*N + dbit = 81 + 0*9 + 0 = 81
        // bn_var = 3*N*N + block*N + dbit = 243 + 0 + 0 = 243
        let [rn, bn] = t.csp_vars_of[0];
        assert_eq!(rn, 81, "rn_var for glabel 0 should be 81");
        assert_eq!(bn, 243, "bn_var for glabel 0 should be 243");
    }
}
