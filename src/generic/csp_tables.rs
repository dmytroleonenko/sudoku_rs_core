//! CSP-Variable tables and link graphs for arbitrary `(N, BR, BC)` grids.
//!
//! Per spec §2.1 (CSP-Variable families), §2.3 (links), §6.2 (build phase),
//! §13 (`csp_tables.rs`).
//!
//! ## Label encoding (consistent with chain_model.rs and aic.rs)
//! `label = row * N * N + col * N + digit_bit`  (digit_bit = digit − 1, 0-based).
//! For N=9: 729 labels total (0..728).
//!
//! ## CSP-Variable encoding
//! Variables are laid out as:
//!   - rc: vars 0          .. N*N         (81 for 9×9): var = row*N + col
//!   - rn: vars N*N        .. 2*N*N       : var = N*N + row*N + digit_bit
//!   - cn: vars 2*N*N      .. 3*N*N       : var = 2*N*N + col*N + digit_bit
//!   - bn: vars 3*N*N      .. 4*N*N       : var = 3*N*N + block_id*N + digit_bit
//! Total: 4*N*N variables (324 for N=9).
//!
//! ## Bitset width
//! For N labels total = N³, we need `W = (N³ + 63) / 64` u64 words.
//! N=9  → 729 labels → W=12
//! N=16 → 4096 labels → W=64
//! We use `const W: usize` as a type parameter on `BitSet<W>`.
//!
//! ## Caching
//! Tables are built once per `(N, BR, BC)` instantiation and cached via a
//! `Mutex<HashMap>` (same pattern as `peer_tables.rs`).

use std::sync::{Arc, OnceLock};

use super::bitboard::BitSet;
use super::chain_model::{CspVarId, CspVarKind, Label};

// ─── Bitset width constants ───────────────────────────────────────────────────

/// Width in u64 words for N=9 (729 labels).
pub const W9: usize = 12; // (9*9*9 + 63)/64 = (729+63)/64 = 792/64 = 12
/// Width in u64 words for N=16 (4096 labels).
pub const W16: usize = 64; // (16*16*16 + 63)/64 = 4096/64 = 64

/// Named constant for the 9×9 label count.
pub const N_LABELS_9X9: usize = 729;
/// Named constant for the 9×9 CSP-Variable count.
pub const N_CSP_VARS_9X9: usize = 324;

// ─── Public types ────────────────────────────────────────────────────────────

/// Maps CSP-Variable ids to their labels and vice versa.
/// Per spec §2.1 and §13.
#[derive(Clone, Debug)]
pub struct CspVarTable {
    /// `labels_of[var_id]` = the labels that belong to this CSP-Variable.
    /// Each label appears in exactly 4 vars (one rc, rn, cn, bn).
    /// Per spec §2.1.
    pub labels_of: Vec<Vec<Label>>,
    /// `vars_of[label]` = the 4 CSP-Variable ids for this label.
    /// Slot order: [rc_var, rn_var, cn_var, bn_var]. Per spec §2.1.
    pub vars_of: Vec<[CspVarId; 4]>,
    /// `kind_of[var_id]` = which family this variable belongs to. Per spec §2.1.
    pub kind_of: Vec<CspVarKind>,
    /// The grid size N this table was built for.
    pub n: usize,
}

/// Symmetric "exists-link" graph: any two labels sharing a CSP-Variable are linked.
/// Per spec §2.3.
///
/// `linked[a].test(b)` is true iff labels `a` and `b` are peers (linked by any constraint).
/// For N=9 this is a 729×729-bit matrix stored as 729 × BitSet<12>.
#[derive(Clone, Debug)]
pub struct LinkGraph<const W: usize> {
    /// Per-label link bitsets. `linked[a]` has bit `b` set iff label `a` is linked to label `b`.
    pub linked: Vec<BitSet<W>>,
}

/// Per-label, per-CSP-family alternative lists.
/// Per spec §2.3 and §6.1.
///
/// `alternatives[label][slot]` = the other labels in the same CSP-Variable as `label`
/// for slot 0=rc, 1=rn, 2=cn, 3=bn. Each list has length N−1 (the other N−1 labels
/// in that variable).
#[derive(Clone, Debug)]
pub struct CspLinkGraph {
    /// Indexed as `alternatives[label][slot]`.
    pub alternatives: Vec<[Vec<Label>; 4]>,
    /// The grid size N this was built for.
    pub n: usize,
}

// ─── Build function ──────────────────────────────────────────────────────────

/// Build all three tables for an `(N, BR, BC)` grid instantiation.
///
/// Per spec §6.2 (build phase):
/// 1. Enumerate 4*N*N CSP-Variables with their labels.
/// 2. Build the `LinkGraph` (symmetric existence-link: share any CSP-Variable).
/// 3. Build `CspLinkGraph` (per-family alternative lists).
///
/// Returns `(CspVarTable, LinkGraph<W>, CspLinkGraph)`.
/// Panics if `BR * BC != N`.
///
/// For N=9 (W=12): one-time cost ~1ms.
/// For N=16 (W=64): one-time cost ~30ms.
pub fn build_csp_tables_n9() -> (CspVarTable, LinkGraph<W9>, CspLinkGraph) {
    build_csp_tables_generic::<9, 3, 3, W9>()
}

/// Generic builder — exposed for testing and future N=16 support.
pub fn build_csp_tables_generic<
    const N: usize,
    const BR: usize,
    const BC: usize,
    const W: usize,
>() -> (CspVarTable, LinkGraph<W>, CspLinkGraph) {
    assert_eq!(BR * BC, N, "BR*BC must equal N");
    assert_eq!((N * N * N + 63) / 64, W, "W must equal (N³+63)/64");

    let n_labels = N * N * N;
    let n_vars = 4 * N * N;

    // ── Build CspVarTable ────────────────────────────────────────────────────

    let mut labels_of: Vec<Vec<Label>> = vec![Vec::new(); n_vars];
    let mut vars_of: Vec<[CspVarId; 4]> = vec![[0u32; 4]; n_labels];
    let mut kind_of: Vec<CspVarKind> = Vec::with_capacity(n_vars);

    // rc: var_id = row*N + col; domain = digits (label = row*N*N + col*N + dbit)
    for _v in 0..N * N {
        kind_of.push(CspVarKind::Rc);
    }
    // rn: var_id = N*N + row*N + dbit; domain = columns
    for _v in 0..N * N {
        kind_of.push(CspVarKind::Rn);
    }
    // cn: var_id = 2*N*N + col*N + dbit; domain = rows
    for _v in 0..N * N {
        kind_of.push(CspVarKind::Cn);
    }
    // bn: var_id = 3*N*N + block_id*N + dbit; domain = cells-in-block
    for _v in 0..N * N {
        kind_of.push(CspVarKind::Bn);
    }

    let n_box_cols = N / BC; // = BR

    for row in 0..N {
        for col in 0..N {
            let block_id = (row / BR) * n_box_cols + (col / BC);
            for dbit in 0..N {
                let label = (row * N * N + col * N + dbit) as u32;

                let rc_var = (row * N + col) as u32;
                let rn_var = (N * N + row * N + dbit) as u32;
                let cn_var = (2 * N * N + col * N + dbit) as u32;
                let bn_var = (3 * N * N + block_id * N + dbit) as u32;

                vars_of[label as usize] = [rc_var, rn_var, cn_var, bn_var];

                labels_of[rc_var as usize].push(label);
                labels_of[rn_var as usize].push(label);
                labels_of[cn_var as usize].push(label);
                labels_of[bn_var as usize].push(label);
            }
        }
    }

    let csp_var_table = CspVarTable {
        labels_of,
        vars_of,
        kind_of,
        n: N,
    };

    // ── Build LinkGraph and CspLinkGraph ─────────────────────────────────────

    let mut linked: Vec<BitSet<W>> = vec![BitSet::empty(); n_labels];

    // alternatives[label][slot] — start as empty vecs
    let mut alternatives: Vec<[Vec<Label>; 4]> =
        (0..n_labels).map(|_| [Vec::new(), Vec::new(), Vec::new(), Vec::new()]).collect();

    for var_id in 0..n_vars {
        let labs = &csp_var_table.labels_of[var_id];
        // For every pair in this variable, mark them as linked and record alternatives.
        for (i, &la) in labs.iter().enumerate() {
            for (j, &lb) in labs.iter().enumerate() {
                if i == j {
                    continue;
                }
                // Link graph: symmetric.
                linked[la as usize].set(lb as usize);

                // CSP-link alternatives: which slot does this var correspond to for la?
                let slot = slot_for_label(la, var_id as u32, &csp_var_table);
                alternatives[la as usize][slot].push(lb);
            }
        }
    }

    // Deduplicate alternatives (a label may appear via multiple CSP families in
    // the same "slot" only if var ids are distinct — they are by construction — but
    // let's be safe).
    for lab_alts in &mut alternatives {
        for slot_vec in lab_alts.iter_mut() {
            slot_vec.sort_unstable();
            slot_vec.dedup();
        }
    }

    let link_graph = LinkGraph { linked };
    let csp_link_graph = CspLinkGraph { alternatives, n: N };

    (csp_var_table, link_graph, csp_link_graph)
}

/// Return the slot index (0=rc, 1=rn, 2=cn, 3=bn) for `var_id` w.r.t. `label`.
/// Per spec §2.1 variable layout.
#[inline]
fn slot_for_label(label: Label, var_id: CspVarId, t: &CspVarTable) -> usize {
    let slots = &t.vars_of[label as usize];
    for s in 0..4 {
        if slots[s] == var_id {
            return s;
        }
    }
    // Should never happen if the table was built correctly.
    panic!("var_id {} not found in vars_of[{}]", var_id, label);
}

// ─── Cached accessor for N=9 ─────────────────────────────────────────────────

type CachedN9 = (CspVarTable, LinkGraph<W9>, CspLinkGraph);

static CSP_TABLES_9X9: OnceLock<Arc<CachedN9>> = OnceLock::new();

/// Return a cached Arc to the N=9 CSP tables.
/// Built on first call (~1ms); subsequent calls return the cached Arc.
pub fn get_csp_tables_9x9() -> Arc<CachedN9> {
    CSP_TABLES_9X9
        .get_or_init(|| Arc::new(build_csp_tables_n9()))
        .clone()
}

// ─── Helper methods on LinkGraph ─────────────────────────────────────────────

impl<const W: usize> LinkGraph<W> {
    /// True iff labels `a` and `b` are linked (peer). Per spec §2.3.
    #[inline(always)]
    pub fn is_linked(&self, a: Label, b: Label) -> bool {
        self.linked[a as usize].test(b as usize)
    }

    /// True iff label `a` is linked to at least one label in the bitset `bs`.
    /// Per spec §2.3; this is the hot-path `linked-or` operation.
    #[inline(always)]
    pub fn is_linked_or_bitset(&self, a: Label, bs: &BitSet<W>) -> bool {
        let row = &self.linked[a as usize];
        // bitwise AND of each word; true iff any bit overlaps.
        for w in 0..W {
            if row.words[w] & bs.words[w] != 0 {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For N=9 the CSP-Variable count must be exactly 324 (4 × 81). Per spec §2.1.
    #[test]
    fn n9_csp_var_count_is_324() {
        let (t, _, _) = build_csp_tables_n9();
        assert_eq!(t.labels_of.len(), N_CSP_VARS_9X9,
            "expected 324 CSP-Variables for N=9");
    }

    /// Each CSP-Variable must contain exactly 9 labels. Per spec §2.1.
    #[test]
    fn n9_each_csp_var_has_9_labels() {
        let (t, _, _) = build_csp_tables_n9();
        for (v, labs) in t.labels_of.iter().enumerate() {
            assert_eq!(labs.len(), 9,
                "CSP-Variable {} has {} labels, expected 9", v, labs.len());
        }
    }

    /// Every label belongs to exactly 4 CSP-Variables. Per spec §2.1.
    #[test]
    fn n9_label_belongs_to_4_vars() {
        let (t, _, _) = build_csp_tables_n9();
        assert_eq!(t.vars_of.len(), N_LABELS_9X9);
        for lab in 0..N_LABELS_9X9 {
            let vars = &t.vars_of[lab];
            // All 4 slots should be distinct.
            let mut unique_vars: Vec<u32> = vars.to_vec();
            unique_vars.sort_unstable();
            unique_vars.dedup();
            assert_eq!(unique_vars.len(), 4,
                "label {} belongs to {} vars, expected 4", lab, unique_vars.len());
        }
    }

    /// Two labels in the same row and same digit must be linked. Per spec §2.3.
    #[test]
    fn n9_row_peers_are_linked() {
        let (_, lg, _) = build_csp_tables_n9();
        // label(row=0, col=0, dbit=0) and label(row=0, col=5, dbit=0) share rn variable
        let la = (0 * 9 * 9 + 0 * 9 + 0) as usize; // row0, col0, dbit0
        let lb = (0 * 9 * 9 + 5 * 9 + 0) as usize; // row0, col5, dbit0
        assert!(lg.linked[la].test(lb),
            "row-peer labels must be linked");
        assert!(lg.linked[lb].test(la),
            "link graph must be symmetric");
    }

    /// Two labels in the same cell (different digits) must be linked via rc. Per spec §2.3.
    #[test]
    fn n9_same_cell_different_digits_linked() {
        let (_, lg, _) = build_csp_tables_n9();
        // Same cell (row=3, col=4), different digits (dbit=0 and dbit=5)
        let la = 3 * 81 + 4 * 9 + 0;
        let lb = 3 * 81 + 4 * 9 + 5;
        assert!(lg.linked[la].test(lb), "same-cell labels must be linked");
    }

    /// CspLinkGraph alternatives: for a given label, each of its 4 CSP-Variable
    /// slots should have exactly N−1=8 alternatives on N=9. Per spec §6.2.
    #[test]
    fn n9_csp_alternatives_count() {
        let (_, _, cg) = build_csp_tables_n9();
        // Check label 0 (row=0, col=0, dbit=0)
        let lab = 0usize;
        for slot in 0..4 {
            assert_eq!(cg.alternatives[lab][slot].len(), 8,
                "slot {} should have 8 alternatives, got {}", slot, cg.alternatives[lab][slot].len());
        }
    }
}
