//! Resolution State (RS) for CSP-Rules chain search.
//!
//! Per spec §3, §13 (`resolution_state.rs`), and remediation log C2.
//!
//! ## Design choice: N=9 hard-coded, NOT const-generic on N
//!
//! This file uses hard-coded bitset widths for N=9:
//!   - `W_LAB  = 12`  (N³ = 729 labels → 12 u64 words)
//!   - `W_GLAB = 8`   (486 glabels → 8 u64 words)
//!   - `W_VAL  = 2`   (N² = 81 values → 2 u64 words)
//!
//! This avoids const-generic complexity (`generic_const_exprs` feature) at the cost
//! of restricting the RS to N=9 grids.
//!
//! TODO: const-generic on N for 16×16 support.
//!   To extend: replace the concrete `BitSet<W_LAB>`, `BitSet<W_GLAB>`, `BitSet<W_VAL>`
//!   fields with `BitSet<{(N*N*N+63)/64}>`, `BitSet<{N*N*N*(BR+BC)/(BR*BC+63)/64}>`, and
//!   `BitSet<{(N*N+63)/64}>` respectively. This requires the `generic_const_exprs` unstable
//!   feature (or a macro) for the const expressions. For 16×16 the widths would be
//!   W_LAB=64, W_GLAB=32 (2048/64), W_VAL=4.
//!
//! ## Dynamic g_alive (per spec §2.2, §3, C2 fix)
//! `g_alive` is NOT static — it is truth-maintained in CLIPS via logical dependencies.
//! A glabel is alive iff ≥ 2 of its member labels are still in `cand_alive`.
//! The `g_support` vec tracks alive-member counts per glabel.
//! **ALL candidate eliminations MUST go through `eliminate_candidate`** — direct
//! `cand_alive.clear(L)` is forbidden (would silently miss g-alive cascade).

use super::bitboard::BitSet;
use super::chain_model::{GLabel, Label, Rlc};
use super::csp_tables::{LinkGraph, W9};
use super::glabel_tables::{GLabelTable, GLinkGraph, WG9};
// W_LAB_9 = W9 alias for use in method signatures (avoids confusion with W_LAB const)
use super::csp_tables::W9 as W_LAB_9;
use super::grid::Grid;

/// Width for the label bitset (N=9: 729 labels, 12 words).
pub const W_LAB: usize = 12;
/// Width for the glabel bitset (N=9: 486 glabels, 8 words).
pub const W_GLAB: usize = 8;
/// Width for the value (placed-digit) bitset (N=9: 81 cells, 2 words).
pub const W_VAL: usize = 2;

/// The Resolution State used by all chain search passes.
///
/// Per spec §3: a snapshot of `{alive candidates, alive g-candidates, placed values}`.
/// The link and CSP-link graphs are immutable and held externally; this struct tracks
/// only the mutable alive-set state.
///
/// ## Invariant
/// `g_alive.test(g)` ⟺ `g_support[g] >= 2`.
/// Maintained by `eliminate_candidate`. NEVER call `cand_alive.clear(l)` directly.
#[derive(Clone, Debug)]
pub struct ResolutionState {
    /// Alive candidate bitset: bit L set iff label L is still undecided. Per spec §3.
    pub cand_alive: BitSet<W_LAB>,
    /// Alive glabel bitset: bit g set iff ≥ 2 member labels of g are alive. Per spec §3.
    pub g_alive: BitSet<W_GLAB>,
    /// Per-glabel alive-member support count.
    /// Invariant: `g_alive.test(g) ⟺ g_support[g] >= 2`.
    /// Per spec C2 fix, backed by init-glinks.clp logical-rule semantics.
    pub g_support: Vec<u8>,
    /// Placed values bitset: bit `(row*N + col)` set iff that cell is solved.
    pub values: BitSet<W_VAL>,
}

impl ResolutionState {
    /// Construct from a 9×9 grid and glabel table.
    ///
    /// - `cand_alive`: set for every label L=(row,col,dbit) where
    ///   `grid.candidates[cell] & (1 << dbit) != 0` and cell is not yet solved.
    /// - `values`: set for every cell that is solved.
    /// - `g_support`: count of alive members per glabel.
    /// - `g_alive`: set for every glabel with support ≥ 2.
    ///
    /// Per spec §13.
    pub fn from_grid_9x9(g: &Grid<9, 3, 3>, t: &GLabelTable<W_LAB_9>) -> Self {
        let mut cand_alive: BitSet<W_LAB> = BitSet::empty();
        let mut values: BitSet<W_VAL> = BitSet::empty();
        let n_glabels = t.members_of.len();
        let mut g_support = vec![0u8; n_glabels];

        for row in 0..9usize {
            for col in 0..9usize {
                let cell = row * 9 + col;
                if g.solved[cell] != 0 {
                    values.set(cell);
                    // Committed digit: mark its label as alive too (c-value).
                    // We do NOT set cand_alive for c-values; they are not "cand" labels.
                } else {
                    let mask = g.candidates[cell];
                    for dbit in 0..9usize {
                        if mask & (1u32 << dbit) != 0 {
                            let label = row * 81 + col * 9 + dbit;
                            cand_alive.set(label);
                        }
                    }
                }
            }
        }

        // Count alive members per glabel.
        for gid in 0..n_glabels {
            let count = t.members_of[gid].iter()
                .filter(|&&lab| cand_alive.test(lab as usize))
                .count() as u8;
            g_support[gid] = count;
        }

        let mut g_alive: BitSet<W_GLAB> = BitSet::empty();
        for gid in 0..n_glabels {
            if g_support[gid] >= 2 {
                g_alive.set(gid);
            }
        }

        ResolutionState {
            cand_alive,
            g_alive,
            g_support,
            values,
        }
    }

    /// Eliminate candidate label `l`.
    ///
    /// **This is the ONLY supported elimination entry-point.**
    /// Direct `cand_alive.clear(L)` is forbidden (will silently miss g-alive cascade).
    ///
    /// Actions:
    /// 1. Clear `l` from `cand_alive`.
    /// 2. For every glabel `g` containing `l`, decrement `g_support[g]`.
    ///    If `g_support[g]` drops below 2, clear `g` from `g_alive`.
    ///
    /// Idempotent: eliminating an already-dead label is a no-op.
    ///
    /// Per spec §13 C2 fix, CLIPS semantics: `init-glinks.clp:68-80, 93-105`
    /// logical rules auto-retract g-candidates when member count < 2.
    pub fn eliminate_candidate(&mut self, l: Label, t: &GLabelTable<W_LAB_9>) {
        if !self.cand_alive.test(l as usize) {
            return; // Already eliminated — idempotent.
        }
        self.cand_alive.clear(l as usize);

        // Update g_support for every glabel containing l.
        // We iterate over all glabels whose member_bits include l.
        // For N=9 with 486 glabels this is a small constant per elimination.
        for (gid, mb) in t.member_bits.iter().enumerate() {
            if mb.test(l as usize) {
                if self.g_support[gid] > 0 {
                    self.g_support[gid] -= 1;
                }
                if self.g_support[gid] < 2 {
                    self.g_alive.clear(gid);
                }
            }
        }
    }

    /// True iff glabel `g` is currently alive (≥ 2 member labels alive).
    /// Per spec §13.
    #[inline(always)]
    pub fn g_cand_alive(&self, g: GLabel) -> bool {
        self.g_alive.test(g as usize)
    }

    /// True iff both labels `a` and `b` are alive AND linked (per the link graph).
    /// Per spec §13.
    #[inline(always)]
    pub fn linked(&self, a: Label, b: Label, lg: &LinkGraph<W9>) -> bool {
        self.cand_alive.test(a as usize)
            && self.cand_alive.test(b as usize)
            && lg.is_linked(a, b)
    }

    /// True iff label `a` is alive AND is linked to at least one label in bitset `bs`
    /// (restricted to alive labels). Per spec §13.
    ///
    /// Note: does NOT filter `bs` by aliveness — callers are responsible for
    /// passing a bitset of already-alive or "committed" labels.
    #[inline]
    pub fn linked_or(&self, a: Label, bs: &BitSet<W_LAB>, lg: &LinkGraph<W9>) -> bool {
        if !self.cand_alive.test(a as usize) {
            return false;
        }
        lg.is_linked_or_bitset(a, bs)
    }

    /// True iff label `a` is alive AND is linked (via exists-link OR exists-glink) to
    /// at least one element of `rlcs`.
    ///
    /// - `Rlc::Cand(b)`: check `lg.is_linked(a, b)`.
    /// - `Rlc::GCand(g)`: check `gg.glinked[a].test(g)`.
    ///
    /// Per spec §13 and §8.3.
    pub fn glinked_or(
        &self,
        a: Label,
        rlcs: &[Rlc],
        gg: &GLinkGraph<W9, WG9>,
        lg: &LinkGraph<W9>,
    ) -> bool {
        if !self.cand_alive.test(a as usize) {
            return false;
        }
        for rlc in rlcs {
            match rlc {
                Rlc::Cand(b) => {
                    if lg.is_linked(a, *b) {
                        return true;
                    }
                }
                Rlc::GCand(g) => {
                    if gg.glinked[a as usize].test(*g as usize) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::csp_tables::build_csp_tables_n9;
    use crate::generic::glabel_tables::build_glabel_tables_n9;
    use crate::generic::grid::Grid;

    fn make_rs_from_full_grid() -> (ResolutionState, GLabelTable<W9>) {
        let (csp, _, _) = build_csp_tables_n9();
        let (glab, _) = build_glabel_tables_n9(&csp);
        let grid = Grid::<9, 3, 3>::empty();
        let rs = ResolutionState::from_grid_9x9(&grid, &glab);
        (rs, glab)
    }

    /// `eliminate_candidate` removes the label from `cand_alive`.
    #[test]
    fn eliminate_removes_from_cand_alive() {
        let (mut rs, t) = make_rs_from_full_grid();
        let label = 0u32; // first label
        assert!(rs.cand_alive.test(0), "label 0 should be alive initially");
        rs.eliminate_candidate(label, &t);
        assert!(!rs.cand_alive.test(0), "label 0 should be dead after elimination");
    }

    /// Eliminating an already-dead label is idempotent (no panic, no change).
    #[test]
    fn eliminate_is_idempotent() {
        let (mut rs, t) = make_rs_from_full_grid();
        let label = 5u32;
        rs.eliminate_candidate(label, &t);
        let g_support_before: Vec<u8> = rs.g_support.clone();
        rs.eliminate_candidate(label, &t); // second call — should be no-op
        assert_eq!(rs.g_support, g_support_before, "idempotent on g_support");
    }

    /// When the next-to-last member of a glabel is eliminated, that glabel becomes
    /// dead in `g_alive`. Per spec C2 fix.
    #[test]
    fn eliminate_kills_glabel_when_support_drops_below_2() {
        let (csp, _, _) = build_csp_tables_n9();
        let (t, _) = build_glabel_tables_n9(&csp);

        // Find glabel 0 and its members.
        let members = t.members_of[0].clone();
        assert_eq!(members.len(), 3, "glabel 0 should have 3 members");

        let grid = Grid::<9, 3, 3>::empty();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &t);

        assert!(rs.g_alive.test(0), "glabel 0 should be alive initially");
        assert_eq!(rs.g_support[0], 3, "glabel 0 should have 3 alive members");

        // Eliminate 2 of 3 members.
        rs.eliminate_candidate(members[0], &t);
        assert_eq!(rs.g_support[0], 2);
        assert!(rs.g_alive.test(0), "still alive with 2 members");

        rs.eliminate_candidate(members[1], &t);
        assert_eq!(rs.g_support[0], 1);
        assert!(!rs.g_alive.test(0), "glabel should be dead with only 1 member");
    }

    /// `g_cand_alive` reflects `g_alive` correctly.
    #[test]
    fn g_cand_alive_reflects_g_alive() {
        let (csp, _, _) = build_csp_tables_n9();
        let (t, _) = build_glabel_tables_n9(&csp);
        let grid = Grid::<9, 3, 3>::empty();
        let mut rs = ResolutionState::from_grid_9x9(&grid, &t);

        assert!(rs.g_cand_alive(0), "glabel 0 should be alive");

        let members = t.members_of[0].clone();
        rs.eliminate_candidate(members[0], &t);
        rs.eliminate_candidate(members[1], &t);
        assert!(!rs.g_cand_alive(0), "glabel 0 should be dead after 2 member eliminations");
    }
}
