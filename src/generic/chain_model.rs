//! Shared types for CSP-Rules chain techniques: whip[k], braid[k], g-whip[k], g-braid[k].
//!
//! Per spec §13 (chain_model.rs section) and §2 (formal preliminaries).
//!
//! ## Label encoding convention (per spec §2.2)
//! `Label = row * N * N + col * N + digit_bit` where `digit_bit` = digit−1 (0-based).
//! For N=9: 729 labels total (cells 0..81, digits 0..9 → bits 0..8).
//! This matches the `node_idx` convention used in `techniques/aic.rs`.
//!
//! ## Dedup key design (per spec §14.1)
//! - Whip / partial-whip: **positional** sequence equality on `(target, rlcs[0..k])`.
//!   Key = hash of the ordered Vec<Rlc>.
//! - Braid / partial-braid: **multiset** (set) equality on `(target, rlcs as sorted set)`.
//!   Key = hash of the sorted-then-hashed BTreeSet-style representation.
//! - G-variants: same positional/set split as their non-grouped counterparts.
//!
//! We use `std::collections::hash_map::DefaultHasher` (SipHash 1-3 by default) to
//! combine fields into a u64 dedup key. No new crates introduced.
//! **Limitation:** collision probability is negligible in practice (729 labels, k ≤ 36)
//! but the key is advisory (dedup table also stores the full chain for exact comparison).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A label is a packed `(row, col, digit_bit)` integer. For N=9: range 0..728.
/// `label = row * N * N + col * N + digit_bit`  (digit_bit = digit − 1, 0-based).
/// Per spec §2.2.
pub type Label = u32;

/// A grouped label (glabel) id — opaque integer naming a set of labels sharing
/// a block-row or block-column segment for a fixed digit. Per spec §2.2.
pub type GLabel = u32;

/// CSP-Variable identifier. For N=9: 0..323 (324 total = 4 × 81).
/// Encoding: rc vars first (0..81), then rn (81..162), then cn (162..243), then bn (243..324).
/// Per spec §2.1.
pub type CspVarId = u32;

/// Which of the four CSP-Variable families a variable belongs to.
/// Per spec §2.1.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum CspVarKind {
    /// Cell variable: (row, col) → digit. Per spec §2.1.
    Rc,
    /// Row×digit variable: (row, digit) → column. Per spec §2.1.
    Rn,
    /// Column×digit variable: (col, digit) → row. Per spec §2.1.
    Cn,
    /// Block×digit variable: (block, digit) → cell-in-block. Per spec §2.1.
    Bn,
}

/// A right-linking candidate: either a regular label or a grouped label (glabel).
/// Per spec §6.1 and §8.4.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub enum Rlc {
    /// Regular (non-grouped) candidate. Per spec §6.1.
    Cand(Label),
    /// Grouped candidate. Per spec §8.4.
    GCand(GLabel),
}

/// Discriminates among the eight chain kinds used by the search.
/// Per spec §13.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum ChainKind {
    /// Partial whip under construction (not yet eliminated). Per spec §6.3.1.
    PartialWhip,
    /// Completed whip (elimination fired). Per spec §5.
    Whip,
    /// Partial braid under construction. Per spec §7.
    PartialBraid,
    /// Completed braid (elimination fired). Per spec §7.
    Braid,
    /// Partial g-whip under construction. Per spec §8.4.
    PartialGWhip,
    /// Completed g-whip. Per spec §8.4.
    GWhip,
    /// Partial g-braid under construction. Per spec §8.5.
    PartialGBraid,
    /// Completed g-braid. Per spec §8.5.
    GBraid,
}

/// One chain instance (partial or complete).
///
/// `llcs`, `rlcs`, and `csp_vars` are parallel vectors of length `length`.
/// Per spec §13 (chain_model.rs), §6.1.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Which variety of chain this is. Per spec §13.
    pub kind: ChainKind,
    /// The target candidate Z to be eliminated. Per spec §5.1.
    pub target: Label,
    /// Number of completed steps. Per spec §5.1.
    pub length: u8,
    /// Left-linking candidates per step (the "L_i" in the spec). Per spec §5.1.
    pub llcs: Vec<Label>,
    /// Right-linking candidates per step (the "R_i" in the spec). Per spec §5.1.
    pub rlcs: Vec<Rlc>,
    /// CSP-Variable id for each step. Per spec §5.1.
    pub csp_vars: Vec<CspVarId>,
}

impl Chain {
    /// Return the last right-linking candidate. Panics if `length == 0`.
    /// Per spec §13.
    #[inline]
    pub fn last_rlc(&self) -> Rlc {
        *self.rlcs.last().expect("chain has at least one step")
    }

    /// Compute a u64 dedup key for this chain per spec §14.1.
    ///
    /// - Whip / partial-whip: hash of `(target, ordered rlcs sequence)`.
    /// - Braid / partial-braid: hash of `(target, sorted rlcs set)`.
    /// - G-whip / partial-g-whip: positional (ordered) on Rlc sequence.
    /// - G-braid / partial-g-braid: set (sorted) on Rlc sequence.
    ///
    /// The key is advisory for fast reject; callers should do exact comparison
    /// when the key matches.
    pub fn dedup_key(&self) -> u64 {
        self.dedup_key_inner(true)
    }

    /// **Cross-type guard key.** Used ONLY for CLIPS-style union-type guards
    /// (e.g. `partial-whip|partial-braid` in `Braids[5].clp:128-139`,
    /// `partial-gwhip|partial-gbraid` in `gBraids[5].clp:138-145`).
    ///
    /// **DO NOT use for self-extension dedup** — for whip extension (which is
    /// positional per spec §14.1), use [`Chain::dedup_key`] instead. Using this
    /// cross-type key for positional self-extension would collapse legitimately
    /// distinct ordered chains.
    ///
    /// This key hashes `(is_grouped, target, rlcs)` but NOT `is_braid`, so that
    /// a partial-whip and a partial-braid with the same `(target, rlcs)` get
    /// identical keys. Two chains with different `is_grouped` → DIFFERENT keys
    /// (gwhip ≠ whip).
    ///
    /// FX-4 fix: the F8 discriminator split (keep `is_braid` in `dedup_key()` for
    /// same-technique extension) is CORRECT for self-extension guards, but BREAKS
    /// the union-type guard. This method is the correct key for union-type guards.
    pub fn dedup_key_cross_type(&self) -> u64 {
        self.dedup_key_inner(false)
    }

    fn dedup_key_inner(&self, include_is_braid: bool) -> u64 {
        let mut h = DefaultHasher::new();
        self.target.hash(&mut h);
        // F8 fix: include kind discriminator to prevent cross-kind collisions.
        // Two chains with identical (target, rlcs) but different kinds now get different keys.
        // Encode as a pair (is_grouped: bool, is_braid: bool) for forward compatibility.
        // FX-4 note: dedup_key_cross_type() passes include_is_braid=false to drop the braid
        // discriminator for CLIPS union-type guards (partial-whip|partial-braid).
        let is_grouped = matches!(
            self.kind,
            ChainKind::PartialGWhip | ChainKind::GWhip | ChainKind::PartialGBraid | ChainKind::GBraid
        );
        let is_braid = matches!(
            self.kind,
            ChainKind::PartialBraid | ChainKind::Braid | ChainKind::PartialGBraid | ChainKind::GBraid
        );
        is_grouped.hash(&mut h);
        if include_is_braid {
            is_braid.hash(&mut h);
        }
        if !include_is_braid {
            // Cross-type dedup: CLIPS `same-sets-of-rlcs` is always set equality.
            // Both whip and braid get sorted-set hash for the union-type guard.
            let mut sorted = self.rlcs.clone();
            sorted.sort_unstable();
            sorted.hash(&mut h);
        } else {
            match self.kind {
                ChainKind::PartialWhip
                | ChainKind::Whip
                | ChainKind::PartialGWhip
                | ChainKind::GWhip => {
                    // Positional: hash ordered rlcs sequence. Per spec §14.1.
                    self.rlcs.hash(&mut h);
                }
                ChainKind::PartialBraid
                | ChainKind::Braid
                | ChainKind::PartialGBraid
                | ChainKind::GBraid => {
                    // Set semantics: sort rlcs then hash. Per spec §14.1.
                    let mut sorted = self.rlcs.clone();
                    sorted.sort_unstable();
                    sorted.hash(&mut h);
                }
            }
        }
        h.finish()
    }
}

/// Result of a chain elimination rule firing.
/// Per spec §13.
#[derive(Copy, Clone, Debug)]
pub struct ChainElimination {
    /// The candidate that was eliminated (the Z target). Per spec §5.1.
    pub target: Label,
    /// Which rule produced the elimination, including the chain length k.
    pub rule: ChainRule,
}

/// The specific rule that produced an elimination.
/// Per spec §9.1.
#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub enum ChainRule {
    /// Whip of length k. Per spec §5.
    Whip(u8),
    /// Braid of length k. Per spec §7.
    Braid(u8),
    /// G-whip of length k. Per spec §8.4.
    GWhip(u8),
    /// G-braid of length k. Per spec §8.5.
    GBraid(u8),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_whip(target: Label, rlcs: &[u32]) -> Chain {
        let len = rlcs.len() as u8;
        Chain {
            kind: ChainKind::PartialWhip,
            target,
            length: len,
            llcs: vec![0u32; len as usize],
            rlcs: rlcs.iter().map(|&r| Rlc::Cand(r)).collect(),
            csp_vars: vec![0u32; len as usize],
        }
    }

    fn make_braid(target: Label, rlcs: &[u32]) -> Chain {
        let len = rlcs.len() as u8;
        Chain {
            kind: ChainKind::PartialBraid,
            target,
            length: len,
            llcs: vec![0u32; len as usize],
            rlcs: rlcs.iter().map(|&r| Rlc::Cand(r)).collect(),
            csp_vars: vec![0u32; len as usize],
        }
    }

    /// Per spec §14.1: two whips differing only in rlcs ORDER must NOT be deduped
    /// (whip dedup is positional / sequence-equality).
    #[test]
    fn whip_dedup_is_positional_not_set() {
        let w1 = make_whip(0, &[10, 20, 30]);
        let w2 = make_whip(0, &[30, 20, 10]); // same rlcs set, different order
        // Must have DIFFERENT dedup keys (positional semantics).
        assert_ne!(
            w1.dedup_key(),
            w2.dedup_key(),
            "whip dedup must be positional: [10,20,30] ≠ [30,20,10]"
        );
    }

    /// Per spec §14.1: two braids differing only in rlcs ORDER MUST be deduped
    /// (braid dedup is set / multiset-equality).
    #[test]
    fn braid_dedup_is_set_not_positional() {
        let b1 = make_braid(0, &[10, 20, 30]);
        let b2 = make_braid(0, &[30, 10, 20]); // same rlcs set, different order
        // Must have the SAME dedup key (set semantics).
        assert_eq!(
            b1.dedup_key(),
            b2.dedup_key(),
            "braid dedup must be set-based: [10,20,30] == [30,10,20]"
        );
    }

    /// `last_rlc` returns the correct Rlc variant.
    #[test]
    fn last_rlc_returns_correct_variant() {
        let w = make_whip(5, &[100, 200]);
        assert_eq!(w.last_rlc(), Rlc::Cand(200));

        let mut g = w.clone();
        g.kind = ChainKind::PartialGWhip;
        g.rlcs.push(Rlc::GCand(42));
        assert_eq!(g.last_rlc(), Rlc::GCand(42));
    }

    /// Chain round-trips through Clone.
    #[test]
    fn chain_clone_round_trip() {
        let c = Chain {
            kind: ChainKind::Braid,
            target: 77,
            length: 2,
            llcs: vec![1, 2],
            rlcs: vec![Rlc::Cand(3), Rlc::GCand(9)],
            csp_vars: vec![0, 1],
        };
        let c2 = c.clone();
        assert_eq!(c2.target, 77);
        assert_eq!(c2.length, 2);
        assert_eq!(c2.llcs, vec![1, 2]);
        assert_eq!(c2.rlcs, vec![Rlc::Cand(3), Rlc::GCand(9)]);
    }

    /// Whip and braid with identical rlcs content must have different keys (F8 fix:
    /// `kind` discriminator is included in the hash). This is a defensive check;
    /// cross-kind dedup collisions are already unlikely in practice since different
    /// modules use separate dedup tables.
    #[test]
    fn whip_braid_same_rlcs_different_keys() {
        let w = make_whip(0, &[5, 10]);
        let b = make_braid(0, &[5, 10]);
        // F8 fix: kind discriminator is now in the key, so these must differ.
        assert_ne!(
            w.dedup_key(),
            b.dedup_key(),
            "whip and braid with same rlcs must have different dedup keys (F8 kind discriminator)"
        );
    }

    /// FX-4 test (a): Two chains with same `(target, rlcs)` but one PartialWhip
    /// and one PartialBraid → `dedup_key_cross_type()` returns EQUAL keys.
    /// CLIPS `Braids[5].clp:128-139` treats `partial-whip|partial-braid` as a union.
    #[test]
    fn cross_type_dedup_whip_braid_same_key() {
        let w = make_whip(10, &[20, 30]);
        let b = make_braid(10, &[20, 30]);
        // Cross-type key: is_braid discriminator dropped → same key.
        assert_eq!(
            w.dedup_key_cross_type(),
            b.dedup_key_cross_type(),
            "FX-4: partial-whip and partial-braid with same (target, rlcs) must share dedup_key_cross_type()"
        );
        // Regular dedup_key must still differ (F8 preserves kind for same-technique extension).
        assert_ne!(
            w.dedup_key(),
            b.dedup_key(),
            "dedup_key() still distinguishes whip from braid"
        );
    }

    /// FX-4 test (b): Two chains with same `(target, rlcs)` but one PartialWhip
    /// and one PartialGWhip → `dedup_key_cross_type()` returns DIFFERENT keys.
    /// `is_grouped` flag distinguishes the grouped from non-grouped family.
    #[test]
    fn cross_type_dedup_whip_gwhip_different_keys() {
        let w = make_whip(10, &[20, 30]);
        let gw = Chain {
            kind: ChainKind::PartialGWhip,
            target: 10,
            length: 2,
            llcs: vec![0u32; 2],
            rlcs: vec![Rlc::Cand(20), Rlc::Cand(30)],
            csp_vars: vec![0u32; 2],
        };
        // Different is_grouped → different cross-type key.
        assert_ne!(
            w.dedup_key_cross_type(),
            gw.dedup_key_cross_type(),
            "FX-4: partial-whip and partial-gwhip must have DIFFERENT dedup_key_cross_type() (is_grouped differs)"
        );
    }
}
