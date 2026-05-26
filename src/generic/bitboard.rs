//! Const-generic bitboard helpers over `[u64; W]` words.
//!
//! Used by `peer_tables.rs` to store cell-sets (peer sets, unit membership) for
//! grids of up to N×N = 256 cells (W=4 covers 16×16 = 256 cells exactly).

/// A cell-set stored as `W` u64 words, addressing cells `0 .. 64*W`.
#[derive(Clone, Copy, Debug)]
pub struct BitSet<const W: usize> {
    pub words: [u64; W],
}

impl<const W: usize> BitSet<W> {
    #[inline(always)]
    pub const fn empty() -> Self {
        BitSet { words: [0u64; W] }
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize) {
        let w = idx >> 6;
        let b = idx & 63;
        self.words[w] |= 1u64 << b;
    }

    #[inline(always)]
    pub fn clear(&mut self, idx: usize) {
        let w = idx >> 6;
        let b = idx & 63;
        self.words[w] &= !(1u64 << b);
    }

    #[inline(always)]
    pub fn test(&self, idx: usize) -> bool {
        let w = idx >> 6;
        let b = idx & 63;
        (self.words[w] >> b) & 1 == 1
    }

    #[inline(always)]
    pub fn popcount(&self) -> u32 {
        let mut c = 0u32;
        let mut i = 0;
        while i < W {
            c += self.words[i].count_ones();
            i += 1;
        }
        c
    }

    /// Iterate set bits in ascending index order, calling `f(idx)` for each.
    #[inline]
    pub fn for_each<F: FnMut(usize)>(&self, mut f: F) {
        for w in 0..W {
            let mut m = self.words[w];
            let base = w * 64;
            while m != 0 {
                let b = m.trailing_zeros() as usize;
                m &= m - 1;
                f(base + b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_test_clear_popcount() {
        let mut b: BitSet<2> = BitSet::empty();
        b.set(0);
        b.set(63);
        b.set(64);
        b.set(127);
        assert!(b.test(0));
        assert!(b.test(63));
        assert!(b.test(64));
        assert!(b.test(127));
        assert!(!b.test(1));
        assert_eq!(b.popcount(), 4);
        b.clear(63);
        assert!(!b.test(63));
        assert_eq!(b.popcount(), 3);
    }

    #[test]
    fn for_each_ascending() {
        let mut b: BitSet<3> = BitSet::empty();
        for &i in &[5usize, 17, 64, 130] {
            b.set(i);
        }
        let mut got = Vec::new();
        b.for_each(|i| got.push(i));
        assert_eq!(got, vec![5, 17, 64, 130]);
    }
}
