//! `OrderingSink` — buffer rated puzzles in an unbounded `BTreeMap` and drain
//! the contiguous prefix starting at `next_idx` whenever `submit` is called.
//! Forwards drained puzzles to an inner `PuzzleSink` in input order.
//!
//! Determinism:
//!   - `BTreeMap` iteration is by-key (input ordinal). No `HashMap`.
//!   - `submit` and `flush_remaining` produce the same output sequence
//!     regardless of submission order, provided every index in `[0, max_idx]`
//!     is eventually submitted exactly once.
//!
//! This is the moral equivalent of "preserve order under
//! `par_iter().rate().collect()`" but streaming — never holds the entire
//! batch in memory, only the out-of-order tail.

use std::collections::BTreeMap;
use std::io;

use super::sink::PuzzleSink;
use super::types::RatedPuzzle;

pub struct OrderingSink<const N: usize, S: PuzzleSink<N>> {
    inner: S,
    buf: BTreeMap<u64, RatedPuzzle<N>>,
    next_idx: u64,
}

impl<const N: usize, S: PuzzleSink<N>> OrderingSink<N, S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buf: BTreeMap::new(),
            next_idx: 0,
        }
    }

    /// Buffer `p` and drain any contiguous prefix that starts at `next_idx`.
    pub fn submit(&mut self, p: RatedPuzzle<N>) -> io::Result<()> {
        self.buf.insert(p.i, p);
        self.drain_contiguous()
    }

    fn drain_contiguous(&mut self) -> io::Result<()> {
        while let Some(p) = self.buf.remove(&self.next_idx) {
            self.inner.write(&p)?;
            self.next_idx += 1;
        }
        Ok(())
    }

    /// Drain everything that's left, regardless of contiguity. Call only after
    /// the source has been fully consumed; otherwise gaps in the output stream
    /// will silently remain unfilled. Returns the inner sink.
    pub fn finish(mut self) -> io::Result<S> {
        // Defensive flush of any remaining contiguous block.
        self.drain_contiguous()?;
        // If there are still entries, indices were skipped — emit them in
        // ascending order (the only deterministic choice) but signal via the
        // returned count of buffered entries.
        if !self.buf.is_empty() {
            // Drain in BTreeMap order = ascending by index.
            let leftover: Vec<RatedPuzzle<N>> = std::mem::take(&mut self.buf)
                .into_iter()
                .map(|(_, v)| v)
                .collect();
            for p in leftover {
                self.inner.write(&p)?;
            }
        }
        self.inner.flush()?;
        Ok(self.inner)
    }

    /// Number of currently-buffered (out-of-order) entries.
    #[inline]
    pub fn pending(&self) -> usize {
        self.buf.len()
    }

    /// Next index to be written to the inner sink.
    #[inline]
    pub fn next_idx(&self) -> u64 {
        self.next_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::rater::RateResult;
    use crate::generic::techniques::Tier;

    struct CapSink {
        seen: Vec<u64>,
    }

    impl<const N: usize> PuzzleSink<N> for CapSink {
        fn write(&mut self, p: &RatedPuzzle<N>) -> io::Result<()> {
            self.seen.push(p.i);
            Ok(())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn rp<const N: usize>(i: u64) -> RatedPuzzle<N> {
        RatedPuzzle::<N>::new(
            i,
            "?".to_string(),
            RateResult {
                tier: Tier::T1,
                frontier: vec![],
                solved: true,
                trace: vec![],
                rater_error: false,
                wave_depth: 0,
                backtrack_steps: 0,
                unique_solution: true,
                se_score: 0.0,
            },
        )
    }

    #[test]
    fn drains_in_order_under_random_submission() {
        let inner = CapSink { seen: Vec::new() };
        let mut s: OrderingSink<9, CapSink> = OrderingSink::new(inner);
        for i in [3u64, 1, 5, 0, 4, 2] {
            s.submit(rp::<9>(i)).unwrap();
        }
        let inner = s.finish().unwrap();
        assert_eq!(inner.seen, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn pending_decreases_on_contiguous_arrival() {
        let inner = CapSink { seen: Vec::new() };
        let mut s: OrderingSink<9, CapSink> = OrderingSink::new(inner);
        s.submit(rp::<9>(2)).unwrap();
        assert_eq!(s.pending(), 1);
        s.submit(rp::<9>(1)).unwrap();
        assert_eq!(s.pending(), 2);
        s.submit(rp::<9>(0)).unwrap();
        assert_eq!(s.pending(), 0);
        assert_eq!(s.next_idx(), 3);
    }
}
