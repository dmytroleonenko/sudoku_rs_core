//! Per-puzzle data carriers for the rate-batch pipeline.

use std::marker::PhantomData;

use crate::generic::rater::RateResult;

/// Raw (un-rated) puzzle. `puzzle` is the canonical text encoding of length
/// N×N (see crate::io module docs). `i` is the input ordinal (0-based) — used
/// downstream by [`crate::io::OrderingSink`] to restore input order after
/// parallel rating.
#[derive(Debug, Clone)]
pub struct RawPuzzle<const N: usize> {
    pub i: u64,
    pub puzzle: String,
    _marker: PhantomData<()>,
}

impl<const N: usize> RawPuzzle<N> {
    #[inline]
    pub fn new(i: u64, puzzle: String) -> Self {
        Self {
            i,
            puzzle,
            _marker: PhantomData,
        }
    }
}

/// Rated puzzle: input ordinal + original text + cascade outcome. Carries the
/// full [`RateResult`] so the sink can serialize whichever subset of fields it
/// chooses.
#[derive(Debug, Clone)]
pub struct RatedPuzzle<const N: usize> {
    pub i: u64,
    pub puzzle: String,
    pub rate: RateResult,
}

impl<const N: usize> RatedPuzzle<N> {
    #[inline]
    pub fn new(i: u64, puzzle: String, rate: RateResult) -> Self {
        Self { i, puzzle, rate }
    }
}
