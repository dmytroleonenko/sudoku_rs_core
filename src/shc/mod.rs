//! Clean-room Rust port of François Cordoliani's SHC (Sudoku Hierarchical
//! Classifier). Phase 1: foundation — board state, base propagator (TB),
//! uniqueness check.
//!
//! Source-of-truth for the algorithmic content is the decompiled reference
//! at `/tmp/shc_decompiled/SHC/SHC/`. No Java code is copied verbatim — this
//! is an idiomatic Rust rewrite (Result/Option/enum/Vec, no global mutable
//! state, no exceptions-as-control-flow).

pub mod board;
pub mod braid_engine;
pub mod error;
pub mod tb;
pub mod te;
pub mod tridagon;
pub mod ua;
pub mod uniqueness;
pub mod wave;

pub use board::{ApplyOutcome, Board, Cell, Region};
pub use braid_engine::{tuple_ctr, BraidResult, EliminationProof};
pub use error::{BoardError, UniquenessError};
pub use tb::{propagate, TbLevel};
pub use te::{rate_bxb, rate_bxbb, rate_te_depth};
pub use uniqueness::verify_unique_solution;
pub use wave::{rate_b, RateError};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod braid_engine_tests;
