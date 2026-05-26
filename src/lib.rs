//! sudoku_rs_core — fast-core solver/rater/generator for sudoku.
//!
//! All public surface is the const-generic pipeline under `generic::*`,
//! parameterised over `(N, BR, BC)`. The legacy 9×9-only modules
//! (`rater`, `backtracker`, `grid`, `techniques`, `generator`, `pipeline_writer`,
//! `rerate`, `ingest_text`, `reverse_construct`) were retired in R3.1a once
//! the generic cascade reached 100% tier parity on a 400-puzzle T2/T3 sweep.
//!
//! Clean-room implementation. Algorithmic ideas (SoA candidate layout, triads,
//! balanced bi-value guessing) inspired by reading the descriptions in the
//! Schoku README and HoDoKu solver class names. **No source code was copied.**
//!
//! Licensed under Apache-2.0 OR MIT.

pub mod bitboard;
pub mod generic;
pub mod schema;
pub mod pipeline_writer_generic;
pub mod rerate_generic;
pub mod generic_writer_helpers;
pub mod io;
pub mod shc;
