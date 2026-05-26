//! Stream I/O for the rate-batch pipeline (R3.0a).
//!
//! Layout:
//!  - [`types`]   — `RawPuzzle<N>`, `RatedPuzzle<N>` (size-marker generics).
//!  - [`source`]  — `PuzzleSource<N>` trait + `TextFileSource<N>`.
//!  - [`sink`]    — `PuzzleSink<N>` trait + `JsonlSink<N>`.
//!  - [`ordered`] — `OrderingSink<N, S>` wrapper that drains contiguous indices
//!                  from an unbounded `BTreeMap` buffer to preserve input order
//!                  under parallel rating.
//!
//! Encoding (text format, one puzzle per line):
//!   - `'.'` or `'0'` = blank
//!   - `'1'..='9'` digits 1..=9
//!   - `'A'..='G'` (or lowercase) digits 10..=16 for sizes 12×12 / 16×16
//!
//! Lines are right-trimmed. Blank lines and `#`-prefixed comment lines are
//! skipped. Lines whose length exceeds 2·N² are rejected (defensive guard
//! against accidentally feeding wrong-size puzzle files).

pub mod types;
pub mod source;
pub mod sink;
pub mod ordered;
pub mod ingest;

pub use types::{RawPuzzle, RatedPuzzle};
pub use source::{NextLossy, PuzzleSource, TextFileSource};
pub use sink::{PuzzleSink, JsonlSink};
pub use ordered::OrderingSink;
