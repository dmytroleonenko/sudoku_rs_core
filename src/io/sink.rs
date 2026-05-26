//! Puzzle sinks. The only `PuzzleSink` shipped in R3.0a is [`JsonlSink`] — one
//! JSON object per line, emitting the schema described below.
//!
//! Schema:
//! ```text
//! {
//!   "i":                 u64,                 // input ordinal
//!   "puzzle":            String,              // canonical text (length N²)
//!   "tier":              "T1"|"T2"|"T3"|"T4Plus",
//!   "wave_depth":        u32,                 // cascade outer-loop iterations
//!   "frontier":          [String, ...],       // distinct technique names
//!   "trace":             [String, ...],       // ordered firings
//!   "backtrack_steps":   u64,                 // count_solutions_with_steps(2)
//!   "unique_solution":   bool,                // n_solutions == 1
//!   "solved":            bool,                // cascade reached solved state
//!   "rater_error":       bool                 // contradiction during cascade
//! }
//! ```

use std::fs::File;
use std::io::{self, LineWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::generic::reverse_construct::technique_id_str;
use crate::generic::techniques::{TechniqueId, Tier};

use super::types::RatedPuzzle;

/// Streaming sink for rated puzzles. Implementors must accept puzzles in any
/// order; the wrapping [`crate::io::OrderingSink`] is responsible for restoring
/// input order before forwarding here.
pub trait PuzzleSink<const N: usize> {
    fn write(&mut self, p: &RatedPuzzle<N>) -> io::Result<()>;
    fn flush(&mut self) -> io::Result<()>;
}

#[inline]
fn tier_str(t: Tier) -> &'static str {
    match t {
        Tier::T1 => "T1",
        Tier::T2 => "T2",
        Tier::T3 => "T3",
        Tier::T4Plus => "T4Plus",
    }
}

#[inline]
fn techs_to_strs(ids: &[TechniqueId]) -> Vec<&'static str> {
    ids.iter().copied().map(technique_id_str).collect()
}

/// JSONL sink. One object per line, terminated with `\n`. Uses `LineWriter`
/// so every completed line is flushed to the underlying writer on newline —
/// readers tailing the file see rows within a few milliseconds of production.
pub struct JsonlSink<const N: usize, W: Write> {
    w: LineWriter<W>,
}

impl<const N: usize> JsonlSink<N, File> {
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let f = File::create(path.as_ref())?;
        Ok(Self::from_writer(f))
    }
}

impl<const N: usize, W: Write> JsonlSink<N, W> {
    pub fn from_writer(w: W) -> Self {
        Self {
            w: LineWriter::new(w),
        }
    }

    pub fn into_inner(self) -> io::Result<W> {
        self.w.into_inner().map_err(|e| e.into_error())
    }
}

#[derive(Serialize)]
struct Row<'a> {
    i: u64,
    puzzle: &'a str,
    tier: &'static str,
    wave_depth: u32,
    frontier: Vec<&'static str>,
    trace: Vec<&'static str>,
    backtrack_steps: u64,
    unique_solution: bool,
    solved: bool,
    rater_error: bool,
}

impl<const N: usize, W: Write> PuzzleSink<N> for JsonlSink<N, W> {
    fn write(&mut self, p: &RatedPuzzle<N>) -> io::Result<()> {
        let row = Row {
            i: p.i,
            puzzle: &p.puzzle,
            tier: tier_str(p.rate.tier),
            wave_depth: p.rate.wave_depth,
            frontier: techs_to_strs(&p.rate.frontier),
            trace: techs_to_strs(&p.rate.trace),
            backtrack_steps: p.rate.backtrack_steps,
            unique_solution: p.rate.unique_solution,
            solved: p.rate.solved,
            rater_error: p.rate.rater_error,
        };
        // serde_json::to_writer doesn't add a newline.
        serde_json::to_writer(&mut self.w, &row)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.w.write_all(b"\n")?;
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}
