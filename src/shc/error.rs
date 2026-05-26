//! Error types for the SHC port.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoardError {
    /// Input string did not have exactly 81 characters.
    BadLength(usize),
    /// Character was not in `[1-9]` or one of the empty markers `.0`.
    BadChar { index: usize, ch: char },
    /// Initial assignments produced an immediate contradiction.
    InitialContradiction,
}

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoardError::BadLength(n) => write!(f, "expected 81 characters, got {}", n),
            BoardError::BadChar { index, ch } => {
                write!(f, "invalid character {:?} at index {}", ch, index)
            }
            BoardError::InitialContradiction => write!(f, "initial puzzle is contradictory"),
        }
    }
}

impl std::error::Error for BoardError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniquenessError {
    NoSolution,
    Multiple,
}

impl fmt::Display for UniquenessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniquenessError::NoSolution => write!(f, "puzzle has no solution"),
            UniquenessError::Multiple => write!(f, "puzzle has multiple solutions"),
        }
    }
}

impl std::error::Error for UniquenessError {}
