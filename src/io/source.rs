//! Puzzle sources. The only `PuzzleSource` shipped in R3.0a is
//! [`TextFileSource`] — newline-delimited text encoding (see crate::io docs).

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

use super::types::RawPuzzle;

/// Streaming source of puzzles. Implementors should be cheap to construct and
/// must yield puzzles in deterministic order (the `i` field is assigned in
/// receipt order).
pub trait PuzzleSource<const N: usize> {
    /// Returns the next puzzle, `None` at EOS, or an `io::Error` on read
    /// failure / malformed input.
    fn next(&mut self) -> io::Result<Option<RawPuzzle<N>>>;
}

/// R3.2: outcome of [`TextFileSource::next_lossy`] — for non-strict
/// `rate-batch` mode that must keep input-index alignment under malformed
/// rows.
pub enum NextLossy<const N: usize> {
    /// One puzzle, valid. Advances the index.
    Ok(RawPuzzle<N>),
    /// Malformed line (length mismatch, wrong charset, etc). The index has
    /// already been advanced; the caller should emit a synthetic
    /// `rater_error: true` row at this index. `i` is the index that was
    /// assigned to the malformed entry; `msg` is the parse-error string.
    Bad { i: u64, msg: String },
    /// EOS.
    Eos,
}

/// Reads puzzles from a text file (one per line). See crate::io module docs
/// for the accepted encoding.
///
/// Hard rules enforced here:
///  - line length, after right-trimming whitespace, must equal N*N.
///  - lines longer than 2·N² (before trim) are rejected as a guardrail
///    against feeding a 16×16 puzzle file to a 9×9 pipeline (or vice versa).
///  - blank lines and lines whose first non-whitespace byte is `#` are
///    silently skipped (and do *not* advance the puzzle index).
pub struct TextFileSource<const N: usize> {
    reader: BufReader<Box<dyn Read + Send>>,
    next_idx: u64,
    line_buf: String,
    line_no: u64,
    eof: bool,
}

impl<const N: usize> TextFileSource<N> {
    /// Open `path` for streaming. Use [`Self::from_reader`] for tests.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let f = File::open(path.as_ref())?;
        Ok(Self::from_reader(Box::new(f)))
    }

    pub fn from_reader(r: Box<dyn Read + Send>) -> Self {
        Self {
            reader: BufReader::new(r),
            next_idx: 0,
            line_buf: String::new(),
            line_no: 0,
            eof: false,
        }
    }
}

impl<const N: usize> TextFileSource<N> {
    /// R3.2: lossy variant. On a malformed line returns `NextLossy::Bad`
    /// with the consumed index, allowing callers to emit a synthetic
    /// `rater_error: true` row and keep input-index alignment. Read I/O
    /// failures still surface as [`io::Result::Err`].
    pub fn next_lossy(&mut self) -> io::Result<NextLossy<N>> {
        let nn = N * N;
        let max_len = 2 * nn;
        loop {
            if self.eof {
                return Ok(NextLossy::Eos);
            }
            self.line_buf.clear();
            let n = self.reader.read_line(&mut self.line_buf)?;
            if n == 0 {
                self.eof = true;
                return Ok(NextLossy::Eos);
            }
            self.line_no += 1;
            let trimmed = self.line_buf.trim_end_matches(['\r', '\n', ' ', '\t']);
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('#') {
                continue;
            }
            if self.line_buf.len() > max_len + 8 {
                let i = self.next_idx;
                self.next_idx += 1;
                return Ok(NextLossy::Bad {
                    i,
                    msg: format!(
                        "line {}: length {} exceeds 2·N² ({}) — wrong-size input?",
                        self.line_no,
                        trimmed.len(),
                        max_len
                    ),
                });
            }
            if trimmed.len() != nn {
                let i = self.next_idx;
                self.next_idx += 1;
                return Ok(NextLossy::Bad {
                    i,
                    msg: format!(
                        "line {}: expected length {} (N×N for N={}), got {}",
                        self.line_no,
                        nn,
                        N,
                        trimmed.len()
                    ),
                });
            }
            let i = self.next_idx;
            self.next_idx += 1;
            return Ok(NextLossy::Ok(RawPuzzle::<N>::new(i, trimmed.to_string())));
        }
    }
}

impl<const N: usize> PuzzleSource<N> for TextFileSource<N> {
    fn next(&mut self) -> io::Result<Option<RawPuzzle<N>>> {
        let nn = N * N;
        let max_len = 2 * nn;
        loop {
            if self.eof {
                return Ok(None);
            }
            self.line_buf.clear();
            let n = self.reader.read_line(&mut self.line_buf)?;
            if n == 0 {
                self.eof = true;
                return Ok(None);
            }
            self.line_no += 1;
            if self.line_buf.len() > max_len + 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "line {}: length {} exceeds 2·N² ({}) — wrong-size input?",
                        self.line_no,
                        self.line_buf.trim_end().len(),
                        max_len
                    ),
                ));
            }
            let trimmed = self.line_buf.trim_end_matches(['\r', '\n', ' ', '\t']);
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with('#') {
                continue;
            }
            if trimmed.len() != nn {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "line {}: expected length {} (N×N for N={}), got {}",
                        self.line_no, nn, N, trimmed.len()
                    ),
                ));
            }
            let i = self.next_idx;
            self.next_idx += 1;
            return Ok(Some(RawPuzzle::<N>::new(i, trimmed.to_string())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn cursor(s: &str) -> Box<dyn Read + Send> {
        Box::new(Cursor::new(s.to_string()))
    }

    #[test]
    fn empty_yields_none() {
        let mut src: TextFileSource<9> = TextFileSource::from_reader(cursor(""));
        assert!(src.next().unwrap().is_none());
    }

    #[test]
    fn single_line_9x9() {
        let s = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79\n";
        let mut src: TextFileSource<9> = TextFileSource::from_reader(cursor(s));
        let p = src.next().unwrap().unwrap();
        assert_eq!(p.i, 0);
        assert_eq!(p.puzzle.len(), 81);
        assert!(src.next().unwrap().is_none());
    }

    #[test]
    fn skips_blank_and_comment() {
        let s = "\n# comment\n\
                 53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79\n\
                 # another\n\
                 ................................................................................." ;
        // last line is 81 dots — still a valid input
        let s = format!("{}\n", s);
        let mut src: TextFileSource<9> = TextFileSource::from_reader(cursor(&s));
        let p0 = src.next().unwrap().unwrap();
        assert_eq!(p0.i, 0);
        let p1 = src.next().unwrap().unwrap();
        assert_eq!(p1.i, 1);
        assert!(src.next().unwrap().is_none());
    }

    #[test]
    fn rejects_short_line() {
        let s = "12345\n";
        let mut src: TextFileSource<9> = TextFileSource::from_reader(cursor(s));
        let e = src.next().unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn rejects_line_exceeding_2nn() {
        // 9×9 → 2N² = 162. Build a 200-char line.
        let s = format!("{}\n", "1".repeat(200));
        let mut src: TextFileSource<9> = TextFileSource::from_reader(cursor(&s));
        let e = src.next().unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn parses_16x16_hex_digits() {
        // 16×16 → 256 chars; use solution-like content with hex.
        let line: String = (0..256)
            .map(|i| {
                let v = ((i % 16) + 1) as u8;
                if v <= 9 {
                    (b'0' + v) as char
                } else {
                    (b'A' + v - 10) as char
                }
            })
            .collect();
        let s = format!("{}\n", line);
        let mut src: TextFileSource<16> = TextFileSource::from_reader(cursor(&s));
        let p = src.next().unwrap().unwrap();
        assert_eq!(p.puzzle.len(), 256);
        assert!(p.puzzle.contains('G'));
    }
}
