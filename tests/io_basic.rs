//! R3.0a — io module tests.
//!
//! Covers:
//!   - TextFileSource: empty, single, mixed (blank/comment), oversized,
//!     16×16 hex.
//!   - OrderingSink: out-of-order submission collapses to in-order output.
//!   - Determinism: rate-batch run with threads=1 vs threads=8 gives byte-
//!     identical SHA-256 over the JSONL output.

use std::io::{Cursor, Read, Write};

use md5::Md5;
use sudoku_rs_core::generic::rater::RateResult;
use sudoku_rs_core::generic::techniques::{TechniqueId, Tier};
use sudoku_rs_core::io::{
    OrderingSink, PuzzleSink, PuzzleSource, RatedPuzzle, TextFileSource,
};

fn cur(s: &str) -> Box<dyn Read + Send> {
    Box::new(Cursor::new(s.to_string()))
}

#[test]
fn text_source_empty() {
    let mut src: TextFileSource<9> = TextFileSource::from_reader(cur(""));
    assert!(src.next().unwrap().is_none());
}

#[test]
fn text_source_single_line() {
    let p = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79\n";
    let mut src: TextFileSource<9> = TextFileSource::from_reader(cur(p));
    let r = src.next().unwrap().unwrap();
    assert_eq!(r.i, 0);
    assert_eq!(r.puzzle.len(), 81);
    assert!(src.next().unwrap().is_none());
}

#[test]
fn text_source_mixed_blank_and_comment() {
    let s = "\n# header\n\
             53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79\n\
             \n# another\n\
             .................................................................................\n";
    let mut src: TextFileSource<9> = TextFileSource::from_reader(cur(s));
    let p0 = src.next().unwrap().unwrap();
    let p1 = src.next().unwrap().unwrap();
    assert_eq!(p0.i, 0);
    assert_eq!(p1.i, 1);
    assert!(src.next().unwrap().is_none());
}

#[test]
fn text_source_rejects_oversize_line() {
    // 9×9 → 2N²=162. Provide a 200-char line.
    let s = format!("{}\n", "1".repeat(200));
    let mut src: TextFileSource<9> = TextFileSource::from_reader(cur(&s));
    assert!(src.next().is_err());
}

#[test]
fn text_source_16x16_hex_letters() {
    // Build one full row plus blanks for the remaining 15 rows.
    let mut s = String::new();
    for _ in 0..16 {
        for v in 1..=16u8 {
            let c = if v <= 9 { (b'0' + v) as char } else { (b'A' + v - 10) as char };
            s.push(c);
        }
    }
    s.push('\n');
    let mut src: TextFileSource<16> = TextFileSource::from_reader(cur(&s));
    let p = src.next().unwrap().unwrap();
    assert_eq!(p.puzzle.len(), 256);
    assert!(p.puzzle.contains('G'));
}

// --------------------------------------------------------------------------
// OrderingSink determinism
// --------------------------------------------------------------------------

struct CapSink<const N: usize> {
    seen: Vec<u64>,
}
impl<const N: usize> PuzzleSink<N> for CapSink<N> {
    fn write(&mut self, p: &RatedPuzzle<N>) -> std::io::Result<()> {
        self.seen.push(p.i);
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn mk_rp<const N: usize>(i: u64) -> RatedPuzzle<N> {
    RatedPuzzle::<N>::new(
        i,
        "?".repeat(N * N),
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
fn ordering_sink_drains_in_order() {
    let mut s: OrderingSink<9, CapSink<9>> = OrderingSink::new(CapSink { seen: Vec::new() });
    for i in [3u64, 1, 5, 0, 4, 2] {
        s.submit(mk_rp::<9>(i)).unwrap();
    }
    let inner = s.finish().unwrap();
    assert_eq!(inner.seen, vec![0, 1, 2, 3, 4, 5]);
}

// --------------------------------------------------------------------------
// rate-batch CLI parity: threads=1 vs threads=N → same JSONL bytes.
// We invoke the binary built by `cargo test` via std::process::Command.
// --------------------------------------------------------------------------

fn md5_of_file(p: &std::path::Path) -> String {
    use md5::Digest;
    let mut h = Md5::new();
    let bytes = std::fs::read(p).expect("read");
    h.update(&bytes);
    let d = h.finalize();
    hex::encode(d)
}

fn write_test_input(path: &std::path::Path, n: usize) {
    // n random-ish 9×9 puzzles. Use a small known T1/T2 mix.
    let bases = [
        "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79",
        "..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..",
        "1.......2.9.4...5...6...7...5.9.3.......7.......85..4.7.....6...3...9.8...2.....1",
        ".................................................................................",
    ];
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..n {
        f.write_all(bases[i % bases.len()].as_bytes()).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

fn cargo_bin() -> std::path::PathBuf {
    // Resolve the test-mode bin via env var set by Cargo.
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_sudoku_rs_core"))
}

#[test]
fn rate_batch_thread_count_determinism() {
    let dir = tempfile::tempdir().unwrap();
    let inp = dir.path().join("in.txt");
    let out1 = dir.path().join("out1.jsonl");
    let out8 = dir.path().join("out8.jsonl");
    write_test_input(&inp, 100);

    let bin = cargo_bin();
    for (threads, out) in [(1usize, &out1), (8usize, &out8)] {
        let st = std::process::Command::new(&bin)
            .args([
                "rate-batch",
                "--input",
                inp.to_str().unwrap(),
                "--output",
                out.to_str().unwrap(),
                "--size",
                "9x3x3",
                "--threads",
                &threads.to_string(),
            ])
            .status()
            .unwrap();
        assert!(st.success(), "rate-batch failed at threads={}", threads);
    }

    let h1 = md5_of_file(&out1);
    let h8 = md5_of_file(&out8);
    assert_eq!(
        h1, h8,
        "rate-batch output must be identical regardless of thread count"
    );

    // sanity: 100 lines
    let n_lines = std::fs::read_to_string(&out1)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .count();
    assert_eq!(n_lines, 100);
}

// Reference TechniqueId constant is kept live (silences unused-import warning
// for the integration test crate).
#[allow(dead_code)]
const _T: Option<TechniqueId> = None;
