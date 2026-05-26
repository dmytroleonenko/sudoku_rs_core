//! `ingest-seeds` — import plain-text puzzle lists into parquet (Stage S, R3.4).
//!
//! Schema:
//!   - `puzzle: utf8`       — normalized (`.` for empties), 81 chars
//!   - `clue_count: int32`  — number of given digits
//!   - `unique: bool`       — true if solver confirmed exactly one solution
//!                           (null when `--solve` is off)
//!   - `solution: utf8`     — 81-char solution (null when not solved or non-unique)
//!
//! Only 9×9 (`N=9`, `BR=3`, `BC=3`) is supported in Stage S; the size is
//! hard-wired at the call sites. Extending to other sizes requires routing on
//! the `--size` flag (deferred to Stage T+).

use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Int32Builder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::{EnabledStatistics, WriterProperties};

use crate::generic::grid::Grid;
use crate::generic::search::{count_solutions_up_to, solve_unique};

// ── Schema ──────────────────────────────────────────────────────────────────

/// Arrow schema for the ingest-seeds parquet output.
pub fn ingest_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("puzzle", DataType::Utf8, false),
        Field::new("clue_count", DataType::Int32, false),
        Field::new("unique", DataType::Boolean, true),
        Field::new("solution", DataType::Utf8, true),
    ]))
}

// ── Public config ────────────────────────────────────────────────────────────

pub struct IngestConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Human-readable label written to the manifest sidecar.
    pub label: String,
    /// If true, solve each puzzle and populate `unique` / `solution`.
    pub solve: bool,
    /// Cap on number of input lines processed (0 = unlimited).
    pub max: usize,
}

// ── Row type ─────────────────────────────────────────────────────────────────

struct IngestRow {
    puzzle: String,   // normalized
    clue_count: i32,
    unique: Option<bool>,
    solution: Option<String>,
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

/// Normalize a raw 81-char line: replace `'0'` with `'.'`, trim to 81.
/// Returns `None` if the line is not exactly 81 printable ASCII chars.
fn normalize_9x9(raw: &str) -> Option<String> {
    let trimmed = raw.trim_end_matches(['\r', '\n', ' ', '\t']);
    // Accept puzzle strings optionally followed by a space + solution (ignore the rest)
    let puzzle_part = trimmed.split_whitespace().next().unwrap_or(trimmed);
    if puzzle_part.len() != 81 {
        return None;
    }
    let mut out = String::with_capacity(81);
    for b in puzzle_part.bytes() {
        match b {
            b'0' => out.push('.'),
            b'.' | b'1'..=b'9' => out.push(b as char),
            _ => return None,
        }
    }
    Some(out)
}

fn count_clues(puzzle: &str) -> i32 {
    puzzle.chars().filter(|&c| c != '.').count() as i32
}

// ── Core ingest logic ─────────────────────────────────────────────────────────

/// Parse + optionally solve puzzles from `cfg.input`.
/// Returns `(rows, count_in, count_kept, count_unique)`.
fn collect_rows(cfg: &IngestConfig) -> io::Result<(Vec<IngestRow>, usize, usize, usize)> {
    let file = std::fs::File::open(&cfg.input)?;
    let reader = BufReader::new(file);

    let mut rows: Vec<IngestRow> = Vec::new();
    let mut count_in: usize = 0;
    let mut count_unique: usize = 0;

    for line_result in reader.lines() {
        let line = line_result?;
        let trimmed = line.trim_end_matches(['\r', '\n', ' ', '\t']);
        // Skip blank lines and comments.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        count_in += 1;
        if cfg.max > 0 && count_in > cfg.max {
            // We've already incremented count_in past max; correct it.
            count_in -= 1;
            break;
        }
        let puzzle = match normalize_9x9(trimmed) {
            Some(p) => p,
            None => {
                eprintln!("warning: skipping malformed line {}: {:?}", count_in, &trimmed[..trimmed.len().min(20)]);
                count_in -= 1;
                continue;
            }
        };
        let clue_count = count_clues(&puzzle);

        let (unique, solution) = if cfg.solve {
            // Parse into Grid<9,3,3>.
            match Grid::<9, 3, 3>::from_str(&puzzle) {
                None => {
                    eprintln!("warning: puzzle {} failed to parse into grid — skipping", count_in);
                    count_in -= 1;
                    continue;
                }
                Some(g) => {
                    let n_sol = count_solutions_up_to::<9, 3, 3>(&g, 2);
                    if n_sol == 1 {
                        count_unique += 1;
                        let sol_grid = solve_unique::<9, 3, 3>(&g);
                        (Some(true), sol_grid.map(|sg| sg.to_string_grid()))
                    } else {
                        (Some(false), None)
                    }
                }
            }
        } else {
            (None, None)
        };

        rows.push(IngestRow { puzzle, clue_count, unique, solution });
    }

    let count_kept = rows.len();
    Ok((rows, count_in, count_kept, count_unique))
}

// ── Parquet writer ────────────────────────────────────────────────────────────

fn write_parquet(path: &Path, rows: &[IngestRow]) -> io::Result<()> {
    let schema = ingest_schema();
    let n = rows.len();

    let mut puzzle_b = StringBuilder::with_capacity(n, n * 82);
    let mut clue_b = Int32Builder::with_capacity(n);
    let mut unique_b = BooleanBuilder::with_capacity(n);
    let mut solution_b = StringBuilder::with_capacity(n, n * 82);

    for r in rows {
        puzzle_b.append_value(&r.puzzle);
        clue_b.append_value(r.clue_count);
        match r.unique {
            Some(v) => unique_b.append_value(v),
            None => unique_b.append_null(),
        }
        match &r.solution {
            Some(s) => solution_b.append_value(s),
            None => solution_b.append_null(),
        }
    }

    let cols: Vec<ArrayRef> = vec![
        Arc::new(puzzle_b.finish()),
        Arc::new(clue_b.finish()),
        Arc::new(unique_b.finish()),
        Arc::new(solution_b.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), cols)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = std::fs::File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    writer
        .close()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    Ok(())
}

// ── Manifest sidecar ──────────────────────────────────────────────────────────

fn write_manifest(
    out_path: &Path,
    label: &str,
    source: &Path,
    count_in: usize,
    count_kept: usize,
    count_unique: usize,
) -> io::Result<()> {
    let manifest_path = {
        let mut p = out_path.to_path_buf();
        let ext = p.extension().unwrap_or_default().to_string_lossy().into_owned();
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        p.set_file_name(format!("{}.{}.manifest.json", stem, ext));
        p
    };
    let ts = chrono_now_iso();
    let src_str = source.display().to_string();
    let json = format!(
        r#"{{
  "label": {label},
  "source": {source},
  "count_in": {count_in},
  "count_kept": {count_kept},
  "count_unique": {count_unique},
  "generated_at": {ts}
}}"#,
        label = serde_json::to_string(label).unwrap(),
        source = serde_json::to_string(&src_str).unwrap(),
        count_in = count_in,
        count_kept = count_kept,
        count_unique = count_unique,
        ts = serde_json::to_string(&ts).unwrap(),
    );
    std::fs::write(&manifest_path, json)?;
    eprintln!("manifest: {}", manifest_path.display());
    Ok(())
}

fn chrono_now_iso() -> String {
    // stdlib only — no chrono dep.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as ISO 8601 UTC using only integer arithmetic.
    let mut s = secs;
    let sec = s % 60; s /= 60;
    let min = s % 60; s /= 60;
    let hour = s % 24; s /= 24;
    // Days since epoch → date (Gregorian, non-leap approximation good ±1 day).
    let days = s;
    // Zeller-ish: use a simple cumulative day table.
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hour, min, sec)
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Days since 1970-01-01.
    let mut year: u64 = 1970;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month: u64 = 1;
    for md in &month_days {
        if days < *md { break; }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run the ingest-seeds pipeline according to `cfg`.
/// Writes parquet + manifest sidecar, prints a summary to stderr.
pub fn run(cfg: &IngestConfig) -> io::Result<()> {
    eprintln!(
        "ingest-seeds: reading {:?}  solve={} max={}",
        cfg.input, cfg.solve, if cfg.max == 0 { "∞".to_string() } else { cfg.max.to_string() }
    );

    let (rows, count_in, count_kept, count_unique) = collect_rows(cfg)?;

    eprintln!(
        "ingest-seeds: parsed {} lines, kept {} puzzles, {} unique",
        count_in, count_kept, count_unique
    );

    write_parquet(&cfg.output, &rows)?;
    eprintln!("ingest-seeds: wrote {:?}", cfg.output);

    write_manifest(
        &cfg.output,
        &cfg.label,
        &cfg.input,
        count_in,
        count_kept,
        count_unique,
    )?;

    Ok(())
}
