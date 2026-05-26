//! Output record schema. Mirrors the Python parquet writer at
//! `tools/sudoku_jax/sudoku_jax/dataset_gen/pipeline.py::_build_record`.
//!
//! Schema (column → arrow type):
//!   puzzle                   : List<Int64>(81)
//!   solution                 : List<Int64>(81)
//!   tier                     : Int64
//!   tier_name                : Utf8
//!   propagation_wave_depth   : Int64
//!   technique_frontier       : List<Utf8>
//!   longest_inference_chain  : Int64
//!   backtracking_entropy     : Float64
//!   trace_techniques         : List<Int64>
//!   trace_actions            : Utf8   (JSON string of [[tech_id, {}], ...])
//!   trace_length             : Int64
//!   search_backtracks        : Int64
//!   search_unique            : Boolean
//!   se_rating_estimate       : Float64
//!   puzzle_id                : Utf8
//!   clue_count               : Int64
//!   split                    : Utf8

use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

pub fn record_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "puzzle",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
    true,
        ),
        Field::new(
            "solution",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
    true,
        ),
        Field::new("tier", DataType::Int64, true),
        Field::new("tier_name", DataType::Utf8, true),
        Field::new("propagation_wave_depth", DataType::Int64, true),
        Field::new(
            "technique_frontier",
            DataType::List(Arc::new(Field::new("element", DataType::Utf8, true))),
    true,
        ),
        Field::new("longest_inference_chain", DataType::Int64, true),
        Field::new("backtracking_entropy", DataType::Float64, true),
        Field::new(
            "trace_techniques",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
    true,
        ),
        Field::new("trace_actions", DataType::Utf8, true),
        Field::new("trace_length", DataType::Int64, true),
        Field::new("search_backtracks", DataType::Int64, true),
        Field::new("search_unique", DataType::Boolean, true),
        Field::new("se_rating_estimate", DataType::Float64, true),
        Field::new("puzzle_id", DataType::Utf8, true),
        Field::new("clue_count", DataType::Int64, true),
        Field::new("split", DataType::Utf8, true),
        // Stage 0: SE-equivalent cascade score (nullable for backward compat).
        Field::new("se_score", DataType::Float64, true),
    ]))
}

/// Heuristic backtracking entropy used by the Python pipeline.
pub fn backtracking_entropy(tier: i64, longest_chain: i64) -> f64 {
    if tier <= 2 {
        0.0
    } else if tier == 3 {
        std::f64::consts::LN_2 * longest_chain.max(1) as f64
    } else {
        std::f64::consts::LN_2 * longest_chain.max(1) as f64 + 1.0
    }
}

pub fn se_rating_estimate(tier: i64) -> f64 {
    match tier {
        1 => 1.5,
        2 => 3.5,
        3 => 6.5,
        _ => 9.0,
    }
}

pub fn split_for_id(pid: &str) -> &'static str {
    let h = u32::from_str_radix(&pid[..8], 16).unwrap_or(0) % 100;
    if h < 70 {
        "train"
    } else if h < 80 {
        "validation"
    } else {
        "test"
    }
}

/// Compute a stable puzzle_id matching Python's md5("solution_digits|clue_mask").
pub fn puzzle_id(solution: &[u8; 81], puzzle: &[u8; 81]) -> String {
    use md5::{Digest, Md5};
    let mut s = String::with_capacity(81 + 1 + 81);
    for &d in solution.iter() {
        s.push((b'0' + d) as char);
    }
    s.push('|');
    for &d in puzzle.iter() {
        s.push(if d != 0 { '1' } else { '0' });
    }
    let mut hasher = Md5::new();
    hasher.update(s.as_bytes());
    let digest = hasher.finalize();
    hex::encode(digest)[..16].to_string()
}
