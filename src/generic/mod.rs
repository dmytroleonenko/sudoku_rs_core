//! Generic (const-generic) sudoku substrate.
//!
//! Parallel implementation to the existing N=9-specialised modules
//! (`crate::grid`, `crate::bitboard`, `crate::backtracker`). The goal is to
//! support arbitrary block sizes — currently exercised at (6,2,3), (9,3,3),
//! (12,3,4), (16,4,4) — under a single code path.
//!
//! Design notes:
//!  - The struct is parameterised by `<const N: usize, const BR: usize, const BC: usize>`
//!    where `BR * BC == N` (debug_assert at construction). We do NOT expose
//!    `[T; N*N]` arrays in struct fields because that requires the unstable
//!    `generic_const_exprs` feature. Instead, candidate / solved storage is
//!    `Vec<...>` of length `N*N`, allocated once at construction.
//!  - Candidate masks use `u32` (sufficient for N ≤ 32; for the four target
//!    sizes the high bits are unused).
//!  - Peer tables are computed once per `(N, BR, BC)` instantiation and cached
//!    behind a `OnceLock` (see `peer_tables.rs`).
//!  - This module mirrors the *semantics* of `crate::grid::Grid::assign`,
//!    `crate::grid::Grid::eliminate` and
//!    `crate::backtracker::propagate_singles` exactly. Existing 9×9 paths are
//!    unaffected.

pub mod bitboard;
pub mod peer_tables;
pub mod grid;
pub mod chain_model;
pub mod csp_tables;
pub mod glabel_tables;
pub mod resolution_state;
pub mod backtracker;
pub mod techniques;
pub mod rater;
pub mod search;
pub mod generator;
pub mod spec;
pub mod generator_constrained;
pub mod reverse_construct;
pub mod chain_rating;
pub mod aic_reverse;
pub mod fish_reverse;
pub mod ur_type2_reverse;
pub mod als_xz_reverse;
pub mod naked_quad_reverse;
pub mod hidden_quad_reverse;
pub mod canonical;
pub mod nested_aic_reverse;
pub mod nested_fc_reverse;
pub mod cfc_reverse;
pub mod rfc_reverse;
pub mod dfc_reverse;
pub mod gbraid_reverse;
pub mod vicinity;

pub use grid::{Grid, AssignErr};
pub use rater::{
    rate, rate_excluding, rate_with_mode, rate_with_uniqueness, rate_with_uniqueness_mode,
    RateResult, SolverMode,
};
pub use peer_tables::PeerTable;
pub use search::{count_solutions_up_to, count_solutions_with_steps, solve_unique, search_random};
pub use generator::{gen_unique_puzzle, gen_unique_puzzle_with_solution, random_solution, GenConfig};
pub use spec::TechniqueChainSpec;
pub use generator_constrained::{gen_constrained, try_generate_constrained};
pub use reverse_construct::{
    batch_reverse_construct, parse_technique_id, reverse_construct, technique_id_str,
    ReverseResult, ReverseSpec, ALL_TECHNIQUE_IDS,
};
pub use aic_reverse::{
    aic_reverse_construct, batch_aic_reverse_construct, find_first_aic_chain_length,
    AicReverseSpec,
};
pub use fish_reverse::{
    batch_fish_reverse_construct, find_first_fish_size, fish_reverse_construct,
    FishReverseSpec,
};
pub use ur_type2_reverse::{
    batch_ur_type2_reverse_construct, find_first_ur_type2, try_ur_eliminate_dryrun,
    ur_type2_reverse_construct, UrType2ReverseSpec,
};
pub use als_xz_reverse::{
    als_xz_reverse_construct, batch_als_xz_reverse_construct, find_first_als_xz,
    try_als_xz_eliminate_dryrun, AlsXzReverseSpec,
};
pub use naked_quad_reverse::{
    batch_naked_quad_reverse_construct, find_first_naked_quad,
    naked_quad_reverse_construct, try_naked_quad_eliminate_dryrun, NakedQuadReverseSpec,
};
pub use canonical::{canonical_form, canonical_hash};
pub use hidden_quad_reverse::{
    batch_hidden_quad_reverse_construct, find_first_hidden_quad,
    hidden_quad_reverse_construct, try_hidden_quad_eliminate_dryrun, HiddenQuadReverseSpec,
};
pub use nested_aic_reverse::{
    construct as nested_aic_construct,
    batch_construct as nested_aic_batch_construct,
    NestedAicConfig, NestedAicEntry,
};
pub use nested_fc_reverse::{
    batch_nested_fc_reverse_construct,
    NestedFcReverseSpec, NestedFcEntry,
};
pub use cfc_reverse::{batch_cfc_reverse_construct, cfc_reverse_construct, CfcReverseSpec};
pub use rfc_reverse::{
    batch_rfc_reverse_construct, rfc_reverse_spec, ConstructedPuzzle as RfcConstructedPuzzle,
    RfcReverseSpec,
};
pub use dfc_reverse::{batch_dfc_reverse_construct, DfcReverseSpec, ConstructedPuzzle as DfcConstructedPuzzle};
