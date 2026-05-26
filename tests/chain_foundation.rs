//! Integration test: CSP chain foundation round-trip for N=9.
//!
//! Loads a 9×9 grid, builds CSP+glabel tables, builds a ResolutionState,
//! eliminates one candidate, and asserts g_alive correctness.

use sudoku_rs_core::generic::csp_tables::build_csp_tables_n9;
use sudoku_rs_core::generic::glabel_tables::{build_glabel_tables_n9, label_in_glabel};
use sudoku_rs_core::generic::resolution_state::ResolutionState;
use sudoku_rs_core::generic::grid::Grid;

/// Build CSP and glabel tables, construct ResolutionState from an empty
/// 9×9 grid, then eliminate all but one member of glabel 0 and verify the
/// glabel becomes dead in `g_alive`.
#[test]
fn chain_foundation_glabel_cascade() {
    // Build tables.
    let (csp, _lg, _cg) = build_csp_tables_n9();
    let (glab_table, _gg) = build_glabel_tables_n9(&csp);

    // Verify table sizes.
    assert_eq!(glab_table.members_of.len(), 486,
        "N=9 must have 486 glabels");
    assert_eq!(csp.labels_of.len(), 324,
        "N=9 must have 324 CSP-Variables");

    // Create a fully-open 9×9 grid.
    let grid = Grid::<9, 3, 3>::empty();
    let mut rs = ResolutionState::from_grid_9x9(&grid, &glab_table);

    // All 729 candidates should be alive initially.
    let alive_count = rs.cand_alive.popcount();
    assert_eq!(alive_count, 729, "all 729 labels alive in empty grid");

    // Glabel 0 should be alive with 3 members.
    assert!(rs.g_cand_alive(0), "glabel 0 must be alive");
    let members = glab_table.members_of[0].clone();
    assert_eq!(members.len(), 3);

    // Verify label_in_glabel for each member.
    for &m in &members {
        assert!(label_in_glabel(m, 0, &glab_table),
            "member {} should be in glabel 0", m);
    }

    // Eliminate first two members → glabel goes from alive to dead.
    rs.eliminate_candidate(members[0], &glab_table);
    assert!(rs.g_cand_alive(0), "glabel 0 still alive with 2 members");

    rs.eliminate_candidate(members[1], &glab_table);
    assert!(!rs.g_cand_alive(0), "glabel 0 dead after 2 eliminations (only 1 member left)");

    // Third member is still alive as a regular candidate.
    assert!(rs.cand_alive.test(members[2] as usize),
        "third member still alive as regular candidate");

    // Idempotency: re-eliminating member[0] changes nothing.
    let g_support_before = rs.g_support.clone();
    rs.eliminate_candidate(members[0], &glab_table);
    assert_eq!(rs.g_support, g_support_before, "double-elimination must be idempotent");
}

/// CSP table structure: verify the 4-family layout and link graph symmetry.
#[test]
fn chain_foundation_csp_structure() {
    let (csp, lg, cg) = build_csp_tables_n9();

    // 4 families × 81 = 324 variables.
    assert_eq!(csp.labels_of.len(), 324);
    assert_eq!(csp.kind_of.len(), 324);
    assert_eq!(csp.vars_of.len(), 729);

    // Every label belongs to exactly 4 distinct CSP-Variables.
    for lab in 0usize..729 {
        let vars = csp.vars_of[lab];
        let mut sorted = vars;
        sorted.sort_unstable();
        assert_eq!(sorted[0], sorted[0]); // trivial
        // Check all 4 are distinct.
        for i in 0..3 {
            assert_ne!(sorted[i], sorted[i + 1],
                "label {} has duplicate CSP-Variable", lab);
        }
    }

    // Link graph must be symmetric.
    // Spot-check: labels 0 and 9 (same row, same digit) should be linked.
    // label 0 = (row=0, col=0, dbit=0); label 9 = (row=0, col=1, dbit=0).
    assert!(lg.linked[0].test(9), "row-peer labels must be linked");
    assert!(lg.linked[9].test(0), "link graph must be symmetric");

    // CspLinkGraph: label 0 should have 8 alternatives in each of its 4 slots.
    for slot in 0..4 {
        assert_eq!(cg.alternatives[0][slot].len(), 8,
            "slot {} should have 8 alternatives for label 0", slot);
    }
}
