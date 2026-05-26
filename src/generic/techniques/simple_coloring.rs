//! # SimpleColoring
//!
//! ## Inputs
//! Reads from `Grid<N,BR,BC>`: candidate bitmasks per cell, placed digits.
//!
//! ## Mutates
//! May eliminate candidates and/or place digits in `grid` (in place). Does
//! not touch any state outside the passed-in `Grid`.
//!
//! ## Returns
//! `Some(TechniqueProgress)` if at least one elimination/placement happened
//! on this call; `None` if the technique did not fire. Sets
//! `progress.contradiction = true` if elimination drove a cell to 0
//! candidates.
//!
//! ## Performance budget
//! Target: < 100 µs/grid on 9×9 on commodity x86_64.
//!
//! ## Algorithm reference
//! Single-digit coloring: 2-color conjugate-pair graph; Rule 2 (same-color
//! conflict → eliminate that color), Rule 4 (cell sees both colors → loses
//! digit). (Hodoku simple coloring glossary.)
//!
//! ## AlphaEvolve contract
//! This file is self-contained. No cross-file state. `Technique` /
//! `RatedTechnique` traits are stable contract; do not change their
//! signatures. May freely refactor internal helpers, data structures, and
//! per-instance scratch.

use super::super::grid::Grid;
use super::{Technique, TechniqueId, TechniqueProgress, Tier};
use super::RatedTechnique;

pub struct SimpleColoring;

fn peer_lookup<const N: usize, const BR: usize, const BC: usize>(
    grid: &Grid<N, BR, BC>,
    cell: usize,
) -> Vec<bool> {
    let nn = N * N;
    let mut v = vec![false; nn];
    let table = grid.table();
    for &p in &table.cells[cell].peers {
        v[p as usize] = true;
    }
    v
}

impl<const N: usize, const BR: usize, const BC: usize> Technique<N, BR, BC> for SimpleColoring {
    fn id(&self) -> TechniqueId { TechniqueId::SimpleColoring }
    fn tier(&self) -> Tier { Tier::T3 }
    fn name(&self) -> &'static str { "simple_coloring" }

    fn apply(&self, grid: &mut Grid<N, BR, BC>) -> Option<TechniqueProgress> {
        let nn = N * N;
        let table = grid.table().clone();
        for d_bit in 0..(N as u8) {
            let bit = 1u32 << d_bit;
            let digit = d_bit + 1;

            // Build adjacency: for each unit, if exactly 2 unsolved cells have
            // d as candidate, add an edge between them.
            let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nn];
            for unit in &table.units {
                let mut found: [usize; 32] = [0; 32];
                let mut nf = 0usize;
                for &c in unit {
                    let c = c as usize;
                    if grid.solved[c] != 0 { continue; }
                    if grid.candidates[c] & bit != 0 {
                        if nf < 32 { found[nf] = c; }
                        nf += 1;
                        if nf > 2 { break; }
                    }
                }
                if nf == 2 {
                    let (a, b) = (found[0], found[1]);
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }

            // 2-color each connected component via BFS.
            let mut color: Vec<i8> = vec![-1; nn];
            let mut comp_id: Vec<i32> = vec![-1; nn];
            let mut comps: Vec<Vec<(usize, i8)>> = Vec::new();
            for start in 0..nn {
                if color[start] != -1 { continue; }
                if adj[start].is_empty() { continue; }
                if grid.solved[start] != 0 { continue; }
                if grid.candidates[start] & bit == 0 { continue; }
                let cid = comps.len() as i32;
                let mut comp_cells: Vec<(usize, i8)> = Vec::new();
                color[start] = 0;
                comp_id[start] = cid;
                let mut queue: Vec<usize> = vec![start];
                let mut head = 0;
                while head < queue.len() {
                    let c = queue[head]; head += 1;
                    comp_cells.push((c, color[c]));
                    for &n in &adj[c] {
                        if color[n] == -1 {
                            color[n] = 1 - color[c];
                            comp_id[n] = cid;
                            queue.push(n);
                        }
                    }
                }
                comps.push(comp_cells);
            }

            // Helper closure: shares-unit (via peer_lookup).
            // Pre-compute peer lookups lazily per cell.
            let mut elims: Vec<(usize, u8)> = Vec::new();

            // Rule 2: same-color same-unit conflict → that color is bad.
            for comp in &comps {
                let mut bad_color: Option<i8> = None;
                'outer: for i in 0..comp.len() {
                    let pi = peer_lookup::<N, BR, BC>(grid, comp[i].0);
                    for j in (i + 1)..comp.len() {
                        if comp[i].1 == comp[j].1 && pi[comp[j].0] {
                            bad_color = Some(comp[i].1);
                            break 'outer;
                        }
                    }
                }
                if let Some(bc) = bad_color {
                    for &(c, col) in comp {
                        if col == bc {
                            match grid.eliminate(c, digit) {
                                Ok(true) => elims.push((c, digit)),
                                Ok(false) => {}
                                Err(_) => {
                                    let mut prog = TechniqueProgress::default();
                                    prog.eliminations = elims;
                                    prog.contradiction = true;
                                    return Some(prog);
                                }
                            }
                        }
                    }
                }
            }
            if !elims.is_empty() {
                let mut prog = TechniqueProgress::default();
                prog.eliminations = elims;
                return Some(prog);
            }

            // Rule 4: cells outside component peering with both colors lose d.
            for cell in 0..nn {
                if grid.solved[cell] != 0 { continue; }
                if grid.candidates[cell] & bit == 0 { continue; }
                if comp_id[cell] != -1 { continue; }
                let pl = peer_lookup::<N, BR, BC>(grid, cell);
                let mut fired_for_cell = false;
                for comp in &comps {
                    let mut sees0 = false;
                    let mut sees1 = false;
                    for &(cc, col) in comp {
                        if pl[cc] {
                            if col == 0 { sees0 = true; } else { sees1 = true; }
                            if sees0 && sees1 { break; }
                        }
                    }
                    if sees0 && sees1 {
                        match grid.eliminate(cell, digit) {
                            Ok(true) => { elims.push((cell, digit)); fired_for_cell = true; }
                            Ok(false) => {}
                            Err(_) => {
                                let mut prog = TechniqueProgress::default();
                                prog.eliminations = elims;
                                prog.contradiction = true;
                                return Some(prog);
                            }
                        }
                        if fired_for_cell { break; }
                    }
                }
            }
            if !elims.is_empty() {
                let mut prog = TechniqueProgress::default();
                prog.eliminations = elims;
                return Some(prog);
            }
        }
        None
    }
}

impl<const N: usize, const BR: usize, const BC: usize> RatedTechnique<N, BR, BC>
    for SimpleColoring
{
    fn base_rating(&self) -> f64 { 4.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::grid::Grid;

    #[test]
    fn coloring_no_fire_empty_9x9() {
        let mut g: Grid<9, 3, 3> = Grid::empty();
        let t = SimpleColoring;
        assert!(<SimpleColoring as Technique<9,3,3>>::apply(&t, &mut g).is_none());
    }

    #[test]
    fn coloring_smoke_6x6() {
        let mut g: Grid<6, 2, 3> = Grid::empty();
        let t = SimpleColoring;
        let _ = <SimpleColoring as Technique<6,2,3>>::apply(&t, &mut g);
    }

    #[test]
    fn coloring_smoke_12x12() {
        let mut g: Grid<12, 3, 4> = Grid::empty();
        let t = SimpleColoring;
        let _ = <SimpleColoring as Technique<12,3,4>>::apply(&t, &mut g);
    }

    #[test]
    fn coloring_smoke_16x16() {
        let mut g: Grid<16, 4, 4> = Grid::empty();
        let t = SimpleColoring;
        let _ = <SimpleColoring as Technique<16,4,4>>::apply(&t, &mut g);
    }
}
