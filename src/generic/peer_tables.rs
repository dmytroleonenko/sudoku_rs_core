//! Peer / unit tables for arbitrary `(N, BR, BC)` sudoku sizes.
//!
//! For each cell we store its peer-set (other cells sharing a row, column, or
//! box). Peer sets are built once per `(N, BR, BC)` triple and cached behind a
//! global `OnceLock`-keyed map.
//!
//! Note: we use a `Mutex<HashMap>` over a sync map for the cache because a
//! handful of concurrent first-touches per process is the worst case here, and
//! the lock is held only for the lookup/insert. Once present, lookups are
//! cloning the `Arc` reference.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::sync::Arc;

/// Per-cell peer information.
#[derive(Clone, Debug)]
pub struct CellPeers {
    /// Indices of all peer cells (no duplicates, excludes the cell itself).
    /// Length = 3*N - 2*BR - 2*BC + BR*BC - 1 (for N=BR*BC):
    ///   row peers (N-1) + col peers (N-1) + box-only peers ((BR-1)*(BC-1)).
    /// Concretely: 6→11, 9→20, 12→25, 16→39.
    pub peers: Vec<u16>,
    /// Row index of this cell.
    pub row: u16,
    /// Column index of this cell.
    pub col: u16,
    /// Box index of this cell (0 .. N-1).
    pub box_id: u16,
}

/// Tables for one `(N, BR, BC)` instantiation.
#[derive(Clone, Debug)]
pub struct PeerTable {
    pub n: usize,
    pub br: usize,
    pub bc: usize,
    /// Per-cell peer info, length = N*N.
    pub cells: Vec<CellPeers>,
    /// `units[u]` = list of cell indices in unit `u`.
    /// Order: rows 0..N, cols N..2N, boxes 2N..3N. Each row/col/box has length N.
    pub units: Vec<Vec<u16>>,
    /// `cell_units[cell]` = (row_unit_id, col_unit_id, box_unit_id).
    pub cell_units: Vec<[u16; 3]>,
}

impl PeerTable {
    fn build(n: usize, br: usize, bc: usize) -> Self {
        debug_assert_eq!(br * bc, n, "BR*BC must equal N");
        let nn = n * n;
        let n_units = 3 * n;
        let mut units: Vec<Vec<u16>> = vec![Vec::with_capacity(n); n_units];

        // Rows: unit r (0..N) holds cells r*N .. r*N+N
        for r in 0..n {
            for c in 0..n {
                units[r].push((r * n + c) as u16);
            }
        }
        // Columns: unit N+c holds cells c, c+N, c+2N, ...
        for c in 0..n {
            for r in 0..n {
                units[n + c].push((r * n + c) as u16);
            }
        }
        // Boxes: a box has BR rows and BC cols. There are BC box-cols and BR box-rows
        //   so we lay out box id = (br_idx * BC) + bc_idx, where 0 ≤ br_idx < BC
        //   (number of box-rows = N/BR = BC) and 0 ≤ bc_idx < BR (number of
        //   box-cols = N/BC = BR). Wait — easier: box id = (r/BR)*BR_cols + (c/BC)
        //   where BR_cols = N/BC = BR. So box id has range 0..N (since
        //   (N/BR)*(N/BC) = BC*BR = N).
        let n_box_cols = n / bc; // = BR
        for r in 0..n {
            for c in 0..n {
                let b = (r / br) * n_box_cols + (c / bc);
                units[2 * n + b].push((r * n + c) as u16);
            }
        }

        let mut cell_units = vec![[0u16; 3]; nn];
        for cell in 0..nn {
            let r = cell / n;
            let c = cell % n;
            let b = (r / br) * n_box_cols + (c / bc);
            cell_units[cell] = [r as u16, (n + c) as u16, (2 * n + b) as u16];
        }

        // Build peer sets.
        let mut cells: Vec<CellPeers> = Vec::with_capacity(nn);
        for cell in 0..nn {
            let r = cell / n;
            let c = cell % n;
            let b = (r / br) * n_box_cols + (c / bc);
            // Use a bitset to dedup peers.
            let mut seen = vec![false; nn];
            seen[cell] = true; // exclude self
            let mut peers: Vec<u16> = Vec::new();
            // Row peers
            for cc in 0..n {
                let p = r * n + cc;
                if !seen[p] {
                    seen[p] = true;
                    peers.push(p as u16);
                }
            }
            // Col peers
            for rr in 0..n {
                let p = rr * n + c;
                if !seen[p] {
                    seen[p] = true;
                    peers.push(p as u16);
                }
            }
            // Box peers
            let br_start = (r / br) * br;
            let bc_start = (c / bc) * bc;
            for rr in br_start..br_start + br {
                for cc in bc_start..bc_start + bc {
                    let p = rr * n + cc;
                    if !seen[p] {
                        seen[p] = true;
                        peers.push(p as u16);
                    }
                }
            }
            cells.push(CellPeers {
                peers,
                row: r as u16,
                col: c as u16,
                box_id: b as u16,
            });
        }

        PeerTable {
            n,
            br,
            bc,
            cells,
            units,
            cell_units,
        }
    }
}

type CacheKey = (usize, usize, usize);
fn cache() -> &'static Mutex<HashMap<CacheKey, Arc<PeerTable>>> {
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<PeerTable>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get the cached peer table for `(N, BR, BC)`. Builds on first call.
pub fn get(n: usize, br: usize, bc: usize) -> Arc<PeerTable> {
    debug_assert_eq!(br * bc, n, "BR*BC must equal N");
    let key = (n, br, bc);
    let mut guard = cache().lock().expect("peer-table cache mutex poisoned");
    if let Some(t) = guard.get(&key) {
        return t.clone();
    }
    let t = Arc::new(PeerTable::build(n, br, bc));
    guard.insert(key, t.clone());
    t
}

/// Const-generic accessor: returns the peer table for the given `<N,BR,BC>`.
#[inline]
pub fn get_for<const N: usize, const BR: usize, const BC: usize>() -> Arc<PeerTable> {
    get(N, BR, BC)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_count_9x9() {
        let t = get(9, 3, 3);
        assert_eq!(t.cells.len(), 81);
        assert_eq!(t.cells[0].peers.len(), 20);
        // sanity: every cell has 20 peers
        for c in 0..81 {
            assert_eq!(t.cells[c].peers.len(), 20, "cell {} should have 20 peers", c);
        }
    }

    #[test]
    fn peer_count_16x16() {
        let t = get(16, 4, 4);
        assert_eq!(t.cells.len(), 256);
        assert_eq!(t.cells[0].peers.len(), 39);
        for c in 0..256 {
            assert_eq!(t.cells[c].peers.len(), 39);
        }
    }

    #[test]
    fn peer_count_6x6() {
        // Per-cell peers = 3N - BR - BC - 1 = 18 - 2 - 3 - 1 = 12.
        let t = get(6, 2, 3);
        assert_eq!(t.cells.len(), 36);
        assert_eq!(t.cells[0].peers.len(), 12);
        for c in 0..36 {
            assert_eq!(t.cells[c].peers.len(), 12);
        }
    }

    #[test]
    fn peer_count_12x12() {
        // Per-cell peers = 3N - BR - BC - 1 = 36 - 3 - 4 - 1 = 28.
        let t = get(12, 3, 4);
        assert_eq!(t.cells.len(), 144);
        assert_eq!(t.cells[0].peers.len(), 28);
        for c in 0..144 {
            assert_eq!(t.cells[c].peers.len(), 28);
        }
    }

    #[test]
    fn cache_returns_same_arc() {
        let a = get(9, 3, 3);
        let b = get(9, 3, 3);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn box_count_each_size() {
        // Number of box-units == N for all our sizes.
        for &(n, br, bc) in &[(6usize, 2usize, 3usize), (9, 3, 3), (12, 3, 4), (16, 4, 4)] {
            let t = get(n, br, bc);
            // boxes are units[2N .. 3N], each must have N cells
            for b in 0..n {
                assert_eq!(t.units[2 * n + b].len(), n);
            }
            // every cell appears in exactly 3 units
            let mut counts = vec![0usize; n * n];
            for u in &t.units {
                for &c in u {
                    counts[c as usize] += 1;
                }
            }
            for c in 0..n * n {
                assert_eq!(counts[c], 3, "cell {} of size {}", c, n);
            }
        }
    }
}
