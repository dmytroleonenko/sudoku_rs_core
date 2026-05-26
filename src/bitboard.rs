//! Bitboard helpers and precomputed peer/unit masks.
//!
//! A "unit" is a row, column, or box (27 total). Each cell belongs to 3 units.
//! Each cell has 20 peers (other cells sharing at least one unit).
//!
//! Cell candidates are stored as a u16 with bits 0..=8 set when digit (1..=9)
//! is still possible. Digit `d` (1-based) corresponds to bit `d-1`.

pub const N: usize = 9;
pub const NCELLS: usize = 81;
pub const NUNITS: usize = 27;
pub const ALL_DIGITS: u16 = 0x1FF; // bits 0..8

#[inline(always)]
pub fn rcb(cell: usize) -> (usize, usize, usize) {
    let r = cell / 9;
    let c = cell % 9;
    let b = (r / 3) * 3 + (c / 3);
    (r, c, b)
}

/// 27 unit-cell lists: rows 0..9, cols 9..18, boxes 18..27.
pub const fn units() -> [[u8; 9]; 27] {
    let mut u = [[0u8; 9]; 27];
    let mut r = 0;
    while r < 9 {
        let mut c = 0;
        while c < 9 {
            u[r][c] = (r * 9 + c) as u8;
            c += 1;
        }
        r += 1;
    }
    let mut c = 0;
    while c < 9 {
        let mut r = 0;
        while r < 9 {
            u[9 + c][r] = (r * 9 + c) as u8;
            r += 1;
        }
        c += 1;
    }
    let mut b = 0;
    while b < 9 {
        let br = (b / 3) * 3;
        let bc = (b % 3) * 3;
        let mut k = 0;
        let mut i = 0;
        while i < 3 {
            let mut j = 0;
            while j < 3 {
                u[18 + b][k] = ((br + i) * 9 + (bc + j)) as u8;
                k += 1;
                j += 1;
            }
            i += 1;
        }
        b += 1;
    }
    u
}

/// `cell_units[cell]` = (row_unit_id, col_unit_id, box_unit_id).
pub const fn cell_units() -> [[u8; 3]; 81] {
    let mut t = [[0u8; 3]; 81];
    let mut i = 0;
    while i < 81 {
        let r = i / 9;
        let c = i % 9;
        let b = (r / 3) * 3 + (c / 3);
        t[i] = [r as u8, (9 + c) as u8, (18 + b) as u8];
        i += 1;
    }
    t
}

/// 81×128-bit peer mask: bit i set if cell i is a peer of cell c (excludes c itself).
/// Stored as two u64s (low: 0..64, high: 64..81).
pub const fn peers_u64x2() -> [[u64; 2]; 81] {
    let units_arr = units();
    let mut out = [[0u64; 2]; 81];
    let mut c = 0;
    while c < 81 {
        let r = c / 9;
        let col = c % 9;
        let bx = (r / 3) * 3 + (col / 3);
        let mut mask = [0u64; 2];
        let mut k = 0;
        while k < 9 {
            // row peers
            let p = units_arr[r][k] as usize;
            if p != c {
                if p < 64 {
                    mask[0] |= 1u64 << p;
                } else {
                    mask[1] |= 1u64 << (p - 64);
                }
            }
            // col peers
            let p2 = units_arr[9 + col][k] as usize;
            if p2 != c {
                if p2 < 64 {
                    mask[0] |= 1u64 << p2;
                } else {
                    mask[1] |= 1u64 << (p2 - 64);
                }
            }
            // box peers
            let p3 = units_arr[18 + bx][k] as usize;
            if p3 != c {
                if p3 < 64 {
                    mask[0] |= 1u64 << p3;
                } else {
                    mask[1] |= 1u64 << (p3 - 64);
                }
            }
            k += 1;
        }
        out[c] = mask;
        c += 1;
    }
    out
}

#[inline(always)]
pub fn popcount9(x: u16) -> u32 {
    (x & ALL_DIGITS).count_ones()
}

/// Returns the digit (1..=9) of a single-bit u16 mask, or 0 if not single.
#[inline(always)]
pub fn single_digit(mask: u16) -> u8 {
    if mask.count_ones() == 1 {
        (mask.trailing_zeros() + 1) as u8
    } else {
        0
    }
}

/// Iterate digits 1..=9 set in mask.
#[inline(always)]
pub fn iter_bits(mask: u16) -> impl Iterator<Item = u8> {
    let mut m = mask & ALL_DIGITS;
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let b = m.trailing_zeros() as u8;
            m &= m - 1;
            Some(b + 1)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peers_count() {
        let p = peers_u64x2();
        for c in 0..81 {
            let cnt = p[c][0].count_ones() + p[c][1].count_ones();
            assert_eq!(cnt, 20, "cell {} peer count", c);
        }
    }

    #[test]
    fn units_cover() {
        let u = units();
        // Each cell appears in exactly 3 units.
        for cell in 0..81u8 {
            let mut hits = 0;
            for k in 0..27 {
                if u[k].contains(&cell) {
                    hits += 1;
                }
            }
            assert_eq!(hits, 3);
        }
    }
}
