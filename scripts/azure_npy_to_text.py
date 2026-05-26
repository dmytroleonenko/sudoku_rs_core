#!/usr/bin/env python3
"""Convert Azure HRM-style sudoku npy bundles to plain-text puzzle files.

Azure dataset.json layout (per the HRM-sudoku 16×16 release):
  - all__inputs_<N>.npy  : shape (M, N*N) uint8.
                           Encoding: 1 = blank cell, 2..(N+1) = digit 1..N.
                           (Offset by +1 because PAD=0 is reserved.)
  - all__labels_<N>.npy  : shape (M, N*N) uint8, full solution; same
                           offset-by-1 encoding (no blanks).
  - all__group_indices.npy / all__puzzle_indices.npy / all__clue_count_<N>.npy
    / all__tier_<N>.npy  : metadata; we ignore them here.

Output: one puzzle per line, N*N chars, '.' = blank, '1'..'9'/'A'.. for digits
1..N. Compatible with `sudoku_rs_core rerate --size NxBRxBC --input-glob`.

Usage:
  python azure_npy_to_text.py --input-dir SHARD_DIR --output FILE.txt --size 16
                              [--limit M] [--include-solutions]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def encode_row(row: np.ndarray, n: int) -> str:
    # Azure encoding: 1 = blank, 2..(N+1) = digit 1..N.
    out = []
    for v in row:
        v = int(v)
        if v <= 1:
            out.append(".")
            continue
        d = v - 1
        if not (1 <= d <= n):
            raise ValueError(
                f"unexpected cell value {v} (n={n}); expected 1..{n + 1}"
            )
        if d <= 9:
            out.append(str(d))
        else:
            out.append(chr(ord("A") + d - 10))
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--size", required=True, type=int, help="N (e.g. 16)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument(
        "--include-solutions",
        action="store_true",
        help="If set, emit two-line format (puzzle then solution).",
    )
    args = ap.parse_args()

    n = args.size
    nn = n * n
    inputs_path = args.input_dir / f"all__inputs_{n}.npy"
    labels_path = args.input_dir / f"all__labels_{n}.npy"
    if not inputs_path.exists():
        # Fall back to the un-suffixed names some Azure releases use.
        inputs_path = args.input_dir / "all__inputs.npy"
        labels_path = args.input_dir / "all__labels.npy"
    if not inputs_path.exists():
        print(f"error: not found: {inputs_path}", file=sys.stderr)
        return 2

    inputs = np.load(inputs_path)
    if inputs.ndim != 2 or inputs.shape[1] != nn:
        print(
            f"error: inputs shape {inputs.shape} does not match (M, {nn})",
            file=sys.stderr,
        )
        return 2
    if args.include_solutions:
        labels = np.load(labels_path)
        if labels.shape != inputs.shape:
            print(
                f"error: labels shape {labels.shape} != inputs {inputs.shape}",
                file=sys.stderr,
            )
            return 2

    m = inputs.shape[0]
    if args.limit and args.limit < m:
        m = args.limit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w") as f:
        for i in range(m):
            puz = encode_row(inputs[i], n)
            f.write(puz + "\n")
            if args.include_solutions:
                sol = encode_row(labels[i], n)
                f.write(sol + "\n")
            written += 1
    print(
        f"wrote {written} puzzle line(s) to {args.output} (size={n}x{n})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
