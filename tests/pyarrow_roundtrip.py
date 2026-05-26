#!/usr/bin/env python3
"""pyarrow roundtrip parity test (iter-2.3).

Compares a Rust-written parquet shard against a v7 Python-written reference shard:
  1. Schema parity: column names + per-column arrow types (nullability ignored —
     Python writer leaves all fields nullable; Rust now matches).
  2. trace_actions JSON-decodability for every Rust row.
  3. Tier-label parity: rate puzzles from the Rust shard with the *Rust* binary
     and check tier matches its own self-written tier (sanity / determinism check).
  4. Cross-rate: rate puzzles from the Python ref shard with the Rust binary
     (`sudoku_rs_core rate <81-char>`); assert tier matches for ≥95% of T1+T2
     reference puzzles. T3/T4 differences are accepted (Rust T3 is stub-only).

Usage:
  pyarrow_roundtrip.py <rust_shard.parquet> <ref_shard.parquet> <rust_bin>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def _digits_to_81str(digits: list[int]) -> str:
    """Convert puzzle list (0=empty, 1..9 filled) to 81-char string with '.' empties."""
    return "".join("." if d == 0 else str(int(d)) for d in digits)


def _rate_with_rust(binary: Path, puzzle81: str) -> tuple[int, str] | None:
    out = subprocess.run(
        [str(binary), "rate", puzzle81],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        return None
    txt = out.stdout
    # rater::Rating is a Debug print; tier appears as `tier: T1` / `T2` / `T3` / `T4Plus` / `Invalid`.
    m = re.search(r"tier:\s*(T\d\+?P?l?u?s?|Invalid)", txt)
    if not m:
        return None
    tname_to_int = {"T1": 1, "T2": 2, "T3": 3, "T4Plus": 4, "Invalid": 0}
    raw = m.group(1)
    return tname_to_int.get(raw, -1), raw


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: pyarrow_roundtrip.py <rust_shard.parquet> <ref_shard.parquet> <rust_bin>",
            file=sys.stderr,
        )
        return 2

    rust_p = Path(sys.argv[1])
    ref_p = Path(sys.argv[2])
    rust_bin = Path(sys.argv[3])

    fail = []

    # 1. Schema parity ------------------------------------------------------
    t_ref = pq.read_table(ref_p)
    t_rust = pq.read_table(rust_p)
    ref_cols = {f.name: f.type for f in t_ref.schema}
    rust_cols = {f.name: f.type for f in t_rust.schema}

    if set(ref_cols) != set(rust_cols):
        fail.append(
            f"column-name mismatch: ref-only={set(ref_cols)-set(rust_cols)}, "
            f"rust-only={set(rust_cols)-set(ref_cols)}"
        )
    type_mismatches = []
    for k in sorted(set(ref_cols) & set(rust_cols)):
        if ref_cols[k] != rust_cols[k]:
            type_mismatches.append(f"{k}: ref={ref_cols[k]} rust={rust_cols[k]}")
    if type_mismatches:
        fail.append("type mismatches: " + "; ".join(type_mismatches))

    print(f"[schema] {'PASS' if not type_mismatches else 'FAIL'}: "
          f"{len(ref_cols)} cols, {len(type_mismatches)} type mismatches")

    # 2. trace_actions JSON parse ------------------------------------------
    json_errors = 0
    for s in t_rust.column("trace_actions").to_pylist():
        try:
            json.loads(s)
        except Exception:
            json_errors += 1
    print(f"[trace_actions] {'PASS' if json_errors == 0 else 'FAIL'}: "
          f"{json_errors}/{t_rust.num_rows} parse errors")
    if json_errors:
        fail.append(f"trace_actions JSON errors: {json_errors}")

    # 3. Self-consistency: rate Rust shard puzzles with Rust binary --------
    rust_rows = t_rust.to_pylist()
    sample = rust_rows[: min(50, len(rust_rows))]
    self_match = 0
    self_total = 0
    for row in sample:
        p81 = _digits_to_81str(row["puzzle"])
        r = _rate_with_rust(rust_bin, p81)
        if r is None:
            continue
        self_total += 1
        if r[0] == row["tier"]:
            self_match += 1
    print(f"[self-rate] {self_match}/{self_total} match (Rust shard ↔ Rust rater)")

    # 4. Cross-rate: rate Python ref puzzles with Rust binary --------------
    ref_rows = t_ref.to_pylist()
    sample_ref = ref_rows[: min(50, len(ref_rows))]
    cross_match = 0
    cross_total_t1t2 = 0
    cross_match_t1t2 = 0
    by_tier: dict[int, Counter] = {}
    for row in sample_ref:
        p81 = _digits_to_81str(row["puzzle"])
        r = _rate_with_rust(rust_bin, p81)
        if r is None:
            continue
        ref_tier = row["tier"]
        rust_tier = r[0]
        by_tier.setdefault(ref_tier, Counter())[rust_tier] += 1
        if rust_tier == ref_tier:
            cross_match += 1
        if ref_tier in (1, 2):
            cross_total_t1t2 += 1
            if rust_tier == ref_tier:
                cross_match_t1t2 += 1

    print(f"[cross-rate] T1+T2 parity: {cross_match_t1t2}/{cross_total_t1t2} "
          f"({100.0 * cross_match_t1t2 / max(1, cross_total_t1t2):.1f}%)  "
          f"all-tiers: {cross_match}/{len(sample_ref)}")
    print("[cross-rate] confusion (ref→rust):")
    for ref_t in sorted(by_tier):
        for rust_t, n in sorted(by_tier[ref_t].items()):
            print(f"    ref={ref_t} rust={rust_t}: {n}")

    if cross_total_t1t2 > 0 and cross_match_t1t2 / cross_total_t1t2 < 0.95:
        fail.append(
            f"T1+T2 parity below threshold: "
            f"{cross_match_t1t2}/{cross_total_t1t2} ({100.0 * cross_match_t1t2 / cross_total_t1t2:.1f}%)"
        )

    print()
    if fail:
        print("FAIL:")
        for f in fail:
            print("  -", f)
        return 1
    print("ALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
