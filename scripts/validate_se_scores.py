#!/usr/bin/env python3
"""validate_se_scores.py — histogram and summary stats for se_score column.

Usage:
    python scripts/validate_se_scores.py <path-to-parquet> [<path2> ...]

Input: parquet file(s) with at least `puzzle` and `se_score` columns.
Output: printed summary stats + ASCII histogram of se_score distribution.

Requires: pyarrow (no other external deps).
Memory-efficient: streams row groups, no full materialisation.
"""
import sys
import math
import collections
from pathlib import Path


def _quantile(values_sorted, q):
    """Linear-interpolation quantile on a sorted list."""
    if not values_sorted:
        return float("nan")
    n = len(values_sorted)
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac


def process_file(path: Path, scores: list):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg, columns=["se_score"])
        col = table.column("se_score")
        for chunk in col.chunks:
            for val in chunk:
                if val.is_valid:
                    scores.append(val.as_py())
    return scores


def ascii_histogram(values, n_bins=20, width=60):
    if not values:
        print("  (no data)")
        return
    lo = min(values)
    hi = max(values)
    if lo == hi:
        print(f"  all values = {lo:.3f}")
        return
    bin_width = (hi - lo) / n_bins
    counts = collections.Counter()
    for v in values:
        b = min(int((v - lo) / bin_width), n_bins - 1)
        counts[b] += 1
    max_count = max(counts.values()) if counts else 1
    print(f"\n  se_score histogram [{lo:.2f} – {hi:.2f}]  (bin width={bin_width:.3f})")
    print(f"  {'bin range':<22}  {'count':>7}  bar")
    print("  " + "-" * (width + 32))
    for b in range(n_bins):
        cnt = counts.get(b, 0)
        bar_len = int(cnt / max_count * width) if max_count else 0
        lo_b = lo + b * bin_width
        hi_b = lo_b + bin_width
        bar = "█" * bar_len
        print(f"  [{lo_b:6.2f} – {hi_b:6.2f})  {cnt:>7}  {bar}")


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_se_scores.py <parquet> [<parquet> ...]", file=sys.stderr)
        sys.exit(1)

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("ERROR: pyarrow not installed. pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    scores = []
    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"WARNING: {path} does not exist, skipping", file=sys.stderr)
            continue
        print(f"Reading {path} …")
        try:
            process_file(path, scores)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    if not scores:
        print("No se_score values found.", file=sys.stderr)
        sys.exit(1)

    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    total = sum(scores_sorted)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in scores_sorted) / n
    stddev = math.sqrt(variance)

    print(f"\n=== se_score summary ({n} puzzles) ===")
    print(f"  count : {n}")
    print(f"  min   : {scores_sorted[0]:.4f}")
    print(f"  p5    : {_quantile(scores_sorted, 0.05):.4f}")
    print(f"  p25   : {_quantile(scores_sorted, 0.25):.4f}")
    print(f"  median: {_quantile(scores_sorted, 0.50):.4f}")
    print(f"  mean  : {mean:.4f}")
    print(f"  stddev: {stddev:.4f}")
    print(f"  p75   : {_quantile(scores_sorted, 0.75):.4f}")
    print(f"  p95   : {_quantile(scores_sorted, 0.95):.4f}")
    print(f"  max   : {scores_sorted[-1]:.4f}")

    # Tier breakdown by SE bucket
    buckets = {"0.0": 0, "2.6–3.1": 0, "3.2–3.9": 0, "4.0–4.9": 0,
               "5.0–5.9": 0, "6.0–6.9": 0, "7.0+": 0}
    for v in scores_sorted:
        if v == 0.0:
            buckets["0.0"] += 1
        elif v < 3.2:
            buckets["2.6–3.1"] += 1
        elif v < 4.0:
            buckets["3.2–3.9"] += 1
        elif v < 5.0:
            buckets["4.0–4.9"] += 1
        elif v < 6.0:
            buckets["5.0–5.9"] += 1
        elif v < 7.0:
            buckets["6.0–6.9"] += 1
        else:
            buckets["7.0+"] += 1

    print("\n  SE bucket distribution:")
    for label, cnt in buckets.items():
        pct = 100 * cnt / n if n else 0
        print(f"    {label:<10}: {cnt:>6}  ({pct:5.1f}%)")

    ascii_histogram(scores_sorted)
    print()


if __name__ == "__main__":
    main()
