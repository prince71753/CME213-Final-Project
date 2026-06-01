#!/usr/bin/env python3
"""Collect null-variance benchmark summaries into one small CSV."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    patterns = [
        f"single_gpu_repeated_bench_summary_{args.tag}_null_*.csv",
        f"training_bucket_sweep_summary_{args.tag}_null_*.csv",
    ]

    rows = []
    for pattern in patterns:
        for path in sorted(RESULTS.glob(pattern)):
            for row in read_rows(path):
                rows.append({
                    "source": path.name,
                    "hidden": row.get("hidden", ""),
                    "backend": row.get("backend", ""),
                    "requested_sync_mode": row.get("requested_sync_mode", ""),
                    "effective_sync": row.get("effective_sync", ""),
                    "bucket_kb": row.get("bucket_kb", ""),
                    "requested_steps": row.get("requested_steps", ""),
                    "runs": row.get("runs", ""),
                    "valid_runs": row.get("valid_runs", ""),
                    "throughput_mean_mtok_s": row.get("throughput_mean_mtok_s", ""),
                    "throughput_median_mtok_s": row.get("throughput_median_mtok_s", ""),
                    "throughput_std_mtok_s": row.get("throughput_std_mtok_s", ""),
                    "throughput_cv_pct": row.get("throughput_cv_pct", ""),
                    "all_valid": row.get("all_valid", ""),
                })

    if not rows:
        raise SystemExit(f"no null variance summaries found for tag {args.tag}")

    out = RESULTS / f"null_variance_summary_{args.tag}.csv"
    fields = [
        "source", "hidden", "backend", "requested_sync_mode",
        "effective_sync", "bucket_kb", "requested_steps", "runs",
        "valid_runs", "throughput_mean_mtok_s", "throughput_median_mtok_s",
        "throughput_std_mtok_s", "throughput_cv_pct", "all_valid",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
