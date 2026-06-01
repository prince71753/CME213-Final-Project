#!/usr/bin/env python3
"""Derive preliminary communication overlap tables from repeated-run CSVs."""

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


INPUTS = [
    RESULTS / "training_bucket_sweep_h128_mpi_backend_final_clean_h128_mpi.csv",
    RESULTS / "training_bucket_sweep_comm_thread_final_clean_h256_comm.csv",
    RESULTS / "training_bucket_sweep_comm_thread_final_clean_h512_comm_buckets.csv",
]


BEST_CASES = {
    128: ("custom", "overlap", 256),
    256: ("openmp_thread", "overlap", 1024),
    512: ("openmp_thread", "overlap", 2048),
}


def to_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def sample_std(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def read_rows():
    rows = []
    for path in INPUTS:
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                if row.get("valid") != "yes":
                    continue
                row["source"] = path.name
                rows.append(row)
    return rows


def summarize_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            int(row["hidden"]), row["backend"], row["requested_sync_mode"],
            row["effective_sync"], int(row["bucket_kb"]), int(row["requested_steps"]),
        )
        groups[key].append(row)
    summary = {}
    for key, group in groups.items():
        steps = key[-1]
        summary[key] = {
            "runs": len(group),
            "throughput_mean_mtok_s": mean([to_float(r["throughput_mtok_s"]) for r in group]),
            "throughput_std_mtok_s": sample_std([to_float(r["throughput_mtok_s"]) for r in group]),
            "step_time_mean_ms": mean([to_float(r["time_ms"]) / steps for r in group]),
            "step_time_std_ms": sample_std([to_float(r["time_ms"]) / steps for r in group]),
            "avg_grad_sync_mean_ms": mean([to_float(r["avg_grad_sync_ms"]) for r in group]),
            "avg_grad_start_mean_ms": mean([to_float(r["avg_grad_start_ms"]) for r in group]),
            "avg_grad_finish_mean_ms": mean([to_float(r["avg_grad_finish_ms"]) for r in group]),
        }
    return summary


def build_breakdown(summary):
    rows = []
    for hidden, (backend, mode, bucket) in BEST_CASES.items():
        case_key = next((k for k in summary if k[0] == hidden and k[1] == backend
                         and k[2] == mode and k[4] == bucket), None)
        block_key = next((k for k in summary if k[0] == hidden and k[2] == "blocking"
                          and k[4] == 0), None)
        if not case_key:
            continue
        case = summary[case_key]
        block = summary.get(block_key, {})
        blocking_sync = block.get("avg_grad_sync_mean_ms", 0.0)
        exposed_wait = case["avg_grad_finish_mean_ms"] if mode == "overlap" else case["avg_grad_sync_mean_ms"]
        enqueue_start = case["avg_grad_start_mean_ms"]
        non_exposed = max(0.0, case["step_time_mean_ms"] - exposed_wait)
        overlap_eff = ""
        if blocking_sync > 0 and exposed_wait > 0:
            overlap_eff = 1.0 - exposed_wait / blocking_sync
        rows.append({
            "hidden": hidden,
            "selected_backend": backend,
            "selected_mode": mode,
            "bucket_kb": bucket,
            "runs": case["runs"],
            "throughput_mean_mtok_s": case["throughput_mean_mtok_s"],
            "throughput_std_mtok_s": case["throughput_std_mtok_s"],
            "step_time_mean_ms": case["step_time_mean_ms"],
            "step_time_std_ms": case["step_time_std_ms"],
            "blocking_comm_proxy_ms": blocking_sync,
            "enqueue_or_start_ms": enqueue_start,
            "exposed_wait_ms": exposed_wait,
            "non_exposed_step_ms": non_exposed,
            "exposed_wait_fraction": exposed_wait / case["step_time_mean_ms"]
            if case["step_time_mean_ms"] > 0 else "",
            "overlap_efficiency_vs_blocking_proxy": overlap_eff,
        })
    return rows


def build_overlap_rows(summary):
    rows = []
    for key, case in sorted(summary.items()):
        hidden, backend, requested, effective, bucket, steps = key
        if requested != "overlap":
            continue
        block_key = next((k for k in summary if k[0] == hidden and k[2] == "blocking"
                          and k[4] == 0), None)
        if not block_key:
            continue
        total_comm = summary[block_key]["avg_grad_sync_mean_ms"]
        exposed = case["avg_grad_finish_mean_ms"]
        if total_comm <= 0 or exposed <= 0:
            continue
        rows.append({
            "hidden": hidden,
            "backend": backend,
            "bucket_kb": bucket,
            "runs": case["runs"],
            "blocking_comm_proxy_ms": total_comm,
            "enqueue_or_start_ms": case["avg_grad_start_mean_ms"],
            "exposed_wait_ms": exposed,
            "overlap_efficiency": 1.0 - exposed / total_comm,
            "throughput_mean_mtok_s": case["throughput_mean_mtok_s"],
            "throughput_std_mtok_s": case["throughput_std_mtok_s"],
        })
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_breakdown(path, rows):
    if not rows:
        return
    width, height = 780, 450
    left, right, top, bottom = 86, 150, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(r["step_time_mean_ms"] for r in rows) * 1.14
    bar_w = min(90, plot_w / (len(rows) * 2.2))

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Step Time: Exposed Communication</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">Selected best overlap case by hidden size</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">ms per step</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>')
    for idx, row in enumerate(rows):
        x = left + (idx + 0.55) * plot_w / len(rows)
        y0 = y_pos(row["non_exposed_step_ms"])
        y1 = y_pos(row["step_time_mean_ms"])
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{y0:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y0:.1f}" fill="#93c5fd"/>')
        parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{y1:.1f}" width="{bar_w:.1f}" height="{y0-y1:.1f}" fill="#f97316"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">h{row["hidden"]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y1-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{row["exposed_wait_fraction"]*100:.0f}%</text>')
    lx = left + plot_w + 18
    parts.append(f'<rect x="{lx}" y="{top+34}" width="14" height="14" fill="#93c5fd"/>')
    parts.append(f'<text x="{lx+22}" y="{top+46}" font-family="Arial" font-size="12">non-exposed step</text>')
    parts.append(f'<rect x="{lx}" y="{top+58}" width="14" height="14" fill="#f97316"/>')
    parts.append(f'<text x="{lx+22}" y="{top+70}" font-family="Arial" font-size="12">exposed wait</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def main():
    rows = read_rows()
    if not rows:
        raise SystemExit("no input rows found")
    summary = summarize_groups(rows)
    breakdown = build_breakdown(summary)
    overlap = build_overlap_rows(summary)
    breakdown_fields = [
        "hidden", "selected_backend", "selected_mode", "bucket_kb", "runs",
        "throughput_mean_mtok_s", "throughput_std_mtok_s", "step_time_mean_ms",
        "step_time_std_ms", "blocking_comm_proxy_ms", "enqueue_or_start_ms",
        "exposed_wait_ms", "non_exposed_step_ms", "exposed_wait_fraction",
        "overlap_efficiency_vs_blocking_proxy",
    ]
    overlap_fields = [
        "hidden", "backend", "bucket_kb", "runs", "blocking_comm_proxy_ms",
        "enqueue_or_start_ms", "exposed_wait_ms", "overlap_efficiency",
        "throughput_mean_mtok_s", "throughput_std_mtok_s",
    ]
    write_csv(RESULTS / "comm_breakdown_preliminary.csv", breakdown, breakdown_fields)
    write_csv(RESULTS / "overlap_efficiency_preliminary.csv", overlap, overlap_fields)
    svg_breakdown(PLOTS / "comm_breakdown_preliminary.svg", breakdown)
    print("Wrote results/comm_breakdown_preliminary.csv")
    print("Wrote results/overlap_efficiency_preliminary.csv")
    print("Wrote plots/comm_breakdown_preliminary.svg")


if __name__ == "__main__":
    main()
