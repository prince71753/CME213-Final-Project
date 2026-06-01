#!/usr/bin/env python3
"""Build report-facing bucket-size U-curve artifacts from a sweep summary."""

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"

PARAM_COUNTS = {
    128: 222592,
    256: 838400,
    512: 3249664,
}


def read_csv(path):
    if not path.exists():
        raise SystemExit(f"missing CSV: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_text(src, dst):
    dst.write_text(src.read_text())


def read_device_fit():
    for row in read_csv(RESULTS / "allreduce_alpha_beta_fit.csv"):
        if row["backend"] == "device" and int(row["ranks"]) == 4:
            return float(row["alpha_ms"]), float(row["beta_ms_per_byte"])
    raise SystemExit("missing 4-rank device alpha/beta fit")


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize(rows, hidden):
    alpha_ms, beta_ms_per_byte = read_device_fit()
    params = PARAM_COUNTS[hidden]
    grad_bytes = params * 4
    selected = [row for row in rows if int(row["hidden"]) == hidden]
    blocking_rows = [
        row for row in selected
        if row["backend"] == "direct" and row["effective_sync"] == "blocking"
    ]
    if not blocking_rows:
        raise SystemExit(f"missing h{hidden} direct blocking baseline")
    blocking = blocking_rows[0]
    blocking_tput = float(blocking["throughput_mean_mtok_s"])
    blocking_comm = float(blocking["avg_grad_sync_mean_ms"])

    out = []
    for row in selected:
        if row["effective_sync"] != "overlap":
            continue
        bucket_kb = int(row["bucket_kb"])
        bucket_bytes = bucket_kb * 1024
        modeled_buckets = max(1, math.ceil(grad_bytes / bucket_bytes))
        pred_ms = modeled_buckets * alpha_ms + beta_ms_per_byte * grad_bytes
        tput = float(row["throughput_mean_mtok_s"])
        exposed_wait = float(row["avg_grad_finish_mean_ms"])
        out.append({
            "hidden": hidden,
            "backend": row["backend"],
            "bucket_kb": bucket_kb,
            "valid_runs": int(row["valid_runs"]),
            "runs": int(row["runs"]),
            "throughput_mean_mtok_s": tput,
            "throughput_std_mtok_s": float(row["throughput_std_mtok_s"]),
            "time_mean_ms": float(row["time_mean_ms"]),
            "time_std_ms": float(row["time_std_ms"]),
            "exposed_wait_ms": exposed_wait,
            "speedup_vs_blocking": tput / blocking_tput if blocking_tput > 0.0 else "",
            "comm_tail_reduction_pct": 100.0 * (1.0 - exposed_wait / blocking_comm)
            if blocking_comm > 0.0 else "",
            "modeled_buckets": modeled_buckets,
            "alpha_beta_pred_comm_ms": pred_ms,
            "max_checksum_span": row["max_checksum_span"],
            "all_valid": row["all_valid"],
        })
    return sorted(out, key=lambda row: (row["backend"], row["bucket_kb"])), blocking


def svg_plot(path, rows, blocking):
    if not rows:
        return
    width, height = 860, 470
    left, right, top, bottom = 76, 170, 56, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    buckets = sorted({row["bucket_kb"] for row in rows})
    ymax = max(float(blocking["throughput_mean_mtok_s"]),
               max(row["throughput_mean_mtok_s"] + row["throughput_std_mtok_s"] for row in rows)) * 1.16
    colors = {
        "openmp_thread": "#2563eb",
        "pinned": "#f97316",
        "blocking": "#6b7280",
    }

    def x_pos(bucket):
        if len(buckets) == 1:
            return left + plot_w / 2
        idx = buckets.index(bucket)
        return left + idx * plot_w / (len(buckets) - 1)

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">h{rows[0]["hidden"]} Bucket-Size U-Curve</text>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="13">Bucket size (KB), categorical log sweep</text>',
        f'<text x="20" y="{top + plot_h/2}" transform="rotate(-90 20 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Throughput (M tok/s)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = y_pos(value)
        parts += [
            f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>',
        ]
    blocking_y = y_pos(float(blocking["throughput_mean_mtok_s"]))
    parts.append(f'<line x1="{left}" y1="{blocking_y:.1f}" x2="{left+plot_w}" y2="{blocking_y:.1f}" stroke="{colors["blocking"]}" stroke-dasharray="6 4" stroke-width="2"/>')

    for backend in ["pinned", "openmp_thread"]:
        series = [row for row in rows if row["backend"] == backend]
        if not series:
            continue
        pts = []
        for row in series:
            pts.append(f'{x_pos(row["bucket_kb"]):.1f},{y_pos(row["throughput_mean_mtok_s"]):.1f}')
        parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[backend]}" stroke-width="3"/>')
        for row in series:
            x = x_pos(row["bucket_kb"])
            y = y_pos(row["throughput_mean_mtok_s"])
            y_hi = y_pos(row["throughput_mean_mtok_s"] + row["throughput_std_mtok_s"])
            y_lo = y_pos(max(0.0, row["throughput_mean_mtok_s"] - row["throughput_std_mtok_s"]))
            parts += [
                f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{colors[backend]}" stroke-width="1.4"/>',
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colors[backend]}"/>',
            ]
    for bucket in buckets:
        x = x_pos(bucket)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+23}" text-anchor="middle" font-family="Arial" font-size="11">{bucket}</text>')

    lx = left + plot_w + 20
    legend = [
        ("openmp_thread", "OpenMP thread"),
        ("pinned", "pinned overlap"),
        ("blocking", "blocking baseline"),
    ]
    for idx, (key, label) in enumerate(legend):
        y = top + 38 + 24 * idx
        dash = ' stroke-dasharray="6 4"' if key == "blocking" else ""
        parts.append(f'<line x1="{lx}" y1="{y-5}" x2="{lx+18}" y2="{y-5}" stroke="{colors[key]}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{lx+26}" y="{y}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def write_md(path, tag, hidden, rows, blocking):
    all_valid = all(row["all_valid"] == "yes" for row in rows)
    best_openmp = max((row for row in rows if row["backend"] == "openmp_thread"),
                      key=lambda row: row["throughput_mean_mtok_s"], default=None)
    best_pinned = max((row for row in rows if row["backend"] == "pinned"),
                      key=lambda row: row["throughput_mean_mtok_s"], default=None)
    lines = [
        f"# h{hidden} Bucket-Size U-Curve: Job {tag}",
        "",
        f"Overall result: {'PASS' if all_valid else 'FAIL'}",
        "",
        "Configuration: 4 ranks, batch 32, 50 steps, 5 repeats with repeat 1",
        "dropped as warmup. Blocking is a CUDA-aware direct MPI baseline.",
        "Overlap rows compare host-pinned staging against the OpenMP",
        "communication-thread path.",
        "",
        f"Blocking baseline: {float(blocking['throughput_mean_mtok_s']):.3f} +/- {float(blocking['throughput_std_mtok_s']):.3f} M tok/s, "
        f"comm timer {float(blocking['avg_grad_sync_mean_ms']):.3f} ms.",
        "",
    ]
    if best_openmp:
        lines.append(
            f"Best OpenMP-thread bucket: {best_openmp['bucket_kb']} KB, "
            f"{best_openmp['throughput_mean_mtok_s']:.3f} +/- {best_openmp['throughput_std_mtok_s']:.3f} M tok/s, "
            f"{best_openmp['speedup_vs_blocking']:.3f}x vs blocking."
        )
    if best_pinned:
        lines.append(
            f"Best pinned-overlap bucket: {best_pinned['bucket_kb']} KB, "
            f"{best_pinned['throughput_mean_mtok_s']:.3f} +/- {best_pinned['throughput_std_mtok_s']:.3f} M tok/s, "
            f"{best_pinned['speedup_vs_blocking']:.3f}x vs blocking."
        )
    lines += [
        "",
        "| Backend | Bucket KB | M tok/s | Speedup vs blocking | Exposed wait ms | Tail reduction | Modeled buckets | Alpha/beta comm ms | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backend']} | {row['bucket_kb']} | "
            f"{row['throughput_mean_mtok_s']:.3f} +/- {row['throughput_std_mtok_s']:.3f} | "
            f"{row['speedup_vs_blocking']:.3f} | "
            f"{row['exposed_wait_ms']:.3f} | "
            f"{row['comm_tail_reduction_pct']:.1f}% | "
            f"{row['modeled_buckets']} | "
            f"{row['alpha_beta_pred_comm_ms']:.3f} | "
            f"{row['valid_runs']}/{row['runs']} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- The alpha/beta communication-only model penalizes small buckets through",
        "  the repeated latency term. It does not fully model delayed gradient",
        "  readiness for very large buckets, so measured throughput is the deciding",
        "  evidence for the bucket choice.",
        "- Use this result as the report's algorithmic-variant study for overlap",
        "  bucket size.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", help="training_bucket_sweep_summary CSV")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hidden", type=int, default=256)
    args = parser.parse_args()

    summary_path = Path(args.summary) if args.summary else RESULTS / f"training_bucket_sweep_summary_comm_thread_{args.tag}.csv"
    rows, blocking = summarize(read_csv(summary_path), args.hidden)
    if not rows:
        raise SystemExit(f"no h{args.hidden} overlap rows in {summary_path}")

    fields = [
        "hidden", "backend", "bucket_kb", "valid_runs", "runs",
        "throughput_mean_mtok_s", "throughput_std_mtok_s", "time_mean_ms",
        "time_std_ms", "exposed_wait_ms", "speedup_vs_blocking",
        "comm_tail_reduction_pct", "modeled_buckets",
        "alpha_beta_pred_comm_ms", "max_checksum_span", "all_valid",
    ]
    out_csv = RESULTS / f"bucket_ucurve_h{args.hidden}_{args.tag}.csv"
    out_md = RESULTS / f"bucket_ucurve_h{args.hidden}_{args.tag}.md"
    out_svg = PLOTS / f"bucket_ucurve_h{args.hidden}_{args.tag}.svg"
    write_csv(out_csv, rows, fields)
    write_md(out_md, args.tag, args.hidden, rows, blocking)
    svg_plot(out_svg, rows, blocking)
    copy_text(out_csv, RESULTS / f"bucket_ucurve_h{args.hidden}.csv")
    copy_text(out_md, RESULTS / f"bucket_ucurve_h{args.hidden}.md")
    copy_text(out_svg, PLOTS / f"bucket_ucurve_h{args.hidden}.svg")
    if not all(row["all_valid"] == "yes" for row in rows):
        raise SystemExit("one or more bucket U-curve rows had invalid runs")
    print(f"Wrote {out_csv.relative_to(ROOT)}")
    print(f"Wrote {out_md.relative_to(ROOT)}")
    print(f"Wrote {out_svg.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
