#!/usr/bin/env python3
"""Summarize MPI-device vs NCCL Allreduce microbenchmark output."""

import argparse
import csv
import math
from pathlib import Path

import summarize_allreduce_alpha_beta as alpha_beta


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def parse_valid(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("allreduce_alpha_beta_valid,"):
            continue
        parts = line.split(",")
        if len(parts) != 5:
            continue
        rows.append({
            "backend": parts[1],
            "ranks": int(parts[2]),
            "bytes": int(parts[3]),
            "valid": parts[4],
        })
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_comparison(summary):
    by_key = {
        (row["ranks"], row["bytes"], row["backend"]): row
        for row in summary
    }
    out = []
    for ranks, bytes_ in sorted({(row["ranks"], row["bytes"]) for row in summary}):
        mpi = by_key.get((ranks, bytes_, "device"))
        nccl = by_key.get((ranks, bytes_, "nccl"))
        if not mpi or not nccl:
            continue
        mpi_ms = mpi["time_mean_ms"]
        nccl_ms = nccl["time_mean_ms"]
        out.append({
            "ranks": ranks,
            "bytes": bytes_,
            "message_mb": bytes_ / (1024.0 * 1024.0),
            "mpi_device_time_mean_ms": mpi_ms,
            "mpi_device_time_std_ms": mpi["time_std_ms"],
            "nccl_time_mean_ms": nccl_ms,
            "nccl_time_std_ms": nccl["time_std_ms"],
            "nccl_speedup_vs_mpi_device": mpi_ms / nccl_ms if nccl_ms > 0 else 0.0,
            "mpi_device_payload_gb_s": mpi["effective_payload_gb_s"],
            "nccl_payload_gb_s": nccl["effective_payload_gb_s"],
        })
    return out


def svg_plot(path, comparison):
    if not comparison:
        return
    width, height = 780, 440
    left, right, top, bottom = 90, 165, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [math.log2(row["bytes"]) for row in comparison]
    ys = [row["nccl_speedup_vs_mpi_device"] for row in comparison]
    xmin, xmax = min(xs), max(xs)
    ymin = min(0.0, min(ys) * 0.95)
    ymax = max(1.05, max(ys) * 1.10)
    colors = {2: "#2563eb", 4: "#dc2626"}

    def x_pos(bytes_):
        if xmin == xmax:
            return left + plot_w / 2
        return left + (math.log2(bytes_) - xmin) / (xmax - xmin) * plot_w

    def y_pos(value):
        return top + plot_h - (value - ymin) / (ymax - ymin) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">NCCL vs MPI Device Allreduce</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">Message size</text>',
        f'<text x="20" y="{top + plot_h/2}" transform="rotate(-90 20 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">NCCL speedup vs MPI-device</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    one_y = y_pos(1.0)
    parts.append(f'<line x1="{left}" y1="{one_y:.1f}" x2="{left+plot_w}" y2="{one_y:.1f}" stroke="#6b7280" stroke-dasharray="4 4"/>')
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for bytes_ in sorted({row["bytes"] for row in comparison}):
        x = x_pos(bytes_)
        label = f'{bytes_ // 1024}KB' if bytes_ < 1024 * 1024 else f'{bytes_ // (1024 * 1024)}MB'
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    for idx, ranks in enumerate(sorted({row["ranks"] for row in comparison})):
        color = colors.get(ranks, "#16a34a")
        series = sorted([row for row in comparison if row["ranks"] == ranks],
                        key=lambda row: row["bytes"])
        coords = " ".join(
            f'{x_pos(row["bytes"]):.1f},{y_pos(row["nccl_speedup_vs_mpi_device"]):.1f}'
            for row in series
        )
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in series:
            parts.append(f'<circle cx="{x_pos(row["bytes"]):.1f}" cy="{y_pos(row["nccl_speedup_vs_mpi_device"]):.1f}" r="3.5" fill="{color}"/>')
        ly = top + 42 + idx * 24
        parts.append(f'<line x1="{left+plot_w+22}" y1="{ly-4}" x2="{left+plot_w+48}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left+plot_w+56}" y="{ly}" font-family="Arial" font-size="12">{ranks} ranks</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def fmt_ms(value):
    return f"{value:.3f}"


def write_markdown(path, tag, fit_rows, comparison, valid_rows):
    valid_bad = [row for row in valid_rows if row["valid"] != "yes"]
    selected = [
        row for row in comparison
        if row["bytes"] in {1024 * 1024, 4 * 1024 * 1024,
                            16 * 1024 * 1024, 32 * 1024 * 1024}
    ]
    lines = [
        f"# NCCL Allreduce Baseline: Job {tag}",
        "",
        f"Overall result: {'PASS' if not valid_bad else 'FAIL'}",
        "",
        "This compares CUDA-aware MPI device-buffer Allreduce against NCCL",
        "Allreduce using the same message-size sweep as the alpha/beta",
        "microbenchmark. It is a communication-only baseline, not a training",
        "path validation.",
        "",
        "## Alpha/Beta Fits",
        "",
        "| Backend | Ranks | Alpha ms | Beta ns/B | Payload GB/s | R^2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(fit_rows, key=lambda r: (r["backend"], r["ranks"])):
        lines.append(
            f"| {row['backend']} | {row['ranks']} | {row['alpha_ms']:.3f} | "
            f"{row['beta_ns_per_byte']:.3f} | {row['model_payload_gb_s']:.3f} | "
            f"{row['r2']:.3f} |"
        )
    lines += [
        "",
        "## Selected Message Sizes",
        "",
        "| Ranks | Message | MPI-device ms | NCCL ms | NCCL speedup | MPI GB/s | NCCL GB/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['ranks']} | {row['message_mb']:.0f} MB | "
            f"{fmt_ms(row['mpi_device_time_mean_ms'])} | "
            f"{fmt_ms(row['nccl_time_mean_ms'])} | "
            f"{row['nccl_speedup_vs_mpi_device']:.3f}x | "
            f"{row['mpi_device_payload_gb_s']:.3f} | "
            f"{row['nccl_payload_gb_s']:.3f} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- Treat this as the production-collective communication ceiling.",
        "- Small messages can be latency dominated; large messages better expose",
        "  bandwidth and topology differences.",
        "- This result does not replace the validated MPI/OpenMP training path.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--job-id")
    parser.add_argument("--tag")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"nccl_allreduce_baseline_{args.job_id}_raw.txt"
    else:
        raise SystemExit("pass --log or --job-id")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    tag = args.tag or args.job_id or log_path.stem

    rows = alpha_beta.parse_log(log_path)
    if not rows:
        raise SystemExit(f"no benchmark rows parsed from {log_path}")
    summary = alpha_beta.summarize(rows)
    fit_rows = alpha_beta.fits(summary)
    comparison = make_comparison(summary)
    valid_rows = parse_valid(log_path)
    valid_bad = [row for row in valid_rows if row["valid"] != "yes"]

    expected_backends = {"device", "nccl"}
    seen_backends = {row["backend"] for row in summary}
    if seen_backends != expected_backends:
        raise SystemExit(f"expected backends {expected_backends}, saw {seen_backends}")
    if valid_bad:
        raise SystemExit(f"NCCL baseline validity failures: {valid_bad[:3]}")

    raw_fields = ["backend", "ranks", "bytes", "count", "iteration", "time_ms"]
    summary_fields = [
        "backend", "ranks", "bytes", "count", "samples", "time_mean_ms",
        "time_median_ms", "time_std_ms", "effective_payload_gb_s",
    ]
    fit_fields = [
        "backend", "ranks", "points", "alpha_ms", "beta_ms_per_byte",
        "beta_ns_per_byte", "model_payload_gb_s", "r2",
    ]
    comparison_fields = [
        "ranks", "bytes", "message_mb", "mpi_device_time_mean_ms",
        "mpi_device_time_std_ms", "nccl_time_mean_ms", "nccl_time_std_ms",
        "nccl_speedup_vs_mpi_device", "mpi_device_payload_gb_s",
        "nccl_payload_gb_s",
    ]
    valid_fields = ["backend", "ranks", "bytes", "valid"]

    raw_path = RESULTS / f"nccl_allreduce_baseline_{tag}.csv"
    summary_path = RESULTS / f"nccl_allreduce_baseline_summary_{tag}.csv"
    fit_path = RESULTS / f"nccl_allreduce_baseline_fit_{tag}.csv"
    comparison_path = RESULTS / f"nccl_allreduce_baseline_comparison_{tag}.csv"
    valid_path = RESULTS / f"nccl_allreduce_baseline_validity_{tag}.csv"
    md_path = RESULTS / f"nccl_allreduce_baseline_{tag}.md"

    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summary, summary_fields)
    write_csv(fit_path, fit_rows, fit_fields)
    write_csv(comparison_path, comparison, comparison_fields)
    write_csv(valid_path, valid_rows, valid_fields)
    write_markdown(md_path, tag, fit_rows, comparison, valid_rows)
    svg_plot(PLOTS / f"nccl_allreduce_baseline_{tag}.svg", comparison)

    (RESULTS / "nccl_allreduce_baseline.csv").write_text(raw_path.read_text())
    (RESULTS / "nccl_allreduce_baseline_summary.csv").write_text(summary_path.read_text())
    (RESULTS / "nccl_allreduce_baseline_fit.csv").write_text(fit_path.read_text())
    (RESULTS / "nccl_allreduce_baseline_comparison.csv").write_text(comparison_path.read_text())
    (RESULTS / "nccl_allreduce_baseline_validity.csv").write_text(valid_path.read_text())
    (RESULTS / "nccl_allreduce_baseline.md").write_text(md_path.read_text())
    (PLOTS / "nccl_allreduce_baseline.svg").write_text(
        (PLOTS / f"nccl_allreduce_baseline_{tag}.svg").read_text()
    )

    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {fit_path.relative_to(ROOT)}")
    print(f"Wrote {comparison_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
