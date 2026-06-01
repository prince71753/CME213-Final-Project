#!/usr/bin/env python3
"""Summarize MPI Allreduce alpha/beta microbenchmark output."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def sample_std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def median(vals):
    if not vals:
        return 0.0
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def parse_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("allreduce_alpha_beta,"):
            continue
        parts = line.split(",")
        if len(parts) != 7 or parts[1] == "backend":
            continue
        rows.append({
            "backend": parts[1],
            "ranks": int(parts[2]),
            "bytes": int(parts[3]),
            "count": int(parts[4]),
            "iteration": int(parts[5]),
            "time_ms": float(parts[6]),
        })
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["backend"], row["ranks"], row["bytes"], row["count"])].append(row)
    out = []
    for (backend, ranks, bytes_, count), group in sorted(groups.items()):
        times = [r["time_ms"] for r in group]
        out.append({
            "backend": backend,
            "ranks": ranks,
            "bytes": bytes_,
            "count": count,
            "samples": len(group),
            "time_mean_ms": mean(times),
            "time_median_ms": median(times),
            "time_std_ms": sample_std(times),
            "effective_payload_gb_s": (bytes_ / 1e9) / (mean(times) / 1000.0)
            if mean(times) > 0 else 0.0,
        })
    return out


def fit_line(points):
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    den = n * sxx - sx * sx
    if n < 2 or den == 0:
        return 0.0, 0.0, 0.0
    beta = (n * sxy - sx * sy) / den
    alpha = (sy - beta * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for _, y in points)
    ss_res = sum((y - (alpha + beta * x)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return alpha, beta, r2


def fits(summary):
    groups = defaultdict(list)
    for row in summary:
        groups[(row["backend"], row["ranks"])].append(
            (row["bytes"], row["time_mean_ms"])
        )
    out = []
    for (backend, ranks), points in sorted(groups.items()):
        points = sorted(points)
        alpha_ms, beta_ms_per_byte, r2 = fit_line(points)
        model_gb_s = 1e-6 / beta_ms_per_byte if beta_ms_per_byte > 0 else 0.0
        out.append({
            "backend": backend,
            "ranks": ranks,
            "points": len(points),
            "alpha_ms": alpha_ms,
            "beta_ms_per_byte": beta_ms_per_byte,
            "beta_ns_per_byte": beta_ms_per_byte * 1e6,
            "model_payload_gb_s": model_gb_s,
            "r2": r2,
        })
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_plot(path, summary, fit_rows):
    if not summary:
        return
    width, height = 840, 500
    left, right, top, bottom = 95, 180, 55, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [math.log2(r["bytes"]) for r in summary if r["bytes"] > 0]
    ys = [r["time_mean_ms"] for r in summary]
    xmin, xmax = min(xs), max(xs)
    ymax = max(ys) * 1.12
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    keys = sorted({(r["backend"], r["ranks"]) for r in summary})
    fit_map = {(r["backend"], r["ranks"]): r for r in fit_rows}

    def x_pos(bytes_):
        x = math.log2(bytes_)
        if xmin == xmax:
            return left + plot_w / 2
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">MPI Allreduce Alpha/Beta Fit</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">Message size</text>',
        f'<text x="20" y="{top + plot_h/2}" transform="rotate(-90 20 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Time (ms)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for bytes_ in sorted({r["bytes"] for r in summary}):
        x = x_pos(bytes_)
        label = f'{bytes_ // 1024}KB' if bytes_ < 1024 * 1024 else f'{bytes_ // (1024 * 1024)}MB'
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="10">{label}</text>')
    for idx, key in enumerate(keys):
        color = colors[idx % len(colors)]
        series = sorted([r for r in summary if (r["backend"], r["ranks"]) == key],
                        key=lambda r: r["bytes"])
        coords = " ".join(f'{x_pos(r["bytes"]):.1f},{y_pos(r["time_mean_ms"]):.1f}' for r in series)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in series:
            parts.append(f'<circle cx="{x_pos(row["bytes"]):.1f}" cy="{y_pos(row["time_mean_ms"]):.1f}" r="3.5" fill="{color}"/>')
        fit = fit_map.get(key)
        if fit:
            x0 = series[0]["bytes"]
            x1 = series[-1]["bytes"]
            y0 = fit["alpha_ms"] + fit["beta_ms_per_byte"] * x0
            y1 = fit["alpha_ms"] + fit["beta_ms_per_byte"] * x1
            parts.append(f'<line x1="{x_pos(x0):.1f}" y1="{y_pos(y0):.1f}" x2="{x_pos(x1):.1f}" y2="{y_pos(y1):.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 5"/>')
        ly = top + 42 + idx * 23
        label = f'{key[0]} {key[1]} ranks'
        parts.append(f'<line x1="{left+plot_w+18}" y1="{ly-4}" x2="{left+plot_w+42}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left+plot_w+50}" y="{ly}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id")
    parser.add_argument("--log")
    parser.add_argument("--tag")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"allreduce_alpha_beta_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    tag = args.tag or args.job_id or log_path.stem

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no benchmark rows parsed from {log_path}")
    summary = summarize(rows)
    fit_rows = fits(summary)

    raw_fields = ["backend", "ranks", "bytes", "count", "iteration", "time_ms"]
    summary_fields = [
        "backend", "ranks", "bytes", "count", "samples", "time_mean_ms",
        "time_median_ms", "time_std_ms", "effective_payload_gb_s",
    ]
    fit_fields = [
        "backend", "ranks", "points", "alpha_ms", "beta_ms_per_byte",
        "beta_ns_per_byte", "model_payload_gb_s", "r2",
    ]
    raw_path = RESULTS / f"allreduce_alpha_beta_{tag}.csv"
    summary_path = RESULTS / f"allreduce_alpha_beta_summary_{tag}.csv"
    fit_path = RESULTS / f"allreduce_alpha_beta_fit_{tag}.csv"
    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summary, summary_fields)
    write_csv(fit_path, fit_rows, fit_fields)
    (RESULTS / "allreduce_alpha_beta.csv").write_text(raw_path.read_text())
    (RESULTS / "allreduce_alpha_beta_summary.csv").write_text(summary_path.read_text())
    (RESULTS / "allreduce_alpha_beta_fit.csv").write_text(fit_path.read_text())
    svg_plot(PLOTS / f"allreduce_alpha_beta_{tag}.svg", summary, fit_rows)
    svg_plot(PLOTS / "allreduce_alpha_beta.svg", summary, fit_rows)
    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {fit_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
