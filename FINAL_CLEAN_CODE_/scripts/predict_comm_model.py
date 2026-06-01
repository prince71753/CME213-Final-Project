#!/usr/bin/env python3
"""Compare alpha/beta Allreduce predictions with measured training timers."""

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


def read_fit():
    path = RESULTS / "allreduce_alpha_beta_fit.csv"
    if not path.exists():
        raise SystemExit(f"missing fit file: {path}")
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["backend"] == "device" and int(row["ranks"]) == 4:
                return float(row["alpha_ms"]), float(row["beta_ms_per_byte"])
    raise SystemExit("missing device 4-rank fit")


def read_breakdown():
    path = RESULTS / "comm_breakdown_preliminary.csv"
    if not path.exists():
        raise SystemExit(f"missing breakdown file: {path}")
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_plot(path, rows):
    if not rows:
        return
    width, height = 820, 460
    left, right, top, bottom = 86, 160, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(max(r["pred_bucketed_ms"], r["measured_blocking_comm_ms"],
                   r["measured_exposed_wait_ms"]) for r in rows) * 1.16
    group_w = plot_w / len(rows)
    bar_w = min(46, group_w / 5)

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Allreduce Model vs Training Timers</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">Hidden size</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">ms</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>')
    colors = {
        "pred": "#2563eb",
        "blocking": "#f97316",
        "exposed": "#16a34a",
    }
    for idx, row in enumerate(rows):
        center = left + group_w * (idx + 0.5)
        bars = [
            ("pred", row["pred_bucketed_ms"], -bar_w * 1.15),
            ("blocking", row["measured_blocking_comm_ms"], 0.0),
            ("exposed", row["measured_exposed_wait_ms"], bar_w * 1.15),
        ]
        for name, value, offset in bars:
            x = center + offset
            y = y_pos(value)
            parts.append(f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y:.1f}" fill="{colors[name]}"/>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">h{row["hidden"]}</text>')
    lx = left + plot_w + 18
    legend = [
        ("pred", "pred bucketed"),
        ("blocking", "measured blocking"),
        ("exposed", "measured exposed"),
    ]
    for idx, (name, label) in enumerate(legend):
        y = top + 36 + idx * 24
        parts.append(f'<rect x="{lx}" y="{y-12}" width="14" height="14" fill="{colors[name]}"/>')
        parts.append(f'<text x="{lx+22}" y="{y}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def main():
    alpha_ms, beta_ms_per_byte = read_fit()
    rows = []
    for row in read_breakdown():
        hidden = int(row["hidden"])
        params = PARAM_COUNTS[hidden]
        bytes_ = params * 4
        bucket_kb = int(row["bucket_kb"])
        bucket_bytes = bucket_kb * 1024
        buckets = max(1, math.ceil(bytes_ / bucket_bytes)) if bucket_bytes > 0 else 1
        pred_single = alpha_ms + beta_ms_per_byte * bytes_
        pred_bucketed = buckets * alpha_ms + beta_ms_per_byte * bytes_
        measured_block = float(row["blocking_comm_proxy_ms"])
        measured_exposed = float(row["exposed_wait_ms"])
        rows.append({
            "hidden": hidden,
            "params": params,
            "gradient_bytes": bytes_,
            "bucket_kb": bucket_kb,
            "modeled_buckets": buckets,
            "alpha_ms": alpha_ms,
            "beta_ms_per_byte": beta_ms_per_byte,
            "pred_single_message_ms": pred_single,
            "pred_bucketed_ms": pred_bucketed,
            "measured_blocking_comm_ms": measured_block,
            "measured_exposed_wait_ms": measured_exposed,
            "blocking_over_pred_bucketed": measured_block / pred_bucketed
            if pred_bucketed > 0 else "",
            "exposed_over_pred_bucketed": measured_exposed / pred_bucketed
            if pred_bucketed > 0 else "",
        })
    fields = [
        "hidden", "params", "gradient_bytes", "bucket_kb", "modeled_buckets",
        "alpha_ms", "beta_ms_per_byte", "pred_single_message_ms",
        "pred_bucketed_ms", "measured_blocking_comm_ms",
        "measured_exposed_wait_ms", "blocking_over_pred_bucketed",
        "exposed_over_pred_bucketed",
    ]
    write_csv(RESULTS / "allreduce_prediction_vs_measurement.csv", rows, fields)
    svg_plot(PLOTS / "allreduce_prediction_vs_measurement.svg", rows)
    print("Wrote results/allreduce_prediction_vs_measurement.csv")
    print("Wrote plots/allreduce_prediction_vs_measurement.svg")


if __name__ == "__main__":
    main()
