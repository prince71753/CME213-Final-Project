#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


TITLE_RE = re.compile(
    r"strong hidden=(?P<hidden>[0-9]+) ranks=(?P<ranks>[0-9]+) "
    r"local_batch=(?P<local_batch>[0-9]+) mode=(?P<mode>[a-z]+)"
)
EPOCH_RE = re.compile(
    r"Epoch 1: avg_logged_loss=(?P<loss>[0-9.]+|nan).*?"
    r" (?P<ms>[0-9]+)ms\s+(?P<tps>[0-9]+) tok/s"
    r"(?:\s+avg_grad_sync=(?P<sync>[0-9.]+)ms|"
    r"\s+avg_grad_start=(?P<start>[0-9.]+)ms\s+avg_grad_finish=(?P<finish>[0-9.]+)ms)?"
    r"(?:\s+checksum_span=(?P<checksum>[0-9.eE+-]+|nan))?"
)


def parse_sections(text):
    current = None
    body = []
    for line in text.splitlines():
        if line.startswith("===") and line.endswith("==="):
            if current is not None:
                yield current, "\n".join(body)
            current = line.strip("= ").strip()
            body = []
        else:
            body.append(line)
    if current is not None:
        yield current, "\n".join(body)


def parse_log(path):
    rows = []
    for title, body in parse_sections(path.read_text()):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        ep = EPOCH_RE.search(body)
        if not ep:
            continue
        loss = ep.group("loss")
        checksum = ep.group("checksum") or ""
        valid = loss != "nan" and checksum != "nan"
        sync = ep.group("sync")
        start = ep.group("start")
        finish = ep.group("finish")
        rows.append({
            "hidden": int(tm.group("hidden")),
            "ranks": int(tm.group("ranks")),
            "local_batch": int(tm.group("local_batch")),
            "mode": tm.group("mode"),
            "avg_logged_loss": loss,
            "time_ms": float(ep.group("ms")),
            "throughput_tok_s": float(ep.group("tps")),
            "throughput_mtok_s": float(ep.group("tps")) / 1e6,
            "avg_grad_sync_ms": float(sync) if sync else "",
            "avg_grad_start_ms": float(start) if start else "",
            "avg_grad_finish_ms": float(finish) if finish else "",
            "checksum_span": checksum,
            "valid": "yes" if valid else "no",
        })
    return rows


def add_speedups(rows):
    baselines = {}
    for row in rows:
        if row["mode"] == "blocking" and row["ranks"] == 1 and row["valid"] == "yes":
            baselines[row["hidden"]] = row["time_ms"]
    for row in rows:
        base = baselines.get(row["hidden"])
        if base and row["valid"] == "yes":
            speedup = base / row["time_ms"]
            row["strong_speedup"] = speedup
            row["strong_efficiency"] = speedup / row["ranks"]
        else:
            row["strong_speedup"] = ""
            row["strong_efficiency"] = ""


def svg_scaling_plot(path, rows, metric, title, ylabel, ideal=False):
    valid_rows = [r for r in rows if r["valid"] == "yes"]
    hidden_values = sorted({r["hidden"] for r in valid_rows})
    ranks = sorted({r["ranks"] for r in valid_rows})
    width, height = 820, 470
    left, right, top, bottom = 80, 160, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {
        (128, "blocking"): "#2563eb",
        (128, "overlap"): "#60a5fa",
        (256, "blocking"): "#dc2626",
        (256, "overlap"): "#f97316",
    }
    series = []
    for hidden in hidden_values:
        for mode in ["blocking", "overlap"]:
            points = []
            for rank in ranks:
                match = next((r for r in valid_rows
                              if r["hidden"] == hidden
                              and r["mode"] == mode
                              and r["ranks"] == rank), None)
                if match:
                    points.append((rank, float(match[metric])))
            if points:
                series.append((f"H{hidden} {mode}", hidden, mode, points))
    ymax = max([v for _, _, _, points in series for _, v in points] + ([max(ranks)] if ideal else [1.0]))
    ymax *= 1.12
    xmin, xmax = min(ranks), max(ranks)

    def x_pos(rank):
        if xmin == xmax:
            return left + plot_w / 2
        return left + (rank - xmin) / (xmax - xmin) * plot_w

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">MPI ranks</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(5):
        value = ymax * tick / 4
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for rank in ranks:
        x = x_pos(rank)
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">{rank}</text>')
    if ideal:
        ideal_points = [(rank, rank) for rank in ranks]
        d = " ".join(f'{x_pos(rank):.1f},{y_pos(value):.1f}' for rank, value in ideal_points)
        parts.append(f'<polyline points="{d}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{left+plot_w+18}" y="{top+18}" font-family="Arial" font-size="12">ideal</text>')
    for idx, (name, hidden, mode, points) in enumerate(series):
        color = colors.get((hidden, mode), "#4b5563")
        d = " ".join(f'{x_pos(rank):.1f},{y_pos(value):.1f}' for rank, value in points)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="3"/>')
        for rank, value in points:
            x = x_pos(rank)
            y = y_pos(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        ly = top + 45 + idx * 22
        parts.append(f'<line x1="{left+plot_w+18}" y1="{ly-4}" x2="{left+plot_w+42}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left+plot_w+50}" y="{ly}" font-family="Arial" font-size="12">{name}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    log_path = LOGS / f"strong_scaling_{args.job_id}.out"
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no rows parsed from {log_path}")
    add_speedups(rows)

    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    out = RESULTS / f"strong_scaling_{args.job_id}.csv"
    fields = [
        "hidden", "ranks", "local_batch", "mode", "avg_logged_loss",
        "time_ms", "throughput_tok_s", "throughput_mtok_s",
        "avg_grad_sync_ms", "avg_grad_start_ms", "avg_grad_finish_ms",
        "checksum_span", "valid", "strong_speedup", "strong_efficiency",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    canonical = RESULTS / "strong_scaling.csv"
    canonical.write_text(out.read_text())
    svg_scaling_plot(
        PLOTS / f"strong_scaling_speedup_{args.job_id}.svg",
        rows,
        "strong_speedup",
        "Strong Scaling Speedup",
        "Speedup vs 1 rank",
        ideal=True,
    )
    svg_scaling_plot(
        PLOTS / "strong_scaling_speedup.svg",
        rows,
        "strong_speedup",
        "Strong Scaling Speedup",
        "Speedup vs 1 rank",
        ideal=True,
    )
    svg_scaling_plot(
        PLOTS / f"strong_scaling_efficiency_{args.job_id}.svg",
        rows,
        "strong_efficiency",
        "Strong Scaling Efficiency",
        "Parallel efficiency",
    )
    svg_scaling_plot(
        PLOTS / "strong_scaling_efficiency.svg",
        rows,
        "strong_efficiency",
        "Strong Scaling Efficiency",
        "Parallel efficiency",
    )
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
