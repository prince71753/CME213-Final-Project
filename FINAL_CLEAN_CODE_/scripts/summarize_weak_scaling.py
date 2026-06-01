#!/usr/bin/env python3
"""Parse weak-scaling logs into raw/summary CSVs."""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


TITLE_RE = re.compile(
    r"weak hidden=(?P<hidden>[0-9]+)\s+ranks=(?P<ranks>[0-9]+)\s+"
    r"local_batch=(?P<local_batch>[0-9]+)\s+total_batch=(?P<total_batch>[0-9]+)\s+"
    r"backend=(?P<backend>[A-Za-z0-9_]+)\s+sync_variant=(?P<variant>[A-Za-z0-9_]+)\s+"
    r"bucket_kb=(?P<bucket>[0-9]+)\s+repeat=(?P<repeat>[0-9]+)\s+"
    r"steps=(?P<steps>[0-9]+)\s+lr=(?P<lr>[0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?world_size=(?P<world>[0-9]+)\s+sync_mode=(?P<requested>[a-z]+)\s+"
    r"effective_sync=(?P<effective>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)"
)
EPOCH_RE = re.compile(
    r"Epoch 1: avg_logged_loss=(?P<loss>[-+0-9.eE]+|nan|inf|-nan|-inf).*?"
    r"steps/rank=(?P<steps_rank>[0-9]+)\s+(?P<ms>[0-9]+)ms\s+"
    r"(?P<tps>[0-9]+) tok/s"
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


def finite_text(value):
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def parse_log(path):
    rows = []
    for title, body in parse_sections(path.read_text()):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        train = TRAIN_RE.search(body)
        epoch = EPOCH_RE.search(body)
        loss = epoch.group("loss") if epoch else "nan"
        checksum = epoch.group("checksum") if epoch else ""
        body_has_bad = re.search(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", body, re.I)
        valid = bool(epoch) and finite_text(loss) and checksum != "nan" and not body_has_bad
        sync = epoch.group("sync") if epoch else None
        start = epoch.group("start") if epoch else None
        finish = epoch.group("finish") if epoch else None
        steps = int(tm.group("steps"))
        time_ms = float(epoch.group("ms")) if epoch else ""
        rows.append({
            "hidden": int(tm.group("hidden")),
            "ranks": int(tm.group("ranks")),
            "local_batch": int(tm.group("local_batch")),
            "total_batch": int(tm.group("total_batch")),
            "backend": tm.group("backend"),
            "sync_variant": tm.group("variant"),
            "requested_sync_mode": train.group("requested") if train else "",
            "effective_sync": train.group("effective") if train else "",
            "bucket_kb": int(tm.group("bucket")),
            "repeat": int(tm.group("repeat")),
            "requested_steps": steps,
            "lr": tm.group("lr"),
            "avg_logged_loss": loss,
            "steps_rank": int(epoch.group("steps_rank")) if epoch else "",
            "time_ms": time_ms,
            "step_time_ms": time_ms / steps if epoch and steps else "",
            "throughput_tok_s": float(epoch.group("tps")) if epoch else "",
            "throughput_mtok_s": float(epoch.group("tps")) / 1e6 if epoch else "",
            "avg_grad_sync_ms": float(sync) if sync else "",
            "avg_grad_start_ms": float(start) if start else "",
            "avg_grad_finish_ms": float(finish) if finish else "",
            "checksum_span": checksum or "",
            "valid": "yes" if valid else "no",
        })
    return rows


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def sample_std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def cv_pct(vals):
    m = mean(vals)
    if not vals or m == 0:
        return 0.0
    return 100.0 * sample_std(vals) / m


def median(vals):
    if not vals:
        return 0.0
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def only_floats(rows, field):
    out = []
    for row in rows:
        value = row[field]
        if isinstance(value, float):
            out.append(value)
    return out


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["hidden"], row["ranks"], row["local_batch"], row["total_batch"],
            row["backend"], row["sync_variant"], row["requested_sync_mode"],
            row["effective_sync"], row["bucket_kb"], row["requested_steps"], row["lr"],
        )
        groups[key].append(row)

    out = []
    for key in sorted(groups):
        group = groups[key]
        valid_rows = [r for r in group if r["valid"] == "yes"]
        tps = only_floats(valid_rows, "throughput_mtok_s")
        times = only_floats(valid_rows, "time_ms")
        step_times = only_floats(valid_rows, "step_time_ms")
        sync = only_floats(valid_rows, "avg_grad_sync_ms")
        start = only_floats(valid_rows, "avg_grad_start_ms")
        finish = only_floats(valid_rows, "avg_grad_finish_ms")
        checksums = []
        for row in valid_rows:
            try:
                checksums.append(float(row["checksum_span"]))
            except ValueError:
                pass
        hidden, ranks, local_batch, total_batch, backend, variant, requested, effective, bucket, steps, lr = key
        out.append({
            "hidden": hidden,
            "ranks": ranks,
            "local_batch": local_batch,
            "total_batch": total_batch,
            "backend": backend,
            "sync_variant": variant,
            "requested_sync_mode": requested,
            "effective_sync": effective,
            "bucket_kb": bucket,
            "requested_steps": steps,
            "lr": lr,
            "runs": len(group),
            "valid_runs": len(valid_rows),
            "throughput_mean_mtok_s": mean(tps),
            "throughput_median_mtok_s": median(tps),
            "throughput_std_mtok_s": sample_std(tps),
            "throughput_cv_pct": cv_pct(tps),
            "time_mean_ms": mean(times),
            "time_std_ms": sample_std(times),
            "step_time_mean_ms": mean(step_times),
            "step_time_std_ms": sample_std(step_times),
            "avg_grad_sync_mean_ms": mean(sync),
            "avg_grad_start_mean_ms": mean(start),
            "avg_grad_finish_mean_ms": mean(finish),
            "max_checksum_span": max(checksums) if checksums else "",
            "all_valid": "yes" if len(group) == len(valid_rows) else "no",
        })

    baselines = {}
    for row in out:
        if row["ranks"] == 1 and row["valid_runs"] > 0:
            key = (row["hidden"], row["local_batch"], row["backend"], row["sync_variant"])
            baselines[key] = row
    for row in out:
        key = (row["hidden"], row["local_batch"], row["backend"], row["sync_variant"])
        base = baselines.get(key)
        if base and row["valid_runs"] > 0 and base["throughput_mean_mtok_s"] > 0:
            weak_speedup = row["throughput_mean_mtok_s"] / base["throughput_mean_mtok_s"]
            row["weak_speedup"] = weak_speedup
            row["weak_efficiency"] = weak_speedup / row["ranks"]
            if row["step_time_mean_ms"] > 0:
                row["step_time_efficiency"] = base["step_time_mean_ms"] / row["step_time_mean_ms"]
            else:
                row["step_time_efficiency"] = ""
        else:
            row["weak_speedup"] = ""
            row["weak_efficiency"] = ""
            row["step_time_efficiency"] = ""
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_efficiency_plot(path, rows):
    valid = [r for r in rows if r["valid_runs"] > 0 and r["weak_efficiency"] != ""]
    if not valid:
        return
    variants = sorted({r["sync_variant"] for r in valid})
    ranks = sorted({r["ranks"] for r in valid})
    width, height = 780, 440
    left, right, top, bottom = 78, 150, 50, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    ymax = 1.08

    def x_pos(rank):
        if len(ranks) == 1:
            return left + plot_w / 2
        return left + (rank - min(ranks)) / (max(ranks) - min(ranks)) * plot_w

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Weak Scaling Efficiency</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">MPI ranks, fixed per-rank batch</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Efficiency</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>')
    for rank in ranks:
        x = x_pos(rank)
        parts.append(f'<line x1="{x:.1f}" y1="{top+plot_h}" x2="{x:.1f}" y2="{top+plot_h+5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">{rank}</text>')
    parts.append(f'<line x1="{left}" y1="{y_pos(1.0):.1f}" x2="{left+plot_w}" y2="{y_pos(1.0):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="5 5"/>')
    for idx, variant in enumerate(variants):
        color = colors[idx % len(colors)]
        points = []
        for rank in ranks:
            match = next((r for r in valid if r["sync_variant"] == variant and r["ranks"] == rank), None)
            if match:
                points.append((rank, float(match["weak_efficiency"])))
        if not points:
            continue
        coords = " ".join(f'{x_pos(rank):.1f},{y_pos(value):.1f}' for rank, value in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
        for rank, value in points:
            parts.append(f'<circle cx="{x_pos(rank):.1f}" cy="{y_pos(value):.1f}" r="4" fill="{color}"/>')
        ly = top + 42 + idx * 23
        parts.append(f'<line x1="{left+plot_w+18}" y1="{ly-4}" x2="{left+plot_w+42}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left+plot_w+50}" y="{ly}" font-family="Arial" font-size="12">{variant}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id")
    parser.add_argument("--log")
    parser.add_argument("--tag")
    parser.add_argument("--min-repeat", type=int, default=1)
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"weak_scaling_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    tag = args.tag or args.job_id or log_path.stem

    rows = parse_log(log_path)
    if args.min_repeat > 1:
        rows = [row for row in rows if row["repeat"] >= args.min_repeat]
    if not rows:
        raise SystemExit(f"no rows parsed from {log_path}")
    summary = summarize(rows)

    raw_fields = [
        "hidden", "ranks", "local_batch", "total_batch", "backend", "sync_variant",
        "requested_sync_mode", "effective_sync", "bucket_kb", "repeat",
        "requested_steps", "lr", "avg_logged_loss", "steps_rank", "time_ms",
        "step_time_ms", "throughput_tok_s", "throughput_mtok_s",
        "avg_grad_sync_ms", "avg_grad_start_ms", "avg_grad_finish_ms",
        "checksum_span", "valid",
    ]
    summary_fields = [
        "hidden", "ranks", "local_batch", "total_batch", "backend", "sync_variant",
        "requested_sync_mode", "effective_sync", "bucket_kb", "requested_steps",
        "lr", "runs", "valid_runs", "throughput_mean_mtok_s",
        "throughput_median_mtok_s", "throughput_std_mtok_s", "throughput_cv_pct",
        "time_mean_ms", "time_std_ms", "step_time_mean_ms", "step_time_std_ms",
        "avg_grad_sync_mean_ms", "avg_grad_start_mean_ms",
        "avg_grad_finish_mean_ms", "max_checksum_span", "all_valid",
        "weak_speedup", "weak_efficiency", "step_time_efficiency",
    ]
    raw_path = RESULTS / f"weak_scaling_{tag}.csv"
    summary_path = RESULTS / f"weak_scaling_summary_{tag}.csv"
    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summary, summary_fields)
    (RESULTS / "weak_scaling.csv").write_text(raw_path.read_text())
    (RESULTS / "weak_scaling_summary.csv").write_text(summary_path.read_text())
    svg_efficiency_plot(PLOTS / f"weak_scaling_efficiency_{tag}.svg", summary)
    svg_efficiency_plot(PLOTS / "weak_scaling_efficiency.svg", summary)
    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
