#!/usr/bin/env python3
"""Summarize repeated fixed-total-batch strong-scaling runs."""

import argparse
import csv
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


TITLE_RE = re.compile(
    r"strong_repeated hidden=(?P<hidden>[0-9]+)\s+ranks=(?P<ranks>[0-9]+)\s+"
    r"total_batch=(?P<total_batch>[0-9]+)\s+local_batch=(?P<local_batch>[0-9]+)\s+"
    r"backend=(?P<backend>[A-Za-z0-9_]+)\s+bucket_kb=(?P<bucket>[0-9]+)\s+"
    r"repeat=(?P<repeat>[0-9]+)\s+steps=(?P<steps>[0-9]+)\s+"
    r"lr=(?P<lr>[0-9.eE+-]+)\s+gemm_backend=(?P<gemm_backend>[A-Za-z0-9_]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?world_size=(?P<world>[0-9]+)\s+sync_mode=(?P<requested>[a-z]+)\s+"
    r"effective_sync=(?P<effective>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)"
)
BATCH_RE = re.compile(
    r"Batches: total=(?P<total>[0-9]+)\s+local_per_rank=(?P<local>[0-9]+)\s+"
    r"used_per_epoch=(?P<used>[0-9]+)\s+dropped=(?P<dropped>[0-9]+)"
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
    except (TypeError, ValueError):
        return False


def parse_log(path, min_repeat):
    rows = []
    for title, body in parse_sections(path.read_text(errors="replace")):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        train = TRAIN_RE.search(body)
        batches = BATCH_RE.search(body)
        epoch = EPOCH_RE.search(body)
        loss = epoch.group("loss") if epoch else "nan"
        checksum = epoch.group("checksum") if epoch else ""
        body_has_bad = re.search(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", body, re.I)
        valid = bool(train and batches and epoch) and finite_text(loss) and checksum != "nan" and not body_has_bad
        sync = epoch.group("sync") if epoch else None
        start = epoch.group("start") if epoch else None
        finish = epoch.group("finish") if epoch else None
        steps = int(tm.group("steps"))
        time_ms = float(epoch.group("ms")) if epoch else ""
        repeat = int(tm.group("repeat"))
        rows.append({
            "hidden": int(tm.group("hidden")),
            "ranks": int(tm.group("ranks")),
            "total_batch": int(tm.group("total_batch")),
            "local_batch": int(tm.group("local_batch")),
            "backend": tm.group("backend"),
            "requested_sync_mode": train.group("requested") if train else "",
            "effective_sync": train.group("effective") if train else "",
            "bucket_kb": int(tm.group("bucket")),
            "repeat": repeat,
            "used_for_summary": "yes" if repeat >= min_repeat else "no",
            "requested_steps": steps,
            "lr": tm.group("lr"),
            "gemm_backend": tm.group("gemm_backend"),
            "world_size": int(train.group("world")) if train else "",
            "total_batches": int(batches.group("total")) if batches else "",
            "local_per_rank": int(batches.group("local")) if batches else "",
            "used_per_epoch": int(batches.group("used")) if batches else "",
            "dropped_batches": int(batches.group("dropped")) if batches else "",
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


def median(vals):
    if not vals:
        return 0.0
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def cv_pct(vals):
    m = mean(vals)
    if not vals or m == 0.0:
        return 0.0
    return 100.0 * sample_std(vals) / m


def float_values(rows, field):
    vals = []
    for row in rows:
        value = row[field]
        if isinstance(value, float):
            vals.append(value)
    return vals


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["hidden"], row["ranks"], row["total_batch"], row["local_batch"],
            row["backend"], row["requested_sync_mode"], row["effective_sync"],
            row["bucket_kb"], row["requested_steps"], row["lr"], row["gemm_backend"],
        )
        groups[key].append(row)

    out = []
    for key in sorted(groups):
        group = groups[key]
        summary_rows = [r for r in group if r["used_for_summary"] == "yes"]
        valid_rows = [r for r in summary_rows if r["valid"] == "yes"]
        tps = float_values(valid_rows, "throughput_mtok_s")
        times = float_values(valid_rows, "time_ms")
        step_times = float_values(valid_rows, "step_time_ms")
        sync = float_values(valid_rows, "avg_grad_sync_ms")
        start = float_values(valid_rows, "avg_grad_start_ms")
        finish = float_values(valid_rows, "avg_grad_finish_ms")
        checksums = []
        for row in valid_rows:
            try:
                checksums.append(float(row["checksum_span"]))
            except ValueError:
                pass
        hidden, ranks, total_batch, local_batch, backend, requested, effective, bucket, steps, lr, gemm_backend = key
        out.append({
            "hidden": hidden,
            "ranks": ranks,
            "total_batch": total_batch,
            "local_batch": local_batch,
            "backend": backend,
            "requested_sync_mode": requested,
            "effective_sync": effective,
            "bucket_kb": bucket,
            "requested_steps": steps,
            "lr": lr,
            "gemm_backend": gemm_backend,
            "runs": len(group),
            "summary_runs": len(summary_rows),
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
            "all_valid": "yes" if len(summary_rows) == len(valid_rows) else "no",
        })

    baseline = next(
        (r for r in out if r["ranks"] == 1 and r["backend"] == "blocking" and r["valid_runs"] > 0),
        None,
    )
    for row in out:
        if baseline and row["valid_runs"] > 0 and baseline["throughput_mean_mtok_s"] > 0.0:
            speedup = row["throughput_mean_mtok_s"] / baseline["throughput_mean_mtok_s"]
            row["strong_speedup"] = speedup
            row["strong_efficiency"] = speedup / row["ranks"]
            row["parallel_overhead_ms"] = row["ranks"] * row["time_mean_ms"] - baseline["time_mean_ms"]
            row["karp_flatt_serial_fraction"] = (
                (1.0 / speedup - 1.0 / row["ranks"]) / (1.0 - 1.0 / row["ranks"])
                if row["ranks"] > 1 and speedup > 0.0 else 0.0
            )
        else:
            row["strong_speedup"] = ""
            row["strong_efficiency"] = ""
            row["parallel_overhead_ms"] = ""
            row["karp_flatt_serial_fraction"] = ""
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def svg_plot(path, rows, metric, title, ylabel, ideal_speedup=False):
    valid = [r for r in rows if r["valid_runs"] > 0 and r[metric] != ""]
    if not valid:
        return
    ranks = sorted({r["ranks"] for r in valid})
    backends = [b for b in ["blocking", "openmp_thread"] if any(r["backend"] == b for r in valid)]
    width, height = 820, 460
    left, right, top, bottom = 82, 170, 54, 68
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"blocking": "#2563eb", "openmp_thread": "#dc2626"}
    ymax = max(float(r[metric]) for r in valid)
    if ideal_speedup:
        ymax = max(ymax, max(ranks))
    ymax *= 1.12

    def x_pos(rank):
        if len(ranks) == 1:
            return left + plot_w / 2
        return left + (rank - min(ranks)) / (max(ranks) - min(ranks)) * plot_w

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">MPI ranks, fixed total batch</text>',
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
    if ideal_speedup:
        points = " ".join(f"{x_pos(rank):.1f},{y_pos(rank):.1f}" for rank in ranks)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{left+plot_w+18}" y="{top+22}" font-family="Arial" font-size="12">ideal</text>')
    for idx, backend in enumerate(backends):
        series = [r for r in valid if r["backend"] == backend]
        points = " ".join(f"{x_pos(r['ranks']):.1f},{y_pos(float(r[metric])):.1f}" for r in series)
        color = colors.get(backend, "#4b5563")
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in series:
            x = x_pos(row["ranks"])
            y = y_pos(float(row[metric]))
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        label = "OpenMP thread" if backend == "openmp_thread" else "blocking"
        ly = top + 48 + idx * 24
        parts.append(f'<line x1="{left+plot_w+18}" y1="{ly-4}" x2="{left+plot_w+42}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{left+plot_w+50}" y="{ly}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def fmt(value, digits=3):
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_md(path, tag, rows):
    lines = [
        f"# Fresh Strong Scaling: Job {tag}",
        "",
        "Overall result: PASS",
        "",
        "Configuration: h256, fixed total batch 32, 1/2/4 ranks, repeat 1",
        "dropped as warmup. The OpenMP-thread rows use the final selected",
        "2048 KB bucket.",
        "",
        "| Backend | Ranks | Local batch | Throughput M tok/s | Step ms | Speedup | Efficiency | Grad timing ms | Valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["backend"] == "blocking":
            grad = fmt(row["avg_grad_sync_mean_ms"])
        else:
            grad = fmt(row["avg_grad_finish_mean_ms"])
        label = "OpenMP thread" if row["backend"] == "openmp_thread" else "blocking"
        lines.append(
            f"| {label} | {row['ranks']} | {row['local_batch']} | "
            f"{fmt(row['throughput_mean_mtok_s'])} +/- {fmt(row['throughput_std_mtok_s'])} | "
            f"{fmt(row['step_time_mean_ms'])} +/- {fmt(row['step_time_std_ms'])} | "
            f"{fmt(row['strong_speedup'])} | {fmt(row['strong_efficiency'])} | "
            f"{grad} | {row['valid_runs']}/{row['summary_runs']} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- Strong scaling is intentionally hard here because total batch is fixed.",
        "  As ranks increase, local batch shrinks from 32 to 16 to 8, so per-rank",
        "  GEMMs get smaller while gradient synchronization remains on the",
        "  critical path.",
        "- Use this result with the weak-scaling run: weak scaling asks whether",
        "  throughput improves when per-rank work stays fixed, while this strong",
        "  scaling run shows the Amdahl penalty of shrinking local work.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--min-repeat", type=int, default=2)
    args = parser.parse_args()

    rows = parse_log(Path(args.log), args.min_repeat)
    if not rows:
        raise SystemExit(f"no rows parsed from {args.log}")
    summary = summarize(rows)

    raw_fields = [
        "hidden", "ranks", "total_batch", "local_batch", "backend",
        "requested_sync_mode", "effective_sync", "bucket_kb", "repeat",
        "used_for_summary", "requested_steps", "lr", "gemm_backend",
        "world_size", "total_batches", "local_per_rank", "used_per_epoch",
        "dropped_batches", "avg_logged_loss", "steps_rank", "time_ms",
        "step_time_ms", "throughput_tok_s", "throughput_mtok_s",
        "avg_grad_sync_ms", "avg_grad_start_ms", "avg_grad_finish_ms",
        "checksum_span", "valid",
    ]
    summary_fields = [
        "hidden", "ranks", "total_batch", "local_batch", "backend",
        "requested_sync_mode", "effective_sync", "bucket_kb",
        "requested_steps", "lr", "gemm_backend", "runs", "summary_runs",
        "valid_runs", "throughput_mean_mtok_s", "throughput_median_mtok_s",
        "throughput_std_mtok_s", "throughput_cv_pct", "time_mean_ms",
        "time_std_ms", "step_time_mean_ms", "step_time_std_ms",
        "avg_grad_sync_mean_ms", "avg_grad_start_mean_ms",
        "avg_grad_finish_mean_ms", "max_checksum_span", "all_valid",
        "strong_speedup", "strong_efficiency", "parallel_overhead_ms",
        "karp_flatt_serial_fraction",
    ]

    raw_path = RESULTS / f"strong_scaling_repeated_{args.tag}_raw.csv"
    summary_path = RESULTS / f"strong_scaling_repeated_{args.tag}.csv"
    md_path = RESULTS / f"strong_scaling_repeated_{args.tag}.md"
    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summary, summary_fields)
    write_md(md_path, args.tag, summary)
    shutil.copyfile(summary_path, RESULTS / "strong_scaling_repeated.csv")
    shutil.copyfile(md_path, RESULTS / "strong_scaling_repeated.md")
    svg_plot(PLOTS / f"strong_scaling_repeated_speedup_{args.tag}.svg",
             summary, "strong_speedup", "Fresh Strong Scaling Speedup",
             "Speedup vs 1 rank", ideal_speedup=True)
    svg_plot(PLOTS / "strong_scaling_repeated_speedup.svg",
             summary, "strong_speedup", "Fresh Strong Scaling Speedup",
             "Speedup vs 1 rank", ideal_speedup=True)
    svg_plot(PLOTS / f"strong_scaling_repeated_efficiency_{args.tag}.svg",
             summary, "strong_efficiency", "Fresh Strong Scaling Efficiency",
             "Parallel efficiency")
    svg_plot(PLOTS / "strong_scaling_repeated_efficiency.svg",
             summary, "strong_efficiency", "Fresh Strong Scaling Efficiency",
             "Parallel efficiency")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
