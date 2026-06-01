#!/usr/bin/env python3
"""Summarize blocking vs OpenMP-thread overlap speedup across hidden sizes."""

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
    r"overlap_speedup hidden=(?P<hidden>[0-9]+)\s+ranks=(?P<ranks>[0-9]+)\s+"
    r"batch=(?P<batch>[0-9]+)\s+backend=(?P<backend>[A-Za-z0-9_]+)\s+"
    r"sync_mode=(?P<mode>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)\s+"
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


def parse_log(path):
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
        rows.append({
            "hidden": int(tm.group("hidden")),
            "ranks": int(tm.group("ranks")),
            "batch": int(tm.group("batch")),
            "backend": tm.group("backend"),
            "sync_mode": tm.group("mode"),
            "requested_sync_mode": train.group("requested") if train else "",
            "effective_sync": train.group("effective") if train else "",
            "bucket_kb": int(tm.group("bucket")),
            "repeat": int(tm.group("repeat")),
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


def cv_pct(vals):
    m = mean(vals)
    if not vals or m == 0.0:
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


def float_values(rows, field):
    vals = []
    for row in rows:
        value = row[field]
        if isinstance(value, float):
            vals.append(value)
    return vals


def summarize_modes(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["hidden"], row["ranks"], row["batch"], row["backend"],
            row["sync_mode"], row["requested_sync_mode"], row["effective_sync"],
            row["bucket_kb"], row["requested_steps"], row["lr"],
            row["gemm_backend"],
        )
        groups[key].append(row)

    out = []
    for key in sorted(groups):
        group = groups[key]
        valid_rows = [row for row in group if row["valid"] == "yes"]
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
        hidden, ranks, batch, backend, mode, requested, effective, bucket, steps, lr, gemm_backend = key
        out.append({
            "hidden": hidden,
            "ranks": ranks,
            "batch": batch,
            "backend": backend,
            "sync_mode": mode,
            "requested_sync_mode": requested,
            "effective_sync": effective,
            "bucket_kb": bucket,
            "requested_steps": steps,
            "lr": lr,
            "gemm_backend": gemm_backend,
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
    return out


def ratio_std(ratio, a_mean, a_std, b_mean, b_std):
    if a_mean <= 0.0 or b_mean <= 0.0:
        return 0.0
    rel_a = a_std / a_mean if a_mean else 0.0
    rel_b = b_std / b_mean if b_mean else 0.0
    return ratio * math.sqrt(rel_a * rel_a + rel_b * rel_b)


def derive_speedups(summary):
    by_hidden = defaultdict(dict)
    for row in summary:
        by_hidden[row["hidden"]][row["backend"]] = row

    out = []
    for hidden in sorted(by_hidden):
        group = by_hidden[hidden]
        blocking = group.get("blocking")
        openmp = group.get("openmp_thread")
        if not blocking or not openmp:
            continue
        tput_speedup = 0.0
        tput_speedup_std = 0.0
        if blocking["throughput_mean_mtok_s"] > 0.0:
            tput_speedup = openmp["throughput_mean_mtok_s"] / blocking["throughput_mean_mtok_s"]
            tput_speedup_std = ratio_std(
                tput_speedup,
                openmp["throughput_mean_mtok_s"], openmp["throughput_std_mtok_s"],
                blocking["throughput_mean_mtok_s"], blocking["throughput_std_mtok_s"],
            )
        step_speedup = 0.0
        if openmp["step_time_mean_ms"] > 0.0:
            step_speedup = blocking["step_time_mean_ms"] / openmp["step_time_mean_ms"]
        comm_tail_reduction = ""
        if blocking["avg_grad_sync_mean_ms"] > 0.0:
            comm_tail_reduction = 100.0 * (
                1.0 - openmp["avg_grad_finish_mean_ms"] / blocking["avg_grad_sync_mean_ms"]
            )
        checksum_vals = []
        for row in (blocking, openmp):
            try:
                checksum_vals.append(float(row["max_checksum_span"]))
            except ValueError:
                pass
        out.append({
            "hidden": hidden,
            "ranks": blocking["ranks"],
            "batch": blocking["batch"],
            "gemm_backend": blocking["gemm_backend"],
            "blocking_bucket_kb": blocking["bucket_kb"],
            "openmp_bucket_kb": openmp["bucket_kb"],
            "requested_steps": blocking["requested_steps"],
            "lr": blocking["lr"],
            "blocking_valid_runs": blocking["valid_runs"],
            "openmp_valid_runs": openmp["valid_runs"],
            "blocking_throughput_mean_mtok_s": blocking["throughput_mean_mtok_s"],
            "blocking_throughput_std_mtok_s": blocking["throughput_std_mtok_s"],
            "openmp_throughput_mean_mtok_s": openmp["throughput_mean_mtok_s"],
            "openmp_throughput_std_mtok_s": openmp["throughput_std_mtok_s"],
            "throughput_speedup": tput_speedup,
            "throughput_speedup_std": tput_speedup_std,
            "blocking_step_time_mean_ms": blocking["step_time_mean_ms"],
            "openmp_step_time_mean_ms": openmp["step_time_mean_ms"],
            "step_time_speedup": step_speedup,
            "blocking_comm_sync_mean_ms": blocking["avg_grad_sync_mean_ms"],
            "openmp_enqueue_mean_ms": openmp["avg_grad_start_mean_ms"],
            "openmp_exposed_wait_mean_ms": openmp["avg_grad_finish_mean_ms"],
            "comm_tail_reduction_pct": comm_tail_reduction,
            "max_checksum_span": max(checksum_vals) if checksum_vals else "",
            "all_valid": "yes" if blocking["all_valid"] == "yes" and openmp["all_valid"] == "yes" else "no",
        })
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def copy_text(src, dst):
    dst.write_text(src.read_text())


def svg_speedup_plot(path, rows):
    if not rows:
        return
    width, height = 780, 440
    left, right, top, bottom = 80, 44, 52, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    hidden_vals = [row["hidden"] for row in rows]
    ymax = max(1.25, max(row["throughput_speedup"] + row["throughput_speedup_std"] for row in rows) * 1.15)

    def x_pos(hidden):
        if len(hidden_vals) == 1:
            return left + plot_w / 2
        idx = hidden_vals.index(hidden)
        return left + idx * plot_w / (len(hidden_vals) - 1)

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">OpenMP Communication-Thread Speedup</text>',
        f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13">Hidden size, 4 ranks, batch 32, repeat 1 dropped</text>',
        f'<text x="22" y="{top + plot_h/2}" transform="rotate(-90 22 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Throughput speedup vs blocking</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    tick_count = 6
    for tick in range(tick_count + 1):
        val = ymax * tick / tick_count
        y = y_pos(val)
        parts += [
            f'<line x1="{left-5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>',
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}x</text>',
        ]
    y1 = y_pos(1.0)
    parts.append(f'<line x1="{left}" y1="{y1:.1f}" x2="{left+plot_w}" y2="{y1:.1f}" stroke="#6b7280" stroke-dasharray="5 4"/>')
    pts = []
    for row in rows:
        x = x_pos(row["hidden"])
        y = y_pos(row["throughput_speedup"])
        pts.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    for row in rows:
        x = x_pos(row["hidden"])
        y = y_pos(row["throughput_speedup"])
        y_hi = y_pos(row["throughput_speedup"] + row["throughput_speedup_std"])
        y_lo = y_pos(max(0.0, row["throughput_speedup"] - row["throughput_speedup_std"]))
        parts += [
            f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="#1d4ed8" stroke-width="1.5"/>',
            f'<line x1="{x-7:.1f}" y1="{y_hi:.1f}" x2="{x+7:.1f}" y2="{y_hi:.1f}" stroke="#1d4ed8" stroke-width="1.5"/>',
            f'<line x1="{x-7:.1f}" y1="{y_lo:.1f}" x2="{x+7:.1f}" y2="{y_lo:.1f}" stroke="#1d4ed8" stroke-width="1.5"/>',
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>',
            f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">h{row["hidden"]}</text>',
            f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700">{row["throughput_speedup"]:.2f}x</text>',
        ]
    parts.append("</svg>")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def write_md(path, tag, speedups):
    all_valid = all(row["all_valid"] == "yes" for row in speedups)
    lines = [
        f"# Overlap Speedup by Hidden Size: Job {tag}",
        "",
        f"Overall result: {'PASS' if all_valid else 'FAIL'}",
        "",
        "Configuration: 4 ranks, batch 32, 50 steps, 5 repeats with repeat 1",
        "dropped as warmup. GEMM backend was `auto`. OpenMP overlap used",
        "hidden-size-specific final buckets: h128=256 KB, h256=1024 KB,",
        "h512=2048 KB.",
        "",
        "| Hidden | Blocking M tok/s | OpenMP M tok/s | Speedup | Blocking step ms | OpenMP step ms | Blocking comm ms | OpenMP wait ms | Valid runs |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in speedups:
        lines.append(
            f"| {row['hidden']} | "
            f"{row['blocking_throughput_mean_mtok_s']:.3f} +/- {row['blocking_throughput_std_mtok_s']:.3f} | "
            f"{row['openmp_throughput_mean_mtok_s']:.3f} +/- {row['openmp_throughput_std_mtok_s']:.3f} | "
            f"{row['throughput_speedup']:.3f} +/- {row['throughput_speedup_std']:.3f} | "
            f"{row['blocking_step_time_mean_ms']:.3f} | {row['openmp_step_time_mean_ms']:.3f} | "
            f"{row['blocking_comm_sync_mean_ms']:.3f} | {row['openmp_exposed_wait_mean_ms']:.3f} | "
            f"{row['blocking_valid_runs']}/{row['openmp_valid_runs']} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- This is the clean central overlap comparison: same rank count, batch,",
        "  step count, learning rate, and GEMM policy; only the synchronization",
        "  mechanism changes.",
        "- Speedup is computed from mean throughput after dropping the first repeat.",
        "- The OpenMP wait column is the exposed tail at `finish_async_gradient_syncs`,",
        "  while blocking comm is the direct synchronous gradient-sync timer.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


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
        log_path = LOGS / f"overlap_speedup_by_hidden_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    tag = args.tag or args.job_id or log_path.stem

    rows = parse_log(log_path)
    if args.min_repeat > 1:
        rows = [row for row in rows if row["repeat"] >= args.min_repeat]
    if not rows:
        raise SystemExit(f"no overlap-speedup rows parsed from {log_path}")
    summary = summarize_modes(rows)
    speedups = derive_speedups(summary)
    if not speedups:
        raise SystemExit("could not derive blocking/openmp speedups")

    raw_fields = [
        "hidden", "ranks", "batch", "backend", "sync_mode",
        "requested_sync_mode", "effective_sync", "bucket_kb", "repeat",
        "requested_steps", "lr", "gemm_backend", "world_size", "total_batches",
        "local_per_rank", "used_per_epoch", "dropped_batches",
        "avg_logged_loss", "steps_rank", "time_ms", "step_time_ms",
        "throughput_tok_s", "throughput_mtok_s", "avg_grad_sync_ms",
        "avg_grad_start_ms", "avg_grad_finish_ms", "checksum_span", "valid",
    ]
    summary_fields = [
        "hidden", "ranks", "batch", "backend", "sync_mode",
        "requested_sync_mode", "effective_sync", "bucket_kb", "requested_steps",
        "lr", "gemm_backend", "runs", "valid_runs", "throughput_mean_mtok_s",
        "throughput_median_mtok_s", "throughput_std_mtok_s", "throughput_cv_pct",
        "time_mean_ms", "time_std_ms", "step_time_mean_ms", "step_time_std_ms",
        "avg_grad_sync_mean_ms", "avg_grad_start_mean_ms",
        "avg_grad_finish_mean_ms", "max_checksum_span", "all_valid",
    ]
    speedup_fields = [
        "hidden", "ranks", "batch", "gemm_backend", "blocking_bucket_kb",
        "openmp_bucket_kb", "requested_steps", "lr", "blocking_valid_runs",
        "openmp_valid_runs", "blocking_throughput_mean_mtok_s",
        "blocking_throughput_std_mtok_s", "openmp_throughput_mean_mtok_s",
        "openmp_throughput_std_mtok_s", "throughput_speedup",
        "throughput_speedup_std", "blocking_step_time_mean_ms",
        "openmp_step_time_mean_ms", "step_time_speedup",
        "blocking_comm_sync_mean_ms", "openmp_enqueue_mean_ms",
        "openmp_exposed_wait_mean_ms", "comm_tail_reduction_pct",
        "max_checksum_span", "all_valid",
    ]

    raw_path = RESULTS / f"overlap_speedup_by_hidden_{tag}.csv"
    summary_path = RESULTS / f"overlap_speedup_by_hidden_summary_{tag}.csv"
    speedup_path = RESULTS / f"overlap_speedup_by_hidden_speedup_{tag}.csv"
    md_path = RESULTS / f"overlap_speedup_by_hidden_{tag}.md"
    plot_path = PLOTS / f"overlap_speedup_by_hidden_{tag}.svg"

    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summary, summary_fields)
    write_csv(speedup_path, speedups, speedup_fields)
    write_md(md_path, tag, speedups)
    svg_speedup_plot(plot_path, speedups)

    copy_text(raw_path, RESULTS / "overlap_speedup_by_hidden.csv")
    copy_text(summary_path, RESULTS / "overlap_speedup_by_hidden_summary.csv")
    copy_text(speedup_path, RESULTS / "overlap_speedup_by_hidden_speedup.csv")
    copy_text(md_path, RESULTS / "overlap_speedup_by_hidden.md")
    copy_text(plot_path, PLOTS / "overlap_speedup_by_hidden.svg")

    if not all(row["all_valid"] == "yes" for row in speedups):
        raise SystemExit("one or more overlap-speedup groups had invalid runs")

    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {speedup_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"Wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
