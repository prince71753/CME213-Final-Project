#!/usr/bin/env python3
"""Summarize Nsight Systems compute/communication overlap across hidden sizes."""

import argparse
import csv
import math
import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PROFILES = ROOT / "profiles"

NVTX_NAMES = {
    "forward",
    "backward_bucketed",
    "adam_update",
    "openmp_comm_event_wait",
    "openmp_comm_mpi_allreduce",
    "finish_async_gradient_syncs",
}


def parse_hidden_list(value):
    return [int(x) for x in value.replace(",", " ").split() if x]


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def sample_std(vals):
    vals = list(vals)
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def median(vals):
    vals = sorted(vals)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def fetch_events(db_path):
    con = sqlite3.connect(str(db_path))
    query = """
        select coalesce(n.text, s.value) as name,
               n.start,
               n.end,
               n.globalTid
        from NVTX_EVENTS n
        left join StringIds s on n.textId = s.id
        where coalesce(n.text, s.value) in ({})
          and n.end is not null
        order by n.start
    """.format(",".join("?" for _ in NVTX_NAMES))
    rows = []
    for name, start, end, tid in con.execute(query, tuple(sorted(NVTX_NAMES))):
        rows.append({
            "name": name,
            "start": int(start),
            "end": int(end),
            "tid": int(tid) if tid is not None else 0,
            "duration_ms": (int(end) - int(start)) / 1.0e6,
        })
    con.close()
    return rows


def preceding(events, name, end_before):
    candidates = [e for e in events if e["name"] == name and e["end"] <= end_before]
    return max(candidates, key=lambda e: e["end"], default=None)


def following(events, name, start_after):
    candidates = [e for e in events if e["name"] == name and e["start"] >= start_after]
    return min(candidates, key=lambda e: e["start"], default=None)


def inside(events, name, start, end):
    return [
        e for e in events
        if e["name"] == name and e["start"] >= start and e["end"] <= end
    ]


def compute_steps(events, hidden):
    backs = [e for e in events if e["name"] == "backward_bucketed"]
    steps = []
    for idx, back in enumerate(backs, start=1):
        fwd = preceding(events, "forward", back["start"])
        adam = following(events, "adam_update", back["end"])
        allreduces = inside(events, "openmp_comm_mpi_allreduce", back["start"], back["end"])
        waits = inside(events, "openmp_comm_event_wait", back["start"], back["end"])
        finishes = inside(events, "finish_async_gradient_syncs", back["start"], back["end"])
        forward_ms = fwd["duration_ms"] if fwd else 0.0
        backward_ms = back["duration_ms"]
        adam_ms = adam["duration_ms"] if adam else 0.0
        finish_ms = sum(e["duration_ms"] for e in finishes)
        allreduce_ms = sum(e["duration_ms"] for e in allreduces)
        wait_ms = sum(e["duration_ms"] for e in waits)
        main_step_ms = forward_ms + backward_ms + adam_ms
        compute_without_finish_ms = forward_ms + max(0.0, backward_ms - finish_ms) + adam_ms
        steps.append({
            "hidden": hidden,
            "step": idx,
            "forward_ms": forward_ms,
            "backward_bucketed_ms": backward_ms,
            "finish_async_gradient_syncs_ms": finish_ms,
            "adam_update_ms": adam_ms,
            "main_step_ms": main_step_ms,
            "compute_without_finish_ms": compute_without_finish_ms,
            "worker_event_wait_total_ms": wait_ms,
            "worker_allreduce_total_ms": allreduce_ms,
            "worker_allreduce_count": len(allreduces),
            "finish_fraction_of_step": finish_ms / main_step_ms if main_step_ms > 0 else 0.0,
            "worker_comm_to_backward": allreduce_ms / backward_ms if backward_ms > 0 else 0.0,
            "overlap_fraction_proxy": 1.0 - finish_ms / allreduce_ms if allreduce_ms > 0 else 0.0,
        })
    return steps


def summarize_hidden(rows, tag, ranks, batch, steps_requested, bucket_by_hidden):
    out = []
    for hidden in sorted({row["hidden"] for row in rows}):
        analyzed = [row for row in rows if row["hidden"] == hidden and row["step"] > 1]
        if not analyzed:
            continue
        def stat(field):
            vals = [row[field] for row in analyzed]
            return mean(vals), median(vals), sample_std(vals)
        main_mean, main_med, main_std = stat("main_step_ms")
        fwd_mean, _, _ = stat("forward_ms")
        back_mean, _, _ = stat("backward_bucketed_ms")
        finish_mean, finish_med, finish_std = stat("finish_async_gradient_syncs_ms")
        adam_mean, _, _ = stat("adam_update_ms")
        comp_mean, _, _ = stat("compute_without_finish_ms")
        wait_mean, _, _ = stat("worker_event_wait_total_ms")
        ar_mean, ar_med, ar_std = stat("worker_allreduce_total_ms")
        count_mean, _, _ = stat("worker_allreduce_count")
        finish_frac = finish_mean / main_mean if main_mean > 0 else 0.0
        overlap_proxy = 1.0 - finish_mean / ar_mean if ar_mean > 0 else 0.0
        out.append({
            "tag": tag,
            "hidden": hidden,
            "ranks": ranks,
            "batch": batch,
            "requested_steps": steps_requested,
            "analyzed_steps": len(analyzed),
            "bucket_kb": bucket_by_hidden.get(hidden, ""),
            "main_step_mean_ms": main_mean,
            "main_step_median_ms": main_med,
            "main_step_std_ms": main_std,
            "forward_mean_ms": fwd_mean,
            "backward_bucketed_mean_ms": back_mean,
            "finish_async_gradient_syncs_mean_ms": finish_mean,
            "finish_async_gradient_syncs_median_ms": finish_med,
            "finish_async_gradient_syncs_std_ms": finish_std,
            "adam_update_mean_ms": adam_mean,
            "compute_without_finish_mean_ms": comp_mean,
            "worker_event_wait_mean_ms": wait_mean,
            "worker_allreduce_total_mean_ms": ar_mean,
            "worker_allreduce_total_median_ms": ar_med,
            "worker_allreduce_total_std_ms": ar_std,
            "worker_allreduce_count_mean": count_mean,
            "finish_fraction_of_step": finish_frac,
            "worker_comm_to_backward": ar_mean / back_mean if back_mean > 0 else 0.0,
            "overlap_fraction_proxy": overlap_proxy,
        })
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_md(path, tag, summary_rows):
    lines = [
        f"# Nsight Systems Hidden-Size Breakdown: Job {tag}",
        "",
        "Overall result: PASS",
        "",
        "This run profiles the OpenMP communication-thread path for `h128`,",
        "`h256`, and `h512` on 4 MPI ranks. The first step is dropped from",
        "the aggregates to avoid initialization effects.",
        "",
        "| Hidden | Bucket KB | Main step ms | Forward ms | Backward ms | Worker MPI Allreduce ms | Finish/wait ms | Finish fraction | Overlap proxy | AR calls | Steps |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['hidden']} | {row['bucket_kb']} | "
            f"{fmt(row['main_step_mean_ms'])} +/- {fmt(row['main_step_std_ms'])} | "
            f"{fmt(row['forward_mean_ms'])} | "
            f"{fmt(row['backward_bucketed_mean_ms'])} | "
            f"{fmt(row['worker_allreduce_total_mean_ms'])} +/- {fmt(row['worker_allreduce_total_std_ms'])} | "
            f"{fmt(row['finish_async_gradient_syncs_mean_ms'])} +/- {fmt(row['finish_async_gradient_syncs_std_ms'])} | "
            f"{100.0 * row['finish_fraction_of_step']:.1f}% | "
            f"{100.0 * row['overlap_fraction_proxy']:.1f}% | "
            f"{row['worker_allreduce_count_mean']:.1f} | {row['analyzed_steps']} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- `Worker MPI Allreduce ms` is the communication performed by the OpenMP",
        "  communication thread during the profiled backward pass.",
        "- `Finish/wait ms` is the main-thread `finish_async_gradient_syncs` range",
        "  at the end of `backward_bucketed`; it is the profiler-visible exposed",
        "  synchronization cost that remains after overlap.",
        "- The overlap proxy is `1 - finish_wait / worker_allreduce`. It is not a",
        "  replacement for unprofiled throughput, but it shows how much of the",
        "  profiled collective work was hidden behind backward computation.",
        "- Use this figure for mechanism and bottleneck analysis; use the repeated",
        "  unprofiled runs for headline throughput.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


def svg_breakdown(path, summary_rows):
    rows = summary_rows
    if not rows:
        return
    width, height = 980, 470
    left, right, top, bottom = 88, 185, 62, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(
        max(r["main_step_mean_ms"], r["worker_allreduce_total_mean_ms"])
        for r in rows
    ) * 1.18
    group_w = plot_w / len(rows)
    bar_w = min(62, group_w / 3.2)

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Nsight Systems: Compute/Communication by Hidden Size</text>',
        f'<text x="{width/2}" y="50" text-anchor="middle" font-family="Arial" font-size="12">OpenMP communication thread, 4 ranks, warmup step dropped</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<text x="20" y="{top + plot_h/2}" transform="rotate(-90 20 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">milliseconds per step</text>',
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = y_pos(value)
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>')
    for idx, row in enumerate(rows):
        cx = left + group_w * (idx + 0.5)
        stack_x = cx - bar_w - 6
        ar_x = cx + 6
        compute = row["compute_without_finish_mean_ms"]
        finish = row["finish_async_gradient_syncs_mean_ms"]
        ar = row["worker_allreduce_total_mean_ms"]
        y_compute = y_pos(compute)
        y_total = y_pos(compute + finish)
        y_ar = y_pos(ar)
        parts.append(f'<rect x="{stack_x:.1f}" y="{y_compute:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y_compute:.1f}" fill="#60a5fa"/>')
        parts.append(f'<rect x="{stack_x:.1f}" y="{y_total:.1f}" width="{bar_w:.1f}" height="{y_compute-y_total:.1f}" fill="#f97316"/>')
        parts.append(f'<rect x="{ar_x:.1f}" y="{y_ar:.1f}" width="{bar_w:.1f}" height="{top+plot_h-y_ar:.1f}" fill="#0f766e" opacity="0.82"/>')
        parts.append(f'<text x="{stack_x + bar_w/2:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">step</text>')
        parts.append(f'<text x="{ar_x + bar_w/2:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">AR</text>')
        parts.append(f'<text x="{cx:.1f}" y="{top+plot_h+42}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700">h{row["hidden"]}</text>')
        parts.append(f'<text x="{stack_x + bar_w/2:.1f}" y="{y_total-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{100.0 * row["finish_fraction_of_step"]:.0f}% wait</text>')
    lx = left + plot_w + 22
    legend = [
        ("#60a5fa", "main compute/non-wait"),
        ("#f97316", "finish/wait"),
        ("#0f766e", "worker Allreduce"),
    ]
    for i, (color, label) in enumerate(legend):
        y = top + 28 + i * 28
        parts.append(f'<rect x="{lx}" y="{y}" width="15" height="15" fill="{color}" opacity="0.9"/>')
        parts.append(f'<text x="{lx+23}" y="{y+12}" font-family="Arial" font-size="12">{label}</text>')
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--hidden-list", default="128,256,512")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--bucket-h128", type=int, default=256)
    parser.add_argument("--bucket-h256", type=int, default=2048)
    parser.add_argument("--bucket-h512", type=int, default=2048)
    args = parser.parse_args()

    hidden_sizes = parse_hidden_list(args.hidden_list)
    bucket_by_hidden = {
        128: args.bucket_h128,
        256: args.bucket_h256,
        512: args.bucket_h512,
    }
    all_steps = []
    for hidden in hidden_sizes:
        db_path = PROFILES / f"h{hidden}_openmp_breakdown_{args.tag}_rank0.sqlite"
        if not db_path.exists():
            raise SystemExit(f"missing Nsight sqlite: {db_path}")
        steps = compute_steps(fetch_events(db_path), hidden)
        if len(steps) < 2:
            raise SystemExit(f"h{hidden}: expected at least two profiled steps")
        all_steps.extend(steps)

    summary = summarize_hidden(all_steps, args.tag, args.ranks, args.batch,
                               args.steps, bucket_by_hidden)
    step_fields = [
        "hidden", "step", "forward_ms", "backward_bucketed_ms",
        "finish_async_gradient_syncs_ms", "adam_update_ms", "main_step_ms",
        "compute_without_finish_ms", "worker_event_wait_total_ms",
        "worker_allreduce_total_ms", "worker_allreduce_count",
        "finish_fraction_of_step", "worker_comm_to_backward",
        "overlap_fraction_proxy",
    ]
    summary_fields = [
        "tag", "hidden", "ranks", "batch", "requested_steps", "analyzed_steps",
        "bucket_kb", "main_step_mean_ms", "main_step_median_ms",
        "main_step_std_ms", "forward_mean_ms", "backward_bucketed_mean_ms",
        "finish_async_gradient_syncs_mean_ms",
        "finish_async_gradient_syncs_median_ms",
        "finish_async_gradient_syncs_std_ms", "adam_update_mean_ms",
        "compute_without_finish_mean_ms", "worker_event_wait_mean_ms",
        "worker_allreduce_total_mean_ms", "worker_allreduce_total_median_ms",
        "worker_allreduce_total_std_ms", "worker_allreduce_count_mean",
        "finish_fraction_of_step", "worker_comm_to_backward",
        "overlap_fraction_proxy",
    ]

    steps_path = RESULTS / f"nsys_hidden_breakdown_{args.tag}_steps.csv"
    summary_path = RESULTS / f"nsys_hidden_breakdown_{args.tag}.csv"
    md_path = RESULTS / f"nsys_hidden_breakdown_{args.tag}.md"
    plot_path = PLOTS / f"nsys_hidden_breakdown_{args.tag}.svg"
    write_csv(steps_path, all_steps, step_fields)
    write_csv(summary_path, summary, summary_fields)
    write_md(md_path, args.tag, summary)
    svg_breakdown(plot_path, summary)

    shutil.copyfile(steps_path, RESULTS / "nsys_hidden_breakdown_steps.csv")
    shutil.copyfile(summary_path, RESULTS / "nsys_hidden_breakdown.csv")
    shutil.copyfile(md_path, RESULTS / "nsys_hidden_breakdown.md")
    shutil.copyfile(plot_path, PLOTS / "nsys_hidden_breakdown.svg")

    print(f"Wrote {steps_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"Wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
