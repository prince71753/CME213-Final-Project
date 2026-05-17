#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


EPOCH_RE = re.compile(
    r"Epoch 1: .*? (?P<ms>[0-9]+)ms\s+(?P<tps>[0-9]+) tok/s"
    r"(?:\s+avg_grad_sync=(?P<sync>[0-9.]+)ms|\s+avg_grad_start=(?P<start>[0-9.]+)ms\s+avg_grad_finish=(?P<finish>[0-9.]+)ms)?"
    r"(?:\s+checksum_span=(?P<checksum>[0-9.eE+-]+|nan))?"
)
PYTORCH_EPOCH_RE = re.compile(
    r"Epoch 1: .*?steps=(?P<steps>[0-9]+)\s+"
    r"(?P<ms>[0-9]+)ms\s+(?P<tps>[0-9]+) tok/s"
)


def read(path):
    return path.read_text() if path.exists() else ""


def parse_sections(path):
    sections = []
    current = None
    body = []
    for line in read(path).splitlines():
        if line.startswith("===") and line.endswith("==="):
            if current is not None:
                sections.append((current, "\n".join(body)))
            current = line.strip("= ").strip()
            body = []
        else:
            body.append(line)
    if current is not None:
        sections.append((current, "\n".join(body)))
    return sections


def parse_epoch(body):
    match = EPOCH_RE.search(body)
    if not match:
        return None
    sync = match.group("sync")
    start = match.group("start")
    finish = match.group("finish")
    checksum = match.group("checksum")
    return {
        "time_ms": float(match.group("ms")),
        "throughput_tok_s": float(match.group("tps")),
        "throughput_mtok_s": float(match.group("tps")) / 1e6,
        "avg_grad_sync_ms": float(sync) if sync else "",
        "avg_grad_start_ms": float(start) if start else "",
        "avg_grad_finish_ms": float(finish) if finish else "",
        "checksum_span": checksum if checksum else "",
    }


def parse_pytorch_epoch(body):
    match = PYTORCH_EPOCH_RE.search(body)
    if not match:
        return None
    return {
        "steps": int(match.group("steps")),
        "time_ms": float(match.group("ms")),
        "throughput_tok_s": float(match.group("tps")),
        "throughput_mtok_s": float(match.group("tps")) / 1e6,
    }


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_bar(path, title, labels, values, ylabel, color="#3b82f6"):
    width, height = 760, 430
    left, right, top, bottom = 80, 30, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    vmax = max(values) * 1.15 if values else 1.0
    bar_gap = 18
    bar_w = (plot_w - bar_gap * (len(values) + 1)) / max(1, len(values))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(5):
        val = vmax * tick / 4
        y = top + plot_h - (val / vmax) * plot_h
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.2f}</text>')
    for i, (label, val) in enumerate(zip(labels, values)):
        x = left + bar_gap + i * (bar_w + bar_gap)
        h = (val / vmax) * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{val:.2f}</text>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def svg_grouped_bar(path, title, labels, series, ylabel):
    colors = ["#2563eb", "#dc2626", "#059669"]
    width, height = 820, 460
    left, right, top, bottom = 80, 30, 60, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_vals = [v for _, vals in series for v in vals]
    vmax = max(all_vals) * 1.15 if all_vals else 1.0
    group_gap = 30
    group_w = (plot_w - group_gap * (len(labels) + 1)) / max(1, len(labels))
    bar_w = group_w / max(1, len(series))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in range(5):
        val = vmax * tick / 4
        y = top + plot_h - (val / vmax) * plot_h
        parts.append(f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{val:.2f}</text>')
    for gi, label in enumerate(labels):
        gx = left + group_gap + gi * (group_w + group_gap)
        for si, (_, vals) in enumerate(series):
            val = vals[gi]
            h = (val / vmax) * plot_h
            x = gx + si * bar_w
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{colors[si % len(colors)]}"/>')
            parts.append(f'<text x="{x+(bar_w-2)/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{val:.2f}</text>')
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
    lx = left + 10
    for si, (name, _) in enumerate(series):
        ly = height - 34 + si * 18
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{colors[si % len(colors)]}"/>')
        parts.append(f'<text x="{lx+18}" y="{ly}" font-family="Arial" font-size="12">{name}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def main():
    RESULTS.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)

    bucket_rows = []
    for title, body in parse_sections(LOGS / "bucket_sweep_81864.out"):
        ep = parse_epoch(body)
        if not ep:
            continue
        if "Blocking" in title:
            label = "blocking"
            bucket = ""
        else:
            m = re.search(r"bucket_kb=([0-9]+)", title)
            label = f"{m.group(1)} KB" if m else title
            bucket = m.group(1) if m else ""
        bucket_rows.append({"run": label, "bucket_kb": bucket, **ep})

    write_csv(
        RESULTS / "bucket_sweep_default.csv",
        ["run", "bucket_kb", "time_ms", "throughput_tok_s", "throughput_mtok_s",
         "avg_grad_sync_ms", "avg_grad_start_ms", "avg_grad_finish_ms", "checksum_span"],
        bucket_rows,
    )
    if bucket_rows:
        svg_bar(
            PLOTS / "bucket_sweep_default.svg",
            "Default Model: Bucketed Overlap Sweep",
            [r["run"] for r in bucket_rows],
            [r["throughput_mtok_s"] for r in bucket_rows],
            "Throughput (M tokens/s)",
        )

    size_rows = []
    for log_name in ["model_size_sweep_81865.out", "model_size_sweep_81867.out"]:
        for title, body in parse_sections(LOGS / log_name):
            ep = parse_epoch(body)
            if not ep or ep["checksum_span"] == "nan":
                continue
            m = re.search(r"hidden=([0-9]+)", title)
            if not m:
                continue
            hidden = int(m.group(1))
            run = "overlap" if "pinned" in title else "blocking"
            bucket = ""
            bm = re.search(r"bucket_kb=([0-9]+)", title)
            if bm:
                bucket = bm.group(1)
            size_rows.append({"hidden": hidden, "run": run, "bucket_kb": bucket, **ep})

    # Keep one blocking and one 256KB overlap row per hidden size.
    dedup = {}
    for row in size_rows:
        key = (row["hidden"], row["run"])
        if key not in dedup:
            dedup[key] = row
    size_rows = [dedup[k] for k in sorted(dedup)]

    write_csv(
        RESULTS / "model_size_sweep.csv",
        ["hidden", "run", "bucket_kb", "time_ms", "throughput_tok_s", "throughput_mtok_s",
         "avg_grad_sync_ms", "avg_grad_start_ms", "avg_grad_finish_ms", "checksum_span"],
        size_rows,
    )
    hidden_values = sorted({r["hidden"] for r in size_rows})
    if hidden_values:
        blocking = [next(r["throughput_mtok_s"] for r in size_rows if r["hidden"] == h and r["run"] == "blocking") for h in hidden_values]
        overlap = [next(r["throughput_mtok_s"] for r in size_rows if r["hidden"] == h and r["run"] == "overlap") for h in hidden_values]
        svg_grouped_bar(
            PLOTS / "model_size_sweep.svg",
            "Model Size Sensitivity: Blocking vs Overlap",
            [str(h) for h in hidden_values],
            [("blocking", blocking), ("overlap", overlap)],
            "Throughput (M tokens/s)",
        )

    fusion_rows = []
    fusion_text = read(LOGS / "full_validation_81876.out")
    current = None
    for line in fusion_text.splitlines():
        if "bias+ReLU" in line:
            current = "bias+ReLU"
        elif "residual+LayerNorm fwd" in line:
            current = "residual+LayerNorm fwd"
        elif "LayerNorm bwd+residual" in line:
            current = "LayerNorm bwd+residual"
        elif current and "unfused" in line and "fused" in line and "speedup" in line:
            m = re.search(r"unfused ([0-9.]+) ms \| fused ([0-9.]+) ms \| speedup ([0-9.]+)x", line)
            if m:
                fusion_rows.append({
                    "kernel": current,
                    "unfused_ms": float(m.group(1)),
                    "fused_ms": float(m.group(2)),
                    "speedup": float(m.group(3)),
                })
            current = None
    write_csv(RESULTS / "fusion_benchmark.csv",
              ["kernel", "unfused_ms", "fused_ms", "speedup"], fusion_rows)
    if fusion_rows:
        svg_bar(
            PLOTS / "fusion_speedups.svg",
            "Fusion Microbenchmark Speedups",
            [r["kernel"] for r in fusion_rows],
            [r["speedup"] for r in fusion_rows],
            "Speedup vs unfused",
            "#059669",
        )

    framework_rows = []
    for title, body in parse_sections(LOGS / "custom_single_baseline_81878.out"):
        if "Custom CUDA baseline" not in title:
            continue
        ep = parse_epoch(body)
        hm = re.search(r"hidden=([0-9]+)", body)
        pm = re.search(r"Parameters:\s+([0-9]+)", body)
        sm = re.search(r"steps/rank=([0-9]+)", body)
        if ep and hm:
            framework_rows.append({
                "hidden": int(hm.group(1)),
                "implementation": "Custom CUDA",
                "params": int(pm.group(1)) if pm else "",
                "steps": int(sm.group(1)) if sm else "",
                "time_ms": ep["time_ms"],
                "throughput_tok_s": ep["throughput_tok_s"],
                "throughput_mtok_s": ep["throughput_mtok_s"],
            })
    for title, body in parse_sections(LOGS / "pytorch_baseline_81877.out"):
        if "PyTorch baseline" not in title:
            continue
        ep = parse_pytorch_epoch(body)
        hm = re.search(r"hidden=([0-9]+)", body)
        pm = re.search(r"Parameters:\s+([0-9,]+)", body)
        if ep and hm:
            framework_rows.append({
                "hidden": int(hm.group(1)),
                "implementation": "PyTorch",
                "params": int(pm.group(1).replace(",", "")) if pm else "",
                "steps": ep["steps"],
                "time_ms": ep["time_ms"],
                "throughput_tok_s": ep["throughput_tok_s"],
                "throughput_mtok_s": ep["throughput_mtok_s"],
            })
    framework_rows = sorted(framework_rows,
                            key=lambda r: (r["hidden"], r["implementation"]))
    write_csv(
        RESULTS / "framework_baseline.csv",
        ["hidden", "implementation", "params", "steps", "time_ms",
         "throughput_tok_s", "throughput_mtok_s"],
        framework_rows,
    )
    hidden_values = sorted({r["hidden"] for r in framework_rows})
    if hidden_values and all(
        any(r["hidden"] == h and r["implementation"] == impl
            for r in framework_rows)
        for h in hidden_values for impl in ["Custom CUDA", "PyTorch"]
    ):
        custom = [
            next(r["throughput_mtok_s"] for r in framework_rows
                 if r["hidden"] == h and r["implementation"] == "Custom CUDA")
            for h in hidden_values
        ]
        pytorch = [
            next(r["throughput_mtok_s"] for r in framework_rows
                 if r["hidden"] == h and r["implementation"] == "PyTorch")
            for h in hidden_values
        ]
        svg_grouped_bar(
            PLOTS / "framework_baseline.svg",
            "Single-GPU Framework Baseline",
            [str(h) for h in hidden_values],
            [("Custom CUDA", custom), ("PyTorch", pytorch)],
            "Throughput (M tokens/s)",
        )

    readme = [
        "# Generated Results",
        "",
        "Generated by `scripts/generate_results.py` from Slurm logs.",
        "",
        "CSV files:",
        "- `bucket_sweep_default.csv`",
        "- `model_size_sweep.csv`",
        "- `fusion_benchmark.csv`",
        "- `framework_baseline.csv`",
        "- `fusion_profile.csv`",
        "- `fusion_profile_case_83615.csv`",
        "- `nsys_overlap_h256_83616_rank0.csv`",
        "- `adaptive_sync_sweep.csv`",
        "- `adaptive_sync_sweep_83627.csv`",
        "- `h512_sync_stability_83631.csv`",
        "- `final_repeated_sync.csv`",
        "- `final_repeated_sync_summary.csv`",
        "- `final_main_results.csv`",
        "- `gradient_sync_bench.csv`",
        "- `gradient_sync_bench_83641.csv`",
        "- `strong_scaling.csv`",
        "- `strong_scaling_83646.csv`",
        "- `h512_lr_sweep.csv`",
        "- `h512_lr_sweep_83724.csv`",
        "- `h512_lr_sweep_summary.csv`",
        "- `h512_lr_sweep_summary_83724.csv`",
        "- `roofline_fusion_83615.csv`",
        "- `ln_bwd_diagnostic_83615.csv`",
        "- `nsys_top_kernels_83616_rank0.csv`",
        "- `hotspot_profile.csv`",
        "- `splitk_tuning.csv`",
        "- `splitk_tuning_summary.csv`",
        "- `final_repeated_sync_summary_83739.csv`",
        "- `roofline_combined.csv`",
        "",
        "SVG plots:",
        "- `plots/bucket_sweep_default.svg`",
        "- `plots/model_size_sweep.svg`",
        "- `plots/fusion_speedups.svg`",
        "- `plots/framework_baseline.svg`",
        "- `plots/roofline_fusion_83615.svg`",
        "- `plots/roofline_combined.svg`",
        "- `plots/strong_scaling_speedup.svg`",
        "- `plots/strong_scaling_efficiency.svg`",
        "",
        "Notes:",
        "- Hidden-512 default learning-rate NaN runs are omitted from `model_size_sweep.csv`.",
        "- Nsight profiling runs are documented in `DISTRIBUTED_PROGRESS.md`; they are not plotted here because profiler overhead changes end-to-end timing.",
        "- Adaptive-sync rows include a `valid` column. Do not use rows with NaN loss or NaN checksum as performance wins.",
        "- `h512_lr_sweep_summary.csv` is a follow-up stability check; it does not replace the valid-only repeated sync table.",
        "- `splitk_tuning.csv` records the H256 split-K tuning pass; the final code uses `SPLITK_TARGET_BLOCKS=216`.",
        "- `roofline_combined.csv` adds training GEMM hotspots to the fusion roofline.",
        "",
    ]
    (RESULTS / "README.md").write_text("\n".join(readme))

    print(f"Wrote {RESULTS.relative_to(ROOT)} and {PLOTS.relative_to(ROOT)} artifacts.")


if __name__ == "__main__":
    main()
