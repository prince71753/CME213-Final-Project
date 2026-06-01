#!/usr/bin/env python3
"""Build report-facing end-to-end cublasLt fusion ablation artifacts."""

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


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


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize(rows):
    by_key = {(int(row["hidden"]), row["backend"]): row for row in rows}
    hidden_sizes = sorted({hidden for hidden, _ in by_key})
    out = []
    for hidden in hidden_sizes:
        base = by_key.get((hidden, "cublas_tc"))
        fused = by_key.get((hidden, "cublas_tc_lt"))
        if not base or not fused:
            raise SystemExit(f"missing cublas_tc/cublas_tc_lt pair for h{hidden}")
        base_tput = float(base["throughput_mean_mtok_s"])
        fused_tput = float(fused["throughput_mean_mtok_s"])
        base_std = float(base["throughput_std_mtok_s"])
        fused_std = float(fused["throughput_std_mtok_s"])
        base_time = float(base["time_mean_ms"])
        fused_time = float(fused["time_mean_ms"])
        speedup = fused_tput / base_tput if base_tput > 0.0 else float("nan")
        slowdown_pct = 100.0 * (fused_time / base_time - 1.0) if base_time > 0.0 else float("nan")
        if base_tput > 0.0 and fused_tput > 0.0:
            rel = math.sqrt((base_std / base_tput) ** 2 + (fused_std / fused_tput) ** 2)
            speedup_std = speedup * rel
        else:
            speedup_std = float("nan")
        out.append({
            "hidden": hidden,
            "baseline_backend": "cublas_tc",
            "fusion_backend": "cublas_tc_lt",
            "steps": int(base["requested_steps"]),
            "lr": base["lr"],
            "baseline_runs": int(base["runs"]),
            "baseline_valid_runs": int(base["valid_runs"]),
            "fusion_runs": int(fused["runs"]),
            "fusion_valid_runs": int(fused["valid_runs"]),
            "baseline_throughput_mean_mtok_s": base_tput,
            "baseline_throughput_std_mtok_s": base_std,
            "fusion_throughput_mean_mtok_s": fused_tput,
            "fusion_throughput_std_mtok_s": fused_std,
            "throughput_speedup": speedup,
            "throughput_speedup_std": speedup_std,
            "baseline_time_mean_ms": base_time,
            "fusion_time_mean_ms": fused_time,
            "fusion_time_slowdown_pct": slowdown_pct,
            "baseline_loss_mean": float(base["final_loss_mean"]),
            "fusion_loss_mean": float(fused["final_loss_mean"]),
            "loss_delta": float(fused["final_loss_mean"]) - float(base["final_loss_mean"]),
            "all_valid": "yes" if base["all_valid"] == "yes" and fused["all_valid"] == "yes" else "no",
        })
    return out


def svg_plot(path, rows):
    width, height = 760, 430
    left, right, top, bottom = 78, 32, 52, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(max(row["baseline_throughput_mean_mtok_s"] + row["baseline_throughput_std_mtok_s"],
                   row["fusion_throughput_mean_mtok_s"] + row["fusion_throughput_std_mtok_s"])
               for row in rows) * 1.18
    group_w = plot_w / len(rows)
    bar_w = min(58, group_w * 0.28)
    colors = {"baseline": "#6b7280", "fusion": "#2563eb"}

    def y_pos(value):
        return top + plot_h - (value / ymax) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">End-to-End Fusion Ablation</text>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="13">Hidden size</text>',
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
    for i, row in enumerate(rows):
        center = left + group_w * (i + 0.5)
        vals = [
            ("baseline", row["baseline_throughput_mean_mtok_s"], row["baseline_throughput_std_mtok_s"]),
            ("fusion", row["fusion_throughput_mean_mtok_s"], row["fusion_throughput_std_mtok_s"]),
        ]
        for j, (kind, mean, std) in enumerate(vals):
            x = center + (j - 0.5) * (bar_w + 10)
            y = y_pos(mean)
            h = top + plot_h - y
            parts.append(f'<rect x="{x - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[kind]}"/>')
            y_hi = y_pos(mean + std)
            y_lo = y_pos(max(0.0, mean - std))
            parts += [
                f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="#111827" stroke-width="1.2"/>',
                f'<line x1="{x-7:.1f}" y1="{y_hi:.1f}" x2="{x+7:.1f}" y2="{y_hi:.1f}" stroke="#111827" stroke-width="1.2"/>',
                f'<line x1="{x-7:.1f}" y1="{y_lo:.1f}" x2="{x+7:.1f}" y2="{y_lo:.1f}" stroke="#111827" stroke-width="1.2"/>',
            ]
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+25}" text-anchor="middle" font-family="Arial" font-size="12">h{row["hidden"]}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+43}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">{row["throughput_speedup"]:.3f}x</text>')

    lx, ly = left + plot_w - 185, top + 12
    parts += [
        f'<rect x="{lx}" y="{ly}" width="14" height="14" fill="{colors["baseline"]}"/>',
        f'<text x="{lx+22}" y="{ly+12}" font-family="Arial" font-size="12">cuBLAS TC</text>',
        f'<rect x="{lx}" y="{ly+24}" width="14" height="14" fill="{colors["fusion"]}"/>',
        f'<text x="{lx+22}" y="{ly+36}" font-family="Arial" font-size="12">cuBLASLt fused FFN</text>',
    ]
    parts.append("</svg>\n")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(parts))


def write_md(path, tag, rows):
    all_valid = all(row["all_valid"] == "yes" for row in rows)
    lines = [
        f"# End-to-End Fusion Ablation: Job {tag}",
        "",
        f"Overall result: {'PASS' if all_valid else 'FAIL'}",
        "",
        "This compares the normal `cublas_tc` path against `cublas_tc_lt`,",
        "which enables the optional cuBLASLt FFN bias+ReLU fusion. Repeat 1",
        "is dropped as warmup. The always-on custom residual/LayerNorm fusion",
        "kernels remain enabled in both variants.",
        "",
        "| Hidden | Steps | cuBLAS TC M tok/s | cuBLASLt fusion M tok/s | Speedup | Time change | Valid |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['hidden']} | {row['steps']} | "
            f"{row['baseline_throughput_mean_mtok_s']:.3f} +/- {row['baseline_throughput_std_mtok_s']:.3f} | "
            f"{row['fusion_throughput_mean_mtok_s']:.3f} +/- {row['fusion_throughput_std_mtok_s']:.3f} | "
            f"{row['throughput_speedup']:.3f} +/- {row['throughput_speedup_std']:.3f} | "
            f"{row['fusion_time_slowdown_pct']:+.1f}% | "
            f"{row['baseline_valid_runs']}/{row['baseline_runs']}, {row['fusion_valid_runs']}/{row['fusion_runs']} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- This is an end-to-end training-loop ablation, not a microbenchmark of",
        "  the fused kernels alone.",
        "- If speedup is near 1.0, the report should say fusion helps local",
        "  memory-bound kernels but is limited by surrounding GEMM, attention,",
        "  launch, and communication costs.",
        "- If the cuBLASLt path is slower at a hidden size, report it as a useful",
        "  negative result rather than hiding it.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    rows = summarize(read_csv(Path(args.summary)))
    fields = [
        "hidden", "baseline_backend", "fusion_backend", "steps", "lr",
        "baseline_runs", "baseline_valid_runs", "fusion_runs",
        "fusion_valid_runs", "baseline_throughput_mean_mtok_s",
        "baseline_throughput_std_mtok_s", "fusion_throughput_mean_mtok_s",
        "fusion_throughput_std_mtok_s", "throughput_speedup",
        "throughput_speedup_std", "baseline_time_mean_ms",
        "fusion_time_mean_ms", "fusion_time_slowdown_pct",
        "baseline_loss_mean", "fusion_loss_mean", "loss_delta", "all_valid",
    ]
    csv_path = RESULTS / f"fusion_ablation_{args.tag}.csv"
    md_path = RESULTS / f"fusion_ablation_{args.tag}.md"
    svg_path = PLOTS / f"fusion_ablation_{args.tag}.svg"
    write_csv(csv_path, rows, fields)
    write_md(md_path, args.tag, rows)
    svg_plot(svg_path, rows)
    (RESULTS / "fusion_ablation.csv").write_text(csv_path.read_text())
    (RESULTS / "fusion_ablation.md").write_text(md_path.read_text())
    (PLOTS / "fusion_ablation.svg").write_text(svg_path.read_text())
    if not all(row["all_valid"] == "yes" for row in rows):
        raise SystemExit("one or more fusion ablation groups had invalid runs")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")
    print(f"Wrote {svg_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
