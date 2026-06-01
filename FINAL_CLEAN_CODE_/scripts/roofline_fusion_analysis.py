#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


CASE_SHAPES = {
    "bias_relu_unfused": (2048, 512),
    "bias_relu_fused": (2048, 512),
    "residual_ln_unfused": (2048, 128),
    "residual_ln_fused": (2048, 128),
    "ln_bwd_residual_unfused": (2048, 128),
    "ln_bwd_residual_fused": (2048, 128),
}


def approx_flops(case, rows, cols):
    n = rows * cols
    if case.startswith("bias_relu"):
        return float(n)
    if case.startswith("residual_ln"):
        return float(9 * n)
    if case == "ln_bwd_residual_unfused":
        return float(18 * n)
    if case == "ln_bwd_residual_fused":
        return float(17 * n)
    return 0.0


def read_rows(path, peak_bw_gbs, peak_fp32_gflops):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if {"runtime_us", "dram_read_bytes", "dram_write_bytes"}.issubset(fieldnames):
            grouped = {}
            for row in reader:
                case = row["case"]
                shape = CASE_SHAPES.get(case)
                if not shape:
                    continue
                entry = grouped.setdefault(case, {
                    "runtime_us": 0.0,
                    "dram_bytes": 0.0,
                    "bottlenecks": set(),
                })
                entry["runtime_us"] += float(row["runtime_us"] or 0.0)
                entry["dram_bytes"] += float(row["dram_read_bytes"] or 0.0) + float(row["dram_write_bytes"] or 0.0)
                if row.get("observed_bottleneck"):
                    entry["bottlenecks"].add(row["observed_bottleneck"])
            source_rows = []
            for case, entry in grouped.items():
                source_rows.append({
                    "case": case,
                    "ncu_kernel_runtime_us": entry["runtime_us"],
                    "dram_total_bytes": entry["dram_bytes"],
                    "observed_bottlenecks": ";".join(sorted(entry["bottlenecks"])),
                })
        else:
            source_rows = []
            for row in reader:
                source_rows.append({
                    "case": row["case"],
                    "ncu_kernel_runtime_us": row["ncu_kernel_runtime_us"],
                    "dram_total_bytes": row["dram_total_bytes"],
                    "observed_bottlenecks": row.get("observed_bottlenecks", row.get("observed_bottleneck", "")),
                })

        for row in source_rows:
            case = row["case"]
            shape = CASE_SHAPES.get(case)
            if not shape:
                continue
            r, c = shape
            runtime_us = float(row["ncu_kernel_runtime_us"])
            dram_bytes = float(row["dram_total_bytes"])
            flops = approx_flops(case, r, c)
            seconds = runtime_us * 1e-6
            bw_gbs = dram_bytes / seconds / 1e9
            perf_gflops = flops / seconds / 1e9
            ai = flops / dram_bytes if dram_bytes > 0 else 0.0
            roof_gflops = min(peak_fp32_gflops, peak_bw_gbs * ai)
            rows.append({
                "case": case,
                "rows": r,
                "cols": c,
                "flops_model": round(flops, 3),
                "dram_bytes": int(dram_bytes),
                "ncu_kernel_runtime_us": runtime_us,
                "arithmetic_intensity_flop_per_byte": ai,
                "measured_gflops": perf_gflops,
                "measured_bandwidth_gbs": bw_gbs,
                "percent_peak_bandwidth": 100.0 * bw_gbs / peak_bw_gbs,
                "roofline_bound_gflops": roof_gflops,
                "percent_roofline": 100.0 * perf_gflops / roof_gflops if roof_gflops > 0 else 0.0,
                "observed_bottlenecks": row["observed_bottlenecks"],
            })
    return rows


def write_csv(path, rows):
    RESULTS.mkdir(exist_ok=True)
    fields = [
        "case", "rows", "cols", "flops_model", "dram_bytes",
        "ncu_kernel_runtime_us", "arithmetic_intensity_flop_per_byte",
        "measured_gflops", "measured_bandwidth_gbs",
        "percent_peak_bandwidth", "roofline_bound_gflops",
        "percent_roofline", "observed_bottlenecks",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def lx(value):
    return math.log10(max(value, 1e-6))


def write_svg(path, rows, peak_bw_gbs, peak_fp32_gflops, source_label):
    PLOTS.mkdir(exist_ok=True)
    width, height = 900, 560
    left, right, top, bottom = 90, 35, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = -2.2, 1.4
    y_min, y_max = -1.0, math.log10(peak_fp32_gflops * 1.4)

    def sx(ai):
        return left + (lx(ai) - x_min) / (x_max - x_min) * plot_w

    def sy(gflops):
        return top + plot_h - (lx(gflops) - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Fusion Kernels: Empirical Roofline</text>',
        f'<text x="{width/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="13">Arithmetic intensity (FLOP/byte, log scale)</text>',
        f'<text x="22" y="{top + plot_h/2}" transform="rotate(-90 22 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Performance (GFLOP/s, log scale)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]

    for tick in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        if x_min <= lx(tick) <= x_max:
            x = sx(tick)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">{tick:g}</text>')
    for tick in [0.1, 1, 10, 100, 1000, 10000]:
        if y_min <= lx(tick) <= y_max:
            y = sy(tick)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:g}</text>')

    ridge = peak_fp32_gflops / peak_bw_gbs
    roof_points = []
    for ai in [0.005, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, ridge, 30]:
        if ai <= 0:
            continue
        gflops = min(peak_fp32_gflops, peak_bw_gbs * ai)
        if x_min <= lx(ai) <= x_max and y_min <= lx(gflops) <= y_max:
            roof_points.append(f"{sx(ai):.1f},{sy(gflops):.1f}")
    parts.append(f'<polyline points="{" ".join(roof_points)}" fill="none" stroke="#dc2626" stroke-width="2"/>')
    parts.append(f'<text x="{sx(ridge):.1f}" y="{sy(peak_fp32_gflops)-8:.1f}" font-family="Arial" font-size="11" fill="#dc2626">FP32 peak</text>')
    parts.append(f'<text x="{sx(0.08):.1f}" y="{sy(peak_bw_gbs*0.08)-8:.1f}" font-family="Arial" font-size="11" fill="#dc2626">memory roof</text>')

    colors = {
        "bias": "#2563eb",
        "residual": "#059669",
        "ln": "#7c3aed",
    }
    for row in rows:
        case = row["case"]
        color = colors["ln"] if case.startswith("ln_") else colors["residual"] if case.startswith("residual") else colors["bias"]
        x = sx(row["arithmetic_intensity_flop_per_byte"])
        y = sy(row["measured_gflops"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        label = case.replace("_", " ")
        parts.append(f'<text x="{x+7:.1f}" y="{y-7:.1f}" font-family="Arial" font-size="10">{label}</text>')

    parts.append(f'<text x="{left}" y="{height-8}" font-family="Arial" font-size="11" fill="#374151">Ceilings: {peak_bw_gbs:g} GB/s memory, {peak_fp32_gflops:g} GFLOP/s FP32. FLOPs are analytical approximations; bytes and times are from {source_label}.</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RESULTS / "fusion_profile_case_83615.csv"))
    parser.add_argument("--output-csv", default=str(RESULTS / "roofline_fusion_83615.csv"))
    parser.add_argument("--output-svg", default=str(PLOTS / "roofline_fusion_83615.svg"))
    parser.add_argument("--peak-bandwidth-gbs", type=float, default=672.0)
    parser.add_argument("--peak-fp32-gflops", type=float, default=16300.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_csv = Path(args.output_csv)
    output_svg = Path(args.output_svg)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_csv.is_absolute():
        output_csv = ROOT / output_csv
    if not output_svg.is_absolute():
        output_svg = ROOT / output_svg
    rows = read_rows(input_path, args.peak_bandwidth_gbs, args.peak_fp32_gflops)
    write_csv(output_csv, rows)
    write_svg(output_svg, rows, args.peak_bandwidth_gbs, args.peak_fp32_gflops, input_path.name)
    print(f"Wrote {output_csv.relative_to(ROOT)} and {output_svg.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
