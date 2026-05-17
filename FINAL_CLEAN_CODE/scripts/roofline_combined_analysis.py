#!/usr/bin/env python3
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
PEAK_BW_GBS = 672.0
PEAK_COMPUTE_GFLOPS = 130000.0


def load_fusion_rows(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "group": "fusion",
                "case": row["case"],
                "ai": float(row["arithmetic_intensity_flop_per_byte"]),
                "gflops": float(row["measured_gflops"]),
                "bandwidth_gbs": float(row["measured_bandwidth_gbs"]),
                "runtime_us": float(row["ncu_kernel_runtime_us"]),
                "dram_bytes": float(row["dram_bytes"]),
                "notes": row.get("observed_bottlenecks", ""),
            })
    return rows


def load_hotspot_rows(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if not row.get("arithmetic_intensity") or not row.get("ncu_gflops"):
                continue
            rows.append({
                "group": "training_hotspot",
                "case": row["case"],
                "ai": float(row["arithmetic_intensity"]),
                "gflops": float(row["ncu_gflops"]),
                "bandwidth_gbs": float(row["measured_bw_gbs"]),
                "runtime_us": float(row["ncu_runtime_us"]),
                "dram_bytes": float(row["dram_total_bytes"]),
                "notes": row["kernel_name"],
            })
    return rows


def roofline_bound(ai):
    return min(PEAK_COMPUTE_GFLOPS, PEAK_BW_GBS * ai)


def write_csv(path, rows):
    RESULTS.mkdir(exist_ok=True)
    fields = [
        "group", "case", "ai", "gflops", "bandwidth_gbs", "runtime_us",
        "dram_bytes", "roofline_bound_gflops", "percent_roofline", "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            bound = roofline_bound(row["ai"])
            out = dict(row)
            out["roofline_bound_gflops"] = bound
            out["percent_roofline"] = 100.0 * row["gflops"] / bound if bound > 0 else 0.0
            writer.writerow(out)


def lx(value):
    return math.log10(max(value, 1e-9))


def write_svg(path, rows):
    PLOTS.mkdir(exist_ok=True)
    width, height = 960, 600
    left, right, top, bottom = 90, 40, 55, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = -2.2, 2.3
    y_min, y_max = -1.0, math.log10(PEAK_COMPUTE_GFLOPS * 1.4)

    def sx(ai):
        return left + (lx(ai) - x_min) / (x_max - x_min) * plot_w

    def sy(gflops):
        return top + plot_h - (lx(gflops) - y_min) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Empirical Roofline: Fusion Kernels and Training Hotspots</text>',
        f'<text x="{width/2}" y="{height-26}" text-anchor="middle" font-family="Arial" font-size="13">Arithmetic intensity (FLOP/byte, log scale)</text>',
        f'<text x="22" y="{top + plot_h/2}" transform="rotate(-90 22 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">Performance (GFLOP/s, log scale)</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    for tick in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]:
        x = sx(tick)
        if left <= x <= left + plot_w:
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{x:.1f}" y="{top+plot_h+20}" text-anchor="middle" font-family="Arial" font-size="11">{tick:g}</text>')
    for tick in [0.1, 1, 10, 100, 1000, 10000, 100000]:
        y = sy(tick)
        if top <= y <= top + plot_h:
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:g}</text>')

    roof_points = []
    ridge = PEAK_COMPUTE_GFLOPS / PEAK_BW_GBS
    for ai in [0.006, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, ridge, 300]:
        gflops = roofline_bound(ai)
        x = sx(ai)
        y = sy(gflops)
        if left <= x <= left + plot_w and top <= y <= top + plot_h:
            roof_points.append(f"{x:.1f},{y:.1f}")
    parts.append(f'<polyline points="{" ".join(roof_points)}" fill="none" stroke="#dc2626" stroke-width="2"/>')
    parts.append(f'<text x="{sx(ridge)+8:.1f}" y="{sy(PEAK_COMPUTE_GFLOPS)-8:.1f}" font-family="Arial" font-size="11" fill="#dc2626">ridge / tensor-core peak</text>')

    colors = {"fusion": "#2563eb", "training_hotspot": "#dc2626"}
    for row in rows:
        x = sx(row["ai"])
        y = sy(row["gflops"])
        color = colors[row["group"]]
        radius = 5 if row["group"] == "fusion" else 6
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
        label = row["case"].replace("_", " ")
        parts.append(f'<text x="{x+8:.1f}" y="{y-7:.1f}" font-family="Arial" font-size="10">{label}</text>')

    parts.append(f'<circle cx="{left+10}" cy="{height-55}" r="5" fill="#2563eb"/>')
    parts.append(f'<text x="{left+22}" y="{height-51}" font-family="Arial" font-size="12">fusion kernels</text>')
    parts.append(f'<circle cx="{left+145}" cy="{height-55}" r="6" fill="#dc2626"/>')
    parts.append(f'<text x="{left+158}" y="{height-51}" font-family="Arial" font-size="12">training GEMM hotspots</text>')
    parts.append(f'<text x="{left}" y="{height-8}" font-family="Arial" font-size="11" fill="#374151">Ceilings: {PEAK_BW_GBS:g} GB/s memory, {PEAK_COMPUTE_GFLOPS:g} GFLOP/s tensor-core compute. Bytes/times from NCU fusion and hotspot profiles.</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def main():
    rows = []
    fusion_path = RESULTS / "roofline_fusion.csv"
    if not fusion_path.exists():
        fusion_path = RESULTS / "roofline_fusion_83615.csv"
    rows.extend(load_fusion_rows(fusion_path))
    rows.extend(load_hotspot_rows(RESULTS / "hotspot_profile.csv"))
    write_csv(RESULTS / "roofline_combined.csv", rows)
    write_svg(PLOTS / "roofline_combined.svg", rows)
    print("Wrote results/roofline_combined.csv and plots/roofline_combined.svg")


if __name__ == "__main__":
    main()
