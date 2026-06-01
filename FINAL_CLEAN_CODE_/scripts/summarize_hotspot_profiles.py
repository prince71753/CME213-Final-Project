#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PROFILES = ROOT / "profiles"
RESULTS = ROOT / "results"


WIDE_METRICS = {
    "gpu__time_duration.sum": ("ncu_runtime_us", 1.0e-3),
    "dram__bytes_read.sum": ("dram_read_bytes", 1.0),
    "dram__bytes_write.sum": ("dram_write_bytes", 1.0),
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_pct", 1.0),
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_pct", 1.0),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_throughput_pct", 1.0),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("achieved_occupancy_pct", 1.0),
}

TIMING_RE = re.compile(
    r"HOTSPOT_TIMING case=(?P<case>\S+)\s+"
    r"m=(?P<m>[0-9]+)\s+n=(?P<n>[0-9]+)\s+k=(?P<k>[0-9]+)\s+"
    r"batch=(?P<batch>[0-9]+)\s+runtime_us=(?P<runtime>[0-9.]+)\s+"
    r"flops=(?P<flops>[0-9.]+)\s+gflops=(?P<gflops>[0-9.]+)"
)


def parse_float(value):
    text = str(value or "").strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return ""
    try:
        return float(text)
    except ValueError:
        return ""


def pick(row, *names):
    for name in names:
        if name in row:
            return row[name]
    return ""


def case_from_path(path):
    match = re.match(r"hotspot_(.+)_[0-9]+_raw\.csv$", path.name)
    return match.group(1) if match else path.name


def parse_timing_logs(job_id):
    timings = {}
    for path in sorted(LOGS.glob(f"profile_hotspots_{job_id}_*.txt")):
        text = path.read_text(errors="ignore")
        for match in TIMING_RE.finditer(text):
            case = match.group("case")
            timings[case] = {
                "case": case,
                "m": int(match.group("m")),
                "n": int(match.group("n")),
                "k": int(match.group("k")),
                "batch": int(match.group("batch")),
                "event_runtime_us": float(match.group("runtime")),
                "flops": float(match.group("flops")),
                "event_gflops": float(match.group("gflops")),
            }
    return timings


def summarize_raw(path):
    case = case_from_path(path)
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            kernel = pick(record, "Kernel Name", "Kernel Name:").strip()
            if not kernel:
                continue
            row = {
                "case": case,
                "kernel_name": kernel,
                "ncu_runtime_us": "",
                "dram_read_bytes": "",
                "dram_write_bytes": "",
                "dram_throughput_pct": "",
                "sm_throughput_pct": "",
                "achieved_occupancy_pct": "",
            }
            for metric, (out_name, scale) in WIDE_METRICS.items():
                value = parse_float(record.get(metric, ""))
                if value != "":
                    row[out_name] = value * scale
            if row["ncu_runtime_us"] != "":
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", default=str(RESULTS / "hotspot_profile.csv"))
    args = parser.parse_args()

    timings = parse_timing_logs(args.job_id)
    raw_rows = []
    for path in sorted(PROFILES.glob(f"hotspot_*_{args.job_id}_raw.csv")):
        raw_rows.extend(summarize_raw(path))

    rows = []
    for row in raw_rows:
        timing = timings.get(row["case"], {})
        merged = {**timing, **row}
        dram_bytes = 0.0
        for key in ["dram_read_bytes", "dram_write_bytes"]:
            if merged.get(key) != "":
                dram_bytes += float(merged[key])
        flops = float(merged.get("flops", 0.0) or 0.0)
        ncu_runtime_us = merged.get("ncu_runtime_us", "")
        merged["dram_total_bytes"] = dram_bytes if dram_bytes else ""
        merged["arithmetic_intensity"] = flops / dram_bytes if dram_bytes else ""
        merged["ncu_gflops"] = flops / (float(ncu_runtime_us) * 1000.0) \
            if ncu_runtime_us not in {"", 0.0} else ""
        merged["measured_bw_gbs"] = dram_bytes / (float(ncu_runtime_us) * 1.0e-6) / 1.0e9 \
            if dram_bytes and ncu_runtime_us not in {"", 0.0} else ""
        rows.append(merged)

    fieldnames = [
        "case", "kernel_name", "m", "n", "k", "batch", "flops",
        "event_runtime_us", "event_gflops", "ncu_runtime_us",
        "ncu_gflops", "dram_read_bytes", "dram_write_bytes",
        "dram_total_bytes", "arithmetic_intensity", "measured_bw_gbs",
        "dram_throughput_pct", "sm_throughput_pct", "achieved_occupancy_pct",
    ]
    RESULTS.mkdir(exist_ok=True)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["case"]):
            writer.writerow(row)
    print(f"Wrote {out.relative_to(ROOT)} with {len(rows)} hotspot rows.")


if __name__ == "__main__":
    main()
