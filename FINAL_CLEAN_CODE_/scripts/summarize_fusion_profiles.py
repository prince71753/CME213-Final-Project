#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PROFILES = ROOT / "profiles"
RESULTS = ROOT / "results"


METRICS = {
    "gpu__time_duration.sum": "runtime_us",
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_write_bytes",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
}

WIDE_METRICS = {
    "gpu__time_duration.sum": ("runtime_us", 1.0e-3),
    "dram__bytes_read.sum": ("dram_read_bytes", 1.0),
    "dram__bytes_write.sum": ("dram_write_bytes", 1.0),
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_pct", 1.0),
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed": ("dram_throughput_pct", 1.0),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("sm_throughput_pct", 1.0),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("achieved_occupancy_pct", 1.0),
}


def parse_float(value):
    if value is None:
        return ""
    text = str(value).strip().replace(",", "")
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


def infer_bottleneck(row):
    dram = row.get("dram_throughput_pct", "")
    sm = row.get("sm_throughput_pct", "")
    occ = row.get("achieved_occupancy_pct", "")
    if dram == "" or sm == "":
        return "unknown"
    dram = float(dram)
    sm = float(sm)
    occ = float(occ) if occ != "" else 0.0
    if dram < 25.0 and sm < 25.0:
        return "latency/launch-bound" if occ < 45.0 else "low-utilization"
    if dram > sm * 1.25:
        return "memory-bound"
    if sm > dram * 1.25:
        return "compute-bound"
    return "mixed"


def case_from_path(path):
    name = path.name
    match = re.match(r"fusion_(.+)_[0-9]+_raw\.csv$", name)
    return match.group(1) if match else name


def summarize_raw(path):
    case = case_from_path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "Metric Name" not in fieldnames and "Metric Name:" not in fieldnames:
            rows = []
            for record in reader:
                kernel = pick(record, "Kernel Name", "Kernel Name:").strip()
                if not kernel:
                    continue
                row = {
                    "case": case,
                    "launch_id": pick(record, "ID", "ID:").strip(),
                    "kernel_name": kernel,
                    "runtime_us": "",
                    "dram_read_bytes": "",
                    "dram_write_bytes": "",
                    "dram_throughput_pct": "",
                    "sm_throughput_pct": "",
                    "achieved_occupancy_pct": "",
                    "observed_bottleneck": "",
                    "counter_source": "ncu",
                }
                for metric, (out_name, scale) in WIDE_METRICS.items():
                    value = parse_float(record.get(metric, ""))
                    if value != "":
                        row[out_name] = value * scale
                row["observed_bottleneck"] = infer_bottleneck(row)
                rows.append(row)
            return rows

    rows = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            metric = pick(record, "Metric Name", "Metric Name:").strip()
            if metric not in METRICS:
                continue
            kernel = pick(record, "Kernel Name", "Kernel Name:").strip()
            if not kernel:
                kernel = pick(record, "Name", "Name:").strip()
            if not kernel:
                kernel = "unknown"
            launch = pick(record, "ID", "ID:").strip()
            key = (case, launch, kernel)
            if key not in rows:
                rows[key] = {
                    "case": case,
                    "launch_id": launch,
                    "kernel_name": kernel,
                    "runtime_us": "",
                    "dram_read_bytes": "",
                    "dram_write_bytes": "",
                    "dram_throughput_pct": "",
                    "sm_throughput_pct": "",
                    "achieved_occupancy_pct": "",
                    "observed_bottleneck": "",
                    "counter_source": "ncu",
                }
            value = parse_float(pick(record, "Metric Value", "Metric Value:"))
            rows[key][METRICS[metric]] = value
    for row in rows.values():
        row["observed_bottleneck"] = infer_bottleneck(row)
    return list(rows.values())


TIMING_RE = re.compile(
    r"PROFILE_TIMING case=(?P<case>\S+)\s+"
    r"runtime_us=(?P<runtime>[0-9.]+)\s+"
    r"estimated_read_bytes=(?P<read>[0-9.]+)\s+"
    r"estimated_write_bytes=(?P<write>[0-9.]+)"
)


def parse_timing_logs(job_id):
    rows_by_case = {}
    pattern = f"profile_fusion_{job_id}_*.txt" if job_id else "profile_fusion_*_*.txt"
    for path in sorted(LOGS.glob(pattern)):
        text = path.read_text(errors="ignore")
        for match in TIMING_RE.finditer(text):
            case = match.group("case")
            runtime_us = parse_float(match.group("runtime"))
            read_bytes = parse_float(match.group("read"))
            write_bytes = parse_float(match.group("write"))
            runtime_s = runtime_us / 1.0e6 if runtime_us != "" else 0.0
            total_bytes = (read_bytes if read_bytes != "" else 0.0) + \
                          (write_bytes if write_bytes != "" else 0.0)
            bw_gbs = total_bytes / runtime_s / 1.0e9 if runtime_s > 0.0 else ""
            row = {
                "case": case,
                "launch_id": "event",
                "kernel_name": case,
                "runtime_us": runtime_us,
                "dram_read_bytes": read_bytes,
                "dram_write_bytes": write_bytes,
                "dram_throughput_pct": "",
                "sm_throughput_pct": "",
                "achieved_occupancy_pct": "",
                "observed_bottleneck": "estimated-bandwidth %.1f GB/s" % bw_gbs
                                      if bw_gbs != "" else "timed",
                "counter_source": "cuda_event_estimated_traffic",
            }
            rows_by_case[case] = row
    return list(rows_by_case.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="",
                        help="Only summarize profiles for this Slurm job id")
    parser.add_argument("--output", default=str(RESULTS / "fusion_profile.csv"))
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    pattern = f"fusion_*_{args.job_id}_raw.csv" if args.job_id else "fusion_*_raw.csv"
    raw_files = sorted(PROFILES.glob(pattern))
    all_rows = []
    for path in raw_files:
        all_rows.extend(summarize_raw(path))

    raw_cases = {row["case"] for row in all_rows}
    for row in parse_timing_logs(args.job_id):
        if row["case"] not in raw_cases:
            all_rows.append(row)

    fieldnames = [
        "case", "launch_id", "kernel_name", "runtime_us",
        "dram_read_bytes", "dram_write_bytes", "dram_throughput_pct",
        "sm_throughput_pct", "achieved_occupancy_pct", "observed_bottleneck",
        "counter_source",
    ]
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["case"], r["launch_id"], r["kernel_name"])):
            writer.writerow(row)

    print(f"Wrote {out.relative_to(ROOT)} with {len(all_rows)} profiled kernel rows.")


if __name__ == "__main__":
    main()
