#!/usr/bin/env python3
"""Parse GEMM backend/precision validation logs into CSV and Markdown."""

import argparse
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


TITLE_RE = re.compile(
    r"gemm_precision case=(?P<case>[A-Za-z0-9_]+)\s+"
    r"requested_backend=(?P<requested>[A-Za-z0-9_]+)\s+"
    r"strict_fp32=(?P<strict>[01])\s+description=(?P<description>\S+)"
)
STATUS_RE = re.compile(r"case_status name=(?P<case>[A-Za-z0-9_]+) status=(?P<status>[0-9]+)")
BACKEND_RE = re.compile(r"GEMM requested backend:\s+(?P<backend>\S+)")
POLICY_RE = re.compile(r"GEMM auto policy:\s+(?P<policy>\S+)")
STRICT_RE = re.compile(r"CME213_STRICT_FP32:\s+(?P<strict>[01])")
TOL_RE = re.compile(r"gemm_tiled tolerance:\s+(?P<tol>[-+0-9.eE]+)")
SIZE_RE = re.compile(r"M=(?P<M>[0-9]+), N=(?P<N>[0-9]+), K=(?P<K>[0-9]+)")
NAIVE_RE = re.compile(
    r"naive\s+max error:\s+(?P<err>[-+0-9.eE]+)\s+(?P<status>PASS|FAIL)"
)
TILED_RE = re.compile(
    r"gemm_tiled\(dispatch\) max error:\s+(?P<err>[-+0-9.eE]+)\s+(?P<status>PASS|FAIL)"
)


def parse_case_sections(text):
    current = None
    body = []
    for line in text.splitlines():
        if line.startswith("=== gemm_precision "):
            if current is not None:
                yield current, "\n".join(body)
            current = line.strip("= ").strip()
            body = []
        elif current is not None:
            body.append(line)
            if STATUS_RE.search(line):
                yield current, "\n".join(body)
                current = None
                body = []
    if current is not None:
        yield current, "\n".join(body)


def parse_statuses(text):
    statuses = {}
    for match in STATUS_RE.finditer(text):
        statuses[match.group("case")] = int(match.group("status"))
    return statuses


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_log(path):
    text = path.read_text(errors="replace")
    rows = []
    for title, body in parse_case_sections(text):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        reported_backend = BACKEND_RE.search(body)
        auto_policy = POLICY_RE.search(body)
        reported_strict = STRICT_RE.search(body)
        tolerance = TOL_RE.search(body)
        status_match = STATUS_RE.search(body)
        case_status = int(status_match.group("status")) if status_match else 999
        current = None
        for line in body.splitlines():
            sm = SIZE_RE.search(line)
            if sm:
                current = {
                    "case": tm.group("case"),
                    "requested_backend_env": tm.group("requested"),
                    "strict_fp32_env": tm.group("strict"),
                    "description": tm.group("description"),
                    "reported_backend": reported_backend.group("backend") if reported_backend else "",
                    "auto_policy": auto_policy.group("policy") if auto_policy else "",
                    "reported_strict_fp32": reported_strict.group("strict") if reported_strict else "",
                    "tolerance": tolerance.group("tol") if tolerance else "",
                    "M": int(sm.group("M")),
                    "N": int(sm.group("N")),
                    "K": int(sm.group("K")),
                    "naive_error": "",
                    "naive_status": "",
                    "gemm_tiled_error": "",
                    "gemm_tiled_status": "",
                    "process_status": case_status,
                    "valid": "no",
                }
                rows.append(current)
                continue
            if current is None:
                continue
            nm = NAIVE_RE.search(line)
            if nm:
                current["naive_error"] = nm.group("err")
                current["naive_status"] = nm.group("status")
                continue
            gm = TILED_RE.search(line)
            if gm:
                current["gemm_tiled_error"] = gm.group("err")
                current["gemm_tiled_status"] = gm.group("status")
                current["valid"] = (
                    "yes"
                    if case_status == 0
                    and current["naive_status"] == "PASS"
                    and current["gemm_tiled_status"] == "PASS"
                    and finite(current["gemm_tiled_error"])
                    else "no"
                )
    return rows


def write_csv(path, rows):
    fields = [
        "case", "requested_backend_env", "strict_fp32_env", "description",
        "reported_backend", "auto_policy", "reported_strict_fp32", "tolerance",
        "M", "N", "K", "naive_error", "naive_status", "gemm_tiled_error",
        "gemm_tiled_status", "process_status", "valid",
    ]
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def case_summary(rows):
    summaries = []
    for case in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        max_err = max(float(row["gemm_tiled_error"]) for row in case_rows)
        row512 = next((row for row in case_rows if int(row["M"]) == 512), None)
        summaries.append({
            "case": case,
            "backend": case_rows[0]["reported_backend"],
            "strict": case_rows[0]["reported_strict_fp32"],
            "tolerance": float(case_rows[0]["tolerance"]),
            "error512": float(row512["gemm_tiled_error"]) if row512 else float("nan"),
            "max_error": max_err,
            "valid": all(row["valid"] == "yes" for row in case_rows),
        })
    return summaries


def write_md(path, tag, rows):
    summaries = case_summary(rows)
    all_valid = all(item["valid"] for item in summaries)
    lines = [
        f"# GEMM Precision Validation Summary: Job {tag}",
        "",
        f"Overall result: {'PASS' if all_valid else 'FAIL'}",
        "",
        "This run separates the throughput-oriented default GEMM path from",
        "strict FP32 correctness checks.",
        "",
        "| Case | Reported backend | Strict FP32 | Tolerance | 512 error | Max tiled error | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['case']} | {item['backend']} | {item['strict']} | "
            f"{item['tolerance']:.1e} | {item['error512']:.6e} | "
            f"{item['max_error']:.6e} | {'PASS' if item['valid'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- The default auto path reports `cublas_tc` policy and is validated with",
        "  the intended non-strict tolerance used for throughput runs.",
        "- The custom CUDA kernel is separately forced with `CME213_GEMM_BACKEND=custom`",
        "  and `CME213_STRICT_FP32=1`, where it must pass the stricter `1e-5`",
        "  tolerance.",
        "- This resolves the old stale-log confusion around the `512x512x512`",
        "  GEMM error: the larger default error is a deliberate precision/performance",
        "  tradeoff, not a broken custom FP32 kernel.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id")
    parser.add_argument("--log")
    parser.add_argument("--tag")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"gemm_precision_validation_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")

    tag = args.tag or args.job_id or log_path.stem
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no GEMM precision rows parsed from {log_path}")

    csv_path = RESULTS / f"gemm_precision_validation_{tag}.csv"
    md_path = RESULTS / f"gemm_precision_validation_{tag}.md"
    write_csv(csv_path, rows)
    write_md(md_path, tag, rows)
    (RESULTS / "gemm_precision_validation.csv").write_text(csv_path.read_text())
    (RESULTS / "gemm_precision_validation.md").write_text(md_path.read_text())

    if not all(row["valid"] == "yes" for row in rows):
        raise SystemExit("one or more GEMM precision rows failed")

    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
