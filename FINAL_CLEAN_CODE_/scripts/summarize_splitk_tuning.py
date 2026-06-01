#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
CASE_WEIGHTS = {
    "splitk_dW1": 1,
    "splitk_dW2": 1,
    "splitk_qkv": 4,
    "splitk_dWout": 1,
}


SECTION_RE = re.compile(r"splitk_tune target=(?P<target>[0-9]+) case=(?P<case>\S+)")
TIMING_RE = re.compile(
    r"HOTSPOT_TIMING case=(?P<case>\S+).*?runtime_us=(?P<runtime>[0-9.]+).*?"
    r"gflops=(?P<gflops>[0-9.]+)"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    path = LOGS / f"splitk_tuning_{args.job_id}.out"
    if not path.exists():
        raise SystemExit(f"missing log: {path}")

    rows = []
    for title, body in parse_sections(path.read_text()):
        sm = SECTION_RE.search(title)
        tm = TIMING_RE.search(body)
        if not sm or not tm:
            continue
        rows.append({
            "target_blocks": int(sm.group("target")),
            "case": sm.group("case"),
            "runtime_us": float(tm.group("runtime")),
            "gflops": float(tm.group("gflops")),
        })

    if not rows:
        raise SystemExit(f"no rows parsed from {path}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"splitk_tuning_{args.job_id}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_blocks", "case", "runtime_us", "gflops"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["case"], r["target_blocks"])))
    (RESULTS / "splitk_tuning.csv").write_text(out.read_text())

    by_target = {}
    for row in rows:
        by_target.setdefault(row["target_blocks"], {})[row["case"]] = row["runtime_us"]
    summary_rows = []
    for target, cases in sorted(by_target.items()):
        if not all(case in cases for case in CASE_WEIGHTS):
            continue
        total = sum(cases[case] * weight for case, weight in CASE_WEIGHTS.items())
        summary_rows.append({
            "target_blocks": target,
            "weighted_runtime_us": total,
            "relative_to_144": "",
        })
    baseline = next((r["weighted_runtime_us"] for r in summary_rows
                     if r["target_blocks"] == 144), None)
    if baseline:
        for row in summary_rows:
            row["relative_to_144"] = row["weighted_runtime_us"] / baseline
    if summary_rows:
        summary = RESULTS / f"splitk_tuning_summary_{args.job_id}.csv"
        with summary.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["target_blocks", "weighted_runtime_us", "relative_to_144"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        (RESULTS / "splitk_tuning_summary.csv").write_text(summary.read_text())
        print(f"Wrote {out.relative_to(ROOT)} and {summary.relative_to(ROOT)}")
    else:
        print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
