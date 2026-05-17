#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


SECTION_RE = re.compile(r"backend=(?P<backend>[a-z_]+) mode=(?P<mode>[a-z]+)")
BENCH_RE = re.compile(
    r"SYNC_BENCH label=(?P<label>\S+) count=(?P<count>[0-9]+) "
    r"mb=(?P<mb>[0-9.]+) mode=(?P<actual_mode>\S+) "
    r"wall_ms=(?P<wall>[0-9.]+) start_ms=(?P<start>[0-9.]+) "
    r"finish_ms=(?P<finish>[0-9.]+) valid=(?P<valid>yes|no)"
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


def parse_log(path):
    rows = []
    for title, body in parse_sections(path.read_text()):
        section = SECTION_RE.search(title)
        if not section:
            continue
        for match in BENCH_RE.finditer(body):
            rows.append({
                "backend": section.group("backend"),
                "requested_mode": section.group("mode"),
                "actual_mode": match.group("actual_mode"),
                "label": match.group("label"),
                "count": int(match.group("count")),
                "mb": float(match.group("mb")),
                "wall_ms": float(match.group("wall")),
                "start_ms": float(match.group("start")),
                "finish_ms": float(match.group("finish")),
                "valid": match.group("valid"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    log_path = LOGS / f"gradient_sync_bench_{args.job_id}.out"
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no rows parsed from {log_path}")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"gradient_sync_bench_{args.job_id}.csv"
    fields = [
        "backend", "requested_mode", "actual_mode", "label", "count", "mb",
        "wall_ms", "start_ms", "finish_ms", "valid",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
