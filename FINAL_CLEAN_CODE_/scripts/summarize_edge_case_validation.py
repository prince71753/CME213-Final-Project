#!/usr/bin/env python3
"""Parse edge-case validation logs into CSV and Markdown summaries."""

import argparse
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


TITLE_RE = re.compile(
    r"edge_case name=(?P<name>[A-Za-z0-9_]+)\s+hidden=(?P<hidden>[0-9]+)\s+"
    r"ranks=(?P<ranks>[0-9]+)\s+batch=(?P<batch>[0-9]+)\s+"
    r"mode=(?P<mode>[a-z]+)\s+comm_thread=(?P<comm_thread>yes|no)\s+"
    r"steps=(?P<steps>[0-9]+)\s+lr=(?P<lr>[0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?world_size=(?P<world>[0-9]+)\s+sync_mode=(?P<requested>[a-z]+)\s+"
    r"effective_sync=(?P<effective>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)"
)
BATCH_RE = re.compile(
    r"Batches: total=(?P<total>[0-9]+)\s+local_per_rank=(?P<local>[0-9]+)\s+"
    r"used_per_epoch=(?P<used>[0-9]+)\s+dropped=(?P<dropped>[0-9]+)"
)
EPOCH_RE = re.compile(
    r"Epoch 1: avg_logged_loss=(?P<loss>[-+0-9.eE]+|nan|inf|-nan|-inf).*?"
    r"steps/rank=(?P<steps_rank>[0-9]+)\s+(?P<ms>[0-9]+)ms\s+"
    r"(?P<tps>[0-9]+) tok/s"
    r"(?:\s+avg_grad_sync=(?P<sync>[0-9.]+)ms|"
    r"\s+avg_grad_start=(?P<start>[0-9.]+)ms\s+avg_grad_finish=(?P<finish>[0-9.]+)ms)?"
    r"(?:\s+checksum_span=(?P<checksum>[0-9.eE+-]+|nan))?"
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


def finite_text(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_log(path):
    rows = []
    for title, body in parse_sections(path.read_text(errors="replace")):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        train = TRAIN_RE.search(body)
        batches = BATCH_RE.search(body)
        epoch = EPOCH_RE.search(body)
        loss = epoch.group("loss") if epoch else "nan"
        checksum = epoch.group("checksum") if epoch else ""
        body_has_bad = re.search(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", body, re.I)
        valid = bool(train and batches and epoch) and finite_text(loss) and checksum != "nan" and not body_has_bad
        rows.append({
            "name": tm.group("name"),
            "hidden": int(tm.group("hidden")),
            "ranks": int(tm.group("ranks")),
            "batch": int(tm.group("batch")),
            "mode": tm.group("mode"),
            "comm_thread": tm.group("comm_thread"),
            "requested_sync_mode": train.group("requested") if train else "",
            "effective_sync": train.group("effective") if train else "",
            "bucket_kb": int(train.group("bucket")) if train else "",
            "requested_steps": int(tm.group("steps")),
            "lr": tm.group("lr"),
            "total_batches": int(batches.group("total")) if batches else "",
            "local_per_rank": int(batches.group("local")) if batches else "",
            "used_per_epoch": int(batches.group("used")) if batches else "",
            "dropped_batches": int(batches.group("dropped")) if batches else "",
            "avg_logged_loss": loss,
            "steps_rank": int(epoch.group("steps_rank")) if epoch else "",
            "time_ms": float(epoch.group("ms")) if epoch else "",
            "throughput_tok_s": float(epoch.group("tps")) if epoch else "",
            "avg_grad_sync_ms": float(epoch.group("sync")) if epoch and epoch.group("sync") else "",
            "avg_grad_start_ms": float(epoch.group("start")) if epoch and epoch.group("start") else "",
            "avg_grad_finish_ms": float(epoch.group("finish")) if epoch and epoch.group("finish") else "",
            "checksum_span": checksum or "",
            "valid": "yes" if valid else "no",
        })
    return rows


def write_csv(path, rows):
    fields = [
        "name", "hidden", "ranks", "batch", "mode", "comm_thread",
        "requested_sync_mode", "effective_sync", "bucket_kb", "requested_steps",
        "lr", "total_batches", "local_per_rank", "used_per_epoch",
        "dropped_batches", "avg_logged_loss", "steps_rank", "time_ms",
        "throughput_tok_s", "avg_grad_sync_ms", "avg_grad_start_ms",
        "avg_grad_finish_ms", "checksum_span", "valid",
    ]
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, tag, rows):
    all_valid = all(row["valid"] == "yes" for row in rows)
    lines = [
        f"# Edge-Case Validation Summary: Job {tag}",
        "",
        f"Overall result: {'PASS' if all_valid else 'FAIL'}",
        "",
        "These cases check non-power-of-two rank counts and dataset partitions",
        "with dropped global batches.",
        "",
        "| Case | Ranks | Batch | Mode | Comm thread | Dropped | Throughput tok/s | Checksum span | Status |",
        "|---|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        checksum = row["checksum_span"] if row["checksum_span"] != "" else "n/a"
        lines.append(
            f"| {row['name']} | {row['ranks']} | {row['batch']} | "
            f"{row['effective_sync']} | {row['comm_thread']} | "
            f"{row['dropped_batches']} | {row['throughput_tok_s']:.0f} | "
            f"{checksum} | {'PASS' if row['valid'] == 'yes' else 'FAIL'} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- `dropped > 0` confirms the run exercised uneven dataset partitioning.",
        "- Nonzero but small checksum spans indicate ranks stayed numerically close",
        "  after synchronization.",
        "- These are smoke tests, not throughput headline numbers.",
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
        log_path = LOGS / f"edge_case_validation_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    tag = args.tag or args.job_id or log_path.stem
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no edge-case rows parsed from {log_path}")
    csv_path = RESULTS / f"edge_case_validation_{tag}.csv"
    md_path = RESULTS / f"edge_case_validation_{tag}.md"
    write_csv(csv_path, rows)
    write_md(md_path, tag, rows)
    (RESULTS / "edge_case_validation.csv").write_text(csv_path.read_text())
    (RESULTS / "edge_case_validation.md").write_text(md_path.read_text())
    if not all(row["valid"] == "yes" for row in rows):
        raise SystemExit("one or more edge-case validations failed")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
