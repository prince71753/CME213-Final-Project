#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


TITLE_RE = re.compile(
    r"h512_lr lr=(?P<lr>[0-9.eE+-]+) backend=(?P<backend>[a-z_]+) "
    r"sync_mode=(?P<sync_mode>[a-z]+) repeat=(?P<repeat>[0-9]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?effective_sync=(?P<effective>[a-z]+).*?bucket_kb=(?P<bucket>[0-9]+)"
)
EPOCH_RE = re.compile(
    r"Epoch 1: avg_logged_loss=(?P<loss>[0-9.eE+-]+|nan|inf|-inf).*?"
    r" (?P<ms>[0-9]+)ms\s+(?P<tps>[0-9]+) tok/s"
    r"(?:\s+avg_grad_sync=(?P<sync>[0-9.]+)ms|"
    r"\s+avg_grad_start=(?P<start>[0-9.]+)ms\s+avg_grad_finish=(?P<finish>[0-9.]+)ms)?"
    r"(?:\s+checksum_span=(?P<checksum>[0-9.eE+-]+|nan|inf|-inf))?"
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


def finite_float(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def parse_log(path):
    rows = []
    for title, body in parse_sections(path.read_text()):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        train = TRAIN_RE.search(body)
        epoch = EPOCH_RE.search(body)
        if not train or not epoch:
            continue
        loss = epoch.group("loss")
        checksum = epoch.group("checksum") or ""
        valid = finite_float(loss) and finite_float(checksum)
        sync = epoch.group("sync")
        start = epoch.group("start")
        finish = epoch.group("finish")
        rows.append({
            "lr": tm.group("lr"),
            "backend": tm.group("backend"),
            "requested_sync_mode": tm.group("sync_mode"),
            "effective_sync": train.group("effective"),
            "repeat": int(tm.group("repeat")),
            "bucket_kb": int(train.group("bucket")),
            "avg_logged_loss": loss,
            "time_ms": float(epoch.group("ms")),
            "throughput_tok_s": float(epoch.group("tps")),
            "throughput_mtok_s": float(epoch.group("tps")) / 1e6,
            "avg_grad_sync_ms": float(sync) if sync else "",
            "avg_grad_start_ms": float(start) if start else "",
            "avg_grad_finish_ms": float(finish) if finish else "",
            "checksum_span": checksum,
            "valid": "yes" if valid else "no",
        })
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["lr"], row["backend"], row["requested_sync_mode"])].append(row)
    out = []
    for key in sorted(grouped, key=lambda k: float(k[0])):
        group = grouped[key]
        valid_rows = [row for row in group if row["valid"] == "yes"]
        throughputs = [row["throughput_mtok_s"] for row in valid_rows]
        syncs = [row["avg_grad_sync_ms"] for row in valid_rows
                 if row["avg_grad_sync_ms"] != ""]
        checksums = [float(row["checksum_span"]) for row in valid_rows]
        lr, backend, sync_mode = key
        out.append({
            "lr": lr,
            "backend": backend,
            "requested_sync_mode": sync_mode,
            "runs": len(group),
            "valid_runs": len(valid_rows),
            "throughput_mean_mtok_s": mean(throughputs),
            "throughput_std_mtok_s": sample_std(throughputs),
            "avg_grad_sync_ms": mean(syncs),
            "max_checksum_span": max(checksums) if checksums else "",
            "all_valid": "yes" if len(valid_rows) == len(group) else "no",
        })
    return out


def write_csv(path, rows, fields):
    RESULTS.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    log_path = LOGS / f"h512_lr_sweep_{args.job_id}.out"
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no rows parsed from {log_path}")

    raw_fields = [
        "lr", "backend", "requested_sync_mode", "effective_sync", "repeat",
        "bucket_kb", "avg_logged_loss", "time_ms", "throughput_tok_s",
        "throughput_mtok_s", "avg_grad_sync_ms", "avg_grad_start_ms",
        "avg_grad_finish_ms", "checksum_span", "valid",
    ]
    summary_fields = [
        "lr", "backend", "requested_sync_mode", "runs", "valid_runs",
        "throughput_mean_mtok_s", "throughput_std_mtok_s",
        "avg_grad_sync_ms", "max_checksum_span", "all_valid",
    ]

    raw_path = RESULTS / f"h512_lr_sweep_{args.job_id}.csv"
    summary_path = RESULTS / f"h512_lr_sweep_summary_{args.job_id}.csv"
    write_csv(raw_path, rows, raw_fields)
    summary = summarize(rows)
    write_csv(summary_path, summary, summary_fields)
    (RESULTS / "h512_lr_sweep.csv").write_text(raw_path.read_text())
    (RESULTS / "h512_lr_sweep_summary.csv").write_text(summary_path.read_text())
    print(f"Wrote {raw_path.relative_to(ROOT)} and {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
