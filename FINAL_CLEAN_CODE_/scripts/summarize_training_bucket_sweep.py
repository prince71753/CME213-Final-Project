#!/usr/bin/env python3
"""Parse final training bucket sweep logs into raw and summary CSVs."""

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
    r"hidden=(?P<hidden>[0-9]+)\s+backend=(?P<backend>[A-Za-z0-9_]+)\s+"
    r"sync_mode=(?P<mode>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)\s+"
    r"repeat=(?P<repeat>[0-9]+)\s+steps=(?P<steps>[0-9]+)\s+"
    r"lr=(?P<lr>[0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?sync_mode=(?P<requested>[a-z]+)\s+"
    r"effective_sync=(?P<effective>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)"
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


def is_finite(value):
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def parse_log(path):
    rows = []
    for title, body in parse_sections(path.read_text()):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        train = TRAIN_RE.search(body)
        epoch = EPOCH_RE.search(body)
        loss = epoch.group("loss") if epoch else "nan"
        checksum = epoch.group("checksum") if epoch else ""
        body_has_bad = re.search(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", body, re.I)
        valid = bool(epoch) and is_finite(loss) and checksum != "nan" and not body_has_bad
        sync = epoch.group("sync") if epoch else None
        start = epoch.group("start") if epoch else None
        finish = epoch.group("finish") if epoch else None
        rows.append({
            "hidden": int(tm.group("hidden")),
            "backend": tm.group("backend"),
            "requested_sync_mode": train.group("requested") if train else tm.group("mode"),
            "effective_sync": train.group("effective") if train else "",
            "bucket_kb": int(tm.group("bucket")),
            "repeat": int(tm.group("repeat")),
            "requested_steps": int(tm.group("steps")),
            "lr": tm.group("lr"),
            "avg_logged_loss": loss,
            "steps_rank": int(epoch.group("steps_rank")) if epoch else "",
            "time_ms": float(epoch.group("ms")) if epoch else "",
            "throughput_tok_s": float(epoch.group("tps")) if epoch else "",
            "throughput_mtok_s": float(epoch.group("tps")) / 1e6 if epoch else "",
            "avg_grad_sync_ms": float(sync) if sync else "",
            "avg_grad_start_ms": float(start) if start else "",
            "avg_grad_finish_ms": float(finish) if finish else "",
            "checksum_span": checksum or "",
            "valid": "yes" if valid else "no",
        })
    return rows


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def sample_std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def cv_pct(vals):
    m = mean(vals)
    if not vals or m == 0:
        return 0.0
    return 100.0 * sample_std(vals) / m


def median(vals):
    if not vals:
        return 0.0
    vals = sorted(vals)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["hidden"], row["backend"], row["requested_sync_mode"],
            row["effective_sync"], row["bucket_kb"], row["requested_steps"],
            row["lr"],
        )
        groups[key].append(row)

    out = []
    for key in sorted(groups):
        group = groups[key]
        valid_rows = [r for r in group if r["valid"] == "yes"]
        tps = [r["throughput_mtok_s"] for r in valid_rows]
        times = [r["time_ms"] for r in valid_rows]
        finish = [
            r["avg_grad_finish_ms"] for r in valid_rows
            if isinstance(r["avg_grad_finish_ms"], float)
        ]
        sync = [
            r["avg_grad_sync_ms"] for r in valid_rows
            if isinstance(r["avg_grad_sync_ms"], float)
        ]
        checksums = []
        for row in valid_rows:
            try:
                checksums.append(float(row["checksum_span"]))
            except ValueError:
                pass
        hidden, backend, requested, effective, bucket, steps, lr = key
        out.append({
            "hidden": hidden,
            "backend": backend,
            "requested_sync_mode": requested,
            "effective_sync": effective,
            "bucket_kb": bucket,
            "requested_steps": steps,
            "lr": lr,
            "runs": len(group),
            "valid_runs": len(valid_rows),
            "throughput_mean_mtok_s": mean(tps),
            "throughput_median_mtok_s": median(tps),
            "throughput_std_mtok_s": sample_std(tps),
            "throughput_cv_pct": cv_pct(tps),
            "time_mean_ms": mean(times),
            "time_std_ms": sample_std(times),
            "avg_grad_sync_mean_ms": mean(sync),
            "avg_grad_finish_mean_ms": mean(finish),
            "max_checksum_span": max(checksums) if checksums else "",
            "all_valid": "yes" if len(group) == len(valid_rows) else "no",
        })
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id")
    parser.add_argument("--log")
    parser.add_argument("--tag")
    parser.add_argument(
        "--min-repeat",
        type=int,
        default=1,
        help="only include rows with repeat >= this value; use 2 to drop repeat 1 warmup",
    )
    args = parser.parse_args()
    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"training_bucket_sweep_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")
    tag = args.tag or args.job_id or log_path.stem

    rows = parse_log(log_path)
    if args.min_repeat > 1:
        rows = [row for row in rows if row["repeat"] >= args.min_repeat]
    if not rows:
        raise SystemExit(f"no rows parsed from {log_path}")

    raw_fields = [
        "hidden", "backend", "requested_sync_mode", "effective_sync",
        "bucket_kb", "repeat", "requested_steps", "lr", "avg_logged_loss",
        "steps_rank", "time_ms", "throughput_tok_s", "throughput_mtok_s",
        "avg_grad_sync_ms", "avg_grad_start_ms", "avg_grad_finish_ms",
        "checksum_span", "valid",
    ]
    summary_fields = [
        "hidden", "backend", "requested_sync_mode", "effective_sync",
        "bucket_kb", "requested_steps", "lr", "runs", "valid_runs",
        "throughput_mean_mtok_s", "throughput_median_mtok_s",
        "throughput_std_mtok_s", "throughput_cv_pct", "time_mean_ms", "time_std_ms",
        "avg_grad_sync_mean_ms", "avg_grad_finish_mean_ms",
        "max_checksum_span", "all_valid",
    ]
    raw_path = RESULTS / f"training_bucket_sweep_{tag}.csv"
    summary_path = RESULTS / f"training_bucket_sweep_summary_{tag}.csv"
    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summarize(rows), summary_fields)
    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
