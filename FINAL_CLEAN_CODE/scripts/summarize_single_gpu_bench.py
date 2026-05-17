#!/usr/bin/env python3
"""Parse repeated single-GPU training logs into raw and summary CSVs."""

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
    r"repeat=(?P<repeat>[0-9]+)\s+steps=(?P<steps>[0-9]+)\s+"
    r"lr=(?P<lr>[0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?world_size=(?P<world>[0-9]+).*?"
    r"sync_mode=(?P<sync_mode>[a-z]+).*?effective_sync=(?P<effective>[a-z]+)"
)
EPOCH_RE = re.compile(
    r"Epoch 1: avg_logged_loss=(?P<loss>[-+0-9.eE]+|nan|inf|-nan|-inf).*?"
    r"steps/rank=(?P<steps_rank>[0-9]+)\s+"
    r"(?P<ms>[0-9]+)ms\s+(?P<tps>[0-9]+) tok/s"
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


def finite_number(value):
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def parse_log(path):
    rows = []
    text = path.read_text()
    for title, body in parse_sections(text):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        train = TRAIN_RE.search(body)
        epoch = EPOCH_RE.search(body)
        diverged = re.search(r"(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", body, re.I)
        valid = bool(epoch) and not diverged
        loss = epoch.group("loss") if epoch else "nan"
        if not finite_number(loss):
            valid = False
        rows.append({
            "hidden": int(tm.group("hidden")),
            "backend": tm.group("backend"),
            "repeat": int(tm.group("repeat")),
            "requested_steps": int(tm.group("steps")),
            "lr": tm.group("lr"),
            "world_size": int(train.group("world")) if train else "",
            "sync_mode": train.group("sync_mode") if train else "",
            "effective_sync": train.group("effective") if train else "",
            "avg_logged_loss": loss,
            "steps_rank": int(epoch.group("steps_rank")) if epoch else "",
            "time_ms": float(epoch.group("ms")) if epoch else "",
            "throughput_tok_s": float(epoch.group("tps")) if epoch else "",
            "throughput_mtok_s": float(epoch.group("tps")) / 1e6 if epoch else "",
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


def summarize(rows, min_repeat=1):
    groups = defaultdict(list)
    for row in rows:
        if row["repeat"] < min_repeat:
            continue
        key = (row["hidden"], row["backend"], row["requested_steps"], row["lr"])
        groups[key].append(row)

    out = []
    for key in sorted(groups):
        group = groups[key]
        valid_rows = [
            r for r in group
            if r["valid"] == "yes" and isinstance(r["throughput_mtok_s"], float)
        ]
        tps = [r["throughput_mtok_s"] for r in valid_rows]
        times = [r["time_ms"] for r in valid_rows]
        losses = [float(r["avg_logged_loss"]) for r in valid_rows]
        hidden, backend, steps, lr = key
        out.append({
            "hidden": hidden,
            "backend": backend,
            "requested_steps": steps,
            "lr": lr,
            "runs": len(group),
            "valid_runs": len(valid_rows),
            "throughput_mean_mtok_s": mean(tps),
            "throughput_median_mtok_s": median(tps),
            "throughput_std_mtok_s": sample_std(tps),
            "throughput_cv_pct": cv_pct(tps),
            "throughput_min_mtok_s": min(tps) if tps else "",
            "throughput_max_mtok_s": max(tps) if tps else "",
            "time_mean_ms": mean(times),
            "time_std_ms": sample_std(times),
            "final_loss_mean": mean(losses),
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
    parser.add_argument("--min-repeat", type=int, default=1)
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"single_gpu_repeated_bench_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")

    tag = args.tag or args.job_id or log_path.stem
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no benchmark rows parsed from {log_path}")

    raw_fields = [
        "hidden", "backend", "repeat", "requested_steps", "lr",
        "world_size", "sync_mode", "effective_sync", "avg_logged_loss",
        "steps_rank", "time_ms", "throughput_tok_s", "throughput_mtok_s",
        "valid",
    ]
    summary_fields = [
        "hidden", "backend", "requested_steps", "lr", "runs", "valid_runs",
        "throughput_mean_mtok_s", "throughput_median_mtok_s",
        "throughput_std_mtok_s", "throughput_min_mtok_s",
        "throughput_cv_pct", "throughput_max_mtok_s", "time_mean_ms", "time_std_ms",
        "final_loss_mean", "all_valid",
    ]
    raw_path = RESULTS / f"single_gpu_repeated_bench_{tag}.csv"
    summary_path = RESULTS / f"single_gpu_repeated_bench_summary_{tag}.csv"
    write_csv(raw_path, rows, raw_fields)
    write_csv(summary_path, summarize(rows, args.min_repeat), summary_fields)
    print(f"Wrote {raw_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
