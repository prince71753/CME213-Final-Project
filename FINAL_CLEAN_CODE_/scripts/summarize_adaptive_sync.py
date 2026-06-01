#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


EPOCH_RE = re.compile(
    r"Epoch 1: avg_logged_loss=(?P<loss>[0-9.]+|nan).*?"
    r" (?P<ms>[0-9]+)ms\s+(?P<tps>[0-9]+) tok/s"
    r"(?:\s+avg_grad_sync=(?P<sync>[0-9.]+)ms|"
    r"\s+avg_grad_start=(?P<start>[0-9.]+)ms\s+avg_grad_finish=(?P<finish>[0-9.]+)ms)?"
    r"(?:\s+checksum_span=(?P<checksum>[0-9.eE+-]+|nan))?"
)
TRAIN_RE = re.compile(
    r"Training: .*?lr=(?P<lr>[0-9.eE+-]+).*?"
    r"sync_mode=(?P<sync_mode>[a-z]+)\s+effective_sync=(?P<effective_sync>[a-z]+)"
    r"\s+bucket_kb=(?P<bucket>[0-9]+)\s+auto_overlap_max_mb=(?P<auto_mb>[0-9.]+)"
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
        hidden_match = re.search(r"hidden=([0-9]+)", title)
        if not hidden_match or "sync_mode=" not in title:
            continue
        epoch = EPOCH_RE.search(body)
        train = TRAIN_RE.search(body)
        if not epoch or not train:
            continue
        sync = epoch.group("sync")
        start = epoch.group("start")
        finish = epoch.group("finish")
        checksum = epoch.group("checksum")
        loss = epoch.group("loss")
        valid = loss != "nan" and checksum != "nan"
        rows.append({
            "hidden": int(hidden_match.group(1)),
            "requested_sync_mode": train.group("sync_mode"),
            "effective_sync": train.group("effective_sync"),
            "bucket_kb": int(train.group("bucket")),
            "auto_overlap_max_mb": float(train.group("auto_mb")),
            "lr": train.group("lr"),
            "avg_logged_loss": loss,
            "time_ms": float(epoch.group("ms")),
            "throughput_tok_s": float(epoch.group("tps")),
            "throughput_mtok_s": float(epoch.group("tps")) / 1e6,
            "avg_grad_sync_ms": float(sync) if sync else "",
            "avg_grad_start_ms": float(start) if start else "",
            "avg_grad_finish_ms": float(finish) if finish else "",
            "checksum_span": checksum if checksum else "",
            "valid": "yes" if valid else "no",
        })
    return rows


def write_csv(path, rows):
    RESULTS.mkdir(exist_ok=True)
    fields = [
        "hidden", "requested_sync_mode", "effective_sync", "bucket_kb",
        "auto_overlap_max_mb", "lr", "avg_logged_loss", "time_ms", "throughput_tok_s",
        "throughput_mtok_s", "avg_grad_sync_ms", "avg_grad_start_ms",
        "avg_grad_finish_ms", "checksum_span", "valid",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    log_path = LOGS / f"adaptive_sync_sweep_{args.job_id}.out"
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")

    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no adaptive sync rows parsed from {log_path}")

    out_path = Path(args.output) if args.output else RESULTS / f"adaptive_sync_sweep_{args.job_id}.csv"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_csv(out_path, rows)
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
