#!/usr/bin/env python3
"""Parse per-step trajectory traces and compare sync modes."""

import argparse
import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

EXPECTED_CASES = ["blocking_direct", "pinned_overlap", "openmp_thread"]
REFERENCE_CASE = "blocking_direct"
PARAM_COUNTS = {
    128: 222592,
    256: 838400,
    512: 3249664,
}

TITLE_RE = re.compile(
    r"trajectory_case name=(?P<name>[A-Za-z0-9_]+)\s+hidden=(?P<hidden>[0-9]+)\s+"
    r"ranks=(?P<ranks>[0-9]+)\s+batch=(?P<batch>[0-9]+)\s+mode=(?P<mode>[a-z]+)\s+"
    r"comm_thread=(?P<comm_thread>yes|no)\s+bucket_kb=(?P<bucket>[0-9]+)\s+"
    r"steps=(?P<steps>[0-9]+)\s+lr=(?P<lr>[-+0-9.eE]+)"
)
TRAIN_RE = re.compile(
    r"Training: .*?world_size=(?P<world>[0-9]+)\s+sync_mode=(?P<requested>[a-z]+)\s+"
    r"effective_sync=(?P<effective>[a-z]+)\s+bucket_kb=(?P<bucket>[0-9]+)"
)
TRAJ_RE = re.compile(
    r"TRAJECTORY epoch=(?P<epoch>[0-9]+)\s+step=(?P<step>[0-9]+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+|nan|inf|-nan|-inf)\s+"
    r"param_sum=(?P<sum>[-+0-9.eE]+|nan|inf|-nan|-inf)\s+"
    r"param_sumsq=(?P<sumsq>[-+0-9.eE]+|nan|inf|-nan|-inf)\s+"
    r"param_maxabs=(?P<maxabs>[-+0-9.eE]+|nan|inf|-nan|-inf)\s+"
    r"param_hash=(?P<hash>[0-9a-fA-F]+)\s+"
    r"sum_span=(?P<sum_span>[-+0-9.eE]+|nan|inf|-nan|-inf)\s+"
    r"sumsq_span=(?P<sumsq_span>[-+0-9.eE]+|nan|inf|-nan|-inf)\s+"
    r"maxabs_span=(?P<maxabs_span>[-+0-9.eE]+|nan|inf|-nan|-inf)"
)


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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
    cases = {}
    for title, body in parse_sections(path.read_text(errors="replace")):
        tm = TITLE_RE.search(title)
        if not tm:
            continue
        name = tm.group("name")
        train = TRAIN_RE.search(body)
        trace_rows = []
        for match in TRAJ_RE.finditer(body):
            trace_rows.append({
                "case": name,
                "hidden": int(tm.group("hidden")),
                "ranks": int(tm.group("ranks")),
                "batch": int(tm.group("batch")),
                "mode": tm.group("mode"),
                "comm_thread": tm.group("comm_thread"),
                "requested_sync_mode": train.group("requested") if train else "",
                "effective_sync": train.group("effective") if train else "",
                "bucket_kb": int(train.group("bucket")) if train else int(tm.group("bucket")),
                "requested_steps": int(tm.group("steps")),
                "lr": tm.group("lr"),
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "param_sum": float(match.group("sum")),
                "param_sumsq": float(match.group("sumsq")),
                "param_maxabs": float(match.group("maxabs")),
                "param_hash": match.group("hash").lower(),
                "sum_span": float(match.group("sum_span")),
                "sumsq_span": float(match.group("sumsq_span")),
                "maxabs_span": float(match.group("maxabs_span")),
            })
        cases[name] = {
            "title": tm.groupdict(),
            "train": train.groupdict() if train else {},
            "rows": trace_rows,
        }
    return cases


def compare_cases(cases, args):
    missing = [name for name in EXPECTED_CASES if name not in cases]
    if missing:
        raise SystemExit(f"missing trajectory cases: {missing}")
    ref_rows = {row["step"]: row for row in cases[REFERENCE_CASE]["rows"]}
    if not ref_rows:
        raise SystemExit("reference case has no trajectory rows")

    detailed = []
    summary = []
    for case_name in EXPECTED_CASES:
        rows = cases[case_name]["rows"]
        hidden = int(cases[case_name]["title"]["hidden"])
        param_count = PARAM_COUNTS.get(hidden)
        if not param_count:
            raise SystemExit(f"missing parameter count for hidden={hidden}")
        steps = [row["step"] for row in rows]
        expected_steps = list(range(1, args.steps + 1))
        complete = steps == expected_steps
        max_loss_delta = 0.0
        max_sum_delta = 0.0
        max_sumsq_delta = 0.0
        max_maxabs_delta = 0.0
        max_sum_span = 0.0
        max_sumsq_span = 0.0
        max_maxabs_span = 0.0
        hash_mismatches = 0
        invalid_values = 0
        failed_steps = 0

        for row in rows:
            ref = ref_rows.get(row["step"])
            if ref is None:
                failed_steps += 1
                continue
            loss_delta = abs(row["loss"] - ref["loss"])
            sum_delta = abs(row["param_sum"] - ref["param_sum"])
            sumsq_delta = abs(row["param_sumsq"] - ref["param_sumsq"])
            maxabs_delta = abs(row["param_maxabs"] - ref["param_maxabs"])
            sum_delta_per_param = sum_delta / param_count
            sumsq_delta_per_param = sumsq_delta / param_count
            sum_span_per_param = row["sum_span"] / param_count
            sumsq_span_per_param = row["sumsq_span"] / param_count
            hash_match = row["param_hash"] == ref["param_hash"]
            rank_span_ok = (
                sum_span_per_param <= args.rank_sum_span_per_param_tol and
                sumsq_span_per_param <= args.rank_sumsq_span_per_param_tol and
                row["maxabs_span"] <= args.maxabs_tol
            )
            values_ok = all(finite(row[field]) for field in [
                "loss", "param_sum", "param_sumsq", "param_maxabs",
                "sum_span", "sumsq_span", "maxabs_span",
            ])
            tolerances_ok = (
                loss_delta <= args.loss_tol and
                sum_delta_per_param <= args.sum_per_param_tol and
                sumsq_delta_per_param <= args.sumsq_per_param_tol and
                maxabs_delta <= args.maxabs_tol and
                rank_span_ok and
                values_ok
            )
            if not values_ok:
                invalid_values += 1
            if not hash_match:
                hash_mismatches += 1
            if not tolerances_ok:
                failed_steps += 1

            max_loss_delta = max(max_loss_delta, loss_delta)
            max_sum_delta = max(max_sum_delta, sum_delta)
            max_sumsq_delta = max(max_sumsq_delta, sumsq_delta)
            max_maxabs_delta = max(max_maxabs_delta, maxabs_delta)
            max_sum_span = max(max_sum_span, row["sum_span"])
            max_sumsq_span = max(max_sumsq_span, row["sumsq_span"])
            max_maxabs_span = max(max_maxabs_span, row["maxabs_span"])

            out = dict(row)
            out.update({
                "ref_case": REFERENCE_CASE,
                "loss_abs_delta": loss_delta,
                "sum_abs_delta": sum_delta,
                "sum_abs_delta_per_param": sum_delta_per_param,
                "sumsq_abs_delta": sumsq_delta,
                "sumsq_abs_delta_per_param": sumsq_delta_per_param,
                "maxabs_abs_delta": maxabs_delta,
                "sum_span_per_param": sum_span_per_param,
                "sumsq_span_per_param": sumsq_span_per_param,
                "hash_matches_ref": "yes" if hash_match else "no",
                "rank_span_ok": "yes" if rank_span_ok else "no",
                "step_valid": "yes" if tolerances_ok else "no",
            })
            detailed.append(out)

        final = rows[-1] if rows else {}
        case_valid = complete and failed_steps == 0
        summary.append({
            "case": case_name,
            "valid": "yes" if case_valid else "no",
            "steps": len(rows),
            "expected_steps": args.steps,
            "max_loss_abs_delta": max_loss_delta,
            "max_sum_abs_delta": max_sum_delta,
            "max_sum_abs_delta_per_param": max_sum_delta / param_count,
            "max_sumsq_abs_delta": max_sumsq_delta,
            "max_sumsq_abs_delta_per_param": max_sumsq_delta / param_count,
            "max_maxabs_abs_delta": max_maxabs_delta,
            "hash_mismatches": hash_mismatches,
            "max_sum_span": max_sum_span,
            "max_sum_span_per_param": max_sum_span / param_count,
            "max_sumsq_span": max_sumsq_span,
            "max_sumsq_span_per_param": max_sumsq_span / param_count,
            "max_maxabs_span": max_maxabs_span,
            "failed_steps": failed_steps,
            "invalid_values": invalid_values,
            "final_loss": final.get("loss", ""),
            "final_param_sum": final.get("param_sum", ""),
            "final_param_sumsq": final.get("param_sumsq", ""),
            "final_param_hash": final.get("param_hash", ""),
        })

    return detailed, summary


def write_csv(path, rows):
    if not rows:
        raise SystemExit(f"no rows for {path}")
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, tag, summary, args):
    all_valid = all(row["valid"] == "yes" for row in summary)
    hash_exact = all(int(row["hash_mismatches"]) == 0 for row in summary)
    lines = [
        f"# Numerical Trajectory Validation: Job {tag}",
        "",
        f"Overall result: {'PASS' if all_valid else 'FAIL'}",
        "",
        "This run compares blocking CUDA-aware MPI, host-pinned overlap, and",
        "OpenMP communication-thread overlap using the same seed, data order,",
        "rank count, batch size, and learning rate. The trace is correctness",
        "instrumentation only; it should not be used for throughput claims.",
        "",
        "Tolerances:",
        "",
        f"- loss: `{args.loss_tol:g}`",
        f"- parameter sum per parameter: `{args.sum_per_param_tol:g}`",
        f"- parameter squared-sum per parameter: `{args.sumsq_per_param_tol:g}`",
        f"- max parameter magnitude: `{args.maxabs_tol:g}`",
        f"- cross-rank sum span per parameter: `{args.rank_sum_span_per_param_tol:g}`",
        f"- cross-rank squared-sum span per parameter: `{args.rank_sumsq_span_per_param_tol:g}`",
        "",
        f"Exact bitwise hash agreement across modes: {'yes' if hash_exact else 'no'}",
        "",
        "| Case | Valid | Steps | Max loss delta | Max sum delta/param | Max sumsq delta/param | Hash mismatches | Max rank sum span/param | Final hash |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['case']} | {'PASS' if row['valid'] == 'yes' else 'FAIL'} | "
            f"{row['steps']}/{row['expected_steps']} | "
            f"{float(row['max_loss_abs_delta']):.3e} | "
            f"{float(row['max_sum_abs_delta_per_param']):.3e} | "
            f"{float(row['max_sumsq_abs_delta_per_param']):.3e} | "
            f"{row['hash_mismatches']} | "
            f"{float(row['max_sum_span_per_param']):.3e} | "
            f"`{row['final_param_hash']}` |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- Matching trajectories mean the overlap variants changed scheduling and",
        "  communication exposure, not the optimization path.",
        "- The per-step rank spans check that all ranks stayed synchronized after",
        "  each optimizer update.",
        "- Bitwise hashes may differ because the communication paths can change",
        "  floating-point ordering; the normalized FP tolerances are the formal",
        "  pass/fail criterion.",
        "",
    ]
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id")
    parser.add_argument("--log")
    parser.add_argument("--tag")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--loss-tol", type=float, default=1e-4)
    parser.add_argument("--sum-per-param-tol", type=float, default=1e-6)
    parser.add_argument("--sumsq-per-param-tol", type=float, default=2e-8)
    parser.add_argument("--maxabs-tol", type=float, default=2e-6)
    parser.add_argument("--rank-sum-span-per-param-tol", type=float, default=1e-9)
    parser.add_argument("--rank-sumsq-span-per-param-tol", type=float, default=1e-9)
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    elif args.job_id:
        log_path = LOGS / f"trajectory_validation_{args.job_id}.out"
    else:
        raise SystemExit("pass --job-id or --log")
    if not log_path.exists():
        raise SystemExit(f"missing log: {log_path}")

    tag = args.tag or args.job_id or log_path.stem
    cases = parse_log(log_path)
    detailed, summary = compare_cases(cases, args)

    detail_path = RESULTS / f"trajectory_validation_{tag}.csv"
    summary_path = RESULTS / f"trajectory_validation_summary_{tag}.csv"
    md_path = RESULTS / f"trajectory_validation_{tag}.md"
    write_csv(detail_path, detailed)
    write_csv(summary_path, summary)
    write_md(md_path, tag, summary, args)

    (RESULTS / "trajectory_validation.csv").write_text(detail_path.read_text())
    (RESULTS / "trajectory_validation_summary.csv").write_text(summary_path.read_text())
    (RESULTS / "trajectory_validation.md").write_text(md_path.read_text())

    if not all(row["valid"] == "yes" for row in summary):
        raise SystemExit("one or more trajectory validations failed")
    print(f"Wrote {detail_path.relative_to(ROOT)}")
    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
