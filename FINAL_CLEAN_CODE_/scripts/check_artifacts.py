#!/usr/bin/env python3
"""Sanity-check generated experiment artifacts for the final package."""

import csv
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
PLOTS = ROOT / "plots"
PROFILES = ROOT / "profiles"
PACKAGE_ROOT = ROOT.parent


failures = []
warnings = []


def fail(msg):
    failures.append(msg)


def warn(msg):
    warnings.append(msg)


def require_file(path):
    if not path.exists():
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path.relative_to(PACKAGE_ROOT)
        fail(f"missing file: {rel}")
        return False
    if path.stat().st_size == 0:
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            rel = path.relative_to(PACKAGE_ROOT)
        fail(f"empty file: {rel}")
        return False
    return True


def read_csv(path):
    if not require_file(path):
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check_job_log(path, marker):
    if not require_file(path):
        return
    text = path.read_text(errors="replace")
    if marker not in text:
        fail(f"missing completion marker in {path.relative_to(ROOT)}: {marker}")


def check_no_bad_tokens(path):
    if not require_file(path):
        return
    text = path.read_text(errors="replace")
    if re.search(r"FAIL|(^|[^A-Za-z])(nan|inf)([^A-Za-z]|$)", text, re.I):
        fail(f"bad token found in {path.relative_to(ROOT)}")


def check_weak_scaling():
    rows = read_csv(RESULTS / "weak_scaling_summary.csv")
    expected = {1, 2, 4}
    seen = {int(r["ranks"]) for r in rows if r.get("ranks")}
    if seen != expected:
        fail(f"weak scaling ranks mismatch: expected {expected}, saw {seen}")
    for row in rows:
        rank = int(row["ranks"])
        if row["all_valid"] != "yes":
            fail(f"weak scaling rank {rank} not all valid")
        if int(row["valid_runs"]) < 3:
            fail(f"weak scaling rank {rank} has too few valid runs")
        for field in ["throughput_mean_mtok_s", "throughput_std_mtok_s",
                      "step_time_mean_ms", "weak_efficiency"]:
            if not finite(row[field]):
                fail(f"weak scaling rank {rank} non-finite {field}")
        eff = float(row["weak_efficiency"])
        if eff <= 0.0 or eff > 1.05:
            fail(f"weak scaling rank {rank} suspicious efficiency {eff}")
    check_job_log(LOGS / "weak_scaling_88902.out", "=== DONE job=88902 status=0 ===")
    require_file(PLOTS / "weak_scaling_efficiency.svg")


def check_strong_scaling_repeated():
    rows = read_csv(RESULTS / "strong_scaling_repeated.csv")
    job_rows = read_csv(RESULTS / "strong_scaling_repeated_89551.csv")
    raw_rows = read_csv(RESULTS / "strong_scaling_repeated_89551_raw.csv")
    require_file(RESULTS / "strong_scaling_repeated.md")
    require_file(RESULTS / "strong_scaling_repeated_89551.md")
    require_file(PLOTS / "strong_scaling_repeated_speedup.svg")
    require_file(PLOTS / "strong_scaling_repeated_efficiency.svg")
    require_file(PLOTS / "strong_scaling_repeated_speedup_89551.svg")
    require_file(PLOTS / "strong_scaling_repeated_efficiency_89551.svg")

    expected = {
        (1, "blocking"),
        (2, "blocking"),
        (2, "openmp_thread"),
        (4, "blocking"),
        (4, "openmp_thread"),
    }
    for label, data in [("canonical", rows), ("job 89551", job_rows)]:
        seen = {(int(row["ranks"]), row["backend"]) for row in data if row.get("ranks")}
        if seen != expected:
            fail(f"fresh strong scaling {label} rows mismatch: expected {expected}, saw {seen}")
        if len(data) != 5:
            fail(f"fresh strong scaling {label} expected 5 rows, saw {len(data)}")
        for row in data:
            ranks = int(row["ranks"])
            backend = row["backend"]
            if int(row["hidden"]) != 256:
                fail(f"fresh strong scaling {backend} ranks={ranks} expected h256")
            if int(row["total_batch"]) != 32:
                fail(f"fresh strong scaling {backend} ranks={ranks} expected total batch 32")
            if int(row["local_batch"]) != 32 // ranks:
                fail(f"fresh strong scaling {backend} ranks={ranks} local batch mismatch")
            if int(row["runs"]) != 4 or int(row["summary_runs"]) != 3:
                fail(f"fresh strong scaling {backend} ranks={ranks} repeat counts wrong")
            if int(row["valid_runs"]) != 3 or row["all_valid"] != "yes":
                fail(f"fresh strong scaling {backend} ranks={ranks} not all valid")
            for field in [
                "throughput_mean_mtok_s",
                "throughput_std_mtok_s",
                "time_mean_ms",
                "time_std_ms",
                "step_time_mean_ms",
                "step_time_std_ms",
                "max_checksum_span",
                "strong_speedup",
                "strong_efficiency",
                "parallel_overhead_ms",
                "karp_flatt_serial_fraction",
            ]:
                if not finite(row[field]):
                    fail(f"fresh strong scaling {backend} ranks={ranks} non-finite {field}")
            speedup = float(row["strong_speedup"])
            efficiency = float(row["strong_efficiency"])
            if speedup <= 0.0 or efficiency <= 0.0 or efficiency > 1.05:
                fail(f"fresh strong scaling {backend} ranks={ranks} invalid speedup/efficiency")
            try:
                checksum = float(row["max_checksum_span"])
                if checksum > 1e-4:
                    fail(f"fresh strong scaling {backend} ranks={ranks} checksum too large: {checksum}")
            except ValueError:
                fail(f"fresh strong scaling {backend} ranks={ranks} missing checksum")

    if len(raw_rows) != 20:
        fail(f"fresh strong scaling expected 20 raw rows, saw {len(raw_rows)}")
    for row in raw_rows:
        if row["valid"] != "yes":
            fail(f"fresh strong scaling raw row invalid: {row}")

    by_key = {(int(row["ranks"]), row["backend"]): row for row in rows}
    if abs(float(by_key[(1, "blocking")]["strong_speedup"]) - 1.0) > 1e-6:
        fail("fresh strong scaling 1-rank baseline speedup should be 1")
    if float(by_key[(4, "openmp_thread")]["strong_efficiency"]) > 0.20:
        fail("fresh strong scaling h256 fixed-batch efficiency unexpectedly high")
    if float(by_key[(4, "openmp_thread")]["throughput_mean_mtok_s"]) < 0.60:
        fail("fresh strong scaling 4-rank OpenMP throughput unexpectedly low")

    check_job_log(LOGS / "strong_scaling_repeated_89551.out",
                  "=== DONE strong scaling repeated job=89551 status=0 ===")
    check_no_bad_tokens(LOGS / "strong_scaling_repeated_89551.out")
    check_no_bad_tokens(LOGS / "strong_scaling_repeated_89551_raw.txt")
    err = LOGS / "strong_scaling_repeated_89551.err"
    if err.exists() and err.stat().st_size > 0:
        warn("strong_scaling_repeated_89551.err is non-empty; inspect before submission")


def check_alpha_beta():
    rows = read_csv(RESULTS / "allreduce_alpha_beta_fit.csv")
    keys = {(r["backend"], int(r["ranks"])) for r in rows if r.get("backend")}
    expected = {
        ("device", 2),
        ("device", 4),
        ("host_pinned", 2),
        ("host_pinned", 4),
    }
    if keys != expected:
        fail(f"alpha/beta fit keys mismatch: expected {expected}, saw {keys}")
    for row in rows:
        backend = row["backend"]
        ranks = int(row["ranks"])
        r2 = float(row["r2"])
        beta = float(row["beta_ms_per_byte"])
        alpha = float(row["alpha_ms"])
        if r2 < 0.99:
            fail(f"alpha/beta {backend} ranks={ranks} low R^2={r2}")
        if beta <= 0.0:
            fail(f"alpha/beta {backend} ranks={ranks} non-positive beta")
        if backend == "device" and alpha <= 0.0:
            fail(f"device alpha should be positive for ranks={ranks}")
        if backend == "host_pinned" and alpha < 0.0:
            warn(f"host_pinned ranks={ranks} negative alpha; document model limitation")
    summary = read_csv(RESULTS / "allreduce_alpha_beta_summary.csv")
    if len(summary) != 56:
        fail(f"expected 56 alpha/beta summary rows, saw {len(summary)}")
    check_job_log(LOGS / "allreduce_alpha_beta_88903.out", "=== DONE job=88903 status=0 ===")
    require_file(PLOTS / "allreduce_alpha_beta.svg")


def check_breakdown_and_prediction():
    breakdown = read_csv(RESULTS / "comm_breakdown_preliminary.csv")
    prediction = read_csv(RESULTS / "allreduce_prediction_vs_measurement.csv")
    for name, rows in [("breakdown", breakdown), ("prediction", prediction)]:
        seen = {int(r["hidden"]) for r in rows if r.get("hidden")}
        if seen != {128, 256, 512}:
            fail(f"{name} hidden sizes mismatch: {seen}")
    for row in breakdown:
        hidden = int(row["hidden"])
        frac = float(row["exposed_wait_fraction"])
        if frac <= 0.0 or frac >= 1.0:
            fail(f"h{hidden} exposed wait fraction out of range: {frac}")
    for row in prediction:
        hidden = int(row["hidden"])
        pred = float(row["pred_bucketed_ms"])
        block = float(row["measured_blocking_comm_ms"])
        if pred <= 0.0 or block <= 0.0:
            fail(f"h{hidden} invalid prediction/blocking values")
    require_file(PLOTS / "comm_breakdown_preliminary.svg")
    require_file(PLOTS / "allreduce_prediction_vs_measurement.svg")


def check_overlap_speedup():
    rows = read_csv(RESULTS / "overlap_speedup_by_hidden_speedup.csv")
    seen = {int(r["hidden"]) for r in rows if r.get("hidden")}
    if seen != {128, 256, 512}:
        fail(f"overlap-speedup hidden sizes mismatch: {seen}")
    if len(rows) != 3:
        fail(f"expected 3 overlap-speedup rows, saw {len(rows)}")

    by_hidden = {int(row["hidden"]): row for row in rows if row.get("hidden")}
    for hidden, row in by_hidden.items():
        if row["all_valid"] != "yes":
            fail(f"h{hidden} overlap-speedup group not all valid")
        if int(row["blocking_valid_runs"]) < 4 or int(row["openmp_valid_runs"]) < 4:
            fail(f"h{hidden} overlap-speedup has too few valid repeats")
        for field in [
            "blocking_throughput_mean_mtok_s",
            "blocking_throughput_std_mtok_s",
            "openmp_throughput_mean_mtok_s",
            "openmp_throughput_std_mtok_s",
            "throughput_speedup",
            "throughput_speedup_std",
            "blocking_step_time_mean_ms",
            "openmp_step_time_mean_ms",
            "blocking_comm_sync_mean_ms",
            "openmp_exposed_wait_mean_ms",
        ]:
            if not finite(row[field]):
                fail(f"h{hidden} overlap-speedup non-finite {field}")
        speedup = float(row["throughput_speedup"])
        if speedup <= 0.0:
            fail(f"h{hidden} overlap-speedup non-positive speedup {speedup}")
        try:
            checksum = float(row["max_checksum_span"])
            if checksum > 1e-4:
                fail(f"h{hidden} overlap-speedup checksum span too large: {checksum}")
        except ValueError:
            fail(f"h{hidden} overlap-speedup missing checksum span")

    if 128 in by_hidden and not (0.90 <= float(by_hidden[128]["throughput_speedup"]) <= 1.05):
        fail(f"h128 overlap speedup outside expected launch-overhead/no-benefit band: {by_hidden[128]['throughput_speedup']}")
    if 256 in by_hidden and float(by_hidden[256]["throughput_speedup"]) < 1.25:
        fail(f"h256 overlap speedup lower than expected: {by_hidden[256]['throughput_speedup']}")
    if 512 in by_hidden and float(by_hidden[512]["throughput_speedup"]) < 1.20:
        fail(f"h512 overlap speedup lower than expected: {by_hidden[512]['throughput_speedup']}")

    check_job_log(LOGS / "overlap_speedup_by_hidden_89072.out", "=== DONE job=89072 status=0 ===")
    check_no_bad_tokens(LOGS / "overlap_speedup_by_hidden_89072.out")
    require_file(RESULTS / "overlap_speedup_by_hidden.md")
    require_file(PLOTS / "overlap_speedup_by_hidden.svg")


def check_bucket_ucurve_h256():
    rows = read_csv(RESULTS / "bucket_ucurve_h256.csv")
    expected_buckets = {64, 128, 256, 512, 1024, 2048, 4096}
    expected_backends = {"openmp_thread", "pinned"}
    seen_backends = {row["backend"] for row in rows}
    seen_buckets = {int(row["bucket_kb"]) for row in rows if row.get("bucket_kb")}
    if seen_backends != expected_backends:
        fail(f"h256 bucket U-curve backend mismatch: expected {expected_backends}, saw {seen_backends}")
    if seen_buckets != expected_buckets:
        fail(f"h256 bucket U-curve bucket mismatch: expected {expected_buckets}, saw {seen_buckets}")
    if len(rows) != len(expected_backends) * len(expected_buckets):
        fail(f"expected 14 h256 bucket U-curve rows, saw {len(rows)}")

    by_backend_bucket = {}
    for row in rows:
        hidden = int(row["hidden"])
        backend = row["backend"]
        bucket = int(row["bucket_kb"])
        by_backend_bucket[(backend, bucket)] = row
        if hidden != 256:
            fail(f"bucket U-curve row has hidden={hidden}, expected 256")
        if row["all_valid"] != "yes":
            fail(f"h256 bucket U-curve {backend} bucket={bucket} not all valid")
        if int(row["valid_runs"]) < 4:
            fail(f"h256 bucket U-curve {backend} bucket={bucket} too few valid repeats")
        for field in [
            "throughput_mean_mtok_s",
            "throughput_std_mtok_s",
            "time_mean_ms",
            "exposed_wait_ms",
            "speedup_vs_blocking",
            "comm_tail_reduction_pct",
            "alpha_beta_pred_comm_ms",
        ]:
            if not finite(row[field]):
                fail(f"h256 bucket U-curve {backend} bucket={bucket} non-finite {field}")
        try:
            checksum = float(row["max_checksum_span"])
            if checksum > 1e-4:
                fail(f"h256 bucket U-curve {backend} bucket={bucket} checksum too large: {checksum}")
        except ValueError:
            fail(f"h256 bucket U-curve {backend} bucket={bucket} missing checksum")

    openmp_best = max(
        (row for row in rows if row["backend"] == "openmp_thread"),
        key=lambda row: float(row["throughput_mean_mtok_s"]),
        default=None,
    )
    pinned_best = max(
        (row for row in rows if row["backend"] == "pinned"),
        key=lambda row: float(row["throughput_mean_mtok_s"]),
        default=None,
    )
    if not openmp_best or int(openmp_best["bucket_kb"]) != 2048:
        fail("h256 OpenMP U-curve best bucket should be 2048 KB")
    if openmp_best and float(openmp_best["throughput_mean_mtok_s"]) < 2.30:
        fail(f"h256 OpenMP 2048 KB throughput too low: {openmp_best['throughput_mean_mtok_s']}")
    if not pinned_best or int(pinned_best["bucket_kb"]) != 512:
        fail("h256 pinned U-curve best bucket should be 512 KB")
    pinned_4096 = by_backend_bucket.get(("pinned", 4096))
    if pinned_4096 and float(pinned_4096["speedup_vs_blocking"]) >= 1.0:
        fail("h256 pinned 4096 KB should be slower than blocking in this U-curve")

    check_job_log(LOGS / "comm_thread_sweep_89083.out", "=== DONE job=89083 status=0 ===")
    check_no_bad_tokens(LOGS / "comm_thread_sweep_89083.out")
    require_file(RESULTS / "training_bucket_sweep_summary_comm_thread_89083.csv")
    require_file(RESULTS / "bucket_ucurve_h256.md")
    require_file(PLOTS / "bucket_ucurve_h256.svg")


def check_bucket_ucurve_h512():
    rows = read_csv(RESULTS / "bucket_ucurve_h512.csv")
    expected_buckets = {512, 1024, 2048, 4096, 8192, 16384}
    expected_backends = {"openmp_thread", "pinned"}
    seen_backends = {row["backend"] for row in rows}
    seen_buckets = {int(row["bucket_kb"]) for row in rows if row.get("bucket_kb")}
    if seen_backends != expected_backends:
        fail(f"h512 bucket U-curve backend mismatch: expected {expected_backends}, saw {seen_backends}")
    if seen_buckets != expected_buckets:
        fail(f"h512 bucket U-curve bucket mismatch: expected {expected_buckets}, saw {seen_buckets}")
    if len(rows) != len(expected_backends) * len(expected_buckets):
        fail(f"expected 12 h512 bucket U-curve rows, saw {len(rows)}")

    by_backend_bucket = {}
    for row in rows:
        hidden = int(row["hidden"])
        backend = row["backend"]
        bucket = int(row["bucket_kb"])
        by_backend_bucket[(backend, bucket)] = row
        if hidden != 512:
            fail(f"bucket U-curve row has hidden={hidden}, expected 512")
        if row["all_valid"] != "yes":
            fail(f"h512 bucket U-curve {backend} bucket={bucket} not all valid")
        if int(row["valid_runs"]) < 4:
            fail(f"h512 bucket U-curve {backend} bucket={bucket} too few valid repeats")
        for field in [
            "throughput_mean_mtok_s",
            "throughput_std_mtok_s",
            "time_mean_ms",
            "exposed_wait_ms",
            "speedup_vs_blocking",
            "comm_tail_reduction_pct",
            "alpha_beta_pred_comm_ms",
        ]:
            if not finite(row[field]):
                fail(f"h512 bucket U-curve {backend} bucket={bucket} non-finite {field}")
        try:
            checksum = float(row["max_checksum_span"])
            if checksum > 1e-4:
                fail(f"h512 bucket U-curve {backend} bucket={bucket} checksum too large: {checksum}")
        except ValueError:
            fail(f"h512 bucket U-curve {backend} bucket={bucket} missing checksum")

    openmp_best = max(
        (row for row in rows if row["backend"] == "openmp_thread"),
        key=lambda row: float(row["throughput_mean_mtok_s"]),
        default=None,
    )
    pinned_best = max(
        (row for row in rows if row["backend"] == "pinned"),
        key=lambda row: float(row["throughput_mean_mtok_s"]),
        default=None,
    )
    if not openmp_best or int(openmp_best["bucket_kb"]) not in {2048, 4096}:
        fail("h512 OpenMP U-curve best bucket should be in the measured 2048-4096 KB plateau")
    if openmp_best and float(openmp_best["throughput_mean_mtok_s"]) < 0.85:
        fail(f"h512 OpenMP best-bucket throughput too low: {openmp_best['throughput_mean_mtok_s']}")
    if openmp_best and float(openmp_best["speedup_vs_blocking"]) < 1.6:
        fail(f"h512 OpenMP best-bucket speedup too low: {openmp_best['speedup_vs_blocking']}")
    if not pinned_best or int(pinned_best["bucket_kb"]) != 8192:
        fail("h512 pinned U-curve best bucket should be 8192 KB")
    openmp_16384 = by_backend_bucket.get(("openmp_thread", 16384))
    pinned_16384 = by_backend_bucket.get(("pinned", 16384))
    if openmp_best and openmp_16384:
        if float(openmp_16384["throughput_mean_mtok_s"]) >= float(openmp_best["throughput_mean_mtok_s"]):
            fail("h512 OpenMP 16384 KB should be slower than the measured best bucket")
    if pinned_16384 and float(pinned_16384["speedup_vs_blocking"]) >= 1.0:
        fail("h512 pinned 16384 KB should be slower than blocking in this U-curve")

    check_job_log(LOGS / "comm_thread_sweep_89243.out", "=== DONE job=89243 status=0 ===")
    check_no_bad_tokens(LOGS / "comm_thread_sweep_89243.out")
    require_file(RESULTS / "training_bucket_sweep_summary_comm_thread_89243.csv")
    require_file(RESULTS / "bucket_ucurve_h512_89243.csv")
    require_file(RESULTS / "bucket_ucurve_h512_89243.md")
    require_file(RESULTS / "bucket_ucurve_h512.md")
    require_file(PLOTS / "bucket_ucurve_h512_89243.svg")
    require_file(PLOTS / "bucket_ucurve_h512.svg")


def check_nsys_timeline_h256_openmp():
    summary = read_csv(RESULTS / "nsys_timeline_h256_openmp_summary.csv")
    timeline = read_csv(RESULTS / "nsys_timeline_h256_openmp_timeline.csv")
    expected_metrics = {
        "forward_ms",
        "backward_bucketed_ms",
        "adam_update_ms",
        "worker_wait_total_ms",
        "worker_allreduce_total_ms",
        "exposed_tail_after_last_allreduce_ms",
        "worker_allreduce_count",
    }
    seen_metrics = {row["metric"] for row in summary}
    if seen_metrics != expected_metrics:
        fail(f"Nsight timeline metrics mismatch: expected {expected_metrics}, saw {seen_metrics}")
    by_metric = {row["metric"]: row for row in summary}
    for metric, row in by_metric.items():
        if int(row["steps"]) != 19:
            fail(f"Nsight timeline metric {metric} expected 19 warmup-dropped steps")
        for field in ["mean_ms", "median_ms", "std_ms"]:
            if not finite(row[field]):
                fail(f"Nsight timeline metric {metric} non-finite {field}")
    if "worker_allreduce_count" in by_metric:
        if abs(float(by_metric["worker_allreduce_count"]["median_ms"]) - 2.0) > 1e-6:
            fail("Nsight timeline should show two worker Allreduces per representative h256 step")
    if "worker_allreduce_total_ms" in by_metric:
        if float(by_metric["worker_allreduce_total_ms"]["median_ms"]) <= 1.0:
            fail("Nsight timeline worker Allreduce median is unexpectedly small")
    if "exposed_tail_after_last_allreduce_ms" in by_metric:
        tail = float(by_metric["exposed_tail_after_last_allreduce_ms"]["median_ms"])
        if tail < 0.0 or tail > 0.25:
            fail(f"Nsight timeline exposed tail median out of expected range: {tail}")

    timeline_events = {row["event"] for row in timeline}
    required_events = {
        "forward",
        "backward_bucketed",
        "exposed_tail_after_last_allreduce",
        "adam_update",
        "event_wait_1",
        "event_wait_2",
        "mpi_allreduce_1",
        "mpi_allreduce_2",
    }
    if timeline_events != required_events:
        fail(f"Nsight representative timeline events mismatch: expected {required_events}, saw {timeline_events}")
    for row in timeline:
        if not finite(row["start_ms"]) or not finite(row["end_ms"]) or not finite(row["duration_ms"]):
            fail(f"Nsight representative timeline non-finite event timing: {row}")
        if float(row["duration_ms"]) < 0.0:
            fail(f"Nsight representative timeline negative duration: {row}")

    for backend in ["direct", "pinned", "openmp_thread"]:
        require_file(PROFILES / f"h256_comm_{backend}_89088_rank0.sqlite")
        require_file(PROFILES / f"h256_comm_{backend}_89088_rank0.nsys-rep")
        require_file(RESULTS / f"h256_comm_{backend}_89088_rank0_nsys_summary.csv")
    check_job_log(LOGS / "profile_comm_thread_pair_89088.out",
                  "=== DONE profile comm-thread pair job=89088 ===")
    require_file(RESULTS / "nsys_timeline_h256_openmp.md")
    require_file(PLOTS / "nsys_timeline_h256_openmp.svg")


def check_nsys_hidden_breakdown():
    rows = read_csv(RESULTS / "nsys_hidden_breakdown.csv")
    job_rows = read_csv(RESULTS / "nsys_hidden_breakdown_89529.csv")
    steps = read_csv(RESULTS / "nsys_hidden_breakdown_steps.csv")
    require_file(RESULTS / "nsys_hidden_breakdown_89529_steps.csv")
    require_file(RESULTS / "nsys_hidden_breakdown.md")
    require_file(RESULTS / "nsys_hidden_breakdown_89529.md")
    require_file(PLOTS / "nsys_hidden_breakdown.svg")
    require_file(PLOTS / "nsys_hidden_breakdown_89529.svg")

    expected_hidden = {128, 256, 512}
    expected_buckets = {128: 256, 256: 2048, 512: 2048}
    for label, data in [("canonical", rows), ("job 89529", job_rows)]:
        seen = {int(row["hidden"]) for row in data if row.get("hidden")}
        if seen != expected_hidden:
            fail(f"Nsys hidden breakdown {label} hidden sizes mismatch: {seen}")
        if len(data) != 3:
            fail(f"Nsys hidden breakdown {label} expected 3 rows, saw {len(data)}")
        for row in data:
            hidden = int(row["hidden"])
            if int(row["tag"]) != 89529:
                fail(f"Nsys hidden breakdown h{hidden} unexpected tag {row['tag']}")
            if int(row["ranks"]) != 4:
                fail(f"Nsys hidden breakdown h{hidden} expected 4 ranks")
            if int(row["batch"]) != 32:
                fail(f"Nsys hidden breakdown h{hidden} expected batch 32")
            if int(row["analyzed_steps"]) != 19:
                fail(f"Nsys hidden breakdown h{hidden} expected 19 analyzed steps")
            if int(row["bucket_kb"]) != expected_buckets[hidden]:
                fail(f"Nsys hidden breakdown h{hidden} unexpected bucket {row['bucket_kb']}")
            for field in [
                "main_step_mean_ms",
                "main_step_std_ms",
                "forward_mean_ms",
                "backward_bucketed_mean_ms",
                "finish_async_gradient_syncs_mean_ms",
                "finish_async_gradient_syncs_std_ms",
                "adam_update_mean_ms",
                "compute_without_finish_mean_ms",
                "worker_event_wait_mean_ms",
                "worker_allreduce_total_mean_ms",
                "worker_allreduce_total_std_ms",
                "worker_allreduce_count_mean",
                "finish_fraction_of_step",
                "worker_comm_to_backward",
                "overlap_fraction_proxy",
            ]:
                if not finite(row[field]):
                    fail(f"Nsys hidden breakdown h{hidden} non-finite {field}")
            main = float(row["main_step_mean_ms"])
            finish = float(row["finish_async_gradient_syncs_mean_ms"])
            worker = float(row["worker_allreduce_total_mean_ms"])
            finish_frac = float(row["finish_fraction_of_step"])
            overlap_proxy = float(row["overlap_fraction_proxy"])
            if main <= 0.0 or finish <= 0.0 or worker <= 0.0:
                fail(f"Nsys hidden breakdown h{hidden} non-positive timing")
            if finish_frac <= 0.0 or finish_frac >= 1.0:
                fail(f"Nsys hidden breakdown h{hidden} finish fraction out of range: {finish_frac}")
            if overlap_proxy <= -1.0 or overlap_proxy >= 1.0:
                fail(f"Nsys hidden breakdown h{hidden} overlap proxy out of sanity range: {overlap_proxy}")
            if hidden == 128 and abs(float(row["worker_allreduce_count_mean"]) - 4.0) > 1e-6:
                fail("Nsys hidden breakdown h128 expected 4 worker Allreduces")
            if hidden == 256 and abs(float(row["worker_allreduce_count_mean"]) - 2.0) > 1e-6:
                fail("Nsys hidden breakdown h256 expected 2 worker Allreduces")
            if hidden == 512 and abs(float(row["worker_allreduce_count_mean"]) - 4.0) > 1e-6:
                fail("Nsys hidden breakdown h512 expected 4 worker Allreduces")

    if len(steps) != 60:
        fail(f"Nsys hidden breakdown expected 60 per-step rows, saw {len(steps)}")
    by_hidden_steps = {}
    for row in steps:
        hidden = int(row["hidden"])
        by_hidden_steps.setdefault(hidden, set()).add(int(row["step"]))
        for field in [
            "forward_ms",
            "backward_bucketed_ms",
            "finish_async_gradient_syncs_ms",
            "adam_update_ms",
            "main_step_ms",
            "worker_event_wait_total_ms",
            "worker_allreduce_total_ms",
            "worker_allreduce_count",
        ]:
            if not finite(row[field]):
                fail(f"Nsys hidden breakdown h{hidden} step={row.get('step')} non-finite {field}")
    for hidden in expected_hidden:
        if by_hidden_steps.get(hidden) != set(range(1, 21)):
            fail(f"Nsys hidden breakdown h{hidden} step set mismatch: {by_hidden_steps.get(hidden)}")

    for hidden in [128, 256, 512]:
        require_file(RESULTS / f"h{hidden}_openmp_breakdown_89529_rank0_nsys_stats.csv")
        require_file(RESULTS / f"h{hidden}_openmp_breakdown_89529_rank0_nsys_summary.csv")
        for rank in range(4):
            require_file(PROFILES / f"h{hidden}_openmp_breakdown_89529_rank{rank}.sqlite")
            require_file(PROFILES / f"h{hidden}_openmp_breakdown_89529_rank{rank}.nsys-rep")

    check_job_log(LOGS / "nsys_hidden_breakdown_89529.out",
                  "=== DONE nsys hidden breakdown job=89529 status=0 ===")
    require_file(LOGS / "nsys_hidden_breakdown_89529_raw.txt")
    err = LOGS / "nsys_hidden_breakdown_89529.err"
    if err.exists() and err.stat().st_size > 0:
        warn("nsys_hidden_breakdown_89529.err is non-empty; inspect before submission")


def check_gemm_precision():
    rows = read_csv(RESULTS / "gemm_precision_validation.csv")
    expected_cases = {
        "default_auto_cublas_tc",
        "cublas_strict_fp32",
        "custom_strict_fp32",
    }
    cases = {row["case"] for row in rows}
    if cases != expected_cases:
        fail(f"GEMM precision cases mismatch: expected {expected_cases}, saw {cases}")
    if len(rows) != 12:
        fail(f"expected 12 GEMM precision rows, saw {len(rows)}")
    for row in rows:
        case = row["case"]
        size = int(row["M"])
        if row["valid"] != "yes":
            fail(f"GEMM precision {case} M={size} invalid")
        if row["gemm_tiled_status"] != "PASS":
            fail(f"GEMM precision {case} M={size} did not pass")
        if not finite(row["gemm_tiled_error"]):
            fail(f"GEMM precision {case} M={size} non-finite error")

    by_case_size = {(row["case"], int(row["M"])): row for row in rows}
    default_512 = float(by_case_size[("default_auto_cublas_tc", 512)]["gemm_tiled_error"])
    custom_512 = float(by_case_size[("custom_strict_fp32", 512)]["gemm_tiled_error"])
    cublas_512 = float(by_case_size[("cublas_strict_fp32", 512)]["gemm_tiled_error"])
    if not (1e-3 < default_512 < 5e-3):
        fail(f"default auto/cublas_tc 512 GEMM error outside expected non-strict band: {default_512}")
    if not (custom_512 < 1e-5):
        fail(f"custom strict FP32 512 GEMM error too large: {custom_512}")
    if not (cublas_512 < 1e-4):
        fail(f"cuBLAS strict FP32 512 GEMM error too large: {cublas_512}")

    check_job_log(LOGS / "gemm_precision_validation_89068.out", "=== DONE job=89068 status=0 ===")
    check_no_bad_tokens(LOGS / "gemm_precision_validation_89068.out")
    require_file(RESULTS / "gemm_precision_validation.md")


def check_validation():
    main_log = LOGS / "full_validation_88905.out"
    check_job_log(main_log, "=== FULL VALIDATION PASSED ===")
    check_no_bad_tokens(main_log)
    required = [
        "test_gemm",
        "test_attention",
        "test_layernorm",
        "test_model_reference",
        "test_fusion_benchmark",
        "single_gpu_train_20_steps",
        "mpi_2_host_staged_20_steps",
        "mpi_4_blocking_direct_20_steps",
        "mpi_4_overlap_pinned_20_steps",
    ]
    for name in required:
        check_no_bad_tokens(LOGS / f"full_validation_88905_{name}.txt")
    err = LOGS / "full_validation_88905.err"
    if require_file(err) and err.stat().st_size > 0:
        warn("full_validation_88905.err is non-empty; currently only an unused-function compiler warning")


def check_submission_validation():
    check_job_log(LOGS / "full_validation_89398.out", "=== FULL VALIDATION PASSED ===")
    check_no_bad_tokens(LOGS / "full_validation_89398.out")
    required = [
        "test_gemm",
        "test_attention",
        "test_layernorm",
        "test_model_reference",
        "test_fusion_benchmark",
        "single_gpu_train_20_steps",
        "mpi_2_host_staged_20_steps",
        "mpi_4_blocking_direct_20_steps",
        "mpi_4_overlap_pinned_20_steps",
    ]
    for name in required:
        check_no_bad_tokens(LOGS / f"full_validation_89398_{name}.txt")
    err = LOGS / "full_validation_89398.err"
    if require_file(err) and err.stat().st_size > 0:
        warn("full_validation_89398.err is non-empty; currently only an unused-function compiler warning")


def check_edge_cases():
    rows = read_csv(RESULTS / "edge_case_validation.csv")
    expected_names = {
        "rank3_batch32_blocking",
        "rank3_batch32_pinned_overlap",
        "rank3_batch32_openmp_thread",
        "rank3_batch31_blocking",
        "rank3_batch31_openmp_thread",
    }
    names = {row["name"] for row in rows}
    if names != expected_names:
        fail(f"edge-case names mismatch: expected {expected_names}, saw {names}")
    for row in rows:
        name = row["name"]
        if row["valid"] != "yes":
            fail(f"edge-case {name} invalid")
        if int(row["ranks"]) != 3:
            fail(f"edge-case {name} did not use 3 ranks")
        if int(row["dropped_batches"]) <= 0:
            fail(f"edge-case {name} did not exercise dropped batches")
        if not finite(row["throughput_tok_s"]):
            fail(f"edge-case {name} non-finite throughput")
        checksum = row["checksum_span"]
        if checksum and checksum != "nan" and float(checksum) > 1e-4:
            fail(f"edge-case {name} large checksum span {checksum}")
    check_job_log(LOGS / "edge_case_validation_88911.out", "=== DONE job=88911 status=0 ===")
    check_no_bad_tokens(LOGS / "edge_case_validation_88911.out")
    check_no_bad_tokens(LOGS / "edge_case_validation_88911_raw.txt")
    require_file(RESULTS / "edge_case_validation.md")


def check_trajectory_validation():
    summary = read_csv(RESULTS / "trajectory_validation_summary.csv")
    detail = read_csv(RESULTS / "trajectory_validation.csv")
    expected_cases = {"blocking_direct", "pinned_overlap", "openmp_thread"}
    cases = {row["case"] for row in summary}
    if cases != expected_cases:
        fail(f"trajectory-validation cases mismatch: expected {expected_cases}, saw {cases}")
    if len(summary) != 3:
        fail(f"expected 3 trajectory-validation summary rows, saw {len(summary)}")
    if len(detail) != 36:
        fail(f"expected 36 trajectory-validation per-step rows, saw {len(detail)}")

    seen_steps = {}
    for row in detail:
        case = row["case"]
        seen_steps.setdefault(case, set()).add(int(row["step"]))
        for field in [
            "loss", "param_sum", "param_sumsq", "param_maxabs",
            "sum_abs_delta_per_param", "sumsq_abs_delta_per_param",
            "maxabs_abs_delta", "sum_span_per_param",
            "sumsq_span_per_param", "maxabs_span",
        ]:
            if not finite(row[field]):
                fail(f"trajectory-validation {case} step={row.get('step')} non-finite {field}")
        if row["step_valid"] != "yes":
            fail(f"trajectory-validation {case} step={row.get('step')} marked invalid")
    for case in expected_cases:
        if seen_steps.get(case) != set(range(1, 13)):
            fail(f"trajectory-validation {case} steps mismatch: {seen_steps.get(case)}")

    for row in summary:
        case = row["case"]
        if row["valid"] != "yes":
            fail(f"trajectory-validation {case} not valid")
        if int(row["steps"]) != 12 or int(row["expected_steps"]) != 12:
            fail(f"trajectory-validation {case} did not run 12/12 steps")
        if int(row["failed_steps"]) != 0 or int(row["invalid_values"]) != 0:
            fail(f"trajectory-validation {case} has failed/invalid steps")
        checks = [
            ("max_loss_abs_delta", 1e-4),
            ("max_sum_abs_delta_per_param", 1e-6),
            ("max_sumsq_abs_delta_per_param", 2e-8),
            ("max_maxabs_abs_delta", 2e-6),
            ("max_sum_span_per_param", 1e-9),
            ("max_sumsq_span_per_param", 1e-9),
            ("max_maxabs_span", 2e-6),
        ]
        for field, limit in checks:
            if not finite(row[field]) or float(row[field]) > limit:
                fail(f"trajectory-validation {case} {field} exceeds {limit}: {row[field]}")
    check_job_log(LOGS / "trajectory_validation_89250.out", "=== DONE job=89250 status=0 ===")
    check_no_bad_tokens(LOGS / "trajectory_validation_89250.out")
    check_no_bad_tokens(LOGS / "trajectory_validation_89250_raw.txt")
    require_file(RESULTS / "trajectory_validation_89250.md")
    require_file(RESULTS / "trajectory_validation_summary_89250.csv")
    require_file(RESULTS / "trajectory_validation.md")


def check_fusion_ablation():
    rows = read_csv(RESULTS / "fusion_ablation.csv")
    expected_hidden = {128, 256, 512}
    seen_hidden = {int(row["hidden"]) for row in rows if row.get("hidden")}
    if seen_hidden != expected_hidden:
        fail(f"fusion-ablation hidden sizes mismatch: expected {expected_hidden}, saw {seen_hidden}")
    if len(rows) != 3:
        fail(f"expected 3 fusion-ablation rows, saw {len(rows)}")

    for row in rows:
        hidden = int(row["hidden"])
        if row["baseline_backend"] != "cublas_tc":
            fail(f"h{hidden} fusion-ablation baseline backend mismatch: {row['baseline_backend']}")
        if row["fusion_backend"] != "cublas_tc_lt":
            fail(f"h{hidden} fusion-ablation fusion backend mismatch: {row['fusion_backend']}")
        if row["all_valid"] != "yes":
            fail(f"h{hidden} fusion-ablation group not all valid")
        if int(row["baseline_valid_runs"]) < 3 or int(row["fusion_valid_runs"]) < 3:
            fail(f"h{hidden} fusion-ablation has too few valid repeats")
        for field in [
            "baseline_throughput_mean_mtok_s",
            "baseline_throughput_std_mtok_s",
            "fusion_throughput_mean_mtok_s",
            "fusion_throughput_std_mtok_s",
            "throughput_speedup",
            "throughput_speedup_std",
            "baseline_time_mean_ms",
            "fusion_time_mean_ms",
            "fusion_time_slowdown_pct",
            "baseline_loss_mean",
            "fusion_loss_mean",
            "loss_delta",
        ]:
            if not finite(row[field]):
                fail(f"h{hidden} fusion-ablation non-finite {field}")
        speedup = float(row["throughput_speedup"])
        if speedup < 0.50 or speedup > 1.10:
            fail(f"h{hidden} fusion-ablation speedup out of sanity range: {speedup}")

    check_job_log(LOGS / "fusion_ablation_89254.out", "=== DONE job=89254 status=0 ===")
    check_no_bad_tokens(LOGS / "fusion_ablation_89254.out")
    check_no_bad_tokens(LOGS / "fusion_ablation_89254_raw.txt")
    require_file(RESULTS / "single_gpu_repeated_bench_summary_fusion_ablation_89254.csv")
    require_file(RESULTS / "fusion_ablation_89254.csv")
    require_file(RESULTS / "fusion_ablation_89254.md")
    require_file(RESULTS / "fusion_ablation.md")
    require_file(PLOTS / "fusion_ablation_89254.svg")
    require_file(PLOTS / "fusion_ablation.svg")


def check_nccl_allreduce_baseline():
    summary = read_csv(RESULTS / "nccl_allreduce_baseline_summary.csv")
    fit_rows = read_csv(RESULTS / "nccl_allreduce_baseline_fit.csv")
    comparison = read_csv(RESULTS / "nccl_allreduce_baseline_comparison.csv")
    validity = read_csv(RESULTS / "nccl_allreduce_baseline_validity.csv")

    expected_backends = {"device", "nccl"}
    expected_ranks = {2, 4}
    seen_backends = {row["backend"] for row in summary}
    seen_ranks = {int(row["ranks"]) for row in summary if row.get("ranks")}
    if seen_backends != expected_backends:
        fail(f"NCCL baseline backend mismatch: expected {expected_backends}, saw {seen_backends}")
    if seen_ranks != expected_ranks:
        fail(f"NCCL baseline ranks mismatch: expected {expected_ranks}, saw {seen_ranks}")
    if len(summary) != 56:
        fail(f"expected 56 NCCL baseline summary rows, saw {len(summary)}")
    if len(comparison) != 28:
        fail(f"expected 28 NCCL baseline comparison rows, saw {len(comparison)}")
    if len(validity) != 56:
        fail(f"expected 56 NCCL baseline validity rows, saw {len(validity)}")

    for row in validity:
        if row["valid"] != "yes":
            fail(f"NCCL baseline validity failure: {row}")

    fit_keys = {(row["backend"], int(row["ranks"])) for row in fit_rows if row.get("backend")}
    expected_fit_keys = {
        ("device", 2),
        ("device", 4),
        ("nccl", 2),
        ("nccl", 4),
    }
    if fit_keys != expected_fit_keys:
        fail(f"NCCL baseline fit keys mismatch: expected {expected_fit_keys}, saw {fit_keys}")
    for row in fit_rows:
        backend = row["backend"]
        ranks = int(row["ranks"])
        if float(row["r2"]) < 0.99:
            fail(f"NCCL baseline {backend} ranks={ranks} low R^2={row['r2']}")
        if float(row["beta_ms_per_byte"]) <= 0.0:
            fail(f"NCCL baseline {backend} ranks={ranks} non-positive beta")

    for row in comparison:
        ranks = int(row["ranks"])
        bytes_ = int(row["bytes"])
        for field in [
            "mpi_device_time_mean_ms",
            "mpi_device_time_std_ms",
            "nccl_time_mean_ms",
            "nccl_time_std_ms",
            "nccl_speedup_vs_mpi_device",
            "mpi_device_payload_gb_s",
            "nccl_payload_gb_s",
        ]:
            if not finite(row[field]):
                fail(f"NCCL baseline ranks={ranks} bytes={bytes_} non-finite {field}")
        speedup = float(row["nccl_speedup_vs_mpi_device"])
        if speedup <= 1.0:
            fail(f"NCCL baseline ranks={ranks} bytes={bytes_} not faster than MPI device: {speedup}")

    check_job_log(LOGS / "nccl_allreduce_baseline_89261.out",
                  "=== DONE job=89261 status=0 ===")
    check_no_bad_tokens(LOGS / "nccl_allreduce_baseline_89261.out")
    check_no_bad_tokens(LOGS / "nccl_allreduce_baseline_89261_raw.txt")
    require_file(RESULTS / "nccl_allreduce_baseline_89261.md")
    require_file(RESULTS / "nccl_allreduce_baseline_summary_89261.csv")
    require_file(RESULTS / "nccl_allreduce_baseline_fit_89261.csv")
    require_file(RESULTS / "nccl_allreduce_baseline_comparison_89261.csv")
    require_file(RESULTS / "nccl_allreduce_baseline_validity_89261.csv")
    require_file(RESULTS / "nccl_allreduce_baseline.md")
    require_file(PLOTS / "nccl_allreduce_baseline_89261.svg")
    require_file(PLOTS / "nccl_allreduce_baseline.svg")


def main():
    require_file(PACKAGE_ROOT / "SUBMISSION_OVERVIEW.md")
    require_file(PACKAGE_ROOT / "FINAL_TRUTH_TABLE.md")
    require_file(PACKAGE_ROOT / "FRESH_PACKAGE_VALIDATION.md")
    require_file(PACKAGE_ROOT / "REPORT_FRAMING_DRAFT.md")
    require_file(RESULTS / "REAL_RESULTS.md")
    check_strong_scaling_repeated()
    check_weak_scaling()
    check_alpha_beta()
    check_breakdown_and_prediction()
    check_overlap_speedup()
    check_bucket_ucurve_h256()
    check_bucket_ucurve_h512()
    check_nsys_timeline_h256_openmp()
    check_nsys_hidden_breakdown()
    check_gemm_precision()
    check_validation()
    check_submission_validation()
    check_edge_cases()
    check_trajectory_validation()
    check_fusion_ablation()
    check_nccl_allreduce_baseline()
    if warnings:
        print("WARNINGS:")
        for msg in warnings:
            print(f"- {msg}")
    if failures:
        print("FAILURES:")
        for msg in failures:
            print(f"- {msg}")
        sys.exit(1)
    print("ARTIFACT CHECK PASSED")


if __name__ == "__main__":
    main()
