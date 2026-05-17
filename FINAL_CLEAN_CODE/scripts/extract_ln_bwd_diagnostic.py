#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"
RESULTS = ROOT / "results"


METRICS = {
    "runtime_us": ("gpu__time_duration.sum", 1e-3),
    "dram_read_bytes": ("dram__bytes_read.sum", 1.0),
    "dram_write_bytes": ("dram__bytes_write.sum", 1.0),
    "dram_throughput_pct": ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 1.0),
    "sm_throughput_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1.0),
    "achieved_occupancy_pct": ("sm__warps_active.avg.pct_of_peak_sustained_active", 1.0),
    "registers_per_thread": ("launch__registers_per_thread", 1.0),
    "block_size": ("launch__block_size", 1.0),
    "grid_size": ("launch__grid_size", 1.0),
    "waves_per_sm": ("launch__waves_per_multiprocessor", 1.0),
    "occupancy_limit_blocks": ("launch__occupancy_limit_blocks", 1.0),
    "occupancy_limit_registers": ("launch__occupancy_limit_registers", 1.0),
    "occupancy_limit_shared_mem": ("launch__occupancy_limit_shared_mem", 1.0),
    "occupancy_limit_warps": ("launch__occupancy_limit_warps", 1.0),
    "l1_global_atom_requests": ("l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum", 1.0),
    "l1_global_atom_wavefronts": ("l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom.sum", 1.0),
    "smsp_global_red_insts": ("smsp__inst_executed_op_global_red.sum", 1.0),
    "stall_barrier": ("smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", 1.0),
    "stall_long_scoreboard": ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", 1.0),
    "stall_lg_throttle": ("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio", 1.0),
    "stall_wait": ("smsp__average_warps_issue_stalled_wait_per_issue_active.ratio", 1.0),
    "stall_not_selected": ("smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio", 1.0),
}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def extract(case):
    path = PROFILES / f"fusion_{case}_83615_raw.csv"
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        kernel_index = 0
        for row in reader:
            if not row.get("Kernel Name"):
                continue
            out = {
                "case": case,
                "kernel_index": kernel_index,
                "kernel_name": row.get("Kernel Name", ""),
            }
            for name, (column, scale) in METRICS.items():
                out[name] = to_float(row.get(column, "")) * scale
            out["dram_total_bytes"] = out["dram_read_bytes"] + out["dram_write_bytes"]
            rows.append(out)
            kernel_index += 1
    if not rows:
        raise RuntimeError(f"no kernel row in {path}")
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    rows.extend(extract("ln_bwd_residual_unfused"))
    rows.extend(extract("ln_bwd_residual_fused"))
    fields = ["case", "kernel_index", "kernel_name"] + list(METRICS.keys()) + ["dram_total_bytes"]
    out = RESULTS / "ln_bwd_diagnostic_83615.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
