#!/usr/bin/env python3
import argparse
import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles"
RESULTS = ROOT / "results"


def kernel_rows(db_path, label, limit):
    con = sqlite3.connect(str(db_path))
    query = """
        select s.value as kernel_name,
               count(*) as launches,
               sum(k.end - k.start) / 1.0e6 as total_ms,
               avg(k.end - k.start) / 1.0e3 as avg_us
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on k.demangledName = s.id
        group by s.value
        order by sum(k.end - k.start) desc
        limit ?
    """
    return [
        {
            "profile": label,
            "kind": "kernel",
            "name": name,
            "calls": calls,
            "total_ms": total_ms,
            "avg_us": avg_us,
            "total_mb": "",
        }
        for name, calls, total_ms, avg_us in con.execute(query, (limit,))
    ]


def memcpy_rows(db_path, label):
    con = sqlite3.connect(str(db_path))
    query = """
        select e.label,
               count(*) as copies,
               sum(m.end - m.start) / 1.0e6 as total_ms,
               avg(m.end - m.start) / 1.0e3 as avg_us,
               sum(m.bytes) / 1.0e6 as total_mb
        from CUPTI_ACTIVITY_KIND_MEMCPY m
        join ENUM_CUDA_MEMCPY_OPER e on m.copyKind = e.id
        group by e.label
        order by sum(m.end - m.start) desc
    """
    return [
        {
            "profile": label,
            "kind": "memcpy",
            "name": name,
            "calls": calls,
            "total_ms": total_ms,
            "avg_us": avg_us,
            "total_mb": total_mb,
        }
        for name, calls, total_ms, avg_us, total_mb in con.execute(query)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default="83616")
    parser.add_argument("--rank", default="0")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for mode in ["blocking", "overlap"]:
        path = PROFILES / f"h256_{mode}_{args.job_id}_rank{args.rank}.sqlite"
        if not path.exists():
            raise SystemExit(f"missing profile: {path}")
        label = f"h256_{mode}_rank{args.rank}"
        rows.extend(kernel_rows(path, label, args.limit))
        rows.extend(memcpy_rows(path, label))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"nsys_top_kernels_{args.job_id}_rank{args.rank}.csv"
    fields = ["profile", "kind", "name", "calls", "total_ms", "avg_us", "total_mb"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "nsys_top_kernels.csv").write_text(out.read_text())
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
