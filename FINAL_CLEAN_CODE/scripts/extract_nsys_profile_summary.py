#!/usr/bin/env python3
"""Extract top kernels and memcpy rows from one Nsight Systems sqlite file."""

import argparse
import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def table_exists(con, table):
    row = con.execute(
        "select name from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()
    return row is not None


def kernel_rows(con, label, limit):
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
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


def memcpy_rows(con, label):
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        return []
    enum_table = "ENUM_CUDA_MEMCPY_OPER"
    if table_exists(con, enum_table):
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
    else:
        query = """
            select cast(m.copyKind as text),
                   count(*) as copies,
                   sum(m.end - m.start) / 1.0e6 as total_ms,
                   avg(m.end - m.start) / 1.0e3 as avg_us,
                   sum(m.bytes) / 1.0e6 as total_mb
            from CUPTI_ACTIVITY_KIND_MEMCPY m
            group by m.copyKind
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
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    db_path = Path(args.sqlite)
    if not db_path.exists():
        raise SystemExit(f"missing sqlite profile: {db_path}")

    con = sqlite3.connect(str(db_path))
    rows = kernel_rows(con, args.label, args.limit)
    rows.extend(memcpy_rows(con, args.label))

    RESULTS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / f"{args.label}_nsys_summary.csv"
    fields = ["profile", "kind", "name", "calls", "total_ms", "avg_us", "total_mb"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        display = out.resolve().relative_to(ROOT)
    except ValueError:
        display = out
    print(f"Wrote {display}")


if __name__ == "__main__":
    main()
