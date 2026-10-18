#!/usr/bin/env python3
"""按借车年份拆分原始订单，不修改原始 SQLite。

输出：
  data/raw/orders_2021.sqlite       2021 年订单
  data/raw/orders_2017_2020.sqlite  其余年份订单
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS orders_raw (
    record_hash TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    start_lat REAL,
    start_lng REAL,
    end_lat REAL,
    end_lng REAL,
    user_id TEXT,
    com_id TEXT,
    source_page INTEGER NOT NULL
)
"""


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON orders_raw(start_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_end_time ON orders_raw(end_time)")
    conn.commit()


def copy_year(src: sqlite3.Connection, dst: sqlite3.Connection, start: str, end: str, chunk: int) -> int:
    init_db(dst)
    cursor = src.execute(
        """SELECT record_hash, start_time, end_time, start_lat, start_lng,
                  end_lat, end_lng, user_id, com_id, source_page
           FROM orders_raw WHERE start_time >= ? AND start_time < ? ORDER BY rowid""",
        (start, end),
    )
    insert = """INSERT OR IGNORE INTO orders_raw
        (record_hash, start_time, end_time, start_lat, start_lng, end_lat, end_lng,
         user_id, com_id, source_page)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    total = 0
    while True:
        rows = cursor.fetchmany(chunk)
        if not rows:
            break
        dst.executemany(insert, rows)
        dst.commit()
        total += len(rows)
        print(f"已拆分 {total:,} 条", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="按年份拆分共享单车原始订单")
    parser.add_argument("--input", default="data/raw/orders_3m.sqlite")
    parser.add_argument("--year-output", default="data/raw/orders_2021.sqlite")
    parser.add_argument("--other-output", default="data/raw/orders_2017_2020.sqlite")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    src_path = Path(args.input)
    if not src_path.exists():
        raise SystemExit(f"输入数据库不存在：{src_path}")
    Path(args.year_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.other_output).parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(src_path)
    year_db = sqlite3.connect(args.year_output)
    other_db = sqlite3.connect(args.other_output)
    try:
        n2021 = copy_year(src, year_db, "2021-01-01", "2022-01-01", args.chunk_size)
        n_other = copy_year(src, other_db, "0000-01-01", "2021-01-01", args.chunk_size)
    finally:
        src.close()
        year_db.close()
        other_db.close()
    print(f"2021年：{n2021:,} 条 -> {args.year_output}")
    print(f"其他年份：{n_other:,} 条 -> {args.other_output}")
    print(f"原始数据库未修改：{args.input}")


if __name__ == "__main__":
    main()
