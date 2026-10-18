#!/usr/bin/env python3
"""查看订单下载进度和已写入 SQLite 的内容。

示例：
    python3 src/monitor_download.py --db data/raw/orders_5m.sqlite
    python3 src/monitor_download.py --db data/raw/orders_5m.sqlite --interval 10
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def show(db_path: Path) -> None:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        row_count = conn.execute("SELECT COUNT(*) FROM orders_raw").fetchone()[0]
        state = conn.execute(
            "SELECT last_page, api_records FROM fetch_state WHERE id=1"
        ).fetchone()
        time_range = conn.execute(
            "SELECT MIN(start_time), MAX(start_time) FROM orders_raw"
        ).fetchone()
        latest = conn.execute(
            """SELECT start_time, end_time, start_lat, start_lng,
                      end_lat, end_lng, source_page
               FROM orders_raw ORDER BY rowid DESC LIMIT 3"""
        ).fetchall()
        size_mb = db_path.stat().st_size / 1024 / 1024
        print(f"数据库：{db_path}")
        print(f"文件大小：{size_mb:.2f} MB")
        print(f"SQLite已保存：{row_count:,} 条")
        if state:
            print(f"API累计获取：{state[1]:,} 条；最后完成页：{state[0]}")
        print(f"借车时间范围：{time_range[0]} 至 {time_range[1]}")
        print("最近写入的记录：")
        for item in latest:
            print("  ", item)
        conn.close()
    except sqlite3.OperationalError as exc:
        print(f"数据库暂时被写入锁定，稍后重试：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="监控共享单车 SQLite 下载进度")
    parser.add_argument("--db", default="data/raw/orders_5m.sqlite")
    parser.add_argument("--interval", type=int, default=0, help="每隔多少秒刷新；0表示只查看一次")
    args = parser.parse_args()
    path = Path(args.db)
    if not path.exists():
        raise SystemExit(f"数据库不存在：{path}")
    while True:
        print("\033[2J\033[H", end="")
        show(path)
        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
