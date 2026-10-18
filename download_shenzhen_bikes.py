#!/usr/bin/env python3
"""下载深圳共享单车订单到 SQLite。

安装：
    python3 -m pip install requests

运行：
    export SZ_APP_KEY='你的appKey'
    python3 download_shenzhen_bikes.py --limit 1000000

程序支持断点续爬。中断后重新执行同一条命令，会从上次完成的页继续。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import requests


API_URL = "https://opendata.sz.gov.cn/api/29200_00403627/1/service.xhtml"
USER_AGENT = "bikedemand-course-project/1.0"


def request_page(session, app_key: str, page: int, rows: int, retries: int) -> list[dict]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                API_URL,
                params={"appKey": app_key, "page": page, "rows": rows},
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errorCode"):
                raise RuntimeError(
                    f"API错误 {payload['errorCode']}: {payload.get('message', '')}"
                )
            records = payload.get("data", [])
            if not isinstance(records, list):
                raise RuntimeError("API返回的data不是列表")
            return records
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            wait = min(60, 2 ** (attempt - 1))
            print(f"第{page}页失败，{wait}秒后重试：{exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"第{page}页请求失败：{last_error}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
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
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON orders_raw(start_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_end_time ON orders_raw(end_time)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_page INTEGER NOT NULL,
            api_records INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def normalize(record: dict, page: int) -> tuple:
    values = {
        "START_TIME": record.get("START_TIME", ""),
        "END_TIME": record.get("END_TIME", ""),
        "START_LAT": record.get("START_LAT", ""),
        "START_LNG": record.get("START_LNG", ""),
        "END_LAT": record.get("END_LAT", ""),
        "END_LNG": record.get("END_LNG", ""),
        "USER_ID": record.get("USER_ID", ""),
        "COM_ID": record.get("COM_ID", ""),
    }
    digest = hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    def number(value):
        return None if value in (None, "") else float(value)

    return (
        digest,
        values["START_TIME"],
        values["END_TIME"],
        number(values["START_LAT"]),
        number(values["START_LNG"]),
        number(values["END_LAT"]),
        number(values["END_LNG"]),
        values["USER_ID"],
        values["COM_ID"],
        page,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载深圳共享单车订单到SQLite")
    parser.add_argument("--limit", type=int, default=1_000_000, help="目标API记录数")
    parser.add_argument("--rows", type=int, default=10_000, help="每页记录数，最大10000")
    parser.add_argument("--db", default="data/raw/orders.sqlite", help="SQLite文件路径")
    parser.add_argument("--retries", type=int, default=5, help="单页失败重试次数")
    parser.add_argument("--no-resume", action="store_true", help="从第1页重新下载")
    args = parser.parse_args()

    if args.limit <= 0 or not 1 <= args.rows <= 10_000:
        raise SystemExit("--limit必须大于0，--rows必须在1到10000之间")
    app_key = os.getenv("SZ_APP_KEY")
    if not app_key:
        raise SystemExit("没有找到SZ_APP_KEY，请先执行：export SZ_APP_KEY='你的appKey'")

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)
    state = conn.execute("SELECT last_page, api_records FROM fetch_state WHERE id=1").fetchone()
    if state and not args.no_resume:
        page, fetched = state[0] + 1, state[1]
        print(f"断点续爬：从第{page}页开始，已有API记录{fetched}条")
    else:
        page, fetched = 1, 0

    try:
        with requests.Session() as session:
            while fetched < args.limit:
                batch_size = min(args.rows, args.limit - fetched)
                records = request_page(session, app_key, page, batch_size, args.retries)
                if not records:
                    print("接口返回空数据，下载结束。")
                    break
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO orders_raw
                    (record_hash, start_time, end_time, start_lat, start_lng,
                     end_lat, end_lng, user_id, com_id, source_page)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [normalize(record, page) for record in records],
                )
                conn.execute(
                    """
                    INSERT INTO fetch_state(id, last_page, api_records)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                    last_page=excluded.last_page,
                    api_records=excluded.api_records
                    """,
                    (page, fetched + len(records)),
                )
                conn.commit()
                fetched += len(records)
                saved = conn.execute("SELECT COUNT(*) FROM orders_raw").fetchone()[0]
                print(f"第{page}页：收到{len(records)}条，API累计{fetched}条，去重后{saved}条", flush=True)
                if len(records) < batch_size:
                    break
                page += 1
                time.sleep(0.1)
    finally:
        conn.close()
    print(f"完成，数据库：{db_path}")


if __name__ == "__main__":
    main()
