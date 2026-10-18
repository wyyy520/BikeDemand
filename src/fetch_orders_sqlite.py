"""Fetch a bounded sample from the Shenzhen API into a local SQLite database."""

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


def get_page(
    session: requests.Session,
    app_key: str,
    page: int,
    rows: int,
    max_retries: int,
) -> list[dict]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(
                API_URL,
                params={"appKey": app_key, "page": page, "rows": rows},
                headers={"Accept": "application/json", "User-Agent": "bikedemand-course-project/1.0"},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errorCode"):
                raise RuntimeError(
                    f"API error {payload['errorCode']}: {payload.get('message', '')}"
                )
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise RuntimeError("API 返回的 data 不是列表")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait_seconds = min(60, 2 ** (attempt - 1))
            print(
                f"page={page} attempt={attempt}/{max_retries} failed: {exc}; "
                f"retry in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"page={page} 请求失败：{last_error}")


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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_start_time ON orders_raw(start_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_end_time ON orders_raw(end_time)")
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
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True)
    record_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def number(value: object):
        if value in (None, ""):
            return None
        return float(value)

    return (
        record_hash,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=1_000_000,
        help="本次目标 API 记录数；默认100万，不是去重后的数据库行数",
    )
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--db", default="data/raw/orders.sqlite")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--no-resume", action="store_true", help="从第1页重新请求")
    args = parser.parse_args()
    if args.limit <= 0 or args.rows <= 0 or args.rows > 10_000:
        raise SystemExit("--limit必须大于0，--rows必须在1到10000之间")
    app_key = os.environ.get("SZ_APP_KEY")
    if not app_key:
        raise SystemExit("Missing SZ_APP_KEY")

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)
    state = conn.execute("SELECT last_page, api_records FROM fetch_state WHERE id = 1").fetchone()
    if state and not args.no_resume:
        page = state[0] + 1
        fetched = state[1]
        print(f"resume from page={page}, previous_api_records={fetched}")
    else:
        page = 1
        fetched = 0
    try:
        with requests.Session() as session:
            while fetched < args.limit:
                batch_size = min(args.rows, args.limit - fetched)
                records = get_page(session, app_key, page, batch_size, args.max_retries)
                if not records:
                    break
                rows = [normalize(record, page) for record in records]
                conn.executemany(
                    """INSERT OR IGNORE INTO orders_raw
                    (record_hash, start_time, end_time, start_lat, start_lng,
                     end_lat, end_lng, user_id, com_id, source_page)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.execute(
                    "INSERT INTO fetch_state(id, last_page, api_records) VALUES (1, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET last_page=excluded.last_page, api_records=excluded.api_records",
                    (page, fetched + len(records)),
                )
                conn.commit()
                fetched += len(records)
                print(f"page={page} received={len(records)} total={fetched}", flush=True)
                if len(records) < batch_size:
                    break
                page += 1
                time.sleep(0.1)
    finally:
        conn.close()
    print(f"saved database: {db_path}")


if __name__ == "__main__":
    main()
