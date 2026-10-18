"""Download a bounded sample from the Shenzhen open-data API.

Usage:
    export SZ_APP_KEY='your_app_key'
    python3 src/fetch_orders.py --limit 100000

The script writes JSON Lines so a partial download remains inspectable and
does not require loading all records into memory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests


API_URL = "https://opendata.sz.gov.cn/api/29200_00403627/1/service.xhtml"


def fetch_page(session: requests.Session, app_key: str, page: int, rows: int) -> list[dict]:
    response = session.get(
        API_URL,
        params={"appKey": app_key, "page": page, "rows": rows},
        headers={"Accept": "application/json", "User-Agent": "bikedemand-course-project/1.0"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errorCode"):
        if payload.get("errorCode") == "10001":
            raise RuntimeError(
                "该 appKey 尚未订阅此数据接口。请先在共享单车企业每日订单表的接口详情页点击订阅/申请使用。"
            )
        raise RuntimeError(f"API error {payload['errorCode']}: {payload.get('message', '')}")
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected API response: {payload.keys()}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--output", default="data/raw/orders_sample_100k.jsonl")
    args = parser.parse_args()
    if args.limit <= 0 or args.rows <= 0:
        raise SystemExit("--limit and --rows must be positive")

    app_key = os.environ.get("SZ_APP_KEY")
    if not app_key:
        raise SystemExit("Missing SZ_APP_KEY. Set it with: export SZ_APP_KEY='your_app_key'")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fetched = 0
    page = 1
    with requests.Session() as session, output.open("w", encoding="utf-8") as handle:
        while fetched < args.limit:
            batch_size = min(args.rows, args.limit - fetched)
            records = fetch_page(session, app_key, page, batch_size)
            if not records:
                break
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            fetched += len(records)
            print(f"page={page} fetched={fetched}")
            if len(records) < batch_size:
                break
            page += 1
            time.sleep(0.1)
    print(f"saved {fetched} records to {output}")


if __name__ == "__main__":
    main()
