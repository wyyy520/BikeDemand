#!/usr/bin/env python3
"""清洗深圳共享单车订单，剔除明显异常记录并写入新的 SQLite 数据库。

示例：
    python3 src/clean_orders.py \
        --input data/raw/orders_5m.sqlite \
        --output data/processed/orders_5m_clean.sqlite

默认过滤规则：
1. 时间或坐标缺失、无法解析；
2. 结束时间早于或等于开始时间；
3. 骑行时长超过24小时；
4. 起终点超出深圳近似边界；
5. 平均骑行速度超过50 km/h；
6. 起终点直线距离超过100 km。

所有阈值都可以通过命令行调整，清洗统计会写入 cleaning_stats 表。
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from datetime import datetime
from pathlib import Path


DEFAULT_BOUNDS = (22.4, 22.9, 113.7, 114.7)  # min_lat, max_lat, min_lng, max_lng


def parse_time(value: str) -> datetime:
    return datetime.strptime(value.split(".")[0], "%Y-%m-%d %H:%M:%S")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, a)))


def init_output(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders_clean (
            record_hash TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            start_lat REAL NOT NULL,
            start_lng REAL NOT NULL,
            end_lat REAL NOT NULL,
            end_lng REAL NOT NULL,
            user_id TEXT,
            com_id TEXT,
            duration_seconds REAL NOT NULL,
            distance_km REAL NOT NULL,
            speed_kmh REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clean_start_time ON orders_clean(start_time)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clean_end_time ON orders_clean(end_time)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cleaning_stats (
            rule TEXT PRIMARY KEY,
            removed_count INTEGER NOT NULL
        )
        """
    )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗共享单车订单")
    parser.add_argument("--input", default="data/raw/orders_5m.sqlite")
    parser.add_argument("--output", default="data/processed/orders_5m_clean.sqlite")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--max-duration-hours", type=float, default=24.0)
    parser.add_argument("--max-speed-kmh", type=float, default=50.0)
    parser.add_argument("--max-distance-km", type=float, default=100.0)
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"输入数据库不存在：{input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(input_path)
    dst = sqlite3.connect(output_path)
    init_output(dst)
    stats = {
        "total": 0,
        "kept": 0,
        "missing_or_parse_error": 0,
        "invalid_time": 0,
        "duration_too_long": 0,
        "coordinate_outside_shenzhen": 0,
        "distance_too_long": 0,
        "speed_too_high": 0,
    }
    min_lat, max_lat, min_lng, max_lng = DEFAULT_BOUNDS

    cursor = src.execute(
        """SELECT record_hash, start_time, end_time, start_lat, start_lng,
                  end_lat, end_lng, user_id, com_id
           FROM orders_raw ORDER BY rowid"""
    )
    insert_sql = """
        INSERT OR IGNORE INTO orders_clean
        (record_hash, start_time, end_time, start_lat, start_lng, end_lat, end_lng,
         user_id, com_id, duration_seconds, distance_km, speed_kmh)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch = []
    try:
        for row in cursor:
            stats["total"] += 1
            (record_hash, start_raw, end_raw, start_lat_v, start_lng_v,
             end_lat_v, end_lng_v, user_id, com_id) = row
            try:
                start_dt, end_dt = parse_time(start_raw), parse_time(end_raw)
                coords = tuple(float(x) for x in (start_lat_v, start_lng_v, end_lat_v, end_lng_v))
            except (TypeError, ValueError, OverflowError):
                stats["missing_or_parse_error"] += 1
                continue

            duration = (end_dt - start_dt).total_seconds()
            if duration <= 0:
                stats["invalid_time"] += 1
                continue
            if duration > args.max_duration_hours * 3600:
                stats["duration_too_long"] += 1
                continue

            s_lat, s_lng, e_lat, e_lng = coords
            in_bounds = all((min_lat <= lat <= max_lat and min_lng <= lng <= max_lng)
                            for lat, lng in ((s_lat, s_lng), (e_lat, e_lng)))
            if not in_bounds:
                stats["coordinate_outside_shenzhen"] += 1
                continue

            distance = haversine_km(s_lat, s_lng, e_lat, e_lng)
            if distance > args.max_distance_km:
                stats["distance_too_long"] += 1
                continue
            speed = distance / (duration / 3600.0)
            if speed > args.max_speed_kmh:
                stats["speed_too_high"] += 1
                continue

            batch.append((
                record_hash,
                start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                s_lat, s_lng, e_lat, e_lng, user_id, com_id,
                duration, distance, speed,
            ))
            if len(batch) >= args.chunk_size:
                dst.executemany(insert_sql, batch)
                dst.commit()
                stats["kept"] += len(batch)
                batch.clear()
                if stats["total"] % 100_000 == 0:
                    print(f"已处理 {stats['total']:,} 条，保留 {stats['kept']:,} 条", flush=True)
        if batch:
            dst.executemany(insert_sql, batch)
            dst.commit()
            stats["kept"] += len(batch)

        for rule, count in stats.items():
            if rule not in ("total", "kept"):
                dst.execute(
                    "INSERT OR REPLACE INTO cleaning_stats(rule, removed_count) VALUES (?, ?)",
                    (rule, count),
                )
        dst.commit()
    finally:
        src.close()
        dst.close()

    print("清洗完成")
    for key, value in stats.items():
        print(f"{key}: {value:,}")
    print(f"输出数据库：{output_path}")


if __name__ == "__main__":
    main()
