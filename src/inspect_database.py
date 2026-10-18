"""查看共享单车 SQLite 样本数据库。

用法示例：
    python3 src/inspect_database.py
    python3 src/inspect_database.py --limit 20
    python3 src/inspect_database.py --date 2017-04-03
    python3 src/inspect_database.py --sql "SELECT * FROM orders_raw LIMIT 5"
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB = "data/raw/orders_sample_100k.sqlite"


def print_table(headers: list[str], rows: list[tuple]) -> None:
    if not rows:
        print("没有查询到记录。")
        return
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    separator = "-+-".join("-" * width for width in widths)
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print(separator)
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def show_schema(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(orders_raw)").fetchall()
    print("\n[表结构]")
    print_table(["cid", "name", "type", "notnull", "default", "pk"], rows)


def show_summary(conn: sqlite3.Connection) -> None:
    sql = """
    SELECT
        COUNT(*) AS total_records,
        COUNT(DISTINCT user_id) AS distinct_users,
        MIN(start_time) AS min_start_time,
        MAX(start_time) AS max_start_time,
        MIN(end_time) AS min_end_time,
        MAX(end_time) AS max_end_time,
        MIN(start_lat) AS min_start_lat,
        MAX(start_lat) AS max_start_lat,
        MIN(start_lng) AS min_start_lng,
        MAX(start_lng) AS max_start_lng
    FROM orders_raw
    """
    row = conn.execute(sql).fetchone()
    names = [
        "总记录数", "不同用户数", "最早借车时间", "最晚借车时间",
        "最早还车时间", "最晚还车时间", "起点最小纬度", "起点最大纬度",
        "起点最小经度", "起点最大经度",
    ]
    print("\n[数据概览]")
    for name, value in zip(names, row):
        print(f"{name}: {value}")


def show_quality(conn: sqlite3.Connection) -> None:
    sql = """
    SELECT
        SUM(start_time IS NULL OR start_time = '') AS missing_start_time,
        SUM(end_time IS NULL OR end_time = '') AS missing_end_time,
        SUM(start_lat IS NULL OR start_lng IS NULL) AS missing_start_coord,
        SUM(end_lat IS NULL OR end_lng IS NULL) AS missing_end_coord,
        SUM(end_time < start_time) AS invalid_time_order,
        SUM(end_time = start_time) AS zero_duration,
        SUM((julianday(end_time) - julianday(start_time)) * 86400 > 86400) AS over_24h
    FROM orders_raw
    """
    row = conn.execute(sql).fetchone()
    names = [
        "借车时间缺失", "还车时间缺失", "起点坐标缺失", "终点坐标缺失",
        "结束时间早于开始时间", "零时长订单", "超过24小时订单",
    ]
    print("\n[质量检查]")
    for name, value in zip(names, row):
        print(f"{name}: {value}")


def show_sample(conn: sqlite3.Connection, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT start_time, end_time, start_lat, start_lng,
               end_lat, end_lng, user_id, com_id
        FROM orders_raw
        ORDER BY start_time
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    print(f"\n[前 {limit} 条记录（按借车时间排序）]")
    print_table(
        ["start_time", "end_time", "start_lat", "start_lng", "end_lat", "end_lng", "user_id", "com_id"],
        rows,
    )


def show_date(conn: sqlite3.Connection, date: str, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT start_time, end_time, start_lat, start_lng,
               end_lat, end_lng, user_id, com_id
        FROM orders_raw
        WHERE substr(start_time, 1, 10) = ?
        ORDER BY start_time
        LIMIT ?
        """,
        (date, limit),
    ).fetchall()
    print(f"\n[日期 {date} 的记录，最多 {limit} 条]")
    print_table(
        ["start_time", "end_time", "start_lat", "start_lng", "end_lat", "end_lng", "user_id", "com_id"],
        rows,
    )


def show_daily_counts(conn: sqlite3.Connection, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT substr(start_time, 1, 10) AS date, COUNT(*) AS orders
        FROM orders_raw
        GROUP BY date
        ORDER BY date
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    print(f"\n[按日期统计，前 {limit} 个日期]")
    print_table(["date", "orders"], rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="查看共享单车 SQLite 数据库")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite 数据库路径")
    parser.add_argument("--limit", type=int, default=10, help="显示记录条数")
    parser.add_argument("--date", help="只查看指定日期，例如 2017-04-03")
    parser.add_argument("--daily", action="store_true", help="显示按日期的订单量")
    parser.add_argument("--sql", help="执行一条只读 SQL 查询")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"数据库不存在：{db_path}")
    if args.limit <= 0:
        raise SystemExit("--limit 必须大于 0")

    conn = sqlite3.connect(db_path)
    try:
        if args.sql:
            cursor = conn.execute(args.sql)
            rows = cursor.fetchmany(args.limit)
            print_table([description[0] for description in cursor.description], rows)
            return
        show_schema(conn)
        show_summary(conn)
        show_quality(conn)
        if args.date:
            show_date(conn, args.date, args.limit)
        elif args.daily:
            show_daily_counts(conn, args.limit)
        else:
            show_sample(conn, args.limit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
