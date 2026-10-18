#!/usr/bin/env python3
"""把清洗后的订单聚合成 Transformer 可直接使用的 30 分钟网格序列。

输出：
  data/processed/demand_30min.npz
  data/processed/demand_30min_meta.json

每个时间步的 demand 向量按 ``[out_grid_1..N, in_grid_1..N]`` 排列，
time_features 包含日周期、周周期、周末和高峰标记。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def grid_id(lat: pd.Series, lng: pd.Series, min_lat: float, min_lng: float, size: float) -> pd.Series:
    gy = np.floor((lat.to_numpy() - min_lat) / size).astype(np.int32)
    gx = np.floor((lng.to_numpy() - min_lng) / size).astype(np.int32)
    return pd.Series(gy * 1000 + gx, index=lat.index, dtype="int32")


def make_features(index: pd.DatetimeIndex) -> np.ndarray:
    minute = index.hour * 60 + index.minute
    day_phase = 2 * np.pi * minute / 1440.0
    week_phase = 2 * np.pi * (index.dayofweek * 1440 + minute) / (7 * 1440.0)
    weekend = (index.dayofweek >= 5).astype(np.float32)
    peak = (((index.hour.isin([7, 8, 9, 17, 18, 19])))).astype(np.float32)
    return np.column_stack([
        np.sin(day_phase), np.cos(day_phase),
        np.sin(week_phase), np.cos(week_phase), weekend, peak,
    ]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="订单聚合为30分钟空间网格序列")
    parser.add_argument("--input", default="data/processed/orders_5m_clean.sqlite")
    parser.add_argument("--output", default="data/processed/demand_30min.npz")
    parser.add_argument("--top-grids", type=int, default=30)
    parser.add_argument("--grid-size", type=float, default=0.01, help="网格边长，单位为经纬度度数")
    parser.add_argument("--min-lat", type=float, default=22.4)
    parser.add_argument("--min-lng", type=float, default=113.7)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"输入数据库不存在：{source}")
    if args.top_grids <= 0 or args.grid_size <= 0:
        raise SystemExit("--top-grids和--grid-size必须大于0")
    output.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(source)
    query = """SELECT start_time, end_time, start_lat, start_lng, end_lat, end_lng
               FROM orders_clean"""
    parts: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_sql_query(query, conn, chunksize=args.chunk_size):
        total += len(chunk)
        chunk["start_time"] = pd.to_datetime(chunk["start_time"], errors="coerce").dt.floor("30min")
        chunk["end_time"] = pd.to_datetime(chunk["end_time"], errors="coerce").dt.floor("30min")
        chunk = chunk.dropna(subset=["start_time", "end_time", "start_lat", "start_lng", "end_lat", "end_lng"])
        chunk["out_grid"] = grid_id(chunk["start_lat"], chunk["start_lng"], args.min_lat, args.min_lng, args.grid_size)
        chunk["in_grid"] = grid_id(chunk["end_lat"], chunk["end_lng"], args.min_lat, args.min_lng, args.grid_size)
        out = chunk.groupby(["start_time", "out_grid"], observed=True).size().rename("out_count").reset_index()
        out = out.rename(columns={"start_time": "time", "out_grid": "grid"})
        inc = chunk.groupby(["end_time", "in_grid"], observed=True).size().rename("in_count").reset_index()
        inc = inc.rename(columns={"end_time": "time", "in_grid": "grid"})
        parts.append(pd.concat([out, inc], ignore_index=True))
        if total % (args.chunk_size * 5) < len(chunk):
            print(f"已读取 {total:,} 条订单", flush=True)
    conn.close()
    if not parts:
        raise SystemExit("没有可聚合的订单")

    events = pd.concat(parts, ignore_index=True).fillna(0)
    events = events.groupby(["time", "grid"], as_index=False)[["out_count", "in_count"]].sum()
    grid_totals = events.groupby("grid")[["out_count", "in_count"]].sum().sum(axis=1)
    selected = grid_totals.nlargest(args.top_grids).index.to_numpy(dtype=np.int32)
    selected_set = set(int(x) for x in selected)
    events = events[events["grid"].isin(selected_set)]
    if events.empty:
        raise SystemExit("top-grids过滤后没有数据")

    start = events["time"].min()
    end = events["time"].max()
    times = pd.date_range(start=start, end=end, freq="30min")
    grid_to_col = {int(g): i for i, g in enumerate(selected)}
    demand = np.zeros((len(times), 2 * len(selected)), dtype=np.float32)
    time_pos = {t: i for i, t in enumerate(times)}
    for row in events.itertuples(index=False):
        t, g = row.time, int(row.grid)
        i = time_pos.get(t)
        j = grid_to_col.get(g)
        if i is not None and j is not None:
            demand[i, j] += float(row.out_count)
            demand[i, len(selected) + j] += float(row.in_count)

    time_features = make_features(times)
    times_int = (times.asi8 // 10**9).astype(np.int64)
    np.savez_compressed(output, times=times_int, demand=demand, time_features=time_features)
    metadata = {
        "source": str(source), "orders_read": total, "time_frequency": "30min",
        "grid_size": args.grid_size, "min_lat": args.min_lat, "min_lng": args.min_lng,
        "top_grids": [int(x) for x in selected], "num_grids": len(selected),
        "num_time_steps": len(times), "start_time": str(times[0]), "end_time": str(times[-1]),
        "feature_layout": "demand=[out_1..out_N,in_1..in_N], time_features=[day_sin,day_cos,week_sin,week_cos,weekend,peak]",
    }
    meta_path = output.with_name(output.stem + "_meta.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"聚合完成：{output}")
    print(f"时间步：{len(times):,}，网格：{len(selected)}，需求特征：{demand.shape[1]}")


if __name__ == "__main__":
    main()
