#!/usr/bin/env python3
"""训练多区域时序 Transformer，并和两个简单基线比较。

典型运行：
    python src/aggregate_demand.py
    python src/train_transformer.py --epochs 30

目标改为残差：y 减去前一天同期需求（lag48），模型学习对 previous-day 基线的修正量，评估时把残差加回基线后还原为订单数。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformer_model import TemporalTransformer


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, targets: np.ndarray, lookback: int, horizon: int):
        self.x, self.y, self.targets = x, y, targets
        self.lookback, self.horizon = lookback, horizon

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        t = int(self.targets[index])
        lag48 = np.repeat(self.x[t - 48:t - 47], self.lookback, axis=0)
        features = np.concatenate(
            [self.x[t - self.lookback:t], self.y[t - self.lookback:t], lag48], axis=1
        )
        baseline = np.repeat(self.x[t - 48:t - 48 + self.horizon], self.horizon, axis=0)
        target = self.x[t:t + self.horizon] - baseline
        return torch.from_numpy(features).float(), torch.from_numpy(target).float()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_targets(start: int, end: int, lookback: int, horizon: int) -> np.ndarray:
    first = max(start, lookback)
    last = end - horizon + 1
    return np.arange(first, max(first, last), dtype=np.int64)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    denom = np.maximum(np.abs(y_true), 1.0)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_percent": float(np.mean(np.abs(err) / denom) * 100),
        "WMAPE_percent": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y_true)), 1.0) * 100),
    }


@torch.no_grad()
def predict(model, loader, device, mean, std):
    model.eval()
    all_true, all_pred = [], []
    for features, target in loader:
        prediction = model(features.to(device)).cpu().numpy()
        true = target.numpy()
        baseline = np.repeat(features[:, -48:-47, :prediction.shape[2]].cpu().numpy(), prediction.shape[1], axis=1)
        pred_count = np.expm1((prediction + baseline) * std + mean).clip(0)
        true_count = np.expm1((true + baseline) * std + mean).clip(0)
        all_pred.append(pred_count)
        all_true.append(true_count)
    return np.concatenate(all_true), np.concatenate(all_pred)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * len(features)
        n += len(features)
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练共享单车需求 Transformer")
    parser.add_argument("--data", default="data/processed/demand_30min.npz")
    parser.add_argument("--output-dir", default="checkpoints/transformer")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--train-start", default=None, help="如 2021-01-30T13:00")
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-attention", action="store_true", help="启用空间注意力")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"找不到聚合数据：{data_path}，请先运行 aggregate_demand.py")
    raw = np.load(data_path)
    demand = raw["demand"].astype(np.float32)
    time_features = raw["time_features"].astype(np.float32)
    times = raw["times"].astype(np.int64)
    steps, output_dim = demand.shape
    if len(times) != steps or steps <= args.lookback + args.horizon:
        raise SystemExit("时间序列太短，无法构造训练窗口")
    import pandas as pd
    if args.train_start:
        # 按日期划分（针对不均匀数据）
        ts = pd.to_datetime(times, unit="s")
        mask = ts >= pd.Timestamp(args.train_start)
        first_idx = int(np.argmax(mask))
        train_end = int(np.argmax(ts >= pd.Timestamp(args.train_end))) if args.train_end else steps
        val_end = int(np.argmax(ts >= pd.Timestamp(args.val_end))) if args.val_end else steps
        if not (first_idx + args.lookback + args.horizon < train_end < val_end <= steps):
            raise SystemExit("按日期划分后窗口不足，请检查日期范围")
        steps_offset = first_idx
        train_start_idx = first_idx
    else:
        train_start_idx = 0
        train_end = int(steps * args.train_ratio)
        val_end = int(steps * (args.train_ratio + args.val_ratio))
        if not (0 < train_end < val_end < steps):
            raise SystemExit("train/val比例不合法")

    # 只用训练时段估计变换参数，避免测试集信息泄漏。
    log_demand = np.log1p(demand)
    mean = log_demand[train_start_idx:train_end].mean(axis=0).astype(np.float32)
    std = log_demand[train_start_idx:train_end].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    scaled = ((log_demand - mean) / std).astype(np.float32)
    train_targets = make_targets(train_start_idx, train_end, args.lookback, args.horizon)
    val_targets = make_targets(train_end, val_end, args.lookback, args.horizon)
    test_targets = make_targets(val_end, steps, args.lookback, args.horizon)
    if min(len(train_targets), len(val_targets), len(test_targets)) == 0:
        raise SystemExit("某个数据划分没有可用窗口，请减少lookback或调整划分比例")

    train_ds = WindowDataset(scaled, time_features, train_targets, args.lookback, args.horizon)
    val_ds = WindowDataset(scaled, time_features, val_targets, args.lookback, args.horizon)
    test_ds = WindowDataset(scaled, time_features, test_targets, args.lookback, args.horizon)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransformer(
        input_dim=output_dim + time_features.shape[1] + output_dim, output_dim=output_dim,
        d_model=args.d_model, nhead=args.nhead, num_encoder_layers=args.layers,
        dim_feedforward=args.ffn_dim, dropout=args.dropout, horizon=args.horizon,
        num_nodes=output_dim, use_spatial_attention=args.spatial_attention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss()
    best_val, stale = float("inf"), 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"

    print(f"设备：{device}；时间步：{steps:,}；网格：{output_dim // 2}")
    print(f"窗口数：train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for features, target in val_loader:
                loss = criterion(model(features.to(device)), target.to(device))
                val_loss += float(loss.item()) * len(features)
                n += len(features)
        val_loss /= max(n, 1)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"验证集连续 {args.patience} 轮未改善，提前停止")
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    true, pred = predict(model, test_loader, device, mean, std)
    result = {"device": str(device), "best_val_loss": best_val, "transformer": metrics(true, pred)}

    # 前一天同一时刻基线：测试目标 t 使用 t-48 的真实需求。
    if args.horizon == 1:
        baseline_prev = demand[test_targets - 48]
        result["previous_day"] = metrics(true[:, 0], baseline_prev)
    # 训练集的星期几+半小时槽历史均值基线。
    train_dt = __import__("pandas").to_datetime(times[train_start_idx:train_end], unit="s")
    test_dt = __import__("pandas").to_datetime(times[test_targets], unit="s")
    slot = train_dt.dayofweek.to_numpy() * 48 + (train_dt.hour.to_numpy() * 2 + train_dt.minute.to_numpy() // 30)
    mean_by_slot = {int(k): demand[train_start_idx:train_end][slot == k].mean(axis=0) for k in np.unique(slot)}
    hist = np.stack([mean_by_slot.get(int(d.dayofweek * 48 + d.hour * 2 + d.minute // 30), demand[:train_end].mean(axis=0)) for d in test_dt])
    result["historical_mean"] = metrics(true[:, 0], hist) if args.horizon == 1 else {"note": "历史平均基线当前实现为单步"}
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "test_predictions.npz", times=times[test_targets], y_true=true, y_pred=pred)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"模型：{checkpoint}\n指标：{metrics_path}")


if __name__ == "__main__":
    main()
