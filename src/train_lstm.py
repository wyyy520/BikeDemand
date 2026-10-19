#!/usr/bin/env python3
"""训练 LSTM 需求预测模型，评估流程与 train_transformer.py 保持一致。

目标为残差：y 减去前一天同期需求（lag48），评估时加回基线。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lstm_model import DemandLSTM


class WindowDataset(Dataset):
    def __init__(self, demand, time_features, targets, lookback, horizon):
        self.demand = demand
        self.time_features = time_features
        self.targets = targets
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        t = int(self.targets[index])
        lag48 = np.repeat(self.demand[t - 48:t - 47], self.lookback, axis=0)
        x = np.concatenate([
            self.demand[t - self.lookback:t],
            self.time_features[t - self.lookback:t],
            lag48,
        ], axis=1)
        baseline = np.repeat(self.demand[t - 48:t - 48 + self.horizon], self.horizon, axis=0)
        y = self.demand[t:t + self.horizon] - baseline
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def targets(start, end, lookback, horizon):
    first = max(start, lookback)
    last = end - horizon + 1
    return np.arange(first, max(first, last), dtype=np.int64)


def metric(y_true, y_pred):
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
    true_all, pred_all = [], []
    for x, y in loader:
        pred = model(x.to(device)).cpu().numpy()
        y = y.numpy()
        baseline = np.repeat(x[:, -48:-47, :pred.shape[2]].cpu().numpy(), pred.shape[1], axis=1)
        pred_all.append(np.expm1((pred + baseline) * std + mean).clip(0))
        true_all.append(np.expm1((y + baseline) * std + mean).clip(0))
    return np.concatenate(true_all), np.concatenate(pred_all)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="训练共享单车需求 LSTM")
    parser.add_argument("--data", default="data/processed/demand_2021_30min.npz")
    parser.add_argument("--output-dir", default="checkpoints/lstm_2021")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    raw = np.load(args.data)
    demand = raw["demand"].astype(np.float32)
    time_features = raw["time_features"].astype(np.float32)
    times = raw["times"].astype(np.int64)
    steps, output_dim = demand.shape
    train_end = int(steps * args.train_ratio)
    val_end = int(steps * (args.train_ratio + args.val_ratio))
    log_demand = np.log1p(demand)
    mean = log_demand[:train_end].mean(axis=0).astype(np.float32)
    std = log_demand[:train_end].std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    scaled = ((log_demand - mean) / std).astype(np.float32)

    train_ds = WindowDataset(scaled, time_features, targets(0, train_end, args.lookback, args.horizon), args.lookback, args.horizon)
    val_ds = WindowDataset(scaled, time_features, targets(train_end, val_end, args.lookback, args.horizon), args.lookback, args.horizon)
    test_targets = targets(val_end, steps, args.lookback, args.horizon)
    test_ds = WindowDataset(scaled, time_features, test_targets, args.lookback, args.horizon)
    loaders = [
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=args.batch_size),
        DataLoader(test_ds, batch_size=args.batch_size),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemandLSTM(
        input_dim=output_dim + time_features.shape[1] + output_dim, output_dim=output_dim,
        hidden_dim=args.hidden_dim, num_layers=args.layers,
        dropout=args.dropout, horizon=args.horizon,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    best, stale = float("inf"), 0
    print(f"设备：{device}；时间步：{steps:,}；网格：{output_dim // 2}")
    print(f"窗口数：train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders[0], optimizer, criterion, device)
        model.eval()
        val_sum, n = 0.0, 0
        with torch.no_grad():
            for x, y in loaders[1]:
                val_sum += criterion(model(x.to(device)), y.to(device)).item() * len(x)
                n += len(x)
        val_loss = val_sum / max(n, 1)
        print(f"epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f}", flush=True)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
            "mean": mean,
            "std": std,
        }
        torch.save(state, output_dir / f"epoch_{epoch:03d}.pt")
        torch.save(state, output_dir / "last.pt")
        if val_loss < best:
            best, stale = val_loss, 0
            state["best_val_loss"] = best
            torch.save(state, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"验证集连续 {args.patience} 轮未改善，提前停止")
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    true, pred = predict(model, loaders[2], device, mean, std)
    result = {"device": str(device), "best_val_loss": best, "lstm": metric(true, pred)}
    if args.horizon == 1:
        prev = demand[test_targets - 48]
        result["previous_day"] = metric(true[:, 0], prev)
        dt_train = pd.to_datetime(times[:train_end], unit="s")
        dt_test = pd.to_datetime(times[test_targets], unit="s")
        train_slot = dt_train.dayofweek.to_numpy() * 48 + dt_train.hour.to_numpy() * 2 + dt_train.minute.to_numpy() // 30
        test_slot = dt_test.dayofweek.to_numpy() * 48 + dt_test.hour.to_numpy() * 2 + dt_test.minute.to_numpy() // 30
        means = {int(k): demand[:train_end][train_slot == k].mean(axis=0) for k in np.unique(train_slot)}
        hist = np.stack([means.get(int(s), demand[:train_end].mean(axis=0)) for s in test_slot])
        result["historical_mean"] = metric(true[:, 0], hist)
    (output_dir / "test_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output_dir / "test_predictions.npz", times=times[test_targets], y_true=true, y_pred=pred)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"模型：{checkpoint}")


if __name__ == "__main__":
    main()
