# 训练流程

## 1. 生成清洗数据库

下载完成后运行：

```bash
source .venv/bin/activate
python src/clean_orders.py \
  --input data/raw/orders_5m.sqlite \
  --output data/processed/orders_5m_clean.sqlite
```

清洗规则和删除数量会写入 `cleaning_stats` 表。

## 2. 30 分钟网格聚合

```bash
python src/aggregate_demand.py \
  --input data/processed/orders_5m_clean.sqlite \
  --output data/processed/demand_30min.npz \
  --top-grids 30 \
  --grid-size 0.01
```

`0.01` 个经纬度约对应 1 km 网格。脚本选择订单量最高的 30 个网格，输出借车量、还车量和 6 个时间特征。由于原始数据没有固定站点编号，网格是空间单元的替代表示。

Transformer 训练时额外加入显式 `lag48` 特征，即前一天同一时刻的借还需求，避免模型必须自行从历史序列中寻找该位置。

## 3. 训练 Transformer

```bash
python src/train_transformer.py \
  --data data/processed/demand_30min.npz \
  --output-dir checkpoints/transformer \
  --lookback 48 \
  --horizon 1 \
  --epochs 50 \
  --batch-size 64
```

含义：用过去 24 小时的 48 个半小时窗口预测下一个 30 分钟。程序按时间顺序划分 70% 训练、15% 验证、15% 测试；标准化参数只从训练区间估计。目标值先做 `log1p`，预测后用 `expm1` 还原为订单数。

输出文件：

- `checkpoints/transformer/best.pt`：验证集最优权重；
- `checkpoints/transformer/test_metrics.json`：Transformer、前一天同一时刻和历史平均基线的 MAE、RMSE、MAPE、WMAPE；
- `checkpoints/transformer/test_predictions.npz`：测试集真实值和预测值。

## 4. LSTM 对照实验

LSTM 使用完全相同的数据、窗口、时间切分和评价指标：

```bash
python src/train_lstm.py \
  --data data/processed/demand_2021_30min.npz \
  --output-dir checkpoints/lstm_2021 \
  --lookback 48 \
  --horizon 1 \
  --epochs 50 \
  --batch-size 64
```

输出 `best.pt`、`test_metrics.json` 和 `test_predictions.npz`，可以直接与 Transformer 结果比较。

训练过程中还会保存：

- `epoch_001.pt`、`epoch_002.pt` 等每轮权重和优化器状态；
- `last.pt`：最后一轮状态，可用于断点续训；
- `best.pt`：验证集损失最低的状态，用于最终测试。

显存不足时可降低 `--batch-size 32` 或 `--d-model 64`。RTX 4060 可以先用默认配置运行。
