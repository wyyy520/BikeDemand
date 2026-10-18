# 清洗和 Transformer 实现说明

## 清洗规则

`src/clean_orders.py` 将原始订单清洗到新的 SQLite 数据库。默认剔除：

- 时间或坐标缺失、无法解析的记录；
- 结束时间早于或等于开始时间；
- 骑行时长超过 24 小时；
- 起终点超出深圳近似范围（纬度 22.4～22.9，经度 113.7～114.7）；
- 起终点直线距离超过 100 km；
- 平均速度超过 50 km/h。

所有规则都作为异常值处理，而不是把异常记录静默删除；脚本会将每条规则删除的数量写入 `cleaning_stats` 表，报告中可以据此说明清洗影响。

运行：

```bash
python3 src/clean_orders.py \
  --input data/raw/orders_5m.sqlite \
  --output data/processed/orders_5m_clean.sqlite
```

## Transformer 架构

`src/transformer_model.py` 实现多区域时序 Transformer Encoder：

```text
[batch, 48, input_dim]
  → Linear + LayerNorm
  → 正弦位置编码
  → 2层 Transformer Encoder，4头注意力
  → 取最后时间步
  → Linear Prediction Head
  → [batch, horizon, 2×网格数]
```

其中输出前半部分为各网格借车量，后半部分为各网格还车量。第一版设置 `horizon=1`，即预测未来 30 分钟；设置 `horizon=2` 即可预测未来 1 小时。

模型自检：

```bash
python3 src/transformer_model.py
```
