# 共享单车短时需求预测

本项目面向课程作业“基于共享单车数据的交通需求预测”，目标是利用订单数据预测未来 30 分钟各站点的借车量和还车量。

当前确定的主模型是**多区域时序 Transformer Encoder**。由于深圳官方订单数据提供起终点坐标而没有固定站点编号，项目先将 BD09 坐标映射为空间网格，再使用过去 24 小时（48 个 30 分钟时间段）的需求、时间特征和可选外部变量，输出下一时段各区域的借车量和还车量。项目同时实现历史平均法和“前一天同一时刻”两个基线，便于比较。

详细设计见 [模型架构设计](docs/architecture.md)、[数据处理规范](docs/data_contract.md) 和 [实验计划](docs/experiment_plan.md)。

## 目录结构

```text
.
├── configs/              # 可复现实验配置
├── data/raw/             # 原始订单和外部数据，不提交大文件
├── data/processed/       # 聚合后的时序数据
├── docs/                 # 架构、数据和实验文档
├── figures/              # 报告图片
├── notebooks/            # 探索性分析
├── src/                  # 后续实现代码
└── checkpoints/          # 训练得到的模型权重
```

## 后续实现顺序

1. 将课程数据放入 `data/raw/`，按照 `docs/data_contract.md` 确认字段。
2. 实现订单清洗和 30 分钟聚合。
3. 完成探索性分析图表。
4. 实现两个基线模型。
5. 按 `docs/architecture.md` 实现 Transformer 数据集和模型。
6. 训练、评价并生成空间误差图。

完整训练命令见 [训练流程](docs/training_pipeline.md)。核心命令为：

```bash
source .venv/bin/activate
python src/aggregate_demand.py
python src/train_transformer.py --epochs 50
```

## 环境安装

Debian/Ubuntu 的系统 Python 可能启用了 PEP 668，不能直接执行全局 `pip install`。建议在项目目录创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果只运行爬虫，当前系统已经能导入 `requests` 时无需额外安装；爬虫命令如下：

```bash
python3 main.py --limit 5000000 --rows 10000 --db data/raw/orders_5m.sqlite
```

该命令会从 `config_local.py` 读取本地 appKey，并按 SQLite 的 `fetch_state` 断点续爬。下载中断后再次执行同一命令即可继续。
