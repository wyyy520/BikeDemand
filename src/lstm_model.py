#!/usr/bin/env python3
"""多区域时序 LSTM 模型。

输入：[batch, lookback, demand_dim + time_feature_dim]
输出：[batch, horizon, demand_dim]
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DemandLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("输入必须是[batch, time, feature]三维张量")
        z = self.input_projection(x)
        z, _ = self.lstm(z)
        prediction = self.head(z[:, -1])
        return prediction.view(x.size(0), self.horizon, self.output_dim)


if __name__ == "__main__":
    model = DemandLSTM(input_dim=66, output_dim=60)
    sample = torch.randn(4, 48, 66)
    print("input:", tuple(sample.shape))
    print("output:", tuple(model(sample).shape))
