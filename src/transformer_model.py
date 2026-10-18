"""多区域时序 Transformer Encoder。

输入：X.shape == [batch, lookback, input_dim]
输出：y_hat.shape == [batch, horizon, output_dim]

对于 N 个空间网格，output_dim=2*N，前 N 个输出为借车量，后 N 个输出为还车量。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.encoding[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        horizon: int = 1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model必须能被nhead整除")
        self.horizon = horizon
        self.output_dim = output_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.position = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon * output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("输入必须是[batch, time, feature]三维张量")
        z = self.input_projection(x)
        z = self.position(z)
        z = self.encoder(z)
        last_state = z[:, -1]
        prediction = self.head(last_state)
        return prediction.view(x.size(0), self.horizon, self.output_dim)


if __name__ == "__main__":
    model = TemporalTransformer(input_dim=80, output_dim=60)
    sample = torch.randn(4, 48, 80)
    result = model(sample)
    print("input:", tuple(sample.shape))
    print("output:", tuple(result.shape))
