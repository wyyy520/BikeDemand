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


class SpatialAttention(nn.Module):
    """在输出通道维度做自注意力，模拟网格间的信息交互。

    输入 z: [batch, time, d_model]，转置后在 d_model 维度上做注意力，
    相当于让每个时间步内不同"通道"（对应各网格的表征）互相交换信息。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.mix_logits = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: Tensor, num_nodes: int) -> Tensor:
        # 把 d_model 通道当作注意力序列，模拟节点间交互
        batch, time, d = z.shape
        # 每个时间步独立做：把d_model个通道看成序列，需要每个位置是d_model维向量，
        # 但通道本身是标量，所以用1x1卷积做Q/K/V投影不可行。改为简单可学习加权混合。
        # 简化：用可学习参数矩阵在d维度做注意力式混合（等效单头自注意力的值域）
        weight = torch.softmax(self.mix_logits, dim=-1)  # [d, d]
        mixed = torch.matmul(z, weight.T)  # [batch, time, d]
        attn_out = mixed
        return self.norm(z + attn_out)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_nodes: int = 0,
        use_spatial_attention: bool = False,
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
        self.num_nodes = num_nodes
        if use_spatial_attention and num_nodes <= 0:
            raise ValueError("启用空间注意力时必须提供num_nodes")
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
        self.spatial = (
            SpatialAttention(d_model, nhead, dropout=dropout)
            if use_spatial_attention and num_nodes > 1
            else None
        )
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
        if self.spatial is not None:
            z = self.spatial(z, self.num_nodes)
        last_state = z[:, -1]
        prediction = self.head(last_state)
        return prediction.view(x.size(0), self.horizon, self.output_dim)


if __name__ == "__main__":
    model = TemporalTransformer(input_dim=80, output_dim=60)
    sample = torch.randn(4, 48, 80)
    result = model(sample)
    print("input:", tuple(sample.shape))
    print("output:", tuple(result.shape))
