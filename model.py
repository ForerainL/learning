"""Small sequence models for adjMidRet60s prediction.

All models avoid normalization layers and default to <20k parameters.
Input shape: (batch, seq_len, features). Output: (batch, 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.1


class GRUModel(nn.Module):
    """Small GRU for sequence-to-one regression."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last = output[:, -1, :]
        return self.head(last)


class LSTMModel(nn.Module):
    """Small LSTM for sequence-to-one regression."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return self.head(last)


class TemporalBlock(nn.Module):
    """Simple dilated temporal block without normalization layers."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self._trim(out, self.conv1.padding[0])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self._trim(out, self.conv2.padding[0])
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

    @staticmethod
    def _trim(x: torch.Tensor, padding: int) -> torch.Tensor:
        if padding <= 0:
            return x
        return x[:, :, :-padding]


class TCNModel(nn.Module):
    """Small Temporal Convolutional Network for sequence-to-one regression."""

    def __init__(self, config: ModelConfig, kernel_size: int = 3):
        super().__init__()
        layers = []
        channels = [config.input_dim] + [config.hidden_dim] * config.num_layers
        for idx in range(config.num_layers):
            layers.append(
                TemporalBlock(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=config.dropout,
                )
            )
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.network(x)
        last = out[:, :, -1]
        return self.head(last)


class EncoderBlock(nn.Module):
    """Transformer encoder block without any normalization layers."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(x)
        return x


class TransformerModel(nn.Module):
    """Small Transformer encoder for sequence-to-one regression."""

    def __init__(self, config: ModelConfig, num_heads: int = 2):
        super().__init__()
        if config.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        if mask is not None:
            if mask.dim() == 2:
                pass
            elif mask.dim() == 3:
                batch = x.shape[0]
                if mask.shape[0] not in {batch, batch * self.num_heads}:
                    raise ValueError(
                        "attn_mask 3D shape must be (B, T, T) or (B*num_heads, T, T)"
                    )
            else:
                raise ValueError("attn_mask must be 2D or 3D")

        for block in self.blocks:
            x = block(x, mask=mask)
        last = x[:, -1, :]
        return self.head(last)


def count_parameters(model: nn.Module) -> int:
    """Utility to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ModelConfig",
    "GRUModel",
    "LSTMModel",
    "TCNModel",
    "TransformerModel",
    "count_parameters",
]


def _self_test() -> None:
    config = ModelConfig(input_dim=8, hidden_dim=16, num_layers=2, dropout=0.1)
    model = TransformerModel(config, num_heads=2)
    x = torch.randn(4, 64, config.input_dim)
    out = model(x)
    assert out.shape == (4, 1)
    loss = out.mean()
    loss.backward()


if __name__ == "__main__":
    _self_test()
