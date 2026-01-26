"""Small sequence models for adjMidRet60s prediction.

All models avoid normalization layers and default to <20k parameters.
Input shape: (batch, seq_len, features). Output: (batch, 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        out = out[:, :, :-self.conv1.padding[0]]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


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


class TransformerModel(nn.Module):
    """Small Transformer encoder for sequence-to-one regression."""

    def __init__(self, config: ModelConfig, num_heads: int = 2):
        super().__init__()
        if config.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        # Disable normalization by removing the layer norms inside the encoder layer.
        encoder_layer.norm1 = nn.Identity()
        encoder_layer.norm2 = nn.Identity()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)
        encoded = self.encoder(x, mask=mask)
        last = encoded[:, -1, :]
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
