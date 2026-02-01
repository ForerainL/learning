"""Normalized variants of small sequence models for adjMidRet60s prediction.

Raw (no-norm) baselines are re-exported from model.py.
Input shape: (B, T, F). Output shape: (B, 1).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel


# -----------------------------
# Part A — Utilities
# -----------------------------

def make_norm(
    norm_type: Optional[str],
    num_channels: int,
    *,
    num_groups: int = 4,
) -> Optional[nn.Module]:
    """
    norm_type: None | "bn" | "in" | "ln" | "gn"
    - LN is applied on the last dimension (LayerNorm)
    - BN / IN / GN assume channel-first tensors (B, C, T)
    """
    if norm_type is None:
        return None

    norm_key = norm_type.lower()
    if norm_key == "ln":
        return nn.LayerNorm(num_channels)
    if norm_key == "bn":
        return nn.BatchNorm1d(num_channels)
    if norm_key == "in":
        return nn.InstanceNorm1d(num_channels, affine=True)
    if norm_key == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


# -----------------------------
# Part B — TCN (WeightNorm)
# -----------------------------


class TemporalBlockWN(nn.Module):
    """Simple dilated temporal block with WeightNorm on Conv1d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
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


class TemporalBlockWNWithNorm(nn.Module):
    """Temporal block with WeightNorm and optional activation normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        norm_type: Optional[str],
        num_groups: int = 4,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        self.norm1 = make_norm(norm_type, out_channels, num_groups=num_groups)
        self.norm2 = make_norm(norm_type, out_channels, num_groups=num_groups)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self._trim(out, self.conv1.padding[0])
        out = self._apply_norm(self.norm1, out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self._trim(out, self.conv2.padding[0])
        out = self._apply_norm(self.norm2, out)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

    @staticmethod
    def _trim(x: torch.Tensor, padding: int) -> torch.Tensor:
        if padding <= 0:
            return x
        return x[:, :, :-padding]

    @staticmethod
    def _apply_norm(norm: Optional[nn.Module], x: torch.Tensor) -> torch.Tensor:
        if norm is None:
            return x
        if isinstance(norm, nn.LayerNorm):
            x = x.transpose(1, 2)  # (B, T, C)
            x = norm(x)
            return x.transpose(1, 2)  # (B, C, T)
        return norm(x)


class TCNModelWN(nn.Module):
    """
    Standard TCN with WeightNorm on Conv1d (Bai et al. style).
    No activation normalization.
    """

    def __init__(self, config: ModelConfig, kernel_size: int = 3):
        super().__init__()
        layers = []
        channels = [config.input_dim] + [config.hidden_dim] * config.num_layers
        for idx in range(config.num_layers):
            layers.append(
                TemporalBlockWN(
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
        x = x.transpose(1, 2)  # (B, C, T)
        out = self.network(x)
        last = out[:, :, -1]  # (B, C)
        return self.head(last)


class TCNModelWNWithNorm(nn.Module):
    """
    TCN with WeightNorm on Conv1d + optional activation norm.
    """

    def __init__(
        self,
        config: ModelConfig,
        kernel_size: int = 3,
        norm_type: Optional[str] = None,
        num_groups: int = 4,
    ):
        super().__init__()
        layers = []
        channels = [config.input_dim] + [config.hidden_dim] * config.num_layers
        for idx in range(config.num_layers):
            layers.append(
                TemporalBlockWNWithNorm(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=config.dropout,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, T)
        out = self.network(x)
        last = out[:, :, -1]  # (B, C)
        return self.head(last)


# -----------------------------
# Part C — Transformer (Pre-LN)
# -----------------------------


class EncoderBlockPreLN(nn.Module):
    """Transformer encoder block with Pre-LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
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
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerModelPreLN(nn.Module):
    """
    Standard Transformer encoder with Pre-LayerNorm.
    """

    def __init__(self, config: ModelConfig, num_heads: int = 2):
        super().__init__()
        if config.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EncoderBlockPreLN(
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
        _validate_attn_mask(x, mask, self.num_heads)
        for block in self.blocks:
            x = block(x, mask=mask)
        last = x[:, -1, :]
        return self.head(last)


class TransformerModelPreLNWithNorm(nn.Module):
    """
    Transformer with Pre-LN + optional extra normalization.
    """

    def __init__(
        self,
        config: ModelConfig,
        num_heads: int = 2,
        input_norm: bool = True,
        output_norm: bool = False,
    ):
        super().__init__()
        if config.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.input_norm = nn.LayerNorm(config.input_dim) if input_norm else None
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                EncoderBlockPreLN(
                    hidden_dim=config.hidden_dim,
                    num_heads=num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim) if output_norm else None
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.input_proj(x)
        _validate_attn_mask(x, mask, self.num_heads)
        for block in self.blocks:
            x = block(x, mask=mask)
        if self.output_norm is not None:
            x = self.output_norm(x)
        last = x[:, -1, :]
        return self.head(last)


def _validate_attn_mask(x: torch.Tensor, mask: Optional[torch.Tensor], num_heads: int) -> None:
    if mask is None:
        return
    if mask.dim() == 2:
        return
    if mask.dim() == 3:
        batch = x.shape[0]
        if mask.shape[0] not in {batch, batch * num_heads}:
            raise ValueError("attn_mask 3D shape must be (B, T, T) or (B*num_heads, T, T)")
        return
    raise ValueError("attn_mask must be 2D or 3D")


# -----------------------------
# Shared utilities
# -----------------------------


def count_parameters(model: nn.Module) -> int:
    """Utility to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    # raw / no-norm baselines
    "GRUModel",
    "LSTMModel",
    "TCNModel",
    "TransformerModel",
    # standard implementations
    "TCNModelWN",
    "TransformerModelPreLN",
    # norm-augmented variants
    "TCNModelWNWithNorm",
    "TransformerModelPreLNWithNorm",
    "count_parameters",
]
