"""Extended sequence models with optional normalization variants.

Includes raw/no-norm baselines (imported from model.py) and new canonical
implementations for TCN (WeightNorm) and Transformer (Pre-LN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from model import GRUModel, LSTMModel, ModelConfig, TCNModel, TransformerModel


# ======================================================
# Normalization factory
# ======================================================


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
    norm = norm_type.lower()
    if norm == "ln":
        return nn.LayerNorm(num_channels)
    if norm == "bn":
        return nn.BatchNorm1d(num_channels)
    if norm == "in":
        return nn.InstanceNorm1d(num_channels, affine=True)
    if norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    raise ValueError(f"Unknown norm_type: {norm_type}")


def apply_sequence_norm(
    x: torch.Tensor,
    norm: Optional[nn.Module],
    norm_type: Optional[str],
) -> torch.Tensor:
    """Apply normalization to (B, T, C) tensors with optional channel-first norms."""
    if norm is None or norm_type is None:
        return x
    if norm_type == "ln":
        return norm(x)
    x = x.transpose(1, 2)
    x = norm(x)
    return x.transpose(1, 2)


# ======================================================
# RNN variants (optional normalization)
# ======================================================


class GRUModelWithNorm(nn.Module):
    """GRU with optional activation normalization on outputs."""

    def __init__(
        self,
        config: ModelConfig,
        norm_type: Optional[str] = None,
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.norm_type = norm_type.lower() if norm_type is not None else None
        self.norm = make_norm(self.norm_type, config.hidden_dim, num_groups=num_groups)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        output = apply_sequence_norm(output, self.norm, self.norm_type)
        last = output[:, -1, :]
        return self.head(last)


class LSTMModelWithNorm(nn.Module):
    """LSTM with optional activation normalization on outputs."""

    def __init__(
        self,
        config: ModelConfig,
        norm_type: Optional[str] = None,
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.norm_type = norm_type.lower() if norm_type is not None else None
        self.norm = make_norm(self.norm_type, config.hidden_dim, num_groups=num_groups)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = apply_sequence_norm(output, self.norm, self.norm_type)
        last = output[:, -1, :]
        return self.head(last)


# ======================================================
# TCN variants (WeightNorm)
# ======================================================


class TemporalBlockWN(nn.Module):
    """Temporal block with WeightNorm on Conv1d, no activation normalization."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
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
        num_groups: int,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation))
        self.norm1 = make_norm(norm_type, out_channels, num_groups=num_groups)
        self.norm2 = make_norm(norm_type, out_channels, num_groups=num_groups)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = None
        self.norm_type = norm_type.lower() if norm_type is not None else None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv1(x)
        out = self._trim(out, self.conv1.padding[0])
        out = self._apply_norm(out, self.norm1)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self._trim(out, self.conv2.padding[0])
        out = self._apply_norm(out, self.norm2)
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

    def _apply_norm(self, x: torch.Tensor, norm: Optional[nn.Module]) -> torch.Tensor:
        if norm is None:
            return x
        if self.norm_type == "ln":
            # LayerNorm on last dimension -> (B, T, C) then back to (B, C, T)
            x = x.transpose(1, 2)
            x = norm(x)
            return x.transpose(1, 2)
        return norm(x)

    @staticmethod
    def _trim(x: torch.Tensor, padding: int) -> torch.Tensor:
        if padding <= 0:
            return x
        return x[:, :, :-padding]


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
        # x: (B, T, F) -> (B, C, T)
        x = x.transpose(1, 2)
        out = self.network(x)
        last = out[:, :, -1]
        return self.head(last)


class TCNModelWNWithNorm(nn.Module):
    """
    TCN with WeightNorm on Conv1d + optional activation norm.
    Norm choices: BN / IN / GN / LN (LN via transpose).
    """

    def __init__(
        self,
        config: ModelConfig,
        kernel_size: int = 3,
        norm_type: Optional[str] = None,
        num_groups: int = 4,
    ) -> None:
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
        # x: (B, T, F) -> (B, C, T)
        x = x.transpose(1, 2)
        out = self.network(x)
        last = out[:, :, -1]
        return self.head(last)


# ======================================================
# Transformer variants (Pre-LN)
# ======================================================


class EncoderBlockPreLN(nn.Module):
    """Transformer encoder block with Pre-LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ln_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C)
        attn_in = self.ln_attn(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=mask, need_weights=False)
        x = x + self.attn_dropout(attn_out)
        ffn_in = self.ln_ffn(x)
        x = x + self.ffn(ffn_in)
        return x


class TransformerModelPreLN(nn.Module):
    """Standard Transformer encoder with Pre-LayerNorm."""

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
        # x: (B, T, F) -> (B, T, C)
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


class TransformerModelPreLNWithNorm(nn.Module):
    """Transformer with Pre-LN + optional extra normalization."""

    def __init__(
        self,
        config: ModelConfig,
        num_heads: int = 2,
        norm_type: Optional[str] = None,
        num_groups: int = 4,
        input_norm: bool = False,
        output_norm: bool = False,
    ) -> None:
        super().__init__()
        if config.hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm_type = norm_type.lower() if norm_type is not None else None
        self.input_norm = (
            make_norm(self.norm_type, config.input_dim, num_groups=num_groups) if input_norm else None
        )
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
        self.output_norm = (
            make_norm(self.norm_type, config.hidden_dim, num_groups=num_groups) if output_norm else None
        )
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, F)
        if self.input_norm is not None:
            x = apply_sequence_norm(x, self.input_norm, self.norm_type)
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
        if self.output_norm is not None:
            x = apply_sequence_norm(x, self.output_norm, self.norm_type)
        last = x[:, -1, :]
        return self.head(last)


# ======================================================
# Utilities
# ======================================================


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
    "GRUModelWithNorm",
    "LSTMModelWithNorm",
    "TCNModelWNWithNorm",
    "TransformerModelPreLNWithNorm",
    "count_parameters",
]
