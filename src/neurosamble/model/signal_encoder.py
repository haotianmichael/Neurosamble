"""
Signal-domain encoder: Conv front-end + N x Mamba2 (M2Caller-style).

Dimension flow (M2Caller Fig. 1)::

    raw (B, L)            # z-normalized current samples
      -> (B, 1, L)
      -> Conv1d(1 -> C1, k=conv_kernel_1, stride=1, pad=same)      (B, C1, L)
      -> Conv1d(C1 -> C2, k=downsample_factor, stride=downsample)  (B, C2, T)
      -> transpose                                                 (B, T, C2)
      -> Linear(C2 -> D)                                           (B, T, D)
      -> N x Mamba2(d_model=D)                                     (B, T, D) = last_hidden

The returned ``last_hidden`` [B, T, D] plugs straight into ``AveragePooler`` and
therefore into ``ContrastiveTrainer`` unchanged.

**Down-sampling trap:** the conv stride turns the raw length ``L`` into ``T``.
The ``attention_mask`` passed in must already be at the *T* resolution (see
``output_length`` / ``downsample_mask`` helpers used by the dataset), otherwise
it will not align with ``last_hidden`` inside the pooler.

``encoder_type`` selects the sequence body: ``'mamba'`` (default, via the
official ``mamba-ssm`` package — no hand-written SSM), ``'transformer'`` or
``'cnn_rnn'`` (reserved for the M2 ablation).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _PreNormResidual(nn.Module):
    """x -> x + sublayer(norm(x))."""

    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sublayer(self.norm(x))


class SignalEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: str = "mamba",
        conv_channels_1: int = 64,
        conv_channels_2: int = 512,
        conv_kernel_1: int = 11,
        downsample_factor: int = 5,
        n_mamba_blocks: int = 6,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        num_heads: int = 6,
        dropout: float = 0.1,
        embedding_dim: int = 384,
        input_signal_len: int = 2000,
        **kwargs,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.downsample_factor = downsample_factor
        self.embedding_dim = embedding_dim
        self.input_signal_len = input_signal_len

        # --- Conv front-end -------------------------------------------------
        self.conv1 = nn.Conv1d(
            1, conv_channels_1, kernel_size=conv_kernel_1,
            stride=1, padding=conv_kernel_1 // 2,
        )
        # Non-overlapping down-sampling conv (kernel == stride) so that
        # T = L // downsample_factor exactly (matches ``output_length``).
        self.conv2 = nn.Conv1d(
            conv_channels_1, conv_channels_2,
            kernel_size=downsample_factor, stride=downsample_factor, padding=0,
        )
        self.act = nn.GELU()
        self.proj = nn.Linear(conv_channels_2, embedding_dim)

        # --- Sequence body --------------------------------------------------
        self.blocks = self._build_body(
            encoder_type, embedding_dim, n_mamba_blocks,
            d_state, d_conv, expand, num_heads, dropout,
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def _build_body(
        self, encoder_type, d_model, n_blocks,
        d_state, d_conv, expand, num_heads, dropout,
    ) -> nn.ModuleList:
        if encoder_type == "mamba":
            try:
                from mamba_ssm import Mamba2
            except ImportError as e:  # pragma: no cover - env dependent
                raise ImportError(
                    "encoder_type='mamba' requires the official mamba-ssm package: "
                    "pip install mamba-ssm causal-conv1d"
                ) from e
            return nn.ModuleList(
                _PreNormResidual(
                    d_model,
                    Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                )
                for _ in range(n_blocks)
            )

        if encoder_type == "transformer":
            return nn.ModuleList(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                )
                for _ in range(n_blocks)
            )

        if encoder_type == "cnn_rnn":
            return nn.ModuleList(
                _PreNormResidual(
                    d_model,
                    _BiGRU(d_model, dropout),
                )
                for _ in range(n_blocks)
            )

        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    # ------------------------------------------------------------------ #
    def output_length(self, input_len: int) -> int:
        """Down-sampled sequence length T for a raw signal of ``input_len`` samples."""
        return input_len // self.downsample_factor

    def downsample_mask(self, attention_mask_l: torch.Tensor) -> torch.Tensor:
        """Reduce an L-resolution mask to the T resolution via strided max-pool."""
        # (B, L) -> (B, 1, L) -> pool -> (B, T)
        m = attention_mask_l.unsqueeze(1).float()
        pooled = torch.nn.functional.max_pool1d(
            m, kernel_size=self.downsample_factor, stride=self.downsample_factor,
        )
        return (pooled.squeeze(1) > 0).long()

    # ------------------------------------------------------------------ #
    def forward(
        self,
        signal: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Accept (B, L) or (B, L, 1)
        if signal.dim() == 3:
            signal = signal.squeeze(-1)
        x = signal.unsqueeze(1).float()          # (B, 1, L)

        x = self.act(self.conv1(x))              # (B, C1, L)
        x = self.act(self.conv2(x))              # (B, C2, T)
        x = x.transpose(1, 2)                    # (B, T, C2)
        x = self.proj(x)                         # (B, T, D)

        # Zero out padded time steps before the sequence body so padding does
        # not leak through the (stateful) Mamba/GRU layers.
        if attention_mask is not None and attention_mask.shape[1] == x.shape[1]:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)                # (B, T, D) = last_hidden


class _BiGRU(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, dropout=0.0,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.drop(out)


def signal_encoder_from_config(cfg) -> "SignalEncoder":
    """Build a ``SignalEncoder`` (and return the pooler) from a SignalModelConfigSchema."""
    encoder = SignalEncoder(
        encoder_type=cfg.encoder_type,
        conv_channels_1=cfg.conv_channels_1,
        conv_channels_2=cfg.conv_channels_2,
        conv_kernel_1=cfg.conv_kernel_1,
        downsample_factor=cfg.downsample_factor,
        n_mamba_blocks=cfg.n_mamba_blocks,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        embedding_dim=cfg.embedding_dim,
        input_signal_len=cfg.input_signal_len,
    )
    return encoder, cfg.pooling
