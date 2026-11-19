"""Sequence Context Encoder: parallel multi-scale local CNN + Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Stored as a non-persistent buffer so sequence position tables do not inflate
    model checkpoints.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positions to token embeddings ``x`` of shape ``[B, L, D]``."""
        return x + self.pe[:, : x.size(1), :]


class ContextEncoder(nn.Module):
    """Parallel local CNN (k=3, 5) followed by a Pre-LN Transformer Encoder.

    The parallel 1-D convolutions capture local n-gram character patterns (e.g.,
    2-to-5 character Chinese spam phrases), while the subsequent Transformer
    layers capture long-range document semantics to form contextualized query
    vectors ``h_ctx`` for downstream multi-view graph attention.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.15,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        half = hidden_dim // 2
        self.conv3 = nn.Conv1d(hidden_dim, half, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, hidden_dim - half, kernel_size=5, padding=2)
        self.cnn_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Disable nested-tensor fast path for uniform CPU/MPS/CUDA determinism.
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, char_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """Encode character embeddings into contextual hidden states.

        Args:
            char_emb: Multimodal character representations, shape ``[B, L, D]``.
            attn_mask: Boolean mask where ``True`` marks valid tokens, shape ``[B, L]``.

        Returns:
            Contextualized sequence states ``h_ctx``, shape ``[B, L, D]``.
        """
        # Stage 1: Multi-scale local n-gram convolution with residual LayerNorm.
        x_t = char_emb.transpose(1, 2)  # [B, D, L]
        local_feats = torch.cat([self.conv3(x_t), self.conv5(x_t)], dim=1).transpose(1, 2)
        h = self.cnn_norm(char_emb + self.drop(self.act(local_feats)))

        # Stage 2: Global sequence context via Pre-LN Transformer.
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=~attn_mask)
        return self.out_norm(h)
