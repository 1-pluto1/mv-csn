"""Sequence Context Encoder: parallel multi-scale (and dilated) local CNN + Pre-LN Transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
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
        return x + self.pe[:, : x.size(1), :]


class ContextEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.15,
        max_len: int = 512,
        use_dilated_conv: bool = False,
    ) -> None:
        super().__init__()
        self.use_dilated_conv = use_dilated_conv
        if use_dilated_conv:
            c1 = hidden_dim // 3
            c2 = hidden_dim // 3
            c3 = hidden_dim - c1 - c2
            self.conv3 = nn.Conv1d(hidden_dim, c1, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(hidden_dim, c2, kernel_size=5, padding=2)
            self.conv3_d2 = nn.Conv1d(hidden_dim, c3, kernel_size=3, padding=2, dilation=2)
        else:
            half = hidden_dim // 2
            self.conv3 = nn.Conv1d(hidden_dim, half, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(hidden_dim, hidden_dim - half, kernel_size=5, padding=2)
            self.conv3_d2 = None

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
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, char_emb: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        mask_f = attn_mask.unsqueeze(-1).to(char_emb.dtype)
        x_masked = char_emb * mask_f
        x_t = x_masked.transpose(1, 2)
        if self.conv3_d2 is not None:
            local_feats = torch.cat(
                [self.conv3(x_t), self.conv5(x_t), self.conv3_d2(x_t)], dim=1
            ).transpose(1, 2)
        else:
            local_feats = torch.cat([self.conv3(x_t), self.conv5(x_t)], dim=1).transpose(1, 2)
        h = self.cnn_norm(x_masked + self.drop(self.act(local_feats))) * mask_f
        h = self.pos_enc(h) * mask_f
        h = self.transformer(h, src_key_padding_mask=~attn_mask)
        return self.out_norm(h) * mask_f
