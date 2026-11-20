"""Dense Top-K Multi-Head Context-Conditioned Graph Attention (v2 & Fused Dual-View v3)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sim_bias = nn.Parameter(torch.tensor(1.5))
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h_ctx: torch.Tensor,
        input_ids: torch.Tensor,
        emb_weight: torch.Tensor,
        nbr_idx: torch.Tensor,
        nbr_sim: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = h_ctx.shape
        H, Hd = self.num_heads, self.head_dim

        b_idx = nbr_idx[input_ids]
        b_sim = nbr_sim[input_ids]
        has_nbr = b_sim > 0.0
        any_nbr = has_nbr.any(dim=-1, keepdim=True)

        nbr_emb = F.embedding(b_idx, emb_weight)
        K = b_idx.size(-1)

        q = self.q_proj(h_ctx).view(B, L, 1, H, Hd).permute(0, 1, 3, 2, 4)
        k = self.k_proj(nbr_emb).view(B, L, K, H, Hd).permute(0, 1, 3, 2, 4)
        v = self.v_proj(nbr_emb).view(B, L, K, H, Hd).permute(0, 1, 3, 2, 4)

        scores = (q * k).sum(dim=-1) / math.sqrt(Hd)
        edge_bias = self.sim_bias * torch.log(torch.clamp(b_sim, min=1e-4)).unsqueeze(2)
        scores = scores + edge_bias

        mask = has_nbr.unsqueeze(2)
        scores = scores.masked_fill(~mask, -1e4)
        attn = self.drop(F.softmax(scores, dim=-1) * mask.float())

        agg = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(B, L, D)
        out = self.out_proj(agg)
        out = torch.where(any_nbr, out, torch.zeros_like(out))
        return self.norm(out)


class FusedDualViewGAT(nn.Module):
    """Fused low-latency Dual-View Top-K Graph Attention using precomputed log-sim buffers."""

    def __init__(self, hidden_dim: int = 192, num_heads: int = 4, dropout: float = 0.15) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.k_phon = nn.Linear(hidden_dim, hidden_dim)
        self.v_phon = nn.Linear(hidden_dim, hidden_dim)
        self.k_glyph = nn.Linear(hidden_dim, hidden_dim)
        self.v_glyph = nn.Linear(hidden_dim, hidden_dim)
        self.out_phon = nn.Linear(hidden_dim, hidden_dim)
        self.out_glyph = nn.Linear(hidden_dim, hidden_dim)

        self.sim_bias_phon = nn.Parameter(torch.tensor(1.5))
        self.sim_bias_glyph = nn.Parameter(torch.tensor(1.5))
        self.norm_phon = nn.LayerNorm(hidden_dim)
        self.norm_glyph = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _attend_view(
        self,
        q_view: torch.Tensor,
        b_idx: torch.Tensor,
        b_log_sim: torch.Tensor,
        has_nbr: torch.Tensor,
        emb_weight: torch.Tensor,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        sim_bias: nn.Parameter,
        norm: nn.LayerNorm,
    ) -> torch.Tensor:
        B, L, _ = b_idx.shape
        K = b_idx.size(-1)
        H, Hd, D = self.num_heads, self.head_dim, self.hidden_dim

        any_nbr = has_nbr.any(dim=-1, keepdim=True)
        nbr_emb = F.embedding(b_idx, emb_weight)

        k = k_proj(nbr_emb).view(B, L, K, H, Hd).permute(0, 1, 3, 2, 4)
        v = v_proj(nbr_emb).view(B, L, K, H, Hd).permute(0, 1, 3, 2, 4)

        scores = (q_view * k).sum(dim=-1) * self.scale + sim_bias * b_log_sim.unsqueeze(2)
        mask = has_nbr.unsqueeze(2)
        scores = scores.masked_fill(~mask, -1e4)
        attn = self.drop(F.softmax(scores, dim=-1) * mask.to(scores.dtype))

        agg = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(B, L, D)
        out = out_proj(agg)
        out = torch.where(any_nbr, out, torch.zeros_like(out))
        return norm(out)

    def forward(
        self,
        h_ctx: torch.Tensor,
        input_ids: torch.Tensor,
        emb_weight: torch.Tensor,
        phon_nbr_idx: torch.Tensor,
        phon_log_sim: torch.Tensor,
        phon_has_nbr: torch.Tensor,
        glyph_nbr_idx: torch.Tensor,
        glyph_log_sim: torch.Tensor,
        glyph_has_nbr: torch.Tensor,
        use_phonetic: bool = True,
        use_glyph: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = h_ctx.shape
        H, Hd = self.num_heads, self.head_dim
        q_both = self.q_proj(h_ctx).view(B, L, 2, H, 1, Hd)
        q_p = q_both[:, :, 0]
        q_g = q_both[:, :, 1]

        if use_phonetic:
            h_phon = self._attend_view(
                q_p, phon_nbr_idx[input_ids], phon_log_sim[input_ids], phon_has_nbr[input_ids],
                emb_weight, self.k_phon, self.v_phon, self.out_phon, self.sim_bias_phon, self.norm_phon,
            )
        else:
            h_phon = torch.zeros_like(h_ctx)

        if use_glyph:
            h_glyph = self._attend_view(
                q_g, glyph_nbr_idx[input_ids], glyph_log_sim[input_ids], glyph_has_nbr[input_ids],
                emb_weight, self.k_glyph, self.v_glyph, self.out_glyph, self.sim_bias_glyph, self.norm_glyph,
            )
        else:
            h_glyph = torch.zeros_like(h_ctx)

        return h_phon, h_glyph
