"""MPS/CUDA-safe Dense Top-K Multi-Head Context-Conditioned Graph Attention."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewGraphAttention(nn.Module):
    """1-hop relation-aware GAT aggregating Top-K graph neighbors per character."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
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
        attn = F.softmax(scores, dim=-1)
        attn = attn * mask.float()
        attn = self.drop(attn)

        agg = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(B, L, D)
        out = self.out_proj(agg)
        out = torch.where(any_nbr, out, torch.zeros_like(out))
        return self.norm(out)
