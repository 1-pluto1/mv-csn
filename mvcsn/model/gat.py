"""Dense Top-K Multi-Head Context-Conditioned Graph Attention (CPU/MPS/CUDA safe)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewGraphAttention(nn.Module):
    """1-hop relation-aware GAT aggregating Top-K graph neighbors per token.

    For each character token ``i`` with contextual query ``h_ctx[i]`` and Top-K
    graph neighbors ``N(i)`` (either phonetic or visual glyph neighbors) having
    edge similarities ``s_ij``, attention logits combine dot-product context
    compatibility with a learnable log-similarity prior::

        score(i, j) = (q_i · k_j) / sqrt(d_h) + sim_bias * log(max(s_ij, 1e-4))

    This allows surrounding sentence context to disambiguate which homophone or
    homoglyph neighbor restores the adversary's intended canonical character.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 2,
        dropout: float = 0.15,
    ) -> None:
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
        """Aggregate Top-K view neighbors conditioned on sequence context.

        Args:
            h_ctx: Contextual query representations, shape ``[B, L, D]``.
            input_ids: Token indices (including extended OOV IDs), shape ``[B, L]``.
            emb_weight: Base vocabulary embedding matrix, shape ``[V, D]``.
            nbr_idx: Precomputed Top-K neighbor IDs per node, shape ``[V_ext, K]``.
            nbr_sim: Precomputed Top-K edge similarities in ``[0, 1]``, shape ``[V_ext, K]``.

        Returns:
            View-aggregated representations ``h_view``, shape ``[B, L, D]``.
            Tokens with no valid neighbors receive exact zero vectors.
        """
        B, L, D = h_ctx.shape
        H, Hd = self.num_heads, self.head_dim

        # Gather per-token neighbor IDs and edge weights: [B, L, K]
        b_idx = nbr_idx[input_ids]
        b_sim = nbr_sim[input_ids]
        has_nbr = b_sim > 0.0
        any_nbr = has_nbr.any(dim=-1, keepdim=True)  # [B, L, 1]

        nbr_emb = F.embedding(b_idx, emb_weight)  # [B, L, K, D]
        K = b_idx.size(-1)

        # Project queries from context and keys/values from neighbor embeddings:
        # q: [B, L, H, 1, Hd], k/v: [B, L, H, K, Hd]
        q = self.q_proj(h_ctx).view(B, L, 1, H, Hd).permute(0, 1, 3, 2, 4)
        k = self.k_proj(nbr_emb).view(B, L, K, H, Hd).permute(0, 1, 3, 2, 4)
        v = self.v_proj(nbr_emb).view(B, L, K, H, Hd).permute(0, 1, 3, 2, 4)

        scores = (q * k).sum(dim=-1) / math.sqrt(Hd)  # [B, L, H, K]
        edge_bias = self.sim_bias * torch.log(torch.clamp(b_sim, min=1e-4)).unsqueeze(2)
        scores = scores + edge_bias

        mask = has_nbr.unsqueeze(2)  # [B, L, 1, K]
        scores = scores.masked_fill(~mask, -1e4)
        attn = self.drop(F.softmax(scores, dim=-1) * mask.float())

        agg = (attn.unsqueeze(-1) * v).sum(dim=-2).reshape(B, L, D)
        out = self.out_proj(agg)
        out = torch.where(any_nbr, out, torch.zeros_like(out))
        return self.norm(out)
