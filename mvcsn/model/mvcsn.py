"""MV-CSN v2: Multi-View Context-Gated Character Similarity Network."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn

from mvcsn.graph.phonetic import FINALS, INITIALS
from mvcsn.model.encoder import ContextEncoder
from mvcsn.model.gat import ViewGraphAttention
from mvcsn.model.gate import ContextObfuscationGate
from mvcsn.model.heads import AttentionPooling, ObfuscationLocHead, SpamClassifierHead
from mvcsn.utils.config import MVCSNConfig

_GRAPH_BUFFER_KEYS = (
    "phon_nbr_idx",
    "phon_nbr_sim",
    "glyph_nbr_idx",
    "glyph_nbr_sim",
    "glyph_z",
    "pinyin_ids",
)


class MVCSNModel(nn.Module):
    """Dual-view Context-Gated Character Similarity Network (~1.37M parameters).

    Architecture overview:
        1. **Multimodal Character Anchor** (ChineseBERT-inspired): fuses learned
           character ID embeddings with projected 32x32 ``GlyphCAE`` visual
           embeddings and phonetic ``(initial, final, tone)`` embeddings.
        2. **Context Encoder**: parallel local 1-D CNN (k=3, 5) + 2-layer Pre-LN
           Transformer yielding contextual states ``h_ctx``.
        3. **Dual-View Graph Attention**: attends over Top-K phonetic neighbors
           (View 1) and Top-K rendered-glyph neighbors (View 2) conditioned on
           ``h_ctx``, supporting SpellGCN-style runtime OOV node attachment.
        4. **Context Obfuscation Gate**: dynamically interpolates
           ``[h_ctx, h_phon, h_glyph]`` per token via learned simplex weights.
        5. **Dual Prediction Heads**: attention-pooled sequence spam classifier
           plus auxiliary per-token obfuscation localization head.
    """

    def __init__(
        self,
        vocab_size: int,
        phon_nbr_idx: torch.Tensor,
        phon_nbr_sim: torch.Tensor,
        glyph_nbr_idx: torch.Tensor,
        glyph_nbr_sim: torch.Tensor,
        glyph_z: Optional[torch.Tensor] = None,
        pinyin_ids: Optional[torch.Tensor] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        gat_heads: int = 2,
        num_transformer_layers: int = 2,
        transformer_ff_dim: int = 256,
        dropout: float = 0.15,
        max_len: int = 256,
        use_phonetic: bool = True,
        use_glyph: bool = True,
        use_gate: bool = True,
        use_loc_head: bool = True,
        use_multimodal_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.base_vocab_size = vocab_size
        self.use_phonetic = use_phonetic
        self.use_glyph = use_glyph
        self.use_gate = use_gate
        self.use_loc_head = use_loc_head
        self.use_multimodal_anchor = use_multimodal_anchor

        if glyph_z is None:
            glyph_z = torch.zeros((vocab_size, 64), dtype=torch.float32)
        if pinyin_ids is None:
            pinyin_ids = torch.zeros((vocab_size, 3), dtype=torch.long)

        # Register graph adjacency & multimodal feature tables as buffers so they
        # travel cleanly with .to(device) and checkpoints without gradient updates.
        self.register_buffer("phon_nbr_idx", phon_nbr_idx.long())
        self.register_buffer("phon_nbr_sim", phon_nbr_sim.float())
        self.register_buffer("glyph_nbr_idx", glyph_nbr_idx.long())
        self.register_buffer("glyph_nbr_sim", glyph_nbr_sim.float())
        self.register_buffer("glyph_z", glyph_z.float())
        self.register_buffer("pinyin_ids", pinyin_ids.long())
        self._saved_base_buffers: Optional[Dict[str, torch.Tensor]] = None

        # 1. Character & multimodal input anchor projections
        self.char_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)

        glyph_dim = glyph_z.size(1)
        self.init_emb = nn.Embedding(len(INITIALS) + 2, 16, padding_idx=0)
        self.fin_emb = nn.Embedding(len(FINALS) + 2, 24, padding_idx=0)
        self.tone_emb = nn.Embedding(7, 8, padding_idx=0)
        self.pinyin_proj = nn.Linear(48, hidden_dim)
        self.glyph_proj = nn.Linear(glyph_dim, hidden_dim)
        self.anchor_norm = nn.LayerNorm(hidden_dim)

        # 2. Sequence encoder, dual-view GATs, context gate, and prediction heads
        self.encoder = ContextEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ff_dim=transformer_ff_dim,
            dropout=dropout,
            max_len=max_len,
        )
        self.phon_gat = ViewGraphAttention(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)
        self.glyph_gat = ViewGraphAttention(hidden_dim=hidden_dim, num_heads=gat_heads, dropout=dropout)
        self.gate = ContextObfuscationGate(hidden_dim=hidden_dim, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim=hidden_dim)
        self.classifier = SpamClassifierHead(hidden_dim=hidden_dim, dropout=dropout)
        self.loc_head = ObfuscationLocHead(hidden_dim=hidden_dim, dropout=dropout)

    @classmethod
    def from_config(
        cls,
        vocab_size: int,
        graphs: Dict[str, torch.Tensor],
        config: Optional[MVCSNConfig] = None,
    ) -> "MVCSNModel":
        """Construct an :class:`MVCSNModel` from a graph bundle and :class:`MVCSNConfig`.

        Args:
            vocab_size: Number of base vocabulary tokens.
            graphs: Mapping containing ``phon_nbr_idx``, ``phon_nbr_sim``,
                ``glyph_nbr_idx``, ``glyph_nbr_sim``, ``glyph_z``, and ``pinyin_ids``.
            config: Optional architecture/ablation config (defaults to full MV-CSN v2).

        Returns:
            Initialized :class:`MVCSNModel` instance.
        """
        cfg = config or MVCSNConfig()
        return cls(
            vocab_size=vocab_size,
            phon_nbr_idx=graphs["phon_nbr_idx"],
            phon_nbr_sim=graphs["phon_nbr_sim"],
            glyph_nbr_idx=graphs["glyph_nbr_idx"],
            glyph_nbr_sim=graphs["glyph_nbr_sim"],
            glyph_z=graphs.get("glyph_z"),
            pinyin_ids=graphs.get("pinyin_ids"),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            gat_heads=cfg.gat_heads,
            num_transformer_layers=cfg.num_transformer_layers,
            transformer_ff_dim=cfg.transformer_ff_dim,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            use_phonetic=cfg.use_phonetic,
            use_glyph=cfg.use_glyph,
            use_gate=cfg.use_gate,
            use_loc_head=cfg.use_loc_head,
            use_multimodal_anchor=cfg.use_multimodal_anchor,
        )

    def set_inference_graph(self, ext_bundle: Dict[str, torch.Tensor]) -> None:
        """Attach SpellGCN-style dynamic OOV graph rows at inference/evaluation time.

        Prefer :meth:`temporary_inference_graph` when possible so base graph
        buffers are automatically restored even if an exception occurs.
        """
        if self._saved_base_buffers is None:
            self._saved_base_buffers = {
                key: getattr(self, key).detach().clone() for key in _GRAPH_BUFFER_KEYS
            }
        dev = self.char_emb.weight.device
        self.phon_nbr_idx = ext_bundle["phon_nbr_idx"].long().to(dev)
        self.phon_nbr_sim = ext_bundle["phon_nbr_sim"].float().to(dev)
        self.glyph_nbr_idx = ext_bundle["glyph_nbr_idx"].long().to(dev)
        self.glyph_nbr_sim = ext_bundle["glyph_nbr_sim"].float().to(dev)
        self.glyph_z = ext_bundle["glyph_z"].float().to(dev)
        self.pinyin_ids = ext_bundle["pinyin_ids"].long().to(dev)

    def reset_inference_graph(self) -> None:
        """Restore the original training-vocabulary graph buffers after OOV inference."""
        if self._saved_base_buffers is not None:
            dev = self.char_emb.weight.device
            for key, tensor in self._saved_base_buffers.items():
                setattr(self, key, tensor.to(dev))
            self._saved_base_buffers = None

    @contextmanager
    def temporary_inference_graph(
        self, ext_bundle: Optional[Dict[str, torch.Tensor]]
    ) -> Iterator[None]:
        """Context manager attaching an extended OOV graph bundle with guaranteed cleanup.

        Args:
            ext_bundle: Extended graph dictionary from :class:`OOVGraphBridge`,
                or ``None`` to run directly on base vocabulary buffers.
        """
        if ext_bundle is None:
            yield
            return
        self.set_inference_graph(ext_bundle)
        try:
            yield
        finally:
            self.reset_inference_graph()

    def encode_chars(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multimodal anchors, context states, dual-view GATs, and gated fusion.

        Args:
            input_ids: Token indices ``[B, L]``; indices ``>= base_vocab_size``
                resolve via dynamic OOV multimodal & graph tables while falling
                back to ``<UNK>=1`` for raw character embeddings.
            attn_mask: Valid-token boolean mask, shape ``[B, L]``.

        Returns:
            Dictionary of intermediate representations (``e_char``, ``h_ctx``,
            ``h_phon``, ``h_glyph``, ``h_fused``, ``alpha``).
        """
        base_ids = torch.where(
            input_ids < self.base_vocab_size, input_ids, torch.ones_like(input_ids)
        )
        e_raw = self.char_emb(base_ids)  # [B, L, D]

        if self.use_multimodal_anchor:
            gz = self.glyph_z[input_ids]  # [B, L, glyph_dim]
            py = self.pinyin_ids[input_ids]  # [B, L, 3]
            py_cat = torch.cat(
                [
                    self.init_emb(py[..., 0]),
                    self.fin_emb(py[..., 1]),
                    self.tone_emb(py[..., 2]),
                ],
                dim=-1,
            )  # [B, L, 48]
            e_char = self.anchor_norm(
                e_raw + self.emb_drop(self.glyph_proj(gz) + self.pinyin_proj(py_cat))
            )
        else:
            e_char = self.emb_drop(e_raw)

        h_ctx = self.encoder(e_char, attn_mask)

        h_phon = (
            self.phon_gat(h_ctx, input_ids, self.char_emb.weight, self.phon_nbr_idx, self.phon_nbr_sim)
            if self.use_phonetic
            else torch.zeros_like(h_ctx)
        )
        h_glyph = (
            self.glyph_gat(h_ctx, input_ids, self.char_emb.weight, self.glyph_nbr_idx, self.glyph_nbr_sim)
            if self.use_glyph
            else torch.zeros_like(h_ctx)
        )

        h_fused, alpha = self.gate(
            e_char=e_char,
            h_ctx=h_ctx,
            h_phon=h_phon,
            h_glyph=h_glyph,
            use_phonetic=self.use_phonetic,
            use_glyph=self.use_glyph,
            use_gate=self.use_gate,
        )
        return {
            "e_char": e_char,
            "h_ctx": h_ctx,
            "h_phon": h_phon,
            "h_glyph": h_glyph,
            "h_fused": h_fused,
            "alpha": alpha,
        }

    def forward(
        self, input_ids: torch.Tensor, attn_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run full forward pass returning sequence spam logits and token diagnostics.

        Args:
            input_ids: Token indices of shape ``[B, L]``.
            attn_mask: Boolean valid-token mask of shape ``[B, L]``.

        Returns:
            Dictionary containing ``spam_logit`` ``[B]``, ``obf_logits`` ``[B, L]``,
            ``alpha`` ``[B, L, 3]``, ``pool_weights`` ``[B, L]``, ``pooled`` ``[B, D]``,
            and ``h_fused`` ``[B, L, D]``.
        """
        feats = self.encode_chars(input_ids, attn_mask)
        pooled, pool_w = self.pool(feats["h_fused"], attn_mask)
        spam_logit = self.classifier(pooled)
        obf_logits = self.loc_head(feats["h_fused"], feats["h_ctx"], feats["e_char"])
        return {
            "spam_logit": spam_logit,
            "obf_logits": obf_logits,
            "alpha": feats["alpha"],
            "pool_weights": pool_w,
            "pooled": pooled,
            "h_fused": feats["h_fused"],
        }
