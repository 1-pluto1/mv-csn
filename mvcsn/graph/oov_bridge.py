"""SpellGCN-inspired Dynamic OOV Graph Bridge: attach unseen test chars into trained graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from pypinyin import Style, pinyin

from mvcsn.data.preprocessing import is_cjk
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.glyph import GlyphCAE, GlyphRenderer, _blurred_pixel_features
from mvcsn.graph.phonetic import (
    FINAL_TO_ID,
    INIT_TO_ID,
    _build_lookup_tables,
    _load_radicals,
    extract_char_pinyin_features,
    split_syllable,
)

# Polyphone prior weights matching View 1 phonetic graph construction.
_READING_PRIORS = (1.0, 0.88, 0.76)


class OOVGraphBridge:
    """Dynamically computes multimodal anchors & Top-K train-vocab neighbors for unseen chars.

    When adversarial messages contain characters outside the training vocabulary
    (e.g., Japanese Kana ``え``, cross-script ``Ꝗ``, or rare CJK variants ``髙``),
    this bridge renders them into 32x32 bitmaps, encodes them with the frozen
    ``GlyphCAE``, extracts pinyin features, and appends temporary rows
    ``[V .. V + M - 1]`` pointing exclusively into known training-vocab nodes
    ``[0 .. V - 1]``.
    """

    def __init__(
        self,
        vocab: CharVocab,
        graphs: Dict[str, torch.Tensor],
        glyph_meta: Dict[str, object],
        chaizi_path: str | Path = "data/resources/chaizi_map.json.gz",
        font_dir: str | Path = "/content/fonts",
        phon_min_sim: float = 0.35,
        glyph_min_sim: float = 0.55,
    ) -> None:
        self.vocab = vocab
        self.V = len(vocab)
        self.K = int(graphs["phon_nbr_idx"].size(1))
        self.base_graphs = graphs
        self.phon_min_sim = phon_min_sim
        self.glyph_min_sim = glyph_min_sim

        self.renderer = GlyphRenderer(font_dir=font_dir, size=int(glyph_meta["glyph_size"]))
        self.cae = GlyphCAE(emb_dim=int(glyph_meta["emb_dim"]))
        self.cae.load_state_dict(glyph_meta["cae_state"])
        self.cae.eval()

        self.vocab_valid = glyph_meta["valid_mask"].bool()
        self.vocab_z = glyph_meta["glyph_z"].float()
        self.vocab_pix = glyph_meta["glyph_pix"].float()
        self.cnn_w = float(glyph_meta["cnn_weight"])
        self.pix_w = float(glyph_meta["pixel_weight"])

        self.vocab_pinyin = graphs["pinyin_ids"].long()  # [V, 3]
        self.init_mat, self.fin_mat, self.tone_mat = _build_lookup_tables()
        self.char_rads = _load_radicals(chaizi_path)

    def _collect_oov_chars(self, texts: Iterable[str]) -> List[str]:
        """Collect unique characters absent from the base training vocabulary."""
        oov_chars: List[str] = []
        seen = set(self.vocab.stoi.keys())
        for text in texts:
            for ch in text:
                if ch not in seen:
                    seen.add(ch)
                    oov_chars.append(ch)
        return oov_chars

    def _compute_oov_glyph_neighbors(
        self, oov_chars: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render unseen chars to 32x32 bitmaps and find Top-K visual train-vocab neighbors."""
        bmps, ok_flags = self.renderer.render_vocab(oov_chars)
        z_oov = self.cae.encode(bmps.unsqueeze(1))  # [M, 64]
        z_oov = torch.where(ok_flags.unsqueeze(-1), z_oov, torch.zeros_like(z_oov))
        pix_oov = _blurred_pixel_features(bmps)

        g_sim = self.cnn_w * (z_oov @ self.vocab_z.T) + self.pix_w * (pix_oov @ self.vocab_pix.T)
        valid_mask = ok_flags.unsqueeze(1) & self.vocab_valid.unsqueeze(0)
        g_sim = torch.where(valid_mask, g_sim, torch.zeros_like(g_sim))
        g_sim[:, :2] = 0.0
        g_sim[g_sim < self.glyph_min_sim] = 0.0

        g_vals, g_idx = torch.topk(g_sim, k=min(self.K, self.V), dim=-1)
        g_idx = torch.where(g_vals > 0.0, g_idx, torch.zeros_like(g_idx))
        return z_oov, g_idx, g_vals

    def _compute_oov_phonetic_neighbors(
        self, oov_chars: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pinyin features and Top-K phonetic train-vocab neighbors for unseen CJK chars."""
        M = len(oov_chars)
        py_oov = extract_char_pinyin_features(oov_chars)  # [M, 3]
        vi, vf, vt = self.vocab_pinyin[:, 0], self.vocab_pinyin[:, 1], self.vocab_pinyin[:, 2]
        v_valid = ((vi > 0) | (vf > 0)).float()

        p_vals = torch.zeros((M, self.K), dtype=torch.float32)
        p_idx = torch.zeros((M, self.K), dtype=torch.long)
        for idx_o, ch in enumerate(oov_chars):
            if not is_cjk(ch):
                continue
            py_res = pinyin(ch, style=Style.TONE3, heteronym=True, errors="ignore")
            readings = py_res[0] if (py_res and py_res[0]) else []
            best_row = torch.zeros((self.V,), dtype=torch.float32)

            for r_i, reading in enumerate(readings[:3]):
                ini, fin, ton = split_syllable(reading)
                if not ini and not fin:
                    continue
                ii = INIT_TO_ID.get(ini, 0)
                fi = FINAL_TO_ID.get(fin, 0)
                prior = _READING_PRIORS[r_i] if r_i < len(_READING_PRIORS) else 0.70
                score = (
                    0.38 * self.init_mat[ii, vi]
                    + 0.47 * self.fin_mat[fi, vf]
                    + 0.15 * self.tone_mat[ton, vt]
                ) * prior * v_valid
                best_row = torch.maximum(best_row, score)

            best_row[:2] = 0.0
            best_row[best_row < self.phon_min_sim] = 0.0
            vals, inds = torch.topk(best_row, k=min(self.K, self.V))
            p_vals[idx_o] = vals
            p_idx[idx_o] = torch.where(vals > 0.0, inds, torch.zeros_like(inds))
        return py_oov, p_idx, p_vals

    @torch.no_grad()
    def build_extended_bundle(
        self, texts: Iterable[str]
    ) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        """Attach unseen characters in ``texts`` as extended nodes ``[V .. V+M-1]``.

        Returns:
            Tuple ``(oov_map, ext_bundle)`` where ``oov_map`` maps each unseen
            character to its extended integer ID and ``ext_bundle`` contains the
            concatenated graph and multimodal feature tensors.
        """
        oov_chars = self._collect_oov_chars(texts)
        if not oov_chars:
            return {}, {k: v.clone() for k, v in self.base_graphs.items()}

        oov_map = {ch: self.V + i for i, ch in enumerate(oov_chars)}
        z_oov, g_idx, g_vals = self._compute_oov_glyph_neighbors(oov_chars)
        py_oov, p_idx, p_vals = self._compute_oov_phonetic_neighbors(oov_chars)

        ext_bundle = {
            "phon_nbr_idx": torch.cat([self.base_graphs["phon_nbr_idx"], p_idx], dim=0),
            "phon_nbr_sim": torch.cat([self.base_graphs["phon_nbr_sim"], p_vals], dim=0),
            "glyph_nbr_idx": torch.cat([self.base_graphs["glyph_nbr_idx"], g_idx], dim=0),
            "glyph_nbr_sim": torch.cat([self.base_graphs["glyph_nbr_sim"], g_vals], dim=0),
            "glyph_z": torch.cat([self.base_graphs["glyph_z"], z_oov], dim=0),
            "pinyin_ids": torch.cat([self.base_graphs["pinyin_ids"], py_oov], dim=0),
        }
        return oov_map, ext_bundle
