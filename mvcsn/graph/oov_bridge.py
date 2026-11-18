"""SpellGCN-inspired Dynamic OOV Graph Bridge: attach unseen test chars into trained graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from mvcsn.data.vocab import CharVocab
from mvcsn.graph.glyph import GlyphCAE, GlyphRenderer, _blurred_pixel_features
from mvcsn.graph.phonetic import (
    _build_lookup_tables,
    _load_radicals,
    extract_char_pinyin_features,
    split_syllable,
    INIT_TO_ID,
    FINAL_TO_ID,
)
from pypinyin import Style, pinyin


class OOVGraphBridge:
    """Dynamically computes multimodal anchors & Top-K train-vocab neighbors for unseen chars."""

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

    @torch.no_grad()
    def build_extended_bundle(self, texts: Iterable[str]) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        """Attach unseen characters in texts as extended nodes [V .. V+M-1] pointing into [0 .. V-1]."""
        oov_chars: List[str] = []
        seen = set(self.vocab.stoi.keys())
        for t in texts:
            for ch in t:
                if ch not in seen:
                    seen.add(ch)
                    oov_chars.append(ch)

        if not oov_chars:
            return {}, {
                "phon_nbr_idx": self.base_graphs["phon_nbr_idx"].clone(),
                "phon_nbr_sim": self.base_graphs["phon_nbr_sim"].clone(),
                "glyph_nbr_idx": self.base_graphs["glyph_nbr_idx"].clone(),
                "glyph_nbr_sim": self.base_graphs["glyph_nbr_sim"].clone(),
                "glyph_z": self.base_graphs["glyph_z"].clone(),
                "pinyin_ids": self.base_graphs["pinyin_ids"].clone(),
            }

        M = len(oov_chars)
        oov_map = {ch: self.V + i for i, ch in enumerate(oov_chars)}

        # 1. Visual features & top-K visual neighbors within train vocab
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

        # 2. Phonetic features & top-K phonetic neighbors within train vocab
        py_oov = extract_char_pinyin_features(oov_chars)  # [M, 3]
        vi = self.vocab_pinyin[:, 0]
        vf = self.vocab_pinyin[:, 1]
        vt = self.vocab_pinyin[:, 2]
        v_valid = (vi > 0) | (vf > 0)

        p_vals = torch.zeros((M, self.K), dtype=torch.float32)
        p_idx = torch.zeros((M, self.K), dtype=torch.long)
        for idx_o, ch in enumerate(oov_chars):
            code = ord(ch)
            if not ((0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)):
                continue
            py_res = pinyin(ch, style=Style.TONE3, heteronym=True, errors="ignore")
            readings = py_res[0] if (py_res and py_res[0]) else []
            best_row = torch.zeros((self.V,), dtype=torch.float32)
            priors = [1.0, 0.88, 0.76]
            for r_i, r in enumerate(readings[:3]):
                ini, fin, ton = split_syllable(r)
                if not ini and not fin:
                    continue
                ii = INIT_TO_ID.get(ini, 0)
                fi = FINAL_TO_ID.get(fin, 0)
                s_i = self.init_mat[ii, vi]
                s_f = self.fin_mat[fi, vf]
                s_t = self.tone_mat[ton, vt]
                pr = priors[r_i] if r_i < len(priors) else 0.70
                sc = (0.38 * s_i + 0.47 * s_f + 0.15 * s_t) * pr * v_valid.float()
                best_row = torch.maximum(best_row, sc)
            best_row[:2] = 0.0
            best_row[best_row < self.phon_min_sim] = 0.0
            vals, inds = torch.topk(best_row, k=min(self.K, self.V))
            p_vals[idx_o] = vals
            p_idx[idx_o] = torch.where(vals > 0.0, inds, torch.zeros_like(inds))

        ext_bundle = {
            "phon_nbr_idx": torch.cat([self.base_graphs["phon_nbr_idx"], p_idx], dim=0),
            "phon_nbr_sim": torch.cat([self.base_graphs["phon_nbr_sim"], p_vals], dim=0),
            "glyph_nbr_idx": torch.cat([self.base_graphs["glyph_nbr_idx"], g_idx], dim=0),
            "glyph_nbr_sim": torch.cat([self.base_graphs["glyph_nbr_sim"], g_vals], dim=0),
            "glyph_z": torch.cat([self.base_graphs["glyph_z"], z_oov], dim=0),
            "pinyin_ids": torch.cat([self.base_graphs["pinyin_ids"], py_oov], dim=0),
        }
        return oov_map, ext_bundle
