"""Unified MV-CSN v2/v3 artifact serialization and inference engine."""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.utils.config import MVCSNConfig


@dataclass
class DemoBundle:
    vocab: CharVocab
    graphs: Dict[str, torch.Tensor]
    gmeta: Dict[str, Any]
    bitmaps: torch.Tensor
    model_state: Dict[str, torch.Tensor]
    model_cfg: Optional[MVCSNConfig] = None


@dataclass
class PredictionResult:
    raw_text: str
    proc_text: str
    spam_prob: float
    pred_label: str
    oov_resolved: List[str]
    chars: List[str]
    obf_probs: List[float]
    alpha_ctx: List[float]
    alpha_phon: List[float]
    alpha_glyph: List[float]
    pool_weights: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def save_demo_bundle(
    model: MVCSNModel,
    vocab: CharVocab,
    graphs: Dict[str, torch.Tensor],
    glyph_meta: Dict[str, Any],
    out_path: str | Path = "artifacts/demo_bundle.pt",
    model_cfg: Optional[MVCSNConfig] = None,
) -> Path:
    model.reset_inference_graph()
    cfg = model_cfg if model_cfg is not None else model.get_config(teacher_feat_dim=0)
    cfg_dict = asdict(cfg)
    cfg_dict["teacher_feat_dim"] = 0

    bitmaps_u8 = (glyph_meta["bitmaps"] * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    clean_state = {
        k: (v.half().cpu() if v.is_floating_point() else v.cpu())
        for k, v in model.state_dict().items()
        if not k.startswith("kd_proj.")
    }
    payload = {
        "stoi": vocab.stoi,
        "graphs": {k: v.half().cpu() if v.is_floating_point() else v.cpu() for k, v in graphs.items()},
        "bitmaps_zlib": zlib.compress(bitmaps_u8.tobytes(), level=6),
        "bitmaps_shape": list(glyph_meta["bitmaps"].shape),
        "valid_mask": glyph_meta["valid_mask"].cpu(),
        "glyph_z_fp16": glyph_meta["glyph_z"].half().cpu(),
        "glyph_pix_fp16": glyph_meta["glyph_pix"].half().cpu(),
        "cae_state": {
            k: v.half().cpu() if v.is_floating_point() else v.cpu()
            for k, v in glyph_meta["cae_state"].items()
        },
        "glyph_cfg": {
            "emb_dim": glyph_meta["emb_dim"],
            "glyph_size": glyph_meta["glyph_size"],
            "cnn_weight": glyph_meta["cnn_weight"],
            "pixel_weight": glyph_meta["pixel_weight"],
        },
        "model_cfg": cfg_dict,
        "model_state": clean_state,
    }
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)
    return dest


def load_demo_bundle(bundle_path: str | Path = "artifacts/demo_bundle.pt") -> DemoBundle:
    raw = torch.load(Path(bundle_path), map_location="cpu", weights_only=False)
    vocab = CharVocab(raw["stoi"])
    graphs = {k: v.float() if v.is_floating_point() else v for k, v in raw["graphs"].items()}

    if "bitmaps_zlib" in raw:
        decompressed = zlib.decompress(raw["bitmaps_zlib"])
        bmp_np = np.frombuffer(decompressed, dtype=np.uint8).reshape(tuple(raw["bitmaps_shape"])).copy()
        bitmaps = torch.from_numpy(bmp_np).float() / 255.0
    elif "bitmaps_uint8" in raw:
        bitmaps = raw["bitmaps_uint8"].float() / 255.0
    else:
        bitmaps = torch.zeros((len(vocab), 32, 32), dtype=torch.float32)

    gmeta: Dict[str, Any] = {
        "bitmaps": bitmaps,
        "valid_mask": raw["valid_mask"],
        "glyph_z": raw["glyph_z_fp16"].float(),
        "glyph_pix": raw["glyph_pix_fp16"].float(),
        "cae_state": {
            k: v.float() if v.is_floating_point() else v for k, v in raw["cae_state"].items()
        },
        **raw["glyph_cfg"],
    }
    model_state = {
        k: (v.float() if v.is_floating_point() else v)
        for k, v in raw["model_state"].items()
        if not k.startswith("kd_proj.")
    }
    model_cfg = None
    if "model_cfg" in raw and isinstance(raw["model_cfg"], dict):
        cd = dict(raw["model_cfg"])
        cd["teacher_feat_dim"] = 0
        model_cfg = MVCSNConfig(**cd)

    return DemoBundle(
        vocab=vocab,
        graphs=graphs,
        gmeta=gmeta,
        bitmaps=bitmaps,
        model_state=model_state,
        model_cfg=model_cfg,
    )


class MVCSNInferenceEngine:
    def __init__(
        self,
        model: MVCSNModel,
        vocab: CharVocab,
        graphs: Dict[str, torch.Tensor],
        gmeta: Dict[str, Any],
        bitmaps: torch.Tensor,
        preproc: TextPreprocessor,
        bridge: OOVGraphBridge,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.graphs = graphs
        self.gmeta = gmeta
        self.bitmaps = bitmaps
        self.preproc = preproc
        self.bridge = bridge

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path = "artifacts/demo_bundle.pt",
        chaizi_path: str | Path = "data/resources/chaizi_map.json.gz",
        font_dir: str | Path = "/content/fonts",
        device: Optional[torch.device] = None,
    ) -> "MVCSNInferenceEngine":
        bundle = load_demo_bundle(bundle_path)
        preproc = TextPreprocessor(chaizi_path)
        bridge = OOVGraphBridge(
            bundle.vocab,
            bundle.graphs,
            bundle.gmeta,
            chaizi_path=chaizi_path,
            font_dir=font_dir,
        )
        model = MVCSNModel.from_config(len(bundle.vocab), bundle.graphs, config=bundle.model_cfg)
        model.load_state_dict(bundle.model_state, strict=True)
        model.eval()
        return cls(
            model=model,
            vocab=bundle.vocab,
            graphs=bundle.graphs,
            gmeta=bundle.gmeta,
            bitmaps=bundle.bitmaps,
            preproc=preproc,
            bridge=bridge,
            device=device,
        )

    def as_legacy_dict(self) -> Dict[str, Any]:
        return {
            "engine": self,
            "model": self.model,
            "vocab": self.vocab,
            "graphs": self.graphs,
            "gmeta": self.gmeta,
            "bitmaps": self.bitmaps,
            "preproc": self.preproc,
            "bridge": self.bridge,
        }

    @torch.no_grad()
    def predict(self, text: str, max_len: int = 192, threshold: float = 0.5) -> PredictionResult:
        proc_text = self.preproc.process(text, enabled=True) or text
        oov_map, ext_bundle = self.bridge.build_extended_bundle([proc_text])
        ids = self.vocab.encode(proc_text, max_len=max_len, oov_map=oov_map)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones((1, len(ids)), dtype=torch.bool, device=self.device)

        with self.model.temporary_inference_graph(ext_bundle if oov_map else None):
            out = self.model(input_ids, attn_mask)

        L = len(ids)
        chars = list(proc_text[:L]) if proc_text else ["<UNK>"]
        spam_prob = float(torch.sigmoid(out["spam_logit"][0]).item())
        obf_probs = [float(x) for x in torch.sigmoid(out["obf_logits"][0, :L]).tolist()]
        alpha = out["alpha"][0, :L].cpu().tolist()
        pool_w = [float(x) for x in out["pool_weights"][0, :L].tolist()]

        n_tok = min(len(chars), len(alpha))
        return PredictionResult(
            raw_text=text,
            proc_text=proc_text,
            spam_prob=spam_prob,
            pred_label="SPAM" if spam_prob >= threshold else "HAM",
            oov_resolved=list(oov_map.keys()),
            chars=chars[:n_tok],
            obf_probs=obf_probs[:n_tok],
            alpha_ctx=[float(a[0]) for a in alpha[:n_tok]],
            alpha_phon=[float(a[1]) for a in alpha[:n_tok]],
            alpha_glyph=[float(a[2]) for a in alpha[:n_tok]],
            pool_weights=pool_w[:n_tok],
        )

    @torch.no_grad()
    def predict_records(
        self,
        records: Sequence[SpamRecord],
        batch_size: int = 128,
        max_len: int = 192,
    ) -> Dict[str, Any]:
        rec_list = list(records)
        oov_map, ext_bundle = self.bridge.build_extended_bundle([r.proc_text for r in rec_list])
        ds = SpamDataset(rec_list, self.vocab, max_len=max_len, use_preprocessor=True, oov_map=oov_map)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_spam_batch)

        probs: List[float] = []
        mean_alphas: List[List[float]] = []
        max_obf_probs: List[float] = []
        top_tokens: List[List[Dict[str, Any]]] = []

        with self.model.temporary_inference_graph(ext_bundle if oov_map else None):
            offset = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attn_mask = batch["attn_mask"].to(self.device)
                out = self.model(input_ids, attn_mask)

                b_probs = torch.sigmoid(out["spam_logit"]).cpu().tolist()
                b_obf = torch.sigmoid(out["obf_logits"]).cpu()
                b_alpha = out["alpha"].cpu()
                b_pool = out["pool_weights"].cpu()
                b_mask = batch["attn_mask"]

                for i, prob_val in enumerate(b_probs):
                    rec = rec_list[offset + i]
                    L = int(b_mask[i].sum().item())
                    alpha_i = b_alpha[i, :L]
                    obf_i = b_obf[i, :L]
                    pool_i = b_pool[i, :L]

                    probs.append(float(prob_val))
                    mean_alphas.append([round(float(x), 4) for x in alpha_i.mean(dim=0).tolist()])
                    max_obf_probs.append(round(float(obf_i.max().item()), 4) if L > 0 else 0.0)

                    chars = list((rec.proc_text or rec.raw_text)[:L])
                    top_idx = torch.topk(pool_i, k=min(4, L)).indices.tolist() if L > 0 else []
                    top_tokens.append(
                        [
                            {
                                "char": chars[k],
                                "pool_w": round(float(pool_i[k].item()), 4),
                                "obf_p": round(float(obf_i[k].item()), 4),
                                "alpha_ctx": round(float(alpha_i[k, 0].item()), 3),
                                "alpha_phon": round(float(alpha_i[k, 1].item()), 3),
                                "alpha_glyph": round(float(alpha_i[k, 2].item()), 3),
                            }
                            for k in top_idx
                            if k < len(chars)
                        ]
                    )
                offset += len(b_probs)

        return {
            "probs": probs,
            "mean_alphas": mean_alphas,
            "max_obf_probs": max_obf_probs,
            "top_tokens": top_tokens,
            "oov_map": oov_map,
        }
