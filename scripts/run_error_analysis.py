"""CPU Error & Failure-Case Analysis for MV-CSN v2 on Benchmark B Zero-Shot Real Variants.

Loads the verified Seed-42 Clean-trained MV-CSN v2 bundle (`artifacts/demo_bundle.pt`)
and evaluates all 3,395 unseen real adversarial test messages in Benchmark B
(`artifacts/splits_sms_matched.json.gz`), saving a structured FP/FN taxonomy and
per-sample diagnostics to `results/error_analysis.json`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.metrics import compute_classification_metrics

URL_OR_DOMAIN_RE = re.compile(r"(https?://|www\.|\.[a-z]{2,4}\b|t\.cn|dwz)", re.I)
BANK_OR_PHONE_RE = re.compile(r"(卡号|建行|农行|邮政|请联系|1[3-9]\d{9})")
FINANCE_HAM_RE = re.compile(r"(期货|交易所|分级基金|开盘价|库存周报|深成指|H股)")
PUBLIC_OR_NOISE_LABEL_RE = re.compile(r"(气象台|预警信号|疫情防控|戴口罩|公开招标|炒粉|炒面)")
MULTI_QMARK_RE = re.compile(r"(\?{2,}|／.*／)")
CONTACT_RE = re.compile(r"(微信|威信|微芯|薇信|扣扣|qq|tel|电话|热线|客服|退订|回t|加微|电联|联系)", re.I)


def classify_error_pattern(rec: SpamRecord, err_type: str, prob: float, oov_chars: List[str]) -> str:
    """Assign a transparent, rule-backed failure category for error-analysis reporting."""
    txt = rec.proc_text or rec.raw_text
    raw = rec.raw_text
    if err_type == "FP":
        if BANK_OR_PHONE_RE.search(txt):
            return "Benign SMS with bank card / phone contact cues"
        if FINANCE_HAM_RE.search(txt):
            return "Financial / futures market report mimicking spam digits"
        return "Short structured spec or repeated-digit noise in benign text"
    else:
        if PUBLIC_OR_NOISE_LABEL_RE.search(raw):
            return "Upstream corpus label noise (public advisory / tender / food order)"
        if MULTI_QMARK_RE.search(raw):
            return "Multi-symbol insertion (??) or vertical-column transposition"
        if 0.35 <= prob < 0.50:
            return "Borderline gray-market promotion near 0.50 threshold"
        return "Conversational / domain-jargon spam without URL"


def load_bundle_model(
    bundle_path: Path,
    chaizi_path: Path,
    font_dir: Path,
) -> tuple[MVCSNModel, CharVocab, OOVGraphBridge, TextPreprocessor]:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    vocab = CharVocab(bundle["stoi"])
    graphs = {k: v.float() if v.is_floating_point() else v for k, v in bundle["graphs"].items()}

    raw_bytes = zlib.decompress(bundle["bitmaps_zlib"])
    bmp_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(tuple(bundle["bitmaps_shape"])).copy()
    bitmaps = torch.from_numpy(bmp_np).float() / 255.0

    gmeta = {
        "bitmaps": bitmaps,
        "valid_mask": bundle["valid_mask"],
        "glyph_z": bundle["glyph_z_fp16"].float(),
        "glyph_pix": bundle["glyph_pix_fp16"].float(),
        "cae_state": {k: v.float() if v.is_floating_point() else v for k, v in bundle["cae_state"].items()},
        **bundle["glyph_cfg"],
    }
    preproc = TextPreprocessor(chaizi_path)
    bridge = OOVGraphBridge(vocab, graphs, gmeta, chaizi_path=chaizi_path, font_dir=font_dir)

    model = MVCSNModel(
        len(vocab),
        graphs["phon_nbr_idx"],
        graphs["phon_nbr_sim"],
        graphs["glyph_nbr_idx"],
        graphs["glyph_nbr_sim"],
        glyph_z=graphs["glyph_z"],
        pinyin_ids=graphs["pinyin_ids"],
    )
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in bundle["model_state"].items()})
    model.eval()
    return model, vocab, bridge, preproc


@torch.no_grad()
def run_analysis(
    bundle_path: Path,
    splits_path: Path,
    chaizi_path: Path,
    font_dir: Path,
    out_path: Path,
    cascade_low: float = 0.05,
    cascade_high: float = 0.95,
) -> Dict[str, object]:
    model, vocab, bridge, _ = load_bundle_model(bundle_path, chaizi_path, font_dir)
    recs, split_meta = build_or_load_splits(cache_path=splits_path)
    te_real = [r for r in recs if r.split == "test"]

    oov_map, ext_bundle = bridge.build_extended_bundle([r.proc_text for r in te_real])
    ds = SpamDataset(te_real, vocab, max_len=192, use_preprocessor=True, oov_map=oov_map)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_spam_batch)

    model.set_inference_graph(ext_bundle)
    probs: List[float] = []
    mean_alphas: List[List[float]] = []
    max_obf_probs: List[float] = []
    top_tokens: List[List[Dict[str, object]]] = []

    offset = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        out = model(input_ids, attn_mask)
        b_probs = torch.sigmoid(out["spam_logit"]).tolist()
        b_obf = torch.sigmoid(out["obf_logits"])
        b_alpha = out["alpha"]
        b_pool = out["pool_weights"]

        for i in range(len(b_probs)):
            rec = te_real[offset + i]
            L = int(attn_mask[i].sum().item())
            alpha_i = b_alpha[i, :L]
            obf_i = b_obf[i, :L]
            pool_i = b_pool[i, :L]
            probs.append(float(b_probs[i]))
            mean_alphas.append([round(float(x), 4) for x in alpha_i.mean(dim=0).tolist()])
            max_obf_probs.append(round(float(obf_i.max().item()), 4) if L > 0 else 0.0)

            chars = list((rec.proc_text or rec.raw_text)[:L])
            top_idx = torch.topk(pool_i, k=min(4, L)).indices.tolist() if L > 0 else []
            toks = []
            for idx_k in top_idx:
                if idx_k < len(chars):
                    toks.append({
                        "char": chars[idx_k],
                        "pool_w": round(float(pool_i[idx_k].item()), 4),
                        "obf_p": round(float(obf_i[idx_k].item()), 4),
                        "alpha_ctx": round(float(alpha_i[idx_k, 0].item()), 3),
                        "alpha_phon": round(float(alpha_i[idx_k, 1].item()), 3),
                        "alpha_glyph": round(float(alpha_i[idx_k, 2].item()), 3),
                    })
            top_tokens.append(toks)
        offset += len(b_probs)
    model.reset_inference_graph()

    y_true = [r.label for r in te_real]
    overall = compute_classification_metrics(y_true, probs)

    fps: List[Dict[str, object]] = []
    fns: List[Dict[str, object]] = []
    per_attack: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "fp": 0, "fn": 0, "correct": 0})
    fp_cats = Counter()
    fn_cats = Counter()
    err_in_cascade_band = 0
    total_in_cascade_band = 0

    for idx_i, (rec, prob) in enumerate(zip(te_real, probs)):
        pred = int(prob >= 0.5)
        in_band = cascade_low < prob < cascade_high
        if in_band:
            total_in_cascade_band += 1

        atk_bucket = "CLEAN_HAM" if rec.label == 0 else rec.attack_type
        per_attack[atk_bucket]["total"] += 1
        if pred == rec.label:
            per_attack[atk_bucket]["correct"] += 1
            continue

        if in_band:
            err_in_cascade_band += 1

        err_type = "FP" if (rec.label == 0 and pred == 1) else "FN"
        if err_type == "FP":
            per_attack[atk_bucket]["fp"] += 1
        else:
            per_attack[atk_bucket]["fn"] += 1

        proc = rec.proc_text or rec.raw_text
        oov_chars = sorted({c for c in proc if c not in vocab.stoi})
        category = classify_error_pattern(rec, err_type, prob, oov_chars)
        if err_type == "FP":
            fp_cats[category] += 1
        else:
            fn_cats[category] += 1

        entry = {
            "record_idx": rec.idx,
            "error_type": err_type,
            "category": category,
            "true_label": rec.label,
            "pred_label": pred,
            "spam_prob": round(prob, 5),
            "in_uncertainty_band_005_095": in_band,
            "attack_type": rec.attack_type,
            "char_len_raw": len(rec.raw_text),
            "char_len_proc": len(proc),
            "oov_count": len(oov_chars),
            "oov_chars": oov_chars,
            "mean_alpha": {
                "ctx": mean_alphas[idx_i][0],
                "phon": mean_alphas[idx_i][1],
                "glyph": mean_alphas[idx_i][2],
            },
            "max_token_obf_prob": max_obf_probs[idx_i],
            "top_attended_tokens": top_tokens[idx_i],
            "raw_text": rec.raw_text,
            "proc_text": proc,
            "clean_text": rec.clean_text,
        }
        if err_type == "FP":
            fps.append(entry)
        else:
            fns.append(entry)

    fps.sort(key=lambda x: x["spam_prob"], reverse=True)
    fns.sort(key=lambda x: x["spam_prob"])

    per_attack_summary = {}
    for k in ["CLEAN_HAM", "MIXED", "PHON", "GLYPH", "SPLIT", "SPACE", "CLEAN"]:
        if k in per_attack:
            d = per_attack[k]
            err_n = d["fp"] + d["fn"]
            per_attack_summary[k] = {
                **d,
                "errors": err_n,
                "error_rate": round(err_n / max(1, d["total"]), 5),
                "accuracy": round(d["correct"] / max(1, d["total"]), 5),
            }

    total_errors = len(fps) + len(fns)
    payload = {
        "meta": {
            "model": "MV-CSN v2 (Seed-42 Clean-Trained, artifacts/demo_bundle.pt)",
            "benchmark": "Benchmark B (SMS-Matched 16.5K) Zero-Shot Clean -> RealVariant Test Split",
            "test_total": len(te_real),
            "test_ham": sum(1 for y in y_true if y == 0),
            "test_spam": sum(1 for y in y_true if y == 1),
            "resolved_oov_vocab_size": len(oov_map),
            "uncertainty_band": [cascade_low, cascade_high],
        },
        "overall_metrics": overall,
        "error_summary": {
            "total_errors": total_errors,
            "overall_error_rate": round(total_errors / len(te_real), 5),
            "false_positives": len(fps),
            "false_negatives": len(fns),
            "test_samples_in_uncertainty_band": total_in_cascade_band,
            "errors_in_uncertainty_band": err_in_cascade_band,
            "error_capture_rate_by_uncertainty_band": round(err_in_cascade_band / max(1, total_errors), 4),
            "fp_mean_char_len": round(float(np.mean([x["char_len_proc"] for x in fps])), 1) if fps else 0.0,
            "fn_mean_char_len": round(float(np.mean([x["char_len_proc"] for x in fns])), 1) if fns else 0.0,
            "fp_categories": dict(fp_cats.most_common()),
            "fn_categories": dict(fn_cats.most_common()),
        },
        "per_attack_errors": per_attack_summary,
        "false_positives": fps,
        "false_negatives": fns,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MV-CSN v2 error analysis on Benchmark B test split.")
    parser.add_argument("--bundle", type=Path, default=ROOT_DIR / "artifacts/demo_bundle.pt")
    parser.add_argument("--splits", type=Path, default=ROOT_DIR / "artifacts/splits_sms_matched.json.gz")
    parser.add_argument("--chaizi", type=Path, default=ROOT_DIR / "data/resources/chaizi_map.json.gz")
    parser.add_argument("--font-dir", type=Path, default=ROOT_DIR / "fonts")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "results/error_analysis.json")
    args = parser.parse_args()

    res = run_analysis(args.bundle, args.splits, args.chaizi, args.font_dir, args.out)
    print("Overall metrics:", json.dumps(res["overall_metrics"], indent=2))
    print("Error summary:", json.dumps(res["error_summary"], indent=2, ensure_ascii=False))
    print("Per-attack errors:", json.dumps(res["per_attack_errors"], indent=2, ensure_ascii=False))
    print(f"\n--- False Positives ({len(res['false_positives'])}) ---")
    for e in res["false_positives"]:
        print(f"  [FP p={e['spam_prob']:.4f} len={e['char_len_proc']} oov={e['oov_count']}] ({e['category']}) {e['raw_text']}")
    print(f"\n--- False Negatives ({len(res['false_negatives'])}) ---")
    for e in res["false_negatives"]:
        print(f"  [FN p={e['spam_prob']:.4f} atk={e['attack_type']} len={e['char_len_proc']} oov={e['oov_count']}] ({e['category']}) {e['raw_text']}")


if __name__ == "__main__":
    main()
