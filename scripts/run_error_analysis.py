"""CPU Error & Failure-Case Analysis for MV-CSN v2 on Benchmark B Zero-Shot Real Variants.

Delegates bundle loading and dynamic-OOV batch inference to
:class:`mvcsn.inference.MVCSNInferenceEngine`, auditing all FP/FN errors across
the 3,395 Benchmark B real-variant test messages and saving structured results
to ``results/error_analysis.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.inference import MVCSNInferenceEngine
from mvcsn.training.metrics import compute_classification_metrics
from mvcsn.utils.experiment import save_json

BANK_OR_PHONE_RE = re.compile(r"(卡号|建行|农行|邮政|请联系|1[3-9]\d{9})")
FINANCE_HAM_RE = re.compile(r"(期货|交易所|分级基金|开盘价|库存周报|深成指|H股)")
PUBLIC_OR_NOISE_LABEL_RE = re.compile(r"(气象台|预警信号|疫情防控|戴口罩|公开招标|炒粉|炒面)")
MULTI_QMARK_RE = re.compile(r"(\?{2,}|／.*／)")

ATTACK_DISPLAY_ORDER = ("CLEAN_HAM", "MIXED", "PHON", "GLYPH", "SPLIT", "SPACE", "CLEAN")


def classify_error_pattern(rec: SpamRecord, err_type: str, prob: float) -> str:
    """Assign a transparent, rule-backed failure category for error-analysis reporting."""
    txt = rec.proc_text or rec.raw_text
    raw = rec.raw_text
    if err_type == "FP":
        if BANK_OR_PHONE_RE.search(txt):
            return "Benign SMS with bank card / phone contact cues"
        if FINANCE_HAM_RE.search(txt):
            return "Financial / futures market report mimicking spam digits"
        return "Short structured spec or repeated-digit noise in benign text"

    if PUBLIC_OR_NOISE_LABEL_RE.search(raw):
        return "Upstream corpus label noise (public advisory / tender / food order)"
    if MULTI_QMARK_RE.search(raw):
        return "Multi-symbol insertion (??) or vertical-column transposition"
    if 0.35 <= prob < 0.50:
        return "Borderline gray-market promotion near 0.50 threshold"
    return "Conversational / domain-jargon spam without URL"


def _build_error_entry(
    rec: SpamRecord,
    err_type: str,
    prob: float,
    pred: int,
    in_band: bool,
    vocab_stoi: Dict[str, int],
    mean_alpha: Sequence[float],
    max_obf_prob: float,
    top_tokens: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format one misclassified test record with token-level diagnostics."""
    proc = rec.proc_text or rec.raw_text
    oov_chars = sorted({c for c in proc if c not in vocab_stoi})
    return {
        "record_idx": rec.idx,
        "error_type": err_type,
        "category": classify_error_pattern(rec, err_type, prob),
        "true_label": rec.label,
        "pred_label": pred,
        "spam_prob": round(prob, 5),
        "in_uncertainty_band_005_095": in_band,
        "attack_type": rec.attack_type,
        "char_len_raw": len(rec.raw_text),
        "char_len_proc": len(proc),
        "oov_count": len(oov_chars),
        "oov_chars": oov_chars,
        "mean_alpha": {"ctx": mean_alpha[0], "phon": mean_alpha[1], "glyph": mean_alpha[2]},
        "max_token_obf_prob": max_obf_prob,
        "top_attended_tokens": top_tokens,
        "raw_text": rec.raw_text,
        "proc_text": proc,
        "clean_text": rec.clean_text,
    }


def _summarize_per_attack(
    counts: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    """Compute ordered per-attack error counts, error rates, and accuracies."""
    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in ATTACK_DISPLAY_ORDER:
        if bucket not in counts:
            continue
        d = counts[bucket]
        err_n = d["fp"] + d["fn"]
        summary[bucket] = {
            **d,
            "errors": err_n,
            "error_rate": round(err_n / max(1, d["total"]), 5),
            "accuracy": round(d["correct"] / max(1, d["total"]), 5),
        }
    return summary


def run_analysis(
    bundle_path: Path,
    splits_path: Path,
    chaizi_path: Path,
    font_dir: Path,
    out_path: Path,
    cascade_low: float = 0.05,
    cascade_high: float = 0.95,
) -> Dict[str, Any]:
    """Run full Benchmark B error audit on CPU and persist ``results/error_analysis.json``."""
    engine = MVCSNInferenceEngine.from_bundle(
        bundle_path=bundle_path, chaizi_path=chaizi_path, font_dir=font_dir
    )
    recs, _ = build_or_load_splits(cache_path=splits_path)
    te_real = [r for r in recs if r.split == "test"]
    preds = engine.predict_records(te_real)

    probs: List[float] = preds["probs"]
    y_true = [r.label for r in te_real]
    overall = compute_classification_metrics(y_true, probs)

    fps: List[Dict[str, Any]] = []
    fns: List[Dict[str, Any]] = []
    per_attack: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "fp": 0, "fn": 0, "correct": 0}
    )
    fp_cats: Counter[str] = Counter()
    fn_cats: Counter[str] = Counter()
    err_in_band = 0
    total_in_band = 0

    for idx_i, (rec, prob) in enumerate(zip(te_real, probs)):
        pred = int(prob >= 0.5)
        in_band = cascade_low < prob < cascade_high
        total_in_band += int(in_band)

        bucket = "CLEAN_HAM" if rec.label == 0 else rec.attack_type
        per_attack[bucket]["total"] += 1
        if pred == rec.label:
            per_attack[bucket]["correct"] += 1
            continue

        err_in_band += int(in_band)
        err_type = "FP" if rec.label == 0 else "FN"
        per_attack[bucket]["fp" if err_type == "FP" else "fn"] += 1

        entry = _build_error_entry(
            rec=rec,
            err_type=err_type,
            prob=prob,
            pred=pred,
            in_band=in_band,
            vocab_stoi=engine.vocab.stoi,
            mean_alpha=preds["mean_alphas"][idx_i],
            max_obf_prob=preds["max_obf_probs"][idx_i],
            top_tokens=preds["top_tokens"][idx_i],
        )
        if err_type == "FP":
            fp_cats[entry["category"]] += 1
            fps.append(entry)
        else:
            fn_cats[entry["category"]] += 1
            fns.append(entry)

    fps.sort(key=lambda x: x["spam_prob"], reverse=True)
    fns.sort(key=lambda x: x["spam_prob"])
    total_errors = len(fps) + len(fns)

    payload: Dict[str, Any] = {
        "meta": {
            "model": "MV-CSN v2 (Seed-42 Clean-Trained, artifacts/demo_bundle.pt)",
            "benchmark": "Benchmark B (SMS-Matched 16.5K) Zero-Shot Clean -> RealVariant Test Split",
            "test_total": len(te_real),
            "test_ham": sum(1 for y in y_true if y == 0),
            "test_spam": sum(1 for y in y_true if y == 1),
            "resolved_oov_vocab_size": len(preds["oov_map"]),
            "uncertainty_band": [cascade_low, cascade_high],
        },
        "overall_metrics": overall,
        "error_summary": {
            "total_errors": total_errors,
            "overall_error_rate": round(total_errors / len(te_real), 5),
            "false_positives": len(fps),
            "false_negatives": len(fns),
            "test_samples_in_uncertainty_band": total_in_band,
            "errors_in_uncertainty_band": err_in_band,
            "error_capture_rate_by_uncertainty_band": round(err_in_band / max(1, total_errors), 4),
            "fp_mean_char_len": round(float(np.mean([x["char_len_proc"] for x in fps])), 1) if fps else 0.0,
            "fn_mean_char_len": round(float(np.mean([x["char_len_proc"] for x in fns])), 1) if fns else 0.0,
            "fp_categories": dict(fp_cats.most_common()),
            "fn_categories": dict(fn_cats.most_common()),
        },
        "per_attack_errors": _summarize_per_attack(per_attack),
        "false_positives": fps,
        "false_negatives": fns,
    }

    save_json(payload, out_path)
    return payload


def main() -> None:
    """CLI entry point for Benchmark B CPU error analysis."""
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


if __name__ == "__main__":
    main()
