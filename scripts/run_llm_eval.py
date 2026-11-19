"""Evaluate Qwen3-0.6B/4B (Zero-Shot, 4-Shot ICL, Clean-LoRA) and MV-CSN + Qwen3 Cascade."""

from __future__ import annotations

import gc
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.llm.qwen_eval import (
    finetune_qwen_clean_lora,
    format_spam_prompt,
    load_qwen_model,
    score_binary_llm,
)
from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.inference import load_demo_bundle
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.metrics import (
    compute_classification_metrics,
    compute_per_attack_f1,
    mcnemar_test,
)
from mvcsn.utils.experiment import DEFAULT_SEEDS, agg_stats, to_clean_record
from mvcsn.utils.seed import get_device, seed_everything
from scripts.gdrive_sync import bootstrap_from_gdrive, persist_to_gdrive

SEEDS = list(DEFAULT_SEEDS)
QWEN_06B_ID = "Qwen/Qwen3-0.6B"
QWEN_4B_ID = "Qwen/Qwen3-4B-Instruct-2507"


def load_mvcsn_from_bundle(bundle_path: str = "artifacts/demo_bundle.pt"):
    """Load MV-CSN v2 model, vocabulary, and OOV bridge from a packed demo bundle."""
    bundle = load_demo_bundle(bundle_path)
    bridge = OOVGraphBridge(bundle.vocab, bundle.graphs, bundle.gmeta)
    model = MVCSNModel.from_config(len(bundle.vocab), bundle.graphs)
    model.load_state_dict(bundle.model_state, strict=True)
    return model, bundle.vocab, bridge


@torch.no_grad()
def run_mvcsn_with_obf(model: MVCSNModel, records: List[SpamRecord], vocab: CharVocab, oov_map: Dict[str, int], device: torch.device):
    ds = SpamDataset(records, vocab, use_preprocessor=True, oov_map=oov_map)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
    model.eval()
    probs: List[float] = []
    max_obfs: List[float] = []
    flagged_chars: List[List[str]] = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    idx_base = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        out = model(ids, mask)
        p = torch.sigmoid(out["spam_logit"]).cpu().tolist()
        obf = torch.sigmoid(out["obf_logits"]).cpu().numpy()
        bsz = len(p)
        for j in range(bsz):
            rec = records[idx_base + j]
            L = min(len(rec.proc_text), obf.shape[1])
            row_obf = obf[j, :L]
            max_obfs.append(float(row_obf.max()) if L > 0 else 0.0)
            bad = []
            for pos in range(L):
                if row_obf[pos] > 0.50:
                    ch = rec.proc_text[pos]
                    if ch not in bad:
                        bad.append(ch)
            flagged_chars.append(bad[:8])
        probs.extend(p)
        idx_base += bsz

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_sample = (time.time() - t0) * 1000.0 / max(1, len(records))
    return probs, max_obfs, flagged_chars, float(ms_per_sample)


def main() -> None:
    bootstrap_from_gdrive()
    t0 = time.time()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")

    recs_b, _ = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    te_real = [r for r in recs_b if r.split == "test"]
    assert len(recs_b) == 16507 and len(te_real) == 3395

    tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
    va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]
    y_te = [r.label for r in te_real]
    te_texts = [r.raw_text for r in te_real]
    test_atks = [r.attack_type for r in te_real]

    rng = random.Random(42)
    clean_hams = [r.raw_text for r in tr_clean if r.label == 0 and 25 <= len(r.raw_text) <= 80]
    clean_spams = [r.raw_text for r in tr_clean if r.label == 1 and 25 <= len(r.raw_text) <= 80]
    few_shot = [
        (rng.choice(clean_hams[:200]), 0),
        (rng.choice(clean_spams[:200]), 1),
        (rng.choice(clean_hams[200:400]), 0),
        (rng.choice(clean_spams[200:400]), 1),
    ]

    summary: Dict[str, Dict[str, object]] = {}
    per_attack: Dict[str, Dict[str, float]] = {}
    preds_map: Dict[str, List[int]] = {}
    params_map: Dict[str, int] = {}

    # 1. Qwen3-0.6B (Zero-Shot)
    print("[LLM 1/5] Evaluating Qwen3-0.6B (Zero-Shot)...", flush=True)
    m06, tok06 = load_qwen_model(QWEN_06B_ID, device=device)
    params_map["Qwen3-0.6B (Zero-Shot)"] = sum(p.numel() for p in m06.parameters())
    p_06_zs, ms_06 = score_binary_llm(m06, tok06, te_texts, batch_size=32, device=device)
    met_06_zs = compute_classification_metrics(y_te, p_06_zs)
    met_06_zs["infer_ms_per_sample"] = ms_06
    summary["Qwen3-0.6B (Zero-Shot)"] = {k: {"mean": float(v), "std": 0.0} for k, v in met_06_zs.items()}
    per_attack["Qwen3-0.6B (Zero-Shot)"] = compute_per_attack_f1(y_te, p_06_zs, test_atks)
    preds_map["Qwen3-0.6B (Zero-Shot)"] = [int(x >= 0.5) for x in p_06_zs]

    va_06_p, _ = score_binary_llm(m06, tok06, [r.raw_text for r in va_clean], batch_size=32, device=device)
    va_06_y = [r.label for r in va_clean]
    best_thr_06, best_va_f1 = 0.5, -1.0
    for thr in np.linspace(0.005, 0.5, 40):
        vf1 = compute_classification_metrics(va_06_y, [1.0 if p >= thr else 0.0 for p in va_06_p])["f1"]
        if vf1 > best_va_f1:
            best_va_f1 = vf1
            best_thr_06 = float(thr)
    met_06_cal = compute_classification_metrics(y_te, [1.0 if p >= best_thr_06 else 0.0 for p in p_06_zs])
    print(f"  -> F1={met_06_zs['f1']:.4f} (va-calibrated@{best_thr_06:.4f} F1={met_06_cal['f1']:.4f}) HamAcc={met_06_zs['clean_ham_acc']:.4f} AUC={met_06_zs['auc']:.4f} ms={ms_06:.2f}", flush=True)
    del m06
    gc.collect()
    torch.cuda.empty_cache()

    # 2. Qwen3-0.6B (Clean-LoRA, 3 seeds)
    print("[LLM 2/5] Fine-tuning Qwen3-0.6B (Clean-LoRA, 3 seeds)...", flush=True)
    lora_runs: List[Dict[str, float]] = []
    lora_atks: List[Dict[str, float]] = []
    for seed in SEEDS:
        seed_everything(seed)
        save_dir = "artifacts/qwen3_06b_clean_lora" if seed == 42 else None
        m_l, p_l, ms_l, p_cnt = finetune_qwen_clean_lora(
            QWEN_06B_ID, tr_clean, va_clean, te_real, device=device, epochs=2, batch_size=32, lr=2e-4, save_dir=save_dir
        )
        m_l["infer_ms_per_sample"] = ms_l
        lora_runs.append(m_l)
        lora_atks.append(compute_per_attack_f1(y_te, p_l, test_atks))
        params_map["Qwen3-0.6B (Clean-LoRA)"] = p_cnt
        if seed == 42:
            preds_map["Qwen3-0.6B (Clean-LoRA)"] = [int(x >= 0.5) for x in p_l]
        print(f"  [Clean-LoRA seed={seed}] F1={m_l['f1']:.4f} HamAcc={m_l['clean_ham_acc']:.4f} AUC={m_l['auc']:.4f} ms={ms_l:.2f}", flush=True)
    summary["Qwen3-0.6B (Clean-LoRA)"] = agg_stats(lora_runs)
    per_attack["Qwen3-0.6B (Clean-LoRA)"] = {k: v["mean"] for k, v in agg_stats(lora_atks).items()}

    # 3 & 4 & 5: Qwen3-4B-Instruct (Zero-Shot, 4-Shot ICL, and MV-CSN v2 + Qwen3-4B Cascade)
    print("[LLM 3/5] Evaluating Qwen3-4B-Instruct (Zero-Shot & 4-Shot ICL)...", flush=True)
    m4b, tok4b = load_qwen_model(QWEN_4B_ID, device=device)
    params_4b = sum(p.numel() for p in m4b.parameters())
    params_map["Qwen3-4B-Instruct (Zero-Shot)"] = params_4b
    params_map["Qwen3-4B-Instruct (4-Shot ICL)"] = params_4b

    p_4b_zs, ms_4b_zs = score_binary_llm(m4b, tok4b, te_texts, batch_size=16, device=device)
    met_4b_zs = compute_classification_metrics(y_te, p_4b_zs)
    met_4b_zs["infer_ms_per_sample"] = ms_4b_zs
    summary["Qwen3-4B-Instruct (Zero-Shot)"] = {k: {"mean": float(v), "std": 0.0} for k, v in met_4b_zs.items()}
    per_attack["Qwen3-4B-Instruct (Zero-Shot)"] = compute_per_attack_f1(y_te, p_4b_zs, test_atks)
    preds_map["Qwen3-4B-Instruct (Zero-Shot)"] = [int(x >= 0.5) for x in p_4b_zs]
    print(f"  -> 4B Zero-Shot: F1={met_4b_zs['f1']:.4f} HamAcc={met_4b_zs['clean_ham_acc']:.4f} AUC={met_4b_zs['auc']:.4f} ms={ms_4b_zs:.2f}", flush=True)

    p_4b_fs, ms_4b_fs = score_binary_llm(m4b, tok4b, te_texts, batch_size=8, max_length=512, few_shot_examples=few_shot, device=device)
    met_4b_fs = compute_classification_metrics(y_te, p_4b_fs)
    met_4b_fs["infer_ms_per_sample"] = ms_4b_fs
    summary["Qwen3-4B-Instruct (4-Shot ICL)"] = {k: {"mean": float(v), "std": 0.0} for k, v in met_4b_fs.items()}
    per_attack["Qwen3-4B-Instruct (4-Shot ICL)"] = compute_per_attack_f1(y_te, p_4b_fs, test_atks)
    preds_map["Qwen3-4B-Instruct (4-Shot ICL)"] = [int(x >= 0.5) for x in p_4b_fs]
    print(f"  -> 4B 4-Shot:    F1={met_4b_fs['f1']:.4f} HamAcc={met_4b_fs['clean_ham_acc']:.4f} AUC={met_4b_fs['auc']:.4f} ms={ms_4b_fs:.2f}", flush=True)

    # 5. MV-CSN v2 + Qwen3-4B Cascade
    print("[LLM 5/5] Evaluating MV-CSN v2 + Qwen3-4B Cascade...", flush=True)
    mv_model, mv_vocab, bridge = load_mvcsn_from_bundle("artifacts/demo_bundle.pt")
    mv_params = sum(p.numel() for p in mv_model.parameters())
    mv_model.to(device)

    p_va_mv, obf_va_mv, _, _ = run_mvcsn_with_obf(mv_model, va_clean, mv_vocab, {}, device)
    va_uncertain_rate = float(np.mean([1.0 if (0.12 < p < 0.88 or (o > 0.65 and 0.05 < p < 0.92)) else 0.0 for p, o in zip(p_va_mv, obf_va_mv)]))

    oov_te, ext_te = bridge.build_extended_bundle([r.proc_text for r in te_real])
    mv_model.set_inference_graph(ext_te)
    p_mv_te, obf_te, flagged_te, ms_mv = run_mvcsn_with_obf(mv_model, te_real, mv_vocab, oov_te, device)
    met_mv_ref = compute_classification_metrics(y_te, p_mv_te)
    preds_map["MV-CSN v2 (seed=42)"] = [int(x >= 0.5) for x in p_mv_te]

    esc_indices = [
        i for i, (p, o) in enumerate(zip(p_mv_te, obf_te))
        if (0.12 < p < 0.88) or (o > 0.65 and 0.05 < p < 0.92)
    ]
    esc_prompts = []
    for idx in esc_indices:
        rec = te_real[idx]
        bad_str = "、".join(flagged_te[idx]) if flagged_te[idx] else "无"
        user_msg = (
            f"原始短信：{rec.raw_text}\n"
            f"MV-CSN规范化还原参考：{rec.proc_text}\n"
            f"疑似对抗混淆字符：{bad_str}\n"
            "请综合原句与还原线索判断是否为垃圾/诈骗短信："
        )
        esc_prompts.append(format_spam_prompt(tok4b, user_msg))

    if esc_prompts:
        p_esc_llm, ms_esc_each = score_binary_llm(m4b, tok4b, [], batch_size=16, max_length=448, raw_prompts=esc_prompts, device=device)
    else:
        p_esc_llm, ms_esc_each = [], ms_4b_zs

    del m4b
    gc.collect()
    torch.cuda.empty_cache()

    p_cascade = list(p_mv_te)
    for k_idx, test_idx in enumerate(esc_indices):
        p_cascade[test_idx] = 0.5 * p_mv_te[test_idx] + 0.5 * p_esc_llm[k_idx]

    esc_rate = len(esc_indices) / max(1, len(te_real))
    eff_ms = ms_mv + esc_rate * ms_esc_each
    met_cas = compute_classification_metrics(y_te, p_cascade)
    met_cas["infer_ms_per_sample"] = eff_ms
    met_cas["escalation_rate"] = esc_rate

    summary["MV-CSN v2 + Qwen3-4B Cascade"] = {k: {"mean": float(v), "std": 0.0} for k, v in met_cas.items()}
    per_attack["MV-CSN v2 + Qwen3-4B Cascade"] = compute_per_attack_f1(y_te, p_cascade, test_atks)
    preds_map["MV-CSN v2 + Qwen3-4B Cascade"] = [int(x >= 0.5) for x in p_cascade]
    params_map["MV-CSN v2 + Qwen3-4B Cascade"] = mv_params + params_4b

    zs_ref = json.loads(Path("results/zeroshot_real_variants.json").read_text())
    ind_ref = json.loads(Path("results/in_dist_results.json").read_text())["benchmark_b"]
    atk_ref = json.loads(Path("results/per_attack_breakdown.json").read_text())

    for ref_name in ("MV-CSN v2", "RBT3", "RBT3 + MV-CSN v2"):
        ref_sum = dict(zs_ref["summary"][ref_name])
        if "infer_ms_per_sample" in ind_ref["summary"].get(ref_name, {}):
            ref_sum["infer_ms_per_sample"] = ind_ref["summary"][ref_name]["infer_ms_per_sample"]
        summary[ref_name] = ref_sum
        per_attack[ref_name] = {k: v["mean"] for k, v in atk_ref[ref_name].items()}
        params_map[ref_name] = int(ind_ref["params"][ref_name])

    mcnemar = {
        "mvcsn_vs_qwen3_06b_zs": mcnemar_test(y_te, preds_map["MV-CSN v2 (seed=42)"], preds_map["Qwen3-0.6B (Zero-Shot)"]),
        "mvcsn_vs_qwen3_4b_zs": mcnemar_test(y_te, preds_map["MV-CSN v2 (seed=42)"], preds_map["Qwen3-4B-Instruct (Zero-Shot)"]),
        "mvcsn_vs_qwen3_4b_4shot": mcnemar_test(y_te, preds_map["MV-CSN v2 (seed=42)"], preds_map["Qwen3-4B-Instruct (4-Shot ICL)"]),
        "mvcsn_vs_qwen3_06b_lora": mcnemar_test(y_te, preds_map["MV-CSN v2 (seed=42)"], preds_map["Qwen3-0.6B (Clean-LoRA)"]),
        "cascade_vs_qwen3_4b_zs": mcnemar_test(y_te, preds_map["MV-CSN v2 + Qwen3-4B Cascade"], preds_map["Qwen3-4B-Instruct (Zero-Shot)"]),
    }

    payload = {
        "summary": summary,
        "per_attack": per_attack,
        "params": params_map,
        "mcnemar": mcnemar,
        "raw_lora_runs": lora_runs,
        "qwen3_06b_zs_val_calibrated": {"threshold": best_thr_06, "metrics": met_06_cal},
        "cascade_stats": {
            "val_uncertain_rate": va_uncertain_rate,
            "test_escalated_count": len(esc_indices),
            "test_total": len(te_real),
            "test_escalation_rate": esc_rate,
            "mvcsn_ms_per_sample": ms_mv,
            "qwen4b_ms_per_escalated": ms_esc_each,
            "effective_ms_per_sample": eff_ms,
            "mvcsn_seed42_metrics": met_mv_ref,
        },
        "few_shot_demos": [{"text": t, "label": l} for t, l in few_shot],
        "elapsed_seconds": time.time() - t0,
    }
    Path("results/llm_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    persist_to_gdrive()
    print("\n=== LLM COMPARISON SUMMARY ===", flush=True)
    for m_name in [
        "Qwen3-0.6B (Zero-Shot)",
        "Qwen3-4B-Instruct (Zero-Shot)",
        "Qwen3-4B-Instruct (4-Shot ICL)",
        "Qwen3-0.6B (Clean-LoRA)",
        "MV-CSN v2",
        "RBT3 + MV-CSN v2",
        "MV-CSN v2 + Qwen3-4B Cascade",
    ]:
        s = summary[m_name]
        ms_val = s.get("infer_ms_per_sample", {}).get("mean", 0.0)
        print(f"  {m_name:30s}: F1={s['f1']['mean']:.4f}+-{s['f1']['std']:.4f} | AUC={s['auc']['mean']:.4f} | HamAcc={s['clean_ham_acc']['mean']:.4f} | ms={ms_val:.3f} | params={params_map[m_name]}", flush=True)


if __name__ == "__main__":
    main()
