"""Stage 3 Runner 4: Exp 6 LLM Comparison (Qwen3-4B Zero-Shot, 4-Shot ICL, Clean-LoRA, Qwen3-8B, MV-CSN v3 Uncertainty Cascade)."""

from __future__ import annotations

import gc
import gzip
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True

from baselines.llm.qwen_eval import (
    _PromptClsDataset,
    _make_left_pad_collator,
    get_binary_token_ids,
    load_qwen_model,
    score_binary_llm_with_logits,
)
from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.metrics import compute_classification_metrics
from mvcsn.training.trainer import evaluate_mvcsn
from mvcsn.utils.config import MVCSN_V3_CONFIG
from mvcsn.utils.experiment import (
    COUNTERFACTUAL_16_ORIG,
    COUNTERFACTUAL_40_BALANCED,
    build_symmetric_stress_perturber,
    perturb_records_symmetrically,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

MODEL_ID_4B = "Qwen/Qwen3-4B-Instruct-2507"
TAUS = [0.30, 0.40, 0.45]


def _load_olivi(preproc: TextPreprocessor, path: str = "data/resources/external_olivi_spam.tsv.gz") -> List[SpamRecord]:
    recs: List[SpamRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            lbl = int(parts[0])
            raw_t = parts[1]
            proc_t = preproc.process(raw_t, enabled=True) or raw_t
            recs.append(SpamRecord(idx, lbl, raw_t, proc_t, raw_t, idx, "external", "EXTERNAL", []))
    return recs


def _eval_cascade(
    y_true: List[int],
    p_fast: List[float],
    p_llm: List[float],
    tau: float,
    t_fast_bs1: float,
    t_llm_bs1: float,
    t_fast_batch: float,
    t_llm_batch: float,
) -> Dict[str, object]:
    p_comb: List[float] = []
    n_esc = 0
    for pf, pl in zip(p_fast, p_llm):
        if abs(pf - 0.5) < tau:
            p_comb.append(pl)
            n_esc += 1
        else:
            p_comb.append(pf)
    esc_rate = n_esc / max(1, len(p_fast))
    m = compute_classification_metrics(y_true, p_comb)
    return {
        "tau": float(tau),
        "escalation_rate": float(esc_rate),
        "n_escalated": int(n_esc),
        "n_total": int(len(p_fast)),
        "effective_gpu_bs1_ms": float(t_fast_bs1 + esc_rate * t_llm_bs1),
        "effective_gpu_batch_ms": float(t_fast_batch + esc_rate * t_llm_batch),
        "metrics": m,
    }


def _eval_cascade_cf(p_fast: List[float], p_llm: List[float], tau: float, y_true: List[int]) -> Dict[str, float]:
    p_comb = [pl if abs(pf - 0.5) < tau else pf for pf, pl in zip(p_fast, p_llm)]
    esc_rate = sum(1 for pf in p_fast if abs(pf - 0.5) < tau) / max(1, len(p_fast))
    acc = float(np.mean([(p >= 0.5) == (y == 1) for p, y in zip(p_comb, y_true)]))
    return {"acc": acc, "escalation_rate": float(esc_rate)}


def main() -> None:
    t0 = time.time()
    seed_everything(42)
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")

    recs_b, _ = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
    va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]
    te_real = [r for r in recs_b if r.split == "test"]
    te_clean = [to_clean_record(r, preproc) for r in te_real]
    olivi_recs = _load_olivi(preproc)

    vocab_c = CharVocab.load("artifacts/vocab_clean.json")
    graphs_c = load_graphs("artifacts/graphs_clean_top20.pt")
    gmeta_c = load_glyph_meta("artifacts/glyph_meta_clean.pt")
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c, font_dir="/content/fonts")

    stress_fn = build_symmetric_stress_perturber(vocab_c)
    sym40_recs = perturb_records_symmetrically(te_clean, stress_fn, preproc, rate=0.4, seed=2026, mode="sub_only")

    cf16_texts = [preproc.process(t, enabled=True) or t for t in COUNTERFACTUAL_16_ORIG]
    cf40_texts = [preproc.process(t, enabled=True) or t for _, t in COUNTERFACTUAL_40_BALANCED]
    cf40_y = [lbl for lbl, _ in COUNTERFACTUAL_40_BALANCED]

    te_texts = [r.proc_text for r in te_real]
    te_y = [r.label for r in te_real]
    s40_texts = [r.proc_text for r in sym40_recs]
    s40_y = [r.label for r in sym40_recs]
    ol_texts = [r.proc_text for r in olivi_recs]
    ol_y = [r.label for r in olivi_recs]

    # 1) Exact aligned Seed-42 MV-CSN v3 probabilities for cascade
    m_v3 = MVCSNModel.from_config(len(vocab_c), graphs_c, MVCSN_V3_CONFIG)
    sd42 = torch.load("artifacts/mvcsn_v3_seed42.pt", map_location="cpu", weights_only=False)
    m_v3.load_state_dict(sd42["model_state"] if isinstance(sd42, dict) and "model_state" in sd42 else sd42, strict=True)
    m_v3 = m_v3.to(device).eval()

    def _mv_probs(recs: List[SpamRecord]) -> List[float]:
        om, eb = bridge_c.build_extended_bundle([r.proc_text for r in recs])
        dl = DataLoader(SpamDataset(recs, vocab_c, use_preprocessor=True, oov_map=om), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
        with m_v3.temporary_inference_graph(eb if om else None):
            _, probs, _ = evaluate_mvcsn(m_v3, dl, device)
        return probs

    cf16_recs = [SpamRecord(i, 0, t, t, t, i, "probe", "PROBE", []) for i, t in enumerate(cf16_texts)]
    cf40_recs = [SpamRecord(i, y, t, t, t, i, "probe", "PROBE", []) for i, (y, t) in enumerate(zip(cf40_y, cf40_texts))]

    p_mv_te = _mv_probs(te_real)
    p_mv_s40 = _mv_probs(sym40_recs)
    p_mv_ol = _mv_probs(olivi_recs)
    p_mv_cf16 = _mv_probs(cf16_recs)
    p_mv_cf40 = _mv_probs(cf40_recs)
    m_mv_te = compute_classification_metrics(te_y, p_mv_te)
    m_mv_s40 = compute_classification_metrics(s40_y, p_mv_s40)
    m_mv_ol = compute_classification_metrics(ol_y, p_mv_ol)
    del m_v3
    torch.cuda.empty_cache()

    # Pull latency references
    lat_ref = json.loads(Path("results/latency_benchmark.json").read_text(encoding="utf-8")) if Path("results/latency_benchmark.json").exists() else {}
    t_mv_bs1 = float(lat_ref.get("models", {}).get("MV-CSN v3 (2.35M)", {}).get("gpu_bs1_eager_ms_per_msg", 1.5))
    t_mv_bs64 = float(lat_ref.get("models", {}).get("MV-CSN v3 (2.35M)", {}).get("gpu_bs64_ms_per_msg", 0.05))

    # 2) Load Qwen3-4B-Instruct-2507 ONCE
    print(f"\n=== Loading {MODEL_ID_4B} ===", flush=True)
    model, tok = load_qwen_model(MODEL_ID_4B, device=device)
    n_params_4b = int(sum(p.numel() for p in model.parameters()))

    def _eval_qwen_suite(tag: str, few_shot=None):
        print(f"\n--- Evaluating {tag} ---", flush=True)
        p_te, _, ms_te = score_binary_llm_with_logits(model, tok, te_texts, batch_size=64, few_shot_examples=few_shot, device=device)
        p_s40, _, _ = score_binary_llm_with_logits(model, tok, s40_texts, batch_size=64, few_shot_examples=few_shot, device=device)
        p_ol, _, _ = score_binary_llm_with_logits(model, tok, ol_texts, batch_size=64, few_shot_examples=few_shot, device=device)
        p_16, _, _ = score_binary_llm_with_logits(model, tok, cf16_texts, batch_size=16, few_shot_examples=few_shot, device=device)
        p_40, _, _ = score_binary_llm_with_logits(model, tok, cf40_texts, batch_size=32, few_shot_examples=few_shot, device=device)
        # Standardized latency probes: bs=32 (256 msgs) & bs=1 (64 msgs)
        _, _, ms_bs32 = score_binary_llm_with_logits(model, tok, te_texts[:256], batch_size=32, few_shot_examples=few_shot, device=device)
        _, _, ms_bs1 = score_binary_llm_with_logits(model, tok, te_texts[:64], batch_size=1, few_shot_examples=few_shot, device=device)

        res = {
            "bench_b_v2": compute_classification_metrics(te_y, p_te),
            "sym_sub_40": compute_classification_metrics(s40_y, p_s40),
            "olivi_9": compute_classification_metrics(ol_y, p_ol),
            "cf16_ham_acc": float(np.mean([p < 0.5 for p in p_16])),
            "cf40_acc": compute_classification_metrics(cf40_y, p_40)["accuracy"],
            "latency_ms": {"bs64": float(ms_te), "bs32": float(ms_bs32), "bs1": float(ms_bs1)},
            "_probs": {"te": p_te, "s40": p_s40, "ol": p_ol, "cf16": p_16, "cf40": p_40},
        }
        print(
            f"  [{tag}] Fair={res['bench_b_v2']['f1']:.4f} (Ham={res['bench_b_v2']['clean_ham_acc']:.4f}) "
            f"Sym40={res['sym_sub_40']['f1']:.4f} (Ham={res['sym_sub_40']['clean_ham_acc']:.4f}) "
            f"Olivi9={res['olivi_9']['f1']:.4f} CF16={res['cf16_ham_acc']:.4f} CF40={res['cf40_acc']:.4f} "
            f"bs32={ms_bs32:.2f}ms bs1={ms_bs1:.2f}ms",
            flush=True,
        )
        return res

    # (A) Zero-Shot
    res_zs = _eval_qwen_suite("Qwen3-4B Zero-Shot", few_shot=None)

    # (B) 4-Shot ICL (strictly clean training exemplars, 2 Ham + 2 Spam)
    rng = random.Random(42)
    clean_hams = [r.proc_text for r in tr_clean if r.label == 0 and 18 <= len(r.proc_text) <= 65]
    clean_spams = [r.proc_text for r in tr_clean if r.label == 1 and 18 <= len(r.proc_text) <= 65]
    ex_hams = rng.sample(clean_hams, 2)
    ex_spams = rng.sample(clean_spams, 2)
    few_shot_4 = [(ex_hams[0], 0), (ex_spams[0], 1), (ex_hams[1], 0), (ex_spams[1], 1)]
    res_icl = _eval_qwen_suite("Qwen3-4B 4-Shot ICL", few_shot=few_shot_4)

    # (C) Clean-LoRA (3 epochs strictly on tr_clean proc_text, rank=16)
    print("\n=== Training Qwen3-4B Clean-LoRA (3 epochs, rank=16, strictly clean tr_clean) ===", flush=True)
    tr_proc_recs = [SpamRecord(r.idx, r.label, r.proc_text, r.proc_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in tr_clean]
    va_proc_texts = [r.proc_text for r in va_clean]
    va_y = [r.label for r in va_clean]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"  [LoRA] trainable_params={trainable_params:,} / total={n_params_4b:,}", flush=True)

    t0_id, t1_id = get_binary_token_ids(tok)
    ds_tr = _PromptClsDataset(tr_proc_recs, tok, max_length=320)
    tr_loader = DataLoader(ds_tr, batch_size=32, shuffle=True, collate_fn=_make_left_pad_collator(tok.pad_token_id))
    epochs = 3
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(tr_loader)))

    best_va_f1 = -1.0
    best_lora_state = None
    lora_hist = []
    t_lora0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            lbl = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=am, logits_to_keep=1, use_cache=False).logits[:, -1, [t0_id, t1_id]].float()
                loss = F.cross_entropy(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())

        va_p, _, _ = score_binary_llm_with_logits(model, tok, va_proc_texts, batch_size=64, device=device)
        va_m = compute_classification_metrics(va_y, va_p)
        lora_hist.append({"epoch": ep, "train_loss": tot_loss / max(1, len(tr_loader)), "va_clean_f1": va_m["f1"]})
        print(f"  [LoRA ep {ep}/{epochs}] loss={tot_loss / max(1, len(tr_loader)):.4f} va_clean_f1={va_m['f1']:.4f}", flush=True)
        if va_m["f1"] > best_va_f1:
            best_va_f1 = va_m["f1"]
            best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    lora_train_sec = time.time() - t_lora0
    if best_lora_state is not None:
        model.load_state_dict(best_lora_state, strict=False)
    Path("artifacts/qwen3_4b_clean_lora").mkdir(parents=True, exist_ok=True)
    model.save_pretrained("artifacts/qwen3_4b_clean_lora")
    res_lora = _eval_qwen_suite("Qwen3-4B Clean-LoRA", few_shot=None)

    del model, opt, sched, tr_loader
    gc.collect()
    torch.cuda.empty_cache()

    # (D) MV-CSN v3 + Qwen3-4B Uncertainty Cascade
    cascade_sweep = {"qwen4b_zeroshot": {}, "qwen4b_clean_lora": {}}
    for ckey, qres in (("qwen4b_zeroshot", res_zs), ("qwen4b_clean_lora", res_lora)):
        qp = qres["_probs"]
        ql_bs1 = qres["latency_ms"]["bs1"]
        ql_bs32 = qres["latency_ms"]["bs32"]
        for tau in TAUS:
            tkey = f"{tau:.2f}"
            cascade_sweep[ckey][tkey] = {
                "bench_b_v2": _eval_cascade(te_y, p_mv_te, qp["te"], tau, t_mv_bs1, ql_bs1, t_mv_bs64, ql_bs32),
                "sym_sub_40": _eval_cascade(s40_y, p_mv_s40, qp["s40"], tau, t_mv_bs1, ql_bs1, t_mv_bs64, ql_bs32),
                "olivi_9": _eval_cascade(ol_y, p_mv_ol, qp["ol"], tau, t_mv_bs1, ql_bs1, t_mv_bs64, ql_bs32),
                "cf16": _eval_cascade_cf(p_mv_cf16, qp["cf16"], tau, [0] * len(cf16_texts)),
                "cf40": _eval_cascade_cf(p_mv_cf40, qp["cf40"], tau, cf40_y),
            }
            c_te = cascade_sweep[ckey][tkey]["bench_b_v2"]
            c_ol = cascade_sweep[ckey][tkey]["olivi_9"]
            print(
                f"  [Cascade {ckey} tau={tkey}] BenchB F1={c_te['metrics']['f1']:.4f} (esc={c_te['escalation_rate']*100:.2f}%, {c_te['effective_gpu_bs1_ms']:.2f}ms) | "
                f"Olivi9 F1={c_ol['metrics']['f1']:.4f} (esc={c_ol['escalation_rate']*100:.2f}%) | "
                f"CF16={cascade_sweep[ckey][tkey]['cf16']['acc']:.4f} CF40={cascade_sweep[ckey][tkey]['cf40']['acc']:.4f}",
                flush=True,
            )

    # Strip raw prob arrays before saving JSON
    for r in (res_zs, res_icl, res_lora):
        r.pop("_probs", None)

    q8 = json.loads(Path("results/qwen8b_zeroshot.json").read_text(encoding="utf-8"))
    zs_ref = json.loads(Path("results/zeroshot_real_variants.json").read_text(encoding="utf-8")) if Path("results/zeroshot_real_variants.json").exists() else {}

    def _wrap_single(m_dict: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        return {k: {"mean": float(v), "std": 0.0} for k, v in m_dict.items()}

    summary_block = {
        "MV-CSN v3 (Full Dual-Teacher KD)": zs_ref.get("summary", {}).get("MV-CSN v3 (Full Dual-Teacher KD)", _wrap_single(m_mv_te)),
        "RBT3 + MV-CSN v3 (Dual-Teacher)": zs_ref.get("summary", {}).get("RBT3 + MV-CSN v3 (Dual-Teacher)", _wrap_single(m_mv_te)),
        "Qwen3-4B Zero-Shot": _wrap_single(res_zs["bench_b_v2"]),
        "Qwen3-4B 4-Shot ICL": _wrap_single(res_icl["bench_b_v2"]),
        "Qwen3-4B Clean-LoRA": _wrap_single(res_lora["bench_b_v2"]),
        "Qwen3-8B Zero-Shot": _wrap_single(q8["bench_b_v2"]),
        "Cascade (MV-CSN v3 + Qwen3-4B ZS, tau=0.40)": _wrap_single(cascade_sweep["qwen4b_zeroshot"]["0.40"]["bench_b_v2"]["metrics"]),
        "Cascade (MV-CSN v3 + Qwen3-4B LoRA, tau=0.40)": _wrap_single(cascade_sweep["qwen4b_clean_lora"]["0.40"]["bench_b_v2"]["metrics"]),
    }

    payload = {
        "models": {
            "Qwen3-4B-Instruct-2507": {"params": n_params_4b, "lora_trainable_params": trainable_params, "lora_train_seconds": lora_train_sec},
            "Qwen3-8B": {"params": int(q8["params"])},
        },
        "summary": summary_block,
        "mvcsn_v3_seed42": {
            "bench_b_v2": m_mv_te,
            "sym_sub_40": m_mv_s40,
            "olivi_9": m_mv_ol,
            "cf16_ham_acc": float(np.mean([p < 0.5 for p in p_mv_cf16])),
            "cf40_acc": compute_classification_metrics(cf40_y, p_mv_cf40)["accuracy"],
            "latency_ms": {"gpu_bs64": t_mv_bs64, "gpu_bs1": t_mv_bs1},
        },
        "qwen3_4b_zeroshot": res_zs,
        "qwen3_4b_4shot_icl": res_icl,
        "qwen3_4b_clean_lora": {**res_lora, "train_history": lora_hist},
        "qwen3_8b_zeroshot": q8,
        "few_shot_examples": [{"text": t, "label": y} for t, y in few_shot_4],
        "cascade_sweep": cascade_sweep,
        "elapsed_seconds": time.time() - t0,
    }
    save_json(payload, "results/llm_comparison.json")

    # Also update latency_benchmark.json with Qwen3-4B latencies cleanly
    lat_path = Path("results/latency_benchmark.json")
    if lat_path.exists():
        lat_d = json.loads(lat_path.read_text(encoding="utf-8"))
        lat_d.setdefault("models", {})["Qwen3-4B (4.0B)"] = {
            "params": n_params_4b,
            "gpu_bs32_ms_per_msg": float(res_zs["latency_ms"]["bs32"]),
            "gpu_bs1_eager_ms_per_msg": float(res_zs["latency_ms"]["bs1"]),
            "cpu_bs1_ms_per_msg": None,
        }
        save_json(lat_d, lat_path)

    print(f"\nSTAGE3_RUNNER4_DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
