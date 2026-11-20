"""Stage 1b + 30K Unlabeled Transfer Pool Builder (Fast A100 Pipeline)."""

from __future__ import annotations

import gzip
import math
import os
import random
import re
import signal
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

from baselines.roberta.finetune import (
    RBT3Classifier,
    RBT3HybridDataset,
    RBT3MVCSNFusion,
    evaluate_rbt3_family,
    extract_backbone_features,
    make_rbt3_collator,
    train_rbt3_family,
)
from mvcsn.data.preprocessing import TextPreprocessor, normalize_canonical_punct
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.distill import build_pretrained_char_init
from mvcsn.utils.config import MVCSNConfig, MVCSN_V3_CONFIG
from mvcsn.utils.experiment import (
    COUNTERFACTUAL_16_ORIG,
    COUNTERFACTUAL_40_BALANCED,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

SMS_DATASET_URL = (
    "https://raw.githubusercontent.com/hrwhisper/SpamMessage/master/data/"
    "%E5%B8%A6%E6%A0%87%E7%AD%BE%E7%9F%AD%E4%BF%A1.txt"
)
STRIP_PUNCT_WS = re.compile(r"[\s\W_]+", re.UNICODE)


def _char_ngrams(text: str, n: int = 5) -> Set[str]:
    s = STRIP_PUNCT_WS.sub("", text.lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


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


def _logit_from_prob(p: float) -> float:
    pc = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(pc / (1.0 - pc))


def _four_punct_feats(texts: List[str]) -> np.ndarray:
    rows = []
    for t in texts:
        rows.append([
            1.0 if ("," in t or "，" in t or "、" in t) else 0.0,
            1.0 if ("." in t or "。" in t) else 0.0,
            1.0 if ("!" in t or "！" in t) else 0.0,
            1.0 if (":" in t or "：" in t) else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def _four_punct_auc(texts: List[str], labels: List[int]) -> float:
    if len(set(labels)) < 2 or len(labels) < 20:
        return 0.5
    X = _four_punct_feats(texts)
    y = np.asarray(labels, dtype=np.int64)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def _len_bin(t: str) -> int:
    n = len(t)
    if n < 25:
        return 0
    if n < 45:
        return 1
    if n < 70:
        return 2
    if n < 105:
        return 3
    return 4


def _stratum_key(t: str) -> Tuple[int, int, int]:
    return (
        _len_bin(t),
        1 if ("," in t or "，" in t or "、" in t) else 0,
        1 if ("." in t or "。" in t) else 0,
    )


def _match_balanced_pool(
    texts: List[str],
    ham_idx: List[int],
    spam_idx: List[int],
    max_pairs: int = 13000,
    max_auc: float = 0.63,
    step: int = 10,
    seed: int = 42,
) -> Tuple[List[int], float, float]:
    rng = random.Random(seed)
    auc_before = _four_punct_auc(
        [texts[i] for i in ham_idx + spam_idx],
        [0] * len(ham_idx) + [1] * len(spam_idx),
    )

    by_h: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    by_s: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in ham_idx:
        by_h[_stratum_key(texts[i])].append(i)
    for i in spam_idx:
        by_s[_stratum_key(texts[i])].append(i)

    bal_h: List[int] = []
    bal_s: List[int] = []
    used: Set[int] = set()
    for k in sorted(set(by_h.keys()) & set(by_s.keys())):
        hs, ss = by_h[k], by_s[k]
        rng.shuffle(hs)
        rng.shuffle(ss)
        m = min(len(hs), len(ss))
        bal_h.extend(hs[:m])
        bal_s.extend(ss[:m])
        used.update(hs[:m])
        used.update(ss[:m])

    if len(bal_h) > max_pairs:
        perm = list(range(len(bal_h)))
        rng.shuffle(perm)
        perm = perm[:max_pairs]
        bal_h = [bal_h[j] for j in perm]
        bal_s = [bal_s[j] for j in perm]
    else:
        X_all = _four_punct_feats([texts[i] for i in ham_idx + spam_idx])
        y_all = np.asarray([0] * len(ham_idx) + [1] * len(spam_idx), dtype=np.int64)
        prop_lr = LogisticRegression(max_iter=500, random_state=42).fit(X_all, y_all)
        rem_h = [i for i in ham_idx if i not in used]
        rem_s = [i for i in spam_idx if i not in used]
        if rem_h:
            p_h = prop_lr.predict_proba(_four_punct_feats([texts[i] for i in rem_h]))[:, 1]
            pm_h = {i: float(p) for i, p in zip(rem_h, p_h)}
            rem_h.sort(key=lambda idx_val: abs(pm_h[idx_val] - 0.5))
        if rem_s:
            p_s = prop_lr.predict_proba(_four_punct_feats([texts[i] for i in rem_s]))[:, 1]
            pm_s = {i: float(p) for i, p in zip(rem_s, p_s)}
            rem_s.sort(key=lambda idx_val: abs(pm_s[idx_val] - 0.5))

        p_ptr, s_ptr = 0, 0
        while p_ptr + step <= len(rem_h) and s_ptr + step <= len(rem_s) and len(bal_h) + step <= max_pairs:
            cand_h = bal_h + rem_h[p_ptr : p_ptr + step]
            cand_s = bal_s + rem_s[s_ptr : s_ptr + step]
            cand_auc = _four_punct_auc(
                [texts[i] for i in cand_h + cand_s],
                [0] * len(cand_h) + [1] * len(cand_s),
            )
            if cand_auc <= max_auc:
                bal_h, bal_s = cand_h, cand_s
                p_ptr += step
                s_ptr += step
            else:
                break

    auc_after = _four_punct_auc(
        [texts[i] for i in bal_h + bal_s],
        [0] * len(bal_h) + [1] * len(bal_s),
    )
    return bal_h + bal_s, auc_before, auc_after


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
    assert len(tr_clean) == 11706 and len(te_real) == 3216
    olivi_recs = _load_olivi(preproc)

    vocab_c = CharVocab.load("artifacts/vocab_clean.json")
    graphs_c = load_graphs("artifacts/graphs_clean_top20.pt")
    gmeta_c = load_glyph_meta("artifacts/glyph_meta_clean.pt")
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c, font_dir="/content/fonts")

    # --- Phase 1a: Filter restored Qwen3-8B pool (3,146 rows) using fast 35s Clean RBT3 consensus gate ---
    q_pool = torch.load("artifacts/qwen8b_unlabeled_pool.pt", map_location="cpu", weights_only=False)
    q_recs: List[SpamRecord] = q_pool["records"]
    q_probs: List[float] = q_pool["qwen_probs"].float().tolist()
    q_logits: List[float] = q_pool["qwen_logits"].float().tolist()
    q_buckets: List[str] = list(q_pool["buckets"])

    print(f"[Phase 1a] Training fast Clean RBT3 consensus filter on {len(tr_clean)} clean rows...", flush=True)
    rbt_tok = AutoTokenizer.from_pretrained("hfl/rbt3", use_fast=True)
    col_r = make_rbt3_collator(rbt_tok.pad_token_id)
    tr_r_dl = DataLoader(RBT3HybridDataset(tr_clean, vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_r)
    va_r_dl = DataLoader(RBT3HybridDataset(va_clean, vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True), batch_size=128, shuffle=False, collate_fn=col_r)
    q_r_dl = DataLoader(RBT3HybridDataset(q_recs, vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True), batch_size=128, shuffle=False, collate_fn=col_r)

    rbt_filter = RBT3Classifier("hfl/rbt3", use_proc=True)
    rf_info = train_rbt3_family(rbt_filter, tr_r_dl, va_r_dl, device, epochs=4, lr=4e-5, is_fusion=False)
    _, dom_probs, _ = evaluate_rbt3_family(rbt_filter, q_r_dl, device)
    print(f"[Phase 1a] Clean RBT3 filter trained in {rf_info['train_seconds']:.1f}s (val_f1={rf_info['best_val_f1']:.4f})", flush=True)
    del rbt_filter
    torch.cuda.empty_cache()

    ham_idx: List[int] = []
    spam_idx: List[int] = []
    promo_dropped = 0
    for i in range(len(q_recs)):
        qp = q_probs[i]
        dp = dom_probs[i]
        bkt = q_buckets[i]
        if qp < 0.15:
            if bkt == "promo" and dp >= 0.50:
                promo_dropped += 1
                continue
            ham_idx.append(i)
        elif qp > 0.85 and dp > 0.80:
            spam_idx.append(i)

    q_texts = [r.raw_text for r in q_recs]
    sel_cons, auc_b, auc_a = _match_balanced_pool(q_texts, ham_idx, spam_idx, max_pairs=2000, max_auc=0.63, step=10, seed=42)
    cons_records: List[SpamRecord] = []
    cons_probs: List[float] = []
    cons_logits: List[float] = []
    cons_buckets: List[str] = []
    for k, oi in enumerate(sel_cons):
        r = q_recs[oi]
        if q_probs[oi] < 0.15:
            ps = float(q_probs[oi])
            ls = float(q_logits[oi])
            lbl = 0
        else:
            ps = 0.5 * (float(q_probs[oi]) + float(dom_probs[oi]))
            ls = _logit_from_prob(ps)
            lbl = 1
        cons_records.append(SpamRecord(k, lbl, r.raw_text, r.proc_text, r.raw_text, 910000 + k, "consensus", "CLEAN", [0] * min(len(r.proc_text), 192)))
        cons_probs.append(ps)
        cons_logits.append(ls)
        cons_buckets.append(q_buckets[oi])

    print(
        f"[Phase 1a] Consensus Pool={len(cons_records)} (ham={sum(1 for r in cons_records if r.label==0)}, "
        f"spam={sum(1 for r in cons_records if r.label==1)}, promo_dropped={promo_dropped}) | "
        f"4-punct AUC: {auc_b:.4f} -> {auc_a:.4f}",
        flush=True,
    )
    torch.save(
        {
            "records": cons_records,
            "soft_probs": torch.tensor(cons_probs, dtype=torch.float32),
            "soft_logits": torch.tensor(cons_logits, dtype=torch.float32),
            "buckets": cons_buckets,
        },
        "artifacts/consensus_unlabeled_pool.pt",
    )
    save_json(
        {
            "promo_ham_dropped": promo_dropped,
            "final_pool_size": len(cons_records),
            "four_punct_lr_auc_before": auc_b,
            "four_punct_lr_auc_after": auc_a,
            "ham_buckets": dict(Counter(b for r, b in zip(cons_records, cons_buckets) if r.label == 0)),
        },
        "results/consensus_pool_audit.json",
    )

    # --- Phase 1b: Build 192-d Char Init & Train Unified Dual-Teacher (MacBERT + MV-CSN v3, uw=0.40) ---
    init_mat_192, init_stats = build_pretrained_char_init(
        vocab_c,
        backbone_name="hfl/chinese-macbert-base",
        hidden_dim=MVCSN_V3_CONFIG.hidden_dim,
        cache_path="artifacts/pretrained_char_init_192.pt",
        seed=42,
    )
    print("[Phase 1b] 192-d Char Init:", init_stats, flush=True)

    mac_tok = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base", use_fast=True)
    col_m = make_rbt3_collator(mac_tok.pad_token_id)

    seed_everything(42)
    t_mv_cfg = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
    u_mv = MVCSNModel.from_config(len(vocab_c), graphs_c, t_mv_cfg)
    u_mv.load_pretrained_char_emb(init_mat_192)
    unified_teacher = RBT3MVCSNFusion(u_mv, model_name="hfl/chinese-macbert-base", dropout=0.15, use_proc=True)

    tr_ds = RBT3HybridDataset(tr_clean, vocab_c, mac_tok, preproc, max_len=192, use_preprocessor=True)
    va_ds = RBT3HybridDataset(va_clean, vocab_c, mac_tok, preproc, max_len=192, use_preprocessor=True)
    unlab_ds = RBT3HybridDataset(cons_records, vocab_c, mac_tok, preproc, max_len=192, use_preprocessor=True, soft_probs=cons_probs)
    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])
    te_ds = RBT3HybridDataset(te_real, vocab_c, mac_tok, preproc, oov_map=oov_te, max_len=192, use_preprocessor=True)
    oov_ol, ext_ol = bridge_c.build_extended_bundle([r.proc_text for r in olivi_recs])
    ol_ds = RBT3HybridDataset(olivi_recs, vocab_c, mac_tok, preproc, oov_map=oov_ol, max_len=192, use_preprocessor=True)

    tr_loader_shuf = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_m)
    tr_loader_seq = DataLoader(tr_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_m)
    unlab_loader_shuf = DataLoader(unlab_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_m)
    unlab_loader_seq = DataLoader(unlab_ds, batch_size=128, shuffle=False, collate_fn=col_m)
    va_loader = DataLoader(va_ds, batch_size=128, shuffle=False, collate_fn=col_m)
    te_loader = DataLoader(te_ds, batch_size=128, shuffle=False, collate_fn=col_m)
    ol_loader = DataLoader(ol_ds, batch_size=128, shuffle=False, collate_fn=col_m)

    print("[Phase 1b] Training Unified Dual-Teacher (MacBERT + MV-CSN v3, uw=0.40)...", flush=True)
    t_info = train_rbt3_family(
        unified_teacher, tr_loader_shuf, va_loader, device, epochs=5, lr=3e-5,
        is_fusion=True, use_consistency=True, unlabeled_loader=unlab_loader_shuf, unlabeled_weight=0.40,
    )
    print(f"[Phase 1b] Unified Teacher Train: {t_info}", flush=True)

    with unified_teacher.mvcsn.temporary_inference_graph(ext_te if oov_te else None):
        m_b2, _, _ = evaluate_rbt3_family(unified_teacher, te_loader, device)
    with unified_teacher.mvcsn.temporary_inference_graph(ext_ol if oov_ol else None):
        m_ol, _, _ = evaluate_rbt3_family(unified_teacher, ol_loader, device)

    cf16_recs = [SpamRecord(i, 0, t, preproc.process(t, True) or t, t, i, "cf", "CLEAN", []) for i, t in enumerate(COUNTERFACTUAL_16_ORIG)]
    oov_16, ext_16 = bridge_c.build_extended_bundle([r.proc_text for r in cf16_recs])
    dl_16 = DataLoader(RBT3HybridDataset(cf16_recs, vocab_c, mac_tok, preproc, oov_map=oov_16, use_preprocessor=True), batch_size=16, shuffle=False, collate_fn=col_m)
    with unified_teacher.mvcsn.temporary_inference_graph(ext_16 if oov_16 else None):
        _, p_16, _ = evaluate_rbt3_family(unified_teacher, dl_16, device)
    cf16_acc = float(np.mean([p < 0.5 for p in p_16]))

    cf40_recs = [SpamRecord(i, y, t, preproc.process(t, True) or t, t, i, "cf", "CLEAN", []) for i, (y, t) in enumerate(COUNTERFACTUAL_40_BALANCED)]
    oov_40, ext_40 = bridge_c.build_extended_bundle([r.proc_text for r in cf40_recs])
    dl_40 = DataLoader(RBT3HybridDataset(cf40_recs, vocab_c, mac_tok, preproc, oov_map=oov_40, use_preprocessor=True), batch_size=40, shuffle=False, collate_fn=col_m)
    with unified_teacher.mvcsn.temporary_inference_graph(ext_40 if oov_40 else None):
        m_40, _, _ = evaluate_rbt3_family(unified_teacher, dl_40, device)

    print(
        f"[Phase 1b Eval] BenchB_v2 P={m_b2['precision']:.4f} R={m_b2['recall']:.4f} F1={m_b2['f1']:.4f} (Ham={m_b2['clean_ham_acc']:.4f}) | "
        f"Olivi-9 F1={m_ol['f1']:.4f} | CF16={cf16_acc:.4f} | CF40={m_40['accuracy']:.4f}",
        flush=True,
    )
    print(f"[Phase 1b CF16 probs] {[round(float(p), 3) for p in p_16]}", flush=True)

    unified_teacher.mvcsn.reset_inference_graph()
    tr_ext = extract_backbone_features(unified_teacher, tr_loader_seq, device)
    u_ext = extract_backbone_features(unified_teacher, unlab_loader_seq, device)
    u_blended_logits = 0.5 * u_ext["logits"] + 0.5 * torch.tensor(cons_logits, dtype=torch.float32)

    torch.save(
        {
            "tr_logits": tr_ext["logits"],
            "tr_feats": tr_ext["feats"],
            "u_logits": u_blended_logits,
            "u_teacher_logits_only": u_ext["logits"],
            "u_feats": u_ext["feats"],
        },
        "artifacts/unified_teacher_cache.pt",
    )
    torch.save(unified_teacher.state_dict(), "artifacts/unified_macbert_mvcsn_teacher.pt")
    save_json(
        {
            "model": "MacBERT-base + MV-CSN v3 (Unified Consensus Dual-Teacher)",
            "params": sum(p.numel() for p in unified_teacher.parameters()),
            "bench_b_v2": m_b2,
            "olivi_9": m_ol,
            "cf16_ham_acc": cf16_acc,
            "cf16_probs": [round(float(p), 4) for p in p_16],
            "cf40_acc": m_40["accuracy"],
        },
        "results/unified_teacher_eval.json",
    )

    # --- Phase 2: Build FULL 30K Leak-Free Unlabeled Transfer Pool (labels strictly discarded!) ---
    t_p2 = time.time()
    print("[Phase 2] Building 30K Leak-Free Unlabeled Transfer Pool from hrwhisper (labels stripped)...", flush=True)
    blocked_texts = (
        [r.clean_text for r in recs_b]
        + [r.raw_text for r in recs_b]
        + [r.raw_text for r in olivi_recs]
        + list(COUNTERFACTUAL_16_ORIG)
        + [t for _, t in COUNTERFACTUAL_40_BALANCED]
        + [r.raw_text for r in cons_records]
    )
    blocked_exact: Set[str] = set()
    blocked_index: Dict[str, List[Tuple[int, Set[str]]]] = {}
    for pt in blocked_texts:
        blocked_exact.add(STRIP_PUNCT_WS.sub("", pt.lower()))
        ng = _char_ngrams(pt, 5)
        if not ng:
            continue
        item = (len(ng), ng)
        for g in list(ng)[::7]:
            blocked_index.setdefault(g, []).append(item)

    raw_lines = urllib.request.urlopen(SMS_DATASET_URL, timeout=60).read().decode("utf-8", "ignore").splitlines()
    raw_texts: List[str] = []
    for ln in raw_lines:
        parts = ln.split("\t", 1)
        if len(parts) != 2:
            continue
        # Strictly discard parts[0] label
        t = normalize_canonical_punct(parts[1].strip())
        if 12 <= len(t) <= 160:
            raw_texts.append(t)

    rng = random.Random(42)
    rng.shuffle(raw_texts)

    target_cands = 180000
    cands: List[str] = []
    seen_norm: Set[str] = set()
    leak_removed = 0
    for txt in raw_texts:
        if len(cands) >= target_cands:
            break
        norm = STRIP_PUNCT_WS.sub("", txt.lower())
        if len(norm) < 8 or norm in seen_norm:
            continue
        if norm in blocked_exact:
            leak_removed += 1
            continue
        ng = _char_ngrams(txt, 5)
        n_ng = len(ng)
        leaked = False
        for g in ng:
            refs = blocked_index.get(g)
            if not refs:
                continue
            for n_ref, ref in refs:
                if n_ng < n_ref * 0.5 or n_ref < n_ng * 0.5:
                    continue
                inter = len(ng & ref)
                if inter / (n_ng + n_ref - inter) >= 0.50:
                    leaked = True
                    break
            if leaked:
                break
        if leaked:
            leak_removed += 1
            continue
        seen_norm.add(norm)
        cands.append(txt)

    print(
        f"[Phase 2] Collected {len(cands)} leak-free unlabeled SMS (leak_removed={leak_removed}) in {time.time()-t_p2:.1f}s. "
        "Scoring with Unified Dual-Teacher (bs=256)...",
        flush=True,
    )

    # Stream scoring in chunks of 30K to keep RAM flat and GPU saturated
    all_t_logits: List[torch.Tensor] = []
    all_t_feats_fp16: List[torch.Tensor] = []
    cand_proc = [preproc.process(t, enabled=True) or t for t in cands]
    t_sc = time.time()
    chunk_sz = 30000
    for st in range(0, len(cands), chunk_sz):
        ed = min(len(cands), st + chunk_sz)
        sub_recs = [
            SpamRecord(i, 0, cands[i], cand_proc[i], cands[i], 950000 + i, "transfer", "CLEAN", [0] * min(len(cand_proc[i]), 192))
            for i in range(st, ed)
        ]
        oov_s, ext_s = bridge_c.build_extended_bundle([r.proc_text for r in sub_recs])
        dl_s = DataLoader(
            RBT3HybridDataset(sub_recs, vocab_c, mac_tok, preproc, oov_map=oov_s, max_len=192, use_preprocessor=True),
            batch_size=256, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_m,
        )
        with unified_teacher.mvcsn.temporary_inference_graph(ext_s if oov_s else None):
            out_s = extract_backbone_features(unified_teacher, dl_s, device)
        all_t_logits.append(out_s["logits"].float())
        all_t_feats_fp16.append(out_s["feats"].half())
    t_logits = torch.cat(all_t_logits, dim=0)
    t_feats_fp16 = torch.cat(all_t_feats_fp16, dim=0)
    t_probs = torch.sigmoid(t_logits).tolist()
    print(f"[Phase 2] Scored {len(cands)} unlabeled SMS in {time.time()-t_sc:.1f}s", flush=True)

    del unified_teacher
    torch.cuda.empty_cache()

    t_ham_idx = [i for i, p in enumerate(t_probs) if p < 0.25]
    t_spam_idx = [i for i, p in enumerate(t_probs) if p > 0.75]
    print(f"[Phase 2] Confident unlabeled candidates: Ham(<0.25)={len(t_ham_idx)}, Spam(>0.75)={len(t_spam_idx)}", flush=True)

    sel_idx, tb_auc, ta_auc = _match_balanced_pool(cands, t_ham_idx, t_spam_idx, max_pairs=13000, max_auc=0.62, step=200, seed=42)
    sel_ham = sum(1 for i in sel_idx if t_probs[i] < 0.25)
    sel_spam = sum(1 for i in sel_idx if t_probs[i] > 0.75)
    print(
        f"[Phase 2] Matched broad transfer={len(sel_idx)} (ham={sel_ham}, spam={sel_spam}) | "
        f"4-punct AUC: {tb_auc:.4f} -> {ta_auc:.4f}",
        flush=True,
    )

    # Combine broad 26K transfer pool + 2x Qwen3-8B blended consensus pool (~30K total!)
    final_recs: List[SpamRecord] = []
    final_logits_list: List[torch.Tensor] = []
    final_feats_list: List[torch.Tensor] = []
    for oi in sel_idx:
        lbl = 0 if t_probs[oi] < 0.5 else 1
        final_recs.append(SpamRecord(len(final_recs), lbl, cands[oi], cand_proc[oi], cands[oi], 950000 + len(final_recs), "transfer", "CLEAN", [0] * min(len(cand_proc[oi]), 192)))
    final_logits_list.append(t_logits[sel_idx])
    final_feats_list.append(t_feats_fp16[sel_idx])

    c_feats_fp16 = u_ext["feats"].half()
    for _rep in range(2):
        for r in cons_records:
            final_recs.append(SpamRecord(len(final_recs), r.label, r.raw_text, r.proc_text, r.raw_text, 910000 + len(final_recs), "consensus", "CLEAN", [0] * min(len(r.proc_text), 192)))
        final_logits_list.append(u_blended_logits.float())
        final_feats_list.append(c_feats_fp16)

    all_logits = torch.cat(final_logits_list, dim=0)
    all_probs = torch.sigmoid(all_logits)
    all_feats = torch.cat(final_feats_list, dim=0)

    comb_auc = _four_punct_auc([r.raw_text for r in final_recs], [r.label for r in final_recs])
    print(
        f"[Phase 2] FINAL 30K Pool={len(final_recs)} (broad={len(sel_idx)} + 2x consensus={2*len(cons_records)}) | "
        f"combined 4-punct AUC={comb_auc:.4f} | total_sec={time.time()-t0:.1f}s",
        flush=True,
    )
    assert comb_auc <= 0.65, f"TRANSFER_POOL_PUNCT_LEAK: auc={comb_auc:.4f}"

    torch.save(
        {
            "records": final_recs,
            "soft_logits": all_logits,
            "soft_probs": all_probs,
            "feats": all_feats,
        },
        "artifacts/unified_transfer_pool_30k.pt",
    )
    save_json(
        {
            "candidates_scored": len(cands),
            "leak_removed": leak_removed,
            "teacher_confident_ham": len(t_ham_idx),
            "teacher_confident_spam": len(t_spam_idx),
            "broad_matched_size": len(sel_idx),
            "broad_ham": sel_ham,
            "broad_spam": sel_spam,
            "broad_four_punct_auc_before": tb_auc,
            "broad_four_punct_auc_after": ta_auc,
            "consensus_added_2x": 2 * len(cons_records),
            "final_pool_size": len(final_recs),
            "combined_four_punct_auc": comb_auc,
            "elapsed_seconds": time.time() - t0,
        },
        "results/transfer_pool_audit.json",
    )
    print("UNIFIED_AND_30K_POOL_DONE", flush=True)


if __name__ == "__main__":
    main()
