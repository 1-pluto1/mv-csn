"""Build 26K True Dual-Teacher Transfer Pool: Naturally-Punctuated Qwen3-8B+MacBERT Consensus + Broad Transfer."""

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
from collections import defaultdict
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

from baselines.llm.qwen_eval import load_qwen_model, score_binary_llm_with_logits
from baselines.roberta.finetune import (
    RBT3HybridDataset,
    RBT3MVCSNFusion,
    extract_backbone_features,
    make_rbt3_collator,
)
from mvcsn.data.preprocessing import TextPreprocessor, normalize_canonical_punct
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.utils.config import MVCSNConfig, MVCSN_V3_CONFIG
from mvcsn.utils.experiment import (
    COUNTERFACTUAL_16_ORIG,
    COUNTERFACTUAL_40_BALANCED,
    save_json,
)
from mvcsn.utils.seed import get_device, seed_everything

SMS_DATASET_URL = (
    "https://raw.githubusercontent.com/hrwhisper/SpamMessage/master/data/"
    "%E5%B8%A6%E6%A0%87%E7%AD%BE%E7%9F%AD%E4%BF%A1.txt"
)
STRIP_PUNCT_WS = re.compile(r"[\s\W_]+", re.UNICODE)
PUNCT_RE = re.compile(r"[，。！？：,.!?:]")
PROMO_AD_RE = re.compile(r"特惠|促销|打折|折扣|免费领|大促|抢购|优惠券|代金券|充值送|满\d+减|开业酬宾|折起|低至\d折")


def _char_ngrams(text: str, n: int = 5) -> Set[str]:
    s = STRIP_PUNCT_WS.sub("", text.lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _load_olivi_texts(path: str = "data/resources/external_olivi_spam.tsv.gz") -> List[str]:
    out: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out.append(p[1])
    return out


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
    if n < 40:
        return 1
    if n < 65:
        return 2
    if n < 100:
        return 3
    return 4


def _stratum_key(t: str) -> Tuple[int, int, int]:
    return (
        _len_bin(t),
        1 if ("," in t or "，" in t or "、" in t) else 0,
        1 if ("." in t or "。" in t or "!" in t or "！" in t) else 0,
    )


def _match_balanced(
    texts: List[str],
    ham_idx: List[int],
    spam_idx: List[int],
    max_pairs: int = 5000,
    max_auc: float = 0.62,
    step: int = 50,
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
    assert sum(1 for r in recs_b if r.split == "test") == 3216
    olivi_texts = _load_olivi_texts()
    prev_pool = torch.load("artifacts/unified_transfer_pool_30k.pt", map_location="cpu", weights_only=False)

    vocab_c = CharVocab.load("artifacts/vocab_clean.json")
    graphs_c = load_graphs("artifacts/graphs_clean_top20.pt")
    gmeta_c = load_glyph_meta("artifacts/glyph_meta_clean.pt")
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c, font_dir="/content/fonts")

    # 1. Collect 14,000 leak-free NATURALLY PUNCTUATED unlabeled SMS (7K short 12..40 + 7K medium 41..120)
    print("[DualPool] Collecting 14,000 leak-free naturally-punctuated unlabeled SMS from hrwhisper (labels stripped)...", flush=True)
    blocked_texts = (
        [r.clean_text for r in recs_b]
        + [r.raw_text for r in recs_b]
        + olivi_texts
        + list(COUNTERFACTUAL_16_ORIG)
        + [t for _, t in COUNTERFACTUAL_40_BALANCED]
        + [r.raw_text for r in prev_pool["records"]]
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
    short_raw: List[str] = []
    med_raw: List[str] = []
    for ln in raw_lines:
        parts = ln.split("\t", 1)
        if len(parts) != 2:
            continue
        # Strictly discard parts[0] label
        t = normalize_canonical_punct(parts[1].strip())
        if not PUNCT_RE.search(t):
            continue
        if 12 <= len(t) <= 40:
            short_raw.append(t)
        elif 41 <= len(t) <= 120:
            med_raw.append(t)

    rng = random.Random(42)
    rng.shuffle(short_raw)
    rng.shuffle(med_raw)

    seen_norm: Set[str] = set()
    leak_removed = 0

    def _filter_bucket(src_list: List[str], limit: int) -> List[str]:
        nonlocal leak_removed
        out: List[str] = []
        for txt in src_list:
            if len(out) >= limit:
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
            out.append(txt)
        return out

    p_short = _filter_bucket(short_raw, 7000)
    p_med = _filter_bucket(med_raw, 7000)
    p_texts = p_short + p_med
    p_proc = [preproc.process(t, enabled=True) or t for t in p_texts]
    print(f"[DualPool] Kept {len(p_texts)} punctuated unlabeled SMS (short={len(p_short)}, med={len(p_med)}, leak_removed={leak_removed})", flush=True)

    # 2. Score with Unified Teacher (MacBERT + MV-CSN v3, ~6s) for 768d features & domain logits
    mac_tok = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base", use_fast=True)
    col_m = make_rbt3_collator(mac_tok.pad_token_id)
    t_mv_cfg = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
    u_mv = MVCSNModel.from_config(len(vocab_c), graphs_c, t_mv_cfg)
    teacher = RBT3MVCSNFusion(u_mv, model_name="hfl/chinese-macbert-base", dropout=0.15, use_proc=True)
    teacher.load_state_dict(torch.load("artifacts/unified_macbert_mvcsn_teacher.pt", map_location="cpu", weights_only=True), strict=True)
    teacher = teacher.to(device).eval()

    p_recs = [
        SpamRecord(i, 0, p_texts[i], p_proc[i], p_texts[i], 970000 + i, "punct_consensus", "CLEAN", [0] * min(len(p_proc[i]), 192))
        for i in range(len(p_texts))
    ]
    oov_p, ext_p = bridge_c.build_extended_bundle(p_proc)
    p_loader = DataLoader(
        RBT3HybridDataset(p_recs, vocab_c, mac_tok, preproc, oov_map=oov_p, max_len=192, use_preprocessor=True),
        batch_size=256, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_m,
    )
    with teacher.mvcsn.temporary_inference_graph(ext_p if oov_p else None):
        u_out = extract_backbone_features(teacher, p_loader, device)
    u_logits = u_out["logits"].float()
    u_probs = torch.sigmoid(u_logits).tolist()
    u_feats_fp16 = u_out["feats"].half()
    del teacher
    torch.cuda.empty_cache()
    print(f"[DualPool] Unified Teacher scored {len(p_texts)} punctuated rows in {time.time()-t0:.1f}s. Scoring forward-only with Qwen3-8B...", flush=True)

    # 3. Score forward-only with Qwen/Qwen3-8B (~2.5 min on A100-40GB, bs=64)
    t_q = time.time()
    m8b, tok8b = load_qwen_model("Qwen/Qwen3-8B", device=device)
    q_probs, q_logits, _ = score_binary_llm_with_logits(m8b, tok8b, p_proc, batch_size=64, device=device)
    del m8b, tok8b
    torch.cuda.empty_cache()
    print(f"[DualPool] Qwen3-8B scored {len(p_texts)} punctuated rows in {time.time()-t_q:.1f}s", flush=True)

    # 4. High-Purity Dual-Teacher Consensus Filtering on Naturally Punctuated SMS
    ham_idx: List[int] = []
    spam_idx: List[int] = []
    promo_dropped = 0
    for i in range(len(p_texts)):
        qp = q_probs[i]
        up = u_probs[i]
        if qp < 0.10:
            if up > 0.85 and PROMO_AD_RE.search(p_texts[i]):
                promo_dropped += 1
                continue
            ham_idx.append(i)
        elif qp > 0.85 and up > 0.80:
            spam_idx.append(i)

    print(f"[DualPool] Punctuated Consensus Raw: HighPurityHam={len(ham_idx)} (promo_dropped={promo_dropped}), ConsensusSpam={len(spam_idx)}", flush=True)
    sel_p, p_auc_b, p_auc_a = _match_balanced(p_texts, ham_idx, spam_idx, max_pairs=5000, max_auc=0.62, step=50, seed=42)
    sel_p_ham = [i for i in sel_p if q_probs[i] < 0.10]
    sel_p_spam = [i for i in sel_p if q_probs[i] > 0.85]
    short_p_ham = sum(1 for i in sel_p_ham if len(p_texts[i]) <= 35)
    short_p_spam = sum(1 for i in sel_p_spam if len(p_texts[i]) <= 35)
    print(
        f"[DualPool] Punctuated Consensus Matched={len(sel_p)} (ham={len(sel_p_ham)}, spam={len(sel_p_spam)}, "
        f"short<=35 ham={short_p_ham} vs spam={short_p_spam}) | 4-punct AUC: {p_auc_b:.4f} -> {p_auc_a:.4f}",
        flush=True,
    )

    # Build Dual-Teacher calibrated targets for punctuated consensus rows
    final_recs: List[SpamRecord] = []
    final_logits_list: List[torch.Tensor] = []
    final_feats_list: List[torch.Tensor] = []

    p_target_logits: List[float] = []
    for oi in sel_p:
        if q_probs[oi] < 0.10:
            # Trust Qwen3-8B's 16/16 everyday-notification semantics strongly (0.75 Qwen + 0.25 Unified Teacher)
            tl = 0.75 * float(q_logits[oi]) + 0.25 * float(u_logits[oi].item())
            lbl = 0
        else:
            tl = 0.50 * float(q_logits[oi]) + 0.50 * float(u_logits[oi].item())
            lbl = 1
        final_recs.append(SpamRecord(len(final_recs), lbl, p_texts[oi], p_proc[oi], p_texts[oi], 970000 + len(final_recs), "punct_consensus", "CLEAN", [0] * min(len(p_proc[oi]), 192)))
        p_target_logits.append(tl)
    final_logits_list.append(torch.tensor(p_target_logits, dtype=torch.float32))
    final_feats_list.append(u_feats_fp16[sel_p])

    # 5. Merge 16,000 balanced broad transfer rows (8K Ham + 8K Spam) + 1x original consensus rows from prev_pool
    broad_h = [i for i, r in enumerate(prev_pool["records"]) if r.split == "transfer" and r.label == 0]
    broad_s = [i for i, r in enumerate(prev_pool["records"]) if r.split == "transfer" and r.label == 1]
    rng.shuffle(broad_h)
    rng.shuffle(broad_s)
    m_b = min(8000, len(broad_h), len(broad_s))
    broad_sel = broad_h[:m_b] + broad_s[:m_b]
    cons_orig = [i for i, r in enumerate(prev_pool["records"]) if r.split == "consensus"][:1914]
    keep_prev = broad_sel + cons_orig

    for oi in keep_prev:
        r = prev_pool["records"][oi]
        final_recs.append(SpamRecord(len(final_recs), r.label, r.raw_text, r.proc_text, r.raw_text, r.cluster_id, r.split, "CLEAN", [0] * min(len(r.proc_text), 192)))
    final_logits_list.append(prev_pool["soft_logits"][keep_prev].float())
    final_feats_list.append(prev_pool["feats"][keep_prev].half())

    all_logits = torch.cat(final_logits_list, dim=0)
    all_probs = torch.sigmoid(all_logits)
    all_feats = torch.cat(final_feats_list, dim=0)
    comb_auc = _four_punct_auc([r.raw_text for r in final_recs], [r.label for r in final_recs])

    print(
        f"[DualPool] FINAL Dual-Teacher Pool={len(final_recs)} (punct_consensus={len(sel_p)} + broad={len(broad_sel)} + orig_cons={len(cons_orig)}) | "
        f"combined 4-punct AUC={comb_auc:.4f} | elapsed={time.time()-t0:.1f}s",
        flush=True,
    )
    assert comb_auc <= 0.65, f"DUAL_POOL_PUNCT_LEAK: auc={comb_auc:.4f}"

    torch.save(
        {
            "records": final_recs,
            "soft_logits": all_logits,
            "soft_probs": all_probs,
            "feats": all_feats,
        },
        "artifacts/dual_teacher_transfer_pool_30k.pt",
    )
    save_json(
        {
            "punct_candidates_scored": len(p_texts),
            "punct_leak_removed": leak_removed,
            "punct_promo_dropped": promo_dropped,
            "punct_consensus_size": len(sel_p),
            "punct_short35_ham": short_p_ham,
            "punct_short35_spam": short_p_spam,
            "punct_auc_before": p_auc_b,
            "punct_auc_after": p_auc_a,
            "broad_size": len(broad_sel),
            "orig_consensus_size": len(cons_orig),
            "final_pool_size": len(final_recs),
            "combined_four_punct_auc": comb_auc,
            "elapsed_seconds": time.time() - t0,
        },
        "results/dual_teacher_pool_audit.json",
    )
    print("DUAL_TEACHER_POOL_DONE", flush=True)


if __name__ == "__main__":
    main()
