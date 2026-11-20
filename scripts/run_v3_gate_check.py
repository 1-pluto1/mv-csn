"""Stage 2 (v4): Seed-42 Go/No-Go Gate Check using 30K Unified Transfer Pool & Fixed UNK Gate."""

from __future__ import annotations

import copy
import gzip
import os
import re
import signal
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

from baselines.roberta.finetune import (
    RBT3Classifier,
    RBT3HybridDataset,
    RBT3MVCSNFusion,
    evaluate_rbt3_family,
    make_rbt3_collator,
    train_rbt3_family,
)
from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor, is_cjk
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.distill import (
    DistillDataset,
    collate_distill_batch,
    train_mvcsn_distilled,
)
from mvcsn.training.trainer import evaluate_mvcsn
from mvcsn.utils.config import DistillConfig, MVCSNConfig, MVCSN_V3_CONFIG
from mvcsn.utils.experiment import (
    COUNTERFACTUAL_16_ORIG,
    COUNTERFACTUAL_40_BALANCED,
    build_symmetric_stress_perturber,
    perturb_records_symmetrically,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

RARE_GLYPH_CHARS = set("髙穏茺芣箌伱妳佰萬賺賠頂專贈淸徴錢貸網領賬號冊訂")


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


def _make_records(texts: List[str], labels: List[int], preproc: TextPreprocessor, use_proc: bool = True) -> List[SpamRecord]:
    out: List[SpamRecord] = []
    for i, (t, y) in enumerate(zip(texts, labels)):
        pt = (preproc.process(t, enabled=True) or t) if use_proc else t
        out.append(SpamRecord(i, int(y), t, pt, t, i, "probe", "PROBE", []))
    return out


@torch.no_grad()
def eval_mvcsn_texts(
    model: MVCSNModel,
    vocab: CharVocab,
    bridge: OOVGraphBridge,
    preproc: TextPreprocessor,
    texts: List[str],
    labels: List[int],
    device: torch.device,
    use_oov: bool = True,
    use_proc: bool = True,
) -> Tuple[Dict[str, float], List[float]]:
    recs = _make_records(texts, labels, preproc, use_proc=use_proc)
    if use_oov:
        oov_map, ext_b = bridge.build_extended_bundle([r.proc_text for r in recs])
    else:
        oov_map, ext_b = None, None
    ds = SpamDataset(recs, vocab, use_preprocessor=use_proc, oov_map=oov_map)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
    with model.temporary_inference_graph(ext_b if oov_map else None):
        m, p, _ = evaluate_mvcsn(model, loader, device)
    return m, p


@torch.no_grad()
def run_full_token_gate_analysis(
    model: MVCSNModel,
    te_real: List[SpamRecord],
    vocab: CharVocab,
    oov_te: Dict[str, int],
    ext_te: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, object]:
    ds = SpamDataset(te_real, vocab, use_preprocessor=True, oov_map=oov_te)
    loader = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
    model.eval()

    sums = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    cnts = defaultdict(int)
    with model.temporary_inference_graph(ext_te if oov_te else None):
        offset = 0
        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attn_mask"].to(device, non_blocking=True)
            out = model(ids, mask)
            alphas = out["alpha"].cpu().numpy()
            bsz = ids.size(0)
            for j in range(bsz):
                rec = te_real[offset + j]
                L = int(mask[j].sum().item())
                txt = rec.proc_text[:L]
                for pos, ch in enumerate(txt):
                    a = alphas[j, pos]
                    in_v = ch in vocab.stoi
                    is_obf = (pos < len(rec.obf_mask)) and (rec.obf_mask[pos] == 1)
                    if not in_v:
                        key = "obf_oov_bridged" if ch in oov_te else "oov_unbridged"
                    elif not is_obf:
                        key = "clean_in_vocab"
                    else:
                        is_glyph = (
                            rec.attack_type == "GLYPH"
                            or (not is_cjk(ch))
                            or (ch in RARE_GLYPH_CHARS)
                            or (0x3040 <= ord(ch) <= 0x30FF)
                        )
                        key = "obf_in_vocab_glyph" if is_glyph else "obf_in_vocab_phon"
                    sums[key] += a
                    cnts[key] += 1
            offset += bsz

    means = {k: (sums[k] / max(1, cnts[k])).round(4).tolist() for k in cnts}
    return {"counts": dict(cnts), "means": means}


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
    assert len(te_real) == 3216
    olivi_recs = _load_olivi(preproc)

    vocab_c = CharVocab.load("artifacts/vocab_clean.json")
    graphs_c = load_graphs("artifacts/graphs_clean_top20.pt")
    gmeta_c = load_glyph_meta("artifacts/glyph_meta_clean.pt")
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c, font_dir="/content/fonts")

    init_mat = torch.load("artifacts/pretrained_char_init_192.pt", map_location="cpu", weights_only=False)["init_matrix"]
    clean_cache = torch.load("artifacts/teacher_cache.pt", map_location="cpu", weights_only=False)
    uni_cache = torch.load("artifacts/unified_teacher_cache.pt", map_location="cpu", weights_only=False)
    u_pool = torch.load("artifacts/dual_teacher_transfer_pool_30k.pt", map_location="cpu", weights_only=False)
    print(f"[GateCheck] Loaded 30K transfer pool: size={len(u_pool['records'])}", flush=True)

    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])
    oov_ol, ext_ol = bridge_c.build_extended_bundle([r.proc_text for r in olivi_recs])
    te_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in te_real]
    oov_raw, ext_raw = bridge_c.build_extended_bundle([r.raw_text for r in te_raw_recs])

    tr_dist_ds = DistillDataset(tr_clean, vocab_c)
    unlab_ds = DistillDataset(u_pool["records"], vocab_c)
    va_mv_loader = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
    te_fair_loader = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
    te_raw_loader = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
    te_ol_loader = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True, oov_map=oov_ol), batch_size=128, shuffle=False, collate_fn=collate_spam_batch)

    tr_dist_loader = DataLoader(tr_dist_ds, batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    unlab_loader = DataLoader(unlab_ds, batch_size=192, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)

    rbt_tok = AutoTokenizer.from_pretrained("hfl/rbt3", use_fast=True)
    col_r = make_rbt3_collator(rbt_tok.pad_token_id)
    tr_rl_ds = RBT3HybridDataset(tr_clean, vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True)
    va_rl_ds = RBT3HybridDataset(va_clean, vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True)
    te_rl_fair_ds = RBT3HybridDataset(te_real, vocab_c, rbt_tok, preproc, oov_map=oov_te, max_len=192, use_preprocessor=True)
    te_rl_raw_ds = RBT3HybridDataset(te_raw_recs, vocab_c, rbt_tok, preproc, oov_map=oov_raw, max_len=192, use_preprocessor=False)
    ol_rl_ds = RBT3HybridDataset(olivi_recs, vocab_c, rbt_tok, preproc, oov_map=oov_ol, max_len=192, use_preprocessor=True)

    tr_rl_loader = DataLoader(tr_rl_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_r)
    va_rl_loader = DataLoader(va_rl_ds, batch_size=128, shuffle=False, collate_fn=col_r)
    te_rl_fair_loader = DataLoader(te_rl_fair_ds, batch_size=128, shuffle=False, collate_fn=col_r)
    te_rl_raw_loader = DataLoader(te_rl_raw_ds, batch_size=128, shuffle=False, collate_fn=col_r)
    ol_rl_loader = DataLoader(ol_rl_ds, batch_size=128, shuffle=False, collate_fn=col_r)

    stress_fn = build_symmetric_stress_perturber(vocab_c)
    tr_aug_recs = perturb_records_symmetrically(tr_clean, stress_fn, preproc, rate=0.18, seed=42, mode="mixed")
    tr_rl_aug_loader = DataLoader(
        RBT3HybridDataset(tr_aug_recs, vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True),
        batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_r,
    )
    unlab_rl_loader = DataLoader(
        RBT3HybridDataset(
            u_pool["records"], vocab_c, rbt_tok, preproc, max_len=192, use_preprocessor=True,
            soft_probs=u_pool["soft_probs"].tolist(),
        ),
        batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_r,
    )

    cf16_texts = list(COUNTERFACTUAL_16_ORIG)
    cf40_texts = [t for _, t in COUNTERFACTUAL_40_BALANCED]
    cf40_y = [lbl for lbl, _ in COUNTERFACTUAL_40_BALANCED]
    ham_te = [r for r in te_real if r.label == 0]
    spam_te = [r for r in te_real if r.label == 1]
    ham_plus_texts = [r.raw_text.rstrip(".,!?") + ",." for r in ham_te]
    spam_strip_texts = [re.sub(r"[,!:.，！。：]", "", r.raw_text) for r in spam_te]

    def _eval_rbt_probes(m, is_fus: bool) -> Dict[str, float]:
        cf16_recs = _make_records(cf16_texts, [0] * len(cf16_texts), preproc, True)
        cf40_recs = _make_records(cf40_texts, cf40_y, preproc, True)
        hp_recs = _make_records(ham_plus_texts, [0] * len(ham_plus_texts), preproc, True)
        ss_recs = _make_records(spam_strip_texts, [1] * len(spam_strip_texts), preproc, True)

        def _score_r(recs):
            o_m, ex = bridge_c.build_extended_bundle([r.proc_text for r in recs]) if is_fus else (None, None)
            dl = DataLoader(RBT3HybridDataset(recs, vocab_c, rbt_tok, preproc, oov_map=o_m, use_preprocessor=True), batch_size=128, shuffle=False, collate_fn=col_r)
            if is_fus:
                with m.mvcsn.temporary_inference_graph(ex if o_m else None):
                    met, pr, _ = evaluate_rbt3_family(m, dl, device)
            else:
                met, pr, _ = evaluate_rbt3_family(m, dl, device)
            return met, pr

        _, p16 = _score_r(cf16_recs)
        m40, _ = _score_r(cf40_recs)
        mhp, _ = _score_r(hp_recs)
        mss, _ = _score_r(ss_recs)
        return {
            "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
            "balanced_40_acc": m40["accuracy"],
            "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
            "bench_spam_stripped_punct_recall": mss["spam_recall"],
        }

    results: Dict[str, Dict[str, object]] = {}

    def _run_mvcsn_variant(
        name: str,
        cfg: MVCSNConfig,
        d_cfg: DistillConfig,
        t_log: torch.Tensor,
        t_feat: torch.Tensor,
        use_u: bool,
        use_oov: bool = True,
    ):
        seed_everything(42)
        print(f"\n[GateCheck] Training {name} ...", flush=True)
        m = MVCSNModel.from_config(len(vocab_c), graphs_c, cfg)
        m.load_pretrained_char_emb(init_mat)
        info = train_mvcsn_distilled(
            m, tr_dist_loader, va_mv_loader, device, vocab_c, config=d_cfg,
            unlabeled_loader=unlab_loader if use_u else None,
            teacher_logits=t_log, teacher_feats=t_feat,
            unlabeled_logits=u_pool["soft_logits"] if use_u else None,
            unlabeled_feats=u_pool["feats"] if use_u else None,
        )
        with m.temporary_inference_graph(ext_te if (use_oov and oov_te) else None):
            m_fair, _, _ = evaluate_mvcsn(m, te_fair_loader, device)
        with m.temporary_inference_graph(ext_raw if (use_oov and oov_raw) else None):
            m_raw, _, _ = evaluate_mvcsn(m, te_raw_loader, device)
        with m.temporary_inference_graph(ext_ol if (use_oov and oov_ol) else None):
            m_ol, _, _ = evaluate_mvcsn(m, te_ol_loader, device)

        _, p16 = eval_mvcsn_texts(m, vocab_c, bridge_c, preproc, cf16_texts, [0] * 16, device, use_oov=use_oov)
        m40, _ = eval_mvcsn_texts(m, vocab_c, bridge_c, preproc, cf40_texts, cf40_y, device, use_oov=use_oov)
        mhp, _ = eval_mvcsn_texts(m, vocab_c, bridge_c, preproc, ham_plus_texts, [0] * len(ham_te), device, use_oov=use_oov)
        mss, _ = eval_mvcsn_texts(m, vocab_c, bridge_c, preproc, spam_strip_texts, [1] * len(spam_te), device, use_oov=use_oov)

        res = {
            "params_infer": sum(p.numel() for n, p in m.named_parameters() if not n.startswith("kd_proj.")),
            "train_info": info,
            "fair": m_fair,
            "raw": m_raw,
            "olivi9": m_ol,
            "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
            "orig_16_probs": [round(float(p), 4) for p in p16],
            "balanced_40_acc": m40["accuracy"],
            "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
            "bench_spam_stripped_punct_recall": mss["spam_recall"],
        }
        results[name] = res
        print(
            f"  -> {name} ({res['params_infer']:,d}p, ep={info['epochs_run']}, {info['train_seconds']:.1f}s): "
            f"FairF1={m_fair['f1']:.4f} (Ham={m_fair['clean_ham_acc']:.4f}) RawF1={m_raw['f1']:.4f} "
            f"Olivi9={m_ol['f1']:.4f} (P={m_ol['precision']:.3f} R={m_ol['recall']:.3f} AUC={m_ol['auc']:.3f}) "
            f"CF16={res['orig_16_ham_acc']:.4f} CF40={res['balanced_40_acc']:.4f}",
            flush=True,
        )
        return m, res

    # 1. MV-CSN v3 (Clean-KD Only)
    _run_mvcsn_variant(
        "MV-CSN v3 (Clean-KD Only)",
        MVCSN_V3_CONFIG,
        DistillConfig(use_unlabeled_kd=False),
        clean_cache["logits"], clean_cache["feats"], use_u=False,
    )

    # 2. MV-CSN v3 (Full Dual-Teacher KD)
    m_v3_full, r_v3_full = _run_mvcsn_variant(
        "MV-CSN v3 (Full Dual-Teacher KD)",
        MVCSN_V3_CONFIG,
        DistillConfig(use_unlabeled_kd=True),
        uni_cache["tr_logits"], uni_cache["tr_feats"], use_u=True,
    )
    m_v3_full.reset_inference_graph()
    torch.save(m_v3_full.state_dict(), "artifacts/mvcsn_v3_seed42.pt")

    # 3. Plain CNN+Trans (Same Full 30K Dual-Teacher KD)
    plain_cfg = MVCSNConfig(
        hidden_dim=192, num_heads=6, gat_heads=4, num_transformer_layers=3, transformer_ff_dim=384,
        dropout=0.15, use_phonetic=False, use_glyph=False, use_gate=False, use_loc_head=True,
        use_multimodal_anchor=False, use_dilated_conv=True, use_fused_gat=False, teacher_feat_dim=768,
    )
    _, r_plain = _run_mvcsn_variant(
        "Plain CNN+Trans (Same KD)",
        plain_cfg,
        DistillConfig(use_unlabeled_kd=True, unk_prob=1.0),
        uni_cache["tr_logits"], uni_cache["tr_feats"], use_u=True, use_oov=False,
    )

    # 4. RBT3 (Clean)
    seed_everything(42)
    print("\n[GateCheck] Training RBT3 (Clean) ...", flush=True)
    rbt_clean = RBT3Classifier("hfl/rbt3", use_proc=True)
    train_rbt3_family(rbt_clean, tr_rl_loader, va_rl_loader, device, epochs=4, lr=4e-5, is_fusion=False)
    mr_fair, _, _ = evaluate_rbt3_family(rbt_clean, te_rl_fair_loader, device)
    rbt_clean_raw = copy.deepcopy(rbt_clean)
    rbt_clean_raw.use_proc = False
    mr_raw, _, _ = evaluate_rbt3_family(rbt_clean_raw, te_rl_raw_loader, device)
    mr_ol, _, _ = evaluate_rbt3_family(rbt_clean, ol_rl_loader, device)
    results["RBT3 (Clean)"] = {
        "params": sum(p.numel() for p in rbt_clean.parameters()),
        "fair": mr_fair, "raw": mr_raw, "olivi9": mr_ol,
        **_eval_rbt_probes(rbt_clean, is_fus=False),
    }

    # 5. RBT3 (Same Data+Aug — identical 30K transfer pool!)
    seed_everything(42)
    print("\n[GateCheck] Training RBT3 (Same Data+Aug) ...", flush=True)
    rbt_aug = RBT3Classifier("hfl/rbt3", use_proc=True)
    train_rbt3_family(
        rbt_aug, tr_rl_loader, va_rl_loader, device, epochs=4, lr=4e-5, is_fusion=False,
        unlabeled_loader=unlab_rl_loader, unlabeled_weight=0.45, aug_train_loader=tr_rl_aug_loader,
    )
    mra_fair, _, _ = evaluate_rbt3_family(rbt_aug, te_rl_fair_loader, device)
    rbt_aug_raw = copy.deepcopy(rbt_aug)
    rbt_aug_raw.use_proc = False
    mra_raw, _, _ = evaluate_rbt3_family(rbt_aug_raw, te_rl_raw_loader, device)
    mra_ol, _, _ = evaluate_rbt3_family(rbt_aug, ol_rl_loader, device)
    results["RBT3 (Same Data+Aug)"] = {
        "params": sum(p.numel() for p in rbt_aug.parameters()),
        "fair": mra_fair, "raw": mra_raw, "olivi9": mra_ol,
        **_eval_rbt_probes(rbt_aug, is_fus=False),
    }

    # 6. RBT3 + MV-CSN v3 (Full Dual-Teacher — identical 30K transfer pool!)
    seed_everything(42)
    print("\n[GateCheck] Training RBT3 + MV-CSN v3 (Dual-Teacher) ...", flush=True)
    fus_cfg = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
    mv_sub = MVCSNModel.from_config(len(vocab_c), graphs_c, fus_cfg)
    mv_sub.load_pretrained_char_emb(init_mat)
    fus = RBT3MVCSNFusion(mv_sub, "hfl/rbt3", dropout=0.15, use_proc=True)
    train_rbt3_family(
        fus, tr_rl_loader, va_rl_loader, device, epochs=4, lr=4e-5, is_fusion=True, use_consistency=True,
        unlabeled_loader=unlab_rl_loader, unlabeled_weight=0.45, aug_train_loader=tr_rl_aug_loader,
    )
    with fus.mvcsn.temporary_inference_graph(ext_te if oov_te else None):
        mf_fair, _, _ = evaluate_rbt3_family(fus, te_rl_fair_loader, device)
    fus_raw = copy.deepcopy(fus)
    fus_raw.use_proc = False
    with fus_raw.mvcsn.temporary_inference_graph(ext_raw if oov_raw else None):
        mf_raw, _, _ = evaluate_rbt3_family(fus_raw, te_rl_raw_loader, device)
    with fus.mvcsn.temporary_inference_graph(ext_ol if oov_ol else None):
        mf_ol, _, _ = evaluate_rbt3_family(fus, ol_rl_loader, device)
    results["RBT3 + MV-CSN v3 (Dual-Teacher)"] = {
        "params": sum(p.numel() for p in fus.parameters()),
        "fair": mf_fair, "raw": mf_raw, "olivi9": mf_ol,
        **_eval_rbt_probes(fus, is_fus=True),
    }

    gate_stats = run_full_token_gate_analysis(m_v3_full, te_real, vocab_c, oov_te, ext_te, device)
    g_means = gate_stats["means"]

    gate1_ok = (
        r_v3_full["orig_16_ham_acc"] >= 0.875
        and r_v3_full["balanced_40_acc"] >= 0.85
        and r_v3_full["bench_ham_plus_comma_period_acc"] >= 0.95
    )
    gate2_ok = r_v3_full["fair"]["f1"] >= 0.993 and r_v3_full["raw"]["f1"] >= 0.92
    beats_plain_bench = r_v3_full["fair"]["f1"] >= r_plain["fair"]["f1"]
    beats_plain_olivi = r_v3_full["olivi9"]["f1"] > r_plain["olivi9"]["f1"]
    gate3_ok = r_v3_full["olivi9"]["f1"] >= 0.77 and (beats_plain_bench or beats_plain_olivi)

    c_ctx = g_means.get("clean_in_vocab", [0, 0, 0])[0]
    p_gr = sum(g_means.get("obf_in_vocab_phon", [0, 0, 0])[1:])
    oov_gr = sum(g_means.get("obf_oov_bridged", [0, 0, 0])[1:])
    gate4_ok = c_ctx >= 0.85 and p_gr >= 0.65 and oov_gr >= 0.95

    all_pass = bool(gate1_ok and gate2_ok and gate3_ok and gate4_ok)
    gates_report = {
        "gate1_everyday_notif_fix": gate1_ok,
        "gate2_single_domain_and_raw": gate2_ok,
        "gate3_olivi9_and_beats_plain": gate3_ok,
        "beats_plain_bench_b": beats_plain_bench,
        "beats_plain_olivi9": beats_plain_olivi,
        "gate4_gate_interpretability": gate4_ok,
        "all_pass": all_pass,
        "gate_details": {"c_ctx": c_ctx, "phon_graph": p_gr, "oov_bridged_graph": oov_gr},
    }

    save_json({"results": results, "gate_analysis": gate_stats, "gates": gates_report, "elapsed_seconds": time.time() - t0}, "results/stage2_gate_check.json")

    print("\n" + "=" * 88, flush=True)
    print("STAGE 2 (30K UNIFIED TRANSFER + FIXED UNK GATE) SEED-42 GATE SUMMARY", flush=True)
    print("=" * 88, flush=True)
    for k, v in results.items():
        print(
            f"  {k:34s} | FairF1={v['fair']['f1']:.4f} | RawF1={v['raw']['f1']:.4f} | "
            f"Olivi9={v['olivi9']['f1']:.4f} | CF16={v['orig_16_ham_acc']:.4f} | CF40={v['balanced_40_acc']:.4f}",
            flush=True,
        )
    print("Gate counts:", gate_stats["counts"], flush=True)
    print("Gate means:", g_means, flush=True)
    print("Gates:", gates_report, flush=True)
    print("STAGE_2_ALL_GATES_PASS" if all_pass else "STAGE_2_GATE_FAIL", flush=True)


if __name__ == "__main__":
    main()
