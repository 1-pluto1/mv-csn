"""Stage 3 Runner 2: Track 2 Dual-Teacher KD (5 Models x 3 Seeds), Merge, Gate/Error Analysis, Demo Bundle, Latency."""

from __future__ import annotations

import gc
import gzip
import json
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
from scipy.stats import binomtest
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
from mvcsn.data.preprocessing import TextPreprocessor, strip_to_pure_script
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.inference import MVCSNInferenceEngine, save_demo_bundle
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.distill import DistillDataset, collate_distill_batch, train_mvcsn_distilled
from mvcsn.training.metrics import compute_classification_metrics
from mvcsn.training.trainer import evaluate_mvcsn
from mvcsn.utils.config import DistillConfig, MVCSNConfig, MVCSN_V3_CONFIG
from mvcsn.utils.experiment import (
    COUNTERFACTUAL_16_ORIG,
    COUNTERFACTUAL_40_BALANCED,
    DEFAULT_SEEDS,
    agg_stats,
    build_symmetric_stress_perturber,
    measure_fn_ms,
    perturb_records_symmetrically,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

STRESS_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
T1_CACHE = Path("results/stage3_track1_cache.json")
T2_CACHE = Path("results/stage3_track2_cache.json")


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


def _per_attack_metrics(records: List[SpamRecord], probs: List[float]) -> Dict[str, Dict[str, float]]:
    by_atk: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r, p in zip(records, probs):
        by_atk[r.attack_type].append((r.label, p))
    out: Dict[str, Dict[str, float]] = {}
    for atk, pairs in sorted(by_atk.items()):
        ys = [x[0] for x in pairs]
        ps = [x[1] for x in pairs]
        acc = float(np.mean([(p >= 0.5) == (y == 1) for y, p in pairs]))
        pos_ps = [p for y, p in pairs if y == 1]
        neg_ps = [p for y, p in pairs if y == 0]
        rec_entry: Dict[str, float] = {"count": len(pairs), "accuracy": acc}
        if pos_ps:
            rec_entry["spam_recall"] = float(np.mean([p >= 0.5 for p in pos_ps]))
        if neg_ps:
            rec_entry["ham_acc"] = float(np.mean([p < 0.5 for p in neg_ps]))
        if len(set(ys)) == 2:
            rec_entry["f1"] = compute_classification_metrics(ys, ps)["f1"]
        out[atk] = rec_entry
    return out


def _mcnemar_exact(y_true: List[int], p_a: List[float], p_b: List[float]) -> Dict[str, float]:
    b = 0  # A right, B wrong
    c = 0  # A wrong, B right
    for y, pa, pb in zip(y_true, p_a, p_b):
        ca = (pa >= 0.5) == (y == 1)
        cb = (pb >= 0.5) == (y == 1)
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    n = b + c
    pval = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue) if n > 0 else 1.0
    return {"a_right_b_wrong": int(b), "a_wrong_b_right": int(c), "discordant": int(n), "p_value": pval}


def _infer_param_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("kd_proj.")))


def main() -> None:
    t0 = time.time()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")

    recs_b, meta_b = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
    va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]
    te_real = [r for r in recs_b if r.split == "test"]
    te_clean = [to_clean_record(r, preproc) for r in te_real]
    assert len(tr_clean) == 11706 and len(te_real) == 3216

    olivi_recs = _load_olivi(preproc)
    assert len(olivi_recs) == 12011

    vocab_c = CharVocab.load("artifacts/vocab_clean.json")
    graphs_c = load_graphs("artifacts/graphs_clean_top20.pt")
    gmeta_c = load_glyph_meta("artifacts/glyph_meta_clean.pt")
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c, font_dir="/content/fonts")
    init_mat_192 = torch.load("artifacts/pretrained_char_init_192.pt", map_location="cpu", weights_only=False)["init_matrix"]

    ut_cache = torch.load("artifacts/unified_teacher_cache.pt", map_location="cpu", weights_only=False)
    tr_t_logits = ut_cache["tr_logits"]
    tr_t_feats = ut_cache["tr_feats"]
    dp_cache = torch.load("artifacts/dual_teacher_transfer_pool_30k.pt", map_location="cpu", weights_only=False)
    u_recs: List[SpamRecord] = dp_cache["records"]
    u_logits = dp_cache["soft_logits"]
    u_soft_probs = dp_cache["soft_probs"]
    u_feats = dp_cache["feats"]
    assert len(u_recs) == len(u_logits) == len(u_soft_probs) == len(u_feats)
    print(f"[Data] Transfer pool: {len(u_recs)} unlabeled SMS rows", flush=True)

    te_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in te_real]
    te_cjk_recs = [SpamRecord(r.idx, r.label, strip_to_pure_script(r.proc_text) or r.proc_text, strip_to_pure_script(r.proc_text) or r.proc_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in te_real]

    stress_fn = build_symmetric_stress_perturber(vocab_c)
    stress_sets: Dict[str, Dict[float, List[SpamRecord]]] = {"mixed": {}, "sub_only": {}}
    stress_oov: Dict[str, Dict[float, Tuple[Dict[str, int], Dict[str, torch.Tensor]]]] = {"mixed": {}, "sub_only": {}}
    for mode in ("mixed", "sub_only"):
        for rt in STRESS_RATES:
            srecs = perturb_records_symmetrically(te_clean, stress_fn, preproc, rate=rt, seed=2026, mode=mode)
            stress_sets[mode][rt] = srecs
            stress_oov[mode][rt] = bridge_c.build_extended_bundle([r.proc_text for r in srecs])

    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])
    oov_raw, ext_raw = bridge_c.build_extended_bundle([r.raw_text for r in te_raw_recs])
    oov_cjk, ext_cjk = bridge_c.build_extended_bundle([r.proc_text for r in te_cjk_recs])
    oov_ol, ext_ol = bridge_c.build_extended_bundle([r.proc_text for r in olivi_recs])

    cf16_texts = list(COUNTERFACTUAL_16_ORIG)
    cf40_texts = [t for _, t in COUNTERFACTUAL_40_BALANCED]
    cf40_y = [lbl for lbl, _ in COUNTERFACTUAL_40_BALANCED]
    ham_te = [r for r in te_real if r.label == 0]
    spam_te = [r for r in te_real if r.label == 1]
    ham_plus_texts = [r.raw_text.rstrip(".,!?") + ",." for r in ham_te]
    spam_strip_texts = [re.sub(r"[,!:.，！。：]", "", r.raw_text) for r in spam_te]

    cf16_recs = _make_records(cf16_texts, [0] * len(cf16_texts), preproc, True)
    cf40_recs = _make_records(cf40_texts, cf40_y, preproc, True)
    hp_recs = _make_records(ham_plus_texts, [0] * len(ham_plus_texts), preproc, True)
    ss_recs = _make_records(spam_strip_texts, [1] * len(spam_strip_texts), preproc, True)
    oov_16, ext_16 = bridge_c.build_extended_bundle([r.proc_text for r in cf16_recs])
    oov_40, ext_40 = bridge_c.build_extended_bundle([r.proc_text for r in cf40_recs])
    oov_hp, ext_hp = bridge_c.build_extended_bundle([r.proc_text for r in hp_recs])
    oov_ss, ext_ss = bridge_c.build_extended_bundle([r.proc_text for r in ss_recs])

    tr_distill_dl = DataLoader(DistillDataset(tr_clean, vocab_c), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    u_distill_dl = DataLoader(DistillDataset(u_recs, vocab_c), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    va_mv_dl = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    te_mv_fair = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_raw = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_cjk = DataLoader(SpamDataset(te_cjk_recs, vocab_c, use_preprocessor=True, oov_map=oov_cjk), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_ol = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True, oov_map=oov_ol), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    te_mv_fair_nooov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_raw_nooov = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_cjk_nooov = DataLoader(SpamDataset(te_cjk_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_ol_nooov = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    prev = json.loads(T2_CACHE.read_text(encoding="utf-8")) if T2_CACHE.exists() else {}
    fair_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("fair_runs", {}))
    raw_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("raw_runs", {}))
    cjk_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("cjk_runs", {}))
    olivi_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("olivi_runs", {}))
    probe_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("probe_runs", {}))
    per_attack_runs: Dict[str, List[Dict[str, Dict[str, float]]]] = defaultdict(list, prev.get("per_attack_runs", {}))
    stress_runs: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]] = {
        m: defaultdict(lambda: defaultdict(list), {k: defaultdict(list, v) for k, v in prev.get("stress_runs", {}).get(m, {}).items()})
        for m in ("mixed", "sub_only")
    }
    seed42_fair_probs: Dict[str, List[float]] = dict(prev.get("seed42_fair_probs", {}))
    seed42_cf16_probs: Dict[str, List[float]] = dict(prev.get("seed42_cf16_probs", {}))
    params_map: Dict[str, int] = dict(prev.get("params", {}))

    def _flush():
        save_json(
            {
                "seeds": list(DEFAULT_SEEDS),
                "dataset_meta": meta_b,
                "transfer_pool_size": len(u_recs),
                "params": params_map,
                "fair_runs": dict(fair_runs),
                "raw_runs": dict(raw_runs),
                "cjk_runs": dict(cjk_runs),
                "olivi_runs": dict(olivi_runs),
                "probe_runs": dict(probe_runs),
                "per_attack_runs": dict(per_attack_runs),
                "stress_runs": {m: {k: dict(v) for k, v in d.items()} for m, d in stress_runs.items()},
                "seed42_fair_probs": seed42_fair_probs,
                "seed42_cf16_probs": seed42_cf16_probs,
                "elapsed_seconds": time.time() - t0,
            },
            T2_CACHE,
        )

    def _done(name: str) -> bool:
        return len(fair_runs.get(name, [])) >= len(DEFAULT_SEEDS)

    def _record_all(name: str, seed: int, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, p16, n_params: int):
        params_map[name] = int(n_params)
        fair_runs[name].append(m_fair)
        raw_runs[name].append(m_raw)
        cjk_runs[name].append(m_cjk)
        olivi_runs[name].append(m_ol)
        probe_runs[name].append(probes)
        per_attack_runs[name].append(_per_attack_metrics(te_real, p_fair))
        if seed == 42:
            seed42_fair_probs[name] = [round(float(x), 5) for x in p_fair]
            seed42_cf16_probs[name] = [round(float(x), 5) for x in p16]
        print(
            f"  [T2 s={seed}] {name:34s} | Fair={m_fair['f1']:.4f} Raw={m_raw['f1']:.4f} "
            f"CJK={m_cjk['f1']:.4f} Olivi9={m_ol['f1']:.4f} CF16={probes['orig_16_ham_acc']:.4f} CF40={probes['balanced_40_acc']:.4f}",
            flush=True,
        )

    def _run_mvcsn_student(
        name: str,
        model_cfg: MVCSNConfig,
        dcfg: DistillConfig,
        use_oov: bool,
        ckpt_pattern: str | None = None,
        reuse_existing_ckpt: bool = False,
    ) -> None:
        if _done(name):
            print(f"[Skip] {name} already cached", flush=True)
            return
        print(f"\n=== [Track 2] {name} ===", flush=True)
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m = MVCSNModel.from_config(len(vocab_c), graphs_c, model_cfg)
            m.load_pretrained_char_emb(init_mat_192)
            ckpt_p = Path(ckpt_pattern.format(seed=seed)) if ckpt_pattern else None
            if reuse_existing_ckpt and ckpt_p is not None and ckpt_p.exists():
                print(f"  [Reuse] Loading verified checkpoint {ckpt_p}", flush=True)
                sd = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                if isinstance(sd, dict) and "model_state" in sd:
                    sd = sd["model_state"]
                m.load_state_dict(sd, strict=True)
                m = m.to(device)
            else:
                train_mvcsn_distilled(
                    m,
                    tr_distill_dl,
                    va_mv_dl,
                    device,
                    vocab_c,
                    config=dcfg,
                    unlabeled_loader=u_distill_dl if dcfg.use_unlabeled_kd else None,
                    teacher_logits=tr_t_logits,
                    teacher_feats=tr_t_feats,
                    unlabeled_logits=u_logits if dcfg.use_unlabeled_kd else None,
                    unlabeled_feats=u_feats if dcfg.use_unlabeled_kd else None,
                )
                if ckpt_p is not None:
                    ckpt_p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({k: v.cpu() for k, v in m.state_dict().items()}, ckpt_p)

            def _ev(dl, ext_b):
                if use_oov and ext_b is not None:
                    with m.temporary_inference_graph(ext_b):
                        return evaluate_mvcsn(m, dl, device)
                return evaluate_mvcsn(m, dl, device)

            if use_oov:
                m_fair, p_fair, _ = _ev(te_mv_fair, ext_te if oov_te else None)
                m_raw, _, _ = _ev(te_mv_raw, ext_raw if oov_raw else None)
                m_cjk, _, _ = _ev(te_mv_cjk, ext_cjk if oov_cjk else None)
                m_ol, _, _ = _ev(te_mv_ol, ext_ol if oov_ol else None)
            else:
                m_fair, p_fair, _ = _ev(te_mv_fair_nooov, None)
                m_raw, _, _ = _ev(te_mv_raw_nooov, None)
                m_cjk, _, _ = _ev(te_mv_cjk_nooov, None)
                m_ol, _, _ = _ev(te_mv_ol_nooov, None)

            def _ev_recs(recs, om, eb):
                dl = DataLoader(SpamDataset(recs, vocab_c, use_preprocessor=True, oov_map=om if use_oov else None), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
                return _ev(dl, eb if (use_oov and om) else None)

            _, p16, _ = _ev_recs(cf16_recs, oov_16, ext_16)
            m40, _, _ = _ev_recs(cf40_recs, oov_40, ext_40)
            mhp, _, _ = _ev_recs(hp_recs, oov_hp, ext_hp)
            mss, _, _ = _ev_recs(ss_recs, oov_ss, ext_ss)
            probes = {
                "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
                "balanced_40_acc": m40["accuracy"],
                "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
                "bench_spam_stripped_punct_recall": mss["spam_recall"],
            }
            _record_all(name, seed, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, p16, _infer_param_count(m))
            for mode in ("mixed", "sub_only"):
                for rt in STRESS_RATES:
                    om, eb = stress_oov[mode][rt]
                    ms, _, _ = _ev_recs(stress_sets[mode][rt], om, eb)
                    stress_runs[mode][name][f"{rt:.1f}"].append(ms)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        _flush()

    plain_cfg = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "use_phonetic": False, "use_glyph": False, "use_gate": False, "use_multimodal_anchor": False})
    _run_mvcsn_student("Plain CNN+Trans (Same KD)", plain_cfg, DistillConfig(unk_prob=1.0), use_oov=False)
    _run_mvcsn_student("MV-CSN v3 (Clean-KD Only)", MVCSN_V3_CONFIG, DistillConfig(use_unlabeled_kd=False), use_oov=True, ckpt_pattern="artifacts/mvcsn_v3_cleankd_seed{seed}.pt", reuse_existing_ckpt=True)
    _run_mvcsn_student("MV-CSN v3 (Full Dual-Teacher KD)", MVCSN_V3_CONFIG, DistillConfig(), use_oov=True, ckpt_pattern="artifacts/mvcsn_v3_seed{seed}.pt", reuse_existing_ckpt=True)

    # Track 2 RBT3 fair same-data controls
    tok_rbt = AutoTokenizer.from_pretrained("hfl/rbt3", use_fast=True)
    col_rbt = make_rbt3_collator(tok_rbt.pad_token_id)

    def _run_rbt3_dual_teacher(name: str, is_fusion: bool) -> None:
        if _done(name):
            print(f"[Skip] {name} already cached", flush=True)
            return
        print(f"\n=== [Track 2] {name} ===", flush=True)
        tr_dl = DataLoader(RBT3HybridDataset(tr_clean, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_rbt)
        va_dl = DataLoader(RBT3HybridDataset(va_clean, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
        u_dl = DataLoader(RBT3HybridDataset(u_recs, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True, soft_probs=u_soft_probs.tolist()), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_rbt)

        te_fair_dl = DataLoader(RBT3HybridDataset(te_real, vocab_c, tok_rbt, preproc, oov_map=oov_te if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
        te_raw_dl = DataLoader(RBT3HybridDataset(te_raw_recs, vocab_c, tok_rbt, preproc, oov_map=oov_raw if is_fusion else None, max_len=192, use_preprocessor=False), batch_size=256, shuffle=False, collate_fn=col_rbt)
        te_cjk_dl = DataLoader(RBT3HybridDataset(te_cjk_recs, vocab_c, tok_rbt, preproc, oov_map=oov_cjk if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
        ol_dl = DataLoader(RBT3HybridDataset(olivi_recs, vocab_c, tok_rbt, preproc, oov_map=oov_ol if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
        dl_16 = DataLoader(RBT3HybridDataset(cf16_recs, vocab_c, tok_rbt, preproc, oov_map=oov_16 if is_fusion else None, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=col_rbt)
        dl_40 = DataLoader(RBT3HybridDataset(cf40_recs, vocab_c, tok_rbt, preproc, oov_map=oov_40 if is_fusion else None, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=col_rbt)
        dl_hp = DataLoader(RBT3HybridDataset(hp_recs, vocab_c, tok_rbt, preproc, oov_map=oov_hp if is_fusion else None, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
        dl_ss = DataLoader(RBT3HybridDataset(ss_recs, vocab_c, tok_rbt, preproc, oov_map=oov_ss if is_fusion else None, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)

        stress_dls = {"mixed": {}, "sub_only": {}}
        for mode in ("mixed", "sub_only"):
            for rt in STRESS_RATES:
                om, _ = stress_oov[mode][rt]
                stress_dls[mode][rt] = DataLoader(
                    RBT3HybridDataset(stress_sets[mode][rt], vocab_c, tok_rbt, preproc, oov_map=om if is_fusion else None, max_len=192, use_preprocessor=True),
                    batch_size=256, shuffle=False, collate_fn=col_rbt,
                )

        v3_notf = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            tr_aug_recs = perturb_records_symmetrically(tr_clean, stress_fn, preproc, rate=0.20, seed=seed, mode="mixed")
            aug_dl = DataLoader(RBT3HybridDataset(tr_aug_recs, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_rbt)
            if is_fusion:
                mv_sub = MVCSNModel.from_config(len(vocab_c), graphs_c, v3_notf)
                mv_sub.load_pretrained_char_emb(init_mat_192)
                m = RBT3MVCSNFusion(mv_sub, model_name="hfl/rbt3", dropout=0.15, use_proc=True)
            else:
                m = RBT3Classifier(model_name="hfl/rbt3", dropout=0.15, use_proc=True)

            train_rbt3_family(
                m, tr_dl, va_dl, device,
                epochs=4, lr=4e-5, is_fusion=is_fusion, use_consistency=True,
                unlabeled_loader=u_dl, unlabeled_weight=0.35,
                aug_train_loader=aug_dl, aug_cons_weight=0.35,
            )

            def _ev_b(dl, ext_b=None, raw_flag=False):
                prev_up = m.use_proc
                m.use_proc = not raw_flag
                try:
                    if is_fusion:
                        with m.mvcsn.temporary_inference_graph(ext_b):
                            return evaluate_rbt3_family(m, dl, device)
                    return evaluate_rbt3_family(m, dl, device)
                finally:
                    m.use_proc = prev_up

            m_fair, p_fair, _ = _ev_b(te_fair_dl, ext_te if (is_fusion and oov_te) else None, False)
            m_raw, _, _ = _ev_b(te_raw_dl, ext_raw if (is_fusion and oov_raw) else None, True)
            m_cjk, _, _ = _ev_b(te_cjk_dl, ext_cjk if (is_fusion and oov_cjk) else None, False)
            m_ol, _, _ = _ev_b(ol_dl, ext_ol if (is_fusion and oov_ol) else None, False)
            _, p16, _ = _ev_b(dl_16, ext_16 if (is_fusion and oov_16) else None, False)
            m40, _, _ = _ev_b(dl_40, ext_40 if (is_fusion and oov_40) else None, False)
            mhp, _, _ = _ev_b(dl_hp, ext_hp if (is_fusion and oov_hp) else None, False)
            mss, _, _ = _ev_b(dl_ss, ext_ss if (is_fusion and oov_ss) else None, False)
            probes = {
                "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
                "balanced_40_acc": m40["accuracy"],
                "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
                "bench_spam_stripped_punct_recall": mss["spam_recall"],
            }
            _record_all(name, seed, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, p16, sum(p.numel() for p in m.parameters()))
            for mode in ("mixed", "sub_only"):
                for rt in STRESS_RATES:
                    om, eb = stress_oov[mode][rt]
                    ms, _, _ = _ev_b(stress_dls[mode][rt], eb if (is_fusion and om) else None, False)
                    stress_runs[mode][name][f"{rt:.1f}"].append(ms)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        _flush()

    _run_rbt3_dual_teacher("RBT3 (Same Data+Aug)", is_fusion=False)
    _run_rbt3_dual_teacher("RBT3 + MV-CSN v3 (Dual-Teacher)", is_fusion=True)

    # ===== MERGE TRACK 1 + TRACK 2 INTO CANONICAL JSONS =====
    print("\n=== Merging Track 1 + Track 2 into canonical JSONs ===", flush=True)
    t1 = json.loads(T1_CACHE.read_text(encoding="utf-8"))
    t2 = json.loads(T2_CACHE.read_text(encoding="utf-8"))

    t1_models = list(t1["fair_runs"].keys())
    t2_models = list(t2["fair_runs"].keys())
    all_fair = {**t1["fair_runs"], **t2["fair_runs"]}
    all_raw = {**t1["raw_runs"], **t2["raw_runs"]}
    all_cjk = {**t1["cjk_runs"], **t2["cjk_runs"]}
    all_olivi = {**t1["olivi_runs"], **t2["olivi_runs"]}
    all_probes = {**t1["probe_runs"], **t2["probe_runs"]}
    all_pa = {**t1["per_attack_runs"], **t2["per_attack_runs"]}
    all_params = {**t1["params"], **t2["params"]}
    all_s42_probs = {**t1["seed42_fair_probs"], **t2["seed42_fair_probs"]}

    y_te = [r.label for r in te_real]
    mcnemar_pairs = [
        ("MV-CSN v3 (Full Dual-Teacher KD)", "MV-CSN v2 (Clean)"),
        ("MV-CSN v3 (Full Dual-Teacher KD)", "Plain CNN+Trans (Same KD)"),
        ("MV-CSN v3 (Full Dual-Teacher KD)", "MV-CSN v3 (Clean-KD Only)"),
        ("MV-CSN v3 (Full Dual-Teacher KD)", "LZASD (Fixed)"),
        ("MV-CSN v3 (Full Dual-Teacher KD)", "TFIDF-SVM"),
        ("MV-CSN v3 (Full Dual-Teacher KD)", "RBT3 (Clean)"),
        ("MV-CSN v3 (Full Dual-Teacher KD)", "RBT3 (Same Data+Aug)"),
        ("RBT3 + MV-CSN v3 (Dual-Teacher)", "RBT3 (Same Data+Aug)"),
        ("RBT3 + MV-CSN v3 (Dual-Teacher)", "RBT3 (Clean)"),
        ("MacBERT + MV-CSN v3 (Clean)", "MacBERT-base (Clean)"),
    ]
    mcnemar_out = {}
    for a, b in mcnemar_pairs:
        if a in all_s42_probs and b in all_s42_probs:
            mcnemar_out[f"{a} vs {b}"] = _mcnemar_exact(y_te, all_s42_probs[a], all_s42_probs[b])

    save_json(
        {
            "seeds": list(DEFAULT_SEEDS),
            "dataset_meta": meta_b,
            "transfer_pool_size": len(u_recs),
            "params": all_params,
            "summary": {m: agg_stats(v) for m, v in all_fair.items()},
            "track1_clean_only": {m: agg_stats(all_fair[m]) for m in t1_models},
            "track2_dual_teacher_kd": {m: agg_stats(all_fair[m]) for m in t2_models},
            "raw_summary": {m: agg_stats(v) for m, v in all_raw.items()},
            "olivi9_summary": {m: agg_stats(v) for m, v in all_olivi.items()},
            "probes_summary": {m: agg_stats(v) for m, v in all_probes.items()},
            "mcnemar_tests_seed42": mcnemar_out,
            "runs": {"fair": all_fair, "raw": all_raw},
        },
        "results/zeroshot_real_variants.json",
    )

    pa_summary = {}
    for m, run_list in all_pa.items():
        atks = sorted(run_list[0].keys())
        pa_summary[m] = {atk: agg_stats([r[atk] for r in run_list]) for atk in atks}
    save_json({"seeds": list(DEFAULT_SEEDS), "summary": pa_summary, "runs": all_pa}, "results/per_attack_breakdown.json")

    save_json(
        {"seeds": list(DEFAULT_SEEDS), "summary": {m: agg_stats(v) for m, v in all_cjk.items()}, "runs": all_cjk},
        "results/pure_cjk_track.json",
    )

    merged_stress = {}
    merged_stress_summary = {}
    for mode in ("mixed", "sub_only"):
        comb = {**t1["stress_runs"].get(mode, {}), **t2["stress_runs"].get(mode, {})}
        merged_stress[mode] = comb
        merged_stress_summary[mode] = {m: {rt: agg_stats(runs) for rt, runs in rmap.items()} for m, rmap in comb.items()}
    save_json(
        {"seeds": list(DEFAULT_SEEDS), "rates": STRESS_RATES, "summary": merged_stress_summary, "runs": merged_stress},
        "results/intensity_curve.json",
    )

    # External cross-domain Part A (Zero-Shot Olivi-9, Part B In-Domain merged by Runner 3)
    ext_existing = json.loads(Path("results/external_cross_domain.json").read_text(encoding="utf-8")) if Path("results/external_cross_domain.json").exists() else {}
    q8 = json.loads(Path("results/qwen8b_zeroshot.json").read_text(encoding="utf-8"))
    zs_olivi_summary = {m: agg_stats(v) for m, v in all_olivi.items()}
    zs_olivi_summary["Qwen3-8B Zero-Shot"] = {k: {"mean": float(v), "std": 0.0} for k, v in q8["olivi_9"].items()}
    ext_existing["dataset"] = "Olivi-9 External Chinese SMS (12,011 canonical deduplicated rows)"
    ext_existing["n_samples"] = len(olivi_recs)
    ext_existing["zeroshot_olivi9"] = {"summary": zs_olivi_summary, "runs": all_olivi}
    save_json(ext_existing, "results/external_cross_domain.json")

    # ===== EXP 5: GATE ANALYSIS & EXPORT DEMO BUNDLE (Seed-42 MV-CSN v3 Full) =====
    print("\n=== Exp 5: Gate Analysis + Demo Bundle Export ===", flush=True)
    m_s42 = MVCSNModel.from_config(len(vocab_c), graphs_c, MVCSN_V3_CONFIG)
    sd42 = torch.load("artifacts/mvcsn_v3_seed42.pt", map_location="cpu", weights_only=False)
    if isinstance(sd42, dict) and "model_state" in sd42:
        sd42 = sd42["model_state"]
    m_s42.load_state_dict(sd42, strict=True)
    m_s42 = m_s42.to(device).eval()

    V_in = len(vocab_c)
    buckets = {
        "clean_in_vocab": [],
        "obf_in_vocab_phon_or_glyph": [],
        "obf_oov_bridged": [],
        "clean_oov_bridged": [],
        "unbridged_unk": [],
    }
    all_p_s42: List[float] = []
    with torch.no_grad(), m_s42.temporary_inference_graph(ext_te if oov_te else None):
        for batch in te_mv_fair:
            ids = batch["input_ids"].to(device)
            mask = batch["attn_mask"].to(device)
            obf = batch["obf_labels"].to(device)
            out = m_s42(ids, mask)
            probs = torch.sigmoid(out["spam_logit"]).cpu().tolist()
            all_p_s42.extend(probs)
            alpha = out["alpha"].cpu()  # (B, L, 3): ctx, phon, glyph
            ids_c = ids.cpu()
            mask_c = mask.cpu().bool()
            obf_c = obf.cpu()

            m_c_iv = mask_c & (obf_c == 0) & (ids_c >= 2) & (ids_c < V_in)
            m_o_iv = mask_c & (obf_c == 1) & (ids_c >= 2) & (ids_c < V_in)
            m_o_ov = mask_c & (obf_c == 1) & (ids_c >= V_in)
            m_c_ov = mask_c & (obf_c == 0) & (ids_c >= V_in)
            m_unk = mask_c & (ids_c == 1)

            for key, msk in (
                ("clean_in_vocab", m_c_iv),
                ("obf_in_vocab_phon_or_glyph", m_o_iv),
                ("obf_oov_bridged", m_o_ov),
                ("clean_oov_bridged", m_c_ov),
                ("unbridged_unk", m_unk),
            ):
                if msk.any():
                    buckets[key].append(alpha[msk])

    gate_stats = {}
    total_tokens = 0
    for key, chunks in buckets.items():
        if chunks:
            cat = torch.cat(chunks, dim=0)
            n_tok = int(cat.shape[0])
            total_tokens += n_tok
            means = cat.mean(dim=0).tolist()
            stds = cat.std(dim=0).tolist() if n_tok > 1 else [0.0, 0.0, 0.0]
            gate_stats[key] = {
                "token_count": n_tok,
                "alpha_ctx_mean": float(means[0]),
                "alpha_phon_mean": float(means[1]),
                "alpha_glyph_mean": float(means[2]),
                "alpha_graph_total_mean": float(means[1] + means[2]),
                "alpha_ctx_std": float(stds[0]),
                "alpha_phon_std": float(stds[1]),
                "alpha_glyph_std": float(stds[2]),
            }
        else:
            gate_stats[key] = {"token_count": 0}
    print(f"  [Gate] total_tokens={total_tokens} | clean_ctx={gate_stats['clean_in_vocab'].get('alpha_ctx_mean', 0):.4f} "
          f"obf_iv_graph={gate_stats['obf_in_vocab_phon_or_glyph'].get('alpha_graph_total_mean', 0):.4f} "
          f"obf_oov_graph={gate_stats['obf_oov_bridged'].get('alpha_graph_total_mean', 0):.4f}", flush=True)

    # Export & verify demo_bundle.pt
    bundle_path = save_demo_bundle(m_s42, vocab_c, graphs_c, gmeta_c, out_path="artifacts/demo_bundle.pt", model_cfg=MVCSN_V3_CONFIG)
    eng = MVCSNInferenceEngine.from_bundle(bundle_path, chaizi_path="data/resources/chaizi_map.json.gz", font_dir="/content/fonts")
    r_spam = eng.predict("特惠充100送200え，髙额返利稳赚不赔加微十Ꝗ领福利")
    r_ham = eng.predict("我和飞机上一百五十多名旅客眼巴巴期待飞机划破长空腾空而起的时刻")
    print(f"  [DemoBundle] spam_prob={r_spam.spam_prob:.4f} (oov={r_spam.oov_resolved}), ham_prob={r_ham.spam_prob:.4f}", flush=True)
    assert r_spam.spam_prob > 0.90, f"Demo spam prob too low: {r_spam.spam_prob}"
    assert r_ham.spam_prob < 0.10, f"Demo ham prob too high: {r_ham.spam_prob}"
    assert eng.model.kd_proj is None, "kd_proj should be stripped from demo_bundle.pt"

    case_texts = [
        ("glyph_oov_spam", 1, "特惠充100送200え，髙额返利稳赚不赔加微十Ꝗ领福利"),
        ("homophone_spam", 1, "高薪招聘兼职日结三百元加徽信了解详情"),
        ("split_char_spam", 1, "禾兑票代开正夫见保真联纟系电话13800138000"),
        ("space_noise_spam", 1, "免 费 领 取 现 金 红 包 点 击 链 接 立 即 提 现"),
        ("polite_service_ham", 0, "您好，您的快递已放至丰巢快递柜，请凭取件码及时领取。"),
        ("narrative_ham", 0, "我和飞机上一百五十多名旅客眼巴巴期待飞机划破长空腾空而起的时刻"),
    ]
    case_studies = []
    for cid, y_lbl, txt in case_texts:
        res = eng.predict(txt)
        case_studies.append({
            "case_id": cid,
            "true_label": y_lbl,
            "raw_text": res.raw_text,
            "proc_text": res.proc_text,
            "spam_prob": round(float(res.spam_prob), 5),
            "pred_label": res.pred_label,
            "oov_resolved": res.oov_resolved,
            "chars": res.chars,
            "obf_probs": [round(float(x), 4) for x in res.obf_probs],
            "alpha_ctx": [round(float(x), 4) for x in res.alpha_ctx],
            "alpha_phon": [round(float(x), 4) for x in res.alpha_phon],
            "alpha_glyph": [round(float(x), 4) for x in res.alpha_glyph],
        })

    save_json({"total_tokens": total_tokens, "buckets": gate_stats, "case_studies": case_studies}, "results/gate_analysis.json")

    # ===== EXP 7: ERROR ANALYSIS & COUNTERFACTUAL DIAGNOSTICS =====
    print("\n=== Exp 7: Error Analysis ===", flush=True)
    fps, fns = [], []
    fp_by_atk, fn_by_atk = defaultdict(int), defaultdict(int)
    for r, p in zip(te_real, all_p_s42):
        pred = 1 if p >= 0.5 else 0
        if pred == 1 and r.label == 0:
            fp_by_atk[r.attack_type] += 1
            fps.append({"idx": r.idx, "attack_type": r.attack_type, "prob": round(float(p), 5), "len": len(r.raw_text), "text": r.raw_text[:120]})
        elif pred == 0 and r.label == 1:
            fn_by_atk[r.attack_type] += 1
            fns.append({"idx": r.idx, "attack_type": r.attack_type, "prob": round(float(p), 5), "len": len(r.raw_text), "text": r.raw_text[:120]})
    fps.sort(key=lambda x: -x["prob"])
    fns.sort(key=lambda x: x["prob"])

    cf16_detail = []
    for txt, p_v3, p_rbt in zip(
        cf16_texts,
        seed42_cf16_probs["MV-CSN v3 (Full Dual-Teacher KD)"],
        seed42_cf16_probs["RBT3 + MV-CSN v3 (Dual-Teacher)"],
    ):
        cf16_detail.append({
            "text": txt,
            "mvcsn_v3_prob": p_v3,
            "mvcsn_v3_correct": p_v3 < 0.5,
            "rbt3_mvcsn_v3_prob": p_rbt,
            "rbt3_mvcsn_v3_correct": p_rbt < 0.5,
        })

    save_json(
        {
            "model": "MV-CSN v3 (Full Dual-Teacher KD) [Seed 42]",
            "n_test": len(te_real),
            "n_fp": len(fps),
            "n_fn": len(fns),
            "fp_by_attack": dict(fp_by_atk),
            "fn_by_attack": dict(fn_by_atk),
            "top_fps": fps[:25],
            "top_fns": fns[:25],
            "counterfactual_16_detail": cf16_detail,
            "probes_comparison": {m: agg_stats(v) for m, v in all_probes.items()},
        },
        "results/error_analysis.json",
    )

    # ===== LATENCY BENCHMARK (GPU bs=64, GPU bs=1 eager/compile, CPU bs=1) =====
    print("\n=== Latency Benchmark ===", flush=True)
    bench_recs = te_real[:64]
    oov_64, ext_64 = bridge_c.build_extended_bundle([r.proc_text for r in bench_recs])
    mv_b64 = next(iter(DataLoader(SpamDataset(bench_recs, vocab_c, use_preprocessor=True, oov_map=oov_64), batch_size=64, shuffle=False, collate_fn=collate_spam_batch)))
    mv_b1 = next(iter(DataLoader(SpamDataset(bench_recs[:1], vocab_c, use_preprocessor=True, oov_map=oov_64), batch_size=1, shuffle=False, collate_fn=collate_spam_batch)))

    lat_out = {"hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "cpu_threads": torch.get_num_threads()}, "models": {}}

    def _bench_mv(title: str, cfg: MVCSNConfig, use_oov: bool):
        cfg_inf = MVCSNConfig(**{**asdict(cfg), "teacher_feat_dim": 0})
        m = MVCSNModel.from_config(len(vocab_c), graphs_c, cfg_inf).to(device).eval()
        n_p = sum(p.numel() for p in m.parameters())
        ids64, mk64 = mv_b64["input_ids"].to(device), mv_b64["attn_mask"].to(device)
        ids1, mk1 = mv_b1["input_ids"].to(device), mv_b1["attn_mask"].to(device)
        if not use_oov:
            ids64 = ids64.clamp(max=len(vocab_c) - 1)
            ids1 = ids1.clamp(max=len(vocab_c) - 1)
        ctx = m.temporary_inference_graph(ext_64 if (use_oov and oov_64) else None)
        with torch.no_grad(), ctx:
            g64 = measure_fn_ms(lambda: m(ids64, mk64), n_samples=64, warmup=5, repeats=25, cuda_sync=True)
            g1 = measure_fn_ms(lambda: m(ids1, mk1), n_samples=1, warmup=10, repeats=50, cuda_sync=True)
            g1_comp = None
            try:
                mc = torch.compile(m, mode="reduce-overhead")
                g1_comp = measure_fn_ms(lambda: mc(ids1, mk1), n_samples=1, warmup=5, repeats=50, cuda_sync=True)
            except Exception as e:
                print(f"  [compile skip {title}] {e}", flush=True)
        m_cpu = m.cpu().eval()
        ids1_c, mk1_c = ids1.cpu(), mk1.cpu()
        with torch.no_grad(), m_cpu.temporary_inference_graph(ext_64 if (use_oov and oov_64) else None):
            c1 = measure_fn_ms(lambda: m_cpu(ids1_c, mk1_c), n_samples=1, warmup=5, repeats=30, cuda_sync=False)
        lat_out["models"][title] = {
            "params": int(n_p),
            "gpu_bs64_ms_per_msg": float(g64),
            "gpu_bs1_eager_ms_per_msg": float(g1),
            "gpu_bs1_compile_ms_per_msg": float(g1_comp) if g1_comp is not None else None,
            "cpu_bs1_ms_per_msg": float(c1),
        }
        print(f"  [Lat] {title:28s} | GPU64={g64:.3f}ms GPU1={g1:.3f}ms Comp1={g1_comp} CPU1={c1:.3f}ms", flush=True)
        del m, m_cpu
        torch.cuda.empty_cache()

    _bench_mv("MV-CSN v3 (2.35M)", MVCSN_V3_CONFIG, True)
    _bench_mv("Plain CNN+Trans (2.35M)", plain_cfg, False)

    def _bench_hf(title: str, hf_name: str, is_fusion: bool):
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        col = make_rbt3_collator(tok.pad_token_id)
        b64 = next(iter(DataLoader(RBT3HybridDataset(bench_recs, vocab_c, tok, preproc, oov_map=oov_64 if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=col)))
        b1 = next(iter(DataLoader(RBT3HybridDataset(bench_recs[:1], vocab_c, tok, preproc, oov_map=oov_64 if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=1, shuffle=False, collate_fn=col)))
        if is_fusion:
            v3_nf = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
            m = RBT3MVCSNFusion(MVCSNModel.from_config(len(vocab_c), graphs_c, v3_nf), model_name=hf_name, dropout=0.15, use_proc=True).to(device).eval()
        else:
            m = RBT3Classifier(model_name=hf_name, dropout=0.15, use_proc=True).to(device).eval()
        n_p = sum(p.numel() for p in m.parameters())
        b64_g = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b64.items()}
        b1_g = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b1.items()}
        with torch.no_grad():
            if is_fusion:
                with m.mvcsn.temporary_inference_graph(ext_64 if oov_64 else None):
                    g64 = measure_fn_ms(lambda: m(b64_g), n_samples=64, warmup=3, repeats=15, cuda_sync=True)
                    g1 = measure_fn_ms(lambda: m(b1_g), n_samples=1, warmup=5, repeats=30, cuda_sync=True)
            else:
                g64 = measure_fn_ms(lambda: m(b64_g), n_samples=64, warmup=3, repeats=15, cuda_sync=True)
                g1 = measure_fn_ms(lambda: m(b1_g), n_samples=1, warmup=5, repeats=30, cuda_sync=True)
        m_cpu = m.cpu().eval()
        with torch.no_grad():
            if is_fusion:
                with m_cpu.mvcsn.temporary_inference_graph(ext_64 if oov_64 else None):
                    c1 = measure_fn_ms(lambda: m_cpu(b1), n_samples=1, warmup=3, repeats=15, cuda_sync=False)
            else:
                c1 = measure_fn_ms(lambda: m_cpu(b1), n_samples=1, warmup=3, repeats=15, cuda_sync=False)
        lat_out["models"][title] = {
            "params": int(n_p),
            "gpu_bs64_ms_per_msg": float(g64),
            "gpu_bs1_eager_ms_per_msg": float(g1),
            "gpu_bs1_compile_ms_per_msg": None,
            "cpu_bs1_ms_per_msg": float(c1),
        }
        print(f"  [Lat] {title:28s} | GPU64={g64:.3f}ms GPU1={g1:.3f}ms CPU1={c1:.3f}ms", flush=True)
        del m, m_cpu
        gc.collect()
        torch.cuda.empty_cache()

    _bench_hf("RBT3 (38.5M)", "hfl/rbt3", False)
    _bench_hf("RBT3 + MV-CSN v3 (40.8M)", "hfl/rbt3", True)
    _bench_hf("MacBERT + MV-CSN v3 (104.6M)", "hfl/chinese-macbert-base", True)
    lat_out["models"]["Qwen3-8B (8.19B)"] = {
        "params": int(q8["params"]),
        "gpu_bs32_ms_per_msg": float(q8["latency_ms"]["bs32"]),
        "gpu_bs1_eager_ms_per_msg": float(q8["latency_ms"]["bs1"]),
        "cpu_bs1_ms_per_msg": None,
    }
    save_json(lat_out, "results/latency_benchmark.json")
    print(f"\nSTAGE3_RUNNER2_DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
