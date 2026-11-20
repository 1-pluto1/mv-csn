"""Stage 3 Runner 1: Track 1 Strict Clean-Only (7 Models x 3 Seeds, resumable incremental saves)."""

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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True

from baselines.lzasd.model import (
    LZASDModel,
    build_lzasd_smoothed_embeddings,
    evaluate_lzasd,
    train_lzasd,
)
from baselines.roberta.finetune import (
    RBT3Classifier,
    RBT3HybridDataset,
    RBT3MVCSNFusion,
    evaluate_rbt3_family,
    make_rbt3_collator,
    train_rbt3_family,
)
from baselines.tfidf_svm import run_tfidf_svm
from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor, strip_to_pure_script
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.metrics import compute_classification_metrics
from mvcsn.training.trainer import evaluate_mvcsn, train_mvcsn
from mvcsn.utils.config import MVCSNConfig, MVCSN_V3_CONFIG, TrainConfig
from mvcsn.utils.experiment import (
    COUNTERFACTUAL_16_ORIG,
    COUNTERFACTUAL_40_BALANCED,
    DEFAULT_SEEDS,
    build_symmetric_stress_perturber,
    perturb_records_symmetrically,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

STRESS_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
CACHE_PATH = Path("results/stage3_track1_cache.json")


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

    tr_mv_loader = DataLoader(SpamDataset(tr_clean, vocab_c, use_preprocessor=True), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_spam_batch)
    va_mv_loader = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_fair = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_raw = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_cjk = DataLoader(SpamDataset(te_cjk_recs, vocab_c, use_preprocessor=True, oov_map=oov_cjk), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_ol = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True, oov_map=oov_ol), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    te_mv_fair_nooov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_raw_nooov = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_cjk_nooov = DataLoader(SpamDataset(te_cjk_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_ol_nooov = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    prev = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
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
    params_map: Dict[str, int] = dict(prev.get("params", {}))

    def _flush():
        save_json(
            {
                "seeds": list(DEFAULT_SEEDS),
                "dataset_meta": meta_b,
                "params": params_map,
                "fair_runs": dict(fair_runs),
                "raw_runs": dict(raw_runs),
                "cjk_runs": dict(cjk_runs),
                "olivi_runs": dict(olivi_runs),
                "probe_runs": dict(probe_runs),
                "per_attack_runs": dict(per_attack_runs),
                "stress_runs": {m: {k: dict(v) for k, v in d.items()} for m, d in stress_runs.items()},
                "seed42_fair_probs": seed42_fair_probs,
                "elapsed_seconds": time.time() - t0,
            },
            CACHE_PATH,
        )

    def _done(name: str) -> bool:
        return len(fair_runs.get(name, [])) >= len(DEFAULT_SEEDS)

    def _record_all(name: str, seed: int, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, n_params: int):
        params_map[name] = int(n_params)
        fair_runs[name].append(m_fair)
        raw_runs[name].append(m_raw)
        cjk_runs[name].append(m_cjk)
        olivi_runs[name].append(m_ol)
        probe_runs[name].append(probes)
        per_attack_runs[name].append(_per_attack_metrics(te_real, p_fair))
        if seed == 42:
            seed42_fair_probs[name] = [round(float(x), 5) for x in p_fair]
        print(
            f"  [T1 s={seed}] {name:28s} | Fair={m_fair['f1']:.4f} Raw={m_raw['f1']:.4f} "
            f"CJK={m_cjk['f1']:.4f} Olivi9={m_ol['f1']:.4f} CF16={probes['orig_16_ham_acc']:.4f} CF40={probes['balanced_40_acc']:.4f}",
            flush=True,
        )

    tr_proc_texts = [r.proc_text for r in tr_clean]
    tr_y = [r.label for r in tr_clean]
    te_proc_texts = [r.proc_text for r in te_real]
    te_raw_texts = [r.raw_text for r in te_real]
    te_cjk_texts = [r.proc_text for r in te_cjk_recs]
    te_y = [r.label for r in te_real]
    ol_proc_texts = [r.proc_text for r in olivi_recs]
    ol_y = [r.label for r in olivi_recs]

    if not _done("TFIDF-SVM"):
        print("\n=== [Track 1] 1/7 TFIDF-SVM ===", flush=True)
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m_fair, p_fair, _, vec, clf = run_tfidf_svm(tr_proc_texts, tr_y, te_proc_texts, te_y, seed=seed)
            m_raw = compute_classification_metrics(te_y, clf.predict_proba(vec.transform(te_raw_texts))[:, 1].tolist())
            m_cjk = compute_classification_metrics(te_y, clf.predict_proba(vec.transform(te_cjk_texts))[:, 1].tolist())
            m_ol = compute_classification_metrics(ol_y, clf.predict_proba(vec.transform(ol_proc_texts))[:, 1].tolist())
            p16 = clf.predict_proba(vec.transform([r.proc_text for r in cf16_recs]))[:, 1].tolist()
            m40 = compute_classification_metrics(cf40_y, clf.predict_proba(vec.transform([r.proc_text for r in cf40_recs]))[:, 1].tolist())
            mhp = compute_classification_metrics([0] * len(hp_recs), clf.predict_proba(vec.transform([r.proc_text for r in hp_recs]))[:, 1].tolist())
            mss = compute_classification_metrics([1] * len(ss_recs), clf.predict_proba(vec.transform([r.proc_text for r in ss_recs]))[:, 1].tolist())
            probes = {
                "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
                "balanced_40_acc": m40["accuracy"],
                "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
                "bench_spam_stripped_punct_recall": mss["spam_recall"],
            }
            _record_all("TFIDF-SVM", seed, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, len(vec.vocabulary_))
            for mode in ("mixed", "sub_only"):
                for rt in STRESS_RATES:
                    sr = stress_sets[mode][rt]
                    ps = clf.predict_proba(vec.transform([r.proc_text for r in sr]))[:, 1].tolist()
                    stress_runs[mode]["TFIDF-SVM"][f"{rt:.1f}"].append(compute_classification_metrics([r.label for r in sr], ps))
        _flush()

    if not _done("LZASD (Fixed)"):
        print("\n=== [Track 1] 2/7 LZASD (Fixed) ===", flush=True)
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            emb = build_lzasd_smoothed_embeddings(tr_proc_texts, vocab_c, seed=seed, device=device)
            m_lz = LZASDModel(emb)
            train_lzasd(m_lz, tr_mv_loader, va_mv_loader, device, epochs=20, lr=1e-3, patience=6)
            m_fair, p_fair, _ = evaluate_lzasd(m_lz, te_mv_fair_nooov, device)
            m_raw, _, _ = evaluate_lzasd(m_lz, te_mv_raw_nooov, device)
            m_cjk, _, _ = evaluate_lzasd(m_lz, te_mv_cjk_nooov, device)
            m_ol, _, _ = evaluate_lzasd(m_lz, te_mv_ol_nooov, device)
            _, p16, _ = evaluate_lzasd(m_lz, DataLoader(SpamDataset(cf16_recs, vocab_c, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=collate_spam_batch), device)
            m40, _, _ = evaluate_lzasd(m_lz, DataLoader(SpamDataset(cf40_recs, vocab_c, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=collate_spam_batch), device)
            mhp, _, _ = evaluate_lzasd(m_lz, DataLoader(SpamDataset(hp_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch), device)
            mss, _, _ = evaluate_lzasd(m_lz, DataLoader(SpamDataset(ss_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch), device)
            probes = {
                "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
                "balanced_40_acc": m40["accuracy"],
                "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
                "bench_spam_stripped_punct_recall": mss["spam_recall"],
            }
            _record_all("LZASD (Fixed)", seed, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, sum(p.numel() for p in m_lz.parameters()))
            for mode in ("mixed", "sub_only"):
                for rt in STRESS_RATES:
                    sr = stress_sets[mode][rt]
                    dl = DataLoader(SpamDataset(sr, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
                    ms, _, _ = evaluate_lzasd(m_lz, dl, device)
                    stress_runs[mode]["LZASD (Fixed)"][f"{rt:.1f}"].append(ms)
            del m_lz
            torch.cuda.empty_cache()
        _flush()

    if not _done("MV-CSN v2 (Clean)"):
        print("\n=== [Track 1] 3/7 MV-CSN v2 (Clean) ===", flush=True)
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m_v2 = MVCSNModel.from_config(len(vocab_c), graphs_c, MVCSNConfig())
            train_mvcsn(m_v2, tr_mv_loader, va_mv_loader, device, config=TrainConfig())
            with m_v2.temporary_inference_graph(ext_te if oov_te else None):
                m_fair, p_fair, _ = evaluate_mvcsn(m_v2, te_mv_fair, device)
            with m_v2.temporary_inference_graph(ext_raw if oov_raw else None):
                m_raw, _, _ = evaluate_mvcsn(m_v2, te_mv_raw, device)
            with m_v2.temporary_inference_graph(ext_cjk if oov_cjk else None):
                m_cjk, _, _ = evaluate_mvcsn(m_v2, te_mv_cjk, device)
            with m_v2.temporary_inference_graph(ext_ol if oov_ol else None):
                m_ol, _, _ = evaluate_mvcsn(m_v2, te_mv_ol, device)

            def _ev_mv(recs, oov_m, ext_b):
                dl = DataLoader(SpamDataset(recs, vocab_c, use_preprocessor=True, oov_map=oov_m), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
                with m_v2.temporary_inference_graph(ext_b if oov_m else None):
                    return evaluate_mvcsn(m_v2, dl, device)

            _, p16, _ = _ev_mv(cf16_recs, oov_16, ext_16)
            m40, _, _ = _ev_mv(cf40_recs, oov_40, ext_40)
            mhp, _, _ = _ev_mv(hp_recs, oov_hp, ext_hp)
            mss, _, _ = _ev_mv(ss_recs, oov_ss, ext_ss)
            probes = {
                "orig_16_ham_acc": float(np.mean([p < 0.5 for p in p16])),
                "balanced_40_acc": m40["accuracy"],
                "bench_ham_plus_comma_period_acc": mhp["clean_ham_acc"],
                "bench_spam_stripped_punct_recall": mss["spam_recall"],
            }
            _record_all("MV-CSN v2 (Clean)", seed, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, sum(p.numel() for p in m_v2.parameters()))
            for mode in ("mixed", "sub_only"):
                for rt in STRESS_RATES:
                    sr = stress_sets[mode][rt]
                    om, eb = stress_oov[mode][rt]
                    ms, _, _ = _ev_mv(sr, om, eb)
                    stress_runs[mode]["MV-CSN v2 (Clean)"][f"{rt:.1f}"].append(ms)
            del m_v2
            torch.cuda.empty_cache()
        _flush()

    def _run_backbone_family(model_title: str, hf_name: str, is_fusion: bool, mv_cfg: MVCSNConfig | None, lr: float, do_stress: bool):
        if _done(model_title):
            print(f"[Skip] {model_title} already cached", flush=True)
            return
        print(f"\n=== [Track 1] {model_title} ===", flush=True)
        tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        col = make_rbt3_collator(tok.pad_token_id)
        tr_dl = DataLoader(RBT3HybridDataset(tr_clean, vocab_c, tok, preproc, max_len=192, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col)
        va_dl = DataLoader(RBT3HybridDataset(va_clean, vocab_c, tok, preproc, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col)
        te_fair_dl = DataLoader(RBT3HybridDataset(te_real, vocab_c, tok, preproc, oov_map=oov_te if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col)
        te_raw_dl = DataLoader(RBT3HybridDataset(te_raw_recs, vocab_c, tok, preproc, oov_map=oov_raw if is_fusion else None, max_len=192, use_preprocessor=False), batch_size=256, shuffle=False, collate_fn=col)
        te_cjk_dl = DataLoader(RBT3HybridDataset(te_cjk_recs, vocab_c, tok, preproc, oov_map=oov_cjk if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col)
        ol_dl = DataLoader(RBT3HybridDataset(olivi_recs, vocab_c, tok, preproc, oov_map=oov_ol if is_fusion else None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col)

        dl_16 = DataLoader(RBT3HybridDataset(cf16_recs, vocab_c, tok, preproc, oov_map=oov_16 if is_fusion else None, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=col)
        dl_40 = DataLoader(RBT3HybridDataset(cf40_recs, vocab_c, tok, preproc, oov_map=oov_40 if is_fusion else None, use_preprocessor=True), batch_size=64, shuffle=False, collate_fn=col)
        dl_hp = DataLoader(RBT3HybridDataset(hp_recs, vocab_c, tok, preproc, oov_map=oov_hp if is_fusion else None, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col)
        dl_ss = DataLoader(RBT3HybridDataset(ss_recs, vocab_c, tok, preproc, oov_map=oov_ss if is_fusion else None, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col)

        stress_dls = None
        if do_stress:
            stress_dls = {"mixed": {}, "sub_only": {}}
            for mode in ("mixed", "sub_only"):
                for rt in STRESS_RATES:
                    om, _ = stress_oov[mode][rt]
                    stress_dls[mode][rt] = DataLoader(
                        RBT3HybridDataset(stress_sets[mode][rt], vocab_c, tok, preproc, oov_map=om if is_fusion else None, max_len=192, use_preprocessor=True),
                        batch_size=256, shuffle=False, collate_fn=col,
                    )

        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            if is_fusion:
                assert mv_cfg is not None
                mv_sub = MVCSNModel.from_config(len(vocab_c), graphs_c, mv_cfg)
                if mv_cfg.hidden_dim == 192:
                    mv_sub.load_pretrained_char_emb(init_mat_192)
                m = RBT3MVCSNFusion(mv_sub, model_name=hf_name, dropout=0.15, use_proc=True)
            else:
                m = RBT3Classifier(model_name=hf_name, dropout=0.15, use_proc=True)
            train_rbt3_family(m, tr_dl, va_dl, device, epochs=4, lr=lr, is_fusion=is_fusion, use_consistency=True)

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
            _record_all(model_title, seed, m_fair, p_fair, m_raw, m_cjk, m_ol, probes, sum(p.numel() for p in m.parameters()))
            if do_stress and stress_dls is not None:
                for mode in ("mixed", "sub_only"):
                    for rt in STRESS_RATES:
                        om, eb = stress_oov[mode][rt]
                        ms, _, _ = _ev_b(stress_dls[mode][rt], eb if (is_fusion and om) else None, False)
                        stress_runs[mode][model_title][f"{rt:.1f}"].append(ms)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        _flush()

    _run_backbone_family("RBT3 (Clean)", "hfl/rbt3", False, None, lr=4e-5, do_stress=True)
    _run_backbone_family("MacBERT-base (Clean)", "hfl/chinese-macbert-base", False, None, lr=3e-5, do_stress=False)
    _run_backbone_family("RBT3 + MV-CSN v2 (Clean)", "hfl/rbt3", True, MVCSNConfig(), lr=4e-5, do_stress=False)
    v3_notf = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
    _run_backbone_family("MacBERT + MV-CSN v3 (Clean)", "hfl/chinese-macbert-base", True, v3_notf, lr=3e-5, do_stress=False)

    _flush()
    print(f"\nSTAGE3_TRACK1_DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
