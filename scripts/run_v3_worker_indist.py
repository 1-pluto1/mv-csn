"""Parallel Worker for Exp 1 (Bench A & B v2 In-Dist) and Exp 8b (Olivi-9 In-Domain) x 3 Seeds."""

from __future__ import annotations

import gc
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

torch.set_num_threads(2)
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
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.trainer import evaluate_mvcsn, train_mvcsn
from mvcsn.utils.config import MVCSNConfig, MVCSN_V3_CONFIG, TrainConfig
from mvcsn.utils.experiment import DEFAULT_SEEDS
from mvcsn.utils.seed import get_device, seed_everything

INDIST_CACHE = Path("results/stage3_indist_cache.json")
OLIVI_ID_CACHE = Path("results/stage3_olivi_indomain_cache.json")


def _atomic_save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_supervised_6_models(
    tag: str,
    tr_recs: List[SpamRecord],
    va_recs: List[SpamRecord],
    te_recs: List[SpamRecord],
    vocab_c: CharVocab,
    graphs_c: Dict[str, torch.Tensor],
    bridge_c: OOVGraphBridge,
    init_mat_192: torch.Tensor,
    preproc: TextPreprocessor,
    tok_rbt,
    col_rbt,
    device: torch.device,
    cache_dict: Dict[str, List[Dict[str, float]]],
    flush_fn,
) -> None:
    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_recs])
    tr_mv_dl = DataLoader(SpamDataset(tr_recs, vocab_c, use_preprocessor=True), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_spam_batch)
    va_mv_dl = DataLoader(SpamDataset(va_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_dl = DataLoader(SpamDataset(te_recs, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    te_mv_nooov = DataLoader(SpamDataset(te_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    tr_texts = [r.proc_text for r in tr_recs]
    tr_y = [r.label for r in tr_recs]
    te_texts = [r.proc_text for r in te_recs]
    te_y = [r.label for r in te_recs]

    def _done(mname: str) -> bool:
        return len(cache_dict.get(mname, [])) >= len(DEFAULT_SEEDS)

    if not _done("TFIDF-SVM"):
        cache_dict["TFIDF-SVM"] = []
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m_te, _, _, _, _ = run_tfidf_svm(tr_texts, tr_y, te_texts, te_y, seed=seed)
            cache_dict["TFIDF-SVM"].append(m_te)
            print(f"  [{tag} s={seed}] TFIDF-SVM F1={m_te['f1']:.4f}", flush=True)
        flush_fn()

    if not _done("LZASD (Fixed)"):
        cache_dict["LZASD (Fixed)"] = []
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            emb = build_lzasd_smoothed_embeddings(tr_texts, vocab_c, seed=seed, device=device)
            m = LZASDModel(emb)
            train_lzasd(m, tr_mv_dl, va_mv_dl, device, epochs=20, lr=1e-3, patience=6)
            m_te, _, _ = evaluate_lzasd(m, te_mv_nooov, device)
            cache_dict["LZASD (Fixed)"].append(m_te)
            print(f"  [{tag} s={seed}] LZASD (Fixed) F1={m_te['f1']:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        flush_fn()

    if not _done("MV-CSN v2"):
        cache_dict["MV-CSN v2"] = []
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m = MVCSNModel.from_config(len(vocab_c), graphs_c, MVCSNConfig())
            train_mvcsn(m, tr_mv_dl, va_mv_dl, device, config=TrainConfig())
            with m.temporary_inference_graph(ext_te if oov_te else None):
                m_te, _, _ = evaluate_mvcsn(m, te_mv_dl, device)
            cache_dict["MV-CSN v2"].append(m_te)
            print(f"  [{tag} s={seed}] MV-CSN v2 F1={m_te['f1']:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        flush_fn()

    v3_notf = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "teacher_feat_dim": 0})
    if not _done("MV-CSN v3"):
        cache_dict["MV-CSN v3"] = []
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m = MVCSNModel.from_config(len(vocab_c), graphs_c, v3_notf)
            m.load_pretrained_char_emb(init_mat_192)
            train_mvcsn(m, tr_mv_dl, va_mv_dl, device, config=TrainConfig(epochs=20, lr=1.2e-3, patience=7))
            with m.temporary_inference_graph(ext_te if oov_te else None):
                m_te, _, _ = evaluate_mvcsn(m, te_mv_dl, device)
            cache_dict["MV-CSN v3"].append(m_te)
            print(f"  [{tag} s={seed}] MV-CSN v3 F1={m_te['f1']:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        flush_fn()

    tr_rbt_dl = DataLoader(RBT3HybridDataset(tr_recs, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=0, pin_memory=True, collate_fn=col_rbt)
    va_rbt_dl = DataLoader(RBT3HybridDataset(va_recs, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
    te_rbt_dl = DataLoader(RBT3HybridDataset(te_recs, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)
    te_fus_dl = DataLoader(RBT3HybridDataset(te_recs, vocab_c, tok_rbt, preproc, oov_map=oov_te, max_len=192, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=col_rbt)

    if not _done("RBT3"):
        cache_dict["RBT3"] = []
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m = RBT3Classifier(model_name="hfl/rbt3", dropout=0.15, use_proc=True)
            train_rbt3_family(m, tr_rbt_dl, va_rbt_dl, device, epochs=4, lr=4e-5, is_fusion=False, use_consistency=True)
            m_te, _, _ = evaluate_rbt3_family(m, te_rbt_dl, device)
            cache_dict["RBT3"].append(m_te)
            print(f"  [{tag} s={seed}] RBT3 F1={m_te['f1']:.4f}", flush=True)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        flush_fn()

    if not _done("RBT3 + MV-CSN v3"):
        cache_dict["RBT3 + MV-CSN v3"] = []
        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            mv_sub = MVCSNModel.from_config(len(vocab_c), graphs_c, v3_notf)
            mv_sub.load_pretrained_char_emb(init_mat_192)
            m = RBT3MVCSNFusion(mv_sub, model_name="hfl/rbt3", dropout=0.15, use_proc=True)
            train_rbt3_family(m, tr_rbt_dl, va_rbt_dl, device, epochs=4, lr=4e-5, is_fusion=True, use_consistency=True)
            with m.mvcsn.temporary_inference_graph(ext_te if oov_te else None):
                m_te, _, _ = evaluate_rbt3_family(m, te_fus_dl, device)
            cache_dict["RBT3 + MV-CSN v3"].append(m_te)
            print(f"  [{tag} s={seed}] RBT3 + MV-CSN v3 F1={m_te['f1']:.4f}", flush=True)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        flush_fn()


def main() -> None:
    t0 = time.time()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")
    vocab_c = CharVocab.load("artifacts/vocab_clean.json")
    graphs_c = load_graphs("artifacts/graphs_clean_top20.pt")
    gmeta_c = load_glyph_meta("artifacts/glyph_meta_clean.pt")
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c, font_dir="/content/fonts")
    init_mat_192 = torch.load("artifacts/pretrained_char_init_192.pt", map_location="cpu", weights_only=False)["init_matrix"]

    tok_rbt = AutoTokenizer.from_pretrained("hfl/rbt3", use_fast=True)
    col_rbt = make_rbt3_collator(tok_rbt.pad_token_id)

    prev_id = json.loads(INDIST_CACHE.read_text(encoding="utf-8")) if INDIST_CACHE.exists() else {"bench_a": {}, "bench_b_v2": {}}

    def _flush_id():
        _atomic_save_json(prev_id, INDIST_CACHE)

    recs_a, _ = build_or_load_splits(
        cache_path="artifacts/splits.json.gz",
        dataset_path="data/resources/variant_spam_16k.tsv.gz",
    )
    print("\n=== [Worker InDist] Bench A ===", flush=True)
    _run_supervised_6_models(
        "BenchA",
        [r for r in recs_a if r.split == "train"],
        [r for r in recs_a if r.split == "val"],
        [r for r in recs_a if r.split == "test"],
        vocab_c, graphs_c, bridge_c, init_mat_192, preproc, tok_rbt, col_rbt, device,
        prev_id["bench_a"], _flush_id,
    )

    recs_b, _ = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    print("\n=== [Worker InDist] Bench B v2 ===", flush=True)
    _run_supervised_6_models(
        "BenchBv2",
        [r for r in recs_b if r.split == "train"],
        [r for r in recs_b if r.split == "val"],
        [r for r in recs_b if r.split == "test"],
        vocab_c, graphs_c, bridge_c, init_mat_192, preproc, tok_rbt, col_rbt, device,
        prev_id["bench_b_v2"], _flush_id,
    )

    print("\n=== [Worker InDist] Olivi-9 In-Domain (70/10/20 Cluster Split) ===", flush=True)
    recs_ol_id, _ = build_or_load_splits(
        cache_path="artifacts/splits_olivi_indomain.json.gz",
        dataset_path="data/resources/external_olivi_spam.tsv.gz",
        sim_threshold=0.65,
        seed=42,
    )
    prev_olid = json.loads(OLIVI_ID_CACHE.read_text(encoding="utf-8")) if OLIVI_ID_CACHE.exists() else {}

    def _flush_olid():
        _atomic_save_json(prev_olid, OLIVI_ID_CACHE)

    _run_supervised_6_models(
        "Olivi9-ID",
        [r for r in recs_ol_id if r.split == "train"],
        [r for r in recs_ol_id if r.split == "val"],
        [r for r in recs_ol_id if r.split == "test"],
        vocab_c, graphs_c, bridge_c, init_mat_192, preproc, tok_rbt, col_rbt, device,
        prev_olid, _flush_olid,
    )
    print(f"\nWORKER_INDIST_DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
