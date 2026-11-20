"""Stage 3 Runner 3: Exp 4 (12-Row Ablation x 3 Seeds), Exp 1 (In-Dist Bench A & B v2), Exp 8b (Olivi-9 In-Domain), Solo Zero-Contention Latency Refresh."""

from __future__ import annotations

import gc
import gzip
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.distill import DistillDataset, collate_distill_batch, train_mvcsn_distilled
from mvcsn.training.trainer import evaluate_mvcsn, train_mvcsn
from mvcsn.utils.config import DistillConfig, MVCSNConfig, MVCSN_V3_CONFIG, TrainConfig
from mvcsn.utils.experiment import (
    ABLATION_SPECS,
    DEFAULT_SEEDS,
    agg_stats,
    build_symmetric_stress_perturber,
    measure_fn_ms,
    perturb_records_symmetrically,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

ABL_CACHE = Path("results/stage3_ablations_cache.json")
INDIST_CACHE = Path("results/stage3_indist_cache.json")
OLIVI_ID_CACHE = Path("results/stage3_olivi_indomain_cache.json")


def _wait_for_workers(poll_sec: int = 10) -> None:
    while True:
        res = subprocess.run(["pgrep", "-f", "run_v3_worker_"], capture_output=True, text=True)
        pids = [x.strip() for x in res.stdout.splitlines() if x.strip()]
        if not pids:
            print("[Runner 3] All parallel workers exited cleanly; proceeding with merge & remaining work.", flush=True)
            return
        print(f"[Runner 3] Waiting {poll_sec}s for active parallel workers (pids={pids})...", flush=True)
        time.sleep(poll_sec)


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
) -> Dict[str, List[Dict[str, float]]]:
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

    tr_rbt_dl = DataLoader(RBT3HybridDataset(tr_recs, vocab_c, tok_rbt, preproc, oov_map=None, max_len=192, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_rbt)
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

    return cache_dict


def _refresh_solo_latency(
    te_real: List[SpamRecord],
    vocab_c: CharVocab,
    graphs_c: Dict[str, torch.Tensor],
    bridge_c: OOVGraphBridge,
    preproc: TextPreprocessor,
    device: torch.device,
) -> None:
    print("\n=== Solo Zero-Contention Latency Benchmark Refresh ===", flush=True)
    bench_recs = te_real[:64]
    oov_64, ext_64 = bridge_c.build_extended_bundle([r.proc_text for r in bench_recs])
    mv_b64 = next(iter(DataLoader(SpamDataset(bench_recs, vocab_c, use_preprocessor=True, oov_map=oov_64), batch_size=64, shuffle=False, collate_fn=collate_spam_batch)))
    mv_b1 = next(iter(DataLoader(SpamDataset(bench_recs[:1], vocab_c, use_preprocessor=True, oov_map=oov_64), batch_size=1, shuffle=False, collate_fn=collate_spam_batch)))
    plain_cfg = MVCSNConfig(**{**asdict(MVCSN_V3_CONFIG), "use_phonetic": False, "use_glyph": False, "use_gate": False, "use_multimodal_anchor": False})

    lat_out: Dict[str, object] = {
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cpu_threads": torch.get_num_threads(),
            "solo_zero_contention": True,
        },
        "models": {},
    }

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
        print(f"  [SoloLat] {title:28s} | GPU64={g64:.3f}ms GPU1={g1:.3f}ms Comp1={g1_comp} CPU1={c1:.3f}ms", flush=True)
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
        print(f"  [SoloLat] {title:28s} | GPU64={g64:.3f}ms GPU1={g1:.3f}ms CPU1={c1:.3f}ms", flush=True)
        del m, m_cpu
        gc.collect()
        torch.cuda.empty_cache()

    _bench_hf("RBT3 (38.5M)", "hfl/rbt3", False)
    _bench_hf("RBT3 + MV-CSN v3 (40.8M)", "hfl/rbt3", True)
    _bench_hf("MacBERT + MV-CSN v3 (104.6M)", "hfl/chinese-macbert-base", True)
    q8 = json.loads(Path("results/qwen8b_zeroshot.json").read_text(encoding="utf-8"))
    lat_out["models"]["Qwen3-8B (8.19B)"] = {
        "params": int(q8["params"]),
        "gpu_bs32_ms_per_msg": float(q8["latency_ms"]["bs32"]),
        "gpu_bs1_eager_ms_per_msg": float(q8["latency_ms"]["bs1"]),
        "cpu_bs1_ms_per_msg": None,
    }
    save_json(lat_out, "results/latency_benchmark.json")


def main() -> None:
    t0 = time.time()
    _wait_for_workers(poll_sec=10)

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
    u_feats = dp_cache["feats"]

    stress_fn = build_symmetric_stress_perturber(vocab_c)
    sym40_recs = perturb_records_symmetrically(te_clean, stress_fn, preproc, rate=0.4, seed=2026, mode="sub_only")

    te_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in te_real]
    sym40_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in sym40_recs]
    olivi_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in olivi_recs]

    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])
    oov_raw, ext_raw = bridge_c.build_extended_bundle([r.raw_text for r in te_raw_recs])
    oov_s40, ext_s40 = bridge_c.build_extended_bundle([r.proc_text for r in sym40_recs])
    oov_s40_raw, ext_s40_raw = bridge_c.build_extended_bundle([r.raw_text for r in sym40_raw_recs])
    oov_ol, ext_ol = bridge_c.build_extended_bundle([r.proc_text for r in olivi_recs])
    oov_ol_raw, ext_ol_raw = bridge_c.build_extended_bundle([r.raw_text for r in olivi_raw_recs])

    tr_distill_dl = DataLoader(DistillDataset(tr_clean, vocab_c), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    u_distill_dl = DataLoader(DistillDataset(u_recs, vocab_c), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    va_mv_dl = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    tr_clean_raw = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in tr_clean]
    va_clean_raw = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in va_clean]
    u_recs_raw = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in u_recs]
    tr_distill_raw_dl = DataLoader(DistillDataset(tr_clean_raw, vocab_c), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    u_distill_raw_dl = DataLoader(DistillDataset(u_recs_raw, vocab_c), batch_size=128, shuffle=True, pin_memory=True, collate_fn=collate_distill_batch)
    va_mv_raw_dl = DataLoader(SpamDataset(va_clean_raw, vocab_c, use_preprocessor=False), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    dl_fair_oov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_raw_oov = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_s40_oov = DataLoader(SpamDataset(sym40_recs, vocab_c, use_preprocessor=True, oov_map=oov_s40), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_s40_raw_oov = DataLoader(SpamDataset(sym40_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_s40_raw), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_ol_oov = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True, oov_map=oov_ol), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_ol_raw_oov = DataLoader(SpamDataset(olivi_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_ol_raw), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    dl_fair_nooov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_raw_nooov = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_s40_nooov = DataLoader(SpamDataset(sym40_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)
    dl_ol_nooov = DataLoader(SpamDataset(olivi_recs, vocab_c, use_preprocessor=True), batch_size=256, shuffle=False, collate_fn=collate_spam_batch)

    # ===== EXP 4: MERGE WORKER SHARDS & FINISH REMAINING ABLATIONS =====
    print("\n=== Exp 4: Merging Ablation Shards + Checkpoint-Reused Rows ===", flush=True)
    prev_abl = json.loads(ABL_CACHE.read_text(encoding="utf-8")) if ABL_CACHE.exists() else {}
    abl_fair: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev_abl.get("fair", {}))
    abl_raw: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev_abl.get("raw", {}))
    abl_s40: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev_abl.get("symsub40", {}))
    abl_ol: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev_abl.get("olivi9", {}))

    for shard_fn in ("results/stage3_abl_shard1.json", "results/stage3_abl_shard2.json"):
        sp = Path(shard_fn)
        if sp.exists():
            sd = json.loads(sp.read_text(encoding="utf-8"))
            for k, v in sd.get("fair", {}).items():
                if len(v) > len(abl_fair.get(k, [])):
                    abl_fair[k] = v
                    abl_raw[k] = sd.get("raw", {}).get(k, [])
                    abl_s40[k] = sd.get("symsub40", {}).get(k, [])
                    abl_ol[k] = sd.get("olivi9", {}).get(k, [])
            print(f"  [Merge] Loaded {shard_fn}: {list(sd.get('fair', {}).keys())}", flush=True)

    def _flush_abl():
        save_json({"fair": dict(abl_fair), "raw": dict(abl_raw), "symsub40": dict(abl_s40), "olivi9": dict(abl_ol)}, ABL_CACHE)

    _flush_abl()

    for idx_spec, spec in enumerate(ABLATION_SPECS, 1):
        sname = spec.name
        if len(abl_fair.get(sname, [])) >= len(DEFAULT_SEEDS):
            print(f"[Skip Abl {idx_spec}/12] {sname}", flush=True)
            continue
        print(f"\n--- [Abl {idx_spec}/12] {sname} ---", flush=True)
        abl_fair[sname], abl_raw[sname], abl_s40[sname], abl_ol[sname] = [], [], [], []

        mcfg = MVCSNConfig(
            **{
                **asdict(MVCSN_V3_CONFIG),
                "use_phonetic": spec.use_phonetic,
                "use_glyph": spec.use_glyph,
                "use_gate": spec.use_gate,
                "use_multimodal_anchor": spec.use_multimodal_anchor,
            }
        )
        dcfg = DistillConfig(
            use_consistency=spec.use_consistency,
            use_logit_kd=spec.use_domain_kd,
            use_feat_kd=spec.use_domain_kd,
            use_unlabeled_kd=spec.use_unlabeled_kd,
            use_cross_view_kd=spec.use_cross_view_kd,
            use_space_aug=spec.use_space_aug,
        )

        reuse_full = (
            spec.use_phonetic and spec.use_glyph and spec.use_gate and spec.use_multimodal_anchor
            and spec.use_consistency and spec.use_preprocessor and spec.use_pretrained_init
            and spec.use_domain_kd and spec.use_unlabeled_kd and spec.use_cross_view_kd and spec.use_space_aug
        )
        reuse_cleankd = (
            spec.use_phonetic and spec.use_glyph and spec.use_gate and spec.use_multimodal_anchor
            and spec.use_consistency and spec.use_oov_bridge and spec.use_preprocessor and spec.use_pretrained_init
            and spec.use_domain_kd and (not spec.use_unlabeled_kd) and spec.use_cross_view_kd and spec.use_space_aug
        )

        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m = MVCSNModel.from_config(len(vocab_c), graphs_c, mcfg)
            if spec.use_pretrained_init:
                m.load_pretrained_char_emb(init_mat_192)

            loaded = False
            if reuse_full:
                cp = Path(f"artifacts/mvcsn_v3_seed{seed}.pt")
                if cp.exists():
                    sd = torch.load(cp, map_location="cpu", weights_only=False)
                    m.load_state_dict(sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd, strict=True)
                    m = m.to(device)
                    loaded = True
            elif reuse_cleankd:
                cp = Path(f"artifacts/mvcsn_v3_cleankd_seed{seed}.pt")
                if cp.exists():
                    sd = torch.load(cp, map_location="cpu", weights_only=False)
                    m.load_state_dict(sd["model_state"] if isinstance(sd, dict) and "model_state" in sd else sd, strict=True)
                    m = m.to(device)
                    loaded = True

            if not loaded:
                train_mvcsn_distilled(
                    m,
                    tr_distill_dl if spec.use_preprocessor else tr_distill_raw_dl,
                    va_mv_dl if spec.use_preprocessor else va_mv_raw_dl,
                    device,
                    vocab_c,
                    config=dcfg,
                    unlabeled_loader=(u_distill_dl if spec.use_preprocessor else u_distill_raw_dl) if dcfg.use_unlabeled_kd else None,
                    teacher_logits=tr_t_logits if dcfg.use_logit_kd else None,
                    teacher_feats=tr_t_feats if dcfg.use_feat_kd else None,
                    unlabeled_logits=u_logits if dcfg.use_unlabeled_kd else None,
                    unlabeled_feats=u_feats if (dcfg.use_unlabeled_kd and dcfg.use_feat_kd) else None,
                )

            def _ev(dl, eb):
                if spec.use_oov_bridge and eb is not None:
                    with m.temporary_inference_graph(eb):
                        return evaluate_mvcsn(m, dl, device)
                return evaluate_mvcsn(m, dl, device)

            if not spec.use_preprocessor:
                m_f, _, _ = _ev(dl_raw_oov, ext_raw if oov_raw else None)
                m_r = dict(m_f)
                m_s, _, _ = _ev(dl_s40_raw_oov, ext_s40_raw if oov_s40_raw else None)
                m_o, _, _ = _ev(dl_ol_raw_oov, ext_ol_raw if oov_ol_raw else None)
            elif spec.use_oov_bridge:
                m_f, _, _ = _ev(dl_fair_oov, ext_te if oov_te else None)
                m_r, _, _ = _ev(dl_raw_oov, ext_raw if oov_raw else None)
                m_s, _, _ = _ev(dl_s40_oov, ext_s40 if oov_s40 else None)
                m_o, _, _ = _ev(dl_ol_oov, ext_ol if oov_ol else None)
            else:
                m_f, _, _ = _ev(dl_fair_nooov, None)
                m_r, _, _ = _ev(dl_raw_nooov, None)
                m_s, _, _ = _ev(dl_s40_nooov, None)
                m_o, _, _ = _ev(dl_ol_nooov, None)

            abl_fair[sname].append(m_f)
            abl_raw[sname].append(m_r)
            abl_s40[sname].append(m_s)
            abl_ol[sname].append(m_o)
            print(f"  [Abl s={seed}] {sname:36s} | Fair={m_f['f1']:.4f} Raw={m_r['f1']:.4f} Sym40={m_s['f1']:.4f} Olivi9={m_o['f1']:.4f}", flush=True)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        _flush_abl()

    full_name = ABLATION_SPECS[0].name
    sum_f = {s.name: agg_stats(abl_fair[s.name]) for s in ABLATION_SPECS}
    sum_r = {s.name: agg_stats(abl_raw[s.name]) for s in ABLATION_SPECS}
    sum_s = {s.name: agg_stats(abl_s40[s.name]) for s in ABLATION_SPECS}
    sum_o = {s.name: agg_stats(abl_ol[s.name]) for s in ABLATION_SPECS}
    deltas = {}
    for s in ABLATION_SPECS:
        k = s.name
        deltas[k] = {
            "delta_fair_f1": round(sum_f[k]["f1"]["mean"] - sum_f[full_name]["f1"]["mean"], 5),
            "delta_raw_f1": round(sum_r[k]["f1"]["mean"] - sum_r[full_name]["f1"]["mean"], 5),
            "delta_symsub40_f1": round(sum_s[k]["f1"]["mean"] - sum_s[full_name]["f1"]["mean"], 5),
            "delta_olivi9_f1": round(sum_o[k]["f1"]["mean"] - sum_o[full_name]["f1"]["mean"], 5),
        }
    save_json(
        {
            "seeds": list(DEFAULT_SEEDS),
            "specs": [asdict(s) for s in ABLATION_SPECS],
            "summary": sum_f,
            "raw_summary": sum_r,
            "symsub40_summary": sum_s,
            "olivi9_summary": sum_o,
            "deltas_vs_full": deltas,
            "runs": {"fair": dict(abl_fair), "raw": dict(abl_raw), "symsub40": dict(abl_s40), "olivi9": dict(abl_ol)},
        },
        "results/ablations.json",
    )
    print(f"\n[Done] Exp 4 results/ablations.json written ({time.time() - t0:.1f}s)", flush=True)

    # ===== EXP 1: IN-DISTRIBUTION (BENCH A & BENCH B v2) x 3 SEEDS =====
    print("\n=== Exp 1: In-Distribution (Bench A & Bench B v2) x 3 Seeds ===", flush=True)
    tok_rbt = AutoTokenizer.from_pretrained("hfl/rbt3", use_fast=True)
    col_rbt = make_rbt3_collator(tok_rbt.pad_token_id)

    prev_id = json.loads(INDIST_CACHE.read_text(encoding="utf-8")) if INDIST_CACHE.exists() else {"bench_a": {}, "bench_b_v2": {}}

    def _flush_id():
        save_json(prev_id, INDIST_CACHE)

    recs_a, meta_a = build_or_load_splits(
        cache_path="artifacts/splits.json.gz",
        dataset_path="data/resources/variant_spam_16k.tsv.gz",
    )
    _run_supervised_6_models(
        "BenchA",
        [r for r in recs_a if r.split == "train"],
        [r for r in recs_a if r.split == "val"],
        [r for r in recs_a if r.split == "test"],
        vocab_c, graphs_c, bridge_c, init_mat_192, preproc, tok_rbt, col_rbt, device,
        prev_id["bench_a"], _flush_id,
    )
    _run_supervised_6_models(
        "BenchBv2",
        [r for r in recs_b if r.split == "train"],
        [r for r in recs_b if r.split == "val"],
        [r for r in recs_b if r.split == "test"],
        vocab_c, graphs_c, bridge_c, init_mat_192, preproc, tok_rbt, col_rbt, device,
        prev_id["bench_b_v2"], _flush_id,
    )
    save_json(
        {
            "seeds": list(DEFAULT_SEEDS),
            "summary": {m: agg_stats(v) for m, v in prev_id["bench_b_v2"].items()},
            "bench_a_original": {"meta": meta_a, "summary": {m: agg_stats(v) for m, v in prev_id["bench_a"].items()}, "runs": prev_id["bench_a"]},
            "bench_b_v2_sms_matched": {"meta": meta_b, "summary": {m: agg_stats(v) for m, v in prev_id["bench_b_v2"].items()}, "runs": prev_id["bench_b_v2"]},
        },
        "results/in_dist_results.json",
    )
    print(f"\n[Done] Exp 1 results/in_dist_results.json written ({time.time() - t0:.1f}s)", flush=True)

    # ===== EXP 8 PART B: OLIVI-9 IN-DOMAIN (70/10/20 TEMPLATE-CLUSTER SPLIT) =====
    print("\n=== Exp 8b: Olivi-9 In-Domain 70/10/20 Template-Cluster Split ===", flush=True)
    recs_ol_id, meta_ol_id = build_or_load_splits(
        cache_path="artifacts/splits_olivi_indomain.json.gz",
        dataset_path="data/resources/external_olivi_spam.tsv.gz",
        sim_threshold=0.65,
        seed=42,
    )
    prev_olid = json.loads(OLIVI_ID_CACHE.read_text(encoding="utf-8")) if OLIVI_ID_CACHE.exists() else {}

    def _flush_olid():
        save_json(prev_olid, OLIVI_ID_CACHE)

    _run_supervised_6_models(
        "Olivi9-ID",
        [r for r in recs_ol_id if r.split == "train"],
        [r for r in recs_ol_id if r.split == "val"],
        [r for r in recs_ol_id if r.split == "test"],
        vocab_c, graphs_c, bridge_c, init_mat_192, preproc, tok_rbt, col_rbt, device,
        prev_olid, _flush_olid,
    )

    ext_path = Path("results/external_cross_domain.json")
    ext_data = json.loads(ext_path.read_text(encoding="utf-8")) if ext_path.exists() else {}
    ext_data["indomain_olivi9"] = {
        "meta": meta_ol_id,
        "summary": {m: agg_stats(v) for m, v in prev_olid.items()},
        "runs": prev_olid,
    }
    save_json(ext_data, ext_path)

    # Zero-contention solo latency refresh before Runner 4
    _refresh_solo_latency(te_real, vocab_c, graphs_c, bridge_c, preproc, device)
    print(f"\nSTAGE3_RUNNER3_DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
