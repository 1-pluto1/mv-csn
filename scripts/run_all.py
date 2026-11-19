"""Full multi-seed experimental benchmark, zero-shot stress suite, ablations, and gate analysis."""

from __future__ import annotations

import gzip
import json
import os
import random
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import threading
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Dict, List, Tuple

import numpy as np
import torch
from pypinyin import Style, pinyin
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from baselines.lzasd.model import LZASDModel, build_lzasd_smoothed_embeddings, evaluate_lzasd, train_lzasd
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
from mvcsn.data.preprocessing import TextPreprocessor, is_cjk
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import build_multiview_graphs, load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.inference import save_demo_bundle
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.metrics import (
    compute_classification_metrics,
    compute_per_attack_f1,
    mcnemar_test,
)
from mvcsn.training.trainer import evaluate_mvcsn, train_mvcsn
from mvcsn.utils.config import MVCSNConfig, TrainConfig
from mvcsn.utils.experiment import ABLATION_SPECS, DEFAULT_SEEDS, agg_stats, to_clean_record
from mvcsn.utils.seed import get_device, seed_everything
from scripts.gdrive_sync import bootstrap_from_gdrive, persist_to_gdrive

SEEDS = list(DEFAULT_SEEDS)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ProgressMonitor:
    """Background daemon logging stage transitions and GPU memory during long runs."""

    def __init__(self, interval: float = 15.0) -> None:
        self.interval = interval
        self.stage = "init"
        self.t0 = time.time()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thr.start()

    def set_stage(self, name: str) -> None:
        self.stage = name
        print(f"[STAGE] {name} (t={time.time() - self.t0:.1f}s)", flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
            print(f"[WATCHDOG] elapsed={time.time() - self.t0:.1f}s | stage={self.stage} | gpu_mem={mem:.0f}MB", flush=True)

    def stop(self) -> None:
        self._stop.set()


def measure_latency_ms(fn, n_samples: int) -> Tuple[float, object]:
    """Measure average inference latency in milliseconds per sample."""
    t0 = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / max(1, n_samples), out


def run_in_dist_benchmark(
    bench_name: str,
    splits_path: str,
    dataset_path: str,
    vocab_path: str,
    graphs_path: str,
    gmeta_path: str,
    tok,
    preproc: TextPreprocessor,
    device: torch.device,
) -> Dict[str, object]:
    records, meta = build_or_load_splits(cache_path=splits_path, dataset_path=dataset_path)
    tr_recs = [r for r in records if r.split == "train"]
    va_recs = [r for r in records if r.split == "val"]
    te_recs = [r for r in records if r.split == "test"]
    y_te = [r.label for r in te_recs]

    ham_lens = [len(r.raw_text) for r in records if r.label == 0]
    spam_lens = [len(r.raw_text) for r in records if r.label == 1]
    length_audit = {
        "ham_p50": float(np.median(ham_lens)),
        "ham_p90": float(np.percentile(ham_lens, 90)),
        "spam_p50": float(np.median(spam_lens)),
        "spam_p90": float(np.percentile(spam_lens, 90)),
    }

    vocab = CharVocab.load(vocab_path)
    graphs = load_graphs(graphs_path)
    gmeta = load_glyph_meta(gmeta_path)
    bridge = OOVGraphBridge(vocab, graphs, gmeta)
    oov_map, ext_bundle = bridge.build_extended_bundle([r.proc_text for r in va_recs + te_recs])

    col_r = make_rbt3_collator(tok.pad_token_id)
    tr_raw = DataLoader(SpamDataset(tr_recs, vocab, use_preprocessor=False), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    va_raw = DataLoader(SpamDataset(va_recs, vocab, use_preprocessor=False), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    te_raw = DataLoader(SpamDataset(te_recs, vocab, use_preprocessor=False), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

    tr_mv = DataLoader(SpamDataset(tr_recs, vocab, is_train=False, preproc=preproc, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    va_mv = DataLoader(SpamDataset(va_recs, vocab, preproc=preproc, use_preprocessor=True, oov_map=oov_map), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    te_mv = DataLoader(SpamDataset(te_recs, vocab, preproc=preproc, use_preprocessor=True, oov_map=oov_map), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

    tr_rl = DataLoader(RBT3HybridDataset(tr_recs, vocab, tok, preproc, max_len=192), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_r)
    va_rl = DataLoader(RBT3HybridDataset(va_recs, vocab, tok, preproc, oov_map=oov_map, max_len=192), batch_size=64, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_r)
    te_rl = DataLoader(RBT3HybridDataset(te_recs, vocab, tok, preproc, oov_map=oov_map, max_len=192), batch_size=64, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_r)

    per_model = defaultdict(list)
    params_count = {}

    for seed in SEEDS:
        seed_everything(seed)
        # 1. TFIDF
        m_tf, _, _, _, _ = run_tfidf_svm([r.raw_text for r in tr_recs], [r.label for r in tr_recs], [r.raw_text for r in te_recs], y_te, seed=seed)
        per_model["TFIDF-SVM"].append(m_tf)
        params_count["TFIDF-SVM"] = 0

        # 2. LZASD
        lz_emb = build_lzasd_smoothed_embeddings([r.raw_text for r in tr_recs], vocab, seed=seed, device=device)
        lz = LZASDModel(lz_emb)
        params_count["LZASD"] = sum(p.numel() for p in lz.parameters())
        train_lzasd(lz, tr_raw, va_raw, device, epochs=15)
        lat_lz, (m_lz, _, _) = measure_latency_ms(lambda: evaluate_lzasd(lz, te_raw, device), len(te_recs))
        m_lz["infer_ms_per_sample"] = lat_lz
        per_model["LZASD"].append(m_lz)

        # 3. RBT3
        rbt = RBT3Classifier("hfl/rbt3")
        params_count["RBT3"] = sum(p.numel() for p in rbt.parameters())
        train_rbt3_family(rbt, tr_rl, va_rl, device, epochs=4, lr=3e-5, is_fusion=False)
        lat_rb, (m_rb, _, _) = measure_latency_ms(lambda: evaluate_rbt3_family(rbt, te_rl, device), len(te_recs))
        m_rb["infer_ms_per_sample"] = lat_rb
        per_model["RBT3"].append(m_rb)

        # 4. MV-CSN v2
        mv = MVCSNModel(
            len(vocab),
            graphs["phon_nbr_idx"], graphs["phon_nbr_sim"],
            graphs["glyph_nbr_idx"], graphs["glyph_nbr_sim"],
            glyph_z=graphs["glyph_z"], pinyin_ids=graphs["pinyin_ids"],
        )
        params_count["MV-CSN v2"] = sum(p.numel() for p in mv.parameters())
        # Train on base graph, attach OOV bridge for val/test
        train_mvcsn(mv, tr_mv, DataLoader(SpamDataset(va_recs, vocab, preproc=preproc, use_preprocessor=True), batch_size=128, shuffle=False, collate_fn=collate_spam_batch), device, epochs=18, patience=6, use_consistency=True)
        mv.set_inference_graph(ext_bundle)
        lat_mv, (m_mv, _, _) = measure_latency_ms(lambda: evaluate_mvcsn(mv, te_mv, device), len(te_recs))
        m_mv["infer_ms_per_sample"] = lat_mv
        per_model["MV-CSN v2"].append(m_mv)

        # 5. RBT3 + MV-CSN v2
        mv_f = MVCSNModel(
            len(vocab),
            graphs["phon_nbr_idx"], graphs["phon_nbr_sim"],
            graphs["glyph_nbr_idx"], graphs["glyph_nbr_sim"],
            glyph_z=graphs["glyph_z"], pinyin_ids=graphs["pinyin_ids"],
        )
        fus = RBT3MVCSNFusion(mv_f, "hfl/rbt3")
        params_count["RBT3 + MV-CSN v2"] = sum(p.numel() for p in fus.parameters())
        va_rl_base = DataLoader(RBT3HybridDataset(va_recs, vocab, tok, preproc, max_len=192), batch_size=64, shuffle=False, collate_fn=col_r)
        train_rbt3_family(fus, tr_rl, va_rl_base, device, epochs=4, lr=3e-5, is_fusion=True, use_consistency=True)
        fus.mvcsn.set_inference_graph(ext_bundle)
        lat_fb, (m_fb, _, _) = measure_latency_ms(lambda: evaluate_rbt3_family(fus, te_rl, device), len(te_recs))
        m_fb["infer_ms_per_sample"] = lat_fb
        per_model["RBT3 + MV-CSN v2"].append(m_fb)

    return {
        "benchmark": bench_name,
        "meta": meta,
        "length_audit": length_audit,
        "params": params_count,
        "summary": {m: agg_stats(v) for m, v in per_model.items()},
        "raw_runs": dict(per_model),
    }


def build_realistic_stress_perturber(vocab: CharVocab, chaizi_path: str = "data/resources/chaizi_map.json.gz"):
    with gzip.open(chaizi_path, "rt", encoding="utf-8") as f:
        ch_data = json.load(f)
    char_to_pair: Dict[str, str] = {}
    for pair, ch in ch_data.get("pair_to_char", {}).items():
        if len(pair) == 2 and ch not in char_to_pair:
            char_to_pair[ch] = pair

    py_to_chars: Dict[str, List[str]] = {}
    for ch in vocab.itos[2:]:
        if is_cjk(ch):
            py = pinyin(ch, style=Style.NORMAL, heteronym=False, errors="ignore")
            if py and py[0]:
                py_to_chars.setdefault(py[0][0], []).append(ch)

    visual_map = {
        "元": ["え", "兀"], "高": ["髙"], "稳": ["穏"], "充": ["茺"], "天": ["夭"],
        "不": ["芣"], "到": ["箌"], "你": ["伱", "妳"], "百": ["佰"], "万": ["萬"],
        "赚": ["賺"], "赔": ["賠"], "顶": ["頂"], "专": ["專"], "赠": ["贈"],
        "清": ["淸"], "微": ["徴"], "钱": ["錢"], "贷": ["貸"], "网": ["網"],
        "领": ["領"], "账": ["賬"], "号": ["號"], "册": ["冊"], "订": ["訂"],
        "Q": ["Ꝗ"], "q": ["Ꝗ"], "O": ["О", "0"], "o": ["о", "0"], "I": ["Ⅰ", "l"],
        "0": ["O", "〇"], "1": ["Ⅰ", "l"], "8": ["⑧", "捌"], "3": ["③", "叁"], "6": ["⑥", "陆"],
        "v": ["\\/", "ⅴ"], "V": ["\\/", "Ⅴ"], ".": ["﹒"],
    }

    def perturb(text: str, rate: float, rng: random.Random) -> str:
        if rate <= 0.0:
            return text
        out: List[str] = []
        for ch in text:
            if rng.random() < rate:
                cands: List[str] = []
                if ch in visual_map:
                    cands.append(rng.choice(visual_map[ch]))
                if ch in char_to_pair:
                    cands.append(char_to_pair[ch])
                if is_cjk(ch):
                    py = pinyin(ch, style=Style.NORMAL, heteronym=False, errors="ignore")
                    if py and py[0]:
                        homos = [c for c in py_to_chars.get(py[0][0], []) if c != ch]
                        if homos:
                            cands.append(rng.choice(homos[:10]))
                if cands:
                    out.append(rng.choice(cands))
                    # Realistic spacing noise only (no artificial '*' beacon!)
                    if rng.random() < 0.25:
                        out.append(" ")
                    continue
            out.append(ch)
        return "".join(out)

    return perturb


def main() -> None:
    bootstrap_from_gdrive()
    mon = ProgressMonitor(interval=15.0)
    mon.start()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")
    tok = AutoTokenizer.from_pretrained("hfl/rbt3", use_fast=True)
    col_r = make_rbt3_collator(tok.pad_token_id)

    try:
        # Experiment 1: In-Distribution Benchmark A + B
        mon.set_stage("exp1_bench_a")
        res_a = run_in_dist_benchmark(
            "Benchmark A (Course 16K)", "artifacts/splits.json.gz", "data/resources/variant_spam_16k.tsv.gz", "artifacts/vocab.json",
            "artifacts/graphs_top20.pt", "artifacts/glyph_meta.pt", tok, preproc, device
        )
        mon.set_stage("exp1_bench_b")
        res_b = run_in_dist_benchmark(
            "Benchmark B (SMS-Matched 16.5K)", "artifacts/splits_sms_matched.json.gz", "data/resources/variant_spam_sms_matched.tsv.gz", "artifacts/vocab_sms.json",
            "artifacts/graphs_sms_top20.pt", "artifacts/glyph_meta_sms.pt", tok, preproc, device
        )
        (RESULTS_DIR / "in_dist_results.json").write_text(
            json.dumps({"benchmark_a": res_a, "benchmark_b": res_b}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Experiment 2 & 3: Zero-Shot Clean->RealVariants + Intensity Curve (0% -> 60%) on Benchmark B
        mon.set_stage("exp2_3_zeroshot_clean_to_variant")
        recs_b, _ = build_or_load_splits(cache_path="artifacts/splits_sms_matched.json.gz", dataset_path="data/resources/variant_spam_sms_matched.tsv.gz")
        tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
        va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]
        te_real = [r for r in recs_b if r.split == "test"]
        te_clean = [to_clean_record(r, preproc) for r in te_real]
        test_atks = [r.attack_type for r in te_real]
        y_te = [r.label for r in te_real]

        vocab_c = CharVocab.build([r.proc_text for r in tr_clean], min_freq=1, max_size=8000)
        vocab_c.save("artifacts/vocab_clean.json")
        graphs_c, gmeta_c = build_multiview_graphs(vocab_c, top_k=20)
        bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c)
        oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])
        stress_fn = build_realistic_stress_perturber(vocab_c)

        # Precompute intensity test sets across rates
        rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        intensity_packs = {}
        for rate in rates:
            rng = random.Random(2026 + int(rate * 100))
            p_recs = []
            for r in te_clean:
                if r.label == 1:
                    rp = stress_fn(r.raw_text, rate, rng)
                    pp = preproc.process(rp, enabled=True) or rp
                    p_recs.append(SpamRecord(r.idx, r.label, rp, pp, r.clean_text, r.cluster_id, r.split, r.attack_type, []))
                else:
                    p_recs.append(r)
            oov_p, ext_p = bridge_c.build_extended_bundle([x.proc_text for x in p_recs])
            intensity_packs[rate] = (p_recs, oov_p, ext_p)

        zs_runs = defaultdict(list)
        atk_runs = defaultdict(list)
        int_runs = {m: defaultdict(list) for m in ["TFIDF-SVM", "LZASD", "RBT3", "MV-CSN v2", "RBT3 + MV-CSN v2"]}
        seed42_preds: Dict[str, List[int]] = {}
        best_mv_model = None

        tr_lz_c = DataLoader(SpamDataset(tr_clean, vocab_c, use_preprocessor=False), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
        va_lz_c = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=False), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
        te_lz_real = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=False), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

        tr_mv_c = DataLoader(SpamDataset(tr_clean, vocab_c, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
        va_mv_c = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
        te_mv_real = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

        tr_rl_c = DataLoader(RBT3HybridDataset(tr_clean, vocab_c, tok, preproc, max_len=192), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, collate_fn=col_r)
        va_rl_c = DataLoader(RBT3HybridDataset(va_clean, vocab_c, tok, preproc, max_len=192), batch_size=64, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_r)
        te_rl_real = DataLoader(RBT3HybridDataset(te_real, vocab_c, tok, preproc, oov_map=oov_te, max_len=192), batch_size=64, shuffle=False, num_workers=2, pin_memory=True, collate_fn=col_r)

        for seed in SEEDS:
            seed_everything(seed)
            # 1. TFIDF
            m_tf, p_tf, _, vec_c, clf_c = run_tfidf_svm([r.raw_text for r in tr_clean], [r.label for r in tr_clean], [r.raw_text for r in te_real], y_te, seed=seed)
            zs_runs["TFIDF-SVM"].append(m_tf)
            atk_runs["TFIDF-SVM"].append(compute_per_attack_f1(y_te, p_tf, test_atks))
            if seed == 42:
                seed42_preds["TFIDF-SVM"] = [int(p >= 0.5) for p in p_tf]

            # 2. LZASD
            lz_c = LZASDModel(build_lzasd_smoothed_embeddings([r.raw_text for r in tr_clean], vocab_c, seed=seed, device=device))
            train_lzasd(lz_c, tr_lz_c, va_lz_c, device, epochs=15)
            m_lz, p_lz, _ = evaluate_lzasd(lz_c, te_lz_real, device)
            zs_runs["LZASD"].append(m_lz)
            atk_runs["LZASD"].append(compute_per_attack_f1(y_te, p_lz, test_atks))
            if seed == 42:
                seed42_preds["LZASD"] = [int(p >= 0.5) for p in p_lz]

            # 3. RBT3
            rb_c = RBT3Classifier("hfl/rbt3")
            train_rbt3_family(rb_c, tr_rl_c, va_rl_c, device, epochs=4, lr=3e-5, is_fusion=False)
            m_rb, p_rb, _ = evaluate_rbt3_family(rb_c, te_rl_real, device)
            zs_runs["RBT3"].append(m_rb)
            atk_runs["RBT3"].append(compute_per_attack_f1(y_te, p_rb, test_atks))
            if seed == 42:
                seed42_preds["RBT3"] = [int(p >= 0.5) for p in p_rb]

            # 4. MV-CSN v2
            mv_c = MVCSNModel(
                len(vocab_c),
                graphs_c["phon_nbr_idx"], graphs_c["phon_nbr_sim"],
                graphs_c["glyph_nbr_idx"], graphs_c["glyph_nbr_sim"],
                glyph_z=graphs_c["glyph_z"], pinyin_ids=graphs_c["pinyin_ids"],
            )
            train_mvcsn(mv_c, tr_mv_c, va_mv_c, device, epochs=18, patience=6, use_consistency=True)
            mv_c.set_inference_graph(ext_te)
            m_mv, p_mv, _ = evaluate_mvcsn(mv_c, te_mv_real, device)
            zs_runs["MV-CSN v2"].append(m_mv)
            atk_runs["MV-CSN v2"].append(compute_per_attack_f1(y_te, p_mv, test_atks))
            if seed == 42:
                seed42_preds["MV-CSN v2"] = [int(p >= 0.5) for p in p_mv]
                best_mv_model = mv_c

            # 5. RBT3 + MV-CSN v2
            mv_fc = MVCSNModel(
                len(vocab_c),
                graphs_c["phon_nbr_idx"], graphs_c["phon_nbr_sim"],
                graphs_c["glyph_nbr_idx"], graphs_c["glyph_nbr_sim"],
                glyph_z=graphs_c["glyph_z"], pinyin_ids=graphs_c["pinyin_ids"],
            )
            fus_c = RBT3MVCSNFusion(mv_fc, "hfl/rbt3")
            train_rbt3_family(fus_c, tr_rl_c, va_rl_c, device, epochs=4, lr=3e-5, is_fusion=True, use_consistency=True)
            fus_c.mvcsn.set_inference_graph(ext_te)
            m_fb, p_fb, _ = evaluate_rbt3_family(fus_c, te_rl_real, device)
            zs_runs["RBT3 + MV-CSN v2"].append(m_fb)
            atk_runs["RBT3 + MV-CSN v2"].append(compute_per_attack_f1(y_te, p_fb, test_atks))
            if seed == 42:
                seed42_preds["RBT3 + MV-CSN v2"] = [int(p >= 0.5) for p in p_fb]

            # Intensity curve across 0% -> 60%
            for rate in rates:
                p_recs, oov_p, ext_p = intensity_packs[rate]
                f_tf = compute_classification_metrics([x.label for x in p_recs], clf_c.predict_proba(vec_c.transform([x.raw_text for x in p_recs]))[:, 1])["f1"]
                f_lz, _, _ = evaluate_lzasd(lz_c, DataLoader(SpamDataset(p_recs, vocab_c, use_preprocessor=False), batch_size=128, shuffle=False, collate_fn=collate_spam_batch), device)
                rl_p = DataLoader(RBT3HybridDataset(p_recs, vocab_c, tok, preproc, oov_map=oov_p, max_len=192), batch_size=64, shuffle=False, collate_fn=col_r)
                f_rb, _, _ = evaluate_rbt3_family(rb_c, rl_p, device)

                mv_c.reset_inference_graph()
                mv_c.set_inference_graph(ext_p)
                f_mv, _, _ = evaluate_mvcsn(mv_c, DataLoader(SpamDataset(p_recs, vocab_c, use_preprocessor=True, oov_map=oov_p), batch_size=128, shuffle=False, collate_fn=collate_spam_batch), device)

                fus_c.mvcsn.reset_inference_graph()
                fus_c.mvcsn.set_inference_graph(ext_p)
                f_fb, _, _ = evaluate_rbt3_family(fus_c, rl_p, device)

                int_runs["TFIDF-SVM"][str(rate)].append(f_tf)
                int_runs["LZASD"][str(rate)].append(f_lz["f1"])
                int_runs["RBT3"][str(rate)].append(f_rb["f1"])
                int_runs["MV-CSN v2"][str(rate)].append(f_mv["f1"])
                int_runs["RBT3 + MV-CSN v2"][str(rate)].append(f_fb["f1"])

        mcnemar = {
            "mvcsn_vs_lzasd": mcnemar_test(y_te, seed42_preds["MV-CSN v2"], seed42_preds["LZASD"]),
            "mvcsn_vs_rbt3": mcnemar_test(y_te, seed42_preds["MV-CSN v2"], seed42_preds["RBT3"]),
            "mvcsn_vs_tfidf": mcnemar_test(y_te, seed42_preds["MV-CSN v2"], seed42_preds["TFIDF-SVM"]),
            "fusion_vs_rbt3": mcnemar_test(y_te, seed42_preds["RBT3 + MV-CSN v2"], seed42_preds["RBT3"]),
        }

        (RESULTS_DIR / "zeroshot_real_variants.json").write_text(
            json.dumps({"summary": {m: agg_stats(v) for m, v in zs_runs.items()}, "mcnemar": mcnemar, "raw_runs": dict(zs_runs)}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (RESULTS_DIR / "per_attack_breakdown.json").write_text(
            json.dumps({m: agg_stats(v) for m, v in atk_runs.items()}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        int_summary = {
            m: {r: {"mean": float(np.mean(vals)), "std": float(np.std(vals))} for r, vals in rdict.items()}
            for m, rdict in int_runs.items()
        }
        (RESULTS_DIR / "intensity_curve.json").write_text(json.dumps({"rates": rates, "summary": int_summary, "raw": int_runs}, indent=2), encoding="utf-8")

        # Save best MV-CSN v2 checkpoint + compact fp16 demo bundle (~20.5 MB)
        best_mv_model.reset_inference_graph()
        torch.save(best_mv_model.state_dict(), "artifacts/mvcsn_best.pt")
        bundle_dest = save_demo_bundle(best_mv_model, vocab_c, graphs_c, gmeta_c, "artifacts/demo_bundle.pt")
        print("Saved artifacts/demo_bundle.pt:", round(bundle_dest.stat().st_size / 1e6, 2), "MB", flush=True)

        # Experiment 4: 8-Row Ablation Study on MV-CSN v2 (Zero-Shot Clean -> RealVariants, 3 seeds)
        mon.set_stage("exp4_ablations")
        te_mv_no_oov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=None), batch_size=128, shuffle=False, collate_fn=collate_spam_batch)
        te_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in te_real]
        oov_raw, ext_raw = bridge_c.build_extended_bundle([r.raw_text for r in te_raw_recs])
        te_mv_noprep = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=128, shuffle=False, collate_fn=collate_spam_batch)

        abl_runs = defaultdict(list)
        for spec in ABLATION_SPECS:
            model_cfg = MVCSNConfig(
                use_phonetic=spec.use_phonetic,
                use_glyph=spec.use_glyph,
                use_gate=spec.use_gate,
                use_multimodal_anchor=spec.use_multimodal_anchor,
            )
            train_cfg = TrainConfig(epochs=18, patience=6, use_consistency=spec.use_consistency)
            for seed in SEEDS:
                seed_everything(seed)
                m = MVCSNModel.from_config(len(vocab_c), graphs_c, model_cfg)
                train_mvcsn(m, tr_mv_c, va_mv_c, device, config=train_cfg)
                if not spec.use_preprocessor:
                    eval_ext, eval_loader = ext_raw, te_mv_noprep
                elif spec.use_oov_bridge:
                    eval_ext, eval_loader = ext_te, te_mv_real
                else:
                    eval_ext, eval_loader = None, te_mv_no_oov
                with m.temporary_inference_graph(eval_ext):
                    res_m, _, _ = evaluate_mvcsn(m, eval_loader, device)
                abl_runs[spec.name].append(res_m)

        full_f1 = float(np.mean([x["f1"] for x in abl_runs["Full MV-CSN v2"]]))
        abl_summary = {}
        for abl_name, runs in abl_runs.items():
            s = agg_stats(runs)
            s["delta_f1"] = s["f1"]["mean"] - full_f1
            abl_summary[abl_name] = s
        (RESULTS_DIR / "ablations.json").write_text(json.dumps({"summary": abl_summary, "raw_runs": dict(abl_runs)}, indent=2), encoding="utf-8")

        # Experiment 5: Gate Interpretability & Case Studies
        mon.set_stage("exp5_gate_analysis")
        best_mv_model.set_inference_graph(ext_te)
        best_mv_model.eval()
        clean_alphas, phon_alphas, glyph_alphas = [], [], []
        case_studies = []

        # Collect token-level gate distributions across test set
        with torch.no_grad():
            for i in range(0, len(te_real), 128):
                chunk = te_real[i : i + 128]
                ds_c = SpamDataset(chunk, vocab_c, use_preprocessor=True, oov_map=oov_te)
                batch = collate_spam_batch([ds_c[j] for j in range(len(chunk))])
                out = best_mv_model(batch["input_ids"].to(device), batch["attn_mask"].to(device))
                alphas = out["alpha"].cpu().numpy()
                obf_probs = torch.sigmoid(out["obf_logits"]).cpu().numpy()
                probs = torch.sigmoid(out["spam_logit"]).cpu().numpy()

                for j, rec in enumerate(chunk):
                    L = min(len(rec.proc_text), 192)
                    for pos in range(L):
                        a_vec = alphas[j, pos].tolist()
                        is_obf = (pos < len(rec.obf_mask)) and (rec.obf_mask[pos] == 1)
                        if not is_obf:
                            if len(clean_alphas) < 6000:
                                clean_alphas.append(a_vec)
                        else:
                            if rec.attack_type == "GLYPH":
                                glyph_alphas.append(a_vec)
                            else:
                                phon_alphas.append(a_vec)

        # Pick diverse showcase examples across real attack types + tricky clean ham
        showcase_candidates = []
        for atk_pref in ("GLYPH", "PHON", "SPLIT", "MIXED"):
            for r in te_real:
                if r.label == 1 and r.attack_type == atk_pref and r not in showcase_candidates and len(r.raw_text) <= 95:
                    showcase_candidates.append(r)
                    break
        showcase_candidates += [r for r in te_real if r.label == 0 and len(r.raw_text) <= 70][:2]

        with torch.no_grad():
            for rec in showcase_candidates:
                ds_1 = SpamDataset([rec], vocab_c, use_preprocessor=True, oov_map=oov_te)
                b1 = collate_spam_batch([ds_1[0]])
                out1 = best_mv_model(b1["input_ids"].to(device), b1["attn_mask"].to(device))
                L = min(len(rec.proc_text), 192)
                case_studies.append({
                    "label": rec.label,
                    "attack_type": rec.attack_type,
                    "raw_text": rec.raw_text,
                    "proc_text": rec.proc_text[:L],
                    "clean_text": rec.clean_text,
                    "pred_prob": float(torch.sigmoid(out1["spam_logit"][0]).item()),
                    "chars": list(rec.proc_text[:L]),
                    "obf_probs": [round(float(x), 4) for x in torch.sigmoid(out1["obf_logits"][0, :L]).cpu().tolist()],
                    "alpha_ctx": [round(float(x), 4) for x in out1["alpha"][0, :L, 0].cpu().tolist()],
                    "alpha_phon": [round(float(x), 4) for x in out1["alpha"][0, :L, 1].cpu().tolist()],
                    "alpha_glyph": [round(float(x), 4) for x in out1["alpha"][0, :L, 2].cpu().tolist()],
                })

        gate_payload = {
            "counts": {"clean": len(clean_alphas), "phon_obf": len(phon_alphas), "glyph_obf": len(glyph_alphas)},
            "means": {
                "clean": np.mean(clean_alphas, axis=0).tolist() if clean_alphas else [0, 0, 0],
                "phon_obf": np.mean(phon_alphas, axis=0).tolist() if phon_alphas else [0, 0, 0],
                "glyph_obf": np.mean(glyph_alphas, axis=0).tolist() if glyph_alphas else [0, 0, 0],
            },
            "samples": {
                "clean": clean_alphas[:1500],
                "phon_obf": phon_alphas[:1500],
                "glyph_obf": glyph_alphas[:1500],
            },
            "case_studies": case_studies,
        }
        (RESULTS_DIR / "gate_analysis.json").write_text(json.dumps(gate_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Print concise executive summary
        print("\n=== STAGE 2 COMPLETE SUMMARY ===", flush=True)
        print("Zero-Shot Clean->RealVariant Summary:")
        for m, stats in json.loads((RESULTS_DIR / "zeroshot_real_variants.json").read_text())["summary"].items():
            print(f"  {m:18s}: F1={stats['f1']['mean']:.4f}±{stats['f1']['std']:.4f} | HamAcc={stats['clean_ham_acc']['mean']:.4f} | AUC={stats['auc']['mean']:.4f}")
        print("Ablations Summary:")
        for a, stats in abl_summary.items():
            print(f"  {a:28s}: F1={stats['f1']['mean']:.4f}±{stats['f1']['std']:.4f} (dF1={stats['delta_f1']:+.4f})")
        print("Gate Means [ctx, phon, glyph]:", gate_payload["means"], flush=True)
        persist_to_gdrive()

    finally:
        mon.stop()


if __name__ == "__main__":
    main()
