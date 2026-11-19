"""Run the 8-row multi-seed ablation study for MV-CSN v2 on Zero-Shot Clean -> RealVariant."""

from __future__ import annotations

import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import DataLoader

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.inference import load_demo_bundle
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.trainer import evaluate_mvcsn, train_mvcsn
from mvcsn.utils.config import MVCSNConfig, TrainConfig
from mvcsn.utils.experiment import (
    ABLATION_SPECS,
    DEFAULT_SEEDS,
    aggregate_metric_runs,
    save_json,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything
from scripts.gdrive_sync import bootstrap_from_gdrive, persist_to_gdrive


def main() -> None:
    """Execute all 8 ablation configurations x 3 seeds and persist ``results/ablations.json``."""
    bootstrap_from_gdrive()
    t0 = time.time()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")

    recs_b, _ = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    te_real = [r for r in recs_b if r.split == "test"]
    if len(recs_b) != 16507 or len(te_real) != 3395:
        raise RuntimeError(f"Unexpected split sizes: total={len(recs_b)}, test={len(te_real)}")

    tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
    va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]

    bundle = load_demo_bundle("artifacts/demo_bundle.pt")
    vocab_c, graphs_c = bundle.vocab, bundle.graphs
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, bundle.gmeta)
    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])

    tr_mv_c = DataLoader(SpamDataset(tr_clean, vocab_c, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    va_mv_c = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    te_mv_real = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    te_mv_no_oov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=None), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

    te_raw_recs = [
        SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, [])
        for r in te_real
    ]
    oov_raw, ext_raw = bridge_c.build_extended_bundle([r.raw_text for r in te_raw_recs])
    te_mv_noprep = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

    abl_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    out_path = Path("results/ablations.json")
    abl_summary: Dict[str, Dict[str, object]] = {}

    for idx_row, spec in enumerate(ABLATION_SPECS, 1):
        model_cfg = MVCSNConfig(
            use_phonetic=spec.use_phonetic,
            use_glyph=spec.use_glyph,
            use_gate=spec.use_gate,
            use_multimodal_anchor=spec.use_multimodal_anchor,
        )
        train_cfg = TrainConfig(epochs=18, patience=6, use_consistency=spec.use_consistency)

        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            model = MVCSNModel.from_config(len(vocab_c), graphs_c, model_cfg)
            train_mvcsn(model, tr_mv_c, va_mv_c, device, config=train_cfg)

            if not spec.use_preprocessor:
                eval_ext, eval_loader = ext_raw, te_mv_noprep
            elif spec.use_oov_bridge:
                eval_ext, eval_loader = ext_te, te_mv_real
            else:
                eval_ext, eval_loader = None, te_mv_no_oov

            with model.temporary_inference_graph(eval_ext):
                res_m, _, _ = evaluate_mvcsn(model, eval_loader, device)

            abl_runs[spec.name].append(res_m)
            print(
                f"[ABL {idx_row}/{len(ABLATION_SPECS)} {spec.name} | seed={seed}] "
                f"F1={res_m['f1']:.4f} HamAcc={res_m['clean_ham_acc']:.4f} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

        full_f1 = float(np.mean([x["f1"] for x in abl_runs["Full MV-CSN v2"]]))
        abl_summary = {}
        for name_k, runs in abl_runs.items():
            stats = aggregate_metric_runs(runs)
            stats["delta_f1"] = stats["f1"]["mean"] - full_f1
            abl_summary[name_k] = stats
        save_json({"summary": abl_summary, "raw_runs": dict(abl_runs)}, out_path)
        persist_to_gdrive(subdirs=("results",))

    print("\n=== ABLATION SUMMARY ===", flush=True)
    for name_k, stats in abl_summary.items():
        print(
            f"  {name_k:28s}: F1={stats['f1']['mean']:.4f}+-{stats['f1']['std']:.4f} "
            f"(dF1={stats['delta_f1']:+.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
