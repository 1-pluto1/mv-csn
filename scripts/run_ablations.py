"""Run the 8-row multi-seed ablation study for MV-CSN v2 on Zero-Shot Clean -> RealVariant."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

signal.signal(signal.SIGHUP, signal.SIG_IGN)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.trainer import evaluate_mvcsn, train_mvcsn
from mvcsn.utils.seed import get_device, seed_everything
from scripts.gdrive_sync import bootstrap_from_gdrive, persist_to_gdrive

SEEDS = [42, 1024, 2025]


def to_clean_record(r: SpamRecord, preproc: TextPreprocessor) -> SpamRecord:
    ct = r.clean_text if r.clean_text else r.raw_text
    pt = preproc.process(ct, enabled=True) or ct
    return SpamRecord(r.idx, r.label, ct, pt, ct, r.cluster_id, r.split, "CLEAN", [0] * min(len(pt), 192))


def agg_stats(items: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = [k for k in items[0].keys() if isinstance(items[0][k], (int, float))]
    return {
        k: {
            "mean": float(np.mean([float(x[k]) for x in items])),
            "std": float(np.std([float(x[k]) for x in items])),
        }
        for k in keys
    }


def main() -> None:
    bootstrap_from_gdrive()
    t0 = time.time()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")

    recs_b, _ = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    te_real = [r for r in recs_b if r.split == "test"]
    assert len(recs_b) == 16507 and len(te_real) == 3395, f"Unexpected split sizes: {len(recs_b)}, {len(te_real)}"

    tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
    va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]

    bundle = torch.load("artifacts/demo_bundle.pt", map_location="cpu", weights_only=False)
    vocab_c = CharVocab(bundle["stoi"])
    graphs_c = {k: v.float() if v.is_floating_point() else v for k, v in bundle["graphs"].items()}
    gmeta_c = {
        "valid_mask": bundle["valid_mask"],
        "glyph_z": bundle["glyph_z_fp16"].float(),
        "glyph_pix": bundle["glyph_pix_fp16"].float(),
        "cae_state": {k: v.float() if v.is_floating_point() else v for k, v in bundle["cae_state"].items()},
        **bundle["glyph_cfg"],
    }
    bridge_c = OOVGraphBridge(vocab_c, graphs_c, gmeta_c)
    oov_te, ext_te = bridge_c.build_extended_bundle([r.proc_text for r in te_real])

    tr_mv_c = DataLoader(SpamDataset(tr_clean, vocab_c, use_preprocessor=True), batch_size=64, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    va_mv_c = DataLoader(SpamDataset(va_clean, vocab_c, use_preprocessor=True), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    te_mv_real = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=oov_te), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)
    te_mv_no_oov = DataLoader(SpamDataset(te_real, vocab_c, use_preprocessor=True, oov_map=None), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

    te_raw_recs = [SpamRecord(r.idx, r.label, r.raw_text, r.raw_text, r.clean_text, r.cluster_id, r.split, r.attack_type, []) for r in te_real]
    oov_raw, ext_raw = bridge_c.build_extended_bundle([r.raw_text for r in te_raw_recs])
    te_mv_noprep = DataLoader(SpamDataset(te_raw_recs, vocab_c, use_preprocessor=False, oov_map=oov_raw), batch_size=128, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_spam_batch)

    ablation_specs = [
        ("Full MV-CSN v2", dict(use_phonetic=True, use_glyph=True, use_gate=True, use_multimodal_anchor=True, use_consistency=True, use_oov_bridge=True, use_preprocessor=True)),
        ("w/o Phonetic View", dict(use_phonetic=False, use_glyph=True, use_gate=True, use_multimodal_anchor=True, use_consistency=True, use_oov_bridge=True, use_preprocessor=True)),
        ("w/o Glyph Visual View", dict(use_phonetic=True, use_glyph=False, use_gate=True, use_multimodal_anchor=True, use_consistency=True, use_oov_bridge=True, use_preprocessor=True)),
        ("w/o Context Gate (Uniform)", dict(use_phonetic=True, use_glyph=True, use_gate=False, use_multimodal_anchor=True, use_consistency=True, use_oov_bridge=True, use_preprocessor=True)),
        ("w/o RoCBert Consistency", dict(use_phonetic=True, use_glyph=True, use_gate=True, use_multimodal_anchor=True, use_consistency=False, use_oov_bridge=True, use_preprocessor=True)),
        ("w/o Multimodal Anchor", dict(use_phonetic=True, use_glyph=True, use_gate=True, use_multimodal_anchor=False, use_consistency=True, use_oov_bridge=True, use_preprocessor=True)),
        ("w/o Dynamic OOV Bridge", dict(use_phonetic=True, use_glyph=True, use_gate=True, use_multimodal_anchor=True, use_consistency=True, use_oov_bridge=False, use_preprocessor=True)),
        ("w/o Split/Noise Preprocessor", dict(use_phonetic=True, use_glyph=True, use_gate=True, use_multimodal_anchor=True, use_consistency=True, use_oov_bridge=True, use_preprocessor=False)),
    ]

    abl_runs: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    out_path = Path("results/ablations.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    abl_summary = {}

    for idx_row, (abl_name, cfg) in enumerate(ablation_specs, 1):
        for seed in SEEDS:
            seed_everything(seed)
            m = MVCSNModel(
                len(vocab_c),
                graphs_c["phon_nbr_idx"], graphs_c["phon_nbr_sim"],
                graphs_c["glyph_nbr_idx"], graphs_c["glyph_nbr_sim"],
                glyph_z=graphs_c["glyph_z"], pinyin_ids=graphs_c["pinyin_ids"],
                use_phonetic=cfg["use_phonetic"],
                use_glyph=cfg["use_glyph"],
                use_gate=cfg["use_gate"],
                use_multimodal_anchor=cfg["use_multimodal_anchor"],
            )
            train_mvcsn(m, tr_mv_c, va_mv_c, device, epochs=18, patience=6, use_consistency=cfg["use_consistency"])
            if not cfg["use_preprocessor"]:
                m.set_inference_graph(ext_raw)
                res_m, _, _ = evaluate_mvcsn(m, te_mv_noprep, device)
            elif cfg["use_oov_bridge"]:
                m.set_inference_graph(ext_te)
                res_m, _, _ = evaluate_mvcsn(m, te_mv_real, device)
            else:
                res_m, _, _ = evaluate_mvcsn(m, te_mv_no_oov, device)
            abl_runs[abl_name].append(res_m)
            f1_v = res_m["f1"]
            ham_v = res_m["clean_ham_acc"]
            elapsed_v = time.time() - t0
            print(f"[ABL {idx_row}/8 {abl_name} | seed={seed}] F1={f1_v:.4f} HamAcc={ham_v:.4f} elapsed={elapsed_v:.1f}s", flush=True)

        full_f1 = float(np.mean([x["f1"] for x in abl_runs["Full MV-CSN v2"]]))
        abl_summary = {}
        for name_k, runs in abl_runs.items():
            s = agg_stats(runs)
            s["delta_f1"] = s["f1"]["mean"] - full_f1
            abl_summary[name_k] = s
        out_path.write_text(json.dumps({"summary": abl_summary, "raw_runs": dict(abl_runs)}, indent=2), encoding="utf-8")
        persist_to_gdrive(subdirs=("results",))

    print("\n=== ABLATION SUMMARY ===", flush=True)
    for name_k, s in abl_summary.items():
        m_f1 = s["f1"]["mean"]
        s_f1 = s["f1"]["std"]
        d_f1 = s["delta_f1"]
        print(f"  {name_k:28s}: F1={m_f1:.4f}+-{s_f1:.4f} (dF1={d_f1:+.4f})", flush=True)


if __name__ == "__main__":
    main()
