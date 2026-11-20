"""Parallel Worker for Exp 4 Ablation Shards (Runs concurrently without disturbing tmux v3:0)."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import signal
import sys
import time
from collections import defaultdict
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

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = True

from mvcsn.data.dataset import SpamDataset, collate_spam_batch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import SpamRecord, build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_glyph_meta, load_graphs
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.distill import DistillDataset, collate_distill_batch, train_mvcsn_distilled
from mvcsn.training.trainer import evaluate_mvcsn
from mvcsn.utils.config import DistillConfig, MVCSNConfig, MVCSN_V3_CONFIG
from mvcsn.utils.experiment import (
    ABLATION_SPECS,
    DEFAULT_SEEDS,
    build_symmetric_stress_perturber,
    perturb_records_symmetrically,
    to_clean_record,
)
from mvcsn.utils.seed import get_device, seed_everything

SHARD_INDICES = {
    1: [1, 2, 3, 4, 5],
    2: [7, 8, 9, 11],
}


def _atomic_save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True, choices=[1, 2])
    args = parser.parse_args()

    t0 = time.time()
    device = get_device()
    preproc = TextPreprocessor("data/resources/chaizi_map.json.gz")
    cache_path = Path(f"results/stage3_abl_shard{args.shard}.json")

    recs_b, _ = build_or_load_splits(
        cache_path="artifacts/splits_sms_matched.json.gz",
        dataset_path="data/resources/variant_spam_sms_matched.tsv.gz",
    )
    tr_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "train"]
    va_clean = [to_clean_record(r, preproc) for r in recs_b if r.split == "val"]
    te_real = [r for r in recs_b if r.split == "test"]
    te_clean = [to_clean_record(r, preproc) for r in te_real]
    olivi_recs = _load_olivi(preproc)

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

    prev = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    abl_fair: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("fair", {}))
    abl_raw: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("raw", {}))
    abl_s40: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("symsub40", {}))
    abl_ol: Dict[str, List[Dict[str, float]]] = defaultdict(list, prev.get("olivi9", {}))

    def _flush():
        _atomic_save_json({"fair": dict(abl_fair), "raw": dict(abl_raw), "symsub40": dict(abl_s40), "olivi9": dict(abl_ol)}, cache_path)

    for idx_spec in SHARD_INDICES[args.shard]:
        spec = ABLATION_SPECS[idx_spec]
        sname = spec.name
        if len(abl_fair.get(sname, [])) >= len(DEFAULT_SEEDS):
            print(f"[Shard {args.shard} Skip {idx_spec}] {sname}", flush=True)
            continue
        print(f"\n--- [Shard {args.shard} Abl {idx_spec}] {sname} ---", flush=True)
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

        for seed in DEFAULT_SEEDS:
            seed_everything(seed)
            m = MVCSNModel.from_config(len(vocab_c), graphs_c, mcfg)
            if spec.use_pretrained_init:
                m.load_pretrained_char_emb(init_mat_192)

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
            print(f"  [Shard {args.shard} s={seed}] {sname:34s} | Fair={m_f['f1']:.4f} Raw={m_r['f1']:.4f} Sym40={m_s['f1']:.4f} Olivi9={m_o['f1']:.4f}", flush=True)
            del m
            gc.collect()
            torch.cuda.empty_cache()
        _flush()

    print(f"\nWORKER_ABL_SHARD{args.shard}_DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
