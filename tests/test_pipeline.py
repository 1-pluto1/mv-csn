"""Unit tests verifying preprocessing, leakage-free splits, dual-view graphs, OOV Bridge, and MV-CSN v2."""

from __future__ import annotations

import torch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_graphs, load_glyph_meta
from mvcsn.graph.oov_bridge import OOVGraphBridge
from mvcsn.model.mvcsn import MVCSNModel
from mvcsn.training.losses import compute_consistency_loss, compute_mvcsn_loss, generate_gpu_graph_perturbation


def test_preprocessor_recomposition_and_nfkc() -> None:
    pre = TextPreprocessor("data/resources/chaizi_map.json.gz")
    assert pre.process("氵青联系扣\\/") == "清联系扣v"
    assert pre.process("讠青联系扣\\/") == "请联系扣v"
    assert pre.process("特  请，住（册）给") == "特请,住册给"
    # Enclosed digits normalize selectively while cross-script homoglyphs remain intact
    assert pre.process("送⑧⑧⑧え") == "送888え"
    txt, m = pre.process_with_mask("氵青送⑧", [1, 1, 0, 0])
    assert txt == "清送8" and m == [1, 0, 1]
    assert pre.process("") == ""


def test_zero_cluster_leakage_both_benchmarks() -> None:
    for path in ("artifacts/splits.json.gz", "artifacts/splits_sms_matched.json.gz"):
        records, meta = build_or_load_splits(cache_path=path)
        assert meta["total_rows"] >= 16000
        clusters = {"train": set(), "val": set(), "test": set()}
        texts = {"train": set(), "val": set(), "test": set()}
        for r in records:
            clusters[r.split].add(r.cluster_id)
            texts[r.split].add(r.proc_text)
        assert clusters["train"].isdisjoint(clusters["val"])
        assert clusters["train"].isdisjoint(clusters["test"])
        assert clusters["val"].isdisjoint(clusters["test"])
        assert texts["train"].isdisjoint(texts["test"])
        assert texts["train"].isdisjoint(texts["val"])


from pathlib import Path

from mvcsn.inference import load_demo_bundle


def _load_test_graph_artifacts():
    """Load cached graph artifacts, falling back to the committed demo_bundle.pt on fresh clones."""
    if Path("artifacts/vocab.json").exists() and Path("artifacts/glyph_meta.pt").exists():
        return (
            CharVocab.load("artifacts/vocab.json"),
            load_graphs("artifacts/graphs_top20.pt"),
            load_glyph_meta("artifacts/glyph_meta.pt"),
        )
    bundle = load_demo_bundle("artifacts/demo_bundle.pt")
    return bundle.vocab, bundle.graphs, bundle.gmeta


def test_multimodal_graph_and_oov_bridge() -> None:
    vocab, graphs, gmeta = _load_test_graph_artifacts()

    assert graphs["phon_nbr_idx"].shape[0] == len(vocab)
    assert graphs["glyph_nbr_idx"].shape[0] == len(vocab)
    assert graphs["glyph_z"].shape == (len(vocab), 64)
    assert graphs["pinyin_ids"].shape == (len(vocab), 3)

    bridge = OOVGraphBridge(vocab, graphs, gmeta)
    ig = vocab.stoi["高"]
    if "髙" in vocab.stoi:
        igv = vocab.stoi["髙"]
        assert igv in graphs["phon_nbr_idx"][ig].tolist()
        assert igv in graphs["glyph_nbr_idx"][ig].tolist()
    else:
        oov_g, ext_g = bridge.build_extended_bundle(["髙"])
        assert ig in ext_g["glyph_nbr_idx"][oov_g["髙"]].tolist()

    oov_map, ext_bundle = bridge.build_extended_bundle(["50萬Ꝗ送Α"])
    assert "Ꝗ" in oov_map and "Α" in oov_map
    oov_a_id = oov_map["Α"]
    assert oov_a_id >= len(vocab)
    assert vocab.stoi["A"] in ext_bundle["glyph_nbr_idx"][oov_a_id].tolist()


def test_mvcsn_v2_forward_oov_consistency_and_ablations() -> None:
    vocab, graphs, gmeta = _load_test_graph_artifacts()

    for cfg in [
        dict(use_phonetic=True, use_glyph=True, use_gate=True, use_multimodal_anchor=True),
        dict(use_phonetic=False, use_glyph=True, use_gate=True, use_multimodal_anchor=True),
        dict(use_phonetic=True, use_glyph=False, use_gate=True, use_multimodal_anchor=True),
        dict(use_phonetic=True, use_glyph=True, use_gate=False, use_multimodal_anchor=False),
    ]:
        model = MVCSNModel(
            vocab_size=len(vocab),
            phon_nbr_idx=graphs["phon_nbr_idx"],
            phon_nbr_sim=graphs["phon_nbr_sim"],
            glyph_nbr_idx=graphs["glyph_nbr_idx"],
            glyph_nbr_sim=graphs["glyph_nbr_sim"],
            glyph_z=graphs["glyph_z"],
            pinyin_ids=graphs["pinyin_ids"],
            **cfg,
        )
        ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        out1 = model(ids, mask)
        pert_ids, pert_obf = generate_gpu_graph_perturbation(
            ids, mask, model.phon_nbr_idx, model.phon_nbr_sim, model.glyph_nbr_idx, model.glyph_nbr_sim, token_prob=0.5
        )
        out2 = model(pert_ids, mask)
        l = compute_mvcsn_loss(out2, torch.tensor([1.0, 0.0]), pert_obf, mask)
        c = compute_consistency_loss(out1, out2)
        (l["total"] + c).backward()
        assert not torch.isnan(out1["spam_logit"]).any()
        assert torch.allclose(out1["alpha"].sum(dim=-1), torch.ones((2, 4)), atol=1e-4)

    # Verify OOV bridge attachment & reset inside model
    bridge = OOVGraphBridge(vocab, graphs, gmeta)
    oov_map, ext_bundle = bridge.build_extended_bundle(["ꝖΑ"])
    model.set_inference_graph(ext_bundle)
    oov_ids = torch.tensor([[oov_map["Ꝗ"], oov_map["Α"], 2, 0]], dtype=torch.long)
    oov_mask = torch.tensor([[True, True, True, False]])
    out_oov = model(oov_ids, oov_mask)
    assert not torch.isnan(out_oov["spam_logit"]).any()
    model.reset_inference_graph()
