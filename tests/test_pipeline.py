"""Unit tests verifying preprocessing, leakage-free splits, dual-view graphs, and models."""

from __future__ import annotations

import torch
from mvcsn.data.preprocessing import TextPreprocessor
from mvcsn.data.splits import build_or_load_splits
from mvcsn.data.vocab import CharVocab
from mvcsn.graph.builder import load_graphs, load_glyph_meta
from mvcsn.graph.glyph import OOVGlyphResolver
from mvcsn.model.mvcsn import MVCSNModel


def test_preprocessor_recomposition() -> None:
    pre = TextPreprocessor("data/resources/chaizi_map.json.gz")
    assert pre.process("氵青联系扣\\/") == "清联系扣v"
    assert pre.process("讠青联系扣\\/") == "请联系扣v"
    assert pre.process("特  请，住（册）给") == "特请,住册给"
    assert pre.process("") == ""


def test_zero_cluster_leakage() -> None:
    records, meta = build_or_load_splits()
    assert meta["total_rows"] == 16007
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


def test_dual_view_graphs_and_oov() -> None:
    vocab = CharVocab.load("artifacts/vocab.json")
    graphs = load_graphs("artifacts/graphs_top16.pt")
    gmeta = load_glyph_meta("artifacts/glyph_meta.pt")

    assert graphs["phon_nbr_idx"].shape[0] == len(vocab)
    assert graphs["glyph_nbr_idx"].shape[0] == len(vocab)

    ig = vocab.stoi["高"]
    igv = vocab.stoi["髙"]
    assert igv in graphs["phon_nbr_idx"][ig].tolist()
    assert igv in graphs["glyph_nbr_idx"][ig].tolist()

    iw = vocab.stoi["稳"]
    iwv = vocab.stoi["穏"]
    assert iwv in graphs["glyph_nbr_idx"][iw].tolist()

    # Unseen Roman numeral / circled / variant visual twins
    res = OOVGlyphResolver(vocab, gmeta, min_sim=0.95)
    assert "ⅴ" not in vocab.stoi
    idx_v, sim_v = res.resolve_char("ⅴ")
    assert vocab.itos[idx_v] == "v" and sim_v >= 0.95
    assert "﹗" not in vocab.stoi
    idx_ex, sim_ex = res.resolve_char("﹗")
    assert vocab.itos[idx_ex] == "!" and sim_ex >= 0.95


def test_mvcsn_forward_and_ablation_modes() -> None:
    vocab = CharVocab.load("artifacts/vocab.json")
    graphs = load_graphs("artifacts/graphs_top16.pt")
    for cfg in [
        dict(use_phonetic=True, use_glyph=True, use_gate=True),
        dict(use_phonetic=False, use_glyph=True, use_gate=True),
        dict(use_phonetic=True, use_glyph=False, use_gate=True),
        dict(use_phonetic=True, use_glyph=True, use_gate=False),
    ]:
        model = MVCSNModel(
            vocab_size=len(vocab),
            phon_nbr_idx=graphs["phon_nbr_idx"],
            phon_nbr_sim=graphs["phon_nbr_sim"],
            glyph_nbr_idx=graphs["glyph_nbr_idx"],
            glyph_nbr_sim=graphs["glyph_nbr_sim"],
            **cfg,
        )
        ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        out = model(ids, mask)
        assert out["spam_logit"].shape == (2,)
        assert out["obf_logits"].shape == (2, 4)
        assert out["alpha"].shape == (2, 4, 3)
        assert not torch.isnan(out["spam_logit"]).any()
        assert torch.allclose(out["alpha"].sum(dim=-1), torch.ones((2, 4)), atol=1e-4)
