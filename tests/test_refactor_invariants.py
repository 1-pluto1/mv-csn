"""Unit tests verifying engineering refactoring invariants, configs, and inference facade."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mvcsn import (
    CharVocab,
    MVCSNConfig,
    MVCSNInferenceEngine,
    MVCSNModel,
    TextPreprocessor,
    TrainConfig,
    load_demo_bundle,
)
from mvcsn.data.splits import SpamRecord
from mvcsn.utils.experiment import ABLATION_SPECS, DEFAULT_SEEDS, aggregate_metric_runs, to_clean_record

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_and_temporary_oov_graph_cleanup() -> None:
    """Verify MVCSNModel.from_config and exception-safe temporary_inference_graph cleanup."""
    V, K = 16, 4
    graphs = {
        "phon_nbr_idx": torch.zeros((V, K), dtype=torch.long),
        "phon_nbr_sim": torch.zeros((V, K), dtype=torch.float32),
        "glyph_nbr_idx": torch.zeros((V, K), dtype=torch.long),
        "glyph_nbr_sim": torch.zeros((V, K), dtype=torch.float32),
        "glyph_z": torch.randn((V, 64), dtype=torch.float32),
        "pinyin_ids": torch.zeros((V, 3), dtype=torch.long),
    }
    cfg = MVCSNConfig(hidden_dim=64, num_heads=2, gat_heads=2, num_transformer_layers=1, transformer_ff_dim=128)
    model = MVCSNModel.from_config(V, graphs, cfg)
    ext = {k: torch.cat([v, v[:3]], dim=0) for k, v in graphs.items()}

    with pytest.raises(RuntimeError, match="simulated"):
        with model.temporary_inference_graph(ext):
            assert model.phon_nbr_idx.shape[0] == V + 3
            raise RuntimeError("simulated inference error")

    # Even after an exception inside the block, base buffers must be cleanly restored.
    assert model.phon_nbr_idx.shape[0] == V
    assert model._saved_base_buffers is None


def test_preprocessor_mask_parity() -> None:
    """Verify normalize_noise/process and process_with_mask stay strictly in sync."""
    preproc = TextPreprocessor(REPO_ROOT / "data/resources/chaizi_map.json.gz")
    samples = [
        "特  请，住（册）给，（3  8  8 ） ，  记（着）来 ，按 ：94672.cn",
        "氵青联\\/系客服讠主册即送88⑧元彩金，详询薇信",
        "Hello   world  测 试  2026",
    ]
    for raw in samples:
        plain = preproc.process(raw, enabled=True)
        with_mask, mask = preproc.process_with_mask(raw, [0] * len(raw), enabled=True)
        assert with_mask == plain
        assert len(mask) == len(with_mask)


def test_experiment_utilities_and_ablation_matrix() -> None:
    """Verify shared DRY experiment helpers and canonical 8-row ablation spec."""
    assert DEFAULT_SEEDS == (42, 1024, 2025)
    assert len(ABLATION_SPECS) == 8
    assert ABLATION_SPECS[0].name == "Full MV-CSN v2"

    stats = aggregate_metric_runs([{"f1": 0.99, "auc": 1.0}, {"f1": 0.97, "auc": 0.98}])
    assert abs(stats["f1"]["mean"] - 0.98) < 1e-9
    assert abs(stats["f1"]["std"] - 0.01) < 1e-9

    preproc = TextPreprocessor(REPO_ROOT / "data/resources/chaizi_map.json.gz")
    rec = SpamRecord(0, 1, "住（册）送礼", "住册送礼", "注册送礼", 1, "train", "MIXED", [1, 1, 0, 0])
    clean = to_clean_record(rec, preproc)
    assert clean.attack_type == "CLEAN"
    assert clean.raw_text == "注册送礼"
    assert sum(clean.obf_mask) == 0
    assert TrainConfig().cons_weight == 0.50


def test_inference_engine_bundle_strict_load_and_prediction() -> None:
    """Verify MVCSNInferenceEngine loads real demo_bundle.pt strictly and predicts accurately."""
    bundle_path = REPO_ROOT / "artifacts/demo_bundle.pt"
    chaizi_path = REPO_ROOT / "data/resources/chaizi_map.json.gz"
    if not bundle_path.exists():
        pytest.skip("demo_bundle.pt not present")

    bundle = load_demo_bundle(bundle_path)
    assert isinstance(bundle.vocab, CharVocab)
    assert bundle.bitmaps.ndim == 3 and bundle.bitmaps.shape[1:] == (32, 32)

    font_dir = REPO_ROOT / "fonts" if (REPO_ROOT / "fonts/unifont.otf").exists() else Path("/content/fonts")
    engine = MVCSNInferenceEngine.from_bundle(
        bundle_path=bundle_path, chaizi_path=chaizi_path, font_dir=font_dir
    )
    res_spam = engine.predict("特惠充100送200え，髙额返利稳赚不赔加微十Ꝗ领福利")
    res_ham = engine.predict("我和飞机上一百五十多名旅客眼巴巴期待飞机划破长空腾空而起的时刻")
    assert res_spam.pred_label == "SPAM" and res_spam.spam_prob > 0.9
    assert res_ham.pred_label == "HAM" and res_ham.spam_prob < 0.1
    assert engine.model._saved_base_buffers is None
