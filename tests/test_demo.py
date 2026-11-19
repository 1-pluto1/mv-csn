"""Tests for Google Drive sync, Demo inference engine, results schemas, and Streamlit AppTest."""

from __future__ import annotations

import json
import math
from pathlib import Path

from demo.app import build_demo_engine, predict_single_text
from scripts.gdrive_sync import bootstrap_from_gdrive, persist_to_gdrive


def test_gdrive_sync_roundtrip(tmp_path: Path) -> None:
    src_repo = tmp_path / "src_repo"
    src_fonts = tmp_path / "src_fonts"
    drive_dir = tmp_path / "drive" / "SYSU_DSE_MVCSN"
    dst_repo = tmp_path / "dst_repo"
    dst_fonts = tmp_path / "dst_fonts"

    (src_repo / "data" / "resources").mkdir(parents=True)
    (src_repo / "artifacts").mkdir(parents=True)
    (src_repo / "results").mkdir(parents=True)
    src_fonts.mkdir(parents=True)

    (src_repo / "data" / "resources" / "sample.txt").write_text("chaizi", encoding="utf-8")
    (src_repo / "artifacts" / "vocab.json").write_text("{}", encoding="utf-8")
    (src_repo / "results" / "summary.json").write_text('{"f1": 0.99}', encoding="utf-8")
    (src_fonts / "font.otf").write_bytes(b"FONT")

    persist_to_gdrive(drive_dir=drive_dir, repo_root=src_repo, font_dir=src_fonts)
    bootstrap_from_gdrive(drive_dir=drive_dir, repo_root=dst_repo, font_dir=dst_fonts)

    assert (dst_repo / "data" / "resources" / "sample.txt").read_text(encoding="utf-8") == "chaizi"
    assert (dst_repo / "artifacts" / "vocab.json").read_text(encoding="utf-8") == "{}"
    assert (dst_repo / "results" / "summary.json").read_text(encoding="utf-8") == '{"f1": 0.99}'
    assert (dst_fonts / "font.otf").read_bytes() == b"FONT"


def test_demo_engine_inference_and_oov() -> None:
    engine = build_demo_engine()
    spam_res = predict_single_text("特惠充100送200え，髙额返利加微十Ꝗ领福利", engine)
    ham_res = predict_single_text('我和飞机上一百五十多名旅客眼巴巴期待飞机划破长空腾空而起的时刻', engine)

    assert 0.0 <= spam_res["spam_prob"] <= 1.0
    assert 0.0 <= ham_res["spam_prob"] <= 1.0
    assert spam_res["spam_prob"] > 0.5
    assert ham_res["spam_prob"] < 0.5
    assert len(spam_res["chars"]) == len(spam_res["obf_probs"]) == len(spam_res["alpha_ctx"])
    for ac, ap, ag in zip(spam_res["alpha_ctx"], spam_res["alpha_phon"], spam_res["alpha_glyph"]):
        assert math.isclose(ac + ap + ag, 1.0, abs_tol=1e-3)
    # At least one rare glyph token resolves dynamically via OOVGraphBridge
    assert any(ch in spam_res["oov_resolved"] for ch in ("Ꝗ", "え")) or "髙" in engine["vocab"].stoi


def test_results_json_schemas() -> None:
    expected = [
        "in_dist_results.json",
        "zeroshot_real_variants.json",
        "per_attack_breakdown.json",
        "intensity_curve.json",
        "ablations.json",
        "llm_comparison.json",
        "error_analysis.json",
    ]
    for fn in expected:
        p = Path("results") / fn
        assert p.exists(), f"Missing {p}"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    zs = json.loads(Path("results/zeroshot_real_variants.json").read_text(encoding="utf-8"))
    abl = json.loads(Path("results/ablations.json").read_text(encoding="utf-8"))
    llm = json.loads(Path("results/llm_comparison.json").read_text(encoding="utf-8"))
    for block in (zs["summary"], abl["summary"], llm["summary"]):
        for m, stats in block.items():
            assert not math.isnan(stats["f1"]["mean"]), f"NaN F1 in {m}"


def test_streamlit_app_headless() -> None:
    import pytest

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "demo" / "app.py")).run(timeout=60)
    assert not at.exception
