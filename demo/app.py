"""MV-CSN v3 Interactive Streamlit Inspector (Inference, Benchmarks, Graph Explorer)."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mvcsn.inference import MVCSNInferenceEngine
from scripts.gdrive_sync import bootstrap_from_gdrive

PRESET_EXAMPLES: Dict[str, str] = {
    "GLYPH (Cross-Script Lookalike Spam)": "特惠充100送200え，髙额返利稳赚不赔加微十Ꝗ领福利",
    "PHON (Homophone Evasion Spam)": "恭喜中大奖，专员联糸您领取现斤红包，请回复确认",
    "SPLIT (Radical Split & Symbol Bypass)": "氵青联\\/系客服讠主册即送88⑧元彩金，详询薇信",
    "MIXED (Multi-Attack Gambling/Loan)": "免抵押急用钱？Low利息秒到账，手几号讠主册立抢⑧折",
    "CLEAN Ham (Verified Test SMS)": "我和飞机上一百五十多名旅客眼巴巴期待飞机划破长空腾空而起的时刻",
}


def build_demo_engine(
    bundle_path: str | Path = "artifacts/demo_bundle.pt",
    chaizi_path: str | Path = "data/resources/chaizi_map.json.gz",
    font_dir: str | Path = "/content/fonts",
) -> Dict[str, object]:
    """Load MV-CSN v3 inference facade on CPU, auto-bootstrapping from Drive if missing."""
    bp = Path(bundle_path)
    cp = Path(chaizi_path)
    if not bp.exists() or not cp.exists():
        bootstrap_from_gdrive(font_dir=font_dir)

    engine = MVCSNInferenceEngine.from_bundle(
        bundle_path=bp, chaizi_path=cp, font_dir=font_dir
    )
    return engine.as_legacy_dict()


def predict_single_text(
    text: str, engine: Dict[str, object] | MVCSNInferenceEngine, max_len: int = 192
) -> Dict[str, object]:
    """Run MV-CSN v3 inference on one message via :class:`MVCSNInferenceEngine`."""
    facade = engine if isinstance(engine, MVCSNInferenceEngine) else engine["engine"]
    return facade.predict(text, max_len=max_len).to_dict()


def _load_json_safe(path: str | Path) -> Optional[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _render_char_heatmap_html(res: Dict[str, object]) -> str:
    spans: List[str] = []
    for ch, p_obf, ac, ap, ag in zip(
        res["chars"], res["obf_probs"], res["alpha_ctx"], res["alpha_phon"], res["alpha_glyph"]
    ):
        red = 255
        gb = int(max(70, 245 - p_obf * 175))
        bg = f"rgb({red},{gb},{gb})" if p_obf >= 0.35 else "rgb(240,244,248)"
        border = "2px solid #d9534f" if p_obf >= 0.5 else "1px solid #d0d7de"
        safe_ch = html.escape(ch)
        tip = f"obf={p_obf:.2f} | ctx={ac:.2f}, phon={ap:.2f}, glyph={ag:.2f}"
        spans.append(
            f"<span title='{tip}' style='display:inline-block;padding:4px 7px;margin:2px;"
            f"border-radius:6px;background:{bg};border:{border};color:#111;font-size:18px;'>{safe_ch}</span>"
        )
    return "<div style='line-height:2.2;'>" + "".join(spans) + "</div>"


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="MV-CSN v3: Adversarial Variant Spam Inspector", layout="wide")
    st.title("MV-CSN v3: Dual-Teacher Distilled Chinese Adversarial Spam Detection & Gate X-Ray")
    st.caption(
        "Multi-View Context-Gated Character Similarity Network (2.35M) | 32x32 Rendered-Glyph CAE + "
        "Phonetic Priors + Dynamic SpellGCN OOV Bridge + Dual-Teacher Cross-View KD"
    )

    @st.cache_resource
    def _cached_engine():
        return build_demo_engine()

    engine = _cached_engine()

    tab1, tab2, tab3 = st.tabs([
        "1. Live Variant Detector & Gate X-Ray",
        "2. Benchmarks, Ablations, Latency & Qwen3 LLM",
        "3. Dual-View Graph & 32x32 Glyph Inspector",
    ])

    with tab1:
        st.subheader("Live Adversarial Obfuscation Analysis")
        preset_key = st.selectbox("Choose a preset message (or edit freely below)", list(PRESET_EXAMPLES.keys()))
        user_text = st.text_area("Input Chinese SMS / Message", value=PRESET_EXAMPLES[preset_key], height=95)
        if user_text.strip():
            res = predict_single_text(user_text, engine)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction", res["pred_label"])
            c2.metric("Spam Probability", f"{res['spam_prob']:.4f}")
            c3.metric("OOV Characters Bridged", str(len(res["oov_resolved"])))
            c4.metric("Max Token Obfuscation Score", f"{max(res['obf_probs']):.4f}" if res["obf_probs"] else "0.0000")

            st.markdown(f"**Normalized / Chaizi-Recomposed Text:** `{res['proc_text']}`")
            if res["oov_resolved"]:
                st.info(f"Dynamically bridged unseen OOV tokens into train graph: {', '.join(res['oov_resolved'])}")

            st.markdown("**Token-Level Obfuscation Heatmap** (hover a character to inspect `[alpha_ctx, alpha_phon, alpha_glyph]`)")
            st.markdown(_render_char_heatmap_html(res), unsafe_allow_html=True)

            rows = [
                {
                    "pos": idx,
                    "char": ch,
                    "obf_prob": round(obf, 4),
                    "alpha_ctx": round(ac, 4),
                    "alpha_phon": round(ap, 4),
                    "alpha_glyph": round(ag, 4),
                }
                for idx, (ch, obf, ac, ap, ag) in enumerate(
                    zip(res["chars"], res["obf_probs"], res["alpha_ctx"], res["alpha_phon"], res["alpha_glyph"])
                )
            ]
            st.dataframe(rows, use_container_width=True, height=280)

    with tab2:
        st.subheader("Quantitative Benchmark, Cross-Domain, Latency & LLM Comparison Explorer")
        zs = _load_json_safe("results/zeroshot_real_variants.json")
        atk = _load_json_safe("results/per_attack_breakdown.json")
        ic = _load_json_safe("results/intensity_curve.json")
        abl = _load_json_safe("results/ablations.json")
        ext = _load_json_safe("results/external_cross_domain.json")
        lat = _load_json_safe("results/latency_benchmark.json")
        llm = _load_json_safe("results/llm_comparison.json")

        if zs:
            st.markdown("#### Track 1 (Clean-Only) & Track 2 (22K Dual-Teacher KD) — Benchmark B v2 (3 Seeds)")
            raw_sum = zs.get("raw_summary", {})
            ol_sum = zs.get("olivi9_summary", {})
            pr_sum = zs.get("probes_summary", {})
            zs_rows = [
                {
                    "Model": m,
                    "Params": zs.get("params", {}).get(m, 0),
                    "Fair F1": f"{s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}",
                    "Ham Acc": round(s["clean_ham_acc"]["mean"], 4),
                    "Raw F1": round(raw_sum.get(m, {}).get("f1", {}).get("mean", 0.0), 4),
                    "Olivi-9 F1": round(ol_sum.get(m, {}).get("f1", {}).get("mean", 0.0), 4),
                    "CF16": round(pr_sum.get(m, {}).get("orig_16_ham_acc", {}).get("mean", 0.0), 4),
                    "CF40": round(pr_sum.get(m, {}).get("balanced_40_acc", {}).get("mean", 0.0), 4),
                }
                for m, s in zs["summary"].items()
            ]
            st.dataframe(zs_rows, use_container_width=True)

        if atk and "summary" in atk:
            st.markdown("#### Per-Attack Breakdown (Benchmark B v2 Fair Test)")
            atk_rows = []
            for m, s in atk["summary"].items():
                row = {"Model": m}
                for key in ("CLEAN", "PHON", "MIXED", "SPACE", "SPLIT", "GLYPH"):
                    cell = s.get(key, {})
                    metric = cell.get("f1") or cell.get("accuracy") or cell.get("spam_recall") or {}
                    row[key] = round(metric.get("mean", 0.0), 4)
                atk_rows.append(row)
            st.dataframe(atk_rows, use_container_width=True)

        if abl:
            st.markdown("#### 12-Row Multi-Seed Ablation Study")
            deltas = abl.get("deltas_vs_full", {})
            raw_abl = abl.get("raw_summary", {})
            sym_abl = abl.get("symsub40_summary", {})
            olv_abl = abl.get("olivi9_summary", {})
            abl_rows = [
                {
                    "Variant": name,
                    "Fair F1": f"{s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}",
                    "ΔFair": round(deltas.get(name, {}).get("delta_fair_f1", 0.0), 4),
                    "Raw F1": round(raw_abl.get(name, {}).get("f1", {}).get("mean", 0.0), 4),
                    "SymSub40 F1": round(sym_abl.get(name, {}).get("f1", {}).get("mean", 0.0), 4),
                    "Olivi-9 F1": round(olv_abl.get(name, {}).get("f1", {}).get("mean", 0.0), 4),
                    "ΔOlivi-9": round(deltas.get(name, {}).get("delta_olivi9_f1", 0.0), 4),
                }
                for name, s in abl["summary"].items()
            ]
            st.dataframe(abl_rows, use_container_width=True)

        if llm and "mvcsn_v3_seed42" in llm:
            st.markdown("#### Qwen3-4B/8B LLM vs. MV-CSN v3 & Uncertainty Cascade (Seed 42 Comparison)")
            llm_rows = []
            for lbl, entry, esc in [
                ("MV-CSN v3 (Seed 42)", llm["mvcsn_v3_seed42"], 0.0),
                ("Qwen3-4B Zero-Shot", llm["qwen3_4b_zeroshot"], 1.0),
                ("Qwen3-4B 4-Shot ICL", llm["qwen3_4b_4shot_icl"], 1.0),
                ("Qwen3-4B Clean-LoRA (3 ep)", llm["qwen3_4b_clean_lora"], 1.0),
                ("Qwen3-8B Zero-Shot", llm["qwen3_8b_zeroshot"], 1.0),
            ]:
                llm_rows.append(
                    {
                        "Strategy": lbl,
                        "Bench B F1": round(entry["bench_b_v2"]["f1"], 4),
                        "Sym40 F1": round(entry["sym_sub_40"]["f1"], 4),
                        "Sym40 HamAcc": round(entry["sym_sub_40"]["clean_ham_acc"], 4),
                        "Olivi-9 F1": round(entry["olivi_9"]["f1"], 4),
                        "CF16": round(entry["cf16_ham_acc"], 4),
                        "Esc Rate": f"{esc * 100:.1f}%",
                    }
                )
            for cname, ctag in [("qwen4b_clean_lora", "4B LoRA"), ("qwen4b_zeroshot", "4B ZS")]:
                for tau, sw in llm.get("cascade_sweep", {}).get(cname, {}).items():
                    llm_rows.append(
                        {
                            "Strategy": f"Cascade v3 + {ctag} (τ={tau})",
                            "Bench B F1": round(sw["bench_b_v2"]["metrics"]["f1"], 4),
                            "Sym40 F1": round(sw["sym_sub_40"]["metrics"]["f1"], 4),
                            "Sym40 HamAcc": round(sw["sym_sub_40"]["metrics"]["clean_ham_acc"], 4),
                            "Olivi-9 F1": round(sw["olivi_9"]["metrics"]["f1"], 4),
                            "CF16": round(sw["cf16"]["acc"], 4),
                            "Esc Rate": f"{sw['bench_b_v2']['escalation_rate'] * 100:.2f}%",
                        }
                    )
            st.dataframe(llm_rows, use_container_width=True)

        if ext and "indomain_olivi9" in ext:
            st.markdown("#### Olivi-9 External Cross-Domain Zero-Shot vs. In-Domain (70/10/20)")
            ol_rows = []
            zs_olv = ext.get("zeroshot_olivi9", {}).get("summary", {})
            id_olv = ext.get("indomain_olivi9", {}).get("summary", {})
            for m, s in id_olv.items():
                ol_rows.append(
                    {
                        "Model": m,
                        "In-Domain Olivi-9 F1": round(s["f1"]["mean"], 4),
                        "In-Domain AUC": round(s["auc"]["mean"], 4),
                        "Zero-Shot F1 (if matched)": round(zs_olv.get(m, {}).get("f1", {}).get("mean", 0.0), 4),
                    }
                )
            st.dataframe(ol_rows, use_container_width=True)

        if lat and "models" in lat:
            st.markdown(f"#### Solo Zero-Contention Latency ({lat['hardware']['gpu']})")
            lat_rows = [
                {
                    "Model": m,
                    "Params": v["params"],
                    "GPU Batch (ms/msg)": round(v.get("gpu_bs64_ms_per_msg") or v.get("gpu_bs32_ms_per_msg") or 0.0, 4),
                    "GPU bs=1 Eager (ms)": round(v["gpu_bs1_eager_ms_per_msg"], 3),
                    "GPU bs=1 Compile (ms)": round(v["gpu_bs1_compile_ms_per_msg"], 3) if v.get("gpu_bs1_compile_ms_per_msg") else None,
                    "CPU bs=1 (ms)": round(v["cpu_bs1_ms_per_msg"], 3) if v.get("cpu_bs1_ms_per_msg") else None,
                }
                for m, v in lat["models"].items()
            ]
            st.dataframe(lat_rows, use_container_width=True)

        if ic and "summary" in ic:
            st.markdown("#### Symmetric Perturbation Intensity Curve (0% -> 60% Attack Rate)")
            mode = st.selectbox("Perturbation Mode", [k for k in ("sub_only", "mixed") if k in ic["summary"]])
            chart_dict = {
                m: {f"{float(r)*100:.0f}%": ic["summary"][mode][m][str(r)]["f1"]["mean"] for r in ic["rates"]}
                for m in ic["summary"][mode]
            }
            st.line_chart(chart_dict)

    with tab3:
        st.subheader("Dual-View Neighbor & 32x32 Rendered-Glyph Inspector")
        query_ch = st.text_input("Inspect any CJK / Homoglyph / Lookalike character (e.g., 髙, え, Ꝗ, 元, 稳, 茺)", value="え", max_chars=2)
        ch = query_ch.strip()[:1] if query_ch.strip() else "え"
        vocab = engine["vocab"]
        graphs = engine["graphs"]
        bridge = engine["bridge"]
        bitmaps = engine["bitmaps"]

        in_vocab = ch in vocab.stoi
        if in_vocab:
            cid = vocab.stoi[ch]
            bmp = bitmaps[cid].numpy()
            phon_idx = graphs["phon_nbr_idx"][cid].tolist()
            phon_sim = graphs["phon_nbr_sim"][cid].tolist()
            glyph_idx = graphs["glyph_nbr_idx"][cid].tolist()
            glyph_sim = graphs["glyph_nbr_sim"][cid].tolist()
        else:
            oov_map, ext_b = bridge.build_extended_bundle([ch])
            arr, _ = bridge.renderer.render(ch)
            bmp = arr
            if ch in oov_map:
                oid = oov_map[ch]
                phon_idx = ext_b["phon_nbr_idx"][oid].tolist()
                phon_sim = ext_b["phon_nbr_sim"][oid].tolist()
                glyph_idx = ext_b["glyph_nbr_idx"][oid].tolist()
                glyph_sim = ext_b["glyph_nbr_sim"][oid].tolist()
            else:
                phon_idx, phon_sim, glyph_idx, glyph_sim = [], [], [], []

        col_img, col_p, col_g = st.columns([1, 2, 2])
        with col_img:
            st.markdown(f"**Char:** `{ch}` ({'In-Vocab' if in_vocab else 'Dynamic OOV'})")
            st.image((bmp * 255).clip(0, 255).astype(np.uint8), caption="32x32 Glyph Bitmap", width=128)
        with col_p:
            st.markdown("**Top Phonetic Neighbors (View 1)**")
            p_rows = [
                {"neighbor": vocab.itos[i], "sim": round(float(s), 4)}
                for i, s in zip(phon_idx, phon_sim)
                if i > 1 and s > 0
            ][:10]
            st.dataframe(p_rows, use_container_width=True)
        with col_g:
            st.markdown("**Top Rendered-Glyph Neighbors (View 2)**")
            g_rows = [
                {"neighbor": vocab.itos[i], "sim": round(float(s), 4)}
                for i, s in zip(glyph_idx, glyph_sim)
                if i > 1 and s > 0
            ][:10]
            st.dataframe(g_rows, use_container_width=True)


if __name__ == "__main__":
    main()
