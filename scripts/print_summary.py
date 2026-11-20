"""Zero-dependency summary printer for all verified MV-CSN v3 Stage 3 results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load(name: str) -> Optional[Dict[str, Any]]:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _atk_score(row: Dict[str, Any], key: str) -> float:
    cell = row.get(key, {})
    metric = cell.get("f1") or cell.get("accuracy") or cell.get("spam_recall") or {}
    return float(metric.get("mean", 0.0))


def main() -> None:
    """Print formatted summary tables from all 11 canonical ``results/*.json`` files."""
    zs = _load("zeroshot_real_variants.json")
    atk = _load("per_attack_breakdown.json")
    cjk = _load("pure_cjk_track.json")
    ic = _load("intensity_curve.json")
    abl = _load("ablations.json")
    ind = _load("in_dist_results.json")
    ext = _load("external_cross_domain.json")
    gate = _load("gate_analysis.json")
    err = _load("error_analysis.json")
    lat = _load("latency_benchmark.json")
    llm = _load("llm_comparison.json")

    if zs is None:
        raise FileNotFoundError("Missing results/zeroshot_real_variants.json")

    meta = zs["dataset_meta"]
    print("=" * 102)
    print(
        f" MV-CSN v3 · Stage 3 Verified Summary "
        f"(Bench B v2: {meta['total_rows']:,} rows, {meta['num_clusters']:,} clusters, "
        f"pool={zs.get('transfer_pool_size', 0):,}, seeds={zs['seeds']})"
    )
    print("=" * 102)

    for title, track_key in [
        ("[Exp 2 · Track 1] Strict Clean-Only (7 Models, n=3,216 Bench B v2 Test)", "track1_clean_only"),
        ("[Exp 2 · Track 2] Same 22K Dual-Teacher KD / Data+Aug (5 Models, 3 Seeds)", "track2_dual_teacher_kd"),
    ]:
        print(f"\n{title}")
        print(
            f"  {'Model':34s} | {'Params':>11s} | {'Fair F1 (mean±std)':17s} | "
            f"{'HamAcc':6s} | {'RawF1':6s} | {'PureCJK':7s} | {'Olivi9':6s} | {'CF16':6s} | {'CF40':6s}"
        )
        print("  " + "-" * 116)
        for m, s in zs[track_key].items():
            p = int(zs["params"].get(m, 0))
            raw_f1 = zs["raw_summary"].get(m, {}).get("f1", {}).get("mean", 0.0)
            cjk_f1 = (cjk or {}).get("summary", {}).get(m, {}).get("f1", {}).get("mean", 0.0)
            ol_f1 = zs["olivi9_summary"].get(m, {}).get("f1", {}).get("mean", 0.0)
            pr = zs["probes_summary"].get(m, {})
            cf16 = pr.get("orig_16_ham_acc", {}).get("mean", 0.0)
            cf40 = pr.get("balanced_40_acc", {}).get("mean", 0.0)
            print(
                f"  {m:34s} | {p:11,d} | {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}  | "
                f"{s['clean_ham_acc']['mean']:.4f} | {raw_f1:.4f} | {cjk_f1:7.4f} | "
                f"{ol_f1:.4f} | {cf16:.4f} | {cf40:.4f}"
            )

    if atk:
        print("\n[Per-Attack Breakdown] Bench B v2 Test F1 / Accuracy (CLEAN / PHON / MIXED / SPACE / SPLIT / GLYPH)")
        print(f"  {'Model':34s} | {'CLEAN':6s} | {'PHON':6s} | {'MIXED':6s} | {'SPACE':6s} | {'SPLIT':6s} | {'GLYPH':6s}")
        print("  " + "-" * 86)
        for m, row in atk["summary"].items():
            vals = [_atk_score(row, k) for k in ("CLEAN", "PHON", "MIXED", "SPACE", "SPLIT", "GLYPH")]
            print(f"  {m:34s} | " + " | ".join(f"{v:.4f}" for v in vals))

    if ic:
        print("\n[Exp 3 · Symmetric Stress Curves] (rates 0.0 / 0.2 / 0.4 / 0.5 / 0.6)")
        key_rates = ["0.0", "0.2", "0.4", "0.5", "0.6"]
        for mode in ("sub_only", "mixed"):
            if mode not in ic["summary"]:
                continue
            print(f"  -- Mode: {mode} --")
            for m, rmap in ic["summary"][mode].items():
                pts = "  ".join(f"{r}:{rmap[r]['f1']['mean']:.4f}" for r in key_rates if r in rmap)
                print(f"    {m:32s} : {pts}")

    if abl:
        print("\n[Exp 4 · 12-Row Multi-Seed Ablations] (Mean across 3 seeds + Delta vs Full)")
        print(f"  {'Variant':35s} | {'Fair F1 (dFair)':18s} | {'Raw F1 (dRaw)':16s} | {'Sym40 (dSym40)':16s} | {'Olivi9 (dOlv9)':16s}")
        print("  " + "-" * 112)
        for name, s in abl["summary"].items():
            d = abl["deltas_vs_full"][name]
            rf = abl["raw_summary"][name]["f1"]["mean"]
            sf = abl["symsub40_summary"][name]["f1"]["mean"]
            of = abl["olivi9_summary"][name]["f1"]["mean"]
            print(
                f"  {name:35s} | {s['f1']['mean']:.4f} ({d['delta_fair_f1']:+.4f})   | "
                f"{rf:.4f} ({d['delta_raw_f1']:+.4f}) | {sf:.4f} ({d['delta_symsub40_f1']:+.4f}) | "
                f"{of:.4f} ({d['delta_olivi9_f1']:+.4f})"
            )

    if ind or ext:
        print("\n[Exp 1 & 8b · In-Distribution & Olivi-9 In-Domain (3 Seeds)]")
        if ind:
            for bname in ("bench_a_original", "bench_b_v2_sms_matched"):
                if bname in ind:
                    row = "  " + bname + ": " + ", ".join(
                        f"{m}={v['f1']['mean']:.4f} (AUC={v['auc']['mean']:.4f})"
                        for m, v in ind[bname]["summary"].items()
                    )
                    print(row)
        if ext and "indomain_olivi9" in ext:
            row = "  olivi9_indomain_70_10_20: " + ", ".join(
                f"{m}={v['f1']['mean']:.4f} (AUC={v['auc']['mean']:.4f})"
                for m, v in ext["indomain_olivi9"]["summary"].items()
            )
            print(row)

    if gate:
        print(f"\n[Exp 5 · Token-Level Context Gate Distribution] (total_tokens={gate['total_tokens']:,})")
        for bname, b in gate["buckets"].items():
            cnt = int(b["token_count"])
            if cnt == 0:
                print(f"  {bname:28s} : n={cnt:,}")
            else:
                print(
                    f"  {bname:28s} (n={cnt:7,d}) : "
                    f"ctx={b['alpha_ctx_mean']:.4f}  phon={b['alpha_phon_mean']:.4f}  "
                    f"glyph={b['alpha_glyph_mean']:.4f}  (graph_total={b['alpha_graph_total_mean']:.4f})"
                )

    if lat:
        hw = lat["hardware"]
        print(f"\n[Latency Benchmark] ({hw['gpu']}, cpu_threads={hw['cpu_threads']}, solo={hw['solo_zero_contention']})")
        for m, v in lat["models"].items():
            batch_ms = v.get("gpu_bs64_ms_per_msg") or v.get("gpu_bs32_ms_per_msg")
            b_lbl = "bs64" if "gpu_bs64_ms_per_msg" in v else "bs32"
            comp = f"{v['gpu_bs1_compile_ms_per_msg']:.3f}ms" if v.get("gpu_bs1_compile_ms_per_msg") else "—"
            cpu = f"{v['cpu_bs1_ms_per_msg']:.3f}ms" if v.get("cpu_bs1_ms_per_msg") else "—"
            print(
                f"  {m:28s} | {v['params']:13,d} | {b_lbl}={batch_ms:.4f}ms | "
                f"bs1_eager={v['gpu_bs1_eager_ms_per_msg']:.3f}ms | compile={comp:8s} | cpu_bs1={cpu}"
            )

    if llm:
        print("\n[Exp 6 · Qwen3-4B/8B & Uncertainty Cascade (Seed 42 Comparison)]")
        print(
            f"  {'Strategy':34s} | {'BenchB F1 (Ham)':17s} | {'Sym40 F1 (Ham)':17s} | "
            f"{'Olivi9 F1 (Ham)':17s} | {'CF16/CF40':11s} | {'Esc%':6s}"
        )
        print("  " + "-" * 116)
        single_rows = [
            ("MV-CSN v3 (Seed 42)", llm["mvcsn_v3_seed42"], 0.0),
            ("Qwen3-4B Zero-Shot", llm["qwen3_4b_zeroshot"], 1.0),
            ("Qwen3-4B 4-Shot ICL", llm["qwen3_4b_4shot_icl"], 1.0),
            ("Qwen3-4B Clean-LoRA (3 ep)", llm["qwen3_4b_clean_lora"], 1.0),
            ("Qwen3-8B Zero-Shot", llm["qwen3_8b_zeroshot"], 1.0),
        ]
        for label, item, esc in single_rows:
            bb, s40, ol = item["bench_b_v2"], item["sym_sub_40"], item["olivi_9"]
            print(
                f"  {label:34s} | {bb['f1']:.4f} ({bb['clean_ham_acc']*100:5.2f}%)  | "
                f"{s40['f1']:.4f} ({s40['clean_ham_acc']*100:5.2f}%)  | "
                f"{ol['f1']:.4f} ({ol['clean_ham_acc']*100:5.2f}%)  | "
                f"{item['cf16_ham_acc']:.2f}/{item['cf40_acc']:.2f}  | {esc*100:5.1f}%"
            )
        for cname, ctag in [("qwen4b_clean_lora", "4B LoRA"), ("qwen4b_zeroshot", "4B ZS")]:
            for tau in ("0.30", "0.40", "0.45"):
                sw = llm["cascade_sweep"][cname][tau]
                bb, s40, ol = sw["bench_b_v2"], sw["sym_sub_40"], sw["olivi_9"]
                bm, sm, om = bb["metrics"], s40["metrics"], ol["metrics"]
                print(
                    f"  {f'Cascade v3 + {ctag} (tau={tau})':34s} | "
                    f"{bm['f1']:.4f} ({bm['clean_ham_acc']*100:5.2f}%)  | "
                    f"{sm['f1']:.4f} ({sm['clean_ham_acc']*100:5.2f}%)  | "
                    f"{om['f1']:.4f} ({om['clean_ham_acc']*100:5.2f}%)  | "
                    f"{sw['cf16']['acc']:.2f}/{sw['cf40']['acc']:.2f}  | {bb['escalation_rate']*100:5.2f}%"
                )

    if err:
        cf_right = sum(1 for c in err["counterfactual_16_detail"] if c["mvcsn_v3_correct"])
        cf_tot = len(err["counterfactual_16_detail"])
        print(
            f"\n[Exp 7 · Error Analysis (Seed 42)] n_test={err['n_test']:,} | "
            f"FP={err['n_fp']} {err['fp_by_attack']} | FN={err['n_fn']} {err['fn_by_attack']} | "
            f"CF16={cf_right}/{cf_tot}"
        )
    print("=" * 102)


if __name__ == "__main__":
    main()
