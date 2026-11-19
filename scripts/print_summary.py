"""Zero-dependency summary printer for all verified MV-CSN v2 experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load(name: str) -> Dict[str, Any]:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Print formatted summary tables from all verified ``results/*.json`` files."""
    zs = _load("zeroshot_real_variants.json")
    atk = _load("per_attack_breakdown.json")
    llm = _load("llm_comparison.json")
    abl = _load("ablations.json")
    gate = _load("gate_analysis.json")
    err = _load("error_analysis.json")

    print("=" * 84)
    print(" MV-CSN v2 · Verified Experimental Summary (Benchmark B: SMS-Matched 16.5K, 3 Seeds)")
    print("=" * 84)

    print("\n[Exp 2] Zero-Shot Clean -> RealVariant (n=3,395 unseen adversarial test SMS)")
    print(f"  {'Model':22s} | {'F1 (mean±std)':17s} | {'AUC':6s} | {'HamAcc':6s} | {'GLYPH':6s} | {'SPLIT':6s}")
    print("  " + "-" * 76)
    for model in ("LZASD", "RBT3", "TFIDF-SVM", "MV-CSN v2", "RBT3 + MV-CSN v2"):
        s = zs["summary"][model]
        a = atk[model]
        print(
            f"  {model:22s} | {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}   | "
            f"{s['auc']['mean']:.4f} | {s['clean_ham_acc']['mean']:.4f} | "
            f"{a['GLYPH']['mean']:.4f} | {a['SPLIT']['mean']:.4f}"
        )

    print("\n[Exp 6] Open-Weight Qwen3 LLMs vs. MV-CSN v2 & Uncertainty Cascade")
    print(f"  {'Strategy':31s} | {'Params':>13s} | {'ms/SMS':>7s} | {'F1':6s} | {'HamAcc':6s}")
    print("  " + "-" * 76)
    order = [
        "Qwen3-0.6B (Zero-Shot)",
        "Qwen3-4B-Instruct (Zero-Shot)",
        "Qwen3-4B-Instruct (4-Shot ICL)",
        "Qwen3-0.6B (Clean-LoRA)",
        "MV-CSN v2",
        "RBT3 + MV-CSN v2",
        "MV-CSN v2 + Qwen3-4B Cascade",
    ]
    for name in order:
        if name not in llm["summary"]:
            continue
        s = llm["summary"][name]
        params = int(llm["params"].get(name, 0))
        ms = float(s.get("infer_ms_per_sample", {}).get("mean", 0.0))
        print(
            f"  {name:31s} | {params:13,d} | {ms:7.3f} | "
            f"{s['f1']['mean']:.4f} | {s['clean_ham_acc']['mean']:.4f}"
        )

    print("\n[Exp 4] 8-Row Multi-Seed Ablation Study (Delta F1 vs. Full MV-CSN v2)")
    for name, s in abl["summary"].items():
        print(
            f"  {name:30s} : F1={s['f1']['mean']:.4f} ± {s['f1']['std']:.4f} "
            f"(dF1={s['delta_f1']:+.4f})"
        )

    print("\n[Exp 5] Token-Level Context Gate Distribution [alpha_ctx, alpha_phon, alpha_glyph]")
    for key in ("clean", "phon_obf", "glyph_obf"):
        vals = gate["means"][key]
        cnt = gate["counts"][key]
        print(
            f"  {key:10s} (n={cnt:5,d}) : "
            f"ctx={vals[0]:.3f}  phon={vals[1]:.3f}  glyph={vals[2]:.3f}"
        )

    es = err["error_summary"]
    print("\n[Exp 7] Full CPU Error Analysis (Seed-42 MV-CSN v2 on 3,395 Test SMS)")
    print(
        f"  Total errors: {es['total_errors']}/{err['meta']['test_total']} "
        f"({es['overall_error_rate']*100:.2f}%) | FP={es['false_positives']} | FN={es['false_negatives']} | "
        f"Uncertainty band (0.05..0.95) captures {es['errors_in_uncertainty_band']}/{es['total_errors']} "
        f"({es['error_capture_rate_by_uncertainty_band']*100:.1f}%) of all errors"
    )
    print("=" * 84)


if __name__ == "__main__":
    main()
