#!/usr/bin/env bash
# ==============================================================================
# MV-CSN v2 One-Click Reproduction & Verification Script
# Sun Yat-sen University · Data Science & Engineering Capstone
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

step() {
  printf "\n\033[1;32m==>\033[0m \033[1m%s\033[0m\n" "$1"
}

warn() {
  printf "\033[1;33m[WARN]\033[0m %s\n" "$1"
}

usage() {
  cat <<'EOF'
Usage: ./reproduce.sh <mode>

Modes:
  verify          [Default, CPU ~10s] Syntax check + pytest suite + live CPU
                  inference showcase (using committed demo_bundle.pt) + results summary.
  setup           Install Python dependencies and editable mvcsn package.
  summary         Print formatted verified tables directly from results/*.json (instant).
  demo            Launch the 3-tab interactive Streamlit Web Inspector on CPU.
  error-analysis  Reproduce Exp 7 CPU error analysis (3,395 test SMS -> results/error_analysis.json).
  full            Full end-to-end GPU reproduction (Exp 1-5 + Exp 6 Qwen3 LLMs + Exp 7 + tests).
  --help, -h      Show this help message.

Environment overrides:
  PYTHON_BIN      Python interpreter to use (default: python3)
EOF
}

cmd_setup() {
  step "Installing Python dependencies and editable mvcsn package"
  "$PYTHON_BIN" -m pip install -r requirements.txt
  "$PYTHON_BIN" -m pip install -e . --no-deps
  step "Setup complete. Run './reproduce.sh verify' to validate."
}

cmd_summary() {
  "$PYTHON_BIN" scripts/print_summary.py
}

cmd_verify() {
  step "1/4 Checking Python syntax across all repository modules"
  "$PYTHON_BIN" -m py_compile $(find mvcsn baselines demo scripts tests -name '*.py')
  echo "  All .py files compiled cleanly."

  step "2/4 Running automated unit & invariant test suite (pytest)"
  "$PYTHON_BIN" -m pytest -v tests/

  step "3/4 Running live CPU inference showcase (artifacts/demo_bundle.pt)"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from mvcsn import MVCSNInferenceEngine

font_dir = Path("fonts") if Path("fonts/unifont.otf").exists() else Path("assets/fonts")
if not (font_dir / "unifont.otf").exists():
    font_dir = Path("/content/fonts")

engine = MVCSNInferenceEngine.from_bundle(
    bundle_path="artifacts/demo_bundle.pt",
    chaizi_path="data/resources/chaizi_map.json.gz",
    font_dir=font_dir,
)
samples = [
    ("GLYPH", "特惠充100送200え，髙额返利稳赚不赔加微十Ꝗ领福利"),
    ("SPLIT", "氵青联\\/系客服讠主册即送88⑧元彩金，详询薇信"),
    ("HAM  ", "我和飞机上一百五十多名旅客眼巴巴期待飞机划破长空腾空而起的时刻"),
]
for tag, txt in samples:
    r = engine.predict(txt)
    print(f"  [{tag}] pred={r.pred_label:4s} P(spam)={r.spam_prob:.4f} | OOV={r.oov_resolved} | proc={r.proc_text}")
PY

  step "4/4 Verified experimental summary (results/*.json)"
  cmd_summary
}

cmd_demo() {
  step "Launching MV-CSN v2 Interactive Streamlit Inspector"
  exec "$PYTHON_BIN" -m streamlit run demo/app.py
}

cmd_error_analysis() {
  if [[ ! -f "data/resources/variant_spam_sms_matched.tsv.gz" ]]; then
    step "Downloading datasets & character resources"
    "$PYTHON_BIN" scripts/download_data.py
  fi
  if [[ ! -f "artifacts/splits_sms_matched.json.gz" ]]; then
    step "Building deterministic leakage-free Benchmark B splits"
    "$PYTHON_BIN" -c 'from mvcsn.data.splits import build_or_load_splits; build_or_load_splits(cache_path="artifacts/splits_sms_matched.json.gz", dataset_path="data/resources/variant_spam_sms_matched.tsv.gz")'
  fi
  step "Running Exp 7 CPU Error Analysis across all 3,395 Benchmark B test SMS"
  FONT_ARG="fonts"
  [[ -f "fonts/unifont.otf" ]] || FONT_ARG="assets/fonts"
  "$PYTHON_BIN" scripts/run_error_analysis.py --font-dir "$FONT_ARG"
}

cmd_full() {
  step "Checking GPU availability for full multi-seed reproduction"
  if ! "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    warn "No CUDA GPU detected! Full reproduction (Exp 1-6 incl. Qwen3-4B) requires a CUDA GPU (~1.5h on A100/L4)."
    warn "For fast CPU verification without retraining, run: ./reproduce.sh verify"
    exit 1
  fi

  step "Stage 1/5: Data & Knowledge Resource Acquisition (Bench A + B, chaizi, SSC)"
  "$PYTHON_BIN" scripts/download_data.py

  step "Stage 2/5: Exp 1-5 (In-Dist, Zero-Shot, Intensity Curve, 8-Row Ablations, Gate X-Ray)"
  "$PYTHON_BIN" scripts/run_all.py

  step "Stage 3/5: Exp 6 (Qwen3-0.6B/4B Zero-Shot, 4-Shot ICL, Clean-LoRA, Cascade)"
  "$PYTHON_BIN" scripts/run_llm_eval.py

  step "Stage 4/5: Exp 7 (Full Error & Failure-Case Analysis)"
  cmd_error_analysis

  step "Stage 5/5: Full Verification & Summary"
  cmd_verify
}

MODE="${1:-verify}"
case "$MODE" in
  verify)          cmd_verify ;;
  setup)           cmd_setup ;;
  summary)         cmd_summary ;;
  demo)            cmd_demo ;;
  error-analysis)  cmd_error_analysis ;;
  full)            cmd_full ;;
  -h|--help|help)  usage ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac
