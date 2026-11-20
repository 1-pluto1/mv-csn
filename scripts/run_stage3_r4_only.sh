#!/usr/bin/env bash
set -euo pipefail
cd /content/mv-csn
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=== [START] Stage 3 Runner 4 (Qwen3-4B ZS/ICL/LoRA + Cascade) $(date -u) ==="
python3 scripts/run_v3_stage3_llm_eval.py 2>&1 | tee logs/stage3_r4.log
echo "R4_DONE $(date -u)" >> logs/stage3_progress.txt
echo "STAGE3_ALL_COMPLETE $(date -u)" | tee logs/stage3_done.status
