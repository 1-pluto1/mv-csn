#!/usr/bin/env bash
set -euo pipefail
cd /content/mv-csn
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "=== [START] Stage 3 Runner 1 (Track 1) $(date -u) ==="
python3 scripts/run_v3_stage3_track1.py 2>&1 | tee logs/stage3_r1.log
echo "R1_DONE $(date -u)" >> logs/stage3_progress.txt

echo "=== [START] Stage 3 Runner 2 (Track 2 + Merge + Analysis + Latency) $(date -u) ==="
python3 scripts/run_v3_stage3_track2_and_analysis.py 2>&1 | tee logs/stage3_r2.log
echo "R2_DONE $(date -u)" >> logs/stage3_progress.txt

echo "=== [START] Stage 3 Runner 3 (12-Row Ablations + In-Dist + Olivi-9 In-Domain) $(date -u) ==="
python3 scripts/run_v3_stage3_ablations_indist.py 2>&1 | tee logs/stage3_r3.log
echo "R3_DONE $(date -u)" >> logs/stage3_progress.txt

echo "=== [START] Stage 3 Runner 4 (Qwen3-4B ZS/ICL/LoRA + Cascade) $(date -u) ==="
python3 scripts/run_v3_stage3_llm_eval.py 2>&1 | tee logs/stage3_r4.log
echo "R4_DONE $(date -u)" >> logs/stage3_progress.txt

echo "STAGE3_ALL_COMPLETE $(date -u)" | tee logs/stage3_done.status
