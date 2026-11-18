#!/usr/bin/env bash
# Main matrix: 6 estimators x seeds. Finished runs (summary.json present) are skipped, so the
# script can be stopped and restarted (e.g. one batch per night).
# Usage: scripts/run_main.sh [seeds...]      default seeds: 0 1 2
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON="${PYTHON:-$HOME/anaconda3/envs/rlcmp/bin/python}"   # dedicated conda env
SEEDS=("$@"); [ $# -eq 0 ] && SEEDS=(0 1 2)
mkdir -p logs
for seed in "${SEEDS[@]}"; do
  for est in grpo rloo grpo_nostd remax reinforce_ema reinforce; do
    name="${est}_s${seed}"
    if [ -f "runs/$name/summary.json" ]; then echo "skip $name"; continue; fi
    "$PYTHON" -m cfrl.train --config configs/base.yaml estimator=$est seed=$seed overwrite=true 2>&1 | tee "logs/$name.log"
  done
done
"$PYTHON" -m cfrl.analyze runs --out results
