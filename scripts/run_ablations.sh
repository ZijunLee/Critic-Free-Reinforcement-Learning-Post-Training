#!/usr/bin/env bash
# Ablations on GRPO (seed 0 by default): length normalisation and format-reward shaping.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON="${PYTHON:-$HOME/anaconda3/envs/rlcmp/bin/python}"   # dedicated conda env
mkdir -p logs
SEEDS=("$@"); [ $# -eq 0 ] && SEEDS=(0)
for seed in "${SEEDS[@]}"; do
  declare -A VARIANTS=(
    [grpo-seqmean]="estimator=grpo loss_agg=seq_mean"
    [grpo-format0.2]="estimator=grpo format_weight=0.2"
    [rloo-format0.2]="estimator=rloo format_weight=0.2"
  )
  for v in "${!VARIANTS[@]}"; do
    name="${v}_s${seed}"
    if [ -f "runs/$name/summary.json" ]; then echo "skip $name"; continue; fi
    "$PYTHON" -m cfrl.train --config configs/base.yaml ${VARIANTS[$v]} seed=$seed run_name=$name overwrite=true \
      2>&1 | tee "logs/$name.log"
  done
done
"$PYTHON" -m cfrl.analyze runs --out results
