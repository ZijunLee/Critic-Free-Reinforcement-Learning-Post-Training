#!/usr/bin/env bash
# Pilot: 2 short runs to check speed, memory and learning-rate sanity before the full matrix.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON="${PYTHON:-$HOME/anaconda3/envs/rlcmp/bin/python}"   # dedicated conda env
for est in grpo reinforce; do
  "$PYTHON" -m cfrl.train --config configs/base.yaml estimator=$est seed=0 num_steps=40 eval_every=20 \
    eval_size=200 final_eval_size=200 output_dir=runs_pilot overwrite=true 2>&1 | tee runs_pilot_$est.log
done
"$PYTHON" -m cfrl.analyze runs_pilot --out results_pilot
