#!/usr/bin/env bash
# Sanity check for within_group_stats: same seed, 2 steps, logging on vs off.
# The pre-clip gradient norm should agree (the option only changes how the backward passes are split);
# the time_train_s difference is the logging overhead.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON="${PYTHON:-$HOME/anaconda3/envs/rlcmp/bin/python}"
OUT="${OUT:-/tmp/cfrl_within_check}"
for flag in false true; do
  "$PYTHON" -m cfrl.train --config configs/base.yaml estimator=grpo seed=0 num_steps=2 eval_every=0 \
    final_eval_size=8 within_group_stats=$flag output_dir="$OUT" run_name=within_$flag overwrite=true
done
"$PYTHON" - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows = {f: [json.loads(l) for l in (out / f"within_{f}" / "metrics.jsonl").open()] for f in ("false", "true")}
for a, b in zip(rows["false"], rows["true"]):
    print(f"step {a['step']}: gnorm off={a['grad_norm_preclip']:.6g} on={b['grad_norm_preclip']:.6g} "
          f"rel.diff={abs(a['grad_norm_preclip'] - b['grad_norm_preclip']) / a['grad_norm_preclip']:.2e} | "
          f"reward off={a['reward_mean']:.3f} on={b['reward_mean']:.3f} | "
          f"train_s off={a['time_train_s']:.1f} on={b['time_train_s']:.1f} | mem on={b['peak_mem_gb']:.1f}G")
    print("  within keys:", {k: round(v, 6) for k, v in b.items() if "within" in k})
PY
