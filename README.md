# Critic-free RL for LLM reasoning: a controlled comparison of baselines

**Question.** REINFORCE, ReMax, RLOO and GRPO are all the same policy-gradient estimator
`-A · ∇log π(y|x)`; they differ only in how the advantage `A` is formed from rewards (which
baseline is subtracted, and whether it is rescaled). How much do those choices matter for final
accuracy, training stability and compute, when everything else is held fixed?

**Setup.** Qwen2.5-0.5B-Instruct + LoRA, GSM8K with a verifiable reward (numeric match of the
last `\boxed{}`, using Qwen's own chain-of-thought math prompt), one 12 GB GPU (RTX 5070).
All estimators share the rollout code, sampling budget
(16 prompts × 8 samples per step), loss, optimiser and schedule; `src/cfrl/advantages.py` is the
only place where they differ.

**Short answer.** Use a baseline, but not an expensive one. GRPO and RLOO cost the same as
plain REINFORCE and beat it by about 3 points (3 seeds, p < 0.02). ReMax (an extra greedy rollout)
and an EMA baseline (every sample back-propagated) took about 20 % more GPU time with no
measurable gain. And a single run can differ by 3 points from a same-seed rerun, so a choice
between such methods needs several seeds. Details in [Results](#results).

| estimator | advantage of sample *i* of a prompt | extra cost |
|---|---|---|
| `reinforce` | `r_i` | – |
| `reinforce_ema` | `r_i − b`, `b` = EMA of past batch-mean rewards | – |
| `remax` | `r_i − r(greedy)` | one greedy rollout per prompt |
| `rloo` | `r_i − mean_{j≠i} r_j` | – |
| `grpo` | `(r_i − mean r) / (std r + ε)` | – |
| `grpo_nostd` | `r_i − mean r` (as in Dr. GRPO) | – |

## What is measured

- **Accuracy**: greedy GSM8K test accuracy every 25 steps on a fixed 500-item subset, and on the
  full test split (1,319) at the end; 3 seeds per estimator.
- **Gradient variance and signal-to-noise ratio.** Each step's gradient is accumulated over 16
  micro-batches that each hold one prompt group, so the micro-batch gradients are i.i.d. estimates
  of the same expected gradient. From them we estimate `tr Cov(ĝ)` of the step gradient and
  `SNR = (‖ĝ‖² − tr Cov ĝ) / tr Cov ĝ`, which is invariant to rescaling the advantages
  (see `src/cfrl/gradstats.py`). `tr Cov` itself is in units of the advantage, so it is compared
  only between estimators whose advantages are in reward units (all except `grpo`).
- **Where the variance comes from.** With `within_group_stats` every sample is back-propagated
  separately, which splits `tr Cov` into the part caused by sampling completions of the same
  prompt (the only part a baseline can reduce) and the part caused by which prompts were drawn.
- **Dead groups**: fraction of prompts whose 8 samples all got the same reward. Group-baseline
  methods (RLOO, GRPO) get exactly zero gradient from them.
- **Length and compute**: completion length over training, generated tokens, wall clock.

## Design decisions worth knowing

- **Fully on-policy, one gradient step per rollout batch.** The importance ratio is identically 1
  and PPO clipping is inactive, so differences come from the advantage alone.
- **Sampling equals the trained policy.** Qwen's default generation config (top-k 20, top-p 0.8,
  repetition penalty 1.05) is overridden with plain temperature-1 sampling; otherwise the
  log-probabilities in the gradient would belong to a different distribution than the samples.
- **Unbiased loss aggregation by default** (`loss_agg: constant`: token log-prob sums divided by a
  constant). The GRPO paper's per-sequence length normalisation (`seq_mean`) biases the gradient
  towards short wrong answers / long right ones; it is run as an ablation.
- **Prompt the model already follows.** A first CPU check used the XML `<reasoning>/<answer>`
  format of the common open GRPO GSM8K demo: Qwen2.5-0.5B-Instruct ignored it, every reward was
  0, every group had zero variance and no estimator received any gradient. Such demos bootstrap
  with several shaped format rewards; here Qwen's official `\boxed{}` prompt is used instead, so
  the comparison is about learning to solve problems, not about discovering a format.
- **Strict reward.** The last `\boxed{}` must contain exactly one number (commas, `\$`, `\%`,
  `\text{}` units and simple `\frac` are normalised); listing several candidates earns nothing.
  Truncated answers earn 0, so `max_new_tokens = 512` (300 truncated about half of the greedy
  answers in a CPU check) and truncation rates are logged.
- **Zero-advantage samples are skipped** in the backward pass (their gradient is exactly zero when
  `kl_coef = 0`), which makes RLOO/GRPO cheaper on easy or hopeless prompts.
- **Diagnostics that do not change the update.** `within_group_stats` only changes how the
  backward passes are split; on the GPU the first-step gradient norm matched the default path to
  0.14 % (bf16 rounding), at about 5 % extra training time (`scripts/check_within_stats.sh`). It was
  added after seed 0, so the within-prompt share is reported for seeds 1–2.
- **Memory knob without side effects.** `seqs_per_forward` splits a micro-batch into several
  forward/backward passes; the accumulated gradient is unchanged (checked: identical gradient
  norm to 7 digits on CPU) and the variance statistics still treat the micro-batch as one unit.

## Run

```bash
conda create -n rlcmp python=3.11 && conda activate rlcmp
pip install torch --index-url https://download.pytorch.org/whl/cu130   # RTX 50-series needs CUDA 12.8+ (cu130 used here)
pip install -r requirements.txt

PYTHONPATH=src pytest -q                     # unit + model-level tests (tiny random Qwen2)
scripts/pilot.sh                              # 2 × 40 steps: speed / memory / sanity check
scripts/run_main.sh 0                         # one seed of all 6 estimators; restartable
scripts/run_main.sh 1 2
scripts/run_ablations.sh                      # GRPO seq_mean, format-reward shaping
scripts/check_within_stats.sh                 # within_group_stats on/off: same gradient, timing
PYTHONPATH=src python -m cfrl.analyze runs --out results   # summary.md, tests.md, per_run.csv, curves.png
```

Single run with overrides: `PYTHONPATH=src python -m cfrl.train --config configs/base.yaml estimator=rloo seed=1`.
Each run writes `runs/<name>/{config.json, metrics.jsonl, eval.jsonl, samples.jsonl, summary.json, adapter/}`.
The raw run directories (about 0.7 GB) are not included; `results/` holds everything the analysis produced from them.

## Results

All numbers come from `results/summary.md`, `results/tests.md` and `results/per_run.csv`
(generated by `cfrl.analyze`); curves are in `results/curves.png`. Greedy accuracy before
training: 47.8 % on the full test split.

### Main comparison (3 seeds per estimator, GSM8K test, 1,319 problems)

| estimator | final accuracy (mean ± sd) | eval subset at step 25 | samples with nonzero advantage | generation / training s per step | GPU h per run |
|---|---|---|---|---|---|
| `grpo` | **53.1 ± 0.7 %** | 51.9 % | 58 % | 17.3 / 6.1 | 1.5 |
| `rloo` | **52.5 ± 0.5 %** | 50.5 % | 57 % | 17.8 / 6.3 | 1.6 |
| `remax` | 52.1 ± 2.1 % | 49.3 % | 22 % | 27.1 / 2.6 | 1.9 |
| `grpo_nostd` | 51.9 ± 2.6 % | 51.4 % | 58 % | 17.6 / 6.3 | 1.6 |
| `reinforce_ema` | 50.7 ± 1.9 % | 48.8 % | 100 % | 17.8 / 10.8 | 1.8 |
| `reinforce` | 49.6 ± 0.9 % | 48.8 % | 57 % | 17.3 / 6.0 | 1.5 |

Welch t-tests on final accuracy, one value per training run (`results/tests.md`; no correction
for multiple comparisons):

| comparison | difference | p |
|---|---|---|
| `grpo` − `reinforce` | +3.4 pp | 0.008 |
| `rloo` − `reinforce` | +2.9 pp | 0.016 |
| `grpo` − `rloo` | +0.5 pp | 0.33 |
| `grpo` − `grpo_nostd` | +1.2 pp | 0.52 |
| `grpo` − `remax` | +1.0 pp | 0.49 |
| `rloo` − `reinforce_ema` | +1.8 pp | 0.22 |
| group baselines (`grpo`, `rloo`, `grpo_nostd`) − others, 9 vs 9 runs | +1.7 pp | 0.041 |

### Findings

1. **Having a baseline matters more than which one.** GRPO and RLOO beat plain REINFORCE by about
   3 points and were the most stable across seeds. No other pair of estimators differs
   significantly (all 15 pairs tested, smallest other p = 0.15): the largest gap among the five
   baselines is 2.4 points (GRPO vs EMA), and three seeds cannot resolve it.
2. **Group baselines learn faster early.** After 25 steps the three group-baseline methods were
   at 50.5–51.9 % on the eval subset, the others at 48.8–49.3 %. GRPO reached most of its gain by
   step 50 (54.3 % on the subset) and then stayed flat.
3. **Group baselines cut the gradient variance by about 2.5×.** Mean `tr Cov` in reward units:
   `grpo_nostd` 0.0017, `rloo` 0.0020, `reinforce_ema` 0.0035, `remax` 0.0041, `reinforce`
   0.0047. Each run is measured along its own training trajectory, so this is an observed
   difference, not a controlled one.
4. **Almost all remaining noise is within-prompt sampling noise.** In every run, 95–98 % of
   `tr Cov` came from sampling completions of the same prompt and only 2–5 % from which prompts
   were drawn (seeds 1–2, 12 runs) — even after a group baseline has removed what it can.
5. **The trace SNR cannot rank the estimators at this batch size.** The median per-step SNR is
   between −0.003 and 0.010 in all 21 runs: with 128 samples per step, `‖ĝ‖²` is explained by
   noise summed over all LoRA parameters, while accuracy still improves over steps. The
   variance (finding 3) and accuracy separate the estimators; this SNR does not. One REINFORCE run
   (seed 2) had six steps between 139 and 169 with more than twice its median gradient norm and
   gradients aligned across prompts (per-step SNR up to 5.7); they raise its pooled SNR to 0.33 and
   are counted in the `spike steps` column. They stayed below the clipping threshold and accuracy
   did not drop. `grpo_nostd` seed 1 had three such steps; no other run had any.
6. **Compute depends on how many samples get a nonzero advantage.** Zero-advantage samples are
   skipped. About 40 % of prompt groups had all-equal rewards, so group baselines back-propagate
   only ~58 % of samples. The EMA baseline never equals a 0/1 reward, so every sample is
   back-propagated: 1.8× the training time of REINFORCE for +1.1 points (n.s.). ReMax pays +56 %
   generation time for the greedy rollout, but its baseline equals the sample's reward for 78 % of
   samples, so its backward pass is the cheapest; overall 1.9 GPU h per run, the most expensive
   method, without a measurable gain over GRPO or RLOO.

GPU hours include about 5 % for `within_group_stats` in seeds 1–2.

### Ablations (seed 0 only)

| variant | final accuracy | same seed, main run | format rate (test) |
|---|---|---|---|
| GRPO, per-sequence length normalisation (`seq_mean`) | 53.7 % | 52.6 % | 96.4 % (93.7 %) |
| GRPO + format reward 0.2 | 51.4 % | 52.6 % | 98.5 % (93.7 %) |
| RLOO + format reward 0.2 | 53.0 % | 53.1 % | 97.1 % (97.6 %) |

None of the three changes accuracy detectably. Length normalisation did not hurt here; the
format reward raised GRPO's format rate but not its accuracy.

**How large single-run noise is.** The two format-reward runs were first made before a bug fix:
with non-binary rewards (1.2, 0.2), float32 rounding left advantages of up to 1.2·10⁻³ instead of
exactly 0 in all-equal groups (about 0.1 % of the advantage mass; group estimators now set them to
exactly 0, `tests/test_advantages.py`). Re-running with the same seed after the fix changed the
GRPO result from 54.7 % to 51.4 % and the RLOO result from 53.6 % to 53.0 %. A tiny change in the
first update leads to different samples from then on, so the whole trajectory changes. Differences
of about 3 points between single runs are therefore noise, which is why the main comparison uses
three seeds and the ablations are not read as effects. The pre-fix runs are not part of this
repository.

## Scope and limitations

- One small model, one task with a binary verifiable reward, 200 steps. Conclusions about
  estimators may not transfer to larger models, learned reward models or long-horizon agents.
- Three seeds give rough error bars only; differences smaller than the seed spread are reported
  as such. The t-tests are not corrected for multiple comparisons; the ablations have one seed,
  and single runs can differ by about 3 points (see Ablations).
- All runs are evaluated on the same 1,319 test problems, so the seed spread also contains the
  evaluation noise of a single run (binomial standard error about 1.4 points).

## References and credits

- Shao et al. 2024, *DeepSeekMath* (GRPO) · Ahmadian et al. 2024, *Back to Basics* (RLOO) ·
  Li et al. 2023, *ReMax* · Liu et al. 2025, *Understanding R1-Zero-Like Training* (Dr. GRPO) ·
  Williams 1992 (REINFORCE).
- Estimator definitions were cross-checked against the per-algorithm implementations in
  [nanoRLHF](https://github.com/jackfsuia/nanoRLHF); the XML prompt of the widely used open GRPO
  GSM8K demo was tried first and dropped (see above). The training loop, variance diagnostics and analysis in this
  repository are written for this study.

## How this was built

The research question, the experimental design and the interpretation of the results are mine;
I worked them out in discussion with an AI assistant. The code was written with an AI coding tool
(Claude Code). Correctness rests on the unit tests in `tests/` (each estimator against its
definition, batched log-probabilities against an unbatched reference, the variance statistics
against explicitly computed per-sample gradients, reward parsing) and on the checks described
above. All runs were done on my own machine, and every number in this README can be recomputed
from the run logs with `cfrl.analyze`.

## License

MIT, see `LICENSE`.
