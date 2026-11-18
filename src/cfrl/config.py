"""Run configuration: one dataclass, loaded from YAML and overridden by `key=value` CLI args."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

ESTIMATORS = ("reinforce", "reinforce_ema", "remax", "rloo", "grpo", "grpo_nostd")
LOSS_AGGS = ("constant", "seq_mean", "token_mean")


@dataclass
class Config:
    # run
    run_name: str = ""  # default: "<estimator>_s<seed>"
    output_dir: str = "runs"
    overwrite: bool = False
    seed: int = 0
    device: str = "auto"  # auto | cuda | cpu

    # model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"  # bfloat16 | float32
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True

    # algorithm: the estimator is the ONLY thing that should differ between main runs
    estimator: str = "grpo"
    group_size: int = 8  # samples per prompt (K); identical budget for every estimator
    ema_decay: float = 0.9  # reinforce_ema only
    std_eps: float = 1e-4  # grpo only
    kl_coef: float = 0.0  # k3 KL to the frozen base model (LoRA disabled)
    loss_agg: str = "constant"  # constant: sum_t / max_new_tokens (unbiased); seq_mean: GRPO paper; token_mean

    # reward
    format_weight: float = 0.0  # reward = correct + format_weight * well_formatted

    # rollout
    prompts_per_step: int = 16
    prompts_per_micro: int = 1  # micro-batches hold whole groups; also the unit for gradient-variance estimates
    seqs_per_forward: int = 4  # memory knob only: sequences per forward/backward pass inside a micro-batch
    within_group_stats: bool = False  # log within-prompt gradient variance (one backward per sample; slower, same gradient)
    max_new_tokens: int = 300
    temperature: float = 1.0
    gen_batch_size: int = 64

    # optimisation
    num_steps: int = 200
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 10

    # data / eval
    train_size: int = -1  # -1 = full GSM8K train split
    eval_every: int = 25
    eval_size: int = 500  # first N GSM8K test items during training
    final_eval_size: int = -1  # -1 = full test split (1,319)
    eval_batch_size: int = 64
    log_samples: int = 4

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, got {self.estimator!r}")
        if self.loss_agg not in LOSS_AGGS:
            raise ValueError(f"loss_agg must be one of {LOSS_AGGS}, got {self.loss_agg!r}")
        if self.prompts_per_step % self.prompts_per_micro:
            raise ValueError("prompts_per_step must be divisible by prompts_per_micro")
        if self.prompts_per_step // self.prompts_per_micro < 2:
            raise ValueError("need at least 2 micro-batches per step to estimate gradient variance")
        if self.seqs_per_forward < 1:
            raise ValueError("seqs_per_forward must be >= 1")
        if self.estimator == "rloo" and self.group_size < 2:
            raise ValueError("rloo needs group_size >= 2")
        if not self.run_name:
            self.run_name = f"{self.estimator}_s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _cast(value: str, target_type: type):
    if target_type is bool:
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"not a bool: {value!r}")
    return target_type(value)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    values: dict = {}
    if path:
        values.update(yaml.safe_load(Path(path).read_text()) or {})
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    py_types = {"str": str, "int": int, "float": float, "bool": bool}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in types:
            raise ValueError(f"bad override {item!r}; expected key=value with key in Config")
        values[key] = _cast(raw, py_types[types[key]])
    unknown = set(values) - set(types)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return Config(**values)
