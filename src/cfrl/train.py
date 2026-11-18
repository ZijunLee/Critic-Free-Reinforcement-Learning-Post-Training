"""Training loop shared by all estimators.

Usage:
    python -m cfrl.train --config configs/base.yaml estimator=rloo seed=1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time

import numpy as np
import torch

from .advantages import compute_advantages, zero_variance_groups
from .config import Config, load_config
from .data import build_prompt, load_gsm8k
from .evaluate import evaluate
from .gradstats import GradStats, WithinGroupStats
from .reward import compute_reward, is_correct, is_well_formatted
from .rollout import build_batch, completion_logprobs, eos_token_ids, generate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_model(cfg: Config, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.dtype]
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype).to(device)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, tokenizer


def policy_loss(logprobs: torch.Tensor, mask: torch.Tensor, adv: torch.Tensor, cfg: Config,
                n_total: int, tokens_total: int, ref_logprobs: torch.Tensor | None) -> tuple[torch.Tensor, float]:
    """Micro-batch share of the step loss; summing over micro-batches gives the full-batch loss."""
    per_token = -adv.unsqueeze(1) * logprobs
    kl_sum = 0.0
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - logprobs
        kl = (torch.exp(log_ratio) - log_ratio - 1.0) * mask  # k3 estimator, >= 0
        per_token = per_token + cfg.kl_coef * kl
        kl_sum = float((kl.detach().sum(1) / mask.sum(1).clamp(min=1)).sum())
    per_token = per_token * mask
    if cfg.loss_agg == "constant":
        loss = per_token.sum() / (n_total * cfg.max_new_tokens)
    elif cfg.loss_agg == "seq_mean":
        loss = (per_token.sum(1) / mask.sum(1).clamp(min=1)).sum() / n_total
    else:  # token_mean
        loss = per_token.sum() / tokens_total
    return loss, kl_sum


def train(cfg: Config) -> dict:
    run_dir = cfg.run_dir
    if run_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{run_dir} exists; pass overwrite=true to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model, tokenizer = load_model(cfg, device)
    eos_ids = eos_token_ids(tokenizer, model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / cfg.warmup_steps) if cfg.warmup_steps > 0 else 1.0)

    train_items = load_gsm8k("train", cfg.train_size)
    test_items = load_gsm8k("test")
    eval_items = test_items[: cfg.eval_size] if cfg.eval_size > 0 else test_items
    final_items = test_items[: cfg.final_eval_size] if cfg.final_eval_size > 0 else test_items
    order_rng = random.Random(cfg.seed)
    order: list[int] = []

    metrics_f = open(run_dir / "metrics.jsonl", "w")
    eval_f = open(run_dir / "eval.jsonl", "w")
    samples_f = open(run_dir / "samples.jsonl", "w")

    def run_eval(step: int, items: list[dict], tag: str, prefix_n: int = 0) -> dict:
        t0 = time.time()
        m, examples = evaluate(model, tokenizer, items, max_new_tokens=cfg.max_new_tokens,
                               batch_size=cfg.eval_batch_size, eos_ids=eos_ids, num_examples=cfg.log_samples,
                               prefix_n=prefix_n)
        m.update(step=step, split=tag, time_s=time.time() - t0)
        eval_f.write(json.dumps(m) + "\n")
        eval_f.flush()
        for ex in examples:
            samples_f.write(json.dumps({"step": step, "split": tag, **ex}) + "\n")
        samples_f.flush()
        print(f"[eval {tag}] step {step}: acc={m['accuracy']:.4f} format={m['format_rate']:.3f} "
              f"len={m['mean_completion_tokens']:.0f}", flush=True)
        return m

    if cfg.eval_every > 0:
        run_eval(0, eval_items, "test_subset")

    k = cfg.group_size
    num_micro = cfg.prompts_per_step // cfg.prompts_per_micro
    ema_baseline = 0.0
    gen_tokens_cum = 0
    t_start = time.time()

    for step in range(1, cfg.num_steps + 1):
        # ---- rollout ----
        batch_items = []
        for _ in range(cfg.prompts_per_step):
            if not order:
                order = list(range(len(train_items)))
                order_rng.shuffle(order)
            batch_items.append(train_items[order.pop()])
        prompts = [build_prompt(tokenizer, it["question"]) for it in batch_items]

        t0 = time.time()
        model.eval()
        samples = generate(model, tokenizer, prompts, k, sample=True, temperature=cfg.temperature,
                           max_new_tokens=cfg.max_new_tokens, batch_size=cfg.gen_batch_size, eos_ids=eos_ids)
        greedy_rewards = None
        greedy_tokens = 0
        if cfg.estimator == "remax":
            greedy = generate(model, tokenizer, prompts, 1, sample=False, temperature=1.0,
                              max_new_tokens=cfg.max_new_tokens, batch_size=cfg.gen_batch_size, eos_ids=eos_ids)
            greedy_rewards = torch.tensor([compute_reward(s.text, it["answer"], cfg.format_weight)
                                           for s, it in zip(greedy, batch_items)])
            greedy_tokens = sum(len(s.completion_ids) for s in greedy)
        t_gen = time.time() - t0

        golds = [it["answer"] for it in batch_items for _ in range(k)]
        rewards = torch.tensor([compute_reward(s.text, g, cfg.format_weight) for s, g in zip(samples, golds)]
                               ).view(cfg.prompts_per_step, k)
        correct = torch.tensor([is_correct(s.text, g) for s, g in zip(samples, golds)], dtype=torch.float32)
        formatted = torch.tensor([is_well_formatted(s.text) for s in samples], dtype=torch.float32)
        adv = compute_advantages(rewards, cfg.estimator, greedy_rewards=greedy_rewards,
                                 ema_baseline=ema_baseline, std_eps=cfg.std_eps)
        adv_flat = adv.reshape(-1)
        comp_lens = [len(s.completion_ids) for s in samples]
        gen_tokens_cum += sum(comp_lens) + greedy_tokens

        # ---- gradient ----
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        stats = GradStats(params, num_micro)
        within = WithinGroupStats(params) if cfg.within_group_stats else None
        n_total = len(samples)
        tokens_total = sum(comp_lens)
        loss_total, kl_total, rows_used = 0.0, 0.0, 0

        def backward_rows(rows: list[int], chunk_size: int) -> None:
            nonlocal loss_total, kl_total
            for c in range(0, len(rows), chunk_size):
                chunk = rows[c : c + chunk_size]
                batch = build_batch([samples[r] for r in chunk], tokenizer.pad_token_id, device)
                ref_lp = None
                if cfg.kl_coef > 0:
                    with torch.no_grad(), model.disable_adapter():
                        ref_lp = completion_logprobs(model, batch, cfg.temperature)
                lp = completion_logprobs(model, batch, cfg.temperature)
                loss, kl_sum = policy_loss(lp, batch["completion_mask"], adv_flat[chunk].to(device), cfg,
                                           n_total, tokens_total, ref_lp)
                loss.backward()
                loss_total += float(loss)
                kl_total += kl_sum

        def active(row: int) -> bool:
            # zero-advantage samples contribute exactly zero gradient when there is no KL term
            return cfg.kl_coef > 0 or adv_flat[row] != 0

        # a micro-batch holds whole groups; it may be split into several forward passes to bound memory.
        # Gradients accumulate, and GradStats treats the micro-batch as one unit.
        for mb in range(num_micro):
            prompts_mb = range(mb * cfg.prompts_per_micro, (mb + 1) * cfg.prompts_per_micro)
            if within is None:
                rows = [r for p in prompts_mb for r in range(p * k, (p + 1) * k) if active(r)]
                backward_rows(rows, cfg.seqs_per_forward)
                rows_used += len(rows)
            else:  # one backward pass per sample, so each sample's gradient contribution is observable
                for p in prompts_mb:
                    within.start_group()
                    for r in range(p * k, (p + 1) * k):
                        if active(r):
                            backward_rows([r], 1)
                            within.add_sample()
                            rows_used += 1
                        else:
                            within.add_zero_samples(1)
                    within.end_group(k)
            stats.update()
        grad_stats = stats.finalize()
        if within is not None:
            grad_stats.update(within.finalize())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()  # return training activations' blocks before the next generate()
        t_train = time.time() - t0

        batch_mean_reward = float(rewards.mean())
        if cfg.estimator == "reinforce_ema":
            ema_baseline = cfg.ema_decay * ema_baseline + (1 - cfg.ema_decay) * batch_mean_reward

        record = {
            "step": step,
            "reward_mean": batch_mean_reward,
            "sample_accuracy": float(correct.mean()),
            "format_rate": float(formatted.mean()),
            "zero_variance_group_frac": float(zero_variance_groups(rewards).float().mean()),
            "nonzero_adv_frac": rows_used / n_total if cfg.kl_coef == 0.0 else float((adv_flat != 0).float().mean()),
            "adv_abs_mean": float(adv.abs().mean()),
            "completion_tokens_mean": sum(comp_lens) / n_total,
            "truncation_rate": sum(s.truncated for s in samples) / n_total,
            "loss": loss_total,
            "kl": kl_total / n_total if cfg.kl_coef > 0 else 0.0,
            "grad_norm_preclip": grad_norm,
            **grad_stats,
            "lr": scheduler.get_last_lr()[0],
            "ema_baseline": ema_baseline,
            "gen_tokens_cum": gen_tokens_cum,
            "time_gen_s": t_gen,
            "time_train_s": t_train,
            "elapsed_s": time.time() - t_start,
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        }
        if greedy_rewards is not None:
            record["greedy_reward_mean"] = float(greedy_rewards.mean())
        metrics_f.write(json.dumps(record) + "\n")
        metrics_f.flush()
        print(f"step {step}/{cfg.num_steps} reward={batch_mean_reward:.3f} acc={record['sample_accuracy']:.3f} "
              f"zero_var={record['zero_variance_group_frac']:.2f} len={record['completion_tokens_mean']:.0f} "
              f"gnorm={grad_norm:.3g} gen={t_gen:.1f}s train={t_train:.1f}s mem={record['peak_mem_gb']:.1f}G", flush=True)

        if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step != cfg.num_steps:
            run_eval(step, eval_items, "test_subset")

    final = run_eval(cfg.num_steps, final_items, "test_final", prefix_n=len(eval_items))
    model.save_pretrained(run_dir / "adapter")
    summary = {"run_name": cfg.run_name, "estimator": cfg.estimator, "seed": cfg.seed,
               "final_accuracy": final["accuracy"], "final_n": final["n"],
               "gen_tokens_total": gen_tokens_cum, "wall_clock_s": time.time() - t_start}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for f in (metrics_f, eval_f, samples_f):
        f.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("overrides", nargs="*", help="key=value overrides of Config fields")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    print(json.dumps(cfg.to_dict(), indent=2))
    train(cfg)


if __name__ == "__main__":
    main()
