"""Advantage estimators. This module is the only place where the compared methods differ.

All estimators produce one scalar advantage per sampled completion; the policy-gradient loss is
-A * log pi(y|x). With one gradient step per rollout batch (fully on-policy), PPO-style ratios
are identically 1 and clipping is inactive, so differences between runs come from the baseline
and scaling alone.

    reinforce      A = r                                   (no baseline)
    reinforce_ema  A = r - b,  b = EMA of past batch-mean rewards (independent of current samples)
    remax          A = r - r(greedy completion of the same prompt)          [Li et al., 2023]
    rloo           A_i = r_i - mean_{j != i} r_j                             [Ahmadian et al., 2024]
    grpo           A_i = (r_i - mean(r)) / (std(r) + eps)                    [Shao et al., 2024]
    grpo_nostd     A_i = r_i - mean(r)   (GRPO without std scaling, as in Dr. GRPO) [Liu et al., 2025]
"""

from __future__ import annotations

import torch


def compute_advantages(
    rewards: torch.Tensor,
    estimator: str,
    *,
    greedy_rewards: torch.Tensor | None = None,
    ema_baseline: float = 0.0,
    std_eps: float = 1e-4,
) -> torch.Tensor:
    """rewards: [num_prompts, group_size]. Returns advantages with the same shape."""
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [num_prompts, group_size]")
    rewards = rewards.float()
    k = rewards.shape[1]
    mean = rewards.mean(dim=1, keepdim=True)

    if estimator == "reinforce":
        return rewards.clone()
    if estimator == "reinforce_ema":
        return rewards - ema_baseline
    if estimator == "remax":
        if greedy_rewards is None:
            raise ValueError("remax needs greedy_rewards of shape [num_prompts]")
        return rewards - greedy_rewards.float().view(-1, 1)
    if estimator == "rloo":
        if k < 2:
            raise ValueError("rloo needs group_size >= 2")
        loo_mean = (rewards.sum(dim=1, keepdim=True) - rewards) / (k - 1)
        return _zero_flat_groups(rewards, rewards - loo_mean)
    if estimator == "grpo":
        std = rewards.std(dim=1, keepdim=True)  # unbiased, as in TRL
        return _zero_flat_groups(rewards, (rewards - mean) / (std + std_eps))
    if estimator == "grpo_nostd":
        return _zero_flat_groups(rewards, rewards - mean)
    raise ValueError(f"unknown estimator {estimator!r}")


def _zero_flat_groups(rewards: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
    """Group baselines give exactly zero advantage when all rewards in a group are equal. In float32,
    r - mean(r) leaves a rounding residue for rewards like 1.2 (GRPO then divides it by eps), so set it."""
    return adv.masked_fill(zero_variance_groups(rewards).view(-1, 1), 0.0)


def zero_variance_groups(rewards: torch.Tensor) -> torch.Tensor:
    """Groups where every sample got the same reward: group-baseline methods get no signal there."""
    return (rewards.max(dim=1).values - rewards.min(dim=1).values) == 0
