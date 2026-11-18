"""Online estimate of the variance of the policy-gradient estimator.

The step gradient is accumulated over M micro-batches that each hold whole prompt groups, so
the per-micro-batch gradients g_1..g_M are i.i.d. estimates of the same expected gradient
(prompts are sampled independently). If micro-batch m contributes delta_m to the accumulated
gradient, the micro-batch estimate at full-batch scale is g_m = M * delta_m and the step
gradient is g_bar = mean_m g_m. We report

    grad_sq_norm  = ||g_bar||^2
    grad_var_mean = tr Cov(g_bar) = (1/M) * 1/(M-1) * sum_m ||g_m - g_bar||^2

E||g_bar||^2 = ||E g||^2 + tr Cov(g_bar), so the signal-to-noise ratio
(grad_sq_norm - grad_var_mean) / grad_var_mean is invariant to rescaling the advantages,
which makes estimators with different advantage scales (e.g. GRPO's 1/std) comparable.
"""

from __future__ import annotations

import torch


class GradStats:
    def __init__(self, params: list[torch.nn.Parameter], num_micro: int):
        if num_micro < 2:
            raise ValueError("need at least 2 micro-batches")
        self.params = params
        self.m = num_micro
        self.seen = 0
        self.prev = None
        self.sum_g = None
        self.sum_sq = 0.0

    def _flat_grad(self) -> torch.Tensor:
        parts = []
        for p in self.params:
            if p.grad is None:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
            else:
                parts.append(p.grad.detach().reshape(-1).float())
        return torch.cat(parts)

    def update(self) -> None:
        """Call after every micro-batch (also for skipped micro-batches, whose delta is zero)."""
        cur = self._flat_grad()
        if self.prev is None:
            self.prev = torch.zeros_like(cur)
            self.sum_g = torch.zeros_like(cur)
        g_m = self.m * (cur - self.prev)
        self.prev = cur
        self.sum_g += g_m
        self.sum_sq += float((g_m * g_m).sum())
        self.seen += 1

    def finalize(self) -> dict:
        if self.seen != self.m:
            raise RuntimeError(f"expected {self.m} micro-batch updates, got {self.seen}")
        g_bar = self.sum_g / self.m
        sq_norm = float((g_bar * g_bar).sum())
        # sum_m ||g_m - g_bar||^2 = sum_m ||g_m||^2 - M ||g_bar||^2
        scatter = max(self.sum_sq - self.m * sq_norm, 0.0)
        var_mean = scatter / (self.m - 1) / self.m
        return {"grad_sq_norm": sq_norm, "grad_var_mean": var_mean}


class WithinGroupStats:
    """Part of the step-gradient variance that comes from sampling completions of the same prompt.

    GradStats measures tr Cov of the step gradient across micro-batches, which mixes two sources:
    which prompts were drawn, and which completions were sampled for each prompt. A baseline can
    only reduce the second. Here every sample of a group is back-propagated separately, so its
    contribution h_i to the step gradient is the change of the accumulated gradient. With K
    samples per prompt, the group's contribution is sum_i h_i and

        Var(sum_i h_i | prompt) ~= K * s^2,   s^2 = 1/(K-1) * sum_i ||h_i - mean_j h_j||^2

    (exact for independent samples; group baselines make the h_i weakly dependent). Summing over
    the prompts of a step gives grad_var_within; samples skipped because their advantage is 0
    contribute h_i = 0. Logging only: the accumulated gradient is unchanged.
    """

    def __init__(self, params: list[torch.nn.Parameter]):
        self.params = params
        self.total = 0.0
        self.groups = 0
        self._prev = None

    def _flat_grad(self) -> torch.Tensor:
        return GradStats._flat_grad(self)  # same flattening as GradStats

    def start_group(self) -> None:
        self._prev = self._flat_grad()
        self._sum = torch.zeros_like(self._prev)
        self._sum_sq = 0.0
        self._n = 0

    def add_sample(self) -> None:
        """Call after the backward pass of one sample of the current group."""
        cur = self._flat_grad()
        h = cur - self._prev
        self._prev = cur
        self._sum += h
        self._sum_sq += float((h * h).sum())
        self._n += 1

    def add_zero_samples(self, count: int) -> None:
        self._n += count

    def end_group(self, group_size: int) -> None:
        if self._n != group_size:
            raise RuntimeError(f"expected {group_size} samples in the group, got {self._n}")
        k = group_size
        scatter = max(self._sum_sq - float((self._sum * self._sum).sum()) / k, 0.0)
        self.total += k * scatter / (k - 1)
        self.groups += 1

    def finalize(self) -> dict:
        return {"grad_var_within": self.total}
