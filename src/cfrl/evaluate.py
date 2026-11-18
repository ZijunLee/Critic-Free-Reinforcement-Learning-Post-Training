"""Greedy evaluation on GSM8K test items."""

from __future__ import annotations

from .data import build_prompt
from .reward import is_correct, is_well_formatted
from .rollout import generate


def evaluate(model, tokenizer, items: list[dict], *, max_new_tokens: int, batch_size: int, eos_ids: list[int],
             num_examples: int = 0, prefix_n: int = 0) -> tuple[dict, list[dict]]:
    """prefix_n > 0 also reports accuracy on the first prefix_n items (the in-training eval subset),
    so the final full-test evaluation extends the training curve without a second pass."""
    was_training = model.training
    model.eval()
    prompts = [build_prompt(tokenizer, it["question"]) for it in items]
    samples = generate(model, tokenizer, prompts, 1, sample=False, temperature=1.0,
                       max_new_tokens=max_new_tokens, batch_size=batch_size, eos_ids=eos_ids)
    correct = [is_correct(s.text, it["answer"]) for s, it in zip(samples, items)]
    n = max(len(items), 1)
    metrics = {
        "accuracy": sum(correct) / n,
        "format_rate": sum(is_well_formatted(s.text) for s in samples) / n,
        "truncation_rate": sum(s.truncated for s in samples) / n,
        "mean_completion_tokens": sum(len(s.completion_ids) for s in samples) / n,
        "n": len(items),
    }
    if 0 < prefix_n <= len(items):
        metrics["accuracy_prefix"] = sum(correct[:prefix_n]) / prefix_n
        metrics["prefix_n"] = prefix_n
    examples = [
        {"question": it["question"], "gold": it["answer"], "completion": s.text, "correct": c}
        for s, it, c in list(zip(samples, items, correct))[:num_examples]
    ]
    if was_training:
        model.train()
    return metrics, examples
