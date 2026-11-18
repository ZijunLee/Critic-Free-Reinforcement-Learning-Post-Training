"""Sampling completions and computing their log-probabilities under the policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]  # up to and including the first EOS token
    text: str
    truncated: bool  # hit max_new_tokens without EOS


def eos_token_ids(tokenizer, model) -> list[int]:
    ids = set()
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[str],
    num_samples: int,
    *,
    sample: bool,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    eos_ids: list[int],
) -> list[Sample]:
    """Returns len(prompts) * num_samples samples, the samples of each prompt adjacent.

    Sampling uses the plain softmax at `temperature` (top-k/top-p/repetition penalty disabled,
    overriding Qwen's generation_config defaults), so the sampling distribution is exactly the
    policy whose log-probabilities enter the gradient.
    """
    expanded = [p for p in prompts for _ in range(num_samples)]
    eos = set(eos_ids)
    device = next(model.parameters()).device
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
        repetition_penalty=1.0,
        use_cache=True,
    )
    if sample:
        kwargs.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    else:
        kwargs.update(do_sample=False, temperature=None, top_k=None, top_p=None)

    samples: list[Sample] = []
    for start in range(0, len(expanded), batch_size):
        chunk = expanded[start : start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, **kwargs)
        completions = out[:, enc["input_ids"].shape[1] :].tolist()
        for row, completion in enumerate(completions):
            prompt_ids = enc["input_ids"][row][enc["attention_mask"][row].bool()].tolist()
            cut = next((i for i, tok in enumerate(completion) if tok in eos), None)
            completion_ids = completion if cut is None else completion[: cut + 1]
            samples.append(
                Sample(
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    text=tokenizer.decode(completion_ids, skip_special_tokens=True),
                    truncated=cut is None,
                )
            )
    return samples


def build_batch(samples: list[Sample], pad_id: int, device) -> dict:
    """Layout [left pad | prompt | completion | right pad]: every completion starts at column P,
    so logits are only needed for the last C+1 positions."""
    p_len = max(len(s.prompt_ids) for s in samples)
    c_len = max(len(s.completion_ids) for s in samples)
    n = len(samples)
    input_ids = torch.full((n, p_len + c_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, p_len + c_len), dtype=torch.long)
    completion_mask = torch.zeros((n, c_len), dtype=torch.float32)
    for i, s in enumerate(samples):
        p, c = len(s.prompt_ids), len(s.completion_ids)
        input_ids[i, p_len - p : p_len] = torch.tensor(s.prompt_ids)
        input_ids[i, p_len : p_len + c] = torch.tensor(s.completion_ids)
        attention_mask[i, p_len - p : p_len + c] = 1
        completion_mask[i, :c] = 1.0
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(device),
        "completion_mask": completion_mask.to(device),
        "prompt_len": p_len,
    }


def completion_logprobs(model, batch: dict, temperature: float) -> torch.Tensor:
    """Per-token log pi(y_t | x, y_<t) for completion tokens: [n, C] (zeros where masked)."""
    c_len = batch["completion_mask"].shape[1]
    p_len = batch["prompt_len"]
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        logits_to_keep=c_len + 1,
        use_cache=False,
    )
    # kept positions P-1 .. P+C-1; position t predicts token t+1, so drop the last one
    logits = out.logits[:, :-1].float()
    if temperature != 1.0:
        logits = logits / temperature
    targets = batch["input_ids"][:, p_len : p_len + c_len]
    token_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logprobs = token_logits - torch.logsumexp(logits, dim=-1)
    return logprobs * batch["completion_mask"]
