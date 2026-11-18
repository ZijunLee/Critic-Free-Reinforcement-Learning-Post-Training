"""GSM8K loading and prompt construction.

The prompt is Qwen2.5's official chain-of-thought math prompt (final answer in \\boxed{}), which
the instruct model already follows. A custom XML format was tried first: Qwen2.5-0.5B-Instruct
ignored it, every reward was 0 and no estimator received any gradient (see README).
"""

from __future__ import annotations

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_gold(answer_field: str) -> str:
    """GSM8K solutions end with '#### <number>'."""
    return answer_field.split("####")[-1].strip().replace(",", "")


def load_gsm8k(split: str, limit: int = -1) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = [{"question": q, "answer": parse_gold(a)} for q, a in zip(ds["question"], ds["answer"])]
    return items[:limit] if limit > 0 else items


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
