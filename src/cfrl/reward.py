"""Verifiable reward for GSM8K: numeric match of the content of the last \\boxed{...}."""

from __future__ import annotations

import re

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
FRAC_RE = re.compile(r"\\[dt]?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}")
TEXT_RE = re.compile(r"\\(?:text|mbox|mathrm)\{[^{}]*\}")


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, with nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return None  # unbalanced (e.g. truncated)


def to_number(answer: str) -> float | None:
    """Exactly one number is required, so listing several candidates cannot earn reward."""
    s = TEXT_RE.sub("", answer)
    s = s.replace("{,}", "").replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "")
    frac = FRAC_RE.fullmatch(s.strip())
    if frac:
        den = float(frac.group(2))
        return float(frac.group(1)) / den if den else None
    numbers = NUMBER_RE.findall(s)
    if len(numbers) != 1:
        return None
    return float(numbers[0])


def is_correct(text: str, gold: str) -> bool:
    answer = extract_boxed(text)
    if answer is None:
        return False
    value = to_number(answer)
    return value is not None and abs(value - float(gold)) < 1e-6


def is_well_formatted(text: str) -> bool:
    """Ends with a parsable boxed number (the format the reward requires)."""
    answer = extract_boxed(text)
    return answer is not None and to_number(answer) is not None


def compute_reward(text: str, gold: str, format_weight: float = 0.0) -> float:
    reward = float(is_correct(text, gold))
    if format_weight:
        reward += format_weight * float(is_well_formatted(text))
    return reward
