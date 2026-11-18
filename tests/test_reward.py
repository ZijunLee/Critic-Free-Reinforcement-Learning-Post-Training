from cfrl.data import parse_gold
from cfrl.reward import compute_reward, extract_boxed, is_correct, is_well_formatted, to_number

GOOD = "48/2 = 24 and 48 + 24 = 72.\n\nSo the answer is \\(\\boxed{72}\\)."


def test_parse_gold():
    assert parse_gold("blah <<48+24=72>>72\n#### 1,072") == "1072"


def test_extract_last_boxed_with_nesting():
    assert extract_boxed("\\boxed{1} then \\boxed{\\frac{3}{4}}") == "\\frac{3}{4}"
    assert extract_boxed("no box 72") is None
    assert extract_boxed("\\boxed{7") is None  # truncated


def test_to_number_variants():
    assert to_number("1,000") == 1000.0
    assert to_number("1{,}000") == 1000.0
    assert to_number("\\$18") == 18.0
    assert to_number("18 \\text{ dollars}") == 18.0
    assert to_number("50\\%") == 50.0
    assert to_number("-3.5") == -3.5
    assert to_number("\\frac{3}{4}") == 0.75
    assert to_number("5 or 6") is None  # several candidates earn nothing
    assert to_number("x") is None


def test_is_correct():
    assert is_correct(GOOD, "72")
    assert is_correct(GOOD.replace("{72}", "{72.0}"), "72")
    assert not is_correct(GOOD, "73")
    assert not is_correct("The answer is 72.", "72")  # box is required


def test_format_and_reward():
    assert is_well_formatted(GOOD) and not is_well_formatted("72")
    assert compute_reward(GOOD, "72") == 1.0
    assert compute_reward(GOOD, "72", format_weight=0.2) == 1.2
    assert compute_reward(GOOD, "1", format_weight=0.2) == 0.2
