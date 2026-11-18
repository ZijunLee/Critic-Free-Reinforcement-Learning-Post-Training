"""Model-level checks with a tiny randomly initialised Qwen2 (needs the Qwen tokenizer from the Hub)."""

import pytest
import torch

pytest.importorskip("transformers")
pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer unavailable: {exc}")
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = Qwen2ForCausalLM(cfg).eval()
    return model, tok


def _reference_logprobs(model, prompt_ids, completion_ids):
    ids = torch.tensor([prompt_ids + completion_ids])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(logits[:-1], -1)
    targets = ids[0, 1:]
    tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_lp[len(prompt_ids) - 1:]


def test_batched_logprobs_match_unpadded_reference(tiny):
    from cfrl.rollout import Sample, build_batch, completion_logprobs

    model, tok = tiny
    samples = [
        Sample(tok("short prompt").input_ids, tok(" answer one two three").input_ids, "", False),
        Sample(tok("a much longer prompt with many more tokens in it").input_ids, tok(" x").input_ids, "", False),
    ]
    batch = build_batch(samples, tok.pad_token_id, "cpu")
    with torch.no_grad():
        lp = completion_logprobs(model, batch, temperature=1.0)
    for i, s in enumerate(samples):
        ref = _reference_logprobs(model, s.prompt_ids, s.completion_ids)
        got = lp[i, : len(s.completion_ids)]
        assert torch.allclose(got, ref, atol=1e-4), (i, got, ref)
        assert torch.all(lp[i, len(s.completion_ids):] == 0)


def test_generate_cuts_at_eos_and_groups_samples(tiny):
    from cfrl.rollout import eos_token_ids, generate

    model, tok = tiny
    eos = eos_token_ids(tok, model)
    out = generate(model, tok, ["p one", "prompt two is longer"], 3, sample=True, temperature=1.0,
                   max_new_tokens=8, batch_size=4, eos_ids=eos)
    assert len(out) == 6
    assert out[0].prompt_ids == out[2].prompt_ids != out[3].prompt_ids
    for s in out:
        assert 1 <= len(s.completion_ids) <= 8
        assert s.truncated == (s.completion_ids[-1] not in eos)
        assert all(t not in eos for t in s.completion_ids[:-1])
