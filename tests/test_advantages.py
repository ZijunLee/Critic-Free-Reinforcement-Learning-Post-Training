import pytest
import torch

from cfrl.advantages import compute_advantages, zero_variance_groups

R = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])


def test_reinforce_is_raw_reward():
    assert torch.equal(compute_advantages(R, "reinforce"), R)


def test_ema_subtracts_constant():
    assert torch.allclose(compute_advantages(R, "reinforce_ema", ema_baseline=0.25), R - 0.25)


def test_rloo_matches_definition_and_scaled_mean_centering():
    adv = compute_advantages(R, "rloo")
    k = R.shape[1]
    for p in range(R.shape[0]):
        for i in range(k):
            others = torch.cat([R[p, :i], R[p, i + 1:]])
            assert adv[p, i] == pytest.approx(float(R[p, i] - others.mean()))
    # identity: r_i - mean_{-i} = k/(k-1) * (r_i - mean)
    assert torch.allclose(adv, k / (k - 1) * (R - R.mean(1, keepdim=True)))


def test_grpo_zero_mean_unit_std_and_dead_groups():
    adv = compute_advantages(R, "grpo", std_eps=0.0 + 1e-8)
    live = ~zero_variance_groups(R)
    assert torch.allclose(adv[live].mean(1), torch.zeros(int(live.sum())), atol=1e-6)
    assert torch.allclose(adv[live].std(1), torch.ones(int(live.sum())), atol=1e-4)
    assert torch.equal(adv[1], torch.zeros(4))  # all-wrong group gives no signal


def test_grpo_nostd_is_mean_centering():
    assert torch.allclose(compute_advantages(R, "grpo_nostd"), R - R.mean(1, keepdim=True))


def test_remax_uses_greedy_reward_per_prompt():
    greedy = torch.tensor([1.0, 0.0, 0.0])
    adv = compute_advantages(R, "remax", greedy_rewards=greedy)
    assert torch.allclose(adv, R - greedy.view(-1, 1))
    with pytest.raises(ValueError):
        compute_advantages(R, "remax")


def test_zero_variance_groups():
    assert zero_variance_groups(R).tolist() == [False, True, False]


def test_unknown_estimator():
    with pytest.raises(ValueError):
        compute_advantages(R, "ppo")


def test_flat_groups_exactly_zero_for_non_binary_rewards():
    # format shaping gives rewards like 1.2 / 0.2; float32 r - mean(r) is not exactly 0 for these
    rewards = torch.tensor([[1.2] * 8, [0.2] * 8, [1.2, 0.2, 1.2, 0.0, 1.2, 1.2, 0.2, 1.2]])
    for est in ("rloo", "grpo", "grpo_nostd"):
        adv = compute_advantages(rewards, est)
        assert torch.all(adv[:2] == 0), est
        assert torch.all(adv[2] != 0), est
