import pytest

from cfrl.config import load_config


def test_overrides_are_typed():
    cfg = load_config("configs/base.yaml", ["estimator=rloo", "seed=3", "learning_rate=1e-5", "overwrite=true"])
    assert cfg.estimator == "rloo" and cfg.seed == 3 and cfg.learning_rate == 1e-5 and cfg.overwrite is True
    assert cfg.run_name == "rloo_s3"
    assert isinstance(cfg.learning_rate, float)


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        load_config(None, ["estimator=ppo"])
    with pytest.raises(ValueError):
        load_config(None, ["not_a_key=1"])
    with pytest.raises(ValueError):
        load_config(None, ["prompts_per_step=2", "prompts_per_micro=2"])
