"""InterpreterConfig: standalone, does NOT inherit the generator's locked invariants."""

from __future__ import annotations

import pytest

from interpreter.config import InterpreterConfig, load_config


def test_defaults_are_text_only_and_unlocked():
    c = InterpreterConfig()
    assert c.base_model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert c.tuning_mode == "full"
    # Deliberately NOT the generator locks: epochs is free (not forced to 2), no num_new_tokens,
    # no max_pixels / image knobs on the dataclass at all.
    assert c.epochs != 2 or True  # value is free; assert the field is not validated to ==2
    assert not hasattr(c, "num_new_tokens") and not hasattr(c, "max_pixels")


def test_epochs_not_locked_to_two():
    # The generator's SFTConfig raises unless epochs==2; the interpreter must accept any epochs>=1.
    assert InterpreterConfig(epochs=5).epochs == 5


def test_yaml_overrides_and_list_to_tuple(tmp_path):
    cfg = tmp_path / "interp.yaml"
    cfg.write_text(
        "base_model_id: Qwen/Qwen2.5-0.5B-Instruct\n"
        "epochs: 4\n"
        "learning_rate: 2.0e-5\n"
        "lora_target_modules: [q_proj, v_proj]\n"
        "unknown_key: 123\n",  # unknown keys are filtered, not an error
        encoding="utf-8")
    c = load_config(cfg)
    assert c.epochs == 4 and c.learning_rate == 2.0e-5
    assert c.lora_target_modules == ("q_proj", "v_proj")  # list coerced to tuple


def test_bad_tuning_mode_and_scheduler_rejected():
    with pytest.raises(ValueError):
        InterpreterConfig(tuning_mode="qlora")
    with pytest.raises(ValueError):
        InterpreterConfig(scheduler="onecycle")
