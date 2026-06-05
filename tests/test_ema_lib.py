"""Tests for ExponentialMovingAverage (lib/ema_lib.py).

Covers: shadow copy creation, parameter updates via update(), get_state_dict,
and load_ema_state.  All tests are deterministic and run on CPU only.
"""

import pytest
import torch
import torch.nn as nn

from lib.ema_lib import ExponentialMovingAverage as EMA


# pytest .\tests\test_ema_lib.py


@pytest.fixture()
def dummy_model():
    """Simple linear model used for all EMA tests."""
    return nn.Linear(4, 8)


class TestShadowCopyIndependence:
    def test_shadow_is_deep_copy(self, dummy_model):
        ema = EMA(dummy_model, decay=0.99)

        # Shadow parameters must be independent objects (not shared).
        for s_param, t_param in zip(
            ema.shadow_model.parameters(), dummy_model.parameters()
        ):
            assert s_param.data is not t_param.data

    def test_mutating_live_model_does_not_affect_shadow(self, dummy_model):
        from lib.ema_lib import ExponentialMovingAverage as EMA

        ema = EMA(dummy_model, decay=0.99)

        shadow_snapshot = {k: v.clone() for k, v in ema.shadow_model.state_dict().items()}

        # Mutate the live model.
        dummy_model.weight.data.add_(1.0)
        dummy_model.bias.data.mul_(2.0)

        # Shadow must be untouched.
        for key in shadow_snapshot:
            assert torch.equal(ema.shadow_model.state_dict()[key], shadow_snapshot[key])


class TestUpdate:
    def test_update_modifies_only_shadow(self, dummy_model):
        ema = EMA(dummy_model, decay=0.9)

        original_weights = {k: v.clone() for k, v in dummy_model.state_dict().items()}

        # Give the live model a different weight so the blend is visible.
        dummy_model.weight.data.fill_(1.0)
        dummy_model.bias.data.fill_(-1.0)
        ema.update(dummy_model)

        assert not torch.equal(ema.shadow_model.weight, original_weights["weight"])  # type: ignore
        assert not torch.equal(ema.shadow_model.bias, original_weights["bias"])  # type: ignore

    def test_update_formula(self, dummy_model):
        """Verify the exact blend:  ema_new = decay * ema_old + (1-decay) * new."""
        # Use a high initial weight so we can predict the outcome exactly.
        dummy_model.weight.data.fill_(2.0)
        dummy_model.bias.data.fill_(2.0)

        ema = EMA(dummy_model, decay=0.9)  # shadow starts at 2.0

        # Now set live model to a different value.
        dummy_model.weight.data.fill_(1.0)
        ema.update(dummy_model)

        expected_weight = 0.9 * 2.0 + 0.1 * 1.0  # = 1.9
        assert torch.allclose(ema.shadow_model.weight, torch.tensor(expected_weight))  # type: ignore

    def test_zero_decay_skips_update(self, dummy_model):
        ema = EMA(dummy_model, decay=0.0)

        initial_shadow = {k: v.clone() for k, v in ema.shadow_model.state_dict().items()}

        dummy_model.weight.data.fill_(99.0)
        ema.update(dummy_model)

        # Shadow must remain identical to the original clone.
        for key in initial_shadow:
            assert torch.equal(ema.shadow_model.state_dict()[key], initial_shadow[key])


class TestStateDictRoundTrip:
    def test_get_state_dict_structure(self, dummy_model):
        ema = EMA(dummy_model, decay=0.9)
        state = ema.get_state_dict()

        assert isinstance(state, dict)
        assert "decay" in state
        assert "ema_state" in state
        assert state["decay"] == 0.9

    def test_get_state_dict_matches_shadow(self, dummy_model):
        ema = EMA(dummy_model, decay=0.9)
        shadow_sd = ema.shadow_model.state_dict()
        state = ema.get_state_dict()["ema_state"]

        for key in shadow_sd:
            assert torch.equal(shadow_sd[key], state[key])

    def test_load_ema_restores_weights(self, dummy_model):
        """Use the classmethod to load saved EMA state."""
        # Set up a known state.
        dummy_model.weight.data.fill_(3.0)
        ema = EMA(dummy_model, decay=0.95)
        dummy_model.weight.data.fill_(1.0)  # mutate live model (shadow unchanged).
        ema.update(dummy_model)

        saved_state = ema.get_state_dict()

        # Create a fresh EMA using the classmethod and check weights match.
        restored_ema = EMA.load_ema_state(
            dummy_model,
            {"decay": saved_state["decay"], "ema_state": saved_state["ema_state"]},
        )

        for key in saved_state["ema_state"]:
            assert torch.allclose(
                restored_ema.shadow_model.state_dict()[key],
                saved_state["ema_state"][key],
            )

    def test_load_ema_returns_correct_type(self, dummy_model):
        ema = EMA(dummy_model, decay=0.9)
        state = ema.get_state_dict()

        restored = EMA.load_ema_state(dummy_model, state)
        assert isinstance(restored, EMA)

class TestConvergence:
    def test_multiple_updates_converge_to_live_weights(self, dummy_model):
        """After many updates the shadow should approach the live model's weights."""
        decay = 0.95
        ema = EMA(dummy_model, decay=decay)

        # Hold the live model at a fixed weight.
        dummy_model.weight.data.fill_(1.0)

        for _ in range(200):
            ema.update(dummy_model)

        shadow_weight = ema.shadow_model.weight[0][0].item() # type: ignore
        assert abs(shadow_weight - 1.0) < 0.01, (
            f"Shadow weight {shadow_weight} should be close to 1.0 after many updates"
        )
