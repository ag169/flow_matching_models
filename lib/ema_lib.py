"""Exponential Moving Average (EMA) of model parameters.

Maintains a shadow copy of the model that is updated after each training step
using exponential moving average. The shadow model is used for inference /
evaluation and never mutates the live training model.
"""

from typing import Any, Dict

import copy
import torch
import torch.nn as nn


def _deep_clone_model(model: nn.Module) -> nn.Module:
    """Create a deep clone of *model* with independent parameter tensors."""
    return copy.deepcopy(model)


class ExponentialMovingAverage:
    """Maintains an exponential moving average of model parameters.

    A shadow copy of the model is maintained internally; the live training
    model is never mutated.  Call ``update()`` every training step to advance
    the EMA, and use :meth:`get_state_dict` / :meth:`load_state_dict` for
    inference or checkpointing.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay  # typical values: 0.99 – 0.9999; <= 0 disables EMA
        self.shadow_model = _deep_clone_model(model)

    # ------------------------------------------------------------------
    # EMA book-keeping
    # ------------------------------------------------------------------

    def update(self, model: nn.Module) -> None:
        """Blend *model*'s parameters into the shadow copy.

        ``ema_weight ← decay × ema + (1 - decay) × new``
        """
        if self.decay <= 0.0:
            return
        with torch.no_grad():
            for s_param, t_param in zip(
                self.shadow_model.parameters(), model.parameters()
            ):
                s_param.copy_(s_param * self.decay + t_param * (1 - self.decay))

    # ------------------------------------------------------------------
    # Checkpointing / inference helpers
    # ------------------------------------------------------------------

    def get_state_dict(self) -> Dict[str, Any]:
        """Return the shadow model's state dict for saving to disk."""
        return {
            "decay": self.decay,
            "ema_state": self.shadow_model.state_dict(),
        }

    @classmethod
    def load_ema_state(
        cls, model: nn.Module, ema_state: Dict[str, Any]
    ) -> "ExponentialMovingAverage":
        """Create an EMA instance whose shadow starts from *ema_state*.

        Parameters
        ----------
        model :
            The live training model (used only to create the initial shadow).
        ema_state :
            A dict previously returned by :meth:`get_state_dict`.
        """
        decay = ema_state["decay"]
        # Remove the "decay" key before loading into the EMA's internal shadow.
        state_copy = ema_state["ema_state"]

        ema = cls(model, decay)  # fresh shadow from live model
        with torch.no_grad():
            ema.shadow_model.load_state_dict(state_copy)  # overwrite shadow weights
        return ema
