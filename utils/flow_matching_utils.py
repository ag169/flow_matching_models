import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# ============================================================================
# Time Sampling Strategies
# ============================================================================


class TimeSampler:
    """
    Base class for time sampling strategies in flow matching.

    In conditional flow matching, we sample time steps t ∈ [0, 1]
    to interpolate between data distribution and noise prior.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def __call__(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError


class UniformTimeSampler(TimeSampler):
    """
    Uniform sampling of time steps in [0, 1].

    This is the standard approach for conditional flow matching
    where all time steps are equally important.
    """

    def __init__(self, seed: Optional[int] = None):
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, generator=self.generator)


class ODETimeSampler(TimeSampler):
    """
    Time sampling optimized for ODE-based sampling.

    Samples time steps with higher density near t=1 where
    the vector field changes rapidly.
    """

    def __init__(self, sigma: float = 1.0, seed: Optional[int] = None):
        """
        Args:
            sigma: Controls the concentration of samples near t=1.
                   Lower values concentrate more samples near 1.
        """
        self.sigma = sigma

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, batch_size: int) -> torch.Tensor:
        # Sample from a distribution concentrated near 0
        # using log-normal-like distribution
        u = torch.rand(batch_size, generator=self.generator)
        t_zero_skew = self.sigma * u / (1.0 - u + 1e-8)
        t_zero_skew = torch.clamp(t_zero_skew, min=0.0, max=1.0)
        return 1.0 - t_zero_skew


# ============================================================================
# Flow Matching Loss Computation
# ============================================================================


def compute_flow_matching_loss(
    model: nn.Module,
    x_1: torch.Tensor,
    eps: torch.Tensor,
    t: torch.Tensor,
    cls: Optional[torch.Tensor] = None,
    cfg_dropout_prob: float = 0.1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute flow-matching loss with classifier-free guidance support.

    During training, randomly drop conditioning to enable
    classifier-free guidance at inference time.

    Args:
        model: Flow matching model
        x_1: Data samples [B, C, H, W] from original distribution
        eps: Noise samples [B, C, H, W]
        t: Time steps [B]
        cls: Class labels [B] (optional)
        cfg_dropout_prob: Probability of dropping conditioning

    Returns:
        Tuple of (loss, unconditional_loss) where unconditional_loss
        is only computed when CFG training is enabled
    """
    batch_size = x_1.shape[0]

    # x_t = (1 - t) * eps + (t * x_1)
    t_4d = t[:, None, None, None]
    x_t = ((1.0 - t_4d) * eps) + (t_4d * x_1)
    # v_t = d x_t / dt = x_1 - eps
    v_true = x_1 - eps

    # CLS 0 is null-class (unconditional)
    # CLS 1 onwards is for each class in the dataset.
    if cls is not None:
        cls = 1 + cls

    if cfg_dropout_prob > 0:
        assert cls is not None, "Class labels are required for CFG training"
        assert cfg_dropout_prob <= 1.0, "Dropout probability must be between [0, 1]"

        # Generate dropout mask for class labels.
        dropout_mask = torch.rand(batch_size, device=x_1.device) < cfg_dropout_prob

        # Conditional forward pass
        v_cond = model(x_t, t, cls)

        # Unconditional forward pass (use zero class for conditional models)
        cls_zero = torch.zeros_like(t).long()
        v_uncond = model(x_t, t, cls_zero)

        # Combine based on dropout mask
        v_pred = torch.where(dropout_mask[:, None, None, None], v_uncond, v_cond)

        # Compute loss with proper weighting
        loss_cond = torch.mean((v_cond - v_true) ** 2)
        loss_uncond = torch.mean((v_uncond - v_true) ** 2)
        loss = ((1 - cfg_dropout_prob) * loss_cond) + (cfg_dropout_prob * loss_uncond)

        return loss, loss_uncond

    # CFG free training
    if cls is not None:
        v_pred = model(x_t, t, cls)
    else:
        cls_zero = torch.zeros_like(t).long()
        v_pred = model(x_t, t, cls_zero)

    loss = torch.mean((v_pred - v_true) ** 2)

    return loss, None


# ============================================================================
# Multi-Step ODE Solvers for Inference
# ============================================================================


def _predict_velocity(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cls: Optional[torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    """Helper to predict velocity with optional CFG."""
    if cfg_scale > 1.0:
        assert cls is not None
        cls_zero = torch.zeros_like(t).long()

        batched_x = torch.cat([x, x], dim=0)
        batched_t = torch.cat([t, t], dim=0)
        batched_cls = torch.cat([cls + 1, cls_zero], dim=0)
        batched_v = model(batched_x, batched_t, batched_cls)

        v_cond, v_uncond = torch.chunk(batched_v, chunks=2, dim=0)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    if cls is None:
        cls = torch.zeros_like(t).long()
    else:
        cls = cls + 1

    return model(x, t, cls)


class EulerSolver:
    """
    Euler method ODE solver for flow matching inference.

    Simple first-order solver: x_{t+dt} = x_t + dt * v_θ(x_t, t)
    """

    def __init__(self, num_steps: int = 20):
        self.num_steps = num_steps

    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cls: Optional[torch.Tensor] = None,
        device: str | torch.device = "cpu",
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using Euler method.

        Args:
            shape: Desired output shape (batch_size, channels, height, width)
            cls: Class labels for conditional generation [B]
            device: Device to generate on
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            Generated samples [B, C, H, W]
        """
        # Initialize with standard normal noise (corresponds to t=0)
        x = torch.randn(shape, device=device)

        timesteps = torch.linspace(0.0, 1.0, self.num_steps + 1, device=device)
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            t = timesteps[i].unsqueeze(0).expand(x.shape[0])
            v_pred = _predict_velocity(model, x, t, cls, cfg_scale)
            # Euler update: x_{t+dt} = x_t + dt * v_θ(x_t, t)
            x = x + (dt * v_pred)

        return x


class DPMPSolver:
    """
    DPM-Solver++ style solver for flow matching.
    Adds third-order multistep updates for better quality.
    """

    def __init__(self, num_steps: int = 10, schedule: str = "quadratic"):
        self.num_steps = num_steps
        self.schedule = schedule

    def _adaptive_timesteps(self) -> List[float]:
        """Generate adaptive time steps."""
        timesteps = [0.0]
        for i in range(1, self.num_steps + 1):
            fraction = i / self.num_steps

            if self.schedule == "quadratic":
                t = fraction**2
            elif self.schedule == "linear":
                t = fraction
            else:  # cosine schedule
                t = math.sin((fraction * math.pi) / 2)

            timesteps.append(min(1.0, t))

        timesteps = list(set(timesteps))
        timesteps.sort()
        return timesteps

    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cls: Optional[torch.Tensor] = None,
        device: str | torch.device = "cpu",
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples using DPM-Solver++ style updates."""
        # Initialize with standard normal noise (corresponds to t=0)
        x = torch.randn(shape, device=device)
        timesteps = self._adaptive_timesteps()

        v_history = []  # store past velocities

        for i in range(len(timesteps) - 1):
            t_curr, t_next = timesteps[i], timesteps[i + 1]
            dt = t_next - t_curr

            t = torch.full((shape[0],), t_curr, device=device)
            v_pred = _predict_velocity(model, x, t, cls, cfg_scale)

            if i == 0:
                # Euler step
                x_next = x + dt * v_pred
            elif i == 1:
                # Second-order Adams-Bashforth step
                x_next = x + dt * ((1.5 * v_pred) - (0.5 * v_history[-1]))
            else:
                # Third-order Adams–Bashforth step
                x_next = x + dt * (
                    (23 / 12) * v_pred
                    - (16 / 12) * v_history[-1]
                    + (5 / 12) * v_history[-2]
                )

            v_history.append(v_pred)
            x = x_next

        return x
