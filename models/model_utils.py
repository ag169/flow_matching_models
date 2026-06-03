import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


def pad_to_align(x: torch.Tensor, dims_multiple_of: int = 8) -> torch.Tensor:
    x_shape = x.shape

    pad_list = list()

    for ii in range(x.ndim - 1, 1, -1):
        pad_list.append(0)
        pad_list.append(
            dims_multiple_of * (x_shape[ii] + dims_multiple_of - 1) // dims_multiple_of
        )

    return F.pad(x, pad_list)


class SinusoidalTimestepEmbed(nn.Module):
    """
    Generates sinusoidal positional embeddings for [0, 1] normalized time steps 't'.
    Used for injecting timestep conditioning into the model.
    """

    def __init__(self, dim: int, max_period: int = 500) -> None:
        super().__init__()
        self.dim = dim
        self.max_p = max_period

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        freqs = torch.exp(
            torch.arange(0, self.dim, 2, device=time.device, dtype=torch.float)
            * (-math.log(self.max_p) / self.dim)
        )
        # time shape: (B,) -> expanded to (B, D/2)
        embeddings = time[:, None] * freqs[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=1)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Generates sinusoidal positional embeddings for spatial tokens.
    """

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_p = max_period

    def _pos_embed_for_seq_len(self, seq_len: int, x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float, device=x.device)
        freqs = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float, device=x.device)
            * (-math.log(self.max_p) / self.dim)
        )
        embeddings = position[None, :] * freqs[:, None]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 3:
            seq_len = tokens.shape[2]
            pos_embed = self._pos_embed_for_seq_len(seq_len, tokens)
            return tokens + pos_embed[None, :, :]
        elif tokens.ndim == 4:
            seq_h = tokens.shape[2]
            pos_embed_h = self._pos_embed_for_seq_len(seq_h, tokens)

            seq_w = tokens.shape[3]
            pos_embed_w = self._pos_embed_for_seq_len(seq_w, tokens)

            return (
                tokens + pos_embed_h[None, :, :, None] + pos_embed_w[None, :, None, :]
            )
        else:
            raise ValueError("Input tensor must have 3 or 4 dimensions")


class LearnedPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int, embed_dim: Sequence[int] = (8, 8), init_scale: float = 1.0e-2
    ) -> None:
        super().__init__()
        self.dim = dim

        embed_dim_list = [dim]
        embed_dim_list.extend(embed_dim)
        self.pos_embed = nn.Parameter(
            (init_scale * torch.randn(embed_dim_list)), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_shape = list(x.shape)
        target_dims = token_shape[2:]

        if x.ndim == 3:
            mode = "linear"
        elif x.ndim == 4:
            mode = "bilinear"
        elif x.ndim == 5:
            mode = "trilinear"
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}")

        pos_embed = self.pos_embed[None, ...].to(x.device)
        interpolated_embed = F.interpolate(
            pos_embed, target_dims, mode=mode, antialias=False
        )

        return x + interpolated_embed


class ActivationModule(nn.Module):
    def __init__(self, activation: str | None = "swish"):
        super().__init__()
        activation = activation.lower() if activation else None

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "relu6":
            self.activation = nn.ReLU6()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish" or activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "hardswish":
            self.activation = nn.Hardswish()
        elif activation == "hardsigmoid":
            self.activation = nn.Hardsigmoid()
        elif activation == "" or activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation function {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int = -1,
        activation: str = "swish",
        is_gated: bool = False,
        transpose_dim: bool = True,
    ):
        super().__init__()

        self.is_gated = is_gated
        self.transpose_dim = transpose_dim

        if out_dim == -1:
            out_dim = in_dim

        if self.is_gated:
            self.linear_0 = nn.Linear(in_dim, mid_dim * 2)
        else:
            self.linear_0 = nn.Linear(in_dim, mid_dim)

        self.activation = ActivationModule(activation)

        self.linear_1 = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose_dim:
            x = torch.transpose(x, 1, x.ndim - 1)

        out = self.linear_0(x)

        if self.is_gated:
            mid_dim = out.size(-1) // 2
            out = out[..., :mid_dim] * torch.sigmoid(out[..., mid_dim:])

        out = self.linear_1(out)
        if self.transpose_dim:
            out = torch.transpose(out, x.ndim - 1, 1)

        return out
