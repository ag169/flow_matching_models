import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


def pad_to_align(x: torch.Tensor, dims_multiple_of: int = 8) -> torch.Tensor:
    x_shape = x.shape

    pad_list = list()

    for ii in range(x.ndim - 1, 1, -1):
        pad_list.append(0)
        padded_dim = (
            (x_shape[ii] + dims_multiple_of - 1) // dims_multiple_of
        ) * dims_multiple_of
        pad_list.append(padded_dim - x_shape[ii])

    return F.pad(x, pad_list)


class ConditionEmbedding(nn.Module):
    """
    Embeds continuous time 't' and categorical/text conditioning 'c'
    into a joint embedding space.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.time_mlp = MLP(embed_dim, embed_dim, transpose_dim=False)
        self.act = nn.SiLU()
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        # t_emb shape: [B, embed_dim], c_emb shape: [B, embed_dim]
        t_emb = self.time_mlp(t_emb)
        return self.final_proj(self.act(t_emb + c_emb))


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


class ChannelRMSNorm(nn.Module):
    def __init__(
        self, in_dim: int, elementwise_affine: bool = True, transpose_dim: bool = True
    ):
        super().__init__()

        self.transpose_dim = transpose_dim
        self.norm = nn.RMSNorm(
            normalized_shape=[
                in_dim,
            ],
            eps=1.0e-5,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transpose_dim:
            x = torch.transpose(x, 1, x.ndim - 1)

        out = self.norm(x)

        if self.transpose_dim:
            out = torch.transpose(out, x.ndim - 1, 1)

        return out


class MHCA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        head_dim: int = 64,
        num_heads: Optional[int] = None,
        is_gated: bool = True,
        qk_norm: bool = True,
        transpose_dim: bool = True,
    ):
        super().__init__()

        if num_heads is None:
            num_heads = in_dim // head_dim

        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.total_dim = head_dim * num_heads

        self.is_gated = is_gated
        self.qk_norm = qk_norm
        self.transpose_dim = transpose_dim

        self.qkv_dense = nn.Linear(in_dim, 3 * self.total_dim, bias=False)

        if self.is_gated:
            self.gate_dense = nn.Linear(in_dim, self.total_dim, bias=True)
        else:
            self.gate_dense = None

        if self.qk_norm:
            self.q_norm = ChannelRMSNorm(
                self.head_dim, elementwise_affine=True, transpose_dim=False
            )
            self.k_norm = ChannelRMSNorm(
                self.head_dim, elementwise_affine=True, transpose_dim=False
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.op_dense = nn.Linear(self.total_dim, in_dim, bias=True)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4]

        if self.transpose_dim:
            x = torch.transpose(x, 1, x.ndim - 1)

        x_shape = x.shape

        if x.ndim == 4:
            x = x.reshape((x_shape[0], x_shape[1] * x_shape[2], x_shape[3]))

        return x

    def _reshape_output(self, x: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
        if self.transpose_dim:
            x = torch.transpose(x, x.ndim - 1, 1)

        if len(orig_shape) != 3:
            x = x.reshape(orig_shape)

        return x

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        assert x.ndim == 3
        assert x_shape[2] == self.total_dim
        x = x.reshape((x_shape[0], x_shape[1], self.num_heads, self.head_dim))
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x_shape = x.shape
        return x.reshape((x_shape[0], x_shape[1], self.total_dim))

    def _compute_attn(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_dense(x)
        q, k, v = torch.split(qkv, self.total_dim, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.qk_norm:
            assert self.q_norm is not None
            assert self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)
            qk_scale = 1.0
        else:
            qk_scale = math.sqrt(self.head_dim)

        attn_op = F.scaled_dot_product_attention(q, k, v, scale=qk_scale)
        attn_op = self._merge_heads(attn_op)
        return attn_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x_rs = self._reshape_input(x)
        attn_out = self._compute_attn(x_rs)

        if self.is_gated:
            assert self.gate_dense is not None
            attn_gate = self.gate_dense(x_rs)
            attn_out = torch.sigmoid(attn_gate) * attn_out

        attn_out = self.op_dense(attn_out)
        return self._reshape_output(attn_out, x_shape)
