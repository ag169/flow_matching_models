"""
Transformer model for flow-matching.

Uses pure transformer blocks with no spatial convs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import model_utils as mu


class DiTBlock(nn.Module):
    """
    Single transformer block with MHCA attention and MLP FFN.

    Uses AdaLN to inject conditioning into the pre-attention layer norm,
    and applies gating to both attention and MLP outputs.

    Args:
        in_dim: Input dimensionality.
        embed_dim: Dimensionality of the conditioning embedding.
    """

    def __init__(self, in_dim: int, embed_dim: int, attn_head_dim: int = 64):
        super().__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.embed_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.cond_proj = nn.Linear(embed_dim, 6 * in_dim, bias=True)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm1 = mu.ChannelLayerNorm(
            in_dim, elementwise_affine=False, transpose_dim=False
        )
        self.norm2 = mu.ChannelLayerNorm(
            in_dim, elementwise_affine=False, transpose_dim=False
        )

        self.attention = mu.MHCA(
            in_dim=in_dim,
            head_dim=min(in_dim, attn_head_dim),
            num_heads=None,
            qk_norm=True,
            is_gated=True,
            transpose_dim=False,
        )

        self.mlp = mu.MLP(
            in_dim=in_dim, mid_dim=in_dim * 4, is_gated=False, transpose_dim=False
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, L, in_dim].
            emb: Conditioning embedding of shape [B, embed_dim].

        Returns:
            Output after residual connection and gating.
        """
        assert x.ndim == 3
        assert emb.ndim == 2

        emb = emb[:, None, :]
        emb = self.embed_proj(emb)
        emb = F.silu(emb)
        cond_emb = self.cond_proj(emb)
        a1, b1, c1, a2, b2, c2 = torch.chunk(cond_emb, chunks=6, dim=2)

        x1 = self.norm1(x)
        h1 = (x1 * (1 + c1)) + b1
        attn_out = self.attention(h1)
        x = x + (attn_out * a1)

        x2 = self.norm2(x)
        h2 = (x2 * (1 + c2)) + b2
        mlp_out = self.mlp(h2)
        x = x + (mlp_out * a2)

        return x


class Transformer(nn.Module):
    """
    Multi-layer transformer encoder/decoder with configurable layers.

    Args:
        in_dim: Input dimensionality.
        num_layers: Number of transformer blocks to use.
        embed_dim: Dimensionality of the conditioning embedding.
    """

    def __init__(self, in_dim: int, num_layers: int = 12, embed_dim: int = 128):
        super().__init__()

        self.add_pos_embed = mu.SinusoidalPositionalEmbedding(dim=in_dim)
        self.layers = nn.ModuleList(
            [DiTBlock(in_dim=in_dim, embed_dim=embed_dim) for _ in range(num_layers)]
        )

    def forward(self, tokens_4d: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, dim, H, W].
            emb: Conditioning embedding of shape [B, embed_dim].

        Returns:
            Output tensor with the same shape as input.
        """
        x_4d = self.add_pos_embed(tokens_4d)

        x_4d_shape = x_4d.shape
        x = x_4d.reshape((x_4d_shape[0], x_4d_shape[1], x_4d_shape[2] * x_4d_shape[3]))
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x, emb)

        out = x.transpose(1, 2)
        out = out.reshape((x_4d_shape[0], x_4d_shape[1], x_4d_shape[2], x_4d_shape[3]))
        return out


class FlowMatchingTransformer(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_classes: int,
        num_tx_ch: int = 256,
        num_tx_blocks: int = 6,
        patchify_size: int = 4,
        embed_dim: int = 256,
    ):
        super().__init__()

        self.n_classes = num_classes

        self.dims_multiple_of = patchify_size

        self.timestep_embed = mu.SinusoidalTimestepEmbed(embed_dim)
        # Also learn unconditional embedding for classifier-free guidance.
        self.cls_embed = nn.Embedding(self.n_classes + 1, embed_dim)
        self.embed_module = mu.ConditionEmbedding(embed_dim)

        self.patchify = nn.Sequential(
            nn.PixelUnshuffle(downscale_factor=patchify_size),
            nn.Conv2d(
                c_in * patchify_size * patchify_size,
                num_tx_ch,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )

        self.tx_blocks = Transformer(
            num_tx_ch, num_layers=num_tx_blocks, embed_dim=embed_dim
        )

        self.out_block = nn.Sequential(
            mu.ChannelLayerNorm(
                num_tx_ch, transpose_dim=True, elementwise_affine=False
            ),
            nn.Conv2d(
                num_tx_ch,
                c_in * patchify_size * patchify_size,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            nn.PixelShuffle(upscale_factor=patchify_size),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cls: torch.Tensor):
        assert x.ndim == 4
        assert t.ndim == 1
        assert cls.ndim == 1

        t_emb = self.timestep_embed(t)
        cls_emb = self.cls_embed(cls)
        cond_emb = self.embed_module(t_emb, cls_emb)

        x_shape = x.shape
        x_padded = mu.pad_to_align(x, self.dims_multiple_of)

        x_patches = self.patchify(x_padded)
        x_txed = self.tx_blocks(x_patches, cond_emb)
        out = self.out_block(x_txed)

        out = out[:, :, : x_shape[2], : x_shape[3]]
        return out
