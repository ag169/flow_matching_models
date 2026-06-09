"""
Transformer model for flow-matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import model_utils as mu


class AdaLNZero(nn.Module):
    """
    Conditional Adaptive Layer Norm for conditioning injection.

    Args:
        in_dim: Input dimensionality.
        embed_dim: Dimensionality of the conditioning embedding.
    """

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()

        self.embed_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.cond_proj = nn.Linear(embed_dim, 3 * in_dim, bias=True)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

        self.norm = nn.LayerNorm(in_dim)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [B, ..., in_dim].
            emb: Conditioning embedding of shape [B, embed_dim] or broadcastable.

        Returns:
            Normed output along with output scale to be applied after Attn/FFN.
        """
        emb = self.embed_proj(emb)
        emb = F.silu(emb)
        cond_proj = self.cond_proj(emb)
        alpha, beta, gamma = torch.chunk(cond_proj, 3, dim=1)  # [B, in_dim]

        alpha_shape = [1 for _ in range(x.ndim)]
        alpha_shape[0] = alpha.size(0)
        alpha_shape[-1] = alpha.size(-1)

        alpha = alpha.reshape(alpha_shape)
        beta = beta.reshape(alpha_shape)
        gamma = gamma.reshape(alpha_shape)

        x_norm = self.norm(x)
        x_norm = x_norm * (1 + gamma) + beta

        return x_norm, alpha


class TransformerBlock(nn.Module):
    """
    Single transformer block with MHCA attention and gated MLP FFN.

    Uses AdaLN to inject conditioning into the pre-attention layer norm,
    and applies gating to both attention and MLP outputs.

    Args:
        in_dim: Input dimensionality.
        embed_dim: Dimensionality of the conditioning embedding.
    """

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()

        self.norm1 = AdaLNZero(in_dim=in_dim, embed_dim=embed_dim)
        self.attention = mu.MHCA(
            in_dim=in_dim,
            head_dim=min(in_dim // 4, 128),
            num_heads=None,
            is_gated=True,
            transpose_dim=False,
        )

        self.norm2 = AdaLNZero(in_dim=in_dim, embed_dim=embed_dim)
        self.mlp = mu.MLP(in_dim=in_dim, mid_dim=in_dim * 4, transpose_dim=False)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_dim, ...].
            emb: Conditioning embedding of shape [B, embed_dim] or broadcastable.

        Returns:
            Output after residual connection and gating.
        """
        h1, alpha1 = self.norm1(x, emb)
        attn_out = self.attention(h1)
        x1 = x + (attn_out * alpha1)

        h2, alpha2 = self.norm2(x1, emb)
        mlp_out = self.mlp(h2)
        out = x1 + (mlp_out * alpha2)
        return out


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
            [
                TransformerBlock(in_dim=in_dim, embed_dim=embed_dim)
                for _ in range(num_layers)
            ]
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

        self.patchify_conv = nn.Conv2d(
            c_in, num_tx_ch, kernel_size=patchify_size, stride=patchify_size, bias=True
        )

        self.tx_blocks = Transformer(
            num_tx_ch, num_layers=num_tx_blocks, embed_dim=embed_dim
        )

        self.out_block = nn.Sequential(
            nn.GroupNorm(num_tx_ch // 16, num_tx_ch),
            nn.Conv2d(
                num_tx_ch,
                c_in * patchify_size * patchify_size,
                kernel_size=1,
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

        x_patches = self.patchify_conv(x_padded)
        x_txed = self.tx_blocks(x_patches, cond_emb)
        out = self.out_block(x_txed)

        out = out[:, :, : x_shape[2], : x_shape[3]]
        return out
