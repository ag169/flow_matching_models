import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence

from . import model_utils as mu


class ConditionEmbedding(nn.Module):
    """
    Embeds continuous time 't' and categorical/text conditioning 'c'
    into a joint embedding space.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.time_mlp = mu.MLP(embed_dim, embed_dim, transpose_dim=False)
        self.act = nn.SiLU()
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        # t_emb shape: [B, embed_dim], c_emb shape: [B, embed_dim]
        t_emb = self.time_mlp(t_emb)
        return self.final_proj(self.act(t_emb + c_emb))


class CFMResNetBlock(nn.Module):
    """
    A ResNet block that injects conditioning features via
    Adaptive Group Normalization (AdaGN).
    """

    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(in_channels // 8, in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Conditioning projection to generate scale (gamma) and shift (beta)
        self.cond_proj_1 = nn.Linear(emb_dim, 2 * in_channels)
        nn.init.zeros_(self.cond_proj_1.weight)
        nn.init.zeros_(self.cond_proj_1.bias)

        self.cond_proj_2 = nn.Linear(emb_dim, out_channels)
        nn.init.zeros_(self.cond_proj_2.weight)
        nn.init.zeros_(self.cond_proj_2.bias)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection for residual
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = F.silu(emb)
        cond_1 = self.cond_proj_1(emb)[..., None, None]
        scale, shift = torch.chunk(cond_1, 2, dim=1)
        alpha = self.cond_proj_2(emb)[..., None, None]

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(self.conv1(h)))
        h = h * alpha

        return h + self.shortcut(x)


class EncoderStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, embed_dim: int, num_blocks: int = 1):
        super().__init__()
        block_list = list()
        for _ in range(num_blocks):
            block_list.append(CFMResNetBlock(c_in, c_in, embed_dim))
        self.resblocks = nn.ModuleList(block_list)
        self.conv_out = nn.Conv2d(c_in, c_out, 4, 2, padding=1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, emb)
        return self.conv_out(x)


class DecoderStage(nn.Module):
    def __init__(self, c_in: int, c_out: int, embed_dim: int, num_blocks: int = 1):
        super().__init__()

        self.conv_transpose_in = nn.ConvTranspose2d(c_in, c_out, 4, 2, padding=1)
        block_list = list()
        for _ in range(num_blocks):
            block_list.append(CFMResNetBlock(c_out, c_out, embed_dim))
        self.resblocks = nn.ModuleList(block_list)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose_in(x)
        for block in self.resblocks:
            x = block(x, emb)
        return x


class FlowMatchingUNet(nn.Module):
    def __init__(
        self,
        c_in: int,
        num_classes: int,
        channels: Sequence[int] = (32, 128, 256),
        num_blocks_per_stage: int = 2,
        num_bottleneck_resblocks: int = 2,
        embed_dim: int = 128,
    ):
        super().__init__()

        self.n_classes = num_classes

        self.dims_multiple_of = 2 ** (len(channels))

        self.timestep_embed = mu.SinusoidalTimestepEmbed(embed_dim)
        # Also learn unconditional embedding for classifier-free guidance.
        self.cls_embed = nn.Embedding(self.n_classes + 1, embed_dim)
        self.embed_module = ConditionEmbedding(embed_dim)

        self.conv_in = nn.Conv2d(c_in, channels[0], kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels[0], c_in, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)
        self.norm_out = nn.GroupNorm(channels[0] // 8, channels[0])
        self.rb_out = CFMResNetBlock(channels[0], channels[0], embed_dim)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for ii in range(len(channels) - 1):
            c_in = channels[ii]
            c_out = channels[ii + 1]

            self.enc_blocks.append(
                EncoderStage(c_in, c_out, embed_dim, num_blocks_per_stage)
            )
            self.dec_blocks.append(
                DecoderStage(c_out, c_in, embed_dim, num_blocks_per_stage)
            )
            self.skip_convs.append(nn.Conv2d(c_in, c_in, kernel_size=3, padding=1))

        self.bottleneck_resblocks = nn.ModuleList()
        for _ in range(num_bottleneck_resblocks):
            self.bottleneck_resblocks.append(
                CFMResNetBlock(channels[-1], channels[-1], embed_dim)
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
        x_in = self.conv_in(x_padded)

        x_list = list()
        x_out = x_in
        for block in self.enc_blocks:
            x_list.append(x_out)
            x_out = block(x_out, cond_emb)

        for block in self.bottleneck_resblocks:
            x_out = block(x_out, cond_emb)

        num_stages = len(self.enc_blocks)
        for ii in range(num_stages - 1, -1, -1):
            x_out = self.dec_blocks[ii](x_out, cond_emb)
            x_out = x_out + self.skip_convs[ii](x_list[ii])

        x_out = self.rb_out(x_out, cond_emb)
        x_out = self.norm_out(x_out)
        v_out = self.conv_out(x_out)
        v_out = v_out[:, :, : x_shape[2], : x_shape[3]]
        return v_out
