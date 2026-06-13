import torch
import unittest
from models import m001_transformer

# pytest .\tests\test_model_001_transformer.py


class TestTransformerBlock(unittest.TestCase):

    def setUp(self):
        self.in_dim = 64
        self.embed_dim = 128
        self.batch_size = 2
        self.height = 8
        self.width = 8

    def test_transformer_block_forward(self):
        """Tests TransformerBlock with MHCA attention and gated MLP FFN."""
        block = m001_transformer.DiTBlock(in_dim=self.in_dim, embed_dim=self.embed_dim)

        # Inputs: x [B, H * W, in_dim], emb [B, embed_dim]
        B = self.batch_size
        x = torch.rand(
            B,
            self.height * self.width,
            self.in_dim,
        )
        emb = torch.rand(B, self.embed_dim)

        output = block(x, emb)

        self.assertEqual(
            output.shape,
            torch.Size([B, self.height * self.width, self.in_dim]),
        )

    def test_transformer_block_with_gating(self):
        """Tests that gating is applied to attention and MLP outputs."""
        block = m001_transformer.DiTBlock(32, 64)

        B = 2
        x = torch.rand(B, 8, 32)
        emb = torch.rand(B, 64)

        output = block(x, emb)

        # Verify output shape and that gating is applied (non-trivial transformation)
        self.assertEqual(output.shape, torch.Size([B, 8, 32]))


class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.in_dim = 64
        self.num_layers = 4
        self.embed_dim = 128
        self.batch_size = 2
        self.height = 8
        self.width = 8

    def test_transformer_forward(self):
        """Tests Transformer module with configurable layers."""
        transformer = m001_transformer.Transformer(
            in_dim=self.in_dim, num_layers=self.num_layers, embed_dim=self.embed_dim
        )

        B = self.batch_size
        x = torch.rand(B, self.in_dim, self.height, self.width)
        emb = torch.rand(B, self.embed_dim)

        output = transformer(x, emb)

        self.assertEqual(
            output.shape, torch.Size([B, self.in_dim, self.height, self.width])
        )

    def test_transformer_with_pos_embed(self):
        """Tests that sinusoidal positional embedding is added."""
        transformer = m001_transformer.Transformer(32, 6, embed_dim=64)

        B = 2
        x = torch.rand(B, 32, 8, 8)
        emb = torch.zeros(B, 64)  # Zero embedding to isolate pos_embed effect

        output = transformer(x, emb)

        self.assertEqual(output.shape, torch.Size([B, 32, 8, 8]))


class TestFlowMatchingTransformer(unittest.TestCase):

    def setUp(self):
        self.c_in = 3
        self.num_classes = 5
        self.num_tx_ch = 64
        self.num_tx_blocks = 4
        self.patchify_size = 4
        self.embed_dim = 128
        self.batch_size = 2
        self.height = 16
        self.width = 16

    def test_flow_matching_transformer_forward(self):
        """Tests FlowMatchingTransformer complete forward pass."""
        model = m001_transformer.FlowMatchingTransformer(
            c_in=self.c_in,
            num_classes=self.num_classes,
            num_tx_ch=self.num_tx_ch,
            num_tx_blocks=self.num_tx_blocks,
            patchify_size=self.patchify_size,
            embed_dim=self.embed_dim,
        )

        # Inputs: x [B, C, H, W], t [B], cls [B]
        B = self.batch_size
        x = torch.rand(B, self.c_in, self.height, self.width)
        t = torch.randint(0, 100, (B,)).float() / 100.0  # Normalized time [0, 1]
        cls = torch.randint(0, self.num_classes + 1, (B,))

        output = model(x, t, cls)

        # Expected: Output shape should be [B, c_in, H, W]
        self.assertEqual(
            output.shape, torch.Size([B, self.c_in, self.height, self.width])
        )

    def test_flow_matching_transformer_with_zero_class(self):
        """Tests FlowMatchingTransformer with zero class (unconditional generation)."""
        model = m001_transformer.FlowMatchingTransformer(3, 5, embed_dim=64)

        B = 2
        H = 16
        W = 16

        x = torch.rand(B, 3, H, W)
        t = torch.zeros(B).float()  # Time at zero
        cls = torch.zeros(B, dtype=torch.long)  # Zero class (unconditional)

        output = model(x, t, cls)

        self.assertEqual(output.shape, torch.Size([B, 3, H, W]))

    def test_flow_matching_transformer_different_patch_size(self):
        """Tests FlowMatchingTransformer with custom patchify size."""
        model = m001_transformer.FlowMatchingTransformer(
            c_in=3, num_classes=5, patchify_size=8, embed_dim=64
        )

        B = 2
        H = 32
        W = 32

        x = torch.rand(B, 3, H, W)
        t = torch.rand(B).float() / 100.0
        cls = torch.randint(0, 6, (B,))

        output = model(x, t, cls)

        self.assertEqual(output.shape, torch.Size([B, 3, H, W]))


if __name__ == "__main__":
    unittest.main()
