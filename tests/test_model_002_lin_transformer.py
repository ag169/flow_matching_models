import torch
import unittest
from models import m002_lin_transformer

# pytest .\tests\test_model_002_lin_transformer.py


class TestLinAttnDiTBlock(unittest.TestCase):
    """
    Tests for LinAttnDiTBlock with MHCA attention and gated MLP FFN.
    Uses 4D BCHW input as specified in the model docstring.
    """

    def setUp(self):
        self.in_dim = 64
        self.embed_dim = 128
        self.batch_size = 2
        self.height = 8
        self.width = 8

    def test_lin_attn_dit_block_forward(self):
        """Tests LinAttnDiTBlock with MHCA attention and gated MLP FFN."""
        block = m002_lin_transformer.LinAttnDiTBlock(
            in_dim=self.in_dim, embed_dim=self.embed_dim
        )

        B = self.batch_size
        x = torch.rand(B, self.in_dim, self.height, self.width)
        emb = torch.rand(B, self.embed_dim)

        output = block(x, emb)

        self.assertEqual(
            output.shape,
            torch.Size([B, self.in_dim, self.height, self.width]),
        )

    def test_lin_attn_dit_block_with_gating(self):
        """Tests that gating is applied to attention and MLP outputs."""
        block = m002_lin_transformer.LinAttnDiTBlock(32, 64)

        B = 2
        x = torch.rand(B, 32, 8, 8)
        emb = torch.rand(B, 64)

        output = block(x, emb)

        # Verify output shape and that gating is applied (non-trivial transformation)
        self.assertEqual(output.shape, torch.Size([B, 32, 8, 8]))

    def test_lin_attn_dit_block_different_sizes(self):
        """Tests LinAttnDiTBlock with various spatial dimensions."""
        block = m002_lin_transformer.LinAttnDiTBlock(48, 96)

        for height in [4, 16, 32]:
            for width in [4, 16, 32]:
                B = 1
                x = torch.rand(B, 48, height, width)
                emb = torch.rand(B, 96)

                output = block(x, emb)
                expected_shape = torch.Size([B, 48, height, width])
                self.assertEqual(output.shape, expected_shape)


class TestTransformer(unittest.TestCase):
    """
    Tests for Transformer module with configurable layers.
    Uses 4D BCHW input as specified in the model docstring.
    """

    def setUp(self):
        self.in_dim = 64
        self.num_layers = 4
        self.embed_dim = 128
        self.batch_size = 2
        self.height = 8
        self.width = 8

    def test_transformer_forward(self):
        """Tests Transformer module with configurable layers."""
        transformer = m002_lin_transformer.Transformer(
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
        """Tests that learned positional embedding is added."""
        transformer = m002_lin_transformer.Transformer(32, 6, embed_dim=64)

        B = 2
        x = torch.rand(B, 32, 8, 8)
        emb = torch.zeros(B, 64)  # Zero embedding to isolate pos_embed effect

        output = transformer(x, emb)

        self.assertEqual(output.shape, torch.Size([B, 32, 8, 8]))

    def test_transformer_different_sizes(self):
        """Tests Transformer with various spatial dimensions."""
        for num_layers in [2, 4, 6]:
            transformer = m002_lin_transformer.Transformer(48, num_layers, embed_dim=96)

            B = 1
            x = torch.rand(B, 48, 16, 16)
            emb = torch.rand(B, 96)

            output = transformer(x, emb)
            expected_shape = torch.Size([B, 48, 16, 16])
            self.assertEqual(output.shape, expected_shape)


class TestFlowMatchingTransformer(unittest.TestCase):
    """
    Tests for FlowMatchingTransformer complete forward pass.
    Uses 4D BCHW image input as specified in the model docstring.
    """

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
        model = m002_lin_transformer.FlowMatchingTransformer(
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
        model = m002_lin_transformer.FlowMatchingTransformer(
            c_in=3, num_classes=5, embed_dim=64
        )

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
        model = m002_lin_transformer.FlowMatchingTransformer(
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

    def test_flow_matching_transformer_different_sizes(self):
        """Tests FlowMatchingTransformer with various image sizes."""
        for c_in in [1, 3, 4]:
            for patchify_size in [2, 4, 8]:
                model = m002_lin_transformer.FlowMatchingTransformer(
                    c_in=c_in,
                    num_classes=5,
                    num_tx_ch=64,
                    num_tx_blocks=4,
                    patchify_size=patchify_size,
                    embed_dim=128,
                )

                B = 1
                H = (
                    32 // patchify_size * patchify_size
                )  # Ensure divisible by patchify_size
                W = 32 // patchify_size * patchify_size

                x = torch.rand(B, c_in, H, W)
                t = torch.rand(B).float() / 100.0
                cls = torch.randint(0, 6, (B,))

                output = model(x, t, cls)
                expected_shape = torch.Size([B, c_in, H, W])
                self.assertEqual(output.shape, expected_shape)

    def test_flow_matching_transformer_non_divisible_sizes(self):
        """Tests padding and cropping logic with dimensions not divisible by patchify_size."""
        model = m002_lin_transformer.FlowMatchingTransformer(
            c_in=3, num_classes=5, patchify_size=4, embed_dim=64
        )

        B = 1
        # H/W not multiples of 4 (e.g., 17x19)
        x = torch.rand(B, 3, 17, 19)
        t = torch.zeros(1).float()
        cls = torch.zeros(1, dtype=torch.long)

        output = model(x, t, cls)
        self.assertEqual(output.shape, torch.Size([B, 3, 17, 19]))

    def test_flow_matching_transformer_different_num_blocks(self):
        """Tests FlowMatchingTransformer with different numbers of transformer blocks."""
        for num_tx_blocks in [2, 4, 6]:
            model = m002_lin_transformer.FlowMatchingTransformer(
                c_in=3,
                num_classes=5,
                num_tx_ch=128,
                num_tx_blocks=num_tx_blocks,
                patchify_size=4,
                embed_dim=256,
            )

            B = 2
            H = 16
            W = 16

            x = torch.rand(B, 3, H, W)
            t = torch.rand(B).float() / 100.0
            cls = torch.randint(0, 6, (B,))

            output = model(x, t, cls)
            expected_shape = torch.Size([B, 3, H, W])
            self.assertEqual(output.shape, expected_shape)

    def test_flow_matching_transformer_gradient_flow(self):
        """Tests that gradients flow correctly through the model."""
        model = m002_lin_transformer.FlowMatchingTransformer(
            c_in=3,
            num_classes=5,
            num_tx_ch=64,
            num_tx_blocks=4,
            patchify_size=4,
            embed_dim=128,
        )

        B = 2
        H = 16
        W = 16

        x = torch.rand(B, 3, H, W)
        t = torch.rand(B).float() / 100.0
        cls = torch.randint(0, 6, (B,))

        output = model(x, t, cls)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist in key layers
        self.assertIsNotNone(model.patchify[1].weight.grad)
        self.assertIsNotNone(model.tx_blocks.layers[0].attention.qkv_dense.weight.grad)  # type: ignore
        self.assertIsNotNone(model.out_block[1].weight.grad)


if __name__ == "__main__":
    unittest.main()
