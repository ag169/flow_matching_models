import torch
import unittest
from models import m000_unet

# pytest .\tests\test_model_000_unet.py


class TestConditionEmbedding(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 32
        self.batch_size = 4

    def test_condition_embedding_forward(self):
        """Tests ConditionEmbedding module with time and class embeddings."""
        # Setup: Create embedding modules
        embed_module = m000_unet.ConditionEmbedding(self.embed_dim).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Inputs: Time embedding [B, D] and Class embedding [B, D]
        t_emb = torch.rand(self.batch_size, self.embed_dim)
        c_emb = torch.rand(self.batch_size, self.embed_dim)

        output = embed_module(t_emb, c_emb)

        # Expected: Output shape should be [B, D]
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([self.batch_size, self.embed_dim]))

    def test_condition_embedding_with_zero_class(self):
        """Tests ConditionEmbedding with zero class embedding (unconditional)."""
        embed_module = m000_unet.ConditionEmbedding(self.embed_dim)

        # Zero class embedding for unconditional case
        t_emb = torch.rand(2, self.embed_dim)
        c_emb = torch.zeros(2, self.embed_dim)  # Unconditional

        output = embed_module(t_emb, c_emb)

        self.assertEqual(output.shape, torch.Size([2, self.embed_dim]))


class TestCFMResNetBlock(unittest.TestCase):

    def setUp(self):
        self.in_channels = 64
        self.out_channels = 32
        self.emb_dim = 128
        self.batch_size = 2

    def test_cfm_resnet_block_forward(self):
        """Tests CFMResNetBlock with conditioning injection."""
        # Setup: Create ResNet block
        resnet_block = m000_unet.CFMResNetBlock(
            self.in_channels, self.out_channels, self.emb_dim
        )

        # Inputs: x [B, C, H, W], emb [B, D]
        B = self.batch_size
        H = 8
        W = 8
        x = torch.rand(B, self.in_channels, H, W)
        emb = torch.rand(B, self.emb_dim)

        output = resnet_block(x, emb)

        # Expected: Output shape should be [B, out_channels, H, W]
        self.assertEqual(output.shape, torch.Size([B, self.out_channels, H, W]))

    def test_cfm_resnet_block_identity_shortcut(self):
        """Tests CFMResNetBlock when in_channels == out_channels (Identity shortcut)."""
        resnet_block = m000_unet.CFMResNetBlock(64, 64, 128)

        B = 2
        H = 8
        W = 8
        x = torch.rand(B, 64, H, W)
        emb = torch.rand(B, 128)

        output = resnet_block(x, emb)

        self.assertEqual(output.shape, torch.Size([B, 64, H, W]))

    def test_cfm_resnet_block_conv_shortcut(self):
        """Tests CFMResNetBlock when in_channels != out_channels (Conv shortcut)."""
        # Change channels to force conv shortcut
        resnet_block = m000_unet.CFMResNetBlock(128, 64, 128)

        B = 2
        H = 8
        W = 8
        x = torch.rand(B, 128, H, W)
        emb = torch.rand(B, 128)

        output = resnet_block(x, emb)

        self.assertEqual(output.shape, torch.Size([B, 64, H, W]))


class TestEncoderStage(unittest.TestCase):

    def setUp(self):
        self.c_in = 32
        self.c_out = 64
        self.embed_dim = 128

    def test_encoder_stage_forward(self):
        """Tests EncoderStage with multiple ResNet blocks and downsampling."""
        encoder = m000_unet.EncoderStage(
            self.c_in, self.c_out, self.embed_dim, num_blocks=2
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        B = 1
        H = 16
        W = 16
        x = torch.rand(B, self.c_in, H, W)
        emb = torch.rand(B, self.embed_dim)

        output = encoder(x, emb)

        # Expected: Output is downsampled by Conv2d (4, 2, padding=0) -> shape [B, c_out, H/2, W/2]
        expected_h = H // 2
        expected_w = W // 2
        self.assertEqual(
            output.shape, torch.Size([B, self.c_out, expected_h, expected_w])
        )


class TestDecoderStage(unittest.TestCase):

    def setUp(self):
        self.c_in = 64
        self.c_out = 32
        self.embed_dim = 128

    def test_decoder_stage_forward(self):
        """Tests DecoderStage with upsampling via ConvTranspose2d."""
        decoder = m000_unet.DecoderStage(
            self.c_in, self.c_out, self.embed_dim, num_blocks=2
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        B = 1
        H = 8
        W = 4
        x = torch.rand(B, self.c_in, H, W)
        emb = torch.rand(B, self.embed_dim)

        output = decoder(x, emb)

        # Expected: Upsampled by ConvTranspose2d (4, 2, padding=0) -> shape [B, c_out, H*2, W*2]
        expected_h = H * 2
        expected_w = W * 2
        self.assertEqual(
            output.shape, torch.Size([B, self.c_out, expected_h, expected_w])
        )


class TestFlowMatchingUNet(unittest.TestCase):

    def setUp(self):
        self.c_in = 3
        self.num_classes = 5
        self.channels = (16, 32, 64)
        self.embed_dim = 128

    def test_flow_matching_unet_forward(self):
        """Tests FlowMatchingUNet complete forward pass."""
        unet = m000_unet.FlowMatchingUNet(
            c_in=self.c_in,
            num_classes=self.num_classes,
            channels=self.channels,
            embed_dim=self.embed_dim,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        # Inputs: x [B, C, H, W], t [B], cls [B]
        B = 2
        H = 16
        W = 16

        x = torch.rand(B, self.c_in, H, W)
        t = torch.randint(0, 100, (B,)).float() / 100.0  # Normalized time [0, 1]
        cls = torch.randint(0, self.num_classes + 1, (B,))

        output = unet(x, t, cls)

        # Expected: Output shape should be [B, c_in, H, W]
        self.assertEqual(output.shape, torch.Size([B, self.c_in, H, W]))

    def test_flow_matching_unet_with_zero_class(self):
        """Tests FlowMatchingUNet with zero class (unconditional generation)."""
        unet = m000_unet.FlowMatchingUNet(3, 5, embed_dim=128)

        B = 2
        H = 16
        W = 16

        x = torch.rand(B, 3, H, W)
        t = torch.zeros(B).float()  # Time at zero
        cls = torch.zeros(B, dtype=torch.long)  # Zero class (unconditional)

        output = unet(x, t, cls)

        self.assertEqual(output.shape, torch.Size([B, 3, H, W]))

    def test_flow_matching_unet_different_channels(self):
        """Tests FlowMatchingUNet with custom channel configuration."""
        channels = (16, 32, 64)
        unet = m000_unet.FlowMatchingUNet(3, 5, channels=channels, embed_dim=64)

        B = 2
        H = 16
        W = 16

        x = torch.rand(B, 3, H, W)
        t = torch.rand(B).float() / 100.0
        cls = torch.randint(0, 6, (B,))

        output = unet(x, t, cls)

        self.assertEqual(output.shape, torch.Size([B, 3, H, W]))


class TestModelIntegration(unittest.TestCase):

    def setUp(self):
        """Set up a complete model for integration testing."""
        self.model = m000_unet.FlowMatchingUNet(
            c_in=3, num_classes=5, channels=(16, 32, 64), embed_dim=64
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def test_full_pipeline_forward(self):
        """Tests the full forward pipeline with realistic inputs."""
        B = 4
        H = 32
        W = 32

        # Random image input
        x = torch.rand(B, 3, H, W)

        # Random time steps normalized to [0, 1]
        t = torch.rand(B).float()

        # Random class labels (including zero for unconditional)
        cls = torch.randint(0, 6, (B,))

        # Forward pass
        output = self.model(x, t, cls)

        # Verify output dimensions
        self.assertEqual(output.shape, torch.Size([B, 3, H, W]))

        # Verify gradient flow (optional but good for testing)
        output.sum().backward()
        self.assertTrue(self.model.conv_in.weight.grad is not None)


if __name__ == "__main__":
    unittest.main()
