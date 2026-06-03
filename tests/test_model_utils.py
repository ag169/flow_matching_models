import torch
import unittest
from models import model_utils as mu

# pytest .\tests\test_model_utils.py


class TestPosEmbeds(unittest.TestCase):

    def setUp(self):
        # Set up common parameters for tests
        self.dim = 32
        self.batch_size = 4
        self.seq_len = 10
        self.embed_dim = (8, 8)

    def test_sinusoidal_timestep_embed(self):
        """Tests SinusoidalTimestepEmbed for time steps."""
        # Test case: Batch size of 4
        time_embed = mu.SinusoidalTimestepEmbed(dim=self.dim).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Input tensor shape: (B,)
        time_input = torch.rand(self.batch_size)

        output = time_embed(time_input)

        # Expected output shape: (B, D)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([self.batch_size, self.dim]))

    def test_sinusoidal_positional_embedding_sequence(self):
        """Tests SinusoidalPositionalEmbedding for 3D tokens (Sequence)."""
        # Test case: Batch size=4, Sequence length=10, Dim=128
        pos_embed = mu.SinusoidalPositionalEmbedding(dim=self.dim)

        # Input tensor shape: (B, L, D) -> e.g., token features
        tokens = torch.rand(self.batch_size, self.dim, self.seq_len)

        output = pos_embed(tokens)

        # Expected output shape remains the same after addition
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(
            output.shape, torch.Size([self.batch_size, self.dim, self.seq_len])
        )

    def test_sinusoidal_positional_embedding_spatial(self):
        """Tests SinusoidalPositionalEmbedding for 4D tokens (Spatial)."""
        # Test case: Batch=2, Channel/Head=8, Height=10, Width=12
        B = 2
        H = 10
        W = 12

        pos_embed = mu.SinusoidalPositionalEmbedding(dim=self.dim)

        # Input tensor shape: (B, C, H, W) or similar spatial structure where D is the last dim
        tokens = torch.rand(B, self.dim, H, W)

        output = pos_embed(tokens)
        # Expected output shape must match input shape
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, tokens.shape)

    def test_learned_positional_embedding(self):
        """Tests LearnedPositionalEmbedding with a generic input tensor."""
        # Test case: Batch=1, Channels=2, H=5, W=6
        B = 1
        C = self.dim
        H = 5
        W = 6

        pos_embed = mu.LearnedPositionalEmbedding(
            dim=self.dim, embed_dim=self.embed_dim
        )

        # Input tensor shape: (B, C, H, W)
        tokens = torch.rand(B, C, H, W)

        output = pos_embed(tokens)

        # Expected output shape must match input shape
        self.assertEqual(output.shape, tokens.shape)

    def test_learned_positional_embedding_simple(self):
        """Tests LearnedPositionalEmbedding with a 3D tensor (less common but good check)."""
        B = 1
        L = 6  # Sequence length
        C = self.dim

        pos_embed = mu.LearnedPositionalEmbedding(dim=self.dim, embed_dim=(3,))

        # Input tensor shape: (B, C, L) or similar reduced dimension setup
        tokens = torch.rand(B, C, L)

        output = pos_embed(tokens)

        # Expected output shape must match input shape
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, tokens.shape)


class TestMLP(unittest.TestCase):

    def setUp(self):
        self.dim = 32
        self.seq_len = 10

    def test_mlp_standard_2d(self):
        """Tests standard MLP on a 2D tensor (Batch size only)."""
        B = 5
        C = self.dim
        # Test with non-gated structure
        mlp = mu.MLP(in_dim=C, mid_dim=64, activation="swish", is_gated=False)
        tokens = torch.rand(B, C)

        output = mlp(tokens)
        # Expected output shape must match input batch size and the final feature dimension (which should be C if linear_1 maps back to C)
        self.assertEqual(output.shape, torch.Size([B, C]))

    def test_mlp_standard_3d(self):
        """Tests standard MLP on a 3D tensor (Sequence)."""
        B = 4
        L = self.seq_len
        C = self.dim
        # Test with non-gated structure and specific activation
        mlp = mu.MLP(in_dim=C, mid_dim=128, activation="gelu", is_gated=False)
        tokens = torch.rand(B, C, L)

        output = mlp(tokens)
        # Expected output shape must match input shape (assuming linear layer operates on the last dim and preserves other dims)
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([B, C, L]))

    def test_mlp_standard_4d(self):
        """Tests standard MLP on a 4D tensor (Spatial)."""
        B = 2
        H = self.seq_len // 2  # Using sequence length / 2 for H
        W = self.dim // 4  # Using dim / 4 for W
        C = self.dim

        # Test with non-gated structure
        mlp = mu.MLP(in_dim=C, mid_dim=128, activation="swish", is_gated=False)
        tokens = torch.rand(B, C, H, W)

        output = mlp(tokens)
        # Expected output shape must match input shape
        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, tokens.shape)

    def test_mlp_gated_3d_4d(self):
        """Tests gated MLP on 3D and 4D tensors (Sequence/Spatial)."""
        # --- Test 3D Gated ---
        B = 4
        L = self.seq_len
        C = self.dim
        mlp_3d = mu.MLP(in_dim=C, mid_dim=128, activation="swish", is_gated=True)
        tokens_3d = torch.rand(B, C, L)
        output_3d = mlp_3d(tokens_3d)
        self.assertEqual(output_3d.shape, tokens_3d.shape)

        # --- Test 4D Gated ---
        B = 2
        H = self.seq_len // 2
        W = self.dim // 4
        C = self.dim
        mlp_4d = mu.MLP(in_dim=C, mid_dim=128, activation="swish", is_gated=True)
        tokens_4d = torch.rand(B, C, H, W)
        output_4d = mlp_4d(tokens_4d)
        self.assertEqual(output_4d.shape, tokens_4d.shape)


if __name__ == "__main__":
    unittest.main()
