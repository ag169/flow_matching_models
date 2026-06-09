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


class TestConditionEmbedding(unittest.TestCase):

    def setUp(self):
        self.embed_dim = 32
        self.batch_size = 4

    def test_condition_embedding_forward(self):
        """Tests ConditionEmbedding module with time and class embeddings."""
        # Setup: Create embedding modules
        embed_module = mu.ConditionEmbedding(self.embed_dim).to(
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
        embed_module = mu.ConditionEmbedding(self.embed_dim)

        # Zero class embedding for unconditional case
        t_emb = torch.rand(2, self.embed_dim)
        c_emb = torch.zeros(2, self.embed_dim)  # Unconditional

        output = embed_module(t_emb, c_emb)

        self.assertEqual(output.shape, torch.Size([2, self.embed_dim]))


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


class TestChannelRMSNorm:

    def test_channel_rmsnorm_basic_3d(self):
        """Verify output shape matches input for 3D tensor."""
        norm = mu.ChannelRMSNorm(in_dim=64, elementwise_affine=True)
        x = torch.randn(2, 64, 10)
        out = norm(x)
        assert out.shape == x.shape

    def test_channel_rmsnorm_basic_4d(self):
        """Verify output shape matches input for 4D tensor."""
        norm = mu.ChannelRMSNorm(in_dim=32, elementwise_affine=True)
        x = torch.randn(3, 32, 5, 6)
        out = norm(x)
        assert out.shape == x.shape

    def test_channel_rmsnorm_transpose_consistency(self):
        """transpose_dim=True/False should produce identical results."""
        x = torch.randn(4, 128, 7)
        x_nt = x.transpose(1, 2)

        norm_t = mu.ChannelRMSNorm(
            in_dim=128, elementwise_affine=False, transpose_dim=True
        )
        out_t = norm_t(x)

        norm_nt = mu.ChannelRMSNorm(
            in_dim=128, elementwise_affine=False, transpose_dim=False
        )
        out_nt = norm_nt(x_nt)
        out_nt = out_nt.transpose(1, 2)

        assert torch.allclose(out_t, out_nt, atol=1e-6)

    def test_channel_rmsnorm_elementwise_affine(self):
        """With affine=True the output should differ from raw input."""
        x = torch.randn(2, 32, 8)

        norm_on = mu.ChannelRMSNorm(in_dim=32, elementwise_affine=True)
        out_on = norm_on(x)
        assert not torch.allclose(out_on, x, atol=1e-6)


class TestMHCA:

    def test_mhca_forward_shapes_3d(self):
        """Basic forward pass preserves shape for 3D input."""
        m = mu.MHCA(in_dim=64, head_dim=8, num_heads=None)
        x = torch.randn(2, 64, 10)
        out = m(x)
        assert out.shape == x.shape

    def test_mhca_forward_shapes_4d(self):
        """Basic forward pass preserves shape for 4D input."""
        m = mu.MHCA(in_dim=32, head_dim=8, num_heads=None)
        x = torch.randn(3, 32, 5, 6)
        out = m(x)
        assert out.shape == x.shape

    def test_mhca_gated_vs_non_gated(self):
        """Both gated and non-gated produce valid outputs."""
        in_dim = 48
        head_dim = 12
        num_heads = in_dim // head_dim  # 4 heads

        m_gated = mu.MHCA(
            in_dim=in_dim, head_dim=head_dim, num_heads=num_heads, is_gated=True
        )
        m_nongated = mu.MHCA(
            in_dim=in_dim, head_dim=head_dim, num_heads=num_heads, is_gated=False
        )

        x = torch.randn(2, in_dim, 8)

        out_g = m_gated(x)
        out_ng = m_nongated(x)

        assert out_g.shape == x.shape
        assert out_ng.shape == x.shape

    def test_mhca_qk_norm_on_off(self):
        """qk_norm True/False both produce valid outputs."""
        in_dim = 64
        head_dim = 16

        m_with = mu.MHCA(in_dim=in_dim, head_dim=head_dim, qk_norm=True)
        m_without = mu.MHCA(in_dim=in_dim, head_dim=head_dim, qk_norm=False)

        x = torch.randn(2, in_dim, 8)

        assert m_with(x).shape == x.shape
        assert m_without(x).shape == x.shape

    def test_mhca_num_heads_auto_calculation(self):
        """When num_heads=None it should be in_dim // head_dim."""
        for in_dim, head_dim in [(64, 8), (32, 16), (128, 32)]:
            m = mu.MHCA(in_dim=in_dim, head_dim=head_dim)
            assert m.num_heads == in_dim // head_dim

    def test_mhca_split_merge_consistency(self):
        """Internal _split_heads and _merge_heads should round-trip correctly."""
        x = torch.randn(2, 10, 64)  # (B, L, total_dim=64)
        m = mu.MHCA(in_dim=64, head_dim=8, num_heads=None)

        assert m.total_dim == 64
        assert m.num_heads == 8
        assert m.head_dim == 8

        heads = m._split_heads(x)
        assert heads.shape == (2, 8, 10, 8)  # (B, H, L, head_dim)

        merged = m._merge_heads(heads)
        assert torch.allclose(merged, x, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
