import torch
import unittest
from utils import flow_matching_utils as fm_utils

# pytest .\tests\test_flow_matching_utils.py


class TestTimeSamplers(unittest.TestCase):
    """Tests for time sampling strategies."""

    def test_uniform_time_sampler(self):
        """Tests UniformTimeSampler returns correct shape and bounds."""
        sampler = fm_utils.UniformTimeSampler()

        # Test various batch sizes
        for batch_size in [1, 4, 16, 32]:
            t = sampler(batch_size)

            self.assertTrue(torch.is_tensor(t))
            self.assertEqual(t.shape, torch.Size([batch_size]))
            self.assertGreaterEqual(t.min(), 0.0)
            self.assertLessEqual(t.max(), 1.0)

    def test_ode_time_sampler_default(self):
        """Tests ODETimeSampler with default sigma."""
        sampler = fm_utils.ODETimeSampler()
        batch_size = 1000

        t = sampler(batch_size)

        self.assertTrue(torch.is_tensor(t))
        self.assertEqual(t.shape, torch.Size([batch_size]))
        self.assertGreaterEqual(t.min(), 0.0)
        self.assertLessEqual(t.max(), 1.0)

    def test_ode_time_sampler_sigma_effect(self):
        """Tests that lower sigma concentrates samples near t=1."""
        sampler_low = fm_utils.ODETimeSampler(sigma=0.1)
        sampler_high = fm_utils.ODETimeSampler(sigma=2.0)

        batch_size = 5000

        t_low_sigma = sampler_low(batch_size).cpu().numpy()
        t_high_sigma = sampler_high(batch_size).cpu().numpy()

        # Higher sigma should produce smaller mean time (more concentrated near 1)
        self.assertLess(t_high_sigma.mean(), t_low_sigma.mean())


class TestFlowMatchingLoss(unittest.TestCase):
    """Tests for flow matching loss computation."""

    def setUp(self):
        """Set up common test fixtures."""
        torch.manual_seed(42)

        # Create a dummy model that returns tensors of the same shape as input
        class DummyModel(torch.nn.Module):
            def forward(self, x, t, cls=None):
                if cls is not None:
                    return torch.zeros_like(x)
                return torch.ones_like(x) * 0.5

        self.model = DummyModel()
        self.batch_size = 4
        self.channels = 3
        self.height = 8
        self.width = 8

    def _get_inputs(self):
        """Helper to create standard input tensors."""
        x0 = torch.randn(self.batch_size, self.channels, self.height, self.width)
        eps = torch.randn_like(x0)
        t = torch.rand(self.batch_size)
        cls = torch.randint(0, 10, (self.batch_size,))
        return x0, eps, t, cls

    def test_basic_loss_no_cfg(self):
        """Tests loss computation without classifier-free guidance."""
        x0, eps, t, _ = self._get_inputs()

        loss, uncond_loss = fm_utils.compute_flow_matching_loss(
            self.model, x0, eps, t, cls=None, cfg_dropout_prob=0.0
        )

        # Loss should be a scalar tensor
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        # Unconditional loss should be None when CFG is disabled
        self.assertIsNone(uncond_loss)

    def test_basic_loss_with_cls_no_dropout(self):
        """Tests loss computation with class labels but no dropout."""
        x0, eps, t, cls = self._get_inputs()

        loss, uncond_loss = fm_utils.compute_flow_matching_loss(
            self.model, x0, eps, t, cls=cls, cfg_dropout_prob=0.0
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        self.assertIsNone(uncond_loss)

    def test_cfg_loss_with_dropout(self):
        """Tests loss computation with classifier-free guidance dropout."""
        x0, eps, t, cls = self._get_inputs()

        loss, uncond_loss = fm_utils.compute_flow_matching_loss(
            self.model, x0, eps, t, cls=cls, cfg_dropout_prob=0.5
        )

        # Both losses should be computed
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.is_tensor(uncond_loss))
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(uncond_loss is not None)
        assert uncond_loss is not None
        self.assertEqual(uncond_loss.dim(), 0)

    def test_cfg_dropout_prob_one(self):
        """Tests loss computation with full dropout probability."""
        x0, eps, t, cls = self._get_inputs()

        # Should not raise error and should return valid losses
        loss, uncond_loss = fm_utils.compute_flow_matching_loss(
            self.model, x0, eps, t, cls=cls, cfg_dropout_prob=1.0
        )

        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.is_tensor(uncond_loss))

    def test_cfg_requires_cls(self):
        """Tests that CFG dropout requires class labels."""
        x0, eps, t = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8), torch.rand(4)

        with self.assertRaises(AssertionError):
            fm_utils.compute_flow_matching_loss(
                self.model, x0, eps, t, cls=None, cfg_dropout_prob=0.5
            )

    def test_cfg_invalid_prob(self):
        """Tests that invalid dropout probability raises error."""
        x0, eps, t, cls = self._get_inputs()

        with self.assertRaises(AssertionError):
            fm_utils.compute_flow_matching_loss(
                self.model, x0, eps, t, cls=cls, cfg_dropout_prob=1.5
            )


class TestEulerSolver(unittest.TestCase):
    """Tests for Euler ODE solver."""

    def setUp(self):
        torch.manual_seed(42)

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, cls=None):
                return torch.zeros_like(x)

        self.model = DummyModel()
        self.batch_size = 2
        self.channels = 3
        self.height = 8
        self.width = 8

    def test_euler_sample_shape(self):
        """Tests Euler solver output shape."""
        solver = fm_utils.EulerSolver(num_steps=10)

        x = solver.sample(
            model=self.model,
            shape=(self.batch_size, self.channels, self.height, self.width),
            device="cpu",
        )

        expected_shape = torch.Size(
            [self.batch_size, self.channels, self.height, self.width]
        )
        self.assertEqual(x.shape, expected_shape)

    def test_euler_sample_with_cls(self):
        """Tests Euler solver with class labels."""
        solver = fm_utils.EulerSolver(num_steps=10)

        cls = torch.randint(0, 10, (self.batch_size,))
        x = solver.sample(
            model=self.model,
            shape=(self.batch_size, self.channels, self.height, self.width),
            cls=cls,
            device="cpu",
        )

        expected_shape = torch.Size(
            [self.batch_size, self.channels, self.height, self.width]
        )
        self.assertEqual(x.shape, expected_shape)

    def test_euler_cfg_scale(self):
        """Tests Euler solver with CFG scale > 1."""
        solver = fm_utils.EulerSolver(num_steps=10)

        cls = torch.randint(0, 10, (self.batch_size,))
        x = solver.sample(
            model=self.model,
            shape=(self.batch_size, self.channels, self.height, self.width),
            cls=cls,
            device="cpu",
            cfg_scale=2.0,
        )

        expected_shape = torch.Size(
            [self.batch_size, self.channels, self.height, self.width]
        )
        self.assertEqual(x.shape, expected_shape)


class TestDPMPSolver(unittest.TestCase):
    """Tests for DPM-Solver."""

    def setUp(self):
        torch.manual_seed(42)

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, cls=None):
                return torch.zeros_like(x)

        self.model = DummyModel()
        self.batch_size = 2
        self.channels = 3
        self.height = 8
        self.width = 8

    def test_dpm_sample_shape(self):
        """Tests DPM-Solver output shape."""
        solver = fm_utils.DPMPSolver(num_steps=10)

        x = solver.sample(
            model=self.model,
            shape=(self.batch_size, self.channels, self.height, self.width),
            device="cpu",
        )

        expected_shape = torch.Size(
            [self.batch_size, self.channels, self.height, self.width]
        )
        self.assertEqual(x.shape, expected_shape)

    def test_dpm_sample_with_cls(self):
        """Tests DPM-Solver with class labels."""
        solver = fm_utils.DPMPSolver(num_steps=10)

        cls = torch.randint(0, 10, (self.batch_size,))
        x = solver.sample(
            model=self.model,
            shape=(self.batch_size, self.channels, self.height, self.width),
            cls=cls,
            device="cpu",
        )

        expected_shape = torch.Size(
            [self.batch_size, self.channels, self.height, self.width]
        )
        self.assertEqual(x.shape, expected_shape)

    def test_dpm_cfg_scale(self):
        """Tests DPM-Solver with CFG scale > 1."""
        solver = fm_utils.DPMPSolver(num_steps=10)

        cls = torch.randint(0, 10, (self.batch_size,))
        x = solver.sample(
            model=self.model,
            shape=(self.batch_size, self.channels, self.height, self.width),
            cls=cls,
            device="cpu",
            cfg_scale=5.0,
        )

        expected_shape = torch.Size(
            [self.batch_size, self.channels, self.height, self.width]
        )
        self.assertEqual(x.shape, expected_shape)

    def test_adaptive_timesteps(self):
        """Tests DPM-Solver adaptive timestep generation."""
        solver = fm_utils.DPMPSolver(num_steps=10)

        timesteps = solver._adaptive_timesteps()

        # Should have num_steps + 1 time steps (including start and end)
        self.assertEqual(len(timesteps), 11)
        # First timestep should be 0.0
        self.assertAlmostEqual(timesteps[0], 0.0, places=5)
        # Last timestep should be <= 1.0
        self.assertLessEqual(timesteps[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
