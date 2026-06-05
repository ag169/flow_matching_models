import torch
import unittest
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from utils import fid_utils, flow_matching_utils as fm_utils

# pytest .\tests\test_fid_utils.py


class TestFidStatistics(unittest.TestCase):
    """Tests for low-level FID statistics functions."""

    def test_compute_statistics_shape(self):
        """Tests that _compute_statistics returns correct shapes."""
        features = np.random.randn(10, 2048)
        mu, sigma = fid_utils._compute_statistics(features)

        self.assertEqual(mu.shape, (2048,))
        self.assertEqual(sigma.shape, (2048, 2048))

    def test_compute_statistics_values(self):
        """Tests that statistics are computed correctly for known data."""
        # Create data with known mean
        features = np.ones((10, 5)) * 3.0
        mu, sigma = fid_utils._compute_statistics(features)

        np.testing.assert_array_almost_equal(mu, np.ones(5) * 3.0)
        # Covariance of constant data should be zero
        np.testing.assert_array_almost_equal(sigma, np.zeros((5, 5)))

    def test_compute_statistics_nonzero_covariance(self):
        """Tests covariance computation for data with variance."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu, sigma = fid_utils._compute_statistics(features)

        # Mean should be (3.0, 4.0)
        np.testing.assert_array_almost_equal(mu, [3.0, 4.0])
        # Covariance matrix should be non-zero for this data
        self.assertTrue(np.all(sigma > 0))


class TestFrechetDistance(unittest.TestCase):
    """Tests for Frechet distance computation."""

    def test_identical_distributions(self):
        """Tests that FID is zero for identical distributions."""
        mu1 = np.array([1.0, 2.0, 3.0])
        sigma1 = np.eye(3)
        mu2 = mu1.copy()
        sigma2 = sigma1.copy()

        fid = fid_utils._compute_frechet_dist(mu1, sigma1, mu2, sigma2)

        # FID should be zero (or very close) for identical distributions
        self.assertAlmostEqual(fid, 0.0, places=5)

    def test_different_means(self):
        """Tests that FID increases with different means."""
        mu1 = np.array([0.0, 0.0, 0.0])
        sigma1 = np.eye(3)
        mu2 = np.array([5.0, 5.0, 5.0])
        sigma2 = np.eye(3)

        fid = fid_utils._compute_frechet_dist(mu1, sigma1, mu2, sigma2)

        # FID should be positive (distance between means squared)
        self.assertGreater(fid, 0.0)
        # Expected FID = ||mu1 - mu2||^2 = 75
        self.assertAlmostEqual(fid, 75.0, places=5)

    def test_different_covariances(self):
        """Tests that FID increases with different covariances."""
        mu1 = np.array([0.0, 0.0, 0.0])
        sigma1 = np.eye(3)
        mu2 = mu1.copy()
        sigma2 = np.eye(3) * 4.0

        fid = fid_utils._compute_frechet_dist(mu1, sigma1, mu2, sigma2)

        self.assertGreater(fid, 0.0)

    def test_complex_matrix_sqrt(self):
        """Tests that complex matrix square root is handled."""
        # Create a case where sqrtm might return complex numbers
        sigma1 = np.eye(3)
        sigma2 = np.eye(3)

        fid = fid_utils._compute_frechet_dist(np.zeros(3), sigma1, np.zeros(3), sigma2)

        self.assertGreaterEqual(fid, 0.0)


class TestFidCalculator(unittest.TestCase):
    """Tests for the FidCalculator class with dummy data."""

    def setUp(self):
        """Set up test fixtures with dummy data."""
        torch.manual_seed(42)
        self.device = "cpu"
        self.num_samples = 10
        self.batch_size = 4
        self.imgsize = 32
        self.num_classes = 10

        # Create dummy real data (10 random images)
        real_images = torch.rand(self.num_samples, 3, self.imgsize, self.imgsize)
        real_labels = torch.randint(0, self.num_classes, (self.num_samples,))
        self.real_dataset = TensorDataset(real_images, real_labels)
        self.gt_loader = DataLoader(self.real_dataset, batch_size=self.batch_size)

        # Create a dummy model that returns tensors of the same shape as input
        class DummyFMModel(torch.nn.Module):
            def forward(self, x, t, cls=None):
                return torch.zeros_like(x)

        self.dummy_model = DummyFMModel()

        # Create a dummy solver
        self.solver = fm_utils.EulerSolver(num_steps=10)

    def test_fid_calculator_initialization(self):
        """Tests that FidCalculator initializes correctly."""
        calculator = fid_utils.FidCalculator(
            gt_loader=self.gt_loader,
            device=self.device,
            num_samples=self.num_samples,
        )

        self.assertIsNotNone(calculator.real_mu)
        self.assertIsNotNone(calculator.real_sigma)
        self.assertEqual(calculator.real_mu.shape, (2048,))
        self.assertEqual(calculator.real_sigma.shape, (2048, 2048))

    def test_fid_positive_for_different_data(self):
        """Tests that FID is positive for different distributions."""
        calculator = fid_utils.FidCalculator(
            gt_loader=self.gt_loader,
            device=self.device,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
        )

        # Create a dummy model that returns constant values
        class ConstantModel(torch.nn.Module):
            def forward(self, x, t, cls=None):
                return torch.ones_like(x) * 0.5

        constant_model = ConstantModel()

        fid = calculator.compute_fid(
            model=constant_model,
            device=self.device,
            solver=self.solver,
            cfg_scale=1.0,
            num_classes=self.num_classes,
            imgsize=self.imgsize,
        )

        # FID should be positive for different distributions
        self.assertGreater(fid, 0.0)

    def test_fid_with_cfg(self):
        """Tests FID computation with classifier-free guidance."""
        calculator = fid_utils.FidCalculator(
            gt_loader=self.gt_loader,
            device=self.device,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
        )

        class DummyModel(torch.nn.Module):
            def forward(self, x, t, cls=None):
                return torch.zeros_like(x)

        dummy_model = DummyModel()

        fid = calculator.compute_fid(
            model=dummy_model,
            device=self.device,
            solver=self.solver,
            cfg_scale=2.0,
            num_classes=self.num_classes,
            imgsize=self.imgsize,
        )

        self.assertGreaterEqual(fid, 0.0)


if __name__ == "__main__":
    unittest.main()
