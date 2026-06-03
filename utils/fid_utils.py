"""FID (Fréchet Inception Distance) computation utility.

Provides a class-based API that caches real data feature statistics after
the first pass.
"""

from typing import Any

import numpy as np
from scipy.linalg import sqrtm as scipy_sqrtm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

_INCEPTION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------------------------------------------------------------------------
# Inception model loader
# ---------------------------------------------------------------------------


def _get_inception_model(device: torch.device | str) -> models.Inception3:
    """Load a pre-trained Inception v3 (no classifier head) for feature extraction."""
    try:
        # Use pretrained weights from torchvision
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,  # We'll handle normalization separately
        )
        # Disable the final FC layer
        inception.fc = nn.Identity()  # type: ignore
    except Exception:
        raise RuntimeError(
            "Failed to load Inception v3 pretrained weights. "
            "Ensure torchvision is installed and has internet access for weight download."
        )
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    return inception.to(device)


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------


def _compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of a set of features.

    Args:
        features: Array of shape (N, D) containing feature vectors.

    Returns:
        (mu, sigma) where sigma is the unbiased covariance matrix.
    """
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Compute matrix square root using scipy.linalg.sqrtm."""
    sqrtm_op = scipy_sqrtm(matrix)
    if isinstance(sqrtm_op, tuple):
        sqrtm_op = sqrtm_op[0]
    return sqrtm_op


def _compute_frechet_dist(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Compute the Fréchet distance between two multivariate Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrt(sigma1 * sigma2))
    """
    diff = mu1 - mu2
    covmean = _sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = np.real(covmean)

    fid = float(np.dot(diff, diff)) + float(np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------


def _extract_inception_features(
    loader: DataLoader,
    inception_model: models.Inception3,
    device: torch.device | str,
    max_samples: int,
) -> np.ndarray:
    """Extract Inception features from a data loader.

    Args:
        loader: DataLoader yielding (images, labels) batches.
        device: Device for inference.
        max_samples: Maximum number of samples to process.

    Returns:
        Numpy array of shape (N, 2048) with Inception features.
    """

    features_list = []
    total = 0

    with torch.no_grad():
        for batch in loader:
            if total >= max_samples:
                break
            images, _ = batch
            images = images.to(device, non_blocking=True)
            images = _INCEPTION_TRANSFORM(images)
            logits = inception_model(images)
            features = logits.cpu().numpy()
            features_list.append(features)
            total += images.shape[0]

    features = np.concatenate(features_list, axis=0)[:max_samples]
    return features


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def _generate_and_extract_features(
    model: nn.Module,
    inception_model: models.Inception3,
    device: torch.device | str,
    solver: Any,
    num_samples: int,
    batch_size: int,
    cfg_scale: float,
    num_classes: int,
    imgsize: int,
) -> np.ndarray:
    """Generate samples with the model and extract Inception features.

    Args:
        model: The flow-matching model for generation.
        device: Device for inference.
        num_samples: Number of samples to generate.
        batch_size: Batch size for generation.
        cfg_scale: Classifier-free guidance scale.
        num_classes: Number of classes in the dataset.
        imgsize: Image resolution.

    Returns:
        Numpy array of shape (N, 2048) with Inception features.
    """
    features_list = []
    total = 0

    with torch.no_grad():
        while total < num_samples:
            current_batch = min(batch_size, num_samples - total)
            # Sample random class labels for unconditional generation
            cls_indices = torch.randint(0, num_classes, (current_batch,), device=device)
            shape = (current_batch, 3, imgsize, imgsize)

            samples = solver.sample(
                model=model,
                shape=shape,
                cls=cls_indices,
                device=device,
                cfg_scale=cfg_scale,
            )
            samples = torch.clip(samples, 0.0, 1.0)
            samples = _INCEPTION_TRANSFORM(samples)
            logits = inception_model(samples)
            features = logits.cpu().numpy()
            features_list.append(features)
            total += current_batch

    return np.concatenate(features_list, axis=0)


# ---------------------------------------------------------------------------
# Public API — class-based
# ---------------------------------------------------------------------------


class FidCalculator:
    """Compute FID.

    Usage:
        calculator = FidCalculator(real_loader, device, num_samples=50000)
        fid = calculator.compute_fid(model, device, num_samples=50000, ...)
    """

    def __init__(
        self,
        gt_loader: DataLoader,
        device: torch.device | str,
        num_samples: int = 50000,
        batch_size: int = 64,
    ):
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.inception_model = _get_inception_model(device=device)

        # Extract and cache GT data features (one-time cost)
        print("Extracting GT data Inception features...")
        real_features = _extract_inception_features(
            loader=gt_loader,
            inception_model=self.inception_model,
            device=device,
            max_samples=num_samples,
        )
        self.real_mu, self.real_sigma = _compute_statistics(real_features)
        print(
            f"GT features: mu shape={self.real_mu.shape}, "
            f"sigma shape={self.real_sigma.shape}"
        )

    def compute_fid(
        self,
        model: nn.Module,
        device: torch.device | str,
        solver: Any,
        cfg_scale: float = 1.0,
        num_classes: int = 10,
        imgsize: int = 32,
    ) -> float:
        """Generate samples and compute FID using cached real statistics.

        Args:
            model: The flow-matching model for generation.
            device: Device for inference.
            solver: ODE solver for generating samples.
            cfg_scale: Classifier-free guidance scale.
            num_classes: Number of classes in the dataset.
            imgsize: Image resolution.

        Returns:
            FID score (lower is better).
        """
        print(f"Generating {self.num_samples} samples for FID computation...")
        gen_features = _generate_and_extract_features(
            model=model,
            inception_model=self.inception_model,
            device=device,
            solver=solver,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            cfg_scale=cfg_scale,
            num_classes=num_classes,
            imgsize=imgsize,
        )
        gen_mu, gen_sigma = _compute_statistics(gen_features)
        fid = _compute_frechet_dist(self.real_mu, self.real_sigma, gen_mu, gen_sigma)
        return fid
