from typing import Any, Dict

import torch.nn as nn

from .m000_unet import FlowMatchingUNet
from .m001_transformer import FlowMatchingTransformer


def get_model(
    model_arch: str,
    model_params: Dict[str, Any] | None,
    c_in: int = 3,
    num_classes: int = 10,
) -> nn.Module:
    """
    Instantiate a model based on architecture name and parameters.

    Args:
        model_arch: Architecture name (e.g., 'm000_unet')
        model_params: Dictionary of model parameters
        c_in: Number of input channels
        num_classes: Number of output classes

    Returns:
        Instantiated model
    """
    if model_params is None:
        model_params = dict()
    if model_arch == "m000_unet":
        return FlowMatchingUNet(
            c_in=c_in,
            num_classes=num_classes,
            **model_params,
        )
    elif model_arch == "m001_transformer":
        return FlowMatchingTransformer(
            c_in=c_in,
            num_classes=num_classes,
            **model_params,
        )
    else:
        raise ValueError(f"Invalid model architecture: {model_arch}")
