"""
Preprocessing utilities — instance-scaled dataset + batch QT helpers.
"""

from .instance_dataset import (
    InstanceScaledDataset,
    compute_wavelet_level_scales,
    batch_quantile_transform,
    batch_inverse_quantile_transform,
)

__all__ = [
    'InstanceScaledDataset',
    'compute_wavelet_level_scales',
    'batch_quantile_transform',
    'batch_inverse_quantile_transform',
]
