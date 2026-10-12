"""Evaluation metrics for financial time series generation."""
from .stylized_facts import (
    compute_marginal_metrics,
    compute_percentile_wasserstein,
    compute_kurtosis_diff,
    compute_acf_error,
    compute_cross_asset_correlation_error,
)
from .calibration import (
    compute_causal_metrics,
    compute_interval_metrics,
    evaluate_naive_baseline,
)

__all__ = [
    "compute_marginal_metrics",
    "compute_percentile_wasserstein",
    "compute_kurtosis_diff",
    "compute_acf_error",
    "compute_cross_asset_correlation_error",
    "compute_causal_metrics",
    "compute_interval_metrics",
    "evaluate_naive_baseline",
]
