"""RSS-based counterfactual validation and failure-mode analysis.

The equations in :mod:`macad_gym.rss_analysis.model` follow the longitudinal
and lateral definitions in Shalev-Shwartz, Shammah, and Shashua (2017).  The
counterfactual evaluator keeps the project's original collision definition and
adds a stricter, reference-driver-based attribution layer.
"""

from .model import (
    RSSParameters,
    RSSPairEvaluation,
    lateral_safe_distance,
    longitudinal_safe_distance_opposite_direction,
    longitudinal_safe_distance_same_direction,
)
from .metrics import CounterfactualMetrics, classify_counterfactual

__all__ = [
    "RSSParameters",
    "RSSPairEvaluation",
    "CounterfactualMetrics",
    "classify_counterfactual",
    "lateral_safe_distance",
    "longitudinal_safe_distance_opposite_direction",
    "longitudinal_safe_distance_same_direction",
]
