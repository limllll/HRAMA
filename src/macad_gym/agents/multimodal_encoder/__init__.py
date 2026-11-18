"""Self-supervised camera/LiDAR encoders used by HRAMA."""

from macad_gym.agents.multimodal_encoder.models import (
    DrivingImageEncoder,
    DrivingPointCloudEncoder,
    MultimodalAdaptationModel,
)

__all__ = [
    "DrivingImageEncoder",
    "DrivingPointCloudEncoder",
    "MultimodalAdaptationModel",
]
