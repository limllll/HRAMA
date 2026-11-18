"""Runtime-only loader for self-trained HRAMA encoders."""

from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np
import torch

from macad_gym.agents.multimodal_encoder.dataset import (
    sample_and_normalize_points,
)
from macad_gym.agents.multimodal_encoder.models import (
    DrivingImageEncoder,
    DrivingPointCloudEncoder,
)


def resolve_device(device_name):
    device_name = str(device_name or "auto")
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for the multimodal encoder but is unavailable"
        )
    return torch.device(device_name)


class RuntimeMultimodalEncoder(object):
    """Load only image/LiDAR encoders and expose NumPy feature methods."""

    def __init__(self, checkpoint_path, device="auto"):
        checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise IOError(
                "Multimodal encoder checkpoint not found: {}".format(
                    checkpoint_path
                )
            )
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint.get("model_config", {})

        self.image_height = int(model_config.get("image_height", 88))
        self.image_width = int(model_config.get("image_width", 200))
        self.image_feature_dim = int(
            model_config.get("image_feature_dim", 512)
        )
        self.lidar_feature_dim = int(
            model_config.get("lidar_feature_dim", 256)
        )
        self.num_points = int(model_config.get("num_points", 1024))
        self.lidar_range = float(model_config.get("lidar_range", 20.0))
        self.voxel_size = float(model_config.get("voxel_size", 0.1))

        if self.image_feature_dim != 512 or self.lidar_feature_dim != 256:
            raise ValueError(
                "Checkpoint dimensions must remain image=512 and lidar=256; "
                "got image={} lidar={}".format(
                    self.image_feature_dim, self.lidar_feature_dim
                )
            )

        self.image_encoder = DrivingImageEncoder(self.image_feature_dim)
        self.lidar_encoder = DrivingPointCloudEncoder(self.lidar_feature_dim)
        self.image_encoder.load_state_dict(checkpoint["image_encoder"])
        self.lidar_encoder.load_state_dict(checkpoint["lidar_encoder"])
        self.image_encoder.to(self.device).eval()
        self.lidar_encoder.to(self.device).eval()

    def encode_image(self, image):
        """Encode an RGB array in either uint8 or normalized float format."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB image must have shape [H, W, 3]")
        image = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )
        image = image.astype(np.float32, copy=False)
        # The legacy TensorFlow image encoder was trained and deployed with
        # RGB values in [0, 255]. Preserve that input distribution.
        if image.size and image.max() <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            feature = torch.clamp(self.image_encoder(tensor), 0.0, 1.0)
        return self._to_feature_array(feature, self.image_feature_dim)

    def encode_lidar(self, points):
        normalized = sample_and_normalize_points(
            points,
            num_points=self.num_points,
            lidar_range=self.lidar_range,
            voxel_size=self.voxel_size,
            rng=np.random.RandomState(0),
        )
        tensor = torch.from_numpy(normalized.transpose(1, 0)).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            feature = torch.clamp(self.lidar_encoder(tensor), 0.0, 1.0)
        return self._to_feature_array(feature, self.lidar_feature_dim)

    @staticmethod
    def _to_feature_array(feature, expected_size):
        array = feature.detach().cpu().numpy().reshape(-1).astype(
            np.float32, copy=False
        )
        if array.size != int(expected_size):
            raise ValueError(
                "Encoder returned {} values; expected {}".format(
                    array.size, expected_size
                )
            )
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(array, 0.0, 1.0)
