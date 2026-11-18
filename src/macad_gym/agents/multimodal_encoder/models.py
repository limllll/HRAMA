"""Lightweight self-supervised encoders for synchronized CARLA sensors.

The online HRAMA policy keeps its original observation contract:

* image feature: 512 dimensions
* explicit vehicle state: 11 dimensions (handled outside this module)
* LiDAR feature: 256 dimensions

Decoders and contrastive projection heads exist only during offline pre-training.
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from macad_gym.PointNet2.models.pointnet2_utils import (
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)


class ConvBlock(nn.Module):
    """Convolution block matching the legacy TensorFlow image encoder."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=0,
                bias=True,
            ),
            # Match tf.contrib.layers.batch_norm defaults used by the original
            # checkpoint: epsilon=1e-3 and decay=0.999 (PyTorch momentum=0.001).
            nn.BatchNorm2d(out_channels, eps=1.0e-3, momentum=1.0e-3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DrivingImageEncoder(nn.Module):
    """PyTorch equivalent of the project's legacy TensorFlow encoder.

    Keeping the same eight convolutional layers and two 512-D fully connected
    layers makes it possible to initialize this encoder from the checkpoint
    already used by HRAMA instead of silently replacing it with a new network.
    """

    def __init__(self, feature_dim=512, input_size=(88, 200), dropout=0.0):
        super(DrivingImageEncoder, self).__init__()
        self.feature_dim = int(feature_dim)
        if self.feature_dim != 512:
            raise ValueError("The legacy image encoder output dimension is 512")
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 5, 2),
            ConvBlock(32, 32, 3, 1),
            ConvBlock(32, 64, 3, 2),
            ConvBlock(64, 64, 3, 1),
            ConvBlock(64, 128, 3, 2),
            ConvBlock(128, 128, 3, 1),
            ConvBlock(128, 256, 3, 1),
            ConvBlock(256, 256, 3, 1),
        )
        with torch.no_grad():
            probe = torch.zeros(1, 3, int(input_size[0]), int(input_size[1]))
            flattened_dim = int(self.backbone(probe).numel())
        self.fc1 = nn.Linear(flattened_dim, 512)
        self.fc2 = nn.Linear(512, self.feature_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward_with_maps(self, image):
        maps = {}
        feature_map = image
        for index, block in enumerate(self.backbone):
            feature_map = block(feature_map)
            if index == 1:
                maps["shallow"] = feature_map
            elif index == 3:
                maps["skip"] = feature_map
            elif index == 5:
                maps["middle"] = feature_map
        maps["deep"] = feature_map
        feature = feature_map.flatten(1)
        feature = self.dropout(F.relu(self.fc1(feature)))
        feature = self.dropout(F.relu(self.fc2(feature)))
        return feature, maps

    def forward_with_map(self, image):
        feature, maps = self.forward_with_maps(image)
        return feature, maps["deep"]

    def forward(self, image):
        feature, _ = self.forward_with_maps(image)
        return feature


class DrivingPointCloudEncoder(nn.Module):
    """PointNet++ MSG encoder with a fixed 256-dimensional output.

    The set-abstraction layout matches the PointNet++ model previously used by
    the project, while removing the unrelated ModelNet40 classification head.
    """

    def __init__(self, feature_dim=256, dropout1=0.4, dropout2=0.5):
        super(DrivingPointCloudEncoder, self).__init__()
        self.feature_dim = int(feature_dim)
        self.sa1 = PointNetSetAbstractionMsg(
            512,
            [0.1, 0.2, 0.4],
            [16, 32, 128],
            0,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128,
            [0.2, 0.4, 0.8],
            [32, 64, 128],
            320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 640 + 3, [256, 512, 1024], True
        )
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.feature_dim)
        self.bn2 = nn.BatchNorm1d(self.feature_dim)
        self.dropout1 = nn.Dropout(float(dropout1))
        self.dropout2 = nn.Dropout(float(dropout2))

    def forward_with_local(self, points):
        """Return the global feature and first-level PointNet++ features."""
        batch_size = points.shape[0]
        l1_xyz, l1_points = self.sa1(points, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        feature = l3_points.view(batch_size, 1024)
        feature = self.dropout1(F.relu(self.bn1(self.fc1(feature))))
        feature = self.dropout2(F.relu(self.bn2(self.fc2(feature))))
        return feature, l1_xyz, l1_points

    def forward(self, points):
        """Encode normalized XYZ points shaped ``[B, 3, N]``."""
        feature, _, _ = self.forward_with_local(points)
        return feature


class MultiscaleImageDecoder(nn.Module):
    """Training-only decoder that preserves spatial detail from CNN maps."""

    def __init__(self, feature_dim=512):
        super(MultiscaleImageDecoder, self).__init__()
        self.context = nn.Linear(int(feature_dim), 128)
        self.deep = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
        )
        self.fuse_middle = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse_skip = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, maps, feature, output_size):
        deep = self.deep(maps["deep"])
        context = self.context(feature).unsqueeze(-1).unsqueeze(-1)
        deep = deep + context
        deep = F.interpolate(
            deep, size=maps["middle"].shape[-2:], mode="bilinear",
            align_corners=False,
        )
        middle = self.fuse_middle(torch.cat([deep, maps["middle"]], dim=1))
        middle = F.interpolate(
            middle, size=maps["skip"].shape[-2:], mode="bilinear",
            align_corners=False,
        )
        decoded = self.fuse_skip(torch.cat([middle, maps["skip"]], dim=1))
        decoded = F.interpolate(
            decoded, size=tuple(output_size), mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.output(decoded))


class EnhancedMultiscaleImageDecoder(nn.Module):
    """Training-only decoder with an additional high-resolution skip path.

    The encoder itself is unchanged.  This decoder reuses the feature map
    already produced after the second legacy convolution (about 40x96 for an
    88x200 input) to avoid reconstructing all spatial detail from a 17x45 map.
    It is discarded after offline adaptation and never enters the PPO policy.
    """

    def __init__(self, feature_dim=512):
        super(EnhancedMultiscaleImageDecoder, self).__init__()
        self.context = nn.Linear(int(feature_dim), 128)
        self.deep = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        self.fuse_middle = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.fuse_skip = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.fuse_shallow = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )

    def forward(self, maps, feature, output_size):
        deep = self.deep(maps["deep"])
        deep = deep + self.context(feature).unsqueeze(-1).unsqueeze(-1)
        deep = F.interpolate(
            deep, size=maps["middle"].shape[-2:], mode="bilinear",
            align_corners=False,
        )
        middle = self.fuse_middle(torch.cat([deep, maps["middle"]], dim=1))
        middle = F.interpolate(
            middle, size=maps["skip"].shape[-2:], mode="bilinear",
            align_corners=False,
        )
        skip = self.fuse_skip(torch.cat([middle, maps["skip"]], dim=1))
        skip = F.interpolate(
            skip, size=maps["shallow"].shape[-2:], mode="bilinear",
            align_corners=False,
        )
        shallow = self.fuse_shallow(
            torch.cat([skip, maps["shallow"]], dim=1)
        )
        shallow = F.interpolate(
            shallow, size=tuple(output_size), mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(self.output(shallow))


class RayOccupancyHead(nn.Module):
    """ALSO-style occupancy queries conditioned on nearby local features."""

    def __init__(self, local_dim=320, hidden_dim=256):
        super(RayOccupancyHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(int(local_dim) + 3, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, local_xyz, local_features, queries):
        # local_xyz: [B, 3, M], local_features: [B, C, M], queries: [B, Q, 3]
        centers = local_xyz.transpose(1, 2)
        features = local_features.transpose(1, 2)
        nearest = torch.cdist(queries, centers).argmin(dim=2)
        gather_features = nearest.unsqueeze(-1).expand(-1, -1, features.shape[2])
        query_features = features.gather(1, gather_features)
        gather_xyz = nearest.unsqueeze(-1).expand(-1, -1, 3)
        relative = queries - centers.gather(1, gather_xyz)
        return self.decoder(torch.cat([query_features, relative], dim=2)).squeeze(-1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super(ProjectionHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(int(input_dim), int(input_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim), int(projection_dim)),
        )

    def forward(self, feature):
        return F.normalize(self.projector(feature), p=2, dim=1)


class MultimodalAdaptationModel(nn.Module):
    """Joint offline adaptation model; online inference keeps both encoders."""

    def __init__(
        self,
        image_feature_dim=512,
        lidar_feature_dim=256,
        projection_dim=128,
        image_decoder_version="legacy",
    ):
        super(MultimodalAdaptationModel, self).__init__()
        self.image_encoder = DrivingImageEncoder(image_feature_dim)
        self.lidar_encoder = DrivingPointCloudEncoder(lidar_feature_dim)
        self.image_decoder_version = str(image_decoder_version)
        if self.image_decoder_version == "legacy":
            self.image_decoder = MultiscaleImageDecoder(image_feature_dim)
        elif self.image_decoder_version == "enhanced":
            self.image_decoder = EnhancedMultiscaleImageDecoder(
                image_feature_dim
            )
        else:
            raise ValueError(
                "Unknown image decoder version: {}".format(
                    self.image_decoder_version
                )
            )
        self.ray_occupancy_head = RayOccupancyHead(local_dim=320)
        self.image_local_projector = ProjectionHead(64, projection_dim)
        self.lidar_local_projector = ProjectionHead(320, projection_dim)

    def forward_reconstruction(self, images, points, ray_queries):
        """Run the two modality-internal self-supervised objectives."""
        image_feature, image_maps = self.image_encoder.forward_with_maps(images)
        lidar_feature, local_xyz, local_features = self.lidar_encoder.forward_with_local(points)
        return {
            "image_feature": image_feature,
            "lidar_feature": lidar_feature,
            "image_reconstruction": self.image_decoder(
                image_maps, image_feature, images.shape[-2:]
            ),
            "ray_occupancy_logits": self.ray_occupancy_head(
                local_xyz, local_features, ray_queries
            ),
        }

    def forward_alignment(self, images, points):
        """Extract synchronized local features with gradients for both encoders."""
        image_feature, image_maps = self.image_encoder.forward_with_maps(images)
        lidar_feature, local_xyz, local_features = self.lidar_encoder.forward_with_local(points)
        return {
            "image_feature": image_feature,
            "lidar_feature": lidar_feature,
            "image_local_map": image_maps["skip"],
            "lidar_local_xyz": local_xyz,
            "lidar_local_features": local_features,
        }


def project_lidar_to_front_camera(local_xyz, lidar_range=20.0):
    """Project normalized CARLA LiDAR coordinates to normalized image UV."""
    xyz = local_xyz.transpose(1, 2) * float(lidar_range)
    # Fixed collection/runtime transforms. Both sensors have zero rotation.
    camera_x = xyz[:, :, 0] - 1.8
    camera_y = xyz[:, :, 1]
    camera_z = xyz[:, :, 2] + 0.8
    focal_over_width = 0.5  # horizontal FOV is 90 degrees
    u = focal_over_width * camera_y / torch.clamp(camera_x, min=1.0e-6) + 0.5
    # focal/height = 150/120 for the native 300x120 image.
    v = -1.25 * camera_z / torch.clamp(camera_x, min=1.0e-6) + 0.5
    valid = (
        (camera_x > 0.05) & (u >= 0.0) & (u < 1.0)
        & (v >= 0.0) & (v < 1.0)
    )
    return u, v, valid


def local_region_distillation(
    image_map,
    local_xyz,
    local_features,
    image_projector,
    lidar_projector,
    temperature=0.1,
    rows=2,
    columns=4,
    minimum_points=2,
    lidar_range=20.0,
):
    """Projection-guided symmetric region NCE for jointly trained modalities."""
    pooled_image = F.adaptive_avg_pool2d(
        image_map, (int(rows), int(columns))
    ).flatten(2).transpose(1, 2)
    image_regions = []
    lidar_regions = []
    for batch_index in range(local_xyz.shape[0]):
        xyz = local_xyz[batch_index].transpose(0, 1)
        features = local_features[batch_index].transpose(0, 1)
        # Fixed-size PointNet++ inputs may contain repeated points. Merge them
        # before counting valid projected points in a region.
        unique_xyz, inverse = torch.unique(
            xyz, dim=0, sorted=False, return_inverse=True
        )
        unique_features = features.new_zeros(
            (unique_xyz.shape[0], features.shape[1])
        )
        unique_features.index_add_(0, inverse, features)
        counts = features.new_zeros((unique_xyz.shape[0], 1))
        counts.index_add_(0, inverse, features.new_ones((features.shape[0], 1)))
        unique_features = unique_features / torch.clamp(counts, min=1.0)

        u, v, valid = project_lidar_to_front_camera(
            unique_xyz.transpose(0, 1).unsqueeze(0), lidar_range=lidar_range
        )
        u, v, valid = u[0], v[0], valid[0]
        region_x = torch.clamp((u * int(columns)).long(), 0, int(columns) - 1)
        region_y = torch.clamp((v * int(rows)).long(), 0, int(rows) - 1)
        region_ids = region_y * int(columns) + region_x
        for region in range(int(rows) * int(columns)):
            members = valid & (region_ids == region)
            if int(members.sum().item()) < int(minimum_points):
                continue
            image_regions.append(pooled_image[batch_index, region])
            lidar_regions.append(unique_features[members].mean(dim=0))

    if len(image_regions) < 2:
        zero = image_map.sum() * 0.0 + local_features.sum() * 0.0
        return zero, None, None
    image_projection = image_projector(torch.stack(image_regions, dim=0))
    lidar_projection = lidar_projector(torch.stack(lidar_regions, dim=0))
    logits = torch.matmul(lidar_projection, image_projection.t()) / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )
    return loss, image_projection, lidar_projection


def masked_image_mse(reconstruction, target, mask):
    """MSE over masked pixels only; mask is shaped ``[B, 1, H, W]``."""
    expanded_mask = mask.expand_as(target)
    denominator = torch.clamp(expanded_mask.sum(), min=1.0)
    return (((reconstruction - target) ** 2) * expanded_mask).sum() / denominator


def _masked_mean(values, mask):
    expanded_mask = mask.expand_as(values)
    denominator = torch.clamp(expanded_mask.sum(), min=1.0)
    return (values * expanded_mask).sum() / denominator


def masked_image_reconstruction_loss(
    reconstruction,
    target,
    mask,
    l1_weight=1.0,
    ssim_weight=0.2,
    edge_weight=0.1,
):
    """Structure-aware reconstruction loss evaluated on masked regions only.

    L1 reduces the averaging tendency of pixel MSE, SSIM preserves local
    structure, and finite-difference edge consistency emphasizes lane and
    object boundaries.  This objective changes no encoder layers or runtime
    feature dimensions.
    """
    l1 = _masked_mean((reconstruction - target).abs(), mask)

    # Differentiable local SSIM map for images normalized to [0, 1].
    kernel_size = 7
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(
        reconstruction, kernel_size, stride=1, padding=padding
    )
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(
        reconstruction * reconstruction, kernel_size, stride=1,
        padding=padding,
    ) - mu_x * mu_x
    sigma_y = F.avg_pool2d(
        target * target, kernel_size, stride=1, padding=padding
    ) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(
        reconstruction * target, kernel_size, stride=1, padding=padding
    ) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (
        (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        / torch.clamp(
            (mu_x * mu_x + mu_y * mu_y + c1)
            * (sigma_x + sigma_y + c2),
            min=1.0e-8,
        )
    )
    ssim = _masked_mean((1.0 - ssim_map) * 0.5, mask)

    pred_dx = reconstruction[:, :, :, 1:] - reconstruction[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = reconstruction[:, :, 1:, :] - reconstruction[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_x = torch.maximum(mask[:, :, :, 1:], mask[:, :, :, :-1])
    mask_y = torch.maximum(mask[:, :, 1:, :], mask[:, :, :-1, :])
    edge = 0.5 * (
        _masked_mean((pred_dx - target_dx).abs(), mask_x)
        + _masked_mean((pred_dy - target_dy).abs(), mask_y)
    )

    total = (
        float(l1_weight) * l1
        + float(ssim_weight) * ssim
        + float(edge_weight) * edge
    )
    return total, {"l1": l1, "ssim": ssim, "edge": edge}
