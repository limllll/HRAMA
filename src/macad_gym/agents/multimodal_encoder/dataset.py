"""Synchronized CARLA RGB--LiDAR dataset utilities."""

from __future__ import absolute_import, division, print_function

import hashlib
import json
import os
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def carla_image_to_rgb(image):
    """Convert a CARLA BGRA image object to an RGB uint8 array."""
    if isinstance(image, np.ndarray):
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB array must have shape [H, W, 3]")
        if np.issubdtype(rgb.dtype, np.floating):
            if rgb.size and float(np.max(rgb)) <= 1.5:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
        return np.ascontiguousarray(rgb)
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape((image.height, image.width, 4))
    return np.ascontiguousarray(raw[:, :, :3][:, :, ::-1])


def sanitize_points(points, lidar_range=20.0):
    """Return finite XYZ points inside the configured sensor cube."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("LiDAR points must have shape [N, >=3]")
    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    lidar_range = float(lidar_range)
    if lidar_range <= 0:
        raise ValueError("lidar_range must be positive")
    inside = np.all(np.abs(points) <= lidar_range, axis=1)
    points = points[inside]
    if points.shape[0] == 0:
        raise ValueError("LiDAR sample contains no finite in-range points")
    return points


def voxel_downsample_numpy(points, voxel_size=0.1):
    """Dependency-light deterministic voxel downsampling."""
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        return points
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
    unique_indices.sort()
    return points[unique_indices]


def sample_and_normalize_points(
    points,
    num_points=1024,
    lidar_range=20.0,
    voxel_size=0.1,
    rng=None,
):
    """Sanitize, downsample, fixed-size sample and range-normalize XYZ."""
    rng = rng if rng is not None else np.random
    points = sanitize_points(points, lidar_range=lidar_range)
    points = voxel_downsample_numpy(points, voxel_size=voxel_size)
    num_points = int(num_points)
    replace = points.shape[0] < num_points
    selected = rng.choice(points.shape[0], size=num_points, replace=replace)
    sampled = points[selected]
    sampled = np.clip(sampled / float(lidar_range), -1.0, 1.0)
    return sampled.astype(np.float32, copy=False)


class SynchronizedPairWriter(object):
    """Write strict same-actor, same-frame RGB--LiDAR samples."""

    def __init__(
        self,
        output_dir,
        scenario="unknown",
        stride=2,
        max_samples=0,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.sample_dir = os.path.join(self.output_dir, "samples")
        self.manifest_path = os.path.join(self.output_dir, "manifest.jsonl")
        self.scenario = str(scenario)
        self.stride = max(1, int(stride))
        self.max_samples = max(0, int(max_samples))
        self._lock = threading.Lock()
        self._seen = set()
        self._actor_counts = {}
        self.count = 0
        self._episode_offset = 0
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        # Support resuming collection without overwriting files or duplicating
        # manifest records.  The complete identity includes the episode because
        # CARLA frame numbers can restart when the server is relaunched.
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, "r") as manifest:
                for line in manifest:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._seen.add(
                        (
                            str(entry.get("scenario", self.scenario)),
                            int(entry.get("episode", -1)),
                            str(entry.get("actor_id", "unknown")),
                            int(entry.get("frame", -1)),
                        )
                    )
                    if str(entry.get("scenario", "")) == self.scenario:
                        self._episode_offset = max(
                            self._episode_offset,
                            int(entry.get("episode", 0)),
                        )
                    self.count += 1

    @property
    def full(self):
        return self.max_samples > 0 and self.count >= self.max_samples

    def write(self, image, points, actor_id, frame, episode, metadata=None):
        actor_id = str(actor_id)
        frame = int(frame)
        episode = int(episode) + self._episode_offset
        key = (self.scenario, episode, actor_id, frame)

        with self._lock:
            if self.full or key in self._seen:
                return False
            actor_count = self._actor_counts.get(actor_id, 0)
            self._actor_counts[actor_id] = actor_count + 1
            if actor_count % self.stride != 0:
                return False

            rgb = carla_image_to_rgb(image)
            points = sanitize_points(points, lidar_range=1.0e6)
            filename = "{}_ep{:06d}_{}_frame{:09d}.npz".format(
                self.scenario.replace("/", "_"),
                episode,
                actor_id.replace("/", "_"),
                frame,
            )
            relative_path = os.path.join("samples", filename)
            absolute_path = os.path.join(self.output_dir, relative_path)
            temp_path = absolute_path + ".tmp.npz"
            np.savez_compressed(
                temp_path,
                image=rgb,
                points=points.astype(np.float32, copy=False),
            )
            os.replace(temp_path, absolute_path)

            entry = {
                "path": relative_path,
                "scenario": self.scenario,
                "episode": episode,
                "actor_id": actor_id,
                "frame": frame,
                "image_height": int(rgb.shape[0]),
                "image_width": int(rgb.shape[1]),
                "num_points": int(points.shape[0]),
            }
            if metadata:
                reserved = set(entry)
                for name, value in dict(metadata).items():
                    if name in reserved:
                        raise ValueError(
                            "Sample metadata cannot override reserved field {}"
                            .format(name)
                        )
                    entry[str(name)] = value
            with open(self.manifest_path, "a") as manifest:
                manifest.write(json.dumps(entry, sort_keys=True) + "\n")
            self._seen.add(key)
            self.count += 1
            return True


def load_manifest(dataset_dir):
    manifest_path = os.path.join(os.path.abspath(dataset_dir), "manifest.jsonl")
    if not os.path.isfile(manifest_path):
        raise IOError("Dataset manifest not found: {}".format(manifest_path))
    entries = []
    with open(manifest_path, "r") as manifest:
        for line_number, line in enumerate(manifest, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            sample_path = os.path.join(dataset_dir, entry["path"])
            if not os.path.isfile(sample_path):
                raise IOError(
                    "Manifest line {} references missing sample {}".format(
                        line_number, sample_path
                    )
                )
            entries.append(entry)
    if not entries:
        raise ValueError("Dataset manifest contains no samples")
    return entries


def split_entries_by_episode(entries, val_ratio=0.1, seed=2026):
    """Balanced deterministic split with complete episodes kept together."""
    val_ratio = float(val_ratio)
    groups = {}
    for entry in entries:
        group = "{}:{}".format(
            entry.get("scenario", "unknown"),
            entry.get("episode", -1),
        )
        groups.setdefault(group, []).append(entry)
    if len(groups) < 2:
        raise ValueError("At least two complete episodes are required for splitting")

    # Subset-sum over episode sizes chooses a validation set close to the
    # requested sample ratio. Hash ordering makes ties stable across runs.
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha1(
            (item[0] + ":" + str(int(seed))).encode("utf-8")
        ).hexdigest(),
    )
    target = max(1, int(round(len(entries) * val_ratio)))
    maximum = min(len(entries) - 1, target + max(len(v) for _, v in ordered))
    choices = {0: ()}
    for index, (_, members) in enumerate(ordered):
        size = len(members)
        for total, selected in list(choices.items())[::-1]:
            candidate = total + size
            if candidate <= maximum and candidate not in choices:
                choices[candidate] = selected + (index,)
    valid_totals = [total for total in choices if 0 < total < len(entries)]
    best_total = min(valid_totals, key=lambda total: (abs(total - target), total))
    val_group_indices = set(choices[best_total])
    val_group_names = {ordered[index][0] for index in val_group_indices}
    train_entries = []
    val_entries = []
    for group_name, members in groups.items():
        (val_entries if group_name in val_group_names else train_entries).extend(members)
    return train_entries, val_entries


class MultimodalPairDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        entries,
        image_size=(88, 200),
        num_points=1024,
        lidar_range=20.0,
        voxel_size=0.1,
        lidar_mask_ratio=0.5,
        num_ray_queries=384,
        ray_surface_offset=0.1,
        augment=False,
        seed=2026,
    ):
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.entries = list(entries)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.num_points = int(num_points)
        self.lidar_range = float(lidar_range)
        self.voxel_size = float(voxel_size)
        self.lidar_mask_ratio = float(lidar_mask_ratio)
        self.num_ray_queries = int(num_ray_queries)
        self.ray_surface_offset = float(ray_surface_offset)
        self.augment = bool(augment)
        self.seed = int(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        sample_path = os.path.join(self.dataset_dir, entry["path"])
        with np.load(sample_path, allow_pickle=False) as sample:
            image = np.asarray(sample["image"], dtype=np.uint8)
            points = np.asarray(sample["points"], dtype=np.float32)

        image = cv2.resize(
            image,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_AREA,
        )
        if self.augment:
            image, points = self._paired_augmentation(image, points)
        # Preserve the legacy image encoder's [0, 255] input distribution;
        # reconstruction uses a separate normalized target.
        image = image.astype(np.float32)
        image_target = image / 255.0

        points = sanitize_points(points, lidar_range=self.lidar_range)
        points = voxel_downsample_numpy(points, voxel_size=self.voxel_size)
        points = np.clip(points / self.lidar_range, -1.0, 1.0)
        if self.augment:
            rng = np.random
        else:
            digest = hashlib.sha1(
                (entry["path"] + ":" + str(self.seed)).encode("utf-8")
            ).hexdigest()
            rng = np.random.RandomState(int(digest[:8], 16))

        target_indices = rng.choice(
            points.shape[0], self.num_points,
            replace=points.shape[0] < self.num_points,
        )
        target_points = points[target_indices]
        visible_count = max(
            1, int(round(points.shape[0] * (1.0 - self.lidar_mask_ratio)))
        )
        visible_indices = rng.choice(
            points.shape[0], visible_count, replace=False
        )
        visible_points = points[visible_indices]
        input_indices = rng.choice(
            visible_points.shape[0], self.num_points,
            replace=visible_points.shape[0] < self.num_points,
        )
        masked_points = visible_points[input_indices]
        ray_queries, ray_labels = self._ray_occupancy_queries(points, rng)
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)),
            "image_target": torch.from_numpy(image_target.transpose(2, 0, 1)),
            "points": torch.from_numpy(target_points.transpose(1, 0).astype(np.float32)),
            "masked_points": torch.from_numpy(masked_points.transpose(1, 0).astype(np.float32)),
            "ray_queries": torch.from_numpy(ray_queries.astype(np.float32)),
            "ray_labels": torch.from_numpy(ray_labels.astype(np.float32)),
            "path": entry["path"],
            "group": "{}:{}".format(
                entry.get("scenario", "unknown"), entry.get("episode", -1)
            ),
        }

    def _ray_occupancy_queries(self, normalized_points, rng):
        """Create ALSO-style occupied/free queries along measured rays."""
        query_count = max(3, self.num_ray_queries)
        hits_needed = int(np.ceil(query_count / 3.0))
        hit_indices = rng.choice(
            normalized_points.shape[0], hits_needed,
            replace=normalized_points.shape[0] < hits_needed,
        )
        hits = normalized_points[hit_indices]
        norms = np.linalg.norm(hits, axis=1, keepdims=True)
        directions = hits / np.maximum(norms, 1.0e-6)
        offset = self.ray_surface_offset / self.lidar_range

        occupied = hits + directions * rng.uniform(
            0.0, offset, size=(hits_needed, 1)
        ).astype(np.float32)
        free_near = hits - directions * rng.uniform(
            0.0, offset, size=(hits_needed, 1)
        ).astype(np.float32)
        free_ray = hits * rng.uniform(
            0.05, 0.95, size=(hits_needed, 1)
        ).astype(np.float32)
        queries = np.concatenate([occupied, free_near, free_ray], axis=0)
        labels = np.concatenate([
            np.ones(hits_needed, dtype=np.float32),
            np.zeros(2 * hits_needed, dtype=np.float32),
        ])
        order = rng.permutation(queries.shape[0])[:query_count]
        return np.clip(queries[order], -1.0, 1.0), labels[order]

    @staticmethod
    def _paired_augmentation(image, points):
        # A horizontal image flip corresponds to reflecting CARLA's lateral y
        # coordinate. Applying it to both modalities preserves synchronization.
        if np.random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            points = points.copy()
            points[:, 1] *= -1.0

        gain = np.random.uniform(0.9, 1.1)
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return image, points
