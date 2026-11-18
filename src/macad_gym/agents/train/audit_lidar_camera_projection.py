from __future__ import print_function

import argparse
import glob
import json
import os

import cv2
import numpy as np


def project(points, width=300, height=120, fov_deg=90.0):
    # Sensor transforms relative to the ego vehicle in the collection script.
    # Both sensors have zero rotation. CARLA coordinates are x-forward,
    # y-right, z-up; camera optical coordinates are y-right, -z-down, x-depth.
    camera_t = np.array([1.8, 0.0, 1.7], dtype=np.float32)
    lidar_t = np.array([0.0, 0.0, 2.5], dtype=np.float32)
    p_camera_carla = points[:, :3] + lidar_t[None] - camera_t[None]
    optical = np.stack(
        [p_camera_carla[:, 1], -p_camera_carla[:, 2], p_camera_carla[:, 0]],
        axis=1,
    )
    depth = optical[:, 2]
    focal = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    u = focal * optical[:, 0] / np.maximum(depth, 1.0e-6) + width / 2.0
    v = focal * optical[:, 1] / np.maximum(depth, 1.0e-6) + height / 2.0
    valid = (
        (depth > 0.05)
        & (u >= 0.0) & (u < width)
        & (v >= 0.0) & (v < height)
    )
    return u, v, depth, valid


def draw_overlay(image, points):
    u, v, depth, valid = project(points, image.shape[1], image.shape[0])
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    values = np.clip(depth[valid] / 20.0, 0.0, 1.0)
    colors = cv2.applyColorMap((255 * (1.0 - values)).astype(np.uint8), cv2.COLORMAP_TURBO)
    for x, y, color in zip(u[valid], v[valid], colors[:, 0, :]):
        cv2.circle(bgr, (int(round(x)), int(round(y))), 2, tuple(int(c) for c in color), -1)
    return bgr, int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples", type=int, default=24)
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(args.dataset_dir, "samples", "*.npz")))
    if not files:
        raise RuntimeError("no npz samples")
    os.makedirs(args.output_dir, exist_ok=True)
    counts, ratios, total_points = [], [], []
    grid_stats = {"2x4": [], "3x6": [], "4x8": []}
    chosen = np.linspace(0, len(files) - 1, min(args.examples, len(files)), dtype=int)
    chosen = set(chosen.tolist())
    example_rows = []
    for index, path in enumerate(files):
        with np.load(path) as sample:
            image = sample["image"]
            points = sample["points"]
        u, v, _, valid = project(points, image.shape[1], image.shape[0])
        count = int(valid.sum())
        counts.append(count)
        total_points.append(int(points.shape[0]))
        ratios.append(float(count) / max(1, points.shape[0]))
        for name, (rows, columns) in {
            "2x4": (2, 4), "3x6": (3, 6), "4x8": (4, 8)
        }.items():
            gx = np.minimum(columns - 1, (u[valid] / image.shape[1] * columns).astype(np.int64))
            gy = np.minimum(rows - 1, (v[valid] / image.shape[0] * rows).astype(np.int64))
            cell_counts = np.bincount(gy * columns + gx, minlength=rows * columns)
            grid_stats[name].append({
                "nonempty": int((cell_counts > 0).sum()),
                "at_least_2": int((cell_counts >= 2).sum()),
                "at_least_3": int((cell_counts >= 3).sum()),
            })
        if index in chosen:
            overlay, _ = draw_overlay(image, points)
            canvas = np.concatenate([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), overlay], axis=1)
            label = "%s | projected %d/%d" % (os.path.basename(path), count, points.shape[0])
            cv2.putText(canvas, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
            example_rows.append(canvas)
    counts = np.asarray(counts)
    ratios = np.asarray(ratios)
    total_points = np.asarray(total_points)
    metrics = {
        "samples": len(files),
        "points_per_frame": {
            "min": int(total_points.min()), "median": float(np.median(total_points)),
            "mean": float(total_points.mean()), "max": int(total_points.max()),
        },
        "projected_points_per_frame": {
            "min": int(counts.min()), "p05": float(np.percentile(counts, 5)),
            "median": float(np.median(counts)), "mean": float(counts.mean()),
            "p95": float(np.percentile(counts, 95)), "max": int(counts.max()),
        },
        "projected_ratio": {
            "median": float(np.median(ratios)), "mean": float(ratios.mean()),
        },
        "frames_below_8_points": int((counts < 8).sum()),
        "frames_below_16_points": int((counts < 16).sum()),
        "frames_below_32_points": int((counts < 32).sum()),
    }
    metrics["grid_regions_per_frame"] = {}
    for name, values in grid_stats.items():
        metrics["grid_regions_per_frame"][name] = {
            key: {
                "mean": float(np.mean([item[key] for item in values])),
                "median": float(np.median([item[key] for item in values])),
            }
            for key in ("nonempty", "at_least_2", "at_least_3")
        }
    with open(os.path.join(args.output_dir, "projection_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    if example_rows:
        sheet = np.concatenate(example_rows, axis=0)
        cv2.imwrite(os.path.join(args.output_dir, "projection_overlays.png"), sheet)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
