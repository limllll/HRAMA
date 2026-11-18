#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse
import collections
import csv
import glob
import hashlib
import json
import os

import cv2
import numpy as np


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def group_name(scenario):
    return str(scenario).split("_route", 1)[0]


def add_label(image, text):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(image, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def make_extreme_sheet(dataset_dir, rows, output_path):
    if not rows:
        return
    choices = []
    selectors = [
        ("darkest", lambda x: x["image_mean"]),
        ("brightest", lambda x: -x["image_mean"]),
        ("blurriest", lambda x: x["blur"]),
        ("sharpest", lambda x: -x["blur"]),
        ("fewest points", lambda x: x["point_count"]),
        ("most points", lambda x: -x["point_count"]),
    ]
    for label, key in selectors:
        choices.append((label, min(rows, key=key)))
    temporal = [row for row in rows if row["previous_image_mad"] >= 0]
    if temporal:
        choices.append(("most similar previous", min(temporal, key=lambda x: x["previous_image_mad"])))
        choices.append(("most different previous", max(temporal, key=lambda x: x["previous_image_mad"])))
    tiles = []
    for label, row in choices:
        path = os.path.join(dataset_dir, row["path"])
        with np.load(path, allow_pickle=False) as sample:
            rgb = np.asarray(sample["image"], dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        text = "{} | mean {:.1f} blur {:.1f} pts {} mad {:.2f}".format(
            label, row["image_mean"], row["blur"], row["point_count"],
            row["previous_image_mad"],
        )
        tiles.append(add_label(bgr, text))
    width = max(tile.shape[1] for tile in tiles)
    tiles = [cv2.copyMakeBorder(tile, 0, 0, 0, width - tile.shape[1],
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for tile in tiles]
    cv2.imwrite(output_path, np.concatenate(tiles, axis=0))


def main():
    parser = argparse.ArgumentParser(description="Audit synchronized RGB-LiDAR NPZ dataset")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(dataset_dir, "audit"))
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(dataset_dir, "manifest.jsonl")
    with open(manifest_path, "r") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    sample_files = sorted(glob.glob(os.path.join(dataset_dir, "samples", "*.npz")))
    relative_files = set(os.path.relpath(path, dataset_dir) for path in sample_files)
    manifest_paths = [str(entry.get("path", "")) for entry in entries]
    path_counts = collections.Counter(manifest_paths)
    identity_counts = collections.Counter(
        (str(e.get("scenario")), int(e.get("episode", -1)),
         str(e.get("actor_id")), int(e.get("frame", -1))) for e in entries
    )
    duplicate_paths = sorted(path for path, count in path_counts.items() if count > 1)
    duplicate_identities = [list(key) for key, count in identity_counts.items() if count > 1]
    missing_files = sorted(set(manifest_paths) - relative_files)
    orphan_files = sorted(relative_files - set(manifest_paths))

    rows = []
    load_errors = []
    key_sets = collections.Counter()
    image_hash_paths = collections.defaultdict(list)
    group_counts = collections.Counter()
    weather_counts = collections.Counter()
    weather_route_counts = collections.Counter()
    routes = set()
    previous = {}
    image_shapes = collections.Counter()
    image_dtypes = collections.Counter()
    point_dtypes = collections.Counter()
    invalid_image_count = 0
    invalid_point_count = 0
    missing_metadata_count = 0
    frame_gap_below_stride = 0
    expected_stride = 0
    config_path = os.path.join(dataset_dir, "collection_config.json")
    config = json.load(open(config_path)) if os.path.isfile(config_path) else {}
    expected_stride = int(config.get("stride", 0))

    for index, entry in enumerate(entries):
        if index and index % 1000 == 0:
            print("audited {}/{}".format(index, len(entries)))
        path = os.path.join(dataset_dir, entry.get("path", ""))
        try:
            with np.load(path, allow_pickle=False) as sample:
                keys = tuple(sorted(sample.files))
                key_sets[str(keys)] += 1
                image = np.asarray(sample["image"])
                points = np.asarray(sample["points"])
        except Exception as error:
            load_errors.append({"path": entry.get("path"), "error": repr(error)})
            continue
        image_shapes[str(tuple(image.shape))] += 1
        image_dtypes[str(image.dtype)] += 1
        point_dtypes[str(points.dtype)] += 1
        image_valid = (image.ndim == 3 and image.shape[2] == 3 and
                       np.isfinite(image).all() and image.size > 0)
        point_valid = (points.ndim == 2 and points.shape[1] >= 3 and
                       points.shape[0] > 0 and np.isfinite(points[:, :3]).all())
        invalid_image_count += int(not image_valid)
        invalid_point_count += int(not point_valid)
        if not image_valid or not point_valid:
            continue
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        image_mean = float(image.mean())
        image_std = float(image.std())
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        point_count = int(points.shape[0])
        distances = np.linalg.norm(points[:, :3], axis=1)
        point_max_distance = float(distances.max())
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        image_hash_paths[digest].append(entry["path"])
        scenario = str(entry.get("scenario", "unknown"))
        episode = int(entry.get("episode", -1))
        actor = str(entry.get("actor_id", "unknown"))
        frame = int(entry.get("frame", -1))
        route = (scenario, episode)
        routes.add(route)
        group_counts[group_name(scenario)] += 1
        weather = str(entry.get("weather", "missing"))
        weather_counts[weather] += 1
        weather_route_counts[(weather, route)] += 1
        if any(name not in entry for name in
               ("weather", "traffic_tier", "background_vehicles", "sampling_stride")):
            missing_metadata_count += 1
        previous_key = (scenario, episode, actor)
        previous_mad = -1.0
        frame_gap = -1
        if previous_key in previous:
            previous_frame, previous_image = previous[previous_key]
            frame_gap = frame - previous_frame
            previous_mad = float(np.mean(np.abs(
                image.astype(np.float32) - previous_image.astype(np.float32)
            )))
            if expected_stride and frame_gap < expected_stride:
                frame_gap_below_stride += 1
        previous[previous_key] = (frame, image.copy())
        flags = []
        if image_mean < 40 or image_mean > 235:
            flags.append("exposure")
        if image_std < 25:
            flags.append("low_contrast")
        if blur < 100:
            flags.append("blur")
        if point_count < 140:
            flags.append("few_points")
        if point_max_distance > 20.5:
            flags.append("point_range")
        if previous_mad >= 0 and previous_mad < 1.0:
            flags.append("near_duplicate_previous")
        rows.append({
            "path": entry["path"], "scenario": scenario,
            "episode": episode, "frame": frame, "weather": weather,
            "traffic_tier": entry.get("traffic_tier", "missing"),
            "background_vehicles": entry.get("background_vehicles", -1),
            "image_mean": image_mean, "image_std": image_std,
            "blur": blur, "point_count": point_count,
            "point_max_distance": point_max_distance,
            "previous_image_mad": previous_mad, "frame_gap": frame_gap,
            "flags": "|".join(flags),
        })

    exact_duplicate_groups = [paths for paths in image_hash_paths.values() if len(paths) > 1]
    temporal_mad = [row["previous_image_mad"] for row in rows if row["previous_image_mad"] >= 0]
    frame_gaps = [row["frame_gap"] for row in rows if row["frame_gap"] >= 0]
    suspect_rows = [row for row in rows if row["flags"]]
    expected_groups = {}
    for town in config.get("towns", []):
        for tier in config.get("traffic_tiers_per_town", []):
            expected_groups["{}_{}".format(town, tier["name"])] = int(tier["samples"])
    group_mismatches = {
        name: {"actual": int(group_counts.get(name, 0)), "expected": expected}
        for name, expected in expected_groups.items()
        if int(group_counts.get(name, 0)) != expected
    }
    issues = []
    for condition, message in [
        (len(entries) != int(config.get("total_target_samples", len(entries))), "manifest count differs from configured target"),
        (len(sample_files) != len(entries), "sample file count differs from manifest count"),
        (bool(duplicate_paths), "duplicate manifest paths"),
        (bool(duplicate_identities), "duplicate scenario/episode/actor/frame identities"),
        (bool(missing_files), "manifest references missing files"),
        (bool(orphan_files), "unreferenced NPZ files"),
        (bool(load_errors), "NPZ load errors"),
        (invalid_image_count > 0, "invalid images"),
        (invalid_point_count > 0, "invalid point clouds"),
        (missing_metadata_count > 0, "missing V2 metadata"),
        (frame_gap_below_stride > 0, "same-route frame gaps below configured stride"),
        (bool(group_mismatches), "town/traffic-tier count mismatch"),
    ]:
        if condition:
            issues.append(message)
    summary = {
        "passed_integrity": not issues,
        "issues": issues,
        "manifest_entries": len(entries),
        "file_count": len(sample_files),
        "loaded": len(rows),
        "load_errors": load_errors,
        "missing_files": missing_files,
        "orphan_files": orphan_files,
        "duplicate_manifest_paths": duplicate_paths,
        "duplicate_scenario_frames": duplicate_identities,
        "exact_duplicate_image_groups": exact_duplicate_groups,
        "invalid_image_count": invalid_image_count,
        "invalid_point_count": invalid_point_count,
        "missing_metadata_count": missing_metadata_count,
        "frame_gap_below_stride": frame_gap_below_stride,
        "automatic_suspect_count": len(suspect_rows),
        "group_counts": dict(sorted(group_counts.items())),
        "group_mismatches": group_mismatches,
        "weather_counts": dict(sorted(weather_counts.items())),
        "weather_route_counts": dict(sorted(
            (weather, len(set(route for w, route in weather_route_counts if w == weather)))
            for weather in weather_counts
        )),
        "route_count": len(routes),
        "npz_key_sets": dict(key_sets),
        "image_shapes": dict(image_shapes),
        "image_dtypes": dict(image_dtypes),
        "point_dtypes": dict(point_dtypes),
        "metrics": {
            "mean": stats([row["image_mean"] for row in rows]),
            "std": stats([row["image_std"] for row in rows]),
            "blur": stats([row["blur"] for row in rows]),
            "points": stats([row["point_count"] for row in rows]),
            "point_max_distance": stats([row["point_max_distance"] for row in rows]),
            "previous_image_mad": stats(temporal_mad),
            "frame_gap": stats(frame_gaps),
        },
        "near_duplicate_previous_below_1": int(sum(value < 1.0 for value in temporal_mad)),
        "near_duplicate_previous_below_3": int(sum(value < 3.0 for value in temporal_mad)),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    fields = list(rows[0].keys()) if rows else []
    with open(os.path.join(output_dir, "all_metrics.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    make_extreme_sheet(dataset_dir, rows, os.path.join(output_dir, "extreme_samples.jpg"))
    markdown = [
        "# Multimodal Dataset V2 Audit", "",
        "- Integrity: **{}**".format("PASS" if summary["passed_integrity"] else "FAIL"),
        "- Samples: {} manifest / {} files / {} loaded".format(len(entries), len(sample_files), len(rows)),
        "- Routes: {}".format(len(routes)),
        "- Automatic suspects: {} (review flags in `all_metrics.csv`)".format(len(suspect_rows)),
        "- Exact duplicate image groups: {}".format(len(exact_duplicate_groups)),
        "- Consecutive near-duplicates (MAD < 1): {}".format(summary["near_duplicate_previous_below_1"]),
        "- Frame gaps below configured stride: {}".format(frame_gap_below_stride),
        "", "## Weather distribution", "",
    ]
    markdown.extend("- {}: {} samples".format(k, v) for k, v in sorted(weather_counts.items()))
    markdown.extend(["", "## Issues", ""])
    markdown.extend(["- None"] if not issues else ["- " + item for item in issues])
    with open(os.path.join(dataset_dir, "AUDIT_SUMMARY.md"), "w") as handle:
        handle.write("\n".join(markdown) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
