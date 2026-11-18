#!/usr/bin/env python
"""Audit an HRAMA synchronized RGB--LiDAR dataset before pre-training."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np

from macad_gym.agents.multimodal_encoder.dataset import load_manifest


RUN_CONFIG = {
    "dataset_dir": os.path.join(PROJECT_ROOT, "artifacts", "multimodal_dataset"),
    "check_samples": 500,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=RUN_CONFIG["dataset_dir"])
    parser.add_argument(
        "--check-samples", type=int, default=RUN_CONFIG["check_samples"]
    )
    args = parser.parse_args()

    entries = load_manifest(args.dataset_dir)
    scenarios = collections.Counter()
    actors = collections.Counter()
    episodes = collections.defaultdict(set)
    identities = set()
    duplicates = []
    for entry in entries:
        scenario = str(entry.get("scenario", "unknown"))
        actor = str(entry.get("actor_id", "unknown"))
        identity = (
            scenario,
            int(entry.get("episode", -1)),
            actor,
            int(entry.get("frame", -1)),
        )
        if identity in identities:
            duplicates.append(identity)
        identities.add(identity)
        scenarios[scenario] += 1
        actors[actor] += 1
        episodes[scenario].add(int(entry.get("episode", -1)))

    invalid = []
    limit = min(len(entries), max(0, int(args.check_samples)))
    indices = np.linspace(0, len(entries) - 1, num=limit, dtype=np.int64)
    for index in indices:
        entry = entries[int(index)]
        path = os.path.join(args.dataset_dir, entry["path"])
        with np.load(path, allow_pickle=False) as sample:
            image = sample["image"]
            points = sample["points"]
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            invalid.append(entry["path"])

    print("dataset:", os.path.abspath(args.dataset_dir))
    print("samples:", len(entries))
    print("scenarios:", dict(sorted(scenarios.items())))
    print("episodes:", {
        key: len(value) for key, value in sorted(episodes.items())
    })
    print("actors:", dict(sorted(actors.items())))
    print("duplicate identities:", len(duplicates))
    print("invalid inspected samples:", len(invalid))
    if duplicates or invalid:
        raise SystemExit("Dataset audit failed")
    print("Dataset audit passed")


if __name__ == "__main__":
    main()
