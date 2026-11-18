"""DBSCAN failure-mode clustering for reference-avoided collisions."""

import csv
import json
import math
import os

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from .metrics import REFERENCE_AVOIDED


FEATURE_NAMES = [
    "scenario_lane_change",
    "collision_step",
    "danger_onset_step",
    "min_ttc",
    "min_rss_longitudinal_margin",
    "min_rss_lateral_margin",
    "min_center_distance",
    "ego_speed_at_end",
    "max_ego_brake_command",
    "max_ego_deceleration",
    "max_ego_lateral_speed",
    "max_npc_speed",
    "max_npc_acceleration",
    "collision_impulse",
    "relative_speed_at_collision",
    "impact_angle",
    "collision_relative_longitudinal",
    "collision_relative_lateral",
    "collision_with_controlled_npc",
    "npc_count",
]


def load_attributable_records(paths):
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("counterfactual_label") == REFERENCE_AVOIDED:
                    records.append(record)
    return records


def _feature_matrix(records):
    matrix = []
    for record in records:
        summary = record["original_summary"]
        row = []
        for feature in FEATURE_NAMES:
            if feature == "scenario_lane_change":
                value = 1.0 if record.get("scenario_type") == "lane_change" else 0.0
            else:
                value = summary.get(feature)
            row.append(float(value) if value is not None else float("nan"))
        matrix.append(row)
    values = np.asarray(matrix, dtype=np.float64)
    if values.size == 0:
        return values
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    missing_rows, missing_columns = np.where(~np.isfinite(values))
    values[missing_rows, missing_columns] = medians[missing_columns]
    return values


def cluster_records(records, eps=1.25, min_samples=3):
    if not records:
        return np.asarray([], dtype=int), {
            "sample_count": 0,
            "cluster_count": 0,
            "noise_count": 0,
            "noise_ratio": 0.0,
            "failure_mode_entropy": 0.0,
            "silhouette_index": None,
            "calinski_harabasz_index": None,
            "davies_bouldin_index": None,
        }
    values = _feature_matrix(records)
    scaled = StandardScaler().fit_transform(values)
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(scaled)
    cluster_ids = sorted(set(int(label) for label in labels if label >= 0))
    noise_count = int(np.sum(labels < 0))
    clustered_count = len(labels) - noise_count
    entropy = 0.0
    if clustered_count:
        for cluster_id in cluster_ids:
            probability = float(np.sum(labels == cluster_id)) / clustered_count
            entropy -= probability * math.log(probability, 2)

    quality_labels = labels >= 0
    quality_cluster_count = len(set(labels[quality_labels]))
    silhouette = None
    calinski = None
    davies = None
    if quality_cluster_count >= 2 and int(np.sum(quality_labels)) > quality_cluster_count:
        quality_values = scaled[quality_labels]
        selected_labels = labels[quality_labels]
        silhouette = float(silhouette_score(quality_values, selected_labels))
        calinski = float(calinski_harabasz_score(quality_values, selected_labels))
        davies = float(davies_bouldin_score(quality_values, selected_labels))

    return labels, {
        "sample_count": len(records),
        "cluster_count": len(cluster_ids),
        "noise_count": noise_count,
        "noise_ratio": float(noise_count) / len(records),
        "failure_mode_entropy": entropy,
        "silhouette_index": silhouette,
        "calinski_harabasz_index": calinski,
        "davies_bouldin_index": davies,
        "dbscan_eps": float(eps),
        "dbscan_min_samples": int(min_samples),
        "features": FEATURE_NAMES,
    }


def write_cluster_outputs(records, labels, metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    assignments_path = os.path.join(output_dir, "rss_cluster_assignments.csv")
    with open(assignments_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "scenario_type", "cluster_id"] + FEATURE_NAMES)
        for record, label in zip(records, labels):
            summary = record["original_summary"]
            writer.writerow([
                record.get("episode"),
                record.get("scenario_type"),
                int(label),
            ] + [
                (1.0 if record.get("scenario_type") == "lane_change" else 0.0)
                if feature == "scenario_lane_change"
                else summary.get(feature)
                for feature in FEATURE_NAMES
            ])

    summary_path = os.path.join(output_dir, "rss_cluster_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return assignments_path, summary_path
