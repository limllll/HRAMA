"""Archive-backed novelty scoring for attributable ADS failure scenarios.

This module is simulator independent.  It deliberately stores a compact,
interpretable failure descriptor rather than image or LiDAR embeddings.  The
archive is intended for second-stage fine-tuning after a base HRAMA policy has
already learned to generate critical scenarios.
"""

from __future__ import division

import json
import math
import os

import numpy as np

from .metrics import REFERENCE_AVOIDED


DEFAULT_FEATURE_NAMES = (
    "collision_step",
    "danger_onset_step",
    "min_ttc",
    "min_rss_longitudinal_margin",
    "min_rss_lateral_margin",
    "relative_speed_at_collision",
    "impact_angle",
    "collision_relative_longitudinal",
    "collision_relative_lateral",
    "max_ego_deceleration",
    "max_ego_lateral_speed",
    "collision_with_controlled_npc",
)


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _finite_float(value):
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


class FailureFeatureEncoder(object):
    """Create and robustly scale the failure descriptor used by the archive."""

    def __init__(self, feature_names=None, max_steps=80, clip_value=5.0):
        self.feature_names = tuple(feature_names or DEFAULT_FEATURE_NAMES)
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.clip_value = float(clip_value)
        if self.clip_value <= 0.0:
            raise ValueError("clip_value must be positive")
        self.medians = None
        self.scales = None

    def _feature_value(self, summary, feature_name):
        value = _finite_float(summary.get(feature_name))
        if not math.isfinite(value):
            return value
        if feature_name in ("collision_step", "danger_onset_step"):
            return value / float(self.max_steps)
        if feature_name == "impact_angle":
            return value / 180.0
        return value

    def raw_vector(self, summary):
        return np.asarray([
            self._feature_value(summary, name)
            for name in self.feature_names
        ], dtype=np.float64)

    def fit(self, summaries):
        summaries = list(summaries)
        if not summaries:
            raise ValueError("At least one attributable failure is required")
        matrix = np.vstack([self.raw_vector(summary) for summary in summaries])
        medians = []
        scales = []
        for column_index in range(matrix.shape[1]):
            column = matrix[:, column_index]
            finite = column[np.isfinite(column)]
            if not finite.size:
                median = 0.0
                scale = 1.0
            else:
                median = float(np.median(finite))
                q25, q75 = np.percentile(finite, [25.0, 75.0])
                scale = float(q75 - q25)
                if not math.isfinite(scale) or scale < 1e-6:
                    standard_deviation = float(np.std(finite))
                    scale = standard_deviation if standard_deviation >= 1e-6 else 1.0
            medians.append(median)
            scales.append(scale)
        self.medians = np.asarray(medians, dtype=np.float64)
        self.scales = np.asarray(scales, dtype=np.float64)
        return self

    @property
    def fitted(self):
        return self.medians is not None and self.scales is not None

    def transform(self, summary):
        if not self.fitted:
            raise RuntimeError("FailureFeatureEncoder must be fitted first")
        vector = self.raw_vector(summary)
        missing = ~np.isfinite(vector)
        vector[missing] = self.medians[missing]
        scaled = (vector - self.medians) / self.scales
        return np.clip(scaled, -self.clip_value, self.clip_value)

    def to_dict(self):
        if not self.fitted:
            raise RuntimeError("Cannot serialize an unfitted feature encoder")
        return {
            "feature_names": list(self.feature_names),
            "max_steps": self.max_steps,
            "clip_value": self.clip_value,
            "medians": self.medians.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        encoder = cls(
            feature_names=payload["feature_names"],
            max_steps=payload["max_steps"],
            clip_value=payload.get("clip_value", 5.0),
        )
        encoder.medians = np.asarray(payload["medians"], dtype=np.float64)
        encoder.scales = np.asarray(payload["scales"], dtype=np.float64)
        if (len(encoder.medians) != len(encoder.feature_names)
                or len(encoder.scales) != len(encoder.feature_names)):
            raise ValueError("Encoder parameter length does not match feature list")
        return encoder


class NoveltyArchive(object):
    """Prototype archive with a frozen-batch candidate queue.

    ``query`` never mutates the archive.  Candidates observed during one PPO
    iteration are queued and added only when ``commit_pending`` is called by
    the trainer after the iteration.  This keeps the novelty reference fixed
    within a rollout/training batch.
    """

    FORMAT_VERSION = 1

    def __init__(self, encoder, scenario_type, ads_name="behavior_agent", k=5,
                 max_size=1000, reference_scale=1.0, add_threshold=0.0,
                 add_quantile=25.0):
        if not encoder.fitted:
            raise ValueError("NoveltyArchive requires a fitted encoder")
        self.encoder = encoder
        self.scenario_type = str(scenario_type)
        self.ads_name = str(ads_name)
        self.k = max(1, int(k))
        self.max_size = max(1, int(max_size))
        self.reference_scale = max(float(reference_scale), 1e-6)
        self.add_threshold = max(float(add_threshold), 0.0)
        self.add_quantile = float(add_quantile)
        self.version = 0
        self.prototypes = []
        self.pending_candidates = []
        self.duplicate_count = 0
        self.rejected_count = 0

    @staticmethod
    def _load_seed_records(paths, scenario_type=None):
        records = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("counterfactual_label") != REFERENCE_AVOIDED:
                        continue
                    if (scenario_type is not None
                            and record.get("scenario_type") != scenario_type):
                        continue
                    summary = record.get("original_summary")
                    if isinstance(summary, dict):
                        records.append(record)
        return records

    @staticmethod
    def _pairwise_leave_one_out_scores(vectors, k):
        if len(vectors) <= 1:
            return np.asarray([1.0], dtype=np.float64)
        scores = []
        for index, vector in enumerate(vectors):
            distances = np.linalg.norm(vectors - vector, axis=1)
            distances = np.delete(distances, index)
            neighbor_count = min(max(1, int(k)), len(distances))
            scores.append(float(np.mean(np.sort(distances)[:neighbor_count])))
        return np.asarray(scores, dtype=np.float64)

    @classmethod
    def from_seed_jsonl(cls, paths, scenario_type, ads_name="behavior_agent",
                        feature_names=None, max_steps=80, clip_value=5.0,
                        k=5, max_size=1000, add_quantile=25.0):
        paths = list(paths)
        records = cls._load_seed_records(paths, scenario_type=scenario_type)
        if not records:
            raise ValueError(
                "No rss_reference_avoided records found for scenario {} in {}"
                .format(scenario_type, paths))
        summaries = [record["original_summary"] for record in records]
        encoder = FailureFeatureEncoder(
            feature_names=feature_names,
            max_steps=max_steps,
            clip_value=clip_value,
        ).fit(summaries)
        vectors = np.vstack([encoder.transform(summary) for summary in summaries])
        leave_one_out = cls._pairwise_leave_one_out_scores(vectors, k)
        finite_scores = leave_one_out[np.isfinite(leave_one_out)]
        if finite_scores.size:
            reference_scale = max(
                float(np.percentile(finite_scores, 90.0)), 1e-6)
            add_threshold = max(
                float(np.percentile(finite_scores, float(add_quantile))), 1e-6)
        else:
            reference_scale = 1.0
            add_threshold = 1e-6

        archive = cls(
            encoder=encoder,
            scenario_type=scenario_type,
            ads_name=ads_name,
            k=k,
            max_size=max_size,
            reference_scale=reference_scale,
            add_threshold=add_threshold,
            add_quantile=add_quantile,
        )
        for record, vector in zip(records, vectors):
            if len(archive.prototypes) >= archive.max_size:
                break
            archive._append_prototype(
                summary=record["original_summary"],
                vector=vector,
                metadata={
                    "source": "seed_archive",
                    "source_episode": record.get("episode"),
                    "source_path": None,
                },
            )
        return archive

    def _prototype_vectors(self):
        if not self.prototypes:
            return np.empty((0, len(self.encoder.feature_names)), dtype=np.float64)
        return np.asarray(
            [prototype["vector"] for prototype in self.prototypes],
            dtype=np.float64,
        )

    def query(self, summary):
        vector = self.encoder.transform(summary)
        vectors = self._prototype_vectors()
        if not len(vectors):
            return {
                "vector": vector,
                "raw_distance": self.reference_scale,
                "normalized_novelty": 1.0,
                "nearest_prototype_id": None,
                "nearest_distance": None,
                "is_duplicate": False,
                "archive_version": self.version,
                "archive_size": 0,
            }
        distances = np.linalg.norm(vectors - vector, axis=1)
        order = np.argsort(distances)
        neighbor_count = min(self.k, len(order))
        raw_distance = float(np.mean(distances[order[:neighbor_count]]))
        nearest_index = int(order[0])
        nearest_distance = float(distances[nearest_index])
        normalized_novelty = float(np.clip(
            (raw_distance - self.add_threshold)
            / max(self.reference_scale - self.add_threshold, 1e-6),
            0.0,
            1.0,
        ))
        return {
            "vector": vector,
            "raw_distance": raw_distance,
            # Failures inside the archive's duplicate-density threshold do not
            # receive an extra reward.  Beyond that threshold, novelty grows
            # linearly and remains bounded by one.
            "normalized_novelty": normalized_novelty,
            "nearest_prototype_id": self.prototypes[nearest_index]["prototype_id"],
            "nearest_distance": nearest_distance,
            "is_duplicate": raw_distance <= self.add_threshold,
            "archive_version": self.version,
            "archive_size": len(self.prototypes),
        }

    def queue_candidate(self, summary, metadata=None, query_result=None):
        query_result = query_result or self.query(summary)
        self.pending_candidates.append({
            "summary": _json_safe(summary),
            "metadata": _json_safe(metadata or {}),
            "observed_query": {
                key: _json_safe(value)
                for key, value in query_result.items()
                if key != "vector"
            },
        })

    def _append_prototype(self, summary, vector, metadata=None):
        prototype_id = "{}-{:06d}".format(
            self.scenario_type, len(self.prototypes) + 1)
        self.prototypes.append({
            "prototype_id": prototype_id,
            "summary": _json_safe(summary),
            "vector": np.asarray(vector, dtype=np.float64).tolist(),
            "metadata": _json_safe(metadata or {}),
        })
        return prototype_id

    def commit_pending(self):
        added = 0
        duplicate = 0
        rejected = 0
        candidate_count = len(self.pending_candidates)
        for candidate in self.pending_candidates:
            if len(self.prototypes) >= self.max_size:
                rejected += 1
                continue
            query_result = self.query(candidate["summary"])
            if query_result["raw_distance"] <= self.add_threshold:
                duplicate += 1
                continue
            self._append_prototype(
                candidate["summary"],
                query_result["vector"],
                metadata=dict(
                    candidate.get("metadata", {}),
                    observed_query=candidate.get("observed_query", {}),
                    committed_against_version=self.version,
                ),
            )
            added += 1
        self.duplicate_count += duplicate
        self.rejected_count += rejected
        self.pending_candidates = []
        self.version += 1
        return {
            "archive_version": self.version,
            "candidate_count": candidate_count,
            "added_count": added,
            "duplicate_count": duplicate,
            "rejected_count": rejected,
            "archive_size": len(self.prototypes),
            "add_threshold": self.add_threshold,
            "reference_scale": self.reference_scale,
        }

    def to_dict(self):
        return {
            "format_version": self.FORMAT_VERSION,
            "scenario_type": self.scenario_type,
            "ads_name": self.ads_name,
            "k": self.k,
            "max_size": self.max_size,
            "reference_scale": self.reference_scale,
            "add_threshold": self.add_threshold,
            "add_quantile": self.add_quantile,
            "version": self.version,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "encoder": self.encoder.to_dict(),
            "prototypes": self.prototypes,
        }

    def save(self, path):
        path = os.path.abspath(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(self.to_dict()), handle,
                ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary_path, path)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported novelty archive format version")
        archive = cls(
            encoder=FailureFeatureEncoder.from_dict(payload["encoder"]),
            scenario_type=payload["scenario_type"],
            ads_name=payload.get("ads_name", "behavior_agent"),
            k=payload["k"],
            max_size=payload["max_size"],
            reference_scale=payload["reference_scale"],
            add_threshold=payload["add_threshold"],
            add_quantile=payload.get("add_quantile", 25.0),
        )
        archive.version = int(payload.get("version", 0))
        archive.duplicate_count = int(payload.get("duplicate_count", 0))
        archive.rejected_count = int(payload.get("rejected_count", 0))
        archive.prototypes = list(payload.get("prototypes", []))
        for prototype in archive.prototypes:
            vector = prototype.get("vector", [])
            if len(vector) != len(archive.encoder.feature_names):
                raise ValueError("Prototype vector length does not match encoder")
        return archive
