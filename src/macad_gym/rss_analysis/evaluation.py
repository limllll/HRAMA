"""High-level integration used by lane-change and car-following inference."""

import json
import os

from .metrics import CounterfactualMetrics, classify_counterfactual
from .model import RSSParameters
from .recording import (
    EpisodeRecorder,
    JsonlArchive,
    RandomStateSnapshot,
    capture_initial_scene_state,
)
from .replay import run_rss_counterfactual_replay


def load_rss_parameters(path=None):
    if not path:
        return RSSParameters()
    with open(path, "r", encoding="utf-8") as handle:
        values = json.load(handle)
    return RSSParameters(**values)


class RSSCounterfactualEvaluation:
    """Manage original recording, RSS replay, labels, archive, and metrics."""

    def __init__(self, scenario_type, output_dir, params=None,
                 keep_full_trace=False):
        self.scenario_type = str(scenario_type)
        self.params = params or RSSParameters()
        self.keep_full_trace = bool(keep_full_trace)
        self.archive = JsonlArchive(output_dir, self.scenario_type)
        self.metrics = CounterfactualMetrics()
        self._initial_random_state = None
        self._initial_scene_state = None
        self._recorder = None

    def before_original_reset(self):
        self._initial_random_state = RandomStateSnapshot.capture()

    def start_original_episode(self, env):
        if self._initial_random_state is None:
            raise RuntimeError("before_original_reset() must be called first")
        self._initial_scene_state = capture_initial_scene_state(env)
        self._recorder = EpisodeRecorder(
            scenario_type=self.scenario_type,
            params=self.params,
            keep_full_trace=self.keep_full_trace,
        )

    def record_original_step(self, env, action_dict):
        self._recorder.record_step(env, action_dict)

    def finish_original_episode(self, env, episode, ads_collision):
        if self._recorder is None:
            raise RuntimeError("start_original_episode() must be called first")
        original_summary = self._recorder.summary(env, ads_collision)
        post_original_random_state = RandomStateSnapshot.capture()
        replay = None
        if ads_collision:
            replay = run_rss_counterfactual_replay(
                env=env,
                recorded_action_trace=self._recorder.action_trace,
                initial_random_state=self._initial_random_state,
                original_initial_scene_state=self._initial_scene_state,
                post_original_random_state=post_original_random_state,
                scenario_type=self.scenario_type,
                params=self.params,
                keep_full_trace=self.keep_full_trace,
            )
            label = classify_counterfactual(
                ads_collision=True,
                reference_collision=replay["reference_collision"],
                replay_succeeded=replay["succeeded"],
                reference_road_violation=replay.get(
                    "reference_road_violation", False),
            )
        else:
            label = classify_counterfactual(False)

        self.metrics.update(label)
        record = {
            "episode": int(episode),
            "scenario_type": self.scenario_type,
            "counterfactual_label": label,
            "original_summary": original_summary,
            "reference_replay": replay,
            "rss_parameters": self.params.to_dict(),
        }
        if self.keep_full_trace:
            record["original_trace"] = self._recorder.archive_payload()
        self.archive.append(record)
        self._initial_random_state = None
        self._initial_scene_state = None
        self._recorder = None
        return record

    def write_summary(self):
        path = os.path.join(
            self.archive.output_dir,
            "{}_rss_metrics.json".format(self.scenario_type),
        )
        payload = self.metrics.summary()
        payload["scenario_type"] = self.scenario_type
        payload["rss_parameters"] = self.params.to_dict()
        payload["metric_provenance"] = {
            "reference_avoided_collision_frequency": (
                "Reference-avoided collisions / all tests; denominator follows "
                "AuthSim valid collisions / all tests."),
            "reference_avoided_collision_ratio": (
                "Reference-avoided collisions / ADS collision tests; denominator "
                "follows AuthSim valid collisions / all collisions."),
            "reference_avoided_ratio_assessed_replays": (
                "Reference-avoided collisions / successful collision replays; "
                "reported separately so replay failures remain visible."),
            "attribution_rule": (
                "ADS collision and reference-driver avoidance; consistent with "
                "published avoidable-collision/reference-driver comparisons."),
            "reference_collision_rule": (
                "If both ADS and the single RSS-supervised reference collide, "
                "the result is inconclusive, not an invalid-collision label."),
            "safe_avoidance_rule": (
                "A reference avoidance is attributable only if it has neither "
                "a collision nor an off-lane/off-road event."),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return path, payload
