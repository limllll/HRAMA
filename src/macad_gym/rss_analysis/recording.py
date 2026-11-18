"""Episode recording and feature extraction for RSS counterfactual analysis."""

from __future__ import division

import copy
import json
import math
import os
import random

import numpy as np

from .carla_adapter import evaluate_carla_pairs
from .model import RSSParameters


class RandomStateSnapshot:
    def __init__(self, python_state, numpy_state):
        self.python_state = python_state
        self.numpy_state = numpy_state

    @classmethod
    def capture(cls):
        return cls(random.getstate(), np.random.get_state())

    def restore(self):
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)


def _scene_actor_snapshot(actor):
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    return {
        "type_id": str(actor.type_id),
        "x": float(transform.location.x),
        "y": float(transform.location.y),
        "z": float(transform.location.z),
        "yaw": float(transform.rotation.yaw),
        "speed": _vector_norm(velocity),
    }


def capture_initial_scene_state(env):
    """Capture logical actor states immediately after ``env.reset()``.

    CARLA actor IDs change on every reset.  Controlled vehicles are therefore
    keyed by the environment actor name, while SUMO background mirrors are
    keyed by their stable SUMO vehicle IDs.
    """

    actors = {}
    for actor_id, actor in env._actors.items():
        if actor is not None and actor.is_alive:
            actors["controlled:{}".format(actor_id)] = _scene_actor_snapshot(actor)

    sumo_bridge = getattr(env, "_sumo_bridge", None)
    if sumo_bridge is not None:
        for sumo_id, carla_id in getattr(
                sumo_bridge, "_sumo_to_carla_id", {}).items():
            try:
                actor = env.world.get_actor(int(carla_id))
            except (TypeError, ValueError):
                actor = None
            if actor is not None and actor.is_alive:
                actors["sumo:{}".format(sumo_id)] = _scene_actor_snapshot(actor)
    return actors


def compare_initial_scene_states(expected, actual, position_tolerance=0.35,
                                 speed_tolerance=0.75,
                                 yaw_tolerance_degrees=2.0):
    """Check that an RSS replay starts from the same physical scene."""

    expected_keys = set(expected)
    actual_keys = set(actual)
    mismatches = []
    for key in sorted(expected_keys - actual_keys):
        mismatches.append({"actor": key, "reason": "missing_in_replay"})
    for key in sorted(actual_keys - expected_keys):
        mismatches.append({"actor": key, "reason": "extra_in_replay"})

    for key in sorted(expected_keys & actual_keys):
        before = expected[key]
        after = actual[key]
        position_error = math.sqrt(
            (before["x"] - after["x"]) ** 2
            + (before["y"] - after["y"]) ** 2
            + (before["z"] - after["z"]) ** 2)
        speed_error = abs(before["speed"] - after["speed"])
        yaw_error = abs(before["yaw"] - after["yaw"]) % 360.0
        yaw_error = min(yaw_error, 360.0 - yaw_error)
        type_match = before["type_id"] == after["type_id"]
        if (position_error > position_tolerance
                or speed_error > speed_tolerance
                or yaw_error > yaw_tolerance_degrees
                or not type_match):
            mismatches.append({
                "actor": key,
                "reason": "state_mismatch",
                "position_error_m": position_error,
                "speed_error_mps": speed_error,
                "yaw_error_deg": yaw_error,
                "type_match": type_match,
            })
    return {
        "matches": not mismatches,
        "expected_actor_count": len(expected),
        "actual_actor_count": len(actual),
        "position_tolerance_m": float(position_tolerance),
        "speed_tolerance_mps": float(speed_tolerance),
        "yaw_tolerance_deg": float(yaw_tolerance_degrees),
        "mismatches": mismatches,
    }


def _vector_norm(vector):
    return math.sqrt(float(vector.x) ** 2 + float(vector.y) ** 2
                     + float(vector.z) ** 2)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


class EpisodeRecorder:
    """Collect a compact trajectory plus clustering-ready failure features."""

    def __init__(self, scenario_type, params=None, ego_id="ego",
                 keep_full_trace=True):
        self.scenario_type = str(scenario_type)
        self.params = params or RSSParameters()
        self.ego_id = ego_id
        self.keep_full_trace = bool(keep_full_trace)
        self.action_trace = []
        self.state_trace = []
        self._min_ttc = float("inf")
        self._min_long_margin = float("inf")
        self._min_lat_margin = float("inf")
        self._min_center_distance = float("inf")
        self._danger_onset_step = None
        self._max_ego_brake = 0.0
        self._max_ego_deceleration = 0.0
        self._max_ego_lateral_speed = 0.0
        self._max_npc_speed = 0.0
        self._max_npc_acceleration = 0.0
        self._previous_ego_speed = None

    @staticmethod
    def _copy_actions(action_dict):
        copied = {}
        for actor_id, action in action_dict.items():
            if isinstance(action, np.ndarray):
                copied[actor_id] = action.astype(int).tolist()
            elif isinstance(action, (list, tuple)):
                copied[actor_id] = [int(value) for value in action]
            else:
                copied[actor_id] = copy.deepcopy(action)
        return copied

    @staticmethod
    def _actor_snapshot(actor):
        transform = actor.get_transform()
        location = transform.location
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        control = actor.get_control()
        return {
            "carla_id": int(actor.id),
            "type_id": str(actor.type_id),
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
            "yaw": float(transform.rotation.yaw),
            "vx": float(velocity.x),
            "vy": float(velocity.y),
            "vz": float(velocity.z),
            "ax": float(acceleration.x),
            "ay": float(acceleration.y),
            "az": float(acceleration.z),
            "speed": _vector_norm(velocity),
            "acceleration": _vector_norm(acceleration),
            "throttle": float(getattr(control, "throttle", 0.0)),
            "brake": float(getattr(control, "brake", 0.0)),
            "steer": float(getattr(control, "steer", 0.0)),
        }

    def record_step(self, env, action_dict, rss_evaluations=None,
                    controller_restrictions=None):
        step_index = int(env._num_steps.get(self.ego_id, len(self.action_trace)))
        self.action_trace.append(self._copy_actions(action_dict))
        evaluations = (
            list(rss_evaluations) if rss_evaluations is not None
            else evaluate_carla_pairs(env, self.ego_id, self.params)
        )

        actors = {}
        for actor_id, actor in env._actors.items():
            if actor.is_alive:
                actors[actor_id] = self._actor_snapshot(actor)
        ego_state = actors.get(self.ego_id, {})
        ego_speed = ego_state.get("speed", 0.0)
        dt = float(env._env_config.get("fixed_delta_seconds", 0.05))
        if self._previous_ego_speed is not None and dt > 0.0:
            longitudinal_deceleration = max(
                0.0, (self._previous_ego_speed - ego_speed) / dt)
            self._max_ego_deceleration = max(
                self._max_ego_deceleration, longitudinal_deceleration)
        self._previous_ego_speed = ego_speed
        self._max_ego_brake = max(
            self._max_ego_brake, ego_state.get("brake", 0.0))

        for actor_id, actor_state in actors.items():
            if actor_id == self.ego_id:
                continue
            self._max_npc_speed = max(
                self._max_npc_speed, actor_state["speed"])
            self._max_npc_acceleration = max(
                self._max_npc_acceleration, actor_state["acceleration"])

        pair_records = []
        ego_actor = env._actors[self.ego_id]
        ego_location = ego_actor.get_location()
        for evaluation in evaluations:
            pair_record = evaluation.to_dict()
            pair_record["ttc"] = _finite_or_none(evaluation.ttc)
            pair_records.append(pair_record)
            self._max_ego_lateral_speed = max(
                self._max_ego_lateral_speed,
                abs(evaluation.ego_lateral_speed))
            if evaluation.ttc != float("inf"):
                self._min_ttc = min(self._min_ttc, evaluation.ttc)
            self._min_long_margin = min(
                self._min_long_margin, evaluation.longitudinal_margin)
            self._min_lat_margin = min(
                self._min_lat_margin, evaluation.lateral_margin)
            if evaluation.dangerous and self._danger_onset_step is None:
                self._danger_onset_step = step_index

        for actor in env.world.get_actors().filter("vehicle.*"):
            if actor.id == ego_actor.id or not actor.is_alive:
                continue
            location = actor.get_location()
            distance = math.sqrt(
                float(location.x - ego_location.x) ** 2
                + float(location.y - ego_location.y) ** 2)
            self._min_center_distance = min(self._min_center_distance, distance)

        if self.keep_full_trace:
            self.state_trace.append({
                "step": step_index,
                "actors": actors,
                "rss_pairs": pair_records,
                "rss_restrictions": controller_restrictions or {},
            })

    def summary(self, env, collision_flag):
        ego = env._actors.get(self.ego_id)
        ego_speed = _vector_norm(ego.get_velocity()) if ego and ego.is_alive else 0.0
        collision_sensor = env._collisions.get(self.ego_id)
        events = list(getattr(collision_sensor, "events", []))
        last_event = events[-1] if events else {}
        collision_other_id = last_event.get("other_actor_id")
        collision_partner = None
        relative_speed = None
        impact_angle = None
        collision_relative_longitudinal = None
        collision_relative_lateral = None
        collision_with_controlled_npc = None
        if ego and ego.is_alive and collision_other_id is not None:
            other = env.world.get_actor(int(collision_other_id))
            if other is not None and other.is_alive:
                collision_partner = env._actor_id_in_carla.get(
                    other.id, "vehicle_{}".format(other.id))
                ego_velocity = ego.get_velocity()
                other_velocity = other.get_velocity()
                relative_speed = math.sqrt(
                    float(ego_velocity.x - other_velocity.x) ** 2
                    + float(ego_velocity.y - other_velocity.y) ** 2)
                impact_angle = abs(
                    float(ego.get_transform().rotation.yaw)
                    - float(other.get_transform().rotation.yaw)) % 360.0
                if impact_angle > 180.0:
                    impact_angle = 360.0 - impact_angle
                ego_transform = ego.get_transform()
                ego_location = ego_transform.location
                other_location = other.get_location()
                forward = ego_transform.get_forward_vector()
                right = ego_transform.get_right_vector()
                dx = float(other_location.x - ego_location.x)
                dy = float(other_location.y - ego_location.y)
                collision_relative_longitudinal = (
                    dx * float(forward.x) + dy * float(forward.y))
                collision_relative_lateral = (
                    dx * float(right.x) + dy * float(right.y))
                collision_with_controlled_npc = collision_partner in env._actors

        return {
            "scenario_type": self.scenario_type,
            "npc_count": max(0, len(env._actors) - 1),
            "collision": bool(collision_flag),
            "collision_step": env._time_to_firstcollision.get(self.ego_id),
            "danger_onset_step": self._danger_onset_step,
            "min_ttc": _finite_or_none(self._min_ttc),
            "min_rss_longitudinal_margin": _finite_or_none(
                self._min_long_margin),
            "min_rss_lateral_margin": _finite_or_none(self._min_lat_margin),
            "min_center_distance": _finite_or_none(self._min_center_distance),
            "ego_speed_at_end": float(ego_speed),
            "max_ego_brake_command": float(self._max_ego_brake),
            "max_ego_deceleration": float(self._max_ego_deceleration),
            "max_ego_lateral_speed": float(self._max_ego_lateral_speed),
            "max_npc_speed": float(self._max_npc_speed),
            "max_npc_acceleration": float(self._max_npc_acceleration),
            "collision_partner": collision_partner,
            "collision_impulse": last_event.get("impulse"),
            "relative_speed_at_collision": relative_speed,
            "impact_angle": impact_angle,
            "collision_relative_longitudinal": collision_relative_longitudinal,
            "collision_relative_lateral": collision_relative_lateral,
            "collision_with_controlled_npc": collision_with_controlled_npc,
            "rss_parameters": self.params.to_dict(),
        }

    def archive_payload(self):
        return {
            "action_trace": self.action_trace,
            "state_trace": self.state_trace,
        }


class JsonlArchive:
    def __init__(self, output_dir, scenario_type):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.path = os.path.join(
            self.output_dir,
            "{}_rss_counterfactual.jsonl".format(scenario_type),
        )

    def append(self, record):
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                record, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
