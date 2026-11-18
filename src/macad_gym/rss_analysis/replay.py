"""Deterministic NPC-action replay with an RSS-supervised AVUT."""

from __future__ import division

import traceback

import numpy as np

from .model import RSSParameters
from .recording import (
    EpisodeRecorder,
    capture_initial_scene_state,
    compare_initial_scene_states,
)
from .reference_controller import RSSReferenceController


def _ego_collision(env, ego_id="ego"):
    collision = env._collisions[ego_id]
    return bool(collision.collision_vehicles + collision.collision_other
                + collision.collision_pedestrians > 0)


def run_rss_counterfactual_replay(
    env,
    recorded_action_trace,
    initial_random_state,
    original_initial_scene_state,
    post_original_random_state,
    scenario_type,
    params=None,
    ego_id="ego",
    keep_full_trace=False,
):
    """Replay one collision episode and replace only AVUT control.

    The NPC action sequence is held fixed.  Python and NumPy RNG states are
    restored around the replay so the reference reset reproduces the original
    random blueprint/weather choices and does not perturb subsequent episodes.
    """

    params = params or RSSParameters()
    controller = RSSReferenceController(params=params, ego_id=ego_id)
    recorder = EpisodeRecorder(
        scenario_type=scenario_type,
        params=params,
        ego_id=ego_id,
        keep_full_trace=keep_full_trace,
    )
    old_cumulative_reward = env._env_config.get("cummulated_reward", False)
    original_episode_counter = getattr(env, "_episode", None)
    try:
        env._env_config["cummulated_reward"] = False
        initial_random_state.restore()
        env.reset()
        replay_initial_scene_state = capture_initial_scene_state(env)
        initial_state_check = compare_initial_scene_states(
            original_initial_scene_state, replay_initial_scene_state)
        if not initial_state_check["matches"]:
            return {
                "succeeded": False,
                "reference_collision": None,
                "reference_road_violation": None,
                "reference_offlane_events": None,
                "reference_offroad_events": None,
                "steps_replayed": 0,
                "summary": None,
                "trace": None,
                "initial_state_check": initial_state_check,
                "error": "RSS replay initial scene does not match original reset",
            }
        controller.reset()

        for step_index, saved_actions in enumerate(recorded_action_trace):
            action_dict = {
                actor_id: np.asarray(action, dtype=np.int32)
                for actor_id, action in saved_actions.items()
            }
            reference_control = controller.compute_control(env, step_index)
            env.set_external_actor_control(ego_id, reference_control)
            obs, reward, done, info = env.step(action_dict)
            recorder.record_step(
                env,
                action_dict,
                rss_evaluations=controller.last_evaluations,
                controller_restrictions=controller.last_restrictions,
            )
            if _ego_collision(env, ego_id) or done.get("__all__", False):
                break

        reference_collision = _ego_collision(env, ego_id)
        lane_sensor = env._lane_invasions.get(ego_id)
        reference_offlane = int(getattr(lane_sensor, "offlane", 0))
        reference_offroad = int(getattr(lane_sensor, "offroad", 0))
        reference_road_violation = bool(reference_offlane + reference_offroad)
        return {
            "succeeded": True,
            "reference_collision": reference_collision,
            "reference_road_violation": reference_road_violation,
            "reference_offlane_events": reference_offlane,
            "reference_offroad_events": reference_offroad,
            "steps_replayed": len(recorder.action_trace),
            "summary": recorder.summary(env, reference_collision),
            "trace": recorder.archive_payload() if keep_full_trace else None,
            "initial_state_check": initial_state_check,
            "error": None,
        }
    except Exception:
        return {
            "succeeded": False,
            "reference_collision": None,
            "reference_road_violation": None,
            "reference_offlane_events": None,
            "reference_offroad_events": None,
            "steps_replayed": len(recorder.action_trace),
            "summary": None,
            "trace": None,
            "initial_state_check": None,
            "error": traceback.format_exc(),
        }
    finally:
        env._env_config["cummulated_reward"] = old_cumulative_reward
        if original_episode_counter is not None:
            env._episode = original_episode_counter
        post_original_random_state.restore()
