#!/bin/env python
# -*- coding: utf-8 -*-
"""
Following Optimal Strategy (OS) Inference Script.

OS baseline:
- ego/car1/car2/car3 all use CARLA BehaviorAgent(Local Planner) driving.
- No adversarial action.
- Used as cooperative non-adversarial baseline against AMA policies.
"""

import argparse
import atexit
import datetime
import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
IMPORT_ROOTS = [SRC_ROOT, os.path.join(SRC_ROOT, "macad_gym", "agents")]
for import_root in reversed(IMPORT_ROOTS):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
inherited_paths = [
    path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
os.environ["PYTHONPATH"] = os.pathsep.join(IMPORT_ROOTS + [
    path for path in inherited_paths if path not in IMPORT_ROOTS])

import gym
import macad_gym  # noqa: F401
import numpy as np

from macad_gym.envs.straightroads.FollowingRoads3NPC import (
    DEFAULT_NGSIM_CSV,
    apply_following_scenario_config,
)


from macad_gym.agents.experiment_utils import ensure_dir, file_sha256, seed_everything
from macad_gym.agents.inference.following_baseline_results import FollowingBaselineResults

start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Town04FollowingC3-v0", help="Name Gym env")
parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes")
parser.add_argument("--max-steps", type=int, choices=[80], default=80,
                    help="Fixed following scenario and TTD/TTFC fallback: 80 steps")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument(
    "--output-dir", default=os.path.join(PROJECT_ROOT, "artifacts/following_evaluation/OS_500_seed2026"))
parser.add_argument("--carla-reset-retries", type=int, choices=[1, 2], default=2)
parser.add_argument("--following-layout", choices=["fixed", "ngsim"], default="fixed")
parser.add_argument(
    "--ngsim-csv-file",
    default=DEFAULT_NGSIM_CSV,
    help="NGSIM/Lankershim CSV used by the following scenario.",
)
args = parser.parse_args()
if args.episodes <= 0 or args.max_steps <= 0:
    parser.error("episodes and max-steps must be positive")
seed_everything(args.seed)
os.chdir(PROJECT_ROOT)
output_base = os.path.expanduser(args.output_dir)
if not os.path.isabs(output_base):
    output_base = os.path.join(PROJECT_ROOT, output_base)
output_dir = ensure_dir(os.path.join(output_base, "seed{}".format(args.seed)))

ENV_SETTINGS = {
    "enable_sumo_cosim": True,
    "sumo_gui": False,
    "verbose": True,
    "enable_ngsim_background": True,
    "ngsim_csv_file": args.ngsim_csv_file,
    "following_layout": args.following_layout,
    "cummulated_reward": False,
    "sumo_to_carla_sync": True,
    "ngsim_danger_distance_threshold": 5.0,
    "ngsim_danger_ttc_threshold": 2.0,
    "seed": args.seed,
    "sumo_seed": args.seed,
    "carla_reset_retries": args.carla_reset_retries,
    # BehaviorAgent, random actions, and SUMO do not use CARLA TrafficManager.
    # Match the completed following PPO evaluation and its reset stability fix.
    "enable_carla_traffic_manager": False,
}
results = FollowingBaselineResults(output_dir, "OS")
atexit.register(results.close)
env_name = args.env
env = gym.make(
    env_name, env_overrides=ENV_SETTINGS, ngsim_csv_file=args.ngsim_csv_file)
atexit.register(env.close)
env_actor_configs = env.configs
apply_following_scenario_config(
    env_actor_configs, args.ngsim_csv_file, verbose=True, layout=args.following_layout)
env_actor_configs["scenarios"]["max_steps"] = args.max_steps
ENV_SETTINGS["ngsim_excluded_vehicle_ids"] = (
    env_actor_configs["env"].get("ngsim_excluded_vehicle_ids", []))
env_actor_configs["env"].update(ENV_SETTINGS)

# Core OS setup: keep the existing local-planner strategy unchanged.
for aid, cfg in env_actor_configs.get("actors", {}).items():
    cfg["auto_control"] = True

configs = env.configs
actor_ids = list(configs["actors"].keys())
max_steps = args.max_steps
eps = args.episodes
results.write_config({
    "strategy": "OS",
    "seed": args.seed,
    "episodes": args.episodes,
    "max_steps": args.max_steps,
    "following_layout": args.following_layout,
    "environment": ENV_SETTINGS,
    "scenario": env_actor_configs["scenarios"],
    "actors": env_actor_configs["actors"],
    "action_selection": "BehaviorAgent local planner for ego/car1/car2/car3",
    "metric_protocol": "Following_ppo_inference_structural.py; unchanged per-episode calculations",
    "time_metric_unit": "simulation steps; no-event fallback {}".format(args.max_steps),
    "background_metric_scope": "successfully spawned CARLA background vehicles with valid reference samples",
    "rss_counterfactual": False,
    "script_sha256": file_sha256(__file__),
    "logging_sha256": file_sha256(os.path.join(
        os.path.dirname(__file__), "following_baseline_results.py")),
})


def calculate_rc(current, initial):
    if initial == 0:
        return 0
    diff = initial - current
    return max(0, diff / initial)


metrics = {
    "ego_rc": [],
    "ego_ol": [],
    "ego_ttd": [],
    "ego_ttfc": [],
    "ego_collision_flag": [],
    "global_ego_cols": [],
    "global_total_cols": [],
    "npc_lde": [],
    "bg_lde": [],
    "bg_lde_available": [],
    "cosim_available": [],
    "bg_sample_count": [],
}

print("OS baseline: ego/car1/car2/car3 all use BehaviorAgent(local planner).")

for ep in range(eps):
    obs = env.reset()
    results.begin_episode(env)
    total_reward_dict = {aid: 0 for aid in actor_ids}
    done = {"__all__": False}
    step = 0
    start = time.time()

    while not done["__all__"]:
        step += 1
        action_dict = {}

        # Provide placeholders only.
        # With auto_control=True, env ignores these and uses BehaviorAgent output.
        for aid in actor_ids:
            action_dict[aid] = np.array([2, 2], dtype=np.int32)

        obs, reward, done, info = env.step(action_dict)
        results.record_step(env, step)

        for aid in actor_ids:
            if aid in reward:
                total_reward_dict[aid] += reward[aid]

    ego_col_veh = env._collisions["ego"].collision_vehicles
    ego_col_other = env._collisions["ego"].collision_other
    is_ego_hit = 1 if (ego_col_veh + ego_col_other > 0) else 0
    metrics["ego_collision_flag"].append(is_ego_hit)

    ego_idte = env._initial_distance_to_end.get("ego", 0.1)
    ego_edte = env._end_distance_to_end.get("ego", 0)
    rc_norm = None
    if "ego" in info and isinstance(info["ego"], dict):
        rc_norm = info["ego"].get("rc_normalized")
    if rc_norm is None:
        rc_norm = calculate_rc(ego_edte, ego_idte)
    metrics["ego_rc"].append(rc_norm)

    metrics["ego_ol"].append(env._lane_invasions["ego"].offlane)

    ego_ttd = env._time_to_goal.get("ego")
    metrics["ego_ttd"].append(ego_ttd if ego_ttd and ego_ttd > 0 else max_steps)

    ego_ttfc = env._time_to_firstcollision.get("ego")
    metrics["ego_ttfc"].append(
        ego_ttfc if is_ego_hit and ego_ttfc is not None else max_steps)

    current_ego_cols = ego_col_veh + ego_col_other
    current_total_cols = sum(
        [env._collisions[aid].collision_vehicles + env._collisions[aid].collision_other for aid in actor_ids]
    )
    metrics["global_ego_cols"].append(current_ego_cols)
    metrics["global_total_cols"].append(current_total_cols)

    npc_ldes = []
    for aid in actor_ids:
        if aid != "ego":
            lat_dist = env._lateral_distance.get(aid, [])
            if lat_dist:
                npc_ldes.append(math.sqrt(sum([n ** 2 for n in lat_dist]) / len(lat_dist)))
    metrics["npc_lde"].append(np.mean(npc_ldes) if npc_ldes else 0)

    ego_info = info.get("ego", {}) if isinstance(info, dict) else {}
    bg_lde_available = bool(ego_info.get("bg_lde_available", False))
    cosim_available = bool(ego_info.get("cosim_available", False))
    bg_sample_count = int(ego_info.get("bg_sample_count", 0) or 0)
    ep_bg_lde = ego_info.get("bg_lde", float("nan"))
    if not bg_lde_available or not np.isfinite(ep_bg_lde):
        ep_bg_lde = float("nan")
        print(f"Episode {ep + 1}: BG_LDE unavailable; writing NaN and keeping other metrics")
    metrics["bg_lde"].append(ep_bg_lde)
    metrics["bg_lde_available"].append(int(bg_lde_available))
    metrics["cosim_available"].append(int(cosim_available))
    metrics["bg_sample_count"].append(bg_sample_count)
    results.write_episode(ep + 1, metrics, info, env)

    print(
        f"Episode {ep + 1} | FPS: {step / (time.time() - start):.2f} | "
        f"Ego Hit: {is_ego_hit} | BG_LDE: {ep_bg_lde:.4f}"
    )

results.write_final(metrics, eps)

print("\n" + "=" * 44)
print(" FINAL METRICS REPORT (Optimal Strategy)")
print("=" * 44)
print(f"1. R_collision (Attack Success Rate): {np.mean(metrics['ego_collision_flag']):.2%}")
print(f"2. RC (Avg Route Completion): {np.mean(metrics['ego_rc']):.2%}")
print(f"3. OL (Avg Off-lane Frequency): {np.mean(metrics['ego_ol']):.2f}")
print(f"4. TTD (Avg Time to Destination): {np.mean(metrics['ego_ttd']):.2f}")
print(f"5. TTFC (Avg Time to First Collision): {np.mean(metrics['ego_ttfc']):.2f}")
total_e = sum(metrics["global_ego_cols"])
total_t = sum(metrics["global_total_cols"])
print(f"6. C_avut (Effective Collision Ratio): {total_e / total_t if total_t > 0 else 0:.2%}")
print(f"7. LDE (NPC Lateral Error RMSE): {np.mean(metrics['npc_lde']):.4f}")
valid_bg_lde = [value for value in metrics["bg_lde"] if np.isfinite(value)]
if valid_bg_lde:
    print(f"8. BG_LDE (NGSIM Background Vehicle Trajectory Error): {np.mean(valid_bg_lde):.4f} "
          f"({len(valid_bg_lde)}/{eps} valid episodes)")
else:
    print("8. BG_LDE (NGSIM Background Vehicle Trajectory Error): N/A (0 valid episodes)")
print("=" * 44)

print("CSV saved to: {}".format(results.csv_path))
print("Co-simulation diagnostics saved to: {}".format(results.diagnostic_path))
results.close()
atexit.unregister(results.close)
env.close()
atexit.unregister(env.close)
