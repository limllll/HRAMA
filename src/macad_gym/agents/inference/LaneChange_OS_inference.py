#!/bin/env python
# -*- coding: utf-8 -*-
"""
LaneChange Optimal Strategy (OS) Inference Script.

OS baseline:
- ego/car1/car2/car3 all use CARLA BehaviorAgent(Local Planner) driving.
- No adversarial action.
- Used as cooperative non-adversarial baseline against AMA policies.
"""

import argparse
import csv
import datetime
import math
import time

import gym
import macad_gym  # noqa: F401
import numpy as np


start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Town04C3-v0", help="Name Gym env")
parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes")
parser.add_argument("--max-steps", type=int, default=80, help="TTD/TTFC fallback max steps")
args = parser.parse_args()

env_name = args.env
env = gym.make(env_name)
env_actor_configs = env.configs

ENV_SETTINGS = {
    "enable_sumo_cosim": True,
    "sumo_gui": False,
    "verbose": True,
    "enable_ngsim_background": True,
    "ngsim_csv_file": "ngsim_vehicle_final_120.csv",
    "sumo_to_carla_sync": True,
    "ngsim_danger_distance_threshold": 5.0,
    "ngsim_danger_ttc_threshold": 2.0,
}

if "env" not in env_actor_configs:
    env_actor_configs["env"] = {}
env_actor_configs["env"].update(ENV_SETTINGS)

# Core OS setup:
# Force all agents to use local planner through auto_control.
for aid, cfg in env_actor_configs.get("actors", {}).items():
    cfg["auto_control"] = True

configs = env.configs
actor_ids = list(configs["actors"].keys())
max_steps = args.max_steps
eps = args.episodes


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
}

print("OS baseline: ego/car1/car2/car3 all use BehaviorAgent(local planner).")

for ep in range(eps):
    obs = env.reset()
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
    metrics["ego_ttfc"].append(ego_ttfc if is_ego_hit and ego_ttfc else max_steps)

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

    if "ego" in info and "bg_lde" in info["ego"]:
        ep_bg_lde = info["ego"]["bg_lde"]
    else:
        print(f"Episode {ep}: info['ego'] missing 'bg_lde', fallback to 0.0")
        ep_bg_lde = 0.0
    metrics["bg_lde"].append(ep_bg_lde)

    print(
        f"Episode {ep} | FPS: {step / (time.time() - start):.2f} | "
        f"Ego Hit: {is_ego_hit} | BG_LDE: {ep_bg_lde:.4f}"
    )

csv_file = "inference_LaneChange_OS_Results.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["episode", "R_collision", "RC", "OL", "TTD", "TTFC", "Ego_Cols", "Total_Cols", "NPC_LDE", "BG_LDE"]
    )
    for i in range(eps):
        writer.writerow(
            [
                i + 1,
                metrics["ego_collision_flag"][i],
                metrics["ego_rc"][i],
                metrics["ego_ol"][i],
                metrics["ego_ttd"][i],
                metrics["ego_ttfc"][i],
                metrics["global_ego_cols"][i],
                metrics["global_total_cols"][i],
                metrics["npc_lde"][i],
                metrics["bg_lde"][i],
            ]
        )

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
print(f"8. BG_LDE (NGSIM Background Vehicle Trajectory Error): {np.mean(metrics['bg_lde']):.4f}")
print("=" * 44)
print(f"CSV saved to: {csv_file}")
