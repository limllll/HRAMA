#!/bin/env python
# -*- coding: utf-8 -*-
'''
LaneChange Random Strategy (RS) Inference Script.
随机策略：car1, car2, car3 使用随机策略，ego 由 CARLA 行为代理控制
'''
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

configs = env.configs
actor_ids = list(configs["actors"].keys())
max_steps = 80
eps = args.episodes


def calculate_rc(current, initial):
    if initial == 0:
        return 0
    diff = initial - current
    return max(0, diff / initial)


def get_random_action(action_space, actor_id):
    try:
        from gym.spaces import Dict as GymDict
    except ImportError:
        GymDict = None

    if hasattr(action_space, "spaces") or (GymDict and isinstance(action_space, GymDict)):
        try:
            if hasattr(action_space, "spaces") and actor_id in action_space.spaces:
                actor_action_space = action_space.spaces[actor_id]
            else:
                actor_action_space = action_space[actor_id]
        except (KeyError, TypeError):
            actor_action_space = action_space.spaces.get(actor_id) if hasattr(action_space, "spaces") else None
            if actor_action_space is None:
                raise ValueError(f"无法找到智能体 {actor_id} 的动作空间")
    elif isinstance(action_space, dict):
        actor_action_space = action_space[actor_id]
    else:
        actor_action_space = action_space

    if hasattr(actor_action_space, "nvec"):
        action = np.array([np.random.randint(0, n) for n in actor_action_space.nvec], dtype=np.int32)
        return action
    if hasattr(actor_action_space, "sample"):
        sampled_action = actor_action_space.sample()
        if isinstance(sampled_action, dict):
            values = list(sampled_action.values())
            if len(values) >= 2:
                action = np.array([int(values[0]), int(values[1])], dtype=np.int32)
            else:
                action = np.array([2, 2], dtype=np.int32)
        elif isinstance(sampled_action, np.ndarray):
            action = sampled_action.astype(np.int32)
        elif isinstance(sampled_action, (list, tuple)):
            action = np.array(sampled_action, dtype=np.int32)
        else:
            action = np.array([2, 2], dtype=np.int32)

        if len(action) > 2:
            action = action[:2]
        elif len(action) < 2:
            action = np.array([2, 2], dtype=np.int32)

        action[0] = np.clip(int(action[0]), 0, 4)
        action[1] = np.clip(int(action[1]), 0, 4)
        return action.astype(np.int32)

    return np.array([2, 2], dtype=np.int32)


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

print("RS baseline: car1/car2/car3 random, ego by BehaviorAgent.")

for ep in range(eps):
    obs = env.reset()
    total_reward_dict = {aid: 0 for aid in actor_ids}
    done = {"__all__": False}
    step = 0
    start = time.time()

    while not done["__all__"]:
        step += 1
        action_dict = {}
        for aid in actor_ids:
            if aid == "ego":
                action_dict[aid] = np.array([2, 2], dtype=np.int32)
            else:
                action_dict[aid] = get_random_action(env.action_space, aid)

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
        print(f"⚠️ Episode {ep}: info['ego'] 中缺少 'bg_lde' 键，使用默认值 0.0")
        ep_bg_lde = 0.0
    metrics["bg_lde"].append(ep_bg_lde)

    print(f"Episode {ep} | FPS: {step / (time.time() - start):.2f} | Ego Hit: {is_ego_hit} | BG_LDE: {ep_bg_lde:.4f}")

csv_file = "inference_LaneChange_RS_Results.csv"
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
print(" FINAL METRICS REPORT (Random Strategy)")
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
