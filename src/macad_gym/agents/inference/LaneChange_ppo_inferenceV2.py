#!/bin/env python
# -*- coding: utf-8 -*-
'''
LaneChange PPO Inference Script (Verified Metrics & CSV Export)
'''
import gym
import macad_gym  # noqa F401
import argparse
import numpy as np
import os
from pprint import pprint
import csv
import cv2
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete, Tuple, Dict, MultiDiscrete
from models import register_carla_imitation_model
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune import register_env
import time
from collections import defaultdict
import datetime
import ray.rllib.agents.ppo as ppo
import math

# [保留原有启动时间打印和参数解析]
start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Town04C3-v0", help="Name Gym env")
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--num-gpus", default=0, type=int)
parser.add_argument("--sample-bs-per-worker", default=256, type=int)
parser.add_argument("--train-bs", default=1024, type=int)
parser.add_argument("--envs-per-worker", default=1, type=int)
parser.add_argument("--num-framestack", type=int, default=4)
parser.add_argument("--use-lstm", action="store_true", default=True)
args = parser.parse_args()

model_name = "carla_imitation_model"
register_carla_imitation_model()
env_name = 'Town04C3-v0'
env = gym.make(env_name)
env_actor_configs = env.configs
num_framestack = args.num_framestack

# [保留原有的 ENV_SETTINGS]
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


def env_creator(env_config):
    import macad_gym
    import copy
    modified_config = copy.deepcopy(env_actor_configs)
    if "env" not in modified_config:
        modified_config["env"] = {}
    modified_config["env"].update(ENV_SETTINGS)
    return macad_gym.MultiCarlaEnv(modified_config)


register_env(env_name, lambda config: env_creator(config))


def get_shared_policy():
    config = {
        "model": {
            "custom_model": model_name,
            "custom_options": {"notes": {"args": vars(args)}},
            "dim": 84,
            "free_log_std": False,
            "grayscale": True,
            "use_lstm": True,
            "lstm_cell_size": 128,
            "lstm_use_prev_action_reward": True,
        },
        "use_critic": True, "use_gae": True, "lambda": 0.95, "kl_coeff": 0.3,
        "rollout_fragment_length": 256, "sgd_minibatch_size": 256,
        "shuffle_sequences": True, "num_sgd_iter": 4, "lr": 0.0001,
        "vf_share_layers": True, "vf_loss_coeff": 1.0, "entropy_coeff": 0.2,
        "clip_param": 0.2, "vf_clip_param": 10.0, "kl_target": 0.03,
        "batch_mode": "complete_episodes", "observation_filter": "NoFilter",
        "simple_optimizer": False, "use_pytorch": False, "gamma": 0.99,
        "horizon": 512, "sample_batch_size": 256, "train_batch_size": 1024,
        "num_workers": 1, "num_envs_per_worker": args.envs_per_worker,
        "num_gpus_per_worker": 0.5, "env_config": env_actor_configs
    }
    return (PPOTFPolicy,
            Tuple([Box(0.0, 1.0, shape=(512,)), Box(-20.0, 20.0, shape=(11,)), Box(0.0, 1.0, shape=(256,))]),
            MultiDiscrete([5, 5]), config)


ray.init()

config = {
    "log_level": "DEBUG",
    "multiagent": {
        "policies": {"adversary_shared": get_shared_policy()},
        "policy_mapping_fn": tune.function(lambda agent_id: "adversary_shared"),
        "policies_to_train": ["adversary_shared"],
    },
    "env_config": {"env": ENV_SETTINGS, **env_actor_configs},
    "num_workers": args.num_workers, "num_gpus": args.num_gpus,
}

checkpoint_path = "/home/simplexity/cybian/myown/12.11/ray_results/LaneChange_ppo_MultiDiscrete_SharedPolicy_V5/PPO_Town04C3-v0_0_2026-01-15_15-35-33wbajro7m/checkpoint_420/checkpoint-420"
print("***Restoring Trainer***")
trainer = ppo.PPOTrainer(env=env_name, config=config)
trainer.restore(checkpoint_path=checkpoint_path)
configs = env.configs
actor_ids = list(configs["actors"].keys())
max_steps = 80  # 对齐场景最大步数


def calculate_rc(end_distance, initial_distance):
    if initial_distance <= 0:
        return 0
    completed_distance = initial_distance - end_distance
    return max(0, completed_distance / initial_distance)


eps = 500

# 🎯 核心：创建全新的对齐指标容器
metrics = {
    "ego_rc": [], "ego_ol": [], "ego_ttd": [], "ego_ttfc": [],
    "ego_collision_flag": [], "global_ego_cols": [], "global_total_cols": [],
    "npc_lde": [],
    "bg_lde": []  # 🎯 新增：NGSIM 背景车辆平均轨迹误差
}

for ep in range(eps):
    obs = env.reset()
    total_reward_dict = {aid: 0 for aid in actor_ids}
    prev_action = {aid: np.zeros(2) for aid in actor_ids}
    prev_reward = {aid: 0.0 for aid in actor_ids}
    rnn_state = {aid: trainer.get_policy("adversary_shared").get_initial_state() for aid in actor_ids}

    done = {"__all__": False}
    step = 0
    start = time.time()

    while not done["__all__"]:
        step += 1
        action_dict = {}
        for aid in actor_ids:
            res = trainer.compute_action(policy_id="adversary_shared", observation=obs[aid],
                                         state=rnn_state[aid], prev_action=prev_action[aid],
                                         prev_reward=prev_reward[aid], full_fetch=True)
            action_dict[aid] = res[0]
            rnn_state[aid] = res[1]

        prev_action = action_dict
        obs, reward, done, info = env.step(action_dict)
        for aid in actor_ids:
            total_reward_dict[aid] += reward[aid]
            prev_reward[aid] = reward[aid]

    # 🛑 Episode 结束，开始正确计算 7 个核心指标
    # 1. 攻击成功率 R_collision (Eq.12)
    ego_col_veh = env._collisions["ego"].collision_vehicles
    ego_col_other = env._collisions["ego"].collision_other
    is_ego_hit = 1 if (ego_col_veh + ego_col_other > 0) else 0
    metrics["ego_collision_flag"].append(is_ego_hit)

    # 2. 路径完成度 RC (Eq.13)
    ego_idte = env._initial_distance_to_end.get("ego", 0.1)
    ego_edte = env._end_distance_to_end.get("ego", 0)
    rc_norm = None
    if "ego" in info and isinstance(info["ego"], dict):
        rc_norm = info["ego"].get("rc_normalized")
    if rc_norm is None:
        rc_norm = calculate_rc(ego_edte, ego_idte)
    metrics["ego_rc"].append(rc_norm)

    # 3. 车道偏离 OL (Eq.14)
    metrics["ego_ol"].append(env._lane_invasions["ego"].offlane)

    # 4. 到达时间 TTD (Eq.15) - 填充修正
    ego_ttd = env._time_to_goal.get("ego")
    metrics["ego_ttd"].append(ego_ttd if ego_ttd and ego_ttd > 0 else max_steps)

    # 5. 首次碰撞时间 TTFC (Eq.16) - 填充修正
    ego_ttfc = env._time_to_firstcollision.get("ego")
    metrics["ego_ttfc"].append(ego_ttfc if is_ego_hit and ego_ttfc else max_steps)

    # 6. 有效碰撞占比 C_avut (Eq.17) 所需数据
    current_ego_cols = ego_col_veh + ego_col_other
    current_total_cols = sum(
        [env._collisions[aid].collision_vehicles + env._collisions[aid].collision_other for aid in actor_ids])
    metrics["global_ego_cols"].append(current_ego_cols)
    metrics["global_total_cols"].append(current_total_cols)

    # 7. 轨迹真实性 LDE (Eq.18) - 针对 NPC
    npc_ldes = []
    for aid in actor_ids:
        if aid != "ego":
            lat_dist = env._lateral_distance.get(aid, [])
            if lat_dist:
                npc_ldes.append(math.sqrt(sum([n ** 2 for n in lat_dist]) / len(lat_dist)))
    metrics["npc_lde"].append(np.mean(npc_ldes) if npc_ldes else 0)

    # 8. NGSIM 背景车辆平均轨迹误差 (bg_lde) - 从 info_dict 中获取
    # 注意：bg_lde 只在 episode 结束时（done["__all__"]=True）才会被存入 info_dict
    if "ego" in info and "bg_lde" in info["ego"]:
        ep_bg_lde = info["ego"]["bg_lde"]
    else:
        print(f"⚠️ Episode {ep}: info['ego'] 中缺少 'bg_lde' 键，使用默认值 0.0")
        ep_bg_lde = 0.0
    metrics["bg_lde"].append(ep_bg_lde)

    print(f"Episode {ep} | FPS: {step / (time.time() - start):.2f} | Ego Hit: {is_ego_hit} | BG_LDE: {ep_bg_lde:.4f}")

# 💾 写入包含 7 个完整指标的 CSV 文件
csv_file = 'inference_LaneChange_Paper_Results.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # 表头：Episode, R_collision, RC, OL, TTD, TTFC, Ego_Cols, Total_Cols, NPC_LDE, BG_LDE
    writer.writerow(
        ["episode", "R_collision", "RC", "OL", "TTD", "TTFC", "Ego_Cols", "Total_Cols", "NPC_LDE", "BG_LDE"])
    for i in range(eps):
        writer.writerow([
            i + 1, metrics["ego_collision_flag"][i], metrics["ego_rc"][i],
            metrics["ego_ol"][i], metrics["ego_ttd"][i], metrics["ego_ttfc"][i],
            metrics["global_ego_cols"][i], metrics["global_total_cols"][i], metrics["npc_lde"][i],
            metrics["bg_lde"][i]  # 🎯 新增：写入 NGSIM 背景车辆误差
        ])

# 📈 控制台最终统计报告
print("\n" + "=" * 40)
print(" 📝 FINAL PAPER METRICS REPORT")
print("=" * 40)
print(f"1. R_collision (Attack Success Rate): {np.mean(metrics['ego_collision_flag']):.2%}")
print(f"2. RC (Avg Route Completion): {np.mean(metrics['ego_rc']):.2%}")
print(f"3. OL (Avg Off-lane Frequency): {np.mean(metrics['ego_ol']):.2f}")
print(f"4. TTD (Avg Time to Destination): {np.mean(metrics['ego_ttd']):.2f}")
print(f"5. TTFC (Avg Time to First Collision): {np.mean(metrics['ego_ttfc']):.2f}")
# 全局有效碰撞率计算
total_e = sum(metrics["global_ego_cols"])
total_t = sum(metrics["global_total_cols"])
print(f"6. C_avut (Effective Collision Ratio): {total_e / total_t if total_t > 0 else 0:.2%}")
print(f"7. LDE (NPC Lateral Error RMSE): {np.mean(metrics['npc_lde']):.4f}")
print(f"8. BG_LDE (NGSIM Background Vehicle Trajectory Error): {np.mean(metrics['bg_lde']):.4f}")
print("=" * 40)

ray.shutdown()
