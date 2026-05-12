#!/bin/env python
# -*- coding: utf-8 -*-
"""
LaneChange PPO Inference Script aligned with structural-ablation training.
"""

import argparse
import copy
import csv
import datetime
import math
import os
import tempfile
import time

import gym
import macad_gym  # noqa: F401
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import ray.tune as tune
import yaml
from gym.spaces import Box, MultiDiscrete, Tuple
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune import register_env

# Use the same model definition as training to guarantee compatibility.
from macad_gym.agents.train.models import register_carla_imitation_model


start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Town04C3-v0", help="Name Gym env")
parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes")
parser.add_argument("--max-steps", type=int, default=80, help="TTD/TTFC fallback max steps")
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--num-gpus", default=0, type=int)
parser.add_argument("--envs-per-worker", default=1, type=int)
parser.add_argument(
    "--structural-ablation",
    default="full",
    choices=["image_only", "image_state", "image_state_lidar", "full"],
    help=(
        "结构消融: image_only=仅图像; image_state=图像+状态编码; "
        "image_state_lidar=图像+状态+点云(无LSTM); full=同上+LSTM"
    ),
)
parser.add_argument(
    "--checkpoint-path",
    default="",
    help="(Optional) Override in-file checkpoint_path",
)
parser.add_argument(
    "--ego-ads",
    default="wor",
    choices=["behavior", "wor"],
    help="Which ADS controls ego during inference.",
)
parser.add_argument(
    "--wor-agent",
    default="image",
    choices=["image", "lbc"],
    help="World on Rails agent variant to use when --ego-ads wor.",
)
parser.add_argument(
    "--wor-root",
    default=os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "WorldOnRails-release",
        )
    ),
    help="Local WorldOnRails repo root.",
)
parser.add_argument(
    "--wor-config",
    default=os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "leaderboard_weights-selected",
            "config_leaderboard.yaml",
        )
    ),
    help="WoR config yaml to use for leaderboard inference.",
)
parser.add_argument(
    "--wor-main-model",
    default=os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "leaderboard_weights-selected",
            "main_model_10.th",
        )
    ),
    help="Path to WoR phase-2 main model weights. Required for --wor-agent image.",
)
parser.add_argument(
    "--wor-rgb-model",
    default="",
    help="Path to LBC RGB model weights. Required for --wor-agent lbc.",
)
args = parser.parse_args()

STRUCTURAL_ABLATION = args.structural_ablation
USE_LSTM_POLICY = STRUCTURAL_ABLATION == "full"
print("=" * 60)
print("structural_ablation =", STRUCTURAL_ABLATION)
print("model.use_lstm      =", USE_LSTM_POLICY)
print("=" * 60)

# Same style as LaneChange_ppo_inferenceV2.py: fill this path directly.
checkpoint_path = "/home/simplexity/cybian/myown/12.11/ablation_experiment/inference_becausechangethemetrics/ray_results/LaneChange_ppo_MultiDiscrete_SharedPolicy_full/PPO_Town04C3-v0_0_2026-04-03_21-26-261om_jrht/checkpoint_40/checkpoint-40"

model_name = "carla_imitation_model"
register_carla_imitation_model()
env_name = args.env
_base_env = gym.make(env_name)
env_actor_configs = _base_env.configs

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


def build_wor_runtime_config():
    wor_root = os.path.abspath(args.wor_root)
    config_path = os.path.abspath(args.wor_config) if args.wor_config else os.path.join(wor_root, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"WorldOnRails config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["log_wandb"] = False
    config["noise_collect"] = False

    ego_model_path = os.path.join(wor_root, "ego_model.th")
    if os.path.exists(ego_model_path):
        config["ego_model_dir"] = ego_model_path

    if args.wor_agent == "image":
        main_model_path = args.wor_main_model
        if not main_model_path:
            candidate_names = [
                "main_model.th",
                "main_model.pth",
                "phase2_model.th",
                "phase2_model.pth",
            ]
            for name in candidate_names:
                candidate = os.path.join(wor_root, name)
                if os.path.exists(candidate):
                    main_model_path = candidate
                    break
        if not main_model_path or not os.path.exists(main_model_path):
            raise FileNotFoundError(
                "WoR image agent requires phase-2 model weights. "
                "Pass --wor-main-model with the downloaded pretrained model path."
            )
        config["main_model_dir"] = os.path.abspath(main_model_path)
    else:
        rgb_model_path = args.wor_rgb_model
        if not rgb_model_path:
            candidate_names = [
                "rgb_model.th",
                "rgb_model.pth",
                "lbc_rgb_model.th",
                "lbc_rgb_model.pth",
            ]
            for name in candidate_names:
                candidate = os.path.join(wor_root, name)
                if os.path.exists(candidate):
                    rgb_model_path = candidate
                    break
        if not rgb_model_path or not os.path.exists(rgb_model_path):
            raise FileNotFoundError(
                "WoR LBC agent requires RGB model weights. "
                "Pass --wor-rgb-model with the trained/downloaded LBC RGB model path."
            )
        config["rgb_model_dir"] = os.path.abspath(rgb_model_path)

    fd, runtime_config_path = tempfile.mkstemp(
        prefix="wor_lanechange_",
        suffix=".yaml",
    )
    os.close(fd)
    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return wor_root, runtime_config_path


def apply_ego_ads_override(actor_configs):
    modified = copy.deepcopy(actor_configs)
    ego_cfg = modified["actors"]["ego"]

    if args.ego_ads == "behavior":
        ego_cfg["auto_control"] = True
        ego_cfg["control_backend"] = "behavior"
        ego_cfg.pop("control_backend_config", None)
        return modified

    wor_root, runtime_config_path = build_wor_runtime_config()
    agent_file = (
        os.path.join(wor_root, "autoagents", "image_agent.py")
        if args.wor_agent == "image"
        else os.path.join(wor_root, "autoagents", "lbc_agent.py")
    )
    class_name = "ImageAgent" if args.wor_agent == "image" else "LBCAgent"

    ego_cfg["auto_control"] = True
    ego_cfg["control_backend"] = "world_on_rails"
    ego_cfg["control_backend_config"] = {
        "root_dir": wor_root,
        "module": agent_file,
        "class_name": class_name,
        "config_path": runtime_config_path,
        "python_paths": [
            wor_root,
            os.path.join(wor_root, "leaderboard"),
            os.path.join(wor_root, "scenario_runner"),
        ],
    }
    return modified


env_actor_configs = apply_ego_ads_override(env_actor_configs)


def env_creator(env_config):
    import copy
    import macad_gym

    modified_config = copy.deepcopy(env_actor_configs)
    if "env" not in modified_config:
        modified_config["env"] = {}
    modified_config["env"].update(ENV_SETTINGS)
    return macad_gym.MultiCarlaEnv(modified_config)


register_env(env_name, lambda config: env_creator(config))

# IMPORTANT:
# The evaluation loop below should run on the same config that includes the
# ego ADS override (e.g. WoR). Re-create the env after applying overrides.
try:
    _base_env.close()
except Exception:
    pass
env = env_creator({})


def get_shared_policy():
    config = {
        "model": {
            "custom_model": model_name,
            "custom_options": {
                "notes": {"args": vars(args)},
                "structural_ablation": STRUCTURAL_ABLATION,
            },
            "dim": 84,
            "free_log_std": False,
            "grayscale": True,
            "use_lstm": USE_LSTM_POLICY,
            "lstm_cell_size": 128,
            "lstm_use_prev_action_reward": True,
        },
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.95,
        "kl_coeff": 0.3,
        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        "shuffle_sequences": True,
        "num_sgd_iter": 4,
        "lr": 0.0001,
        "vf_share_layers": True,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": 0.2,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "kl_target": 0.03,
        "batch_mode": "complete_episodes",
        "observation_filter": "NoFilter",
        "simple_optimizer": False,
        "use_pytorch": False,
        "gamma": 0.99,
        "horizon": 512,
        "sample_batch_size": 256,
        "train_batch_size": 1024,
        "num_workers": 1,
        "num_envs_per_worker": args.envs_per_worker,
        "num_gpus_per_worker": 0.5,
        "env_config": env_actor_configs,
    }
    return (
        PPOTFPolicy,
        Tuple(
            [
                Box(0.0, 1.0, shape=(512,)),
                Box(-20.0, 20.0, shape=(11,)),
                Box(0.0, 1.0, shape=(256,)),
            ]
        ),
        MultiDiscrete([5, 5]),
        config,
    )


ray.init()

trainer_config = {
    "log_level": "DEBUG",
    "multiagent": {
        "policies": {"adversary_shared": get_shared_policy()},
        "policy_mapping_fn": tune.function(lambda agent_id: "adversary_shared"),
        "policies_to_train": ["adversary_shared"],
    },
    "env_config": {"env": ENV_SETTINGS, **env_actor_configs},
    "num_workers": args.num_workers,
    "num_gpus": args.num_gpus,
}

trainer = ppo.PPOTrainer(env=env_name, config=trainer_config)
trainer.restore(checkpoint_path=(args.checkpoint_path or checkpoint_path))

actor_ids = list(env.configs["actors"].keys())
ego_autocontrol = bool(env.configs["actors"].get("ego", {}).get("auto_control", False))
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
            # If ego is controlled by an autonomous backend (e.g. WoR),
            # the env will ignore its RL action. Provide a stable "no-op"
            # action to keep shapes consistent without wasting inference.
            if aid == "ego" and ego_autocontrol:
                action_dict[aid] = np.array([2, 2], dtype=np.int64)
                continue
            result = trainer.compute_action(
                policy_id="adversary_shared",
                observation=obs[aid],
                state=rnn_state[aid],
                prev_action=prev_action[aid],
                prev_reward=prev_reward[aid],
                full_fetch=True,
            )
            action_dict[aid] = result[0]
            rnn_state[aid] = result[1]

        prev_action = action_dict
        obs, reward, done, info = env.step(action_dict)
        for aid in actor_ids:
            total_reward_dict[aid] += reward[aid]
            prev_reward[aid] = reward[aid]

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

csv_file = f"inference_LaneChange_{STRUCTURAL_ABLATION}_Results.csv"
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
print(f" FINAL METRICS REPORT ({STRUCTURAL_ABLATION})")
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

ray.shutdown()
