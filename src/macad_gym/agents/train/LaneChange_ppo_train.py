#!/bin/env python
'''
cnn model :mnih15, not mnih15_shared_weights
"vf_share_layers": False,
rewardFunction: advrs
除了使用 LSTM，您还可以使用 "preprocessor_pref": "deepmind" 来启用帧堆叠。
'''
import gym
import macad_gym  # noqa F401
import argparse
import os
from pprint import pprint

import cv2
import ray
import ray.tune as tune
from gym.spaces import Box, Discrete, Tuple, Dict, MultiDiscrete  # 务必导入 MultiDiscrete
# from env_wrappers import wrap_deepmind
from models import register_mnih15_net, register_mnih15_shared_weights_net, register_carla_model, \
    register_carla_imitation_model

# from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
# from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy  # 0.8.5
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import register_env
import time
# import tensorflow as tf
# from tensorboardX import SummaryWriter

import datetime

start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    default="Town04C3-v0",
    help="Name Gym env. Used only in debug mode. Default=PongNoFrameskip-v4")
parser.add_argument(
    "--disable-comet",
    action="store_true",
    help="Disables comet logging. Used for local smoke tests")
parser.add_argument(
    "--num-workers",
    default=1,  # 2 #fix
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=0, type=int, help="Number of gpus to use. Default=2")
parser.add_argument(
    "--sample-bs-per-worker",  # one iteration
    default=256,
    type=int,
    help="Number of samples in a batch per worker. Default=50")
parser.add_argument(
    "--train-bs",
    default=1024,
    type=int,
    help="Train batch size. Use as per available GPU mem. Default=500")
parser.add_argument(
    "--envs-per-worker",
    default=1,
    type=int,
    help="Number of env instances per worker. Default=10")
parser.add_argument(
    "--notes",
    default=None,
    help="Custom experiment description to be added to comet logs")
parser.add_argument(
    "--model-arch",
    default="carla_imitation_model",
    help="Model architecture to use. Default=carla")
parser.add_argument(
    "--num-steps",
    default=1000000,
    type=int,
    help="Number of steps to train. Default=20M")
parser.add_argument(
    "--num-iters",
    default=1024,
    type=int,
    help="Number of training iterations. Default=20")
parser.add_argument(
    "--log-graph",
    action="store_true",
    help="Write TF graph on Tensorboard for debugging", default=True)
parser.add_argument(
    "--num-framestack",
    type=int,
    default=4,
    help="Number of obs frames to stack")
parser.add_argument(
    "--debug", action="store_true", help="Run in debug-friendly mode", default=False)
parser.add_argument(
    "--redis-address",
    default=None,
    help="Address of ray head node. Be sure to start ray with"
         "ray start --redis-address <...> --num-gpus<.> before running this script")
parser.add_argument(
    "--use-lstm", action="store_true", help="Append a LSTM cell to the model", default=True)
parser.add_argument(
    "--structural-ablation",
    default="full",
    choices=["image_only", "image_state", "image_state_lidar", "full"],
    help=(
        "结构消融: image_only=仅图像; image_state=图像+状态编码; "
        "image_state_lidar=图像+状态+点云(无LSTM); full=同上+LSTM"))
# SUMO 和 GUI 已設為默認開啟，移除選項

args = parser.parse_args()

STRUCTURAL_ABLATION = args.structural_ablation
# 前三种仅对比多模态骨干，不接 LSTM；full 打开 RLlib LSTM
USE_LSTM_POLICY = STRUCTURAL_ABLATION == "full"
print("=" * 60)
print("structural_ablation =", STRUCTURAL_ABLATION)
print("model.use_lstm      =", USE_LSTM_POLICY)
print("=" * 60)

model_name = "carla_imitation_model"
register_carla_imitation_model()
# Used only in debug mode
env_name = 'Town04C3-v0'
env = gym.make(env_name)

env_actor_configs = env.configs
num_framestack = args.num_framestack

# env_config["env"]["render"] = False

# ========== 统一的环境配置（SUMO/NGSIM设置）==========
# 在这里统一管理，env_creator 和 tune.run 都使用这个配置
ENV_SETTINGS = {
    "enable_sumo_cosim": True,  # 启用 SUMO 联合仿真
    "sumo_gui": False,  # 启用 SUMO GUI
    "verbose": True,  # 启用详细日志
    "enable_ngsim_background": True,  # 🎯 是否启用NGSIM背景交通
    "ngsim_csv_file": "ngsim_vehicle_final_120.csv",  # 🎯 NGSIM数据文件（如 "final_60.csv"）
    "sumo_to_carla_sync": True,  # 🎯 SUMO和CARLA同步

    # ============================================================================
    # 🎯 NGSIM状态机配置（简化版）- "NGSIM回放 + 危险时SUMO永久接管"
    # ============================================================================
    # 两种控制模式:
    #   - REPLAY_MODE: 强制按NGSIM轨迹行驶（默认模式）
    #   - FREE_MODE: 检测到危险时，永久由SUMO接管
    # ============================================================================
    "ngsim_danger_distance_threshold": 5.0,  # 危险距离阈值（米），前车距离<10米触发SUMO接管
    "ngsim_danger_ttc_threshold": 2.0,  # 危险TTC阈值（秒），碰撞时间<3秒触发SUMO接管
}


# 状态机说明（简化版）:
# 1. 危险检测: 每帧检测 前车距离<10m 或 TTC<3s
# 2. 状态转换:
#    - 无危险 → 保持 REPLAY_MODE（按NGSIM轨迹回放）
#    - 检测到危险 → 永久转换为 FREE_MODE（SUMO完全接管）
# 3. 一旦触发接管，车辆永久由SUMO Krauss模型控制，不再尝试恢复NGSIM轨迹
# ====================================================


def env_creator(env_config):
    import macad_gym

    # 創建環境配置的深度副本以避免修改原始配置
    import copy
    modified_config = copy.deepcopy(env_actor_configs)

    # 確保env配置存在
    if "env" not in modified_config:
        modified_config["env"] = {}

    # ✅ 直接使用统一的 ENV_SETTINGS 配置
    modified_config["env"].update(ENV_SETTINGS)

    # Print configuration info for confirmation
    print("=" * 60)
    print("Environment Configuration (from ENV_SETTINGS):")
    print(f"  CARLA-SUMO Co-simulation: {modified_config['env'].get('enable_sumo_cosim', False)}")
    print(f"  SUMO GUI: {modified_config['env'].get('sumo_gui', False)}")
    print(f"  NGSIM Background: {modified_config['env'].get('enable_ngsim_background', False)}")
    print(f"  NGSIM Data File: {modified_config['env'].get('ngsim_csv_file', 'None')}")
    print(f"  SUMO-CARLA Sync: {modified_config['env'].get('sumo_to_carla_sync', False)}")
    print("-" * 60)
    print("NGSIM State Machine Configuration (简化版):")
    print(
        f"  危险距离阈值: {modified_config['env'].get('ngsim_danger_distance_threshold', 10.0)}m (前车距离<此值触发接管)")
    print(f"  危险TTC阈值: {modified_config['env'].get('ngsim_danger_ttc_threshold', 3.0)}s (碰撞时间<此值触发接管)")
    print(f"  ⚠️ 一旦触发危险，车辆将永久由SUMO控制")
    print("=" * 60)

    # 創建帶有修改配置的環境
    env = macad_gym.MultiCarlaEnv(modified_config)

    return env


register_env(env_name, lambda config: env_creator(config))

config = {
    # Model and preprocessor options.
    "model": {
        "custom_model": model_name,
        "custom_options": {
            "notes": {
                "args": vars(args)
            },
            "structural_ablation": STRUCTURAL_ABLATION,
        },
        # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
        # "custom_preprocessor": "sq_im_84",
        "dim": 88,
        "free_log_std": False,  # if args.discrete_actions else True,
        "grayscale": True,
    },

    "env_config": env_actor_configs
}


def get_shared_policy():
    """
    🎯 定义共享策略 (修正动作空间为 MultiDiscrete)
    所有adversary agents (car1, car2, car3) 共享这个策略
    """
    config = {
        # Model and preprocessor options.
        "model": {
            "custom_model": model_name,
            "custom_options": {
                "notes": {
                    "args": vars(args)
                },
                "structural_ablation": STRUCTURAL_ABLATION,
            },
            # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT specified
            # "custom_preprocessor": "sq_im_84",
            "dim": 84,
            "free_log_std": False,  # if args.discrete_actions else True,
            "grayscale": True,
            # conv_filters to be used with the custom CNN model.
            # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2], [16, [3, 3], 2]]
            # full: 832→MLP→LSTM；image_only/image_state/image_state_lidar: 无 LSTM
            "use_lstm": USE_LSTM_POLICY,
            "lstm_cell_size": 128,  # LSTM单元大小
            "lstm_use_prev_action_reward": True,  # LSTM输入包含上一步的动作和奖励
        },
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE(lambda) parameter.
        "lambda": 0.95,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.3,

        "rollout_fragment_length": 256,
        "sgd_minibatch_size": 256,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 4,
        # Stepsize of SGD.
        # tune.grid_search([0.01, 0.001, 0.0001])
        "lr": 0.0001,
        # Learning rate schedule.
        # "lr_schedule": [[0,0.0003],[200000,0.0002],[300000,0.0001]],
        # Share layers for value function. If you set this to True, it's important
        # to tune vf_loss_coeff.
        "vf_share_layers": True,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers: True.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.2,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": [[0, 0.2], [100000, 0.15], [200000, 0.1], [300000, 0.05]],
        # PPO clip parameter.
        "clip_param": 0.2,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of use_lstmgradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.03,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "complete_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Uses the sync samples optimizer instead of the multi-gpu one. This is
        # usually slower, but you might want to try it if you run into issues with
        # the default optimizer.
        "simple_optimizer": False,
        # Use PyTorch as framework?
        "use_pytorch": False,

        # Discount factor of the MDP.
        "gamma": 0.99,
        # Number of steps after which the episode is forced to terminate. Defaults
        # to `env.spec.max_episode_steps` (if present) for Gym envs.
        "horizon": 512,
        "sample_batch_size":
            256,
        "train_batch_size":
            1024,
        # "rollout_fragment_length": 128,
        "num_workers":
            1,
        # Number of environments to evaluate vectorwise per worker.
        "num_envs_per_worker":
            args.envs_per_worker,
        "num_gpus_per_worker":
            0.5,
        "env_config": env_actor_configs
    }

    return (PPOTFPolicy,
            Tuple([
                Box(0.0, 1.0, shape=(512,)),  # 图像特征
                Box(-20.0, 20.0, shape=(11,)),  # 物理信息
                # ✅ 修改为256维，匹配PointNet输出
                Box(0.0, 1.0, shape=(256,)),  # LiDAR特征 (PointNet输出)
            ]),
            # ✅ 修正: 必须匹配 multi_env.py 的定义
            MultiDiscrete([5, 5]),  # [纵向控制(5档), 横向控制(5档)]
            config)


def ego_policy():
    """
    🎯 Ego车辆策略（如果需要训练ego的话）
    这里使用相同的配置，但可以根据需要调整
    """
    return get_shared_policy()


# pprint (args.checkpoint_path)
# pprint(os.path.isfile(args.checkpoint_path))

checkpoint_path = "/home/lim/ray_results/Tjunction_ppo_LSTM64_advr4_Nov28/PPO_Town03I3C2_measure_continuous-v0_0_2024-11-29_00-01-12zhgm0en4/checkpoint_1021/checkpoint-1021"

# ray.init(num_cpus=10, num_gpus=1, memory=20 * 1024 * 1024 * 1024)
ray.init(num_cpus=10, num_gpus=1)
# print(ray.available_resources()['CPU'])

analysis = tune.run(
    "PPO",
    name="LaneChange_ppo_MultiDiscrete_SharedPolicy_{}_rainy".format(STRUCTURAL_ABLATION),
    local_dir="/home/simplexity/cybian/myown/12.11/ablation_experiment/inference_becausechangethemetrics/ray_results/",
    stop={
        "training_iteration": 100000,
        "timesteps_total": 20000000,
        "episodes_total": 5000,
    },
    config={
        "env": env_name,
        "log_level": "DEBUG",
        # "num_sgd_iter": 10,  # Enables Experience Replay
        "multiagent": {
            # 1. 只定义一个共享策略
            "policies": {
                "adversary_shared": get_shared_policy(),
            },

            # 2. 映射函数: 让 car1, car2, car3 都指向这个共享策略
            "policy_mapping_fn": tune.function(
                lambda agent_id: "adversary_shared"
                if agent_id in ["car1", "car2", "car3"]
                else "adversary_shared"  # 如果 ego 也在 keys 里但不需要训练，给个占位或者 None
            ),

            # 3. 指定训练这个共享策略（car1, car2, car3 都会贡献梯度）
            "policies_to_train": ["adversary_shared"],
        },
        "env_config": {
            # ✅ 直接使用顶部定义的 ENV_SETTINGS，确保配置统一
            **env_actor_configs,
            "env": {
                **env_actor_configs.get("env", {}),
                **ENV_SETTINGS  # 直接引用统一配置
            }
        },
        "num_workers": 1,
        "num_gpus": 0,  # args.num_gpus,
        "num_envs_per_worker": args.envs_per_worker,
        "sample_batch_size": args.sample_bs_per_worker,
        "train_batch_size": args.train_bs,
        # "horizon": 512,
        # "lr": 0.0005
        # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        # "horizon": 512, #yet to be fixed
        # "framework": "tf"
    },
    checkpoint_freq=20,
    checkpoint_at_end=True,
    # restore=checkpoint_path,
)

end_time = datetime.datetime.now()
print("endtime:", end_time)

time_delta = end_time - start_time
hours = time_delta.total_seconds() / 3600
print("training time:", hours)

ray.shutdown()