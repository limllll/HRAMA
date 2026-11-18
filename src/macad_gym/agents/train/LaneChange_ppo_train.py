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
import copy
import csv
import glob
import json
import os
import pickle
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
from macad_gym.envs.straightroads.StraightRoads3NPC import StraightRoads3NPC
from macad_gym.agents.experiment_utils import file_sha256, seed_everything
import time
# import tensorflow as tf
# from tensorboardX import SummaryWriter

import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

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
    default=True,
    help="Disables comet logging. Used for local smoke tests")
parser.add_argument(
    "--num-workers",
    default=1,  # 2 #fix
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=0, type=int,
    help="Number of GPUs used by Ray/TensorFlow PPO. Default=0 for this TF runtime")
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
    default=2000000,
    type=int,
    help="Maximum number of environment steps. Default=2000000")
parser.add_argument(
    "--num-iters",
    default=10000,
    type=int,
    help="Maximum number of training iterations. Default=10000")
parser.add_argument(
    "--num-episodes",
    default=5000,
    type=int,
    help="Stop after this many completed episodes. Default=5000")
parser.add_argument(
    "--checkpoint-freq",
    default=5,
    type=int,
    help="Save one PPO checkpoint every N training iterations.")
resume_group = parser.add_mutually_exclusive_group()
resume_group.add_argument(
    "--restore",
    default=None,
    help="Restore a PPO checkpoint into a new Ray Tune trial.")
resume_group.add_argument(
    "--resume-experiment",
    action="store_true",
    help="Resume the existing Ray Tune experiment and its original trial directory.")
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
parser.add_argument(
    "--multimodal-encoder-checkpoint",
    default=os.path.join(
        PROJECT_ROOT,
        "artifacts/multimodal_v2_enhanced_decoder_balanced/best.pt"),
    help="Checkpoint produced by train_multimodal_encoders.py; empty keeps legacy encoders",
)
parser.add_argument(
    "--multimodal-encoder-device",
    default="cpu",
    help="PyTorch device for the custom image/LiDAR encoders (auto, cpu, cuda:0)",
)
parser.add_argument("--seed", default=2026, type=int,
                    help="Global seed for Python, NumPy, TF, PyTorch, RLlib and SUMO")
parser.add_argument(
    "--output-dir",
    default=os.path.join(PROJECT_ROOT, "artifacts/lane_change_ppo"),
    help="Ray Tune result root",
)
parser.add_argument("--experiment-name", default="finetuned_encoder_5000ep",
                    help="Optional experiment name; the seed is appended automatically")
# SUMO 和 GUI 已設為默認開啟，移除選項

args = parser.parse_args()
seed_everything(args.seed)

MULTIMODAL_ENCODER_CHECKPOINT = (
    args.multimodal_encoder_checkpoint
    or os.environ.get("HRAMA_MULTIMODAL_ENCODER_CHECKPOINT", "")
)
if MULTIMODAL_ENCODER_CHECKPOINT:
    MULTIMODAL_ENCODER_CHECKPOINT = os.path.abspath(
        os.path.expanduser(MULTIMODAL_ENCODER_CHECKPOINT))
    if not os.path.isfile(MULTIMODAL_ENCODER_CHECKPOINT):
        raise IOError("Multimodal encoder checkpoint not found: {}".format(
            MULTIMODAL_ENCODER_CHECKPOINT))

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
template_env = StraightRoads3NPC(env_overrides={
    "using_imitation_model": False,
    "encoder_collection_only": True,
})
env_actor_configs = copy.deepcopy(template_env.configs)
env_actor_configs["env"]["using_imitation_model"] = True
env_actor_configs["env"].pop("encoder_collection_only", None)
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
    # Reuse the existing TF image encoder and PointNet++ inference paths;
    # only replace their encoder weights with the domain-adapted checkpoint.
    "use_custom_multimodal_encoder": False,
    "finetuned_multimodal_encoder_checkpoint": MULTIMODAL_ENCODER_CHECKPOINT,
    "multimodal_encoder_device": args.multimodal_encoder_device,
    "seed": args.seed,
    "sumo_seed": args.seed,
}


# 状态机说明（简化版）:
# 1. 危险检测: 每帧检测 前车距离<10m 或 TTC<3s
# 2. 状态转换:
#    - 无危险 → 保持 REPLAY_MODE（按NGSIM轨迹回放）
#    - 检测到危险 → 永久转换为 FREE_MODE（SUMO完全接管）
# 3. 一旦触发接管，车辆永久由SUMO Krauss模型控制，不再尝试恢复NGSIM轨迹
# ====================================================


experiment_name = args.experiment_name or "LaneChange_PPO_{}_finetuned".format(
    STRUCTURAL_ABLATION)
experiment_name = "{}_seed{}".format(experiment_name, args.seed)
REWARD_CSV_PATH = os.path.join(
    PROJECT_ROOT, "LaneChange_PPO_Rewards_inference_full.csv")
REWARD_CHECKPOINT_SCAN_AFTER = 0.0 if args.resume_experiment else time.time()


def _checkpoint_completed_episodes(checkpoint_path):
    """Return episodes_total recorded beside a restored Tune checkpoint."""
    if not checkpoint_path:
        return None
    metadata_paths = []
    metadata_file = checkpoint_path + ".tune_metadata"
    if os.path.isfile(metadata_file):
        metadata_paths.append(metadata_file)
    elif os.path.isdir(checkpoint_path):
        for root, _, files in os.walk(checkpoint_path):
            metadata_paths.extend(
                os.path.join(root, name) for name in files
                if name.endswith(".tune_metadata"))
    metadata = []
    for path in metadata_paths:
        try:
            with open(path, "rb") as stream:
                value = pickle.load(stream)
            if value.get("episodes_total") is not None:
                metadata.append((int(value.get("episodes_total")),
                                 int(value.get("iteration", 0))))
        except Exception as exc:
            print("Could not read checkpoint metadata {}: {}".format(path, exc))
    return max(metadata)[0] if metadata else None


def _reward_log_range(path):
    """Return (first episode, last episode), or None for empty/invalid logs."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as stream:
            rows = [row for row in csv.reader(stream) if row]
        if not rows:
            return None
        episodes = [int(row[0]) for row in rows]
        if episodes != list(range(episodes[0], episodes[0] + len(episodes))):
            return None
        return episodes[0], episodes[-1]
    except (IOError, ValueError, IndexError):
        return None


if args.resume_experiment:
    experiment_dir = os.path.join(
        os.path.abspath(args.output_dir), experiment_name)
    state_files = glob.glob(os.path.join(
        experiment_dir, "experiment_state-*.json"))
    if not state_files:
        raise RuntimeError(
            "--resume-experiment requested, but no Tune experiment state "
            "exists under {}".format(experiment_dir))
    manifest_path = os.path.join(
        os.path.abspath(args.output_dir), experiment_name + "_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            "--resume-experiment requested, but the experiment manifest is "
            "missing: {}".format(manifest_path))
    with open(manifest_path, "r") as stream:
        previous_manifest = json.load(stream)
    previous_reward_path = previous_manifest.get("episode_reward_csv")
    if not previous_reward_path:
        raise RuntimeError(
            "The saved experiment manifest does not identify its reward CSV: "
            "{}".format(manifest_path))
    REWARD_CSV_PATH = previous_reward_path
    print("Resuming existing Tune experiment: {}".format(experiment_dir))
    print("Continuing reward log: {}".format(REWARD_CSV_PATH))
elif args.restore:
    restored_episodes = _checkpoint_completed_episodes(args.restore)
    if restored_episodes is None:
        raise RuntimeError(
            "Cannot safely select a reward log: restored checkpoint has no "
            "readable episodes_total metadata: {}".format(args.restore))
    reward_candidates = [REWARD_CSV_PATH] + sorted(glob.glob(
        REWARD_CSV_PATH + ".resume_from_ep*.csv"))
    matching_logs = []
    for candidate in reward_candidates:
        episode_range = _reward_log_range(candidate)
        if episode_range and episode_range[0] <= restored_episodes + 1:
            matching_logs.append((episode_range[0], candidate))
    if matching_logs:
        # Prefer the segment that starts latest while still covering this checkpoint.
        REWARD_CSV_PATH = max(matching_logs)[1]
    else:
        REWARD_CSV_PATH = "{}.resume_from_ep{}.csv".format(
            REWARD_CSV_PATH, restored_episodes + 1)
    print("Reward log selected for checkpoint episode {}: {}".format(
        restored_episodes, REWARD_CSV_PATH))


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
    modified_config["env"].update({
        "cummulated_reward": True,
        "cummulated_reward_filename": REWARD_CSV_PATH,
        "reward_resume_checkpoint": args.restore,
        "reward_checkpoint_dir": os.path.join(
            os.path.abspath(args.output_dir), experiment_name),
        "reward_checkpoint_scan_after": REWARD_CHECKPOINT_SCAN_AFTER,
    })

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
        , "seed": args.seed
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
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
if (not args.restore and not args.resume_experiment
        and os.path.isfile(REWARD_CSV_PATH)):
    archived_reward_path = "{}.previous_{}".format(
        REWARD_CSV_PATH, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.rename(REWARD_CSV_PATH, archived_reward_path)
    print("Archived previous reward log: {}".format(archived_reward_path))
run_manifest = {
    "seed": args.seed,
    "experiment_name": experiment_name,
    "structural_ablation": STRUCTURAL_ABLATION,
    "multimodal_encoder_checkpoint": MULTIMODAL_ENCODER_CHECKPOINT,
    "multimodal_encoder_sha256": file_sha256(MULTIMODAL_ENCODER_CHECKPOINT),
    "args": vars(args),
    "environment": ENV_SETTINGS,
    "episode_reward_csv": REWARD_CSV_PATH,
}
with open(os.path.join(args.output_dir, experiment_name + "_manifest.json"), "w") as stream:
    json.dump(run_manifest, stream, indent=2, sort_keys=True)

ray.init(num_cpus=10, num_gpus=args.num_gpus)
analysis = tune.run(
    "PPO",
    name=experiment_name,
    local_dir=args.output_dir,
    stop={
        "training_iteration": args.num_iters,
        "timesteps_total": args.num_steps,
        "episodes_total": args.num_episodes,
    },
    config={
        "env": env_name,
        "log_level": "DEBUG",
        # "num_sgd_iter": 10,  # Enables Experience Replay
        "multiagent": {
            # 对抗车辆共享一个可训练策略；ego 仅使用冻结的占位策略。
            # ego 的实际车辆控制仍由环境中的 BehaviorAgent 完成。
            "policies": {
                "adversary_shared": get_shared_policy(),
                "ego_policy": ego_policy(),
            },

            "policy_mapping_fn": tune.function(
                lambda agent_id: "ego_policy" if agent_id == "ego" else "adversary_shared"
            ),

            # 只有 car1/car2/car3 的共享策略贡献梯度。
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
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "num_envs_per_worker": args.envs_per_worker,
        "seed": args.seed,
        "sample_batch_size": args.sample_bs_per_worker,
        "train_batch_size": args.train_bs,
        # "horizon": 512,
        # "lr": 0.0005
        # "lr": tune.grid_search([0.01, 0.001, 0.0001]),
        # "horizon": 512, #yet to be fixed
        # "framework": "tf"
    },
    checkpoint_freq=args.checkpoint_freq,
    checkpoint_at_end=True,
    restore=args.restore,
    resume="LOCAL" if args.resume_experiment else False,
    max_failures=5,
)

end_time = datetime.datetime.now()
print("endtime:", end_time)

time_delta = end_time - start_time
hours = time_delta.total_seconds() / 3600
print("training time:", hours)

ray.shutdown()
