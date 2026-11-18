#!/bin/env python
# -*- coding: utf-8 -*-
"""
PPO training entry for the Town04 following scenario.

This mirrors LaneChange_ppo_train.py: CARLA owns ego/car1/car2/car3 and their
sensors, while SUMO imports the NGSIM/Lankershim background traffic flow.
"""

import argparse
import copy
import csv
import datetime
import glob
import json
import os
import pickle
import shutil
import sys
import tempfile
import time

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
IMPORT_ROOTS = [SRC_ROOT, os.path.dirname(os.path.abspath(__file__)),
                os.path.join(SRC_ROOT, "macad_gym", "agents")]
for import_root in reversed(IMPORT_ROOTS):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
# Ray subprocesses inherit the environment, not the driver's sys.path changes.
inherited_paths = [path for path in os.environ.get("PYTHONPATH", "").split(os.pathsep) if path]
os.environ["PYTHONPATH"] = os.pathsep.join(IMPORT_ROOTS + [
    path for path in inherited_paths if path not in IMPORT_ROOTS])

import gym
import macad_gym  # noqa: F401
import ray
import ray.tune as tune
from gym.spaces import Box, MultiDiscrete, Tuple
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune import register_env

from macad_gym.envs.straightroads.FollowingRoads3NPC import (
    DEFAULT_NGSIM_CSV,
    FollowingRoads3NPC,
    apply_following_scenario_config,
)
from models import register_carla_imitation_model
from macad_gym.agents.experiment_utils import file_sha256, seed_everything
from resume_integrity import (
    RECOVERY_ENV_KEYS, VerifiedResumeExecutor, select_checkpoint, pin_trial_checkpoint,
)


START_TIME = datetime.datetime.now()
print("start-time", START_TIME)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Town04FollowingC3-v0", help="Name Gym env")
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--num-gpus", default=0, type=int)
parser.add_argument("--sample-bs-per-worker", default=256, type=int)
parser.add_argument("--train-bs", default=1024, type=int)
parser.add_argument("--envs-per-worker", default=1, type=int)
parser.add_argument("--num-steps", default=20000000, type=int)
parser.add_argument("--num-iters", default=100000, type=int)
parser.add_argument("--num-episodes", default=5000, type=int)
parser.add_argument("--checkpoint-freq", default=1, type=int,
                    help="Save after this many completed PPO iterations; does not change policy updates.")
parser.add_argument("--carla-reset-retries", default=2, type=int, choices=[1, 2],
                    help="Total reset attempts: at most one extra simulator rebuild in the same worker.")
parser.add_argument("--max-failures", default=2, type=int,
                    help="Bound automatic Ray recovery; simulator groups are cleaned before relaunch.")
resume_group = parser.add_mutually_exclusive_group()
resume_group.add_argument("--restore", default=None,
                         help="Restore a PPO checkpoint into a new Tune trial.")
resume_group.add_argument("--resume-experiment", action="store_true",
                         help="Resume the original Tune experiment and trial directory.")
parser.add_argument("--resume-checkpoint", default=None,
                    help="Optional exact checkpoint to pin for same-directory experiment resume.")
parser.add_argument("--seed", default=2026, type=int)
parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "artifacts/following_ppo"))
parser.add_argument("--experiment-name", default="finetuned_encoder_following_lane_flow_5000ep")
parser.add_argument("--following-layout", choices=["fixed", "ngsim"], default="fixed",
                    help="Fixed spaced layout (default), or original low-speed NGSIM initialization.")
parser.add_argument("--disable-comet", action="store_true", default=True,
                    help="Compatibility option; this entry does not use Comet.")
parser.add_argument("--num-framestack", type=int, default=4)
parser.add_argument(
    "--structural-ablation",
    default="full",
    choices=["image_only", "image_state", "image_state_lidar", "full"],
)
parser.add_argument(
    "--ngsim-csv-file",
    default=DEFAULT_NGSIM_CSV,
    help="NGSIM CSV used by the SUMO bridge; fixed layout shares lane-change traffic.",
)
parser.add_argument(
    "--multimodal-encoder-checkpoint",
    default=os.path.join(PROJECT_ROOT,
                         "artifacts/multimodal_v2_enhanced_decoder_balanced/best.pt"),
    help="Checkpoint produced by train_multimodal_encoders.py; empty keeps legacy encoders",
)
parser.add_argument(
    "--multimodal-encoder-device",
    default="cpu",
    help="Device used to load fine-tuned weights into the existing encoder paths.",
)
args = parser.parse_args()
seed_everything(args.seed)
args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
if min(args.num_episodes, args.num_steps, args.num_iters, args.checkpoint_freq) <= 0:
    parser.error("episode, step, iteration and checkpoint limits must be positive")
if args.max_failures < 0:
    parser.error("max-failures must be non-negative; unlimited automatic retries are disabled")
if args.resume_checkpoint and not args.resume_experiment:
    parser.error("--resume-checkpoint requires --resume-experiment")

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

experiment_name = "{}_seed{}".format(args.experiment_name, args.seed)
REWARD_CSV_PATH = os.path.join(PROJECT_ROOT, "Following_PPO_Rewards_inference_full.csv")
REWARD_CHECKPOINT_SCAN_AFTER = 0.0 if args.resume_experiment else time.time()


def _checkpoint_completed_episodes(checkpoint_path):
    """Read the committed episode boundary from Ray 0.8 checkpoint metadata."""
    if not checkpoint_path:
        return None
    metadata_paths = []
    metadata_file = checkpoint_path + ".tune_metadata"
    if os.path.isfile(metadata_file):
        metadata_paths.append(metadata_file)
    elif os.path.isdir(checkpoint_path):
        for root, _, files in os.walk(checkpoint_path):
            metadata_paths.extend(os.path.join(root, name) for name in files
                                  if name.endswith(".tune_metadata"))
    metadata = []
    for path in metadata_paths:
        try:
            with open(path, "rb") as stream:
                value = pickle.load(stream)
            if value.get("episodes_total") is not None:
                metadata.append((int(value["episodes_total"]), int(value.get("iteration", 0))))
        except Exception as exc:
            print("Could not read checkpoint metadata {}: {}".format(path, exc))
    return max(metadata)[0] if metadata else None


def _reward_log_range(path):
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


def _prepare_experiment_resume(experiment_dir, max_failures, checkpoint_freq, explicit_checkpoint=None):
    """Requeue a failed Ray 0.8 trial in place, preserving its checkpoint/logdir.

    Old Ray's LOCAL resume otherwise leaves ERROR trials terminal. A manual
    resume starts a fresh bounded recovery allowance, not a new PPO experiment.
    Ray restores checkpoint_freq from trial state, so update it here only on a
    manual resume, never while the existing training process is running.
    """
    if checkpoint_freq <= 0:
        raise ValueError("checkpoint_freq must be positive")
    paths = glob.glob(os.path.join(experiment_dir, "experiment_state-*.json"))
    path = max(paths, key=os.path.getmtime)
    with open(path) as stream:
        state = json.load(stream)
    changed = False
    selected_paths = []
    for trial in state["checkpoints"]:
        if trial["status"] == "TERMINATED":
            continue
        selected, metadata = select_checkpoint(
            trial["logdir"], explicit=explicit_checkpoint,
            lineage=None if explicit_checkpoint else trial.get("resume_lineage_id"))
        pin_trial_checkpoint(trial, selected, metadata)
        selected_paths.append(selected)
        changed = True
        if trial["status"] in ("ERROR", "RUNNING"):
            trial["status"] = "PENDING"
            changed = True
        if trial.get("num_failures") != 0 or trial.get("max_failures") != max_failures:
            trial["num_failures"] = 0
            trial["max_failures"] = max_failures
            changed = True
        if trial.get("checkpoint_freq") != checkpoint_freq:
            trial["checkpoint_freq"] = checkpoint_freq
            changed = True
    if len(selected_paths) != 1:
        raise RuntimeError("Expected exactly one unfinished following PPO trial")
    state["runner_data"]["_has_errored"] = False
    backup_dir = os.path.join(experiment_dir, "resume_state_backups")
    os.makedirs(backup_dir, exist_ok=True)
    backup = os.path.join(backup_dir, "{}_{}".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"), os.path.basename(path)))
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=".resume_state_", dir=experiment_dir)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("Prepared same-directory resume; recovery limit {}; checkpoint interval {} "
          "PPO iteration(s); state backup: {}".format(
              max_failures, checkpoint_freq, backup))
    print("Pinned actual PPO checkpoint: {}".format(selected_paths[0]))
    return selected_paths[0]


if args.resume_experiment:
    experiment_dir = os.path.join(args.output_dir, experiment_name)
    if not glob.glob(os.path.join(experiment_dir, "experiment_state-*.json")):
        raise RuntimeError("No Tune experiment state exists under {}".format(experiment_dir))
    manifest_path = os.path.join(args.output_dir, experiment_name + "_manifest.json")
    if not os.path.isfile(manifest_path):
        raise RuntimeError("Experiment manifest is missing: {}".format(manifest_path))
    with open(manifest_path, "r") as stream:
        previous_manifest = json.load(stream)
    previous_reward_path = previous_manifest.get("episode_reward_csv")
    if not previous_reward_path:
        raise RuntimeError("The experiment manifest does not identify its reward CSV")
    REWARD_CSV_PATH = previous_reward_path
    print("Resuming existing Tune experiment: {}".format(experiment_dir))
    print("Continuing reward log: {}".format(REWARD_CSV_PATH))
    _prepare_experiment_resume(experiment_dir, args.max_failures, args.checkpoint_freq,
                               args.resume_checkpoint)
elif args.restore:
    restored_episodes = _checkpoint_completed_episodes(args.restore)
    if restored_episodes is None:
        raise RuntimeError("Restored checkpoint has no readable episodes_total metadata: {}".format(
            args.restore))
    reward_candidates = [REWARD_CSV_PATH] + sorted(glob.glob(
        REWARD_CSV_PATH + ".resume_from_ep*.csv"))
    matching_logs = []
    for candidate in reward_candidates:
        episode_range = _reward_log_range(candidate)
        if episode_range and episode_range[0] <= restored_episodes + 1:
            matching_logs.append((episode_range[0], candidate))
    REWARD_CSV_PATH = (max(matching_logs)[1] if matching_logs else
                       "{}.resume_from_ep{}.csv".format(REWARD_CSV_PATH, restored_episodes + 1))
    print("Reward log selected for checkpoint episode {}: {}".format(
        restored_episodes, REWARD_CSV_PATH))

STRUCTURAL_ABLATION = args.structural_ablation
USE_LSTM_POLICY = STRUCTURAL_ABLATION == "full"
print("=" * 60)
print("following scenario env =", args.env)
print("structural_ablation    =", STRUCTURAL_ABLATION)
print("model.use_lstm         =", USE_LSTM_POLICY)
print("ngsim_csv_file         =", args.ngsim_csv_file)
print("=" * 60)

model_name = "carla_imitation_model"
register_carla_imitation_model()

env_name = args.env
template_env = FollowingRoads3NPC(
    env_overrides={
        "using_imitation_model": False,
        "encoder_collection_only": True,
        "following_layout": args.following_layout,
    },
    ngsim_csv_file=args.ngsim_csv_file,
)
env_actor_configs = copy.deepcopy(template_env.configs)
env_actor_configs["env"]["using_imitation_model"] = True
env_actor_configs["env"].pop("encoder_collection_only", None)
apply_following_scenario_config(
    env_actor_configs, args.ngsim_csv_file, verbose=True, layout=args.following_layout)


def get_controlled_ngsim_vehicle_ids(configs):
    vehicle_ids = []
    for actor_config in configs.get("actors", {}).values():
        vehicle_id = actor_config.get("ngsim_vehicle_id")
        if vehicle_id is not None:
            vehicle_ids.append(int(vehicle_id))
    return vehicle_ids


CONTROLLED_NGSIM_VEHICLE_IDS = get_controlled_ngsim_vehicle_ids(env_actor_configs)
print("controlled_ngsim_ids   =", CONTROLLED_NGSIM_VEHICLE_IDS)
EXCLUDED_NGSIM_VEHICLE_IDS = env_actor_configs["env"].get("ngsim_excluded_vehicle_ids", [])

ENV_SETTINGS = {
    "carla_reset_retries": args.carla_reset_retries,
    # All four actors use PPO/BehaviorAgent; background traffic uses SUMO.
    # An unused CARLA TM can retain a dead world's RPC thread after a restart.
    "enable_carla_traffic_manager": False,
    "enable_sumo_cosim": True,
    "sumo_gui": False,
    "verbose": True,
    "enable_ngsim_background": True,
    "ngsim_csv_file": args.ngsim_csv_file,
    "ngsim_excluded_vehicle_ids": EXCLUDED_NGSIM_VEHICLE_IDS,
    "following_layout": args.following_layout,
    "sumo_to_carla_sync": True,
    "ngsim_danger_distance_threshold": 5.0,
    "ngsim_danger_ttc_threshold": 2.0,
    # Same inference path as the validated lane-change run: only replace weights.
    "use_custom_multimodal_encoder": False,
    "finetuned_multimodal_encoder_checkpoint": MULTIMODAL_ENCODER_CHECKPOINT,
    "multimodal_encoder_device": args.multimodal_encoder_device,
    "seed": args.seed,
    "sumo_seed": args.seed,
    "cummulated_reward": True,
    "cummulated_reward_filename": REWARD_CSV_PATH,
    "reward_resume_checkpoint": args.restore,
    "reward_checkpoint_dir": os.path.join(args.output_dir, experiment_name),
    "reward_checkpoint_scan_after": REWARD_CHECKPOINT_SCAN_AFTER,
}


def env_creator(env_config):
    modified_config = copy.deepcopy(env_actor_configs)
    if "env" not in modified_config:
        modified_config["env"] = {}
    modified_config["env"].update(ENV_SETTINGS)
    # The executor supplies the ACTUAL restore boundary for each Ray restart.
    # Only recovery bookkeeping overrides the launch defaults, not experiment settings.
    incoming_recovery = env_config.get("env", {})
    for key in RECOVERY_ENV_KEYS:
        if key in incoming_recovery:
            modified_config["env"][key] = incoming_recovery[key]

    print("=" * 60)
    print("Following Environment Configuration")
    print("  env:", env_name)
    print("  CARLA-SUMO co-simulation:", modified_config["env"].get("enable_sumo_cosim", False))
    print("  NGSIM background:", modified_config["env"].get("enable_ngsim_background", False))
    print("  NGSIM CSV:", modified_config["env"].get("ngsim_csv_file"))
    print("  excluded NGSIM IDs:", modified_config["env"].get("ngsim_excluded_vehicle_ids"))
    print("=" * 60)

    return macad_gym.MultiCarlaEnv(modified_config)


register_env(env_name, lambda config: env_creator(config))


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
        "entropy_coeff_schedule": [[0, 0.2], [100000, 0.15], [200000, 0.1], [300000, 0.05]],
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "grad_clip": None,
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
        "seed": args.seed,
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


def ego_policy():
    """Frozen RLlib placeholder; ego is physically controlled by BehaviorAgent."""
    return get_shared_policy()


if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
if (not args.restore and not args.resume_experiment and os.path.isfile(REWARD_CSV_PATH)):
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
    "scenario": env_actor_configs["scenarios"],
    "controlled_ngsim_ids": CONTROLLED_NGSIM_VEHICLE_IDS,
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
        "multiagent": {
            "policies": {
                "adversary_shared": get_shared_policy(),
                "ego_policy": ego_policy(),
            },
            "policy_mapping_fn": tune.function(
                lambda agent_id: "ego_policy" if agent_id == "ego" else "adversary_shared"
            ),
            "policies_to_train": ["adversary_shared"],
        },
        "env_config": {
            **env_actor_configs,
            "env": {
                **env_actor_configs.get("env", {}),
                **ENV_SETTINGS,
            },
        },
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "num_envs_per_worker": args.envs_per_worker,
        "seed": args.seed,
        "sample_batch_size": args.sample_bs_per_worker,
        "train_batch_size": args.train_bs,
    },
    checkpoint_freq=args.checkpoint_freq,
    checkpoint_at_end=True,
    restore=args.restore,
    resume="LOCAL" if args.resume_experiment else False,
    max_failures=args.max_failures,
    trial_executor=VerifiedResumeExecutor(require_resume=bool(args.resume_experiment or args.restore)),
)

END_TIME = datetime.datetime.now()
print("endtime:", END_TIME)
print("training time:", (END_TIME - START_TIME).total_seconds() / 3600)

ray.shutdown()
