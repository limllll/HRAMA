"""Common single-worker PPO fine-tuning entry for HRAMA failure diversity."""

from __future__ import print_function

import argparse
import copy
import datetime
import json
import os

import gym
import macad_gym  # noqa: F401
import ray
import ray.rllib.agents.ppo as ppo
import ray.tune as tune
from gym.spaces import Box, MultiDiscrete, Tuple
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune import register_env

from macad_gym.agents.train.models import register_carla_imitation_model
from macad_gym.envs.straightroads.FollowingRoads3NPC import (
    DEFAULT_NGSIM_CSV as DEFAULT_FOLLOWING_NGSIM,
    apply_following_scenario_config,
)
from macad_gym.rss_analysis.diversity_finetune_wrapper import (
    DiversityFineTuneEnv,
)
from macad_gym.rss_analysis.evaluation import load_rss_parameters
from macad_gym.rss_analysis.novelty_archive import NoveltyArchive


DEFAULT_LANE_CHANGE_NGSIM = "ngsim_vehicle_final_120.csv"


def _build_parser(default_scenario):
    parser = argparse.ArgumentParser(
        description=(
            "Restore a trained HRAMA checkpoint and fine-tune it with an "
            "RSS-gated archive novelty reward."))
    parser.add_argument(
        "--scenario",
        default=default_scenario,
        choices=["lane_change", "car_following"],
    )
    parser.add_argument(
        "--restore-checkpoint", required=True,
        help="Base HRAMA PPO checkpoint to restore before fine-tuning.")
    parser.add_argument(
        "--seed-archive", action="append", default=[],
        help=(
            "RSS JSONL archive used to initialize known failures. Repeat the "
            "option to provide multiple archives."),
    )
    parser.add_argument(
        "--archive-state", default="",
        help="Resume from a previously saved novelty_archive.json.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for fine-tuning logs, archive versions and checkpoints.")
    parser.add_argument(
        "--rss-params-file", default="",
        help="Optional RSS parameter JSON; defaults to the current CARLA profile.")
    parser.add_argument("--ads-name", default="behavior_agent")
    parser.add_argument(
        "--structural-ablation", default="full",
        choices=["image_only", "image_state", "image_state_lidar", "full"],
    )
    parser.add_argument("--lambda-div", type=float, default=10.0)
    parser.add_argument("--novelty-k", type=int, default=5)
    parser.add_argument("--archive-max-size", type=int, default=1000)
    parser.add_argument("--archive-add-quantile", type=float, default=25.0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.00002)
    parser.add_argument("--entropy-coeff", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=200000)
    parser.add_argument("--num-iters", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--envs-per-worker", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--num-cpus", type=int, default=10)
    parser.add_argument("--sample-bs-per-worker", type=int, default=256)
    parser.add_argument("--train-bs", type=int, default=1024)
    parser.add_argument("--sgd-minibatch-size", type=int, default=256)
    parser.add_argument("--num-sgd-iter", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--keep-rss-traces", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--ngsim-csv-file", default="",
        help="Override the scenario-specific NGSIM CSV.")
    return parser


def _validate_args(args):
    if args.num_workers != 1:
        raise ValueError(
            "The first archive-reward implementation requires --num-workers 1")
    if args.envs_per_worker != 1:
        raise ValueError(
            "The first archive-reward implementation requires "
            "--envs-per-worker 1")
    if args.lambda_div < 0.0:
        raise ValueError("--lambda-div must be non-negative")
    if not args.archive_state and not args.seed_archive:
        raise ValueError(
            "Provide --seed-archive for a new run or --archive-state to resume")
    for path in args.seed_archive:
        if not os.path.isfile(path):
            raise ValueError("Seed archive does not exist: {}".format(path))
    if args.archive_state and not os.path.isfile(args.archive_state):
        raise ValueError("Archive state does not exist: {}".format(
            args.archive_state))
    if not os.path.exists(args.restore_checkpoint):
        raise ValueError("Checkpoint does not exist: {}".format(
            args.restore_checkpoint))


def _controlled_ngsim_vehicle_ids(configs):
    vehicle_ids = []
    for actor_config in configs.get("actors", {}).values():
        vehicle_id = actor_config.get("ngsim_vehicle_id")
        if vehicle_id is not None:
            vehicle_ids.append(int(vehicle_id))
    return vehicle_ids


def _scenario_configuration(args):
    if args.scenario == "lane_change":
        env_name = "Town04C3-DiversityFineTune-v0"
        gym_env_name = "Town04C3-v0"
        scenario_type = "lane_change"
        ngsim_csv = args.ngsim_csv_file or DEFAULT_LANE_CHANGE_NGSIM
    else:
        env_name = "Town04FollowingC3-DiversityFineTune-v0"
        gym_env_name = "Town04FollowingC3-v0"
        scenario_type = "car_following"
        ngsim_csv = args.ngsim_csv_file or DEFAULT_FOLLOWING_NGSIM

    probe_env = gym.make(gym_env_name)
    actor_configs = copy.deepcopy(probe_env.configs)
    if hasattr(probe_env, "close"):
        probe_env.close()
    if scenario_type == "car_following":
        apply_following_scenario_config(
            actor_configs, ngsim_csv, verbose=True)
    excluded_ids = _controlled_ngsim_vehicle_ids(actor_configs)

    env_settings = {
        "enable_sumo_cosim": True,
        "sumo_gui": False,
        "verbose": bool(args.verbose),
        "enable_ngsim_background": True,
        "ngsim_csv_file": ngsim_csv,
        "sumo_to_carla_sync": True,
        "ngsim_danger_distance_threshold": 5.0,
        "ngsim_danger_ttc_threshold": 2.0,
        "cummulated_reward_filename": os.path.join(
            args.output_dir, "original_cumulative_rewards.csv"),
    }
    if excluded_ids:
        env_settings["ngsim_excluded_vehicle_ids"] = excluded_ids
    actor_configs.setdefault("env", {}).update(env_settings)
    return env_name, scenario_type, actor_configs, env_settings


def _load_archive(args, scenario_type):
    if args.archive_state:
        archive = NoveltyArchive.load(args.archive_state)
        if archive.scenario_type != scenario_type:
            raise ValueError(
                "Archive scenario {} does not match {}".format(
                    archive.scenario_type, scenario_type))
        return archive
    return NoveltyArchive.from_seed_jsonl(
        paths=args.seed_archive,
        scenario_type=scenario_type,
        ads_name=args.ads_name,
        max_steps=args.max_steps,
        k=args.novelty_k,
        max_size=args.archive_max_size,
        add_quantile=args.archive_add_quantile,
    )


def _policy_spec(args, actor_configs, model_name):
    use_lstm_policy = args.structural_ablation == "full"
    policy_config = {
        "model": {
            "custom_model": model_name,
            "custom_options": {
                "notes": {"args": vars(args)},
                "structural_ablation": args.structural_ablation,
            },
            "dim": 84,
            "free_log_std": False,
            "grayscale": True,
            "use_lstm": use_lstm_policy,
            "lstm_cell_size": 128,
            "lstm_use_prev_action_reward": True,
        },
        "use_critic": True,
        "use_gae": True,
        "lambda": 0.95,
        "kl_coeff": 0.3,
        "rollout_fragment_length": args.sample_bs_per_worker,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "shuffle_sequences": True,
        "num_sgd_iter": args.num_sgd_iter,
        "lr": args.learning_rate,
        "vf_share_layers": True,
        "vf_loss_coeff": 1.0,
        "entropy_coeff": args.entropy_coeff,
        "entropy_coeff_schedule": None,
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
        "sample_batch_size": args.sample_bs_per_worker,
        "train_batch_size": args.train_bs,
        "num_workers": args.num_workers,
        "num_envs_per_worker": args.envs_per_worker,
        "num_gpus_per_worker": 0.0,
        "env_config": actor_configs,
    }
    return (
        PPOTFPolicy,
        Tuple([
            Box(0.0, 1.0, shape=(512,)),
            Box(-20.0, 20.0, shape=(11,)),
            Box(0.0, 1.0, shape=(256,)),
        ]),
        MultiDiscrete([5, 5]),
        policy_config,
    )


def _commit_worker_environments(worker):
    return worker.foreach_env(
        lambda environment: environment.commit_archive_candidates())


def _flatten_commit_results(value):
    flattened = []
    if isinstance(value, dict):
        flattened.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            flattened.extend(_flatten_commit_results(item))
    return flattened


def _write_json_line(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")


def run(default_scenario):
    args = _build_parser(default_scenario).parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    args.restore_checkpoint = os.path.abspath(args.restore_checkpoint)
    args.seed_archive = [os.path.abspath(path) for path in args.seed_archive]
    if args.archive_state:
        args.archive_state = os.path.abspath(args.archive_state)
    if args.rss_params_file:
        args.rss_params_file = os.path.abspath(args.rss_params_file)
    os.makedirs(args.output_dir, exist_ok=True)
    _validate_args(args)

    start_time = datetime.datetime.now()
    env_name, scenario_type, actor_configs, env_settings = (
        _scenario_configuration(args))
    model_name = "carla_imitation_model"
    register_carla_imitation_model()
    rss_parameters = load_rss_parameters(args.rss_params_file)
    initial_archive = _load_archive(args, scenario_type)
    initial_archive_path = os.path.join(
        args.output_dir, "initial_novelty_archive.json")
    initial_archive.save(initial_archive_path)

    def env_creator(env_config):
        modified_config = copy.deepcopy(actor_configs)
        modified_config.setdefault("env", {}).update(env_settings)
        base_env = macad_gym.MultiCarlaEnv(modified_config)
        archive = NoveltyArchive.load(initial_archive_path)
        return DiversityFineTuneEnv(
            env=base_env,
            scenario_type=scenario_type,
            novelty_archive=archive,
            lambda_div=args.lambda_div,
            output_dir=args.output_dir,
            rss_params=rss_parameters,
            keep_full_trace=args.keep_rss_traces,
        )

    register_env(env_name, env_creator)
    shared_policy = _policy_spec(args, actor_configs, model_name)
    trainer_config = {
        "log_level": "INFO",
        "multiagent": {
            "policies": {
                "adversary_shared": shared_policy,
                "ego_policy": shared_policy,
            },
            "policy_mapping_fn": tune.function(
                lambda agent_id: (
                    "ego_policy" if agent_id == "ego"
                    else "adversary_shared")),
            "policies_to_train": ["adversary_shared"],
        },
        "env_config": actor_configs,
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "num_envs_per_worker": args.envs_per_worker,
        "sample_batch_size": args.sample_bs_per_worker,
        "train_batch_size": args.train_bs,
        "batch_mode": "complete_episodes",
    }

    experiment_config_path = os.path.join(
        args.output_dir, "experiment_config.json")
    with open(experiment_config_path, "w", encoding="utf-8") as handle:
        json.dump({
            "args": vars(args),
            "scenario_type": scenario_type,
            "env_name": env_name,
            "env_settings": env_settings,
            "rss_parameters": rss_parameters.to_dict(),
            "start_time": start_time.isoformat(),
        }, handle, ensure_ascii=False, indent=2, sort_keys=True)

    ray.init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    trainer = ppo.PPOTrainer(env=env_name, config=trainer_config)
    trainer.restore(args.restore_checkpoint)
    iteration_log = os.path.join(
        args.output_dir, "diversity_finetune_iterations.jsonl")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    added_timesteps = 0
    last_checkpoint = None
    try:
        for fine_tune_iteration in range(1, args.num_iters + 1):
            train_result = trainer.train()
            added_timesteps += int(train_result.get(
                "timesteps_this_iter", 0) or 0)
            commit_nested = trainer.workers.foreach_worker(
                _commit_worker_environments)
            commits = _flatten_commit_results(commit_nested)
            payload = {
                "fine_tune_iteration": fine_tune_iteration,
                "added_timesteps": added_timesteps,
                "timesteps_this_iter": train_result.get(
                    "timesteps_this_iter"),
                "episode_reward_mean": train_result.get(
                    "episode_reward_mean"),
                "episodes_this_iter": train_result.get(
                    "episodes_this_iter"),
                "commits": commits,
            }
            _write_json_line(iteration_log, payload)
            print(json.dumps(payload, ensure_ascii=False, default=str))

            if (args.checkpoint_freq > 0
                    and fine_tune_iteration % args.checkpoint_freq == 0):
                last_checkpoint = trainer.save(checkpoint_dir)
                print("Saved fine-tune checkpoint: {}".format(last_checkpoint))
            if added_timesteps >= args.num_steps:
                break
        last_checkpoint = trainer.save(checkpoint_dir)
        print("Final fine-tune checkpoint: {}".format(last_checkpoint))
    finally:
        trainer.stop()
        ray.shutdown()

    end_time = datetime.datetime.now()
    print("Fine-tuning finished in {:.2f} hours".format(
        (end_time - start_time).total_seconds() / 3600.0))
    print("Output directory: {}".format(args.output_dir))
    print("Last checkpoint: {}".format(last_checkpoint))


def main_lane_change():
    run("lane_change")


def main_car_following():
    run("car_following")
