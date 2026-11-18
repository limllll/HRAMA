#!/bin/env python
# -*- coding: utf-8 -*-
"""
LaneChange PPO Inference Script aligned with structural-ablation training.
"""

import argparse
import copy
import csv
import datetime
import json
import math
import os
import pickle
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
for import_root in reversed([os.path.join(PROJECT_ROOT, "src"), os.path.join(PROJECT_ROOT, "src/macad_gym/agents")]):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import gym
import macad_gym  # noqa: F401
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
import ray.tune as tune
from gym.spaces import Box, MultiDiscrete, Tuple
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune import register_env
from macad_gym.envs.straightroads.StraightRoads3NPC import StraightRoads3NPC
from macad_gym.agents.experiment_utils import ensure_dir, file_sha256, save_json, seed_everything
from macad_gym.ads.wor_evaluation import add_wor_arguments, configure_wor, WORAudit

# Use the same model definition as training to guarantee compatibility.
from macad_gym.agents.train.models import register_carla_imitation_model
from macad_gym.rss_analysis.evaluation import (
    RSSCounterfactualEvaluation,
    load_rss_parameters,
)


start_time = datetime.datetime.now()
print("start-time", start_time)

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Town04C3-v0", help="Name Gym env")
parser.add_argument("--episodes", type=int, default=500, help="Number of evaluation episodes")
parser.add_argument('--npc-count', type=int, choices=[2, 3], default=3,
                    help='2NPC removes target-lane front car2; keeps car1/car3 and the same PPO checkpoint')
parser.add_argument("--max-steps", type=int, default=80, help="TTD/TTFC fallback max steps")
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--num-gpus", default=0, type=int)
parser.add_argument("--seed", default=2026, type=int)
parser.add_argument("--output-dir", default="artifacts/lane_change_evaluation")
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
    "--multimodal-encoder-checkpoint",
    default="",
    help="Self-trained encoder checkpoint used during PPO training",
)
parser.add_argument(
    "--multimodal-encoder-device",
    default="auto",
    help="Device used when loading fine-tuned encoder weights (training used cpu)",
)
parser.add_argument(
    "--rss-counterfactual",
    action="store_true",
    help="Replay AVUT-collision episodes with the RSS-supervised reference driver.",
)
parser.add_argument(
    "--rss-output-dir",
    default="rss_counterfactual_results",
    help="Directory for RSS episode archives and aggregate metrics.",
)
parser.add_argument(
    "--rss-params-file",
    default="",
    help="Optional JSON override for the CARLA RSS parameter profile.",
)
parser.add_argument(
    "--rss-keep-traces",
    action="store_true",
    help="Store full per-step traces in the JSONL archive (large output).",
)
add_wor_arguments(parser, PROJECT_ROOT)
parser.add_argument('--resume', choices=['auto', 'never'], default='never',
                    help='Opt-in CSV-committed recovery for the calibrated WOR launcher')
args = parser.parse_args()
args._wor_actuator_gain = float(os.environ.get('HRAMA_WOR_ACTUATOR_GAIN', '1.25'))
seed_everything(args.seed)

if not args.checkpoint_path:
    raise ValueError("--checkpoint-path is required")
resolved_checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint_path))
if not os.path.isfile(resolved_checkpoint):
    raise IOError("PPO checkpoint not found: {}".format(resolved_checkpoint))


def checkpoint_policy_ids(path):
    """Read policy ids without restoring a Trainer (Ray 0.8 checkpoint format)."""
    try:
        with open(path, "rb") as stream:
            state = pickle.load(stream)
        worker = state.get("worker")
        if isinstance(worker, bytes):
            worker = pickle.loads(worker)
        return set((worker or {}).get("state", {}).keys())
    except Exception as exc:
        print("WARNING: cannot inspect checkpoint policies: {}".format(exc))
        return set()


CHECKPOINT_POLICY_IDS = checkpoint_policy_ids(resolved_checkpoint)
CHECKPOINT_HAS_EGO_POLICY = "ego_policy" in CHECKPOINT_POLICY_IDS
print("checkpoint policies = {}".format(sorted(CHECKPOINT_POLICY_IDS)))

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
output_dir = ensure_dir(os.path.join(args.output_dir, "seed{}".format(args.seed)))
resume_state = None
if args.resume == 'auto':
    if args.ego_ads != 'wor' or os.environ.get('HRAMA_WOR_GAIN_CHILD') != '1':
        parser.error('Use a supervised lane WOR launcher for resume')
    from evaluation_resume import prepare, restored_metrics
    resume_state = prepare(args, output_dir)
start_episode = len(resume_state['rows']) if resume_state else 0

STRUCTURAL_ABLATION = args.structural_ablation
USE_LSTM_POLICY = STRUCTURAL_ABLATION == "full"
print("=" * 60)
print("structural_ablation =", STRUCTURAL_ABLATION)
print("model.use_lstm      =", USE_LSTM_POLICY)
print("=" * 60)

# A checkpoint is intentionally supplied on the command line.  Keeping an old
# in-file path silently restored an incompatible one-policy experiment.
checkpoint_path = ""

model_name = "carla_imitation_model"
register_carla_imitation_model()
env_name = args.env
template_env = StraightRoads3NPC(env_overrides={
    "using_imitation_model": False,
    "encoder_collection_only": True,
})
env_actor_configs = copy.deepcopy(template_env.configs)
from lane_evaluation_layout import apply_lane_npc_count
removed_npc_ids = apply_lane_npc_count(env_actor_configs, args.npc_count)
print('Evaluation NPC count: {}; removed: {}; retained: {}'.format(
    args.npc_count, removed_npc_ids, sorted(env_actor_configs['actors'])))
env_actor_configs["env"]["using_imitation_model"] = True
env_actor_configs["env"].pop("encoder_collection_only", None)

ENV_SETTINGS = {
    "enable_sumo_cosim": True,
    "sumo_gui": False,
    "verbose": True,
    "enable_ngsim_background": True,
    "ngsim_csv_file": "ngsim_vehicle_final_120.csv",
    "sumo_to_carla_sync": True,
    "ngsim_danger_distance_threshold": 5.0,
    "ngsim_danger_ttc_threshold": 2.0,
    # Match PPO training: keep the legacy TF/PointNet++ inference architecture
    # and load only the fine-tuned image-encoder weights into that path.
    "use_custom_multimodal_encoder": False,
    "finetuned_multimodal_encoder_checkpoint": MULTIMODAL_ENCODER_CHECKPOINT,
    "multimodal_encoder_device": args.multimodal_encoder_device,
    "seed": args.seed,
    "sumo_seed": args.seed,
}

if "env" not in env_actor_configs:
    env_actor_configs["env"] = {}
env_actor_configs["env"].update(ENV_SETTINGS)
configure_wor(args, parser, env_actor_configs, ENV_SETTINGS)
env_actor_configs["env"].update(ENV_SETTINGS)
wor_audit = WORAudit(args, output_dir)


def env_creator(env_config):
    import copy
    import macad_gym

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
    """Frozen RLlib placeholder; the selected ADS physically controls ego."""
    return get_shared_policy()


ray.init(num_gpus=args.num_gpus)

policies = {"adversary_shared": get_shared_policy()}
if CHECKPOINT_HAS_EGO_POLICY:
    policies["ego_policy"] = ego_policy()

trainer_config = {
    "log_level": "DEBUG",
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": tune.function(
            lambda agent_id: (
                "ego_policy" if agent_id == "ego" and CHECKPOINT_HAS_EGO_POLICY
                else "adversary_shared")),
        "policies_to_train": ["adversary_shared"],
    },
    "env_config": {"env": ENV_SETTINGS, **env_actor_configs},
    "num_workers": args.num_workers,
    "num_gpus": args.num_gpus,
    "seed": args.seed,
}

trainer = ppo.PPOTrainer(env=env_name, config=trainer_config)
trainer.restore(checkpoint_path=resolved_checkpoint)

# Reuse RLlib's local evaluation environment.  The previous script referenced
# an undefined global `env` after restoring the trainer.
env = trainer.workers.local_worker().env
wor_audit.attach(env)

actor_ids = list(env_actor_configs["actors"].keys())
max_steps = args.max_steps
eps = args.episodes
rss_evaluation = None
if args.rss_counterfactual:
    rss_evaluation = RSSCounterfactualEvaluation(
        scenario_type="lane_change",
        output_dir=args.rss_output_dir,
        params=load_rss_parameters(args.rss_params_file),
        keep_full_trace=args.rss_keep_traces,
    )


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
    "rss_label": [],
    "rss_reference_collision": [],
    "rss_reference_road_violation": [],
    "rss_replay_succeeded": [],
    "bg_candidate_frames": [],
    "bg_valid_frames": [],
    "bg_pending_frames": [],
    "bg_metric_coverage": [],
}

csv_file = os.path.join(
    output_dir, "inference_LaneChange_{}_Results.csv".format(STRUCTURAL_ABLATION))
csv_columns = [
    "record_type", "episode", "R_collision", "RC", "OL", "TTD", "TTFC",
    "Ego_Cols", "Total_Cols", "C_avut", "NPC_LDE", "BG_LDE",
    "BG_LDE_Available", "CoSim_Available", "BG_Sample_Count", "RSS_Label",
    "RSS_Reference_Collision", "RSS_Reference_Road_Violation",
    "RSS_Replay_Succeeded", "RSS_Avoided_Collision_Frequency",
    "RSS_Avoided_Collision_Ratio", "RSS_Avoided_Ratio_Assessed",
    "RSS_Replay_Success_Ratio",
    "CoSim_Info_Present", "SUMO_Bridge_Present", "CoSim_Error",
    "CoSim_First_Failure", "CoSim_Failed_Steps", "CoSim_Last_Status",
    "BG_Candidate_Frames", "BG_Valid_Frames", "BG_Pending_Frames", "BG_Metric_Coverage",
]
if resume_state and resume_state['header'] and resume_state['header'] != csv_columns:
    raise ValueError('CSV schema changed; refusing to append incompatible results')
if resume_state:
    metrics.update(restored_metrics(resume_state['rows']))
csv_handle = open(csv_file, "a" if resume_state else "w", newline="")
csv_writer = csv.writer(csv_handle)
if not resume_state or not resume_state['header']:
    csv_writer.writerow(csv_columns)
csv_handle.flush()
diagnostic_path = os.path.join(output_dir, "cosim_diagnostics.jsonl")
diagnostic_handle = open(diagnostic_path, "a" if resume_state else "w")
diagnostic_failed_episodes = sum(1 for value in metrics['cosim_available'] if not value)
diagnostic_failed_steps = sum(int(row['CoSim_Failed_Steps']) for row in resume_state['rows']) if resume_state else 0


def cosim_snapshot():
    """Record actual environment status, including failures before the first step."""
    return {
        "bridge_present": getattr(env, "_sumo_bridge", None) is not None,
        "startup_available": bool(getattr(env, "_cosim_available", False)),
        "episode_available": bool(getattr(env, "_episode_cosim_available", False)),
        "error": getattr(env, "_cosim_last_error", None),
        "status": copy.deepcopy(getattr(env, "_cosim_last_status", None)),
    }

print('Evaluation progress: {}/{} committed episodes; next episode {}'.format(start_episode, eps, start_episode + 1))
for ep in range(start_episode, eps):
    if rss_evaluation is not None:
        rss_evaluation.before_original_reset()
    obs = env.reset()
    initial_cosim = cosim_snapshot()
    first_cosim_failure = (
        dict(initial_cosim, step=0) if not initial_cosim["episode_available"] else None)
    failed_cosim_steps = 0
    if rss_evaluation is not None:
        rss_evaluation.start_original_episode(env)
    total_reward_dict = {aid: 0 for aid in actor_ids}
    prev_action = {aid: np.zeros(2) for aid in actor_ids}
    prev_reward = {aid: 0.0 for aid in actor_ids}
    rnn_state = {
        aid: trainer.get_policy("adversary_shared").get_initial_state()
        for aid in actor_ids if aid != "ego"
    }
    done = {"__all__": False}
    step = 0
    start = time.time()

    while not done["__all__"]:
        step += 1
        action_dict = {}
        for aid in actor_ids:
            if aid == "ego":
                # The selected ADS controls ego; this is only an RLlib placeholder.
                action_dict[aid] = np.array([2, 2], dtype=np.int32)
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

        wor_audit.before_step(env, ep + 1, step)
        prev_action = action_dict
        obs, reward, done, info = env.step(action_dict)
        wor_audit.after_step(info)
        current_cosim = cosim_snapshot()
        current_status = current_cosim["status"] or {}
        if not current_cosim["bridge_present"] or not current_status.get("ok", False):
            failed_cosim_steps += 1
            if first_cosim_failure is None:
                first_cosim_failure = dict(current_cosim, step=step)
        if rss_evaluation is not None:
            rss_evaluation.record_original_step(env, action_dict)
        for aid in actor_ids:
            total_reward_dict[aid] += reward[aid]
            prev_reward[aid] = reward[aid]

    wor_audit.finish_episode(step)
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
        print(f"⚠️ Episode {ep}: BG_LDE不可用，写入NaN并继续统计其他指标")
    metrics["bg_lde"].append(ep_bg_lde)
    metrics["bg_lde_available"].append(int(bg_lde_available))
    metrics["cosim_available"].append(int(cosim_available))
    metrics["bg_sample_count"].append(bg_sample_count)
    for key in ("bg_candidate_frames", "bg_valid_frames", "bg_pending_frames", "bg_metric_coverage"):
        metrics[key].append(ego_info.get(key, 0))
    final_cosim = cosim_snapshot()
    cosim_info_present = "cosim_available" in ego_info
    cosim_error = ego_info.get("cosim_error") or final_cosim["error"] or ""
    diagnostic_failed_steps += failed_cosim_steps
    diagnostic_failed_episodes += int(not cosim_available)
    diagnostic_record = {
        "episode": ep + 1,
        "initial": initial_cosim,
        "first_failure": first_cosim_failure,
        "failed_steps": failed_cosim_steps,
        "final": final_cosim,
        "ego_metric_info": {key: ego_info.get(key) for key in (
            "bg_lde_available", "cosim_available", "bg_sample_count",
            "cosim_error", "cosim_status", "bg_candidate_frames", "bg_valid_frames",
            "bg_pending_frames", "bg_metric_coverage")},
        "cosim_info_present": cosim_info_present,
    }
    diagnostic_handle.write(json.dumps(diagnostic_record, ensure_ascii=False, default=str) + "\n")
    diagnostic_handle.flush()
    os.fsync(diagnostic_handle.fileno())
    if not cosim_available:
        print("Episode {} co-simulation diagnostic: {}".format(
            ep + 1, json.dumps(diagnostic_record, ensure_ascii=False, default=str)))

    rss_record = None
    if rss_evaluation is not None:
        rss_record = rss_evaluation.finish_original_episode(
            env=env,
            episode=ep + 1,
            ads_collision=bool(is_ego_hit),
        )
        replay = rss_record.get("reference_replay") or {}
        metrics["rss_label"].append(rss_record["counterfactual_label"])
        metrics["rss_reference_collision"].append(
            replay.get("reference_collision"))
        metrics["rss_reference_road_violation"].append(
            replay.get("reference_road_violation"))
        metrics["rss_replay_succeeded"].append(replay.get("succeeded"))
    else:
        metrics["rss_label"].append("")
        metrics["rss_reference_collision"].append("")
        metrics["rss_reference_road_violation"].append("")
        metrics["rss_replay_succeeded"].append("")

    rss_label = rss_record["counterfactual_label"] if rss_record else "disabled"
    print(f"Episode {ep} | FPS: {step / (time.time() - start):.2f} | Ego Hit: {is_ego_hit} | BG_LDE: {ep_bg_lde:.4f} | RSS: {rss_label}")
    episode_ego_cols = metrics["global_ego_cols"][-1]
    episode_total_cols = metrics["global_total_cols"][-1]
    episode_collision_ratio = (
        float(episode_ego_cols) / episode_total_cols if episode_total_cols else 0.0)
    csv_writer.writerow([
        "EPISODE", ep + 1, is_ego_hit, rc_norm, metrics["ego_ol"][-1],
        metrics["ego_ttd"][-1], metrics["ego_ttfc"][-1], episode_ego_cols,
        episode_total_cols, episode_collision_ratio, metrics["npc_lde"][-1],
        ep_bg_lde, bg_lde_available, cosim_available, bg_sample_count,
        metrics["rss_label"][-1], metrics["rss_reference_collision"][-1],
        metrics["rss_reference_road_violation"][-1],
        metrics["rss_replay_succeeded"][-1], "", "", "", "",
        int(cosim_info_present), int(final_cosim["bridge_present"]), cosim_error,
        json.dumps(first_cosim_failure, ensure_ascii=False, default=str),
        failed_cosim_steps, json.dumps(final_cosim["status"], ensure_ascii=False, default=str),
        metrics["bg_candidate_frames"][-1], metrics["bg_valid_frames"][-1],
        metrics["bg_pending_frames"][-1], metrics["bg_metric_coverage"][-1],
    ])
    # Persist each completed episode immediately so interruption does not lose
    # the results already collected.
    csv_handle.flush()
    os.fsync(csv_handle.fileno())

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
valid_bg_lde = [value for value in metrics["bg_lde"] if np.isfinite(value)]
if valid_bg_lde:
    print(f"8. BG_LDE (NGSIM Background Vehicle Trajectory Error): {np.mean(valid_bg_lde):.4f} "
          f"({len(valid_bg_lde)}/{eps} valid episodes)")
else:
    print("8. BG_LDE (NGSIM Background Vehicle Trajectory Error): N/A (0 valid episodes)")
rss_summary = None
if rss_evaluation is not None:
    rss_summary_path, rss_summary = rss_evaluation.write_summary()
    print("9. Reference-avoided collision frequency: {:.2%}".format(
        rss_summary["reference_avoided_collision_frequency"]))
    print("10. Reference-avoided collision ratio: {:.2%}".format(
        rss_summary["reference_avoided_collision_ratio"]))
    print("11. Reference-avoided ratio among assessed replays: {:.2%}".format(
        rss_summary["reference_avoided_ratio_assessed_replays"]))
    print("12. Counterfactual replay success ratio: {:.2%}".format(
        rss_summary["counterfactual_replay_success_ratio"]))
    print("RSS archive: {}".format(rss_evaluation.archive.path))
    print("RSS metrics: {}".format(rss_summary_path))

total_e = sum(metrics["global_ego_cols"])
total_t = sum(metrics["global_total_cols"])
final_row = [
    "FINAL", eps, np.mean(metrics["ego_collision_flag"]),
    np.mean(metrics["ego_rc"]), np.mean(metrics["ego_ol"]),
    np.mean(metrics["ego_ttd"]), np.mean(metrics["ego_ttfc"]),
    total_e, total_t, (float(total_e) / total_t if total_t else 0.0),
    np.mean(metrics["npc_lde"]),
    (np.mean(valid_bg_lde) if valid_bg_lde else ""),
    np.mean(metrics["bg_lde_available"]),
    np.mean(metrics["cosim_available"]), sum(metrics["bg_sample_count"]),
    "", "", "", "",
]
if rss_summary is None:
    final_row.extend(["", "", "", ""])
else:
    final_row.extend([
        rss_summary["reference_avoided_collision_frequency"],
        rss_summary["reference_avoided_collision_ratio"],
        rss_summary["reference_avoided_ratio_assessed_replays"],
        rss_summary["counterfactual_replay_success_ratio"],
    ])
final_row.extend([
    "", "", "{} unavailable episodes".format(diagnostic_failed_episodes),
    "", diagnostic_failed_steps, "",
])
candidate_frames = sum(metrics["bg_candidate_frames"])
valid_frames = sum(metrics["bg_valid_frames"])
final_row.extend([
    candidate_frames, valid_frames, sum(metrics["bg_pending_frames"]),
    (float(valid_frames) / candidate_frames if candidate_frames else 0.0),
])
csv_writer.writerow(final_row)
csv_handle.flush()
csv_handle.close()
diagnostic_handle.close()
wor_audit.finish()
print("=" * 44)
print(f"CSV saved to: {csv_file}")
print("Co-simulation diagnostics saved to: {}".format(diagnostic_path))

save_json(os.path.join(output_dir, "evaluation_config.json"), {
    "seed": args.seed,
    "episodes": args.episodes,
    "ego_ads": args.ego_ads,
    "wor_actuator_gain": args._wor_actuator_gain if args.ego_ads == 'wor' else None,
    "npc_count": args.npc_count,
    "removed_npc_ids": removed_npc_ids,
    "retained_actor_ids": actor_ids,
    "scenario": env_actor_configs['scenarios'],
    "actor_initial_speeds": {key: value.get('initial_speed')
                             for key, value in env_actor_configs['actors'].items()},
    "npc_count_protocol": "same frozen PPO checkpoint; remove target-lane front car2 only for 2NPC",
    "wor_fallback_allowed": False if args.ego_ads == "wor" else None,
    "max_steps": args.max_steps,
    "structural_ablation": STRUCTURAL_ABLATION,
    "ppo_checkpoint": resolved_checkpoint,
    "multimodal_encoder_checkpoint": MULTIMODAL_ENCODER_CHECKPOINT,
    "multimodal_encoder_sha256": file_sha256(MULTIMODAL_ENCODER_CHECKPOINT),
    "environment": ENV_SETTINGS,
})

# Drain/stop/destroy sensors while CARLA is alive, as in following evaluation.
# Otherwise interpreter shutdown races callbacks and can abort in libcarla.
trainer.stop()
ray.shutdown()
