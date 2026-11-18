# -*- coding: utf-8 -*-
"""Durable RS/OS output using the completed following PPO metric protocol.

This module records measurements only. It does not select actions or alter the
existing per-episode metric formulas.
"""
import copy
import csv
import json
import os

import numpy as np

from macad_gym.agents.experiment_utils import save_json


CSV_COLUMNS = [
    "record_type", "episode", "R_collision", "RC", "OL", "TTD", "TTFC",
    "Ego_Cols", "Total_Cols", "C_avut", "NPC_LDE", "BG_LDE",
    "BG_LDE_Available", "CoSim_Available", "BG_Sample_Count", "RSS_Label",
    "RSS_Reference_Collision", "RSS_Reference_Road_Violation",
    "RSS_Replay_Succeeded", "RSS_Avoided_Collision_Frequency",
    "RSS_Avoided_Collision_Ratio", "RSS_Avoided_Ratio_Assessed",
    "RSS_Replay_Success_Ratio",
    "CoSim_Info_Present", "SUMO_Bridge_Present", "CoSim_Error",
    "CoSim_First_Failure", "CoSim_Failed_Steps", "CoSim_Last_Status",
    "BG_Candidate_Frames", "BG_Valid_Frames", "BG_Pending_Frames",
    "BG_Metric_Coverage",
]
FRAME_KEYS = (
    "bg_candidate_frames", "bg_valid_frames", "bg_pending_frames",
    "bg_metric_coverage",
)


def cosim_snapshot(env):
    return {
        "bridge_present": getattr(env, "_sumo_bridge", None) is not None,
        "startup_available": bool(getattr(env, "_cosim_available", False)),
        "episode_available": bool(getattr(env, "_episode_cosim_available", False)),
        "error": getattr(env, "_cosim_last_error", None),
        "status": copy.deepcopy(getattr(env, "_cosim_last_status", None)),
    }


class FollowingBaselineResults:
    def __init__(self, output_dir, strategy, scenario="Following"):
        self.output_dir = output_dir
        self.csv_path = os.path.join(
            output_dir, "inference_{}_{}_Results.csv".format(scenario, strategy))
        self.diagnostic_path = os.path.join(output_dir, "cosim_diagnostics.jsonl")
        self.config_path = os.path.join(output_dir, "evaluation_config.json")
        self.closed = False
        self.completed_episodes = 0
        self.failed_episodes = 0
        self.failed_steps = 0
        self.initial_cosim = None
        self.first_failure = None
        self.episode_failed_steps = 0
        self.csv_handle = None
        self.diagnostic_handle = None
        # Refuse any prior output, including a partial run. No silent overwrite
        # or "resume" that would restart the random sequence at episode one.
        for path in (self.csv_path, self.diagnostic_path, self.config_path):
            if os.path.exists(path):
                raise IOError(
                    "Evaluation output already exists: {}. "
                    "Choose a new --output-dir; existing data is preserved.".format(path))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        self.csv_handle = open(self.csv_path, "x", newline="")
        try:
            self.diagnostic_handle = open(self.diagnostic_path, "x")
        except Exception:
            self.csv_handle.close()
            raise
        self.writer = csv.writer(self.csv_handle)
        self.writer.writerow(CSV_COLUMNS)
        self.csv_handle.flush()

    def write_config(self, payload):
        save_json(self.config_path, payload)

    def begin_episode(self, env):
        self.initial_cosim = cosim_snapshot(env)
        self.first_failure = (
            dict(self.initial_cosim, step=0)
            if not self.initial_cosim["episode_available"] else None)
        self.episode_failed_steps = 0

    def record_step(self, env, step):
        current = cosim_snapshot(env)
        status = current["status"] or {}
        if not current["bridge_present"] or not status.get("ok", False):
            self.episode_failed_steps += 1
            if self.first_failure is None:
                self.first_failure = dict(current, step=step)

    def write_episode(self, episode, metrics, info, env):
        if episode != self.completed_episodes + 1:
            raise ValueError("Episode output must be consecutive from one")
        ego_info = info.get("ego", {}) if isinstance(info, dict) else {}
        for key in FRAME_KEYS:
            metrics.setdefault(key, []).append(ego_info.get(key, 0))
        final = cosim_snapshot(env)
        available = bool(ego_info.get("cosim_available", False))
        info_present = "cosim_available" in ego_info
        error = ego_info.get("cosim_error") or final["error"] or ""
        self.failed_steps += self.episode_failed_steps
        self.failed_episodes += int(not available)
        diagnostic = {
            "episode": episode,
            "initial": self.initial_cosim,
            "first_failure": self.first_failure,
            "failed_steps": self.episode_failed_steps,
            "final": final,
            "ego_metric_info": {key: ego_info.get(key) for key in (
                "bg_lde_available", "cosim_available", "bg_sample_count",
                "cosim_error", "cosim_status", "bg_candidate_frames",
                "bg_valid_frames", "bg_pending_frames", "bg_metric_coverage")},
            "cosim_info_present": info_present,
        }
        self.diagnostic_handle.write(
            json.dumps(diagnostic, ensure_ascii=False, default=str) + "\n")
        self.diagnostic_handle.flush()
        ego_cols = metrics["global_ego_cols"][-1]
        total_cols = metrics["global_total_cols"][-1]
        row = [
            "EPISODE", episode, metrics["ego_collision_flag"][-1],
            metrics["ego_rc"][-1], metrics["ego_ol"][-1],
            metrics["ego_ttd"][-1], metrics["ego_ttfc"][-1],
            ego_cols, total_cols,
            (float(ego_cols) / total_cols if total_cols else 0.0),
            metrics["npc_lde"][-1], metrics["bg_lde"][-1],
            metrics["bg_lde_available"][-1], metrics["cosim_available"][-1],
            metrics["bg_sample_count"][-1],
        ]
        # RSS replay is not enabled by either baseline.
        row.extend([""] * 8)
        row.extend([
            int(info_present), int(final["bridge_present"]), error,
            json.dumps(self.first_failure, ensure_ascii=False, default=str),
            self.episode_failed_steps,
            json.dumps(final["status"], ensure_ascii=False, default=str),
        ])
        row.extend([metrics[key][-1] for key in FRAME_KEYS])
        self.writer.writerow(row)
        self.csv_handle.flush()
        self.completed_episodes = episode
        if not available:
            print("Episode {} co-simulation diagnostic: {}".format(
                episode, json.dumps(diagnostic, ensure_ascii=False, default=str)))

    def write_final(self, metrics, expected_episodes):
        if self.completed_episodes != expected_episodes:
            raise ValueError("Cannot label an incomplete evaluation FINAL")
        total_e = sum(metrics["global_ego_cols"])
        total_t = sum(metrics["global_total_cols"])
        valid_bg = [value for value in metrics["bg_lde"] if np.isfinite(value)]
        row = [
            "FINAL", expected_episodes, np.mean(metrics["ego_collision_flag"]),
            np.mean(metrics["ego_rc"]), np.mean(metrics["ego_ol"]),
            np.mean(metrics["ego_ttd"]), np.mean(metrics["ego_ttfc"]),
            total_e, total_t, (float(total_e) / total_t if total_t else 0.0),
            np.mean(metrics["npc_lde"]), (np.mean(valid_bg) if valid_bg else ""),
            np.mean(metrics["bg_lde_available"]),
            np.mean(metrics["cosim_available"]), sum(metrics["bg_sample_count"]),
        ]
        row.extend([""] * 8)
        row.extend([
            "", "", "{} unavailable episodes".format(self.failed_episodes),
            "", self.failed_steps, "",
        ])
        candidates = sum(metrics["bg_candidate_frames"])
        valid = sum(metrics["bg_valid_frames"])
        row.extend([
            candidates, valid, sum(metrics["bg_pending_frames"]),
            (float(valid) / candidates if candidates else 0.0),
        ])
        self.writer.writerow(row)
        self.csv_handle.flush()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                if self.csv_handle is not None:
                    self.csv_handle.close()
            finally:
                if self.diagnostic_handle is not None:
                    self.diagnostic_handle.close()
