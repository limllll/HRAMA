"""Single-worker environment wrapper for RSS-gated diversity fine-tuning."""

from __future__ import division

import json
import os

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:  # Allows simulator-independent unit tests without Ray.
    MultiAgentEnv = object

from .diversity_reward import (
    compute_diversity_reward,
    public_reward_result,
)
from .evaluation import RSSCounterfactualEvaluation


class DiversityFineTuneEnv(MultiAgentEnv):
    """Add one RSS-gated novelty bonus to the original terminal transition.

    The wrapped environment still computes every original HRAMA step reward.
    When an episode terminates, this wrapper replays an AVUT-collision episode
    with the RSS-supervised reference driver, queries the frozen novelty
    archive, and broadcasts a normalized team bonus across adversarial NPCs.

    Archive candidates are queued during ``trainer.train()`` and committed by
    the driver afterwards through ``commit_archive_candidates``.  The archive
    therefore remains fixed within one PPO iteration.
    """

    def __init__(self, env, scenario_type, novelty_archive, lambda_div,
                 output_dir, rss_params=None, keep_full_trace=False,
                 ego_id="ego"):
        self.env = env
        self.scenario_type = str(scenario_type)
        self.novelty_archive = novelty_archive
        self.lambda_div = float(lambda_div)
        self.output_dir = os.path.abspath(output_dir)
        self.ego_id = str(ego_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.archive_state_path = os.path.join(
            self.output_dir, "novelty_archive.json")
        self.archive_versions_dir = os.path.join(
            self.output_dir, "archive_versions")
        os.makedirs(self.archive_versions_dir, exist_ok=True)
        self.episode_log_path = os.path.join(
            self.output_dir, "diversity_finetune_episodes.jsonl")
        self.rss_evaluation = RSSCounterfactualEvaluation(
            scenario_type=self.scenario_type,
            output_dir=os.path.join(self.output_dir, "rss_training_replays"),
            params=rss_params,
            keep_full_trace=keep_full_trace,
        )
        self._episode_active = False
        self._wrapper_episode = 0
        self._last_commit = None

        # Older RLlib releases inspect these attributes directly.
        for attribute in (
                "observation_space", "action_space", "reward_range",
                "metadata"):
            if hasattr(env, attribute):
                setattr(self, attribute, getattr(env, attribute))
        if hasattr(env, "configs"):
            self.configs = env.configs

    def __getattr__(self, name):
        env = object.__getattribute__(self, "env")
        return getattr(env, name)

    def reset(self):
        self.rss_evaluation.before_original_reset()
        observation = self.env.reset()
        self.rss_evaluation.start_original_episode(self.env)
        self._episode_active = True
        self._wrapper_episode += 1
        return observation

    def _ads_collision(self):
        collision = self.env._collisions.get(self.ego_id)
        if collision is None:
            return False
        # Match the collision definition in the current structural inference.
        return bool(
            collision.collision_vehicles + collision.collision_other > 0)

    def _adversary_ids(self, reward_dict):
        return [
            actor_id for actor_id in reward_dict
            if actor_id != self.ego_id
        ]

    def _write_episode_log(self, record, reward_result):
        replay = record.get("reference_replay") or {}
        payload = {
            "episode": self._wrapper_episode,
            "scenario_type": self.scenario_type,
            "counterfactual_label": record.get("counterfactual_label"),
            "replay_succeeded": replay.get("succeeded"),
            "reference_collision": replay.get("reference_collision"),
            "reference_road_violation": replay.get(
                "reference_road_violation"),
            "reward": public_reward_result(reward_result),
        }
        with open(self.episode_log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                payload, ensure_ascii=False, sort_keys=True,
                allow_nan=False))
            handle.write("\n")

    def step(self, action_dict):
        if not self._episode_active:
            raise RuntimeError("DiversityFineTuneEnv.step called before reset")
        observation, rewards, dones, infos = self.env.step(action_dict)
        self.rss_evaluation.record_original_step(self.env, action_dict)
        if not dones.get("__all__", False):
            return observation, rewards, dones, infos

        # MultiCarlaEnv returns its internal done dictionary.  RSS replay resets
        # the same base environment before this terminal transition is returned
        # to RLlib, so detach all terminal dictionaries first.
        terminal_rewards = dict(rewards)
        terminal_dones = dict(dones)
        terminal_infos = {
            actor_id: (dict(actor_info) if isinstance(actor_info, dict)
                       else actor_info)
            for actor_id, actor_info in infos.items()
        }
        ads_collision = self._ads_collision()
        record = self.rss_evaluation.finish_original_episode(
            env=self.env,
            episode=self._wrapper_episode,
            ads_collision=ads_collision,
        )
        adversary_ids = self._adversary_ids(terminal_rewards)
        if not adversary_ids:
            raise RuntimeError("No adversarial NPC rewards found at terminal step")
        reward_result = compute_diversity_reward(
            record=record,
            archive=self.novelty_archive,
            lambda_div=self.lambda_div,
            adversary_count=len(adversary_ids),
        )
        for actor_id in adversary_ids:
            original_reward = float(terminal_rewards[actor_id])
            terminal_rewards[actor_id] = (
                original_reward + reward_result["per_adversary_bonus"])
            actor_info = terminal_infos.setdefault(actor_id, {})
            actor_info["original_terminal_reward"] = original_reward
            actor_info["diversity_finetune"] = public_reward_result(
                reward_result)
        ego_info = terminal_infos.setdefault(self.ego_id, {})
        ego_info["diversity_finetune"] = public_reward_result(reward_result)

        if reward_result["eligible"]:
            self.novelty_archive.queue_candidate(
                summary=record["original_summary"],
                metadata={
                    "episode": self._wrapper_episode,
                    "scenario_type": self.scenario_type,
                    "counterfactual_label": record.get(
                        "counterfactual_label"),
                },
                query_result=reward_result["_query_result"],
            )
        self._write_episode_log(record, reward_result)
        self._episode_active = False
        return observation, terminal_rewards, terminal_dones, terminal_infos

    def commit_archive_candidates(self):
        """Commit candidates after one PPO iteration and persist the archive."""

        result = self.novelty_archive.commit_pending()
        self.novelty_archive.save(self.archive_state_path)
        versioned_archive_path = os.path.join(
            self.archive_versions_dir,
            "novelty_archive_v{:04d}.json".format(
                self.novelty_archive.version),
        )
        self.novelty_archive.save(versioned_archive_path)
        rss_summary_path, rss_summary = self.rss_evaluation.write_summary()
        result["archive_state_path"] = self.archive_state_path
        result["versioned_archive_path"] = versioned_archive_path
        result["rss_summary_path"] = rss_summary_path
        result["rss_summary"] = rss_summary
        self._last_commit = result
        return result

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None
