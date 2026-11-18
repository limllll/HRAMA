"""Simulator-independent tests for archive-guided diversity fine-tuning."""

import json
import os
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "src", "macad_gym")
macad_package = types.ModuleType("macad_gym")
macad_package.__path__ = [PACKAGE_ROOT]
sys.modules.setdefault("macad_gym", macad_package)

from macad_gym.rss_analysis.diversity_reward import (  # noqa: E402
    compute_diversity_reward,
)
from macad_gym.rss_analysis.metrics import (  # noqa: E402
    REFERENCE_AVOIDED,
    REFERENCE_COLLISION_INCONCLUSIVE,
)
from macad_gym.rss_analysis.novelty_archive import (  # noqa: E402
    NoveltyArchive,
)


def failure_summary(offset):
    return {
        "collision_step": 40 + offset,
        "danger_onset_step": 20 + offset,
        "min_ttc": 1.0 + 0.05 * offset,
        "min_rss_longitudinal_margin": -2.0 - 0.1 * offset,
        "min_rss_lateral_margin": -0.3 - 0.02 * offset,
        "relative_speed_at_collision": 4.0 + 0.2 * offset,
        "impact_angle": 10.0 + offset,
        "collision_relative_longitudinal": 2.0 + 0.1 * offset,
        "collision_relative_lateral": 0.5 + 0.05 * offset,
        "max_ego_deceleration": 3.0 + 0.1 * offset,
        "max_ego_lateral_speed": 0.4 + 0.02 * offset,
        "collision_with_controlled_npc": True,
    }


def make_seed_archive(path):
    with open(path, "w", encoding="utf-8") as handle:
        for episode, offset in enumerate((0, 1, 2, 3, 4), 1):
            handle.write(json.dumps({
                "episode": episode,
                "scenario_type": "lane_change",
                "counterfactual_label": REFERENCE_AVOIDED,
                "original_summary": failure_summary(offset),
            }))
            handle.write("\n")


class NoveltyArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.seed_path = os.path.join(
            self.temporary_directory.name, "seed.jsonl")
        make_seed_archive(self.seed_path)
        self.archive = NoveltyArchive.from_seed_jsonl(
            [self.seed_path], scenario_type="lane_change", k=3)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_distant_failure_is_more_novel_than_seed_failure(self):
        known = self.archive.query(failure_summary(0))
        distant = self.archive.query(failure_summary(40))
        self.assertGreater(
            distant["normalized_novelty"], known["normalized_novelty"])
        self.assertGreater(distant["raw_distance"], known["raw_distance"])

    def test_archive_updates_only_after_commit(self):
        initial_size = len(self.archive.prototypes)
        summary = failure_summary(40)
        query = self.archive.query(summary)
        self.archive.queue_candidate(summary, query_result=query)
        self.assertEqual(len(self.archive.prototypes), initial_size)
        result = self.archive.commit_pending()
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(len(self.archive.prototypes), initial_size + 1)

    def test_archive_round_trip_preserves_score(self):
        state_path = os.path.join(
            self.temporary_directory.name, "novelty_archive.json")
        before = self.archive.query(failure_summary(12))
        self.archive.save(state_path)
        restored = NoveltyArchive.load(state_path)
        after = restored.query(failure_summary(12))
        self.assertAlmostEqual(
            before["raw_distance"], after["raw_distance"], places=10)
        self.assertAlmostEqual(
            before["normalized_novelty"],
            after["normalized_novelty"], places=10)

    def test_missing_feature_uses_fitted_median(self):
        incomplete = failure_summary(10)
        incomplete["impact_angle"] = None
        result = self.archive.query(incomplete)
        self.assertGreaterEqual(result["normalized_novelty"], 0.0)
        self.assertLessEqual(result["normalized_novelty"], 1.0)


class DiversityRewardTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        seed_path = os.path.join(self.temporary_directory.name, "seed.jsonl")
        make_seed_archive(seed_path)
        self.archive = NoveltyArchive.from_seed_jsonl(
            [seed_path], scenario_type="lane_change", k=3)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_reference_avoided_failure_gets_team_bonus(self):
        record = {
            "counterfactual_label": REFERENCE_AVOIDED,
            "original_summary": failure_summary(40),
        }
        result = compute_diversity_reward(
            record, self.archive, lambda_div=10.0, adversary_count=3)
        self.assertTrue(result["eligible"])
        self.assertGreater(result["team_bonus"], 0.0)
        self.assertAlmostEqual(
            result["per_adversary_bonus"] * 3.0,
            result["team_bonus"], places=10)

    def test_inconclusive_collision_gets_no_bonus(self):
        record = {
            "counterfactual_label": REFERENCE_COLLISION_INCONCLUSIVE,
            "original_summary": failure_summary(40),
        }
        result = compute_diversity_reward(
            record, self.archive, lambda_div=10.0, adversary_count=3)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["team_bonus"], 0.0)

    def test_negative_reward_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_diversity_reward(
                {"counterfactual_label": REFERENCE_AVOIDED},
                self.archive,
                lambda_div=-1.0,
                adversary_count=3,
            )


if __name__ == "__main__":
    unittest.main()

