"""Pure-Python regression tests for the RSS equations and metrics."""

import math
import os
import sys
import types
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, "src", "macad_gym")
# The project package registers Gym environments at import time.  These unit
# tests exercise only the simulator-independent RSS modules, so provide a
# lightweight package shell and avoid requiring CARLA/Gym on a development Mac.
macad_package = types.ModuleType("macad_gym")
macad_package.__path__ = [PACKAGE_ROOT]
sys.modules.setdefault("macad_gym", macad_package)

from macad_gym.rss_analysis.metrics import (  # noqa: E402
    NO_ADS_COLLISION,
    REFERENCE_AVOIDED,
    REFERENCE_COLLISION_INCONCLUSIVE,
    REPLAY_FAILED,
    CounterfactualMetrics,
    classify_counterfactual,
)
from macad_gym.rss_analysis.model import (  # noqa: E402
    RSSParameters,
    evaluate_pair,
    lateral_safe_distance,
    longitudinal_safe_distance_same_direction,
)
from macad_gym.rss_analysis.recording import (  # noqa: E402
    compare_initial_scene_states,
)


class RSSModelTests(unittest.TestCase):
    def test_same_direction_distance_matches_lemma_two(self):
        result = longitudinal_safe_distance_same_direction(10.0, 8.0)
        self.assertAlmostEqual(result, 21.15625, places=7)

    def test_lateral_distance_matches_lemma_four(self):
        result = lateral_safe_distance(0.0, 0.0)
        self.assertAlmostEqual(result, 0.1625, places=7)

    def test_danger_requires_both_dimensions_to_be_unsafe(self):
        common = dict(
            other_id="npc1",
            ego_longitudinal_speed=10.0,
            other_longitudinal_speed=8.0,
            ego_lateral_speed=0.0,
            other_lateral_speed=0.0,
            ego_is_rear=True,
            other_on_right=True,
            same_direction=True,
        )
        dangerous = evaluate_pair(
            longitudinal_gap=1.0, lateral_gap=0.0, **common)
        only_longitudinal = evaluate_pair(
            longitudinal_gap=1.0, lateral_gap=5.0, **common)
        only_lateral = evaluate_pair(
            longitudinal_gap=100.0, lateral_gap=0.0, **common)
        self.assertTrue(dangerous.dangerous)
        self.assertFalse(only_longitudinal.dangerous)
        self.assertFalse(only_lateral.dangerous)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            RSSParameters(response_time=0.0)


class CounterfactualMetricTests(unittest.TestCase):
    def test_labels_do_not_call_a_reference_collision_invalid(self):
        self.assertEqual(classify_counterfactual(False), NO_ADS_COLLISION)
        self.assertEqual(classify_counterfactual(True, False), REFERENCE_AVOIDED)
        self.assertEqual(
            classify_counterfactual(True, True),
            REFERENCE_COLLISION_INCONCLUSIVE,
        )
        self.assertEqual(
            classify_counterfactual(
                True, False, reference_road_violation=True),
            REFERENCE_COLLISION_INCONCLUSIVE,
        )
        self.assertEqual(
            classify_counterfactual(True, None, replay_succeeded=False),
            REPLAY_FAILED,
        )

    def test_published_denominators_are_computed_separately(self):
        metrics = CounterfactualMetrics()
        for label in (
                NO_ADS_COLLISION,
                REFERENCE_AVOIDED,
                REFERENCE_COLLISION_INCONCLUSIVE,
                REPLAY_FAILED):
            metrics.update(label)
        summary = metrics.summary()
        self.assertAlmostEqual(summary["collision_rate_all_tests"], 0.75)
        self.assertAlmostEqual(
            summary["reference_avoided_collision_frequency"], 0.25)
        self.assertAlmostEqual(
            summary["reference_avoided_collision_ratio"], 1.0 / 3.0)
        self.assertAlmostEqual(
            summary["reference_avoided_ratio_assessed_replays"], 0.5)
        self.assertAlmostEqual(
            summary["counterfactual_replay_success_ratio"], 2.0 / 3.0)


class ReplayConsistencyTests(unittest.TestCase):
    def test_scene_match_tolerates_small_numeric_error(self):
        expected = {
            "controlled:ego": {
                "type_id": "vehicle.tesla.model3",
                "x": 1.0,
                "y": 2.0,
                "z": 0.1,
                "yaw": 359.5,
                "speed": 10.0,
            }
        }
        actual = {
            "controlled:ego": {
                "type_id": "vehicle.tesla.model3",
                "x": 1.1,
                "y": 2.0,
                "z": 0.1,
                "yaw": 0.5,
                "speed": 10.2,
            }
        }
        self.assertTrue(compare_initial_scene_states(expected, actual)["matches"])

    def test_scene_mismatch_is_reported(self):
        expected = {
            "controlled:ego": {
                "type_id": "vehicle.tesla.model3",
                "x": 1.0,
                "y": 2.0,
                "z": 0.1,
                "yaw": 0.0,
                "speed": 10.0,
            }
        }
        actual = dict(expected)
        actual["controlled:ego"] = dict(expected["controlled:ego"], x=3.0)
        result = compare_initial_scene_states(expected, actual)
        self.assertFalse(result["matches"])
        self.assertEqual(result["mismatches"][0]["reason"], "state_mismatch")


if __name__ == "__main__":
    unittest.main()
