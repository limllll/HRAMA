"""Counterfactual collision labels and aggregate metrics.

The two denominators mirror the published AuthSim evaluation: meaningful
collisions divided by all tests, and meaningful collisions divided by all
collisions.  Here the collision-direction heuristic is replaced by a stricter
reference-driver counterfactual.  The names below are project definitions and
must be reported as such rather than attributed verbatim to AuthSim.
"""

NO_ADS_COLLISION = "no_ads_collision"
REFERENCE_AVOIDED = "rss_reference_avoided"
REFERENCE_COLLISION_INCONCLUSIVE = "rss_reference_collision_inconclusive"
REPLAY_FAILED = "rss_replay_failed"


def classify_counterfactual(ads_collision, reference_collision=None,
                            replay_succeeded=True,
                            reference_road_violation=False):
    if not bool(ads_collision):
        return NO_ADS_COLLISION
    if not replay_succeeded or reference_collision is None:
        return REPLAY_FAILED
    if bool(reference_collision) or bool(reference_road_violation):
        # Both drivers colliding does not prove that the original crash was
        # invalid or inevitable.  Likewise, collision avoidance by leaving the
        # drivable/lawful road area is not accepted as a safe counterfactual.
        return REFERENCE_COLLISION_INCONCLUSIVE
    return REFERENCE_AVOIDED


class CounterfactualMetrics:
    def __init__(self):
        self.total_tests = 0
        self.ads_collision_tests = 0
        self.reference_avoided_collisions = 0
        self.reference_collision_inconclusive = 0
        self.replay_failures = 0

    def update(self, label):
        self.total_tests += 1
        if label == NO_ADS_COLLISION:
            return
        self.ads_collision_tests += 1
        if label == REFERENCE_AVOIDED:
            self.reference_avoided_collisions += 1
        elif label == REFERENCE_COLLISION_INCONCLUSIVE:
            self.reference_collision_inconclusive += 1
        elif label == REPLAY_FAILED:
            self.replay_failures += 1
        else:
            raise ValueError("Unknown counterfactual label: {}".format(label))

    @staticmethod
    def _safe_ratio(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    def summary(self):
        data = {
            "total_tests": self.total_tests,
            "ads_collision_tests": self.ads_collision_tests,
            "reference_avoided_collisions": self.reference_avoided_collisions,
            "reference_collision_inconclusive": (
                self.reference_collision_inconclusive),
            "replay_failures": self.replay_failures,
        }
        assessed_collision_tests = (
            self.reference_avoided_collisions
            + self.reference_collision_inconclusive)
        data.update({
            "assessed_collision_tests": assessed_collision_tests,
            "collision_rate_all_tests": self._safe_ratio(
                self.ads_collision_tests, self.total_tests),
            # Analogous denominator to AuthSim's valid collisions/all tests.
            "reference_avoided_collision_frequency": self._safe_ratio(
                self.reference_avoided_collisions, self.total_tests),
            # Analogous denominator to AuthSim's valid collisions/all collisions.
            "reference_avoided_collision_ratio": self._safe_ratio(
                self.reference_avoided_collisions, self.ads_collision_tests),
            "reference_avoided_ratio_assessed_replays": self._safe_ratio(
                self.reference_avoided_collisions, assessed_collision_tests),
            "reference_collision_inconclusive_ratio": self._safe_ratio(
                self.reference_collision_inconclusive,
                self.ads_collision_tests),
            "counterfactual_replay_success_ratio": self._safe_ratio(
                assessed_collision_tests,
                self.ads_collision_tests),
        })
        return data
