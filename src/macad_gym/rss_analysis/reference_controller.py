"""RSS-supervised reference controller used for counterfactual replay."""

from __future__ import division

from .carla_adapter import evaluate_carla_pairs
from .model import RSSParameters


class _DangerOnset:
    def __init__(self):
        self.longitudinal_step = None
        self.lateral_step = None


class RSSReferenceController:
    """Apply RSS basic proper-response restrictions to a nominal CARLA agent.

    The RSS paper defines safe distances and acceleration constraints, not a
    route-following controller.  This implementation therefore uses the
    project's BehaviorAgent as the nominal route controller and supervises it
    with RSS Definition 10.  A counterfactual is marked attributable only when
    this constrained reference actually avoids the original collision; a
    reference collision is left "not avoided", never declared mathematically
    unavoidable.
    """

    def __init__(self, params=None, ego_id="ego", max_reference_brake=1.0,
                 lateral_steer_limit=0.30):
        self.params = params or RSSParameters()
        self.ego_id = ego_id
        self.max_reference_brake = float(max_reference_brake)
        self.lateral_steer_limit = float(lateral_steer_limit)
        self.reset()

    def reset(self):
        self._onsets = {}
        self.last_evaluations = []
        self.last_restrictions = {
            "dangerous": False,
            "longitudinal_brake": False,
            "lateral_response": False,
            "active_other_ids": [],
        }

    def _update_onsets(self, evaluation, step_index):
        onset = self._onsets.setdefault(evaluation.other_id, _DangerOnset())
        if evaluation.longitudinal_unsafe:
            if onset.longitudinal_step is None:
                onset.longitudinal_step = step_index
        else:
            onset.longitudinal_step = None
        if evaluation.lateral_unsafe:
            if onset.lateral_step is None:
                onset.lateral_step = step_index
        else:
            onset.lateral_step = None
        return onset

    def compute_control(self, env, step_index):
        """Return a CARLA VehicleControl for the current reference step."""

        ego_agent = env._behavior_agents[self.ego_id]
        nominal = ego_agent.run_step()
        nominal.manual_gear_shift = False
        evaluations = evaluate_carla_pairs(env, self.ego_id, self.params)
        self.last_evaluations = evaluations
        brake_required = False
        lateral_directions = []
        active_ids = []
        for evaluation in evaluations:
            onset = self._update_onsets(evaluation, step_index)
            if not evaluation.dangerous:
                continue
            active_ids.append(evaluation.other_id)

            # RSS Definition 10 activates the dimension whose danger threshold
            # occurred last.  Equal thresholds activate both restrictions.
            # We realize the proper response immediately.  This is conservative
            # but RSS-compliant: rho is the latest permitted response time, not
            # a mandatory delay before braking or lateral correction.
            if onset.longitudinal_step >= onset.lateral_step:
                if evaluation.ego_is_rear or not evaluation.same_direction:
                    brake_required = True
            if onset.lateral_step >= onset.longitudinal_step:
                # Positive CARLA steer is right.  Move away from the threatening
                # vehicle while staying within the project's normal steer range.
                lateral_directions.append(-1.0 if evaluation.other_on_right else 1.0)

        if brake_required:
            nominal.throttle = 0.0
            nominal.brake = max(float(nominal.brake), self.max_reference_brake)

        lateral_response = bool(lateral_directions)
        if lateral_response:
            direction_sum = sum(lateral_directions)
            nominal.throttle = 0.0
            nominal.brake = max(float(nominal.brake), 0.2)
            if abs(direction_sum) < 1e-9:
                nominal.steer = 0.0
                brake_required = True
                nominal.brake = max(float(nominal.brake), self.max_reference_brake)
            else:
                nominal.steer = (
                    self.lateral_steer_limit if direction_sum > 0.0
                    else -self.lateral_steer_limit
                )

        self.last_restrictions = {
            "dangerous": bool(active_ids),
            "longitudinal_brake": brake_required,
            "lateral_response": lateral_response,
            "active_other_ids": active_ids,
        }
        return nominal
