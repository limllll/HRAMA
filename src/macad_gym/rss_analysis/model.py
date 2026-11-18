"""Paper-faithful Responsibility-Sensitive Safety distance equations.

The formulas implemented here are Lemma 2 (same-direction longitudinal safe
distance), Lemma 3 (opposite-direction longitudinal safe distance), and Lemma
4 (lateral safe distance) of:

    S. Shalev-Shwartz, S. Shammah, and A. Shashua,
    "On a Formal Model of Safe and Scalable Self-driving Cars," 2017.
    https://arxiv.org/abs/1708.06374

RSS deliberately leaves the numerical parameters configurable.  The default
profile below is the complete vehicle-dynamics profile shipped with CARLA's
official RSS example for CARLA 0.9.13.  Keeping the equations and the numerical
profile separate prevents an implementation parameter from being presented as
a universal constant from the RSS paper.
"""

from __future__ import division

from typing import Dict


class RSSParameters:
    """RSS dynamics parameters, expressed as positive magnitudes in SI units."""

    _FIELD_NAMES = (
        "response_time",
        "max_longitudinal_accel",
        "max_longitudinal_brake",
        "min_longitudinal_brake",
        "min_correct_longitudinal_brake",
        "max_lateral_accel",
        "min_lateral_brake",
        "lateral_fluctuation_margin",
    )

    def __init__(self, response_time=0.5, max_longitudinal_accel=5.0,
                 max_longitudinal_brake=8.0, min_longitudinal_brake=4.0,
                 min_correct_longitudinal_brake=3.0,
                 max_lateral_accel=0.2, min_lateral_brake=0.8,
                 lateral_fluctuation_margin=0.1):
        self.response_time = float(response_time)
        self.max_longitudinal_accel = float(max_longitudinal_accel)
        self.max_longitudinal_brake = float(max_longitudinal_brake)
        self.min_longitudinal_brake = float(min_longitudinal_brake)
        self.min_correct_longitudinal_brake = float(
            min_correct_longitudinal_brake)
        self.max_lateral_accel = float(max_lateral_accel)
        self.min_lateral_brake = float(min_lateral_brake)
        self.lateral_fluctuation_margin = float(lateral_fluctuation_margin)
        positive = (
            "response_time",
            "max_longitudinal_accel",
            "max_longitudinal_brake",
            "min_longitudinal_brake",
            "min_correct_longitudinal_brake",
            "max_lateral_accel",
            "min_lateral_brake",
        )
        for field_name in positive:
            if getattr(self, field_name) <= 0.0:
                raise ValueError("{} must be positive".format(field_name))
        if self.lateral_fluctuation_margin < 0.0:
            raise ValueError("lateral_fluctuation_margin must be non-negative")

    def to_dict(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self._FIELD_NAMES}


def _positive_part(value):
    return max(float(value), 0.0)


def longitudinal_safe_distance_same_direction(
    rear_speed,
    front_speed,
    params=None,
):
    """Return RSS Lemma 2 safe bumper-to-bumper longitudinal distance.

    ``rear_speed`` and ``front_speed`` are non-negative speeds along the lane.
    Acceleration and braking arguments are taken from ``RSSParameters``.
    """

    params = params or RSSParameters()
    rear_speed = max(float(rear_speed), 0.0)
    front_speed = max(float(front_speed), 0.0)
    rho = params.response_time
    rear_after_response = rear_speed + rho * params.max_longitudinal_accel
    distance = (
        rear_speed * rho
        + 0.5 * params.max_longitudinal_accel * rho * rho
        + rear_after_response * rear_after_response
        / (2.0 * params.min_longitudinal_brake)
        - front_speed * front_speed / (2.0 * params.max_longitudinal_brake)
    )
    return _positive_part(distance)


def longitudinal_safe_distance_opposite_direction(
    correct_lane_speed,
    wrong_way_speed_magnitude,
    params=None,
):
    """Return RSS Lemma 3 safe distance for vehicles moving toward each other."""

    params = params or RSSParameters()
    v1 = max(float(correct_lane_speed), 0.0)
    v2 = max(float(wrong_way_speed_magnitude), 0.0)
    rho = params.response_time
    v1_rho = v1 + rho * params.max_longitudinal_accel
    v2_rho = v2 + rho * params.max_longitudinal_accel
    return (
        0.5 * (v1 + v1_rho) * rho
        + v1_rho * v1_rho
        / (2.0 * params.min_correct_longitudinal_brake)
        + 0.5 * (v2 + v2_rho) * rho
        + v2_rho * v2_rho / (2.0 * params.min_longitudinal_brake)
    )


def lateral_safe_distance(left_speed, right_speed, params=None):
    """Return RSS Lemma 4 lateral safe distance.

    Lateral coordinates increase from the left vehicle toward the right
    vehicle.  Therefore, positive ``left_speed`` and negative ``right_speed``
    describe vehicles moving toward each other, exactly as in the paper.
    """

    params = params or RSSParameters()
    left_speed = float(left_speed)
    right_speed = float(right_speed)
    rho = params.response_time
    accel = params.max_lateral_accel
    brake = params.min_lateral_brake
    left_after_response = left_speed + rho * accel
    right_after_response = right_speed - rho * accel
    closing_displacement = (
        0.5 * (left_speed + left_after_response) * rho
        + left_after_response * left_after_response / (2.0 * brake)
        - (
            0.5 * (right_speed + right_after_response) * rho
            - right_after_response * right_after_response / (2.0 * brake)
        )
    )
    return params.lateral_fluctuation_margin + _positive_part(closing_displacement)


class RSSPairEvaluation:
    """Pairwise RSS state in the ego vehicle's lane coordinate frame."""

    _FIELD_NAMES = (
        "other_id", "same_direction", "ego_is_rear", "other_on_right",
        "longitudinal_gap", "lateral_gap", "safe_longitudinal_distance",
        "safe_lateral_distance", "longitudinal_margin", "lateral_margin",
        "longitudinal_unsafe", "lateral_unsafe", "dangerous",
        "ego_longitudinal_speed", "other_longitudinal_speed",
        "ego_lateral_speed", "other_lateral_speed", "ttc",
    )

    def __init__(self, **values):
        missing = [name for name in self._FIELD_NAMES if name not in values]
        unknown = sorted(set(values) - set(self._FIELD_NAMES))
        if missing or unknown:
            raise TypeError(
                "Invalid RSSPairEvaluation fields; missing={}, unknown={}".format(
                    missing, unknown))
        for name in self._FIELD_NAMES:
            setattr(self, name, values[name])

    def to_dict(self):
        return {name: getattr(self, name) for name in self._FIELD_NAMES}


def evaluate_pair(
    other_id,
    longitudinal_gap,
    lateral_gap,
    ego_longitudinal_speed,
    other_longitudinal_speed,
    ego_lateral_speed,
    other_lateral_speed,
    ego_is_rear,
    other_on_right,
    same_direction=True,
    params=None,
):
    """Evaluate the paper's combined longitudinal/lateral danger condition."""

    params = params or RSSParameters()
    long_gap = max(float(longitudinal_gap), 0.0)
    lat_gap = max(float(lateral_gap), 0.0)
    ego_long = float(ego_longitudinal_speed)
    other_long = float(other_longitudinal_speed)
    ego_lat = float(ego_lateral_speed)
    other_lat = float(other_lateral_speed)

    if same_direction:
        if ego_is_rear:
            rear_speed, front_speed = ego_long, other_long
        else:
            rear_speed, front_speed = other_long, ego_long
        safe_long = longitudinal_safe_distance_same_direction(
            rear_speed, front_speed, params)
        closing_speed = rear_speed - front_speed
    else:
        safe_long = longitudinal_safe_distance_opposite_direction(
            abs(ego_long), abs(other_long), params)
        closing_speed = abs(ego_long) + abs(other_long)

    if other_on_right:
        left_speed, right_speed = ego_lat, other_lat
    else:
        left_speed, right_speed = other_lat, ego_lat
    safe_lat = lateral_safe_distance(left_speed, right_speed, params)

    long_margin = long_gap - safe_long
    lat_margin = lat_gap - safe_lat
    long_unsafe = long_margin < 0.0
    lat_unsafe = lat_margin < 0.0
    dangerous = long_unsafe and lat_unsafe
    ttc = float("inf")
    if closing_speed > 1e-6:
        ttc = long_gap / closing_speed

    return RSSPairEvaluation(
        other_id=str(other_id),
        same_direction=bool(same_direction),
        ego_is_rear=bool(ego_is_rear),
        other_on_right=bool(other_on_right),
        longitudinal_gap=long_gap,
        lateral_gap=lat_gap,
        safe_longitudinal_distance=safe_long,
        safe_lateral_distance=safe_lat,
        longitudinal_margin=long_margin,
        lateral_margin=lat_margin,
        longitudinal_unsafe=long_unsafe,
        lateral_unsafe=lat_unsafe,
        dangerous=dangerous,
        ego_longitudinal_speed=ego_long,
        other_longitudinal_speed=other_long,
        ego_lateral_speed=ego_lat,
        other_lateral_speed=other_lat,
        ttc=ttc,
    )
