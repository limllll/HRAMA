"""CARLA adapter for the simulator-independent RSS equations."""

from __future__ import division

import math

from .model import RSSParameters, evaluate_pair


def _dot2(vector, axis):
    return float(vector.x) * axis[0] + float(vector.y) * axis[1]


def _projected_half_extent(actor, axis):
    """Project an oriented CARLA bounding box onto a 2-D axis."""

    transform = actor.get_transform()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    extent = actor.bounding_box.extent
    return (
        abs(_dot2(forward, axis)) * float(extent.x)
        + abs(_dot2(right, axis)) * float(extent.y)
    )


def evaluate_carla_pairs(env, ego_id="ego", params=None):
    """Evaluate the ego vehicle against every live CARLA vehicle.

    This includes policy-controlled actors and CARLA mirrors of SUMO background
    traffic.  Coordinates are projected into the ego's current lane frame, as
    required by RSS Section 3.2.
    """

    params = params or RSSParameters()
    ego = env._actors[ego_id]
    ego_location = ego.get_location()
    waypoint = env.world.get_map().get_waypoint(ego_location, project_to_road=True)
    lane_forward = waypoint.transform.get_forward_vector()
    longitudinal_axis = (float(lane_forward.x), float(lane_forward.y))
    norm = math.hypot(longitudinal_axis[0], longitudinal_axis[1])
    if norm <= 1e-9:
        longitudinal_axis = (1.0, 0.0)
    else:
        longitudinal_axis = (
            longitudinal_axis[0] / norm,
            longitudinal_axis[1] / norm,
        )
    lateral_axis = (-longitudinal_axis[1], longitudinal_axis[0])

    ego_velocity = ego.get_velocity()
    ego_long_speed = _dot2(ego_velocity, longitudinal_axis)
    ego_lat_speed = _dot2(ego_velocity, lateral_axis)
    ego_forward = ego.get_transform().get_forward_vector()
    ego_half_long = _projected_half_extent(ego, longitudinal_axis)
    ego_half_lat = _projected_half_extent(ego, lateral_axis)

    evaluations = []
    for other in env.world.get_actors().filter("vehicle.*"):
        if other.id == ego.id or not other.is_alive:
            continue
        other_location = other.get_location()
        dx = float(other_location.x - ego_location.x)
        dy = float(other_location.y - ego_location.y)
        relative_long = dx * longitudinal_axis[0] + dy * longitudinal_axis[1]
        relative_lat = dx * lateral_axis[0] + dy * lateral_axis[1]
        other_velocity = other.get_velocity()
        other_long_speed = _dot2(other_velocity, longitudinal_axis)
        other_lat_speed = _dot2(other_velocity, lateral_axis)
        other_forward = other.get_transform().get_forward_vector()
        same_direction = (
            float(ego_forward.x) * float(other_forward.x)
            + float(ego_forward.y) * float(other_forward.y)
        ) >= 0.0

        long_gap = (
            abs(relative_long)
            - ego_half_long
            - _projected_half_extent(other, longitudinal_axis)
        )
        lat_gap = (
            abs(relative_lat)
            - ego_half_lat
            - _projected_half_extent(other, lateral_axis)
        )
        actor_name = env._actor_id_in_carla.get(other.id, "vehicle_{}".format(other.id))
        evaluations.append(
            evaluate_pair(
                other_id=actor_name,
                longitudinal_gap=long_gap,
                lateral_gap=lat_gap,
                ego_longitudinal_speed=ego_long_speed,
                other_longitudinal_speed=other_long_speed,
                ego_lateral_speed=ego_lat_speed,
                other_lateral_speed=other_lat_speed,
                ego_is_rear=relative_long >= 0.0,
                other_on_right=relative_lat >= 0.0,
                same_direction=same_direction,
                params=params,
            )
        )
    return evaluations
