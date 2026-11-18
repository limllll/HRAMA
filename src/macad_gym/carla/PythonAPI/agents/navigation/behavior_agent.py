# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from macad_gym.carla.PythonAPI.agents.navigation.basic_agent import BasicAgent
from macad_gym.carla.PythonAPI.agents.navigation.local_planner import RoadOption
from macad_gym.carla.PythonAPI.agents.navigation.behavior_types import Cautious, Aggressive, Normal

from macad_gym.carla.PythonAPI.agents.tools.misc import get_speed, positive, is_within_distance, compute_distance
class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    CONTROL_VERSION = 'shared_distance_pid_v2'

    def __init__(self, vehicle, behavior='normal', ignore_speed_limits=False):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
            :param ignore_speed_limits: bypass road speed caps for simulation
                experiments, without imposing the spawn/initial velocity.
        """

        super(BehaviorAgent, self).__init__(vehicle)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._ignore_speed_limits = bool(ignore_speed_limits)
        self.last_control_diagnostics = {}

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        # Keep the local planner from silently reapplying the road cap when
        # this experiment explicitly opts out of speed-limit following.
        if self._ignore_speed_limits:
            self._local_planner.follow_speed_limits(False)

    def _normal_target_speed(self, junction=False):
        """Behavior-selected target (km/h), unrelated to initial velocity."""
        if self._ignore_speed_limits:
            return float(self._behavior.max_speed)
        offset = 5 if junction else self._behavior.speed_lim_dist
        return max(0.0, min(self._behavior.max_speed, self._speed_limit - offset))

    def _obstacle_detection_distance(self, divisor):
        """Retain the official heuristic; use actual speed when ignoring caps."""
        reference_speed = self._speed if self._ignore_speed_limits else self._speed_limit
        return max(self._behavior.min_proximity_threshold, reference_speed / divisor)

    def _finish_control(self, control, reason, target_speed=None):
        """Publish the decision applied by the caller, not a new control action."""
        self.last_control_diagnostics.update({
            'reason': reason,
            'target_speed_kmh': target_speed,
            'throttle': float(control.throttle),
            'brake': float(control.brake),
            'steer': float(control.steer),
        })
        # The planner's empty-route stop must not look like a hazard brake.
        if (target_speed is not None and control.brake >= 0.999
                and control.throttle == 0.0 and self._local_planner.done()):
            self.last_control_diagnostics['reason'] = 'route_empty'
        return control

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        reference_speed = self._speed if self._ignore_speed_limits else self._speed_limit
        self._look_ahead_steps = max(0, int(reference_speed / 10))

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(
            vehicle_list, self._obstacle_detection_distance(2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(
                    vehicle_list, self._obstacle_detection_distance(2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(
                    vehicle_list, self._obstacle_detection_distance(2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, self._obstacle_detection_distance(2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, self._obstacle_detection_distance(2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, self._obstacle_detection_distance(3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(
                walker_list, self._obstacle_detection_distance(2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(
                walker_list, self._obstacle_detection_distance(2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(
                walker_list, self._obstacle_detection_distance(3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        self.last_control_diagnostics['ttc_s'] = float(ttc)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min(positive(vehicle_speed - self._behavior.speed_decrease),
                               self._normal_target_speed())
            reason = 'following_slowdown'

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min(max(self._min_speed, vehicle_speed),
                               self._normal_target_speed())
            reason = 'following_match'

        # Normal behavior.
        else:
            target_speed = self._normal_target_speed()
            reason = 'following_clear'

        self._local_planner.set_speed(target_speed)
        return self._finish_control(self._local_planner.run_step(debug=debug),
                                    reason, float(target_speed))

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        self.last_control_diagnostics = {
            'controller_version': self.CONTROL_VERSION,
            'speed_kmh': float(self._speed),
            'road_speed_limit_kmh': float(self._speed_limit),
            'ignore_speed_limits': self._ignore_speed_limits,
            'lead_actor_id': None,
            'lead_speed_kmh': None,
            'gap_m': None,
            'ttc_s': None,
        }
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self._finish_control(self.emergency_stop(), 'red_light', 0.0)

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                self.last_control_diagnostics.update({
                    'lead_actor_id': walker.id, 'gap_m': float(distance)})
                return self._finish_control(self.emergency_stop(), 'pedestrian_close', 0.0)

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            lead_speed = get_speed(vehicle)
            self.last_control_diagnostics.update({
                'lead_actor_id': vehicle.id,
                'lead_speed_kmh': float(lead_speed),
                'gap_m': float(distance),
                'ttc_s': float(distance / max(1, (self._speed - lead_speed) / 3.6)),
            })

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self._finish_control(self.emergency_stop(), 'vehicle_close', 0.0)
            else:
                return self.car_following_manager(vehicle, distance, debug=debug)

        # 3: Intersection behavior
        elif (self._incoming_waypoint is not None and self._incoming_waypoint.is_junction
              and self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = self._normal_target_speed(junction=True)
            self._local_planner.set_speed(target_speed)
            return self._finish_control(self._local_planner.run_step(debug=debug),
                                        'junction_turn', float(target_speed))

        # 4: Normal behavior
        else:
            target_speed = self._normal_target_speed()
            self._local_planner.set_speed(target_speed)
            return self._finish_control(self._local_planner.run_step(debug=debug),
                                        'normal', float(target_speed))

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
