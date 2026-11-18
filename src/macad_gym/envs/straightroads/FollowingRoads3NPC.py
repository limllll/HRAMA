import copy
import csv
import os
import xml.etree.ElementTree as ET

from macad_gym.carla.multi_env import MultiCarlaEnv
from macad_gym.carla.scenarios import Scenarios


DEFAULT_NGSIM_CSV = Scenarios.C3_TOWN04_FOLLOWING["ngsim_csv_file"]
DEFAULT_FOLLOWING_LAYOUT = "fixed"
# Same actor speeds as the completed lane-change experiments. Only the
# handwritten routes differ; retain the original shared ADS controller.
FIXED_FOLLOWING_INITIAL_SPEEDS = {"ego": 14.5, "car1": 14.5, "car2": 12.5, "car3": 12.5}
DEFAULT_ACTOR_NGSIM_IDS = {
    "ego": 242,
    "car1": 215,
    "car2": 239,
    "car3": 219,
}
SUMO_NGSIM_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "carla", "sumo_NGSIM")
)
TOWN04_NET_FILE = os.path.join(SUMO_NGSIM_DIR, "carla_town04_official.net.xml")
FEET_TO_METERS = 0.3048
CARLA_VEHICLE_EXTENT_X = 2.3
CARLA_Z = 0.8


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_ngsim_rows(csv_file):
    rows = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {}
            for key, value in raw_row.items():
                row[key] = value
            for key in ("Vehicle_ID", "Frame_ID", "Local_X", "Local_Y", "v_Vel", "Lane_ID"):
                number = _to_float(row.get(key))
                if number is None:
                    row = None
                    break
                row[key] = number
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError("No valid NGSIM rows found in {}".format(csv_file))
    return rows


def _get_net_offset(net_file):
    tree = ET.parse(net_file)
    location = tree.getroot().find("location")
    if location is None:
        return 0.0, 0.0
    raw_offset = location.get("netOffset", "0.0,0.0")
    parts = raw_offset.split(",")
    return float(parts[0]), float(parts[1])


def _resolve_ngsim_csv(csv_file):
    csv_file = csv_file or DEFAULT_NGSIM_CSV
    if not os.path.isabs(csv_file):
        csv_file = os.path.join(SUMO_NGSIM_DIR, csv_file)
    return csv_file


def _lane_range_centers(rows):
    values_by_lane = {}
    for row in rows:
        lane_id = int(row["Lane_ID"])
        values_by_lane.setdefault(lane_id, []).append(row["Local_X"] * FEET_TO_METERS)

    centers = {}
    for lane_id, values in values_by_lane.items():
        centers[lane_id] = {
            "min": min(values),
            "max": max(values),
            "center": (min(values) + max(values)) * 0.5,
        }
    return centers


def _map_ngsim_row_to_carla(row, rows, lane_ranges, net_offset):
    local_y_values = [item["Local_Y"] * FEET_TO_METERS for item in rows]
    ngsim_y_min = min(local_y_values)

    carla_params = {
        "vehicle_half_width": 0.9,
        "safety_margin": 1.0,
        "road_boundaries_edge40": {"min_y": 388.4, "max_y": 398.9},
        "road_boundaries_edge39": {"min_y": 387.9, "max_y": 398.4},
        "lane_centers_edge40": {1: 397.11, 2: 393.61, 3: 390.11},
        "lane_centers_edge39": {1: 396.66, 2: 393.16, 3: 389.66},
        "edge40_x_start": 145.4,
        "edge40_x_end": 382.2,
        "edge39_x_start": 435.4,
    }

    local_y_m = row["Local_Y"] * FEET_TO_METERS
    local_x_m = row["Local_X"] * FEET_TO_METERS
    lane_id = int(row["Lane_ID"])

    sumo_x = carla_params["edge40_x_start"] + (local_y_m - ngsim_y_min)
    if sumo_x <= carla_params["edge40_x_end"]:
        lane_centers = carla_params["lane_centers_edge40"]
        road_boundaries = carla_params["road_boundaries_edge40"]
    elif sumo_x >= carla_params["edge39_x_start"]:
        lane_centers = carla_params["lane_centers_edge39"]
        road_boundaries = carla_params["road_boundaries_edge39"]
    else:
        lane_centers = carla_params["lane_centers_edge40"]
        road_boundaries = carla_params["road_boundaries_edge40"]

    if lane_id not in lane_centers:
        lane_id = 2
    target_sumo_y = lane_centers[lane_id]

    vehicle_half_width = carla_params["vehicle_half_width"]
    safety_margin = carla_params["safety_margin"]
    road_min_y = road_boundaries["min_y"]
    road_max_y = road_boundaries["max_y"]
    lane_safe_offsets = {
        1: {
            "min": -1.5,
            "max": min(1.5, (road_max_y - safety_margin - vehicle_half_width) - lane_centers[1]),
        },
        2: {"min": -1.5, "max": 1.5},
        3: {
            "min": max(-1.5, (road_min_y + safety_margin + vehicle_half_width) - lane_centers[3]),
            "max": 1.5,
        },
    }
    if lane_id == 1:
        lane_safe_offsets[1]["max"] = min(0.6, lane_safe_offsets[1]["max"])
    elif lane_id == 3:
        lane_safe_offsets[3]["min"] = max(-0.6, lane_safe_offsets[3]["min"])

    safe_min = lane_safe_offsets.get(lane_id, {"min": -0.5})["min"]
    safe_max = lane_safe_offsets.get(lane_id, {"max": 0.5})["max"]
    lane_range = lane_ranges.get(lane_id)
    lateral_offset = 0.0
    if lane_range and abs(lane_range["max"] - lane_range["min"]) > 1e-6:
        x_deviation = local_x_m - lane_range["center"]
        x_normalized = x_deviation / (lane_range["max"] - lane_range["min"]) * 2.0
        max_safe_offset = min(abs(safe_min), abs(safe_max))
        lateral_offset = x_normalized * max_safe_offset
        lateral_offset = max(safe_min, min(safe_max, lateral_offset))

    sumo_y = target_sumo_y + lateral_offset
    vehicle_south_edge = sumo_y - vehicle_half_width
    vehicle_north_edge = sumo_y + vehicle_half_width
    road_boundary_min = road_min_y + safety_margin
    road_boundary_max = road_max_y - safety_margin

    if vehicle_south_edge < road_boundary_min:
        sumo_y = road_boundary_min + vehicle_half_width
    elif vehicle_north_edge > road_boundary_max:
        sumo_y = road_boundary_max - vehicle_half_width

    offset_x, offset_y = net_offset
    carla_x = sumo_x - offset_x - CARLA_VEHICLE_EXTENT_X
    carla_y = offset_y - sumo_y
    return [round(carla_x, 2), round(carla_y, 2), CARLA_Z]


def build_ngsim_following_scenario(actor_configs, csv_file=None, net_file=None):
    csv_file = _resolve_ngsim_csv(csv_file)
    net_file = net_file or TOWN04_NET_FILE

    rows = _load_ngsim_rows(csv_file)
    lane_ranges = _lane_range_centers(rows)
    net_offset = _get_net_offset(net_file)

    rows_by_vehicle = {}
    for row in rows:
        vehicle_id = int(row["Vehicle_ID"])
        rows_by_vehicle.setdefault(vehicle_id, []).append(row)

    scenario_actors = {}
    for actor_id, actor_config in actor_configs.items():
        vehicle_id = actor_config.get("ngsim_vehicle_id")
        if vehicle_id is None:
            continue
        vehicle_id = int(vehicle_id)
        vehicle_rows = sorted(rows_by_vehicle.get(vehicle_id, []), key=lambda item: item["Frame_ID"])
        if not vehicle_rows:
            raise ValueError("Vehicle_ID {} not found in {}".format(vehicle_id, csv_file))

        start_row = vehicle_rows[0]
        end_row = vehicle_rows[-1]
        scenario_actors[actor_id] = {
            "start": _map_ngsim_row_to_carla(start_row, rows, lane_ranges, net_offset),
            "end": _map_ngsim_row_to_carla(end_row, rows, lane_ranges, net_offset),
        }
        actor_config["initial_speed"] = start_row["v_Vel"] * FEET_TO_METERS
        actor_config["ngsim_start_frame"] = int(start_row["Frame_ID"])
        actor_config["ngsim_end_frame"] = int(end_row["Frame_ID"])

    if not scenario_actors:
        raise ValueError("No actor has ngsim_vehicle_id; cannot build dynamic following scenario")

    return {
        "map": "Town04",
        "actors": scenario_actors,
        "weather_distribution": [0],
        "max_steps": 80,
    }


def _print_dynamic_following_config(configs, csv_file):
    print("=" * 60)
    print("Dynamic NGSIM following scenario")
    print("  csv:", os.path.basename(csv_file))
    print("  excluded NGSIM IDs:", configs.get("env", {}).get("ngsim_excluded_vehicle_ids", []))
    for actor_id in sorted(configs.get("actors", {}).keys()):
        actor_config = configs["actors"][actor_id]
        vehicle_id = actor_config.get("ngsim_vehicle_id")
        if vehicle_id is None:
            continue
        scenario_actor = configs.get("scenarios", {}).get("actors", {}).get(actor_id, {})
        initial_speed = actor_config.get("initial_speed", 0.0)
        print(
            "  {actor}: NGSIM {veh}, frames {start_frame}->{end_frame}, "
            "start {start}, end {end}, speed {speed:.3f} m/s".format(
                actor=actor_id,
                veh=vehicle_id,
                start_frame=actor_config.get("ngsim_start_frame"),
                end_frame=actor_config.get("ngsim_end_frame"),
                start=scenario_actor.get("start"),
                end=scenario_actor.get("end"),
                speed=initial_speed,
            )
        )
    print("=" * 60)


def apply_dynamic_ngsim_following_config(configs, csv_file=None, verbose=False):
    # Explicit legacy mode: reconstruct the original four NGSIM trajectories.
    for actor_id, vehicle_id in DEFAULT_ACTOR_NGSIM_IDS.items():
        if actor_id in configs.get("actors", {}):
            configs["actors"][actor_id].setdefault("ngsim_vehicle_id", vehicle_id)
    resolved_csv_file = _resolve_ngsim_csv(csv_file)
    actor_configs = configs.get("actors", {})
    scenario = build_ngsim_following_scenario(actor_configs, csv_file=resolved_csv_file)
    configs["scenarios"] = scenario
    configs.setdefault("env", {})["ngsim_csv_file"] = os.path.basename(resolved_csv_file)
    configs["env"]["following_layout"] = "ngsim"
    configs["env"].pop("following_layout_metadata", None)
    configs["env"]["ngsim_excluded_vehicle_ids"] = [
        int(actor_config["ngsim_vehicle_id"])
        for actor_config in actor_configs.values()
        if actor_config.get("ngsim_vehicle_id") is not None
    ]
    configs["env"]["ngsim_following_actor_map"] = {
        actor_id: int(actor_config["ngsim_vehicle_id"])
        for actor_id, actor_config in actor_configs.items()
        if actor_config.get("ngsim_vehicle_id") is not None
    }
    if verbose:
        _print_dynamic_following_config(configs, resolved_csv_file)
    return configs


def apply_fixed_following_config(configs, csv_file=None, verbose=False):
    """Read handwritten start/end coordinates from scenarios.py, like lane change."""
    configs["scenarios"] = copy.deepcopy(Scenarios.C3_TOWN04_FOLLOWING)
    scenario_actors = configs["scenarios"]["actors"]
    for actor_id in scenario_actors:
        actor_config = configs["actors"][actor_id]
        for key in ("ngsim_vehicle_id", "ngsim_start_frame", "ngsim_end_frame"):
            actor_config.pop(key, None)
        actor_config["initial_speed"] = FIXED_FOLLOWING_INITIAL_SPEEDS[actor_id]
    env = configs.setdefault("env", {})
    env["following_layout"] = "fixed"
    env["ngsim_csv_file"] = os.path.basename(_resolve_ngsim_csv(csv_file))
    expected_csv = configs["scenarios"]["ngsim_csv_file"]
    if env["ngsim_csv_file"] != expected_csv:
        raise ValueError(
            "Fixed following coordinates are validated for {} only; got {}. "
            "Re-audit the scene before changing its background dataset, or use "
            "--following-layout ngsim with the original Lankershim CSV.".format(
                expected_csv, env["ngsim_csv_file"]))
    # Coordinates and the small, explicitly documented exclusion list live in
    # the same scene definition. Train and test therefore share one layout.
    env["ngsim_excluded_vehicle_ids"] = list(
        configs["scenarios"].get("ngsim_excluded_vehicle_ids", []))
    env.pop("ngsim_following_actor_map", None)
    env["following_layout_metadata"] = {
        "source": "scenarios.py:Scenarios.C3_TOWN04_FOLLOWING",
        "initial_speeds": dict(FIXED_FOLLOWING_INITIAL_SPEEDS),
    }
    if verbose:
        print("Fixed Town04 following layout (handwritten scenarios.py coordinates)")
        for actor_id in ("ego", "car1", "car2", "car3"):
            print("  {}: start {}, end {}, speed {:.3f} m/s".format(
                actor_id, scenario_actors[actor_id]["start"], scenario_actors[actor_id]["end"],
                configs["actors"][actor_id]["initial_speed"]))
        print("  excluded background NGSIM IDs:", env["ngsim_excluded_vehicle_ids"])
    return configs


def apply_following_scenario_config(configs, csv_file=None, verbose=False, layout=None):
    layout = layout or configs.get("env", {}).get("following_layout", DEFAULT_FOLLOWING_LAYOUT)
    if layout == "fixed":
        return apply_fixed_following_config(configs, csv_file, verbose)
    if layout == "ngsim":
        return apply_dynamic_ngsim_following_config(configs, csv_file, verbose)
    raise ValueError("Unknown following layout: {}".format(layout))


class FollowingRoads3NPC(MultiCarlaEnv):
    def __init__(self, env_overrides=None, ngsim_csv_file=None):
        self.configs = {
            "scenarios": "C3_TOWN04_FOLLOWING",
            "env": {
                "server_map": "/Game/Carla/Maps/Town04",
                "render": True,
                "render_x_res": 600,
                "render_y_res": 450,
                "x_res": 300,
                "y_res": 120,
                "framestack": 1,
                "discrete_actions": True,
                "squash_action_logits": False,
                "verbose": False,
                "use_depth_camera": False,
                "use_semantic_segmentation_camera": True,
                "send_measurements": True,
                "enable_planner": False,
                "spectator_loc": [-300, 36.8, 70],
                "sync_server": True,
                "fixed_delta_seconds": 0.05,
                "using_imitation_model": True,
                "cummulated_reward": True,
                "cummulated_reward_filename": "Following_PPO_Rewards_inference_full.csv",
            },
            "actors": {
                "ego": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": True,
                    "reward_function": "custom",
                    "scenarios": "C3_TOWN04_FOLLOWING_EGO",
                    "manual_control": False,
                    "auto_control": True,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False,
                    "lidar_sensor": True,
                    "ngsim_vehicle_id": DEFAULT_ACTOR_NGSIM_IDS["ego"],
                    "initial_speed": 8.75 * 0.3048,
                },
                "car1": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "myRewardv1",
                    "scenarios": "C3_TOWN04_FOLLOWING_CAR1",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False,
                    "lidar_sensor": True,
                    "ngsim_vehicle_id": DEFAULT_ACTOR_NGSIM_IDS["car1"],
                    "initial_speed": 7.55 * 0.3048,
                },
                "car2": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "myRewardv1",
                    "scenarios": "C3_TOWN04_FOLLOWING_CAR2",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False,
                    "lidar_sensor": True,
                    "ngsim_vehicle_id": DEFAULT_ACTOR_NGSIM_IDS["car2"],
                    "initial_speed": 5.21 * 0.3048,
                },
                "car3": {
                    "type": "vehicle_4W",
                    "enable_planner": True,
                    "convert_images_to_video": False,
                    "early_terminate_on_collision": False,
                    "reward_function": "myRewardv1",
                    "scenarios": "C3_TOWN04_FOLLOWING_CAR3",
                    "manual_control": False,
                    "auto_control": False,
                    "camera_type": "rgb",
                    "collision_sensor": "on",
                    "lane_sensor": "on",
                    "log_images": False,
                    "log_measurements": False,
                    "render": False,
                    "x_res": 300,
                    "y_res": 120,
                    "use_depth_camera": False,
                    "send_measurements": True,
                    "frame_skip": False,
                    "lidar_sensor": True,
                    "ngsim_vehicle_id": DEFAULT_ACTOR_NGSIM_IDS["car3"],
                    "initial_speed": 8.42 * 0.3048,
                },
            },
        }
        if env_overrides:
            self.configs["env"].update(dict(env_overrides))
        apply_following_scenario_config(
            self.configs, ngsim_csv_file or DEFAULT_NGSIM_CSV, verbose=True)
        super(FollowingRoads3NPC, self).__init__(self.configs)
