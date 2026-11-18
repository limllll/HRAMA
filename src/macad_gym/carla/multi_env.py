import threading
import weakref
from collections import defaultdict
from contextlib import contextmanager
import argparse
import atexit
import shutil
from datetime import datetime
import logging
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import socket
import math
import csv
import pickle
import cv2
import tensorflow as tf
import numpy as np  # linalg.norm is used
import GPUtil
from gym.spaces import Box, Discrete, Tuple, Dict, MultiDiscrete
import pygame
import carla

from macad_gym.core.controllers.traffic import apply_traffic
from macad_gym.multi_actor_env import MultiActorEnv
from macad_gym import LOG_DIR
from macad_gym.core.sensors.utils import preprocess_image
from macad_gym.core.maps.nodeid_coord_map import MAP_TO_COORDS_MAPPING
from macad_gym.agents.imitationmodel.imitation_learning import ImitationLearning
# from macad_gym.core.sensors.utils import get_transform_from_nearest_way_point
from macad_gym.carla.reward import Reward
from macad_gym.core.sensors.hud import HUD
from macad_gym.viz.render import Render
from macad_gym.carla.scenarios import Scenarios
from macad_gym.carla.Utils import calculate_forward_velocity
# The following imports require carla to be imported already.
from macad_gym.core.sensors.camera_manager import CameraManager, CAMERA_TYPES
from macad_gym.core.sensors.derived_sensors import LaneInvasionSensor
from macad_gym.core.sensors.derived_sensors import CollisionSensor
from macad_gym.core.sensors.sensor_lifecycle import (
    SensorCallbackGate, stop_sensors, destroy_sensors, destroy_actor_ids,
)
from macad_gym.core.controllers.keyboard_control import KeyboardControl
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_dao import (  # noqa: E501
    GlobalRoutePlannerDAO,
)
from macad_gym.carla.PythonAPI.agents.navigation.behavior_agent import BehaviorAgent
from macad_gym.carla.Utils import calculate_distance
from macad_gym.carla.PythonAPI.agents.tools.misc import draw_waypoints, calculate_lateral_distance, get_distance_to_end
from macad_gym.carla.sumo_bridge import CarlaSumoBridge
from macad_gym.carla.process_guard import (
    launch_owned_process, stop_owned_process, stop_all_owned_processes,
)

# The following imports depend on these paths being in sys path
# TODO: Fix this. This probably won't work after packaging/distribution
sys.path.append("src/macad_gym/carla/PythonAPI")
from macad_gym.core.maps.nav_utils import PathTracker
from macad_gym.carla.PythonAPI.agents.navigation.global_route_planner_revised import (
    GlobalRoutePlannerv1,
)
from macad_gym.carla.PythonAPI.agents.navigation.local_planner import (
    RoadOption,
)
from macad_gym.PointNet2.models.pointnet2_cls_msg import get_model
from macad_gym.agents.multimodal_encoder.runtime import RuntimeMultimodalEncoder
import torch
import open3d as o3d

logger = logging.getLogger(__name__)


def _reward_checkpoint_metadata(checkpoint_path):
    """Read Tune's completed-episode counter from a checkpoint path/directory."""
    if not checkpoint_path:
        return []
    paths = []
    if os.path.isfile(checkpoint_path + ".tune_metadata"):
        paths.append(checkpoint_path + ".tune_metadata")
    elif os.path.isdir(checkpoint_path):
        for root, _, files in os.walk(checkpoint_path):
            paths.extend(os.path.join(root, name) for name in files
                         if name.endswith(".tune_metadata"))
    result = []
    for path in paths:
        try:
            with open(path, "rb") as stream:
                metadata = pickle.load(stream)
            episodes = metadata.get("episodes_total")
            if episodes is not None:
                result.append((int(metadata.get("iteration", 0)), int(episodes),
                               os.path.getmtime(path)))
        except Exception as exc:
            print("Could not read reward checkpoint metadata {}: {}".format(path, exc))
    return result


def _recover_reward_episode_counter(env_config):
    """Recover a full reward log or continue a clearly identified log segment.

    Missing historical rewards are never fabricated. For a segment log, resume
    numbering at the first episode after the restored checkpoint.
    """
    filename = env_config.get("cummulated_reward_filename")
    if not filename:
        return 0
    # The following PPO executor provides the exact checkpoint it is about to
    # load, including in-memory Ray checkpoints. Never replace that boundary
    # with a larger episode counter from a different policy branch on disk.
    exact_completed = env_config.get("reward_resume_episodes")
    if env_config.get("reward_resume_required") and exact_completed is None:
        raise RuntimeError("Required PPO resume has no authoritative reward boundary")
    if exact_completed is not None and int(exact_completed) < 0:
        raise RuntimeError("reward_resume_episodes must be non-negative")
    candidates = ([] if exact_completed is not None else _reward_checkpoint_metadata(
        env_config.get("reward_resume_checkpoint")))
    checkpoint_dir = env_config.get("reward_checkpoint_dir")
    scan_after = float(env_config.get("reward_checkpoint_scan_after", 0.0))
    if exact_completed is None and checkpoint_dir and os.path.isdir(checkpoint_dir):
        for root, _, files in os.walk(checkpoint_dir):
            for name in files:
                if not name.endswith(".tune_metadata"):
                    continue
                metadata_path = os.path.join(root, name)
                try:
                    if os.path.getmtime(metadata_path) < scan_after:
                        continue
                    with open(metadata_path, "rb") as stream:
                        metadata = pickle.load(stream)
                    episodes = metadata.get("episodes_total")
                    if episodes is not None:
                        candidates.append((int(metadata.get("iteration", 0)),
                                           int(episodes),
                                           os.path.getmtime(metadata_path)))
                except Exception as exc:
                    print("Could not scan reward checkpoint {}: {}".format(
                        metadata_path, exc))

    checkpoint_count = int(exact_completed) if exact_completed is not None else None
    if candidates:
        checkpoint_count = max(
            candidates, key=lambda item: (item[1], item[0], item[2]))[1]
    rows = []
    if os.path.isfile(filename):
        with open(filename, "r") as stream:
            rows = [row for row in csv.reader(stream) if row]
    if checkpoint_count is None:
        if env_config.get("reward_resume_checkpoint"):
            raise RuntimeError(
                "Could not read completed-episode metadata for restored PPO "
                "checkpoint {}; refusing to reset reward numbering.".format(
                    env_config.get("reward_resume_checkpoint")))
        # Without a saved policy state there is no committed episode boundary.
        # Discard rewards from a failed, not-yet-checkpointed trial attempt.
        completed = 0
        rows = []
    else:
        completed = checkpoint_count
        if not rows:
            print(
                "Reward history is unavailable; starting a transparent log "
                "segment at episode {}: {}".format(completed + 1, filename))
        else:
            try:
                first_episode = int(rows[0][0])
            except (IndexError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Invalid first reward row in {}: {}".format(filename, exc))
            if first_episode < 1 or first_episode > completed:
                raise RuntimeError(
                    "Reward log starts at episode {}, incompatible with restored "
                    "checkpoint episode {}: {}".format(
                        first_episode, completed, filename))
            expected_rows = completed - first_episode + 1
            if len(rows) < expected_rows:
                raise RuntimeError(
                    "Reward segment {} has {} rows, but checkpoint episode {} "
                    "requires {} rows from episode {} onward.".format(
                        filename, len(rows), completed, expected_rows,
                        first_episode))
            rows = rows[:expected_rows]
    first_episode = int(rows[0][0]) if rows else completed + 1
    for offset, row in enumerate(rows):
        expected = first_episode + offset
        try:
            actual = int(row[0])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid reward CSV row {} in {}: {}".format(
                expected, filename, exc))
        if actual != expected:
            raise RuntimeError(
                "Reward CSV numbering is not contiguous: row {} contains episode {} "
                "in {}".format(expected, actual, filename))
    parent = os.path.dirname(os.path.abspath(filename))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    if rows or not os.path.isfile(filename):
        with open(filename, "w") as stream:
            csv.writer(stream).writerows(rows)
    print("Reward log recovered at completed episode {}: {}".format(
        completed, filename))
    return completed
# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/cybian/CARLA_0.9.13/CarlaUE4.sh")
)

# Check if is using on Windows
IS_WINDOWS_PLATFORM = "win" in sys.platform

assert os.path.exists(SERVER_BINARY), (
    "Make sure CARLA_SERVER environment"
    " variable is set & is pointing to the"
    " CARLA server startup script (Carla"
    "UE4.sh). Refer to the README file/docs."
)

# Carla planner commands
COMMANDS_ENUM = {
    0.0: "REACH_GOAL",
    5.0: "GO_STRAIGHT",
    4.0: "TURN_RIGHT",
    3.0: "TURN_LEFT",
    2.0: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

ROAD_OPTION_TO_COMMANDS_MAPPING = {
    RoadOption.VOID: "REACH_GOAL",
    RoadOption.STRAIGHT: "GO_STRAIGHT",
    RoadOption.RIGHT: "TURN_RIGHT",
    RoadOption.LEFT: "TURN_LEFT",
    RoadOption.LANEFOLLOW: "LANE_FOLLOW",
}

# Threshold to determine that the goal has been reached based on distance
DISTANCE_TO_GOAL_THRESHOLD = 0.5

# Threshold to determine that the goal has been reached based on orientation
# 判断目标达成的另一个阈值，但这个是通过车辆的朝向来判断的。math.pi / 4.0 约等于 45°，
# 即当车辆的朝向和目标的朝向之间的角度差小于 45° 时，认为车辆已经接近目标的方向
ORIENTATION_TO_GOAL_THRESHOLD = math.pi / 4.0

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 2

# Dummy Z coordinate to use when we only care about (x, y)
# 用来作为一个“假定”的 Z 值，仅用于确保算法能继续运行，而不需要专门关注 Z 轴的变化
GROUND_Z = 22

# DISCRETE_ACTIONS = {
#     # middle forward
#     0: [0.5, 0.0],
#     # turn left
#     1: [0.4, -0.5],
#     # turn right
#     2: [0.4, 0.5],
#     # forward
#     3: [0.75, 0.0],
#     # slow forward
#     4: [0.25, 0.0],
#     # forward left
#     5: [0.45, -0.2],
#     # forward right
#     6: [0.45, 0.2],
#     # sharp left
#     7: [0.3, -0.75],
#     # sharp right
#     8: [0.3, 0.75],
# }
# ============================================================================
# 🎯 MultiDiscrete 动作空间（推荐）- 纵向和横向控制解耦
# ============================================================================
# 维度0（纵向控制）: 5个档位 - 控制油门/刹车
LONGITUDINAL_ACTIONS = [-0.4, -0.2, 0.0, 0.2, 0.4]
# 维度1（横向控制）: 5个档位 - 控制转向
LATERAL_ACTIONS = [-0.3, -0.15, 0.0, 0.15, 0.3]

# ============================================================================
# 旧版 Discrete 动作空间（保留用于向后兼容）
# ============================================================================
DISCRETE_ACTIONS = {
    # middle forward
    0: [0.5, 0.0],
    # turn left
    1: [0.5, -0.5],
    # turn right
    2: [0.5, 0.5],
    # forward
    3: [0.75, 0.0],
    # slow forward
    4: [0.25, 0.0],
    # forward left
    5: [0.5, -0.2],
    # forward right
    6: [0.5, 0.2],
    # sharp left
    7: [0.5, -0.75],
    # sharp right
    8: [0.5, 0.75],
}
WEATHERS = {
    0: carla.WeatherParameters.ClearNoon,
    1: carla.WeatherParameters.CloudyNoon,
    2: carla.WeatherParameters.WetNoon,
    3: carla.WeatherParameters.WetCloudyNoon,
    4: carla.WeatherParameters.MidRainyNoon,
    5: carla.WeatherParameters.HardRainNoon,
    6: carla.WeatherParameters.SoftRainNoon,
    7: carla.WeatherParameters.ClearSunset,
    8: carla.WeatherParameters.CloudySunset,
    9: carla.WeatherParameters.WetSunset,
    10: carla.WeatherParameters.WetCloudySunset,
    11: carla.WeatherParameters.MidRainSunset,
    12: carla.WeatherParameters.HardRainSunset,
    13: carla.WeatherParameters.SoftRainSunset,
}

live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    managed_pids = stop_all_owned_processes()
    live_carla_processes.difference_update(managed_pids)
    for pgid in list(live_carla_processes):
        if IS_WINDOWS_PLATFORM:
            # for Windows
            subprocess.call(["taskkill", "/F", "/T", "/PID", str(pgid)])
        else:
            # for Linux
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    live_carla_processes.clear()


def termination_cleanup(*_):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGTERM, termination_cleanup)
signal.signal(signal.SIGINT, termination_cleanup)
atexit.register(cleanup)

MultiAgentEnvBases = [MultiActorEnv]

try:
    from ray.rllib.env import MultiAgentEnv

    MultiAgentEnvBases.append(MultiAgentEnv)
except ImportError:
    logger.warning("\n Disabling RLlib support.", exc_info=True)


class MultiCarlaEnv(*MultiAgentEnvBases):
    def seed(self, seed=None):
        """Gym/RLlib-compatible deterministic seed hook."""
        if seed is None:
            seed = getattr(self, "_experiment_seed", 2026)
        self._experiment_seed = int(seed)
        random.seed(self._experiment_seed)
        np.random.seed(self._experiment_seed)
        if hasattr(self, "_env_config"):
            self._env_config["seed"] = self._experiment_seed
            self._env_config["sumo_seed"] = self._experiment_seed
        return [self._experiment_seed]

    def __init__(self, configs=None):
        """MACAD-Gym environment implementation.

        Provides a generic MACAD-Gym environmenat implementation that can be
        customized further to create new or variations of existing
        multi-agent learning environments. The environment settings, scenarios
        and the actors in the environment can all be configured using
        the `configs` dict.

        Args:
            configs (dict): Configuration for environment specified under the
                `env` key and configurations for each actor specified as dict
                under `actor`.
                Example:
                    # >>> configs = {"env":{
                    "server_map":"/Game/Carla/Maps/Town05",
                    "discrete_actions":True,...},
                    "actor":{
                    "actor_id1":{"enable_planner":True,...},
                    "actor_id2":{"enable_planner":False,...}
                    }}
        """
        if configs is None:
            self.config = DEFAULT_MULTIENV_CONFIG
            configs = self.config
        self._env_config = configs["env"]
        self._reward_episode_global = _recover_reward_episode_counter(
            self._env_config)
        self._experiment_seed = int(self._env_config.get("seed", 2026))
        random.seed(self._experiment_seed)
        np.random.seed(self._experiment_seed)
        self._use_custom_multimodal_encoder = bool(
            self._env_config.get("use_custom_multimodal_encoder", False)
        )
        self._finetuned_multimodal_encoder_checkpoint = self._env_config.get(
            "finetuned_multimodal_encoder_checkpoint", ""
        )
        self._encoder_collection_only = bool(
            self._env_config.get("encoder_collection_only", False)
        )
        self._multimodal_feature_extractor = None
        self._multimodal_encoder_lock = threading.Lock()
        if (self._use_custom_multimodal_encoder
                and not self._env_config.get("using_imitation_model", False)):
            raise ValueError(
                "use_custom_multimodal_encoder requires using_imitation_model=True "
                "to preserve the 512-dimensional image observation contract"
            )
        if (self._env_config.get("using_imitation_model", False)
                and not self._use_custom_multimodal_encoder
                and not self._encoder_collection_only):
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            self._tf_session = tf.Session(config=config)
            self._image_agent = ImitationLearning(self._tf_session)
            start_time = datetime.now()
            self._image_agent.load_model()
            if self._finetuned_multimodal_encoder_checkpoint:
                self._image_agent.load_finetuned_encoder_weights(
                    self._finetuned_multimodal_encoder_checkpoint
                )
            end_time = datetime.now()
            print("time:", (end_time - start_time).total_seconds())
            print("load successfully")
            data = array = np.random.random((88, 200, 3)) * 255
            start_time = datetime.now()
            feature_vec = self._image_agent.compute_feature(data)
            feature_vec = np.array(feature_vec)
            end_time = datetime.now()
            print("time:", (end_time - start_time).total_seconds())
            print("feature:", feature_vec.shape)

        # Functionalities classes
        self._reward_policy = Reward()
        configs["scenarios"] = Scenarios.resolve_scenarios_parameter(
            configs["scenarios"]
        )

        self._scenario_config = configs["scenarios"]
        self._actor_configs = configs["actors"]

        # At most one actor can be manual controlled
        manual_control_count = 0
        for _, actor_config in self._actor_configs.items():
            if actor_config["manual_control"]:
                if "vehicle" not in actor_config["type"]:
                    raise ValueError("Only vehicles can be manual controlled.")

                manual_control_count += 1

        assert manual_control_count <= 1, (
            "At most one actor can be manually controlled. "
            f"Found {manual_control_count} actors with manual_control=True"
        )

        # Camera position is problematic for certain vehicles and even in
        # autopilot they are prone to error
        self.exclude_hard_vehicles = True
        # list of str: Supported values for `type` filed in `actor_configs`
        # for actors than can be actively controlled
        self._supported_active_actor_types = [
            "vehicle_4W",
            "vehicle_2W",
            "pedestrian",
            "traffic_light",
        ]
        # list of str: Supported values for `type` field in `actor_configs`
        # for actors that are passive. Example: A camera mounted on a pole
        self._supported_passive_actor_types = ["camera"]

        # Set attributes as in gym's specs
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": "human"}

        # Belongs to env_config.
        self._server_map = self._env_config["server_map"]
        self._map = self._server_map.split("/")[-1]
        self._render = self._env_config["render"]
        self._framestack = self._env_config["framestack"]
        self._discrete_actions = self._env_config["discrete_actions"]
        self._squash_action_logits = self._env_config["squash_action_logits"]
        self._verbose = self._env_config["verbose"]
        self._render_x_res = self._env_config["render_x_res"]
        self._render_y_res = self._env_config["render_y_res"]
        self._x_res = self._env_config["x_res"]
        self._y_res = self._env_config["y_res"]
        self._use_depth_camera = self._env_config["use_depth_camera"]
        # self._use_semantic_segmentation_camera = self._env_config["use_semantic_segmentation_camera"]
        self._sync_server = self._env_config["sync_server"]
        self._fixed_delta_seconds = self._env_config["fixed_delta_seconds"]
        self._episode = 0
        # Initialize to be compatible with cam_manager to set HUD.
        pygame.font.init()  # for HUD
        self._hud = HUD(self._render_x_res, self._render_y_res)

        # For manual_control
        self._control_clock = None
        self._manual_controller = None
        self._manual_control_camera_manager = None

        # Render related
        Render.resize_screen(self._render_x_res, self._render_y_res)

        self._camera_poses, window_dim = Render.get_surface_poses(
            [self._x_res, self._y_res], self._actor_configs
        )

        if manual_control_count == 0:
            Render.resize_screen(window_dim[0], window_dim[1])
        else:
            self._manual_control_render_pose = (0, window_dim[1])
            Render.resize_screen(
                max(self._render_x_res, window_dim[0]),
                self._render_y_res + window_dim[1],
            )

        # Actions space
        if self._discrete_actions:
            # 🎯 使用 MultiDiscrete 动作空间：[纵向控制, 横向控制]
            # 每个维度有5个离散档位，共25种动作组合
            self.action_space = Dict(
                {
                    actor_id: MultiDiscrete([
                        len(LONGITUDINAL_ACTIONS),  # 纵向: 5档 (急刹/缓刹/滑行/巡航/急加)
                        len(LATERAL_ACTIONS)  # 横向: 5档 (大左/小左/直行/小右/大右)
                    ])
                    for actor_id in self._actor_configs.keys()
                }
            )
        else:
            self.action_space = Dict(
                {
                    actor_id: Box(-1.0, 1.0, shape=(2,))
                    for actor_id in self._actor_configs.keys()
                }
            )

        # Output space of images after preprocessing
        if self._use_depth_camera:
            self._image_space = Box(
                0.0, 255.0, shape=(self._y_res, self._x_res, 1 * self._framestack)
            )
        else:
            self._image_space = Box(
                0.0, 1.0, shape=(self._y_res, self._x_res, 3 * self._framestack)
            )
        self._feature_space = Box(0.0, 1.0, shape=(512,))
        # ✅ 修改为256维，匹配PointNet输出，不再压缩
        self._lidar_feature_space = Box(0.0, 1.0, shape=(256,))
        # Observation space in output
        if self._env_config["send_measurements"]:
            if self._env_config["using_imitation_model"]:
                self.observation_space = Dict(
                    {
                        actor_id: Tuple(
                            [
                                self._feature_space,  # feature
                                # Discrete(len(COMMANDS_ENUM)),  # next_command
                                Box(
                                    -20.0, 20.0, shape=(11,)
                                ),  # directionx/y, rel_speed_x/y, rel_loc_x/y, next-command-onehot encode(0~4)
                                self._lidar_feature_space,
                            ]
                        )
                        for actor_id in self._actor_configs.keys()
                    }
                )
            else:
                self.observation_space = Dict(
                    {
                        actor_id: Tuple(
                            [
                                self._image_space,  # image
                                # Discrete(len(COMMANDS_ENUM)),  # next_command
                                Box(
                                    -20.0, 20.0, shape=(11,)
                                ),  # directionx/y, rel_speed_x/y, rel_loc_x/y, next-command-onehot encode(0~4)
                                self._lidar_feature_space,
                            ]
                        )
                        for actor_id in self._actor_configs.keys()
                    }
                )
        else:
            self.observation_space = Dict(
                {actor_id:
                     Tuple([self._image_space,
                            self._lidar_feature_space]
                           ) for actor_id in
                 self._actor_configs.keys()}
            )

        # Set appropriate node-id to coordinate mappings for Town01 or Town02.
        self.pos_coor_map = MAP_TO_COORDS_MAPPING[self._map]

        self._spec = lambda: None
        self._spec.id = "Carla-v0"
        self._server_port = None
        self._server_process = None
        self._client = None
        self._num_steps = {}
        self._total_reward = {}
        self._target_waypoint = {}
        self._prev_measurement = {}
        self._prev_image = {}
        self._episode_id_dict = {}
        self._measurements_file_dict = {}
        self._weather = None
        self._start_pos = {}  # Start pose for each actor
        self._end_pos = {}  # End pose for each actor
        self._start_coord = {}
        self._end_coord = {}
        self._last_obs = None
        self._image = None
        self._surface = None
        self._video = False
        self._obs_dict = {}
        self._done_dict = {}
        self._previous_actions = {}
        self._previous_rewards = {}
        self._last_reward = {}
        # One-step control overrides are used by the RSS counterfactual
        # reference driver.  Normal training/inference never sets them.
        self._external_control_overrides = {}
        self._external_control_sources = {}
        self._last_actor_control_sources = {}
        self._external_ego_controller = None
        if self._env_config.get("wor_controller"):
            from macad_gym.ads.wor import WORController
            if self._actor_configs["ego"].get("required_external_controller") != "wor":
                raise RuntimeError("WOR requires strict external ego control")
            self._external_ego_controller = WORController(self._env_config["wor_controller"])
        self._npc_vehicles = []  # List of NPC vehicles
        self._npc_pedestrians = []  # List of NPC pedestrians
        self._agents = {}  # Dictionary of macad_agents with agent_id as key
        self._actors = {}  # Dictionary of actors with actor_id as key
        # Sampled initial speeds are applied only after all reset-time sensor
        # ticks have completed, so reset does not advance vehicles away from
        # their configured spawn transforms.
        self._initial_speed_by_actor = {}
        self._actor_id_in_carla = {}  # Dictionary of actor_id, with actor_id_in_carla as key, for a given actor_id_in_carla, we can get the actor_id
        self._cameras = {}  # Dictionary of sensors with actor_id as key
        self._path_trackers = {}  # Dictionary of sensors with actor_id as key
        self._collisions = {}  # Dictionary of sensors with actor_id as key
        self._lane_invasions = {}  # Dictionary of sensors with actor_id as key
        self._scenario_map = {}  # Dictionary with current scenario map config
        self._behavior_agents = {}
        self._initial_distance_to_end = {}
        self._end_distance_to_end = {}
        self._rc_segment_initial = {}
        self._rc_progress_sum = {}
        self._rc_segment_count = {}
        self._rc_normalized = {}
        self._time_to_firstcollision = {}
        self._time_to_goal = {}
        self._episode_reward = defaultdict(list)
        # 🎯 新增：NGSIM 背景车辆轨迹误差累积（用于计算 episode 级别的 RMSE）
        self._ep_bg_total_error = 0.0  # 累积误差的平方和
        self._ep_bg_total_count = 0  # 累积有效车辆数
        self._ep_bg_candidate_frames = 0
        self._ep_bg_valid_frames = 0
        self._ep_bg_pending_frames = 0
        # 添加雷达传感器
        self._radars = {}
        self._lidar_features = {}
        self._lidar_lock = threading.Lock()
        self._lidar_model_lock = threading.Lock()
        self._sensor_callback_gate = SensorCallbackGate()
        self._lidar_range = float(self._env_config.get("lidar_range", 20.0))
        if self._lidar_range <= 0.0:
            raise ValueError("lidar_range must be positive")

        self.lidar_model = None
        if self._use_custom_multimodal_encoder:
            custom_checkpoint = self._env_config.get(
                "multimodal_encoder_checkpoint", ""
            ) or os.environ.get("HRAMA_MULTIMODAL_ENCODER_CHECKPOINT", "")
            if not custom_checkpoint:
                raise ValueError(
                    "Custom multimodal encoder is enabled but no checkpoint "
                    "was supplied"
                )
            self._multimodal_feature_extractor = RuntimeMultimodalEncoder(
                custom_checkpoint,
                device=self._env_config.get(
                    "multimodal_encoder_device", "auto"
                ),
            )
            print(
                "Loaded custom multimodal encoder from {} on {}".format(
                    self._multimodal_feature_extractor.checkpoint_path,
                    self._multimodal_feature_extractor.device,
                )
            )
        elif not self._encoder_collection_only:
            self._load_legacy_lidar_encoder()

        # SUMO 協同橋接器（啟動於 _init_server 之後）
        self._sumo_bridge = None
        self._cosim_available = False
        self._episode_cosim_available = False
        self._episode_bg_metric_available = False
        self._cosim_last_error = None
        self._cosim_last_status = None

    def _load_legacy_lidar_encoder(self):
        """Load the original ModelNet40 PointNet++ path unchanged."""
        self.lidar_model = get_model(num_class=40, normal_channel=False)
        checkpoint_path = self._env_config.get(
            "legacy_lidar_checkpoint",
            "/home/simplexity/cybian/myown/carla-sumo-ads-11.28/"
            "src/macad_gym/PointNet2/log/classification/"
            "pointnet2_msg_normals/checkpoints/best_model.pth",
        )
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_dict = self.lidar_model.state_dict()

        # 1. 筛选可加载的权重
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                           if k in model_dict and v.shape == model_dict[k].shape}

        # 2. 特殊处理第一层卷积权重（从6通道改为3通道）
        for key in ['sa1.conv_blocks.0.0.weight', 'sa1.conv_blocks.1.0.weight', 'sa1.conv_blocks.2.0.weight']:
            if key in checkpoint['model_state_dict']:
                # 取前3个输入通道的权重（XYZ），舍弃后3个（法向量）
                pretrained_dict[key] = checkpoint['model_state_dict'][key][:, :3, :, :]

        model_dict.update(pretrained_dict)
        self.lidar_model.load_state_dict(model_dict)
        if self._finetuned_multimodal_encoder_checkpoint:
            self._load_finetuned_lidar_encoder_weights()
        print("成功加载匹配的权重参数")
        print("PointNet2加载成功")

        self.lidar_model.eval()
        warmup_path = self._env_config.get(
            "legacy_lidar_warmup",
            "/home/simplexity/cybian/myown/carla-sumo-ads-11.28/"
            "src/macad_gym/PointNet2/My/lidar/76.npy",
        )
        point = np.load(warmup_path)
        # print("原始点云形状:", point.shape)

        # 体素化降采样
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point[:, :3])
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # print("降采样后点云形状:", downsampled_points.shape)

        if downsampled_points.size == 0:
            raise RuntimeError("PointNet warm-up point cloud is empty after voxel downsampling")

        # Use the configured physical sensing range instead of per-scan min/max.
        # This preserves the metric meaning of coordinates between frames.
        downsampled_points = np.nan_to_num(
            downsampled_points, nan=0.0, posinf=self._lidar_range,
            neginf=-self._lidar_range)
        downsampled_points = np.clip(
            downsampled_points / self._lidar_range, -1.0, 1.0)
        # print("归一化后点云范围:", np.min(downsampled_points), "到", np.max(downsampled_points))
        # 准备输入数据（只使用XYZ坐标）
        downsampled_points = torch.from_numpy(downsampled_points).unsqueeze(0).float().transpose(1, 2)
        # print("模型输入形状:", downsampled_points.shape)

        # 推理
        with torch.inference_mode():
            self.lidar_model(downsampled_points)
            # print("分类预测结果:", pred.shape)

            # print("my特征向量:", my.shape)
            # print("my张量:", np.array(my).shape)
            # print("my张量最小值:", np.min(np.array(my)))
            # print("my张量最大值:", np.max(np.array(my)))

    def _load_finetuned_lidar_encoder_weights(self):
        """Load adapted PointNet++ weights into the existing LiDAR model."""
        checkpoint = torch.load(
            os.path.abspath(os.path.expanduser(
                self._finetuned_multimodal_encoder_checkpoint
            )),
            map_location=torch.device("cpu"),
        )
        state = checkpoint.get("lidar_encoder")
        if not isinstance(state, dict):
            raise ValueError(
                "Fine-tuned checkpoint has no lidar_encoder state: {}".format(
                    self._finetuned_multimodal_encoder_checkpoint
                )
            )

        current = self.lidar_model.state_dict()
        matched = {}
        for name, value in state.items():
            target_name = name[len("module."):] if name.startswith("module.") else name
            if (target_name in current
                    and tuple(value.shape) == tuple(current[target_name].shape)):
                matched[target_name] = value

        encoder_prefixes = (
            "sa1.", "sa2.", "sa3.", "fc1.", "bn1.", "fc2.", "bn2."
        )
        expected = [
            name for name in current
            if name.startswith(encoder_prefixes)
        ]
        missing = [name for name in expected if name not in matched]
        if missing:
            raise ValueError(
                "Fine-tuned LiDAR weights do not match the original PointNet++ "
                "encoder; missing parameters: {}".format(", ".join(missing))
            )

        current.update(matched)
        self.lidar_model.load_state_dict(current)
        print(
            "Loaded fine-tuned LiDAR weights into the original PointNet++ "
            "encoder ({} tensors)".format(len(matched))
        )

    @staticmethod
    def _get_tcp_port(port=0):
        """
        Get a free tcp port number
        :param port: (default 0) port number. When `0` it will be assigned a free port dynamically
        :return: a port number requested if free otherwise an unhandled exception would be thrown
        """
        s = socket.socket()
        s.bind(("", port))
        server_port = s.getsockname()[1]
        s.close()
        return server_port

    def _init_server(self):
        """Initialize carla server and client

        Returns:
            N/A
        """
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        # First find a port that is free and then use it in order to avoid
        # crashes due to:"...bind:Address already in use"
        self._server_port = self._get_tcp_port()

        multigpu_success = False
        gpus = GPUtil.getGPUs()
        log_file = os.path.join(LOG_DIR, "server_" + str(self._server_port) + ".log")
        logger.info(
            f"1. Port: {self._server_port}\n"
            f"2. Map: {self._server_map}\n"
            f"3. Binary: {SERVER_BINARY}"
        )
        if multigpu_success is False:
            try:
                print("Using single gpu to initialize carla server")

                server_command = [
                    SERVER_BINARY,
                    "-windowed ",
                    "-ResX=", str(self._env_config["render_x_res"]),
                    "-ResY=", str(self._env_config["render_y_res"]),
                    "-benchmark -fps=10",
                    "-carla-server",
                    "-carla-rpc-port={}".format(self._server_port),
                    "-carla-streaming-port=0",
                ]
                server_command.extend(self._env_config.get("carla_extra_args", []))
                with open(log_file, "w") as server_log:
                    self._server_process = launch_owned_process(
                        server_command, stdout=server_log, stderr=server_log,
                        name="carla")
                print("Running simulation in single-GPU mode")
            except Exception:
                self._server_process = None
                logger.exception("Failed to launch CARLA")
                raise

        live_carla_processes.add(self._server_process.pid)

        # Start client
        self._client = None
        startup_timeout = float(self._env_config.get("carla_startup_timeout", 180.0))
        connect_deadline = time.time() + startup_timeout
        while self._client is None:
            if time.time() >= connect_deadline:
                self._clear_server_state()
                raise RuntimeError("Timed out while starting the CARLA server")
            if self._server_process is None or self._server_process.poll() is not None:
                self._clear_server_state()
                raise RuntimeError("CARLA server exited during startup")
            try:
                print("connecting to carla ... server port:", self._server_port)
                self._client = carla.Client("localhost", self._server_port)
                time.sleep(2)
                self._client.set_timeout(20.0)
                print(
                    "Client successfully connected to server, Carla-Server version: ",
                    self._client.get_server_version(),
                )
            except RuntimeError as re:
                if "timeout" not in str(re) and "time-out" not in str(re):
                    print("Could not connect to Carla server because:", re)
                self._client = None
                time.sleep(1.0)

        self._rpc_timeout = 120.0
        self._client.set_timeout(self._rpc_timeout)
        # load map using client api since 0.9.6+
        self._client.load_world(self._server_map)
        self.world = self._client.get_world()
        world_settings = self.world.get_settings()
        world_settings.synchronous_mode = self._sync_server
        if self._sync_server:
            # Synchronous mode
            # try:
            # Available with CARLA version>=0.9.6
            # Set fixed_delta_seconds to have reliable physics between sim steps
            world_settings.fixed_delta_seconds = self._fixed_delta_seconds
        self.world.apply_settings(world_settings)
        # 啟動 SUMO 橋接器（基於 OpenDRIVE 對齊）
        # 检查是否启用SUMO协同仿真
        enable_sumo_cosim = self._env_config.get("enable_sumo_cosim", False)

        if not enable_sumo_cosim:
            print("🚫 SUMO协同仿真已禁用，仅使用CARLA仿真")
            self._sumo_bridge = None
            self._cosim_available = False
            self._cosim_last_error = "SUMO co-simulation is disabled"
            self._cosim_last_status = {
                "ok": False,
                "error": self._cosim_last_error,
            }
        else:
            print("🚗 启用SUMO协同仿真")
            try:
                # 检查是否启用NGSIM背景交通流
                enable_ngsim = self._env_config.get("enable_ngsim_background", False)
                ngsim_csv_file = self._env_config.get("ngsim_csv_file", "final_60.csv")

                # 获取GUI和同步配置
                sumo_gui_enabled = self._env_config.get("sumo_gui", False)
                sumo_sync_enabled = self._env_config.get("sumo_to_carla_sync", False)

                # 🎯 输出SUMO配置信息用于调试
                print(f"🎯 SUMO协同仿真配置:")
                print(f"  🚗 SUMO GUI界面: {sumo_gui_enabled}")
                print(f"  🔄 双向车辆同步: {sumo_sync_enabled}")
                print(f"  📁 NGSIM背景交通: {enable_ngsim}")
                print(f"  📄 NGSIM数据文件: {ngsim_csv_file}")
                print(f"  🗺️  当前地图: {self._map}")
                print(f"  🔍 详细日志: {self._verbose}")

                # 🎯 NGSIM状态机配置参数（简化版：危险时SUMO永久接管）
                # 注：deviation_threshold 和 recovery_frames 已废弃，保留用于向后兼容
                danger_distance = self._env_config.get("ngsim_danger_distance_threshold", 10.0)  # 默认10米
                danger_ttc = self._env_config.get("ngsim_danger_ttc_threshold", 3.0)  # 默认3秒
                excluded_ngsim_vehicle_ids = self._env_config.get("ngsim_excluded_vehicle_ids", [])

                self._sumo_bridge = CarlaSumoBridge(
                    world=self.world,
                    fixed_delta_seconds=self._fixed_delta_seconds,
                    town_map_name=self._map,
                    sumo_gui=sumo_gui_enabled,  # 动态读取GUI配置
                    verbose=self._verbose,  # 傳遞詳細日誌設置
                    sumo_to_carla_sync=sumo_sync_enabled,  # 动态读取双向同步配置
                    enable_ngsim_background=enable_ngsim,  # 启用NGSIM背景交通
                    ngsim_csv_file=ngsim_csv_file,  # NGSIM数据文件
                    excluded_ngsim_vehicle_ids=excluded_ngsim_vehicle_ids,
                    random_seed=self._env_config.get("sumo_seed", self._experiment_seed),
                    # 状态机配置参数（简化版：危险时永久接管）
                    danger_distance_threshold=danger_distance,  # 周围车辆欧氏距离越过阈值时触发接管
                    danger_ttc_threshold=danger_ttc  # TTC<3秒触发接管
                )
                self._sumo_bridge.start()
                self._cosim_available = True
                self._cosim_last_error = None
                self._cosim_last_status = {"ok": True, "phase": "start"}

                # 根据实际配置显示启动信息
                gui_status = "含GUI可视化" if sumo_gui_enabled else "无头模式"
                sync_status = "双向同步" if sumo_sync_enabled else "单向同步"
                print(f"✅ SUMO 橋接器啟動成功（{gui_status}，{sync_status}）- 地圖: {self._map}")

            except Exception as e:
                if self._sumo_bridge is not None:
                    try:
                        self._sumo_bridge.close()
                    except Exception:
                        logger.exception("Failed to clean a partially started SUMO bridge")
                        raise
                self._sumo_bridge = None
                self._cosim_available = False
                self._cosim_last_error = str(e)
                self._cosim_last_status = {
                    "ok": False,
                    "phase": "start",
                    "error": str(e),
                }
                print("❌ SUMO 橋接器啟動失敗（將僅用 CARLA）：{}".format(e))
        self._init_traffic_manager()

        # Set the spectator/server view if rendering is enabled
        if self._render and self._env_config.get("spectator_loc"):
            spectator = self.world.get_spectator()
            spectator_loc = carla.Location(*self._env_config["spectator_loc"])
            d = 6.4
            angle = 160  # degrees
            a = math.radians(angle)
            location = (
                    carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + spectator_loc
            )
            spectator.set_transform(
                # carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))
                carla.Transform(location, carla.Rotation(pitch=-90))
            )

        if self._env_config.get("enable_planner"):
            planner_dao = GlobalRoutePlannerDAO(self.world.get_map())
            self.planner = GlobalRoutePlannerv1(planner_dao)
            self.planner.setup()

    def _validate_traffic_manager_configuration(self):
        """Keep the no-TM path restricted to externally controlled vehicles."""
        if self._env_config.get("enable_carla_traffic_manager", True):
            return
        if self._scenario_map.get("num_vehicles", 0) > 0:
            raise ValueError(
                "enable_carla_traffic_manager=False cannot spawn CARLA-native "
                "autopilot traffic; SUMO-controlled background traffic is supported"
            )
        if any(config.get("manual_control", False)
               for config in self._actor_configs.values()):
            raise ValueError(
                "enable_carla_traffic_manager=False cannot use manual_control "
                "because the keyboard controller can enable CARLA autopilot"
            )

    def _init_traffic_manager(self):
        """Do not create a native TM thread when no vehicle uses it.

        CARLA 0.9.13 keeps Traffic Managers in a process-wide C++ map. Dropping
        a Python reference after a hard server restart does not stop an old
        manager's thread. In the PPO/BehaviorAgent/SUMO-only following scene,
        avoid creating that unused manager in the first place. Other scenes
        retain the original TM settings unless they explicitly opt out.
        """
        self._validate_traffic_manager_configuration()
        if not self._env_config.get("enable_carla_traffic_manager", True):
            self._traffic_manager = None
            self._traffic_manager_port = None
            logger.info(
                "CARLA Traffic Manager disabled: vehicle control stays with "
                "PPO/BehaviorAgent/SUMO; no native autopilot traffic requested"
            )
            return
        self._traffic_manager_port = self._get_tcp_port()
        self._traffic_manager = self._client.get_trafficmanager(self._traffic_manager_port)
        self._traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self._traffic_manager.set_respawn_dormant_vehicles(True)
        self._traffic_manager.set_synchronous_mode(self._sync_server)

    def _check_cleanup_server(self):
        process = self._server_process
        if process is not None and process.poll() is not None:
            raise RuntimeError("CARLA exited during cleanup; refusing further RPCs")

    @contextmanager
    def _cleanup_rpc_timeout(self):
        """Bound reset/close RPCs without changing normal training timeouts."""
        client = self._client
        if client is None:
            raise RuntimeError("No CARLA client is available for cleanup")
        timeout = float(self._env_config.get("carla_cleanup_timeout", 5.0))
        if not 0.0 < timeout <= 30.0:
            raise ValueError("carla_cleanup_timeout must be in (0, 30] seconds")
        client.set_timeout(timeout)
        try:
            self._check_cleanup_server()
            yield timeout
        finally:
            # set_timeout is local; restoring it does not contact a dead server.
            client.set_timeout(getattr(self, "_rpc_timeout", 120.0))

    def _clean_world(self):
        """Destroy all actors cleanly before exiting

        Returns:
            N/A
        """
        # ✅ 关键修复1：检查world对象是否存在
        if not hasattr(self, 'world') or self.world is None:
            print("⚠️ World对象不存在，跳过清理")
            return

        print("🧹 开始清理环境...")

        gate = self._sensor_callback_gate
        gate.begin_close()
        cameras = list(self._cameras.values())
        manual_camera = getattr(self, "_manual_control_camera_manager", None)
        if manual_camera is not None:
            cameras.append(manual_camera)
        # Detach managers first so GC cannot issue a duplicate destroy, and late
        # RGB frames cannot mutate a camera belonging to the next episode.
        sensors = [camera.detach_sensor() for camera in cameras]
        if self._external_ego_controller is not None:
            sensors.extend(self._external_ego_controller.detach_sensors())
        sensors.extend(self._radars.values())
        for owner in list(self._collisions.values()) + list(self._lane_invasions.values()):
            sensors.append(owner.sensor)
            owner.sensor = None

        process = self._server_process
        server_exited = process is not None and process.poll() is not None
        # First finish CPU encoding. Stop must not race an old callback's work.
        gate.wait_for_idle(timeout=10.0)
        if not server_exited:
            with self._cleanup_rpc_timeout() as timeout:
                check = self._check_cleanup_server
                stop_sensors(sensors, ensure_server_alive=check)
                destroy_sensors(sensors, client=self._client,
                                ensure_server_alive=check)
                if self._sync_server and any(sensor is not None for sensor in sensors):
                    check()
                    self.world.tick(timeout)
                vehicles = list(self._actors.values()) + list(self._npc_vehicles)
                pairs = list(zip(*self._npc_pedestrians))
                pedestrians = [pair[0] for pair in pairs]
                controllers = [pair[1] for pair in pairs]
                for controller in controllers:
                    check()
                    if controller.is_alive:
                        controller.stop()
                destroy_actor_ids(self._client,
                    [actor.id for actor in vehicles + list(controllers) + list(pedestrians)
                     if actor is not None and actor.is_alive], check)
                if self._sync_server:
                    check()
                    self.world.tick(timeout)
                check()
                # Retain the existing orphan-vehicle cleanup, but use one batch
                # and never conceal RPC failures or loop over a dead server.
                remaining = self.world.get_actors().filter('vehicle.*')
                if remaining:
                    destroy_actor_ids(self._client, [actor.id for actor in remaining], check)
                    if self._sync_server:
                        check()
                        self.world.tick(timeout)
                print("✅ CARLA sensor and vehicle batches cleaned")
        else:
            print("CARLA has exited; skipping sensor/actor RPCs to the dead server")

        # SUMO-mapped actors are cleaned by the bridge reset before this method.
        self._manual_control_camera_manager = None
        with self._lidar_lock:
            self._lidar_features.clear()
        self._cameras = {}
        self._actors = {}
        self._initial_speed_by_actor = {}
        self._external_control_overrides = {}
        self._external_control_sources = {}
        self._last_actor_control_sources = {}
        self._radars = {}
        self._lidar_features = {}
        self._actor_id_in_carla = {}
        self._npc_vehicles = []
        self._npc_pedestrians = []
        self._path_trackers = {}
        self._collisions = {}
        self._lane_invasions = {}
        self._behavior_agents = {}
        self._initial_distance_to_end = {}
        self._end_distance_to_end = {}
        self._rc_segment_initial = {}
        self._rc_progress_sum = {}
        self._rc_segment_count = {}
        self._rc_normalized = {}
        self._time_to_firstcollision = {}
        self._time_to_goal = {}

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gate.reopen()
        print("✅ 环境清理完成")

    def _clear_server_state(self):
        """Stop owned simulators before discarding stale world/sensor references.

        This is hard teardown after a failed reset, not another actor-destroy
        RPC against an unhealthy server. PPO state and the reward ledger live
        outside this simulator cleanup and must not be restored or truncated.
        """
        print("Clearing Carla server state")
        gate = self._sensor_callback_gate
        gate.begin_close()
        for camera in list(self._cameras.values()):
            camera.detach_sensor()
        if self._manual_control_camera_manager is not None:
            self._manual_control_camera_manager.detach_sensor()
        for owner in list(self._collisions.values()) + list(self._lane_invasions.values()):
            owner.sensor = None

        cleanup_errors = []
        try:
            gate.wait_for_idle(timeout=10.0)
        except Exception as exc:
            cleanup_errors.append(("sensor callbacks", exc))
            # Still stop the owned simulators even if a callback is stuck.

        bridge = self._sumo_bridge
        try:
            if bridge is not None:
                bridge.close()
        except Exception as exc:
            # CarlaSumoBridge.close() stops the owned process in finally, then
            # clears _sumo_proc. A broken TraCI socket alone must not prohibit
            # recovery when process-group cleanup has already succeeded.
            if getattr(bridge, "_sumo_proc", bridge) is None:
                logger.warning("TraCI close failed after owned SUMO process stopped: %s", exc)
            else:
                cleanup_errors.append(("SUMO process", exc))
                logger.exception("SUMO cleanup failed")
        if bridge is None or getattr(bridge, "_sumo_proc", bridge) is None:
            self._sumo_bridge = None
        self._cosim_available = False
        self._episode_cosim_available = False
        self._episode_bg_metric_available = False
        process = self._server_process
        try:
            if process is not None:
                stop_owned_process(process)
                live_carla_processes.discard(process.pid)
        except Exception as exc:
            cleanup_errors.append(("CARLA process", exc))
            logger.exception("CARLA cleanup failed")
        else:
            # Release the client only after the old server group has exited.
            self._server_process = None
            self._server_port = None
            self._client = None
            self.world = None
            self._traffic_manager = None
            self._traffic_manager_port = None
            self.planner = None

        if cleanup_errors:
            raise RuntimeError("Incomplete simulator cleanup ({}); refusing another server launch".format(
                ", ".join(name for name, _ in cleanup_errors))) from cleanup_errors[0][1]

        self._manual_control_camera_manager = None
        if self._external_ego_controller is not None:
            self._external_ego_controller.abandon_episode()
        with self._lidar_lock:
            self._lidar_features.clear()
        # No sensor/actor/manager from the old server may enter the new world.
        for name in (
                "_cameras", "_actors", "_initial_speed_by_actor",
                "_external_control_overrides", "_radars", "_actor_id_in_carla",
                "_external_control_sources", "_last_actor_control_sources",
                "_path_trackers", "_collisions", "_lane_invasions",
                "_behavior_agents", "_initial_distance_to_end", "_end_distance_to_end",
                "_rc_segment_initial", "_rc_progress_sum", "_rc_segment_count",
                "_rc_normalized", "_time_to_firstcollision", "_time_to_goal",
                "_target_waypoint", "_prev_measurement", "_prev_image", "_obs_dict",
                "_done_dict", "_previous_actions", "_previous_rewards"):
            setattr(self, name, {})
        self._npc_vehicles = []
        self._npc_pedestrians = []

    def reset(self):
        """Reset the carla world, call _init_server()

        Returns:
            N/A
        """
        reset_retries = int(self._env_config.get("carla_reset_retries", 1))
        if reset_retries not in (1, 2):
            raise ValueError("carla_reset_retries must be 1 or 2 total attempts")
        # Select the scenario once; a simulator retry is not a new PPO episode.
        self._episode += 1
        self._load_scenario(self._scenario_config)

        for retry in range(reset_retries):
            try:
                if not self._server_process:
                    self._init_server()
                    self._reset(clean_world=False)
                else:
                    self._reset()
                # Sensor warmup is part of reset too: timeout/tick failures
                # here need the same bounded recovery as actor destruction.
                observations = self._prepare_reset_observations()
                print("_reset OK")
                return observations
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                print("reset(): Retry #: {}/{}".format(retry + 1, reset_retries))
                self._clear_server_state()
                if retry + 1 >= reset_retries or not isinstance(e, (RuntimeError, OSError)):
                    raise
                print("CARLA reset failed; rebuilding simulators once in the same "
                      "worker (PPO state and reward log are retained)")
                time.sleep(10.0)
                continue

    def _prepare_reset_observations(self):
        """Initialize sensors/observations without counting a second episode."""
        # Set appropriate initial values for all actors
        self._last_actor_control_sources = {}
        for actor_id, actor_config in self._actor_configs.items():
            if True:
                self._target_waypoint[actor_id] = None
                self._last_reward[actor_id] = None
                self._total_reward[actor_id] = None
                self._num_steps[actor_id] = 0
                # self._behavior_agents[actor_id]=None
                self._time_to_firstcollision[actor_id] = None
                self._time_to_goal[actor_id] = None
                py_measurement = self._read_observation(actor_id)
                self._prev_measurement[actor_id] = py_measurement
                cam = self._cameras[actor_id]
                if self._radars.get(actor_id) is not None:
                    lidar = self._radars[actor_id]
                    # 设置回调函数
                    weak_self = weakref.ref(self)
                    generation = self._sensor_callback_gate.generation
                    lidar.listen(
                        lambda point_cloud, aid=actor_id, gen=generation:
                        MultiCarlaEnv._on_lidar_frame(
                            weak_self, aid, point_cloud, gen))
                # Wait for the sensor (camera) actor to start streaming
                # Shouldn't take too long
                requires_lidar = actor_config.get("lidar_sensor", False)
                sensor_wait_timeout = float(
                    self._env_config.get("sensor_wait_timeout", 30.0))
                sensor_wait_started = time.time()
                while (cam.callback_count == 0 or
                       (requires_lidar and self._lidar_features.get(actor_id) is None)):
                    if time.time() - sensor_wait_started >= sensor_wait_timeout:
                        missing = []
                        if cam.callback_count == 0:
                            missing.append("camera")
                        if requires_lidar and self._lidar_features.get(actor_id) is None:
                            missing.append("lidar")
                        raise RuntimeError(
                            "Timed out waiting {:.1f}s for {} sensor(s) of actor {}".format(
                                sensor_wait_timeout, ", ".join(missing), actor_id))
                    if self._sync_server:
                        self.world.tick()
                        # print("1帧～")
                        # `wait_for_tick` is no longer needed, see https://github.com/carla-simulator/carla/pull/1803
                        # self.world.wait_for_tick()
                    time.sleep(0.05)
                if cam.image is None:
                    print("callback_count:", actor_id, ":", cam.callback_count)
                obs = self._encode_obs(actor_id, cam.image, self._lidar_features.get(actor_id, np.zeros(256)),
                                       py_measurement)
                # del self._lidar_features[actor_id]
                self._obs_dict[actor_id] = obs
                # Actor correctly reset
                self._done_dict[actor_id] = False

        # All reset-time sensor ticks are complete. Restore the exact configured
        if self._external_ego_controller is not None:
            self._external_ego_controller._wait_frame(int(self.world.get_snapshot().frame))
        # poses and set the sampled velocities without advancing another frame.
        self._apply_initial_actor_states()

        # Refresh kinematic measurements after the final pose/velocity write.
        # Camera and LiDAR data were collected at the same spawn poses while the
        # vehicles were stationary, so no additional world tick is needed here.
        for actor_id in self._actors:
            py_measurement = self._read_observation(actor_id)
            self._prev_measurement[actor_id] = py_measurement
            cam = self._cameras[actor_id]
            with self._lidar_lock:
                lidar_feature = self._lidar_features.get(
                    actor_id, np.zeros(256, dtype=np.float32)).copy()
            self._obs_dict[actor_id] = self._encode_obs(
                actor_id, cam.image, lidar_feature, py_measurement)

        return self._obs_dict

    # ! Deprecated method
    def _on_render(self):
        """Render the pygame window.

        Args:

        Returns:
            N/A
        """
        pass

    def _spawn_new_actor(self, actor_id):
        """Spawn an agent as per the blueprint at the given pose

        Args:
            blueprint: Blueprint of the actor. Can be a Vehicle or Pedestrian
            pose: carla.Transform object with location and rotation

        Returns:
            An instance of a subclass of carla.Actor. carla.Vehicle in the case
            of a Vehicle agent.

        """
        actor_type = self._actor_configs[actor_id].get("type", "vehicle_4W")
        if actor_type not in self._supported_active_actor_types:
            print("Unsupported actor type:{}. Using vehicle_4W as the type")
            actor_type = "vehicle_4W"

        if actor_type == "traffic_light":
            # Traffic lights already exist in the world & can't be spawned.
            # Find closest traffic light actor in world.actor_list and return
            from macad_gym.core.controllers import traffic_lights

            loc = carla.Location(
                self._start_pos[actor_id][0],
                self._start_pos[actor_id][1],
                self._start_pos[actor_id][2],
            )
            rot = (
                self.world.get_map()
                .get_waypoint(loc, project_to_road=True)
                .transform.rotation
            )
            #: If yaw is provided in addition to (X, Y, Z), set yaw
            if len(self._start_pos[actor_id]) > 3:
                rot.yaw = self._start_pos[actor_id][3]
            transform = carla.Transform(loc, rot)
            self._actor_configs[actor_id]["start_transform"] = transform
            tls = traffic_lights.get_tls(self.world, transform, sort=True)
            return tls[0][0]  #: Return the key (carla.TrafficLight object) of
            #: closest match

        if actor_type == "pedestrian":
            blueprints = self.world.get_blueprint_library().filter(
                "walker.pedestrian.*"
            )

        elif actor_type == "vehicle_4W":
            blueprints = self.world.get_blueprint_library().filter("vehicle")
            # Further filter down to 4-wheeled vehicles
            blueprints = [
                b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 4
            ]
            if self.exclude_hard_vehicles:
                blueprints = list(
                    filter(
                        lambda x: not (
                                x.id.endswith("microlino")
                                or x.id.endswith("carlacola")
                                or x.id.endswith("cybertruck")
                                or x.id.endswith("t2")
                                or x.id.endswith("sprinter")
                                or x.id.endswith("firetruck")
                                or x.id.endswith("ambulance")
                                or x.id.endswith("patrol_2021")
                                or x.id.endswith("patrol")
                                or x.id.endswith("t2")
                        ),
                        blueprints,
                    )
                )
        elif actor_type == "vehicle_2W":
            # blueprints = self.world.get_blueprint_library().filter("vehicle")
            # # Further filter down to 2-wheeled vehicles
            # blueprints = [
            #     b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 2
            # ]
            blueprints = [self.world.get_blueprint_library().find("vehicle.kawasaki.ninja"),
                          self.world.get_blueprint_library().find("vehicle.harley-davidson.low_rider"),
                          self.world.get_blueprint_library().find("vehicle.yamaha.yzf")]

        blueprint = random.choice(blueprints)
        loc = carla.Location(
            x=self._start_pos[actor_id][0],
            y=self._start_pos[actor_id][1],
            z=self._start_pos[actor_id][2],
        )
        rot = (
            self.world.get_map()
            .get_waypoint(loc, project_to_road=True)
            .transform.rotation
        )
        #: If yaw is provided in addition to (X, Y, Z), set yaw
        if len(self._start_pos[actor_id]) > 3:
            rot.yaw = self._start_pos[actor_id][3]
        transform = carla.Transform(loc, rot)
        self._actor_configs[actor_id]["start_transform"] = transform
        vehicle = None
        if actor_id == "ego":
            blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2017")
            blueprint.set_attribute('role_name', 'ego_vehicle')
        elif actor_id in ["car1", "car2", "car3"]:
            blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2017")
            blueprint.set_attribute('role_name', f'{actor_id}_vehicle')

            # 设置颜色红色
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', '255,0,0')
        print(actor_id, " blueprint:", blueprint)
        for retry in range(RETRIES_ON_ERROR):
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if self._sync_server:
                self.world.tick()
            if vehicle is not None and vehicle.get_location().z > 0.0:
                # Register it under traffic manager
                # Walker vehicle type does not have autopilot. Use walker controller ai
                if actor_type == "pedestrian":
                    # vehicle.set_simulate_physics(False)
                    pass
                else:
                    # A fresh PPO/BehaviorAgent vehicle is not registered with
                    # autopilot. Even set_autopilot(False) creates a native TM
                    # client in CARLA, so skip it on the explicit no-TM path.
                    if self._traffic_manager is not None:
                        vehicle.set_autopilot(False, self._traffic_manager.get_port())

                    # ✅ 为CARLA控制的车辆设置初始速度，使其与SUMO车辆速度匹配
                    # 配置项：可以在actor_configs中设置"initial_speed"参数
                    # 支持两种格式：
                    #   - 固定值: initial_speed = 15.0 (默认15 m/s，约54 km/h)
                    #   - 高斯分布: initial_speed = [mean, std] (例如 [12.5, 2.1])
                    initial_speed_config = self._actor_configs[actor_id].get("initial_speed", 15.0)

                    # 判断是固定值还是高斯分布
                    if isinstance(initial_speed_config, (list, tuple)):
                        # 高斯分布采样
                        mean, std = initial_speed_config[0], initial_speed_config[1]
                        initial_speed = np.random.normal(mean, std)
                        # 确保速度为正值
                        initial_speed = max(0.0, initial_speed)
                        initial_speed = np.clip(initial_speed, 8.33, 16.67)
                        print(
                            f"✅ {actor_id} 初始速度从高斯分布采样: μ={mean:.1f}, σ={std:.1f} → {initial_speed:.2f} m/s ({initial_speed * 3.6:.1f} km/h)")
                    else:
                        # 固定值
                        initial_speed = initial_speed_config
                        print(f"✅ {actor_id} 初始速度已设置: {initial_speed:.1f} m/s ({initial_speed * 3.6:.1f} km/h)")

                    # Do not apply the velocity here. More actors and sensors are
                    # spawned after this point and synchronous sensor warm-up calls
                    # world.tick(), which used to move early-spawned vehicles before
                    # reset() returned. Apply all sampled speeds together after the
                    # final reset-time tick instead.
                    self._initial_speed_by_actor[actor_id] = float(initial_speed)
                break
            # Wait to see if spawn area gets cleared before retrying
            time.sleep(0.5)
            print("spawn_actor: Retry#:{}/{}".format(retry + 1, RETRIES_ON_ERROR))
        if vehicle is None:
            # Request a spawn one last time possibly raising the error
            vehicle = self.world.spawn_actor(blueprint, transform)

        return vehicle

    def _apply_initial_actor_states(self):
        """Restore configured spawn poses and set initial linear velocities.

        CARLA's ``set_target_velocity`` writes the target linear velocity for
        the next physics step; it does not need a throttle-based acceleration
        warm-up. This method must therefore be called only after reset-time
        camera/LiDAR ticks have finished.
        """
        for actor_id, actor in self._actors.items():
            if actor is None or not actor.is_alive:
                continue
            actor_config = self._actor_configs.get(actor_id, {})
            if "vehicle" not in actor_config.get("type", ""):
                continue

            start_transform = actor_config.get("start_transform")
            if start_transform is None:
                continue

            # Eliminate any displacement introduced while reset waited for
            # sensors, then remove residual angular motion.
            actor.set_transform(start_transform)
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

            initial_speed = self._initial_speed_by_actor.get(actor_id, 0.0)
            forward_vector = start_transform.rotation.get_forward_vector()
            actor.set_target_velocity(carla.Vector3D(
                x=forward_vector.x * initial_speed,
                y=forward_vector.y * initial_speed,
                z=0.0,
            ))

    def _reset(self, clean_world=True):
        """Reset the state of the actors.
        A "soft" reset is performed in which the existing actors are destroyed
        and the necessary actors are spawned into the environment without
        affecting other aspects of the environment.
        If the "soft" reset fails, a "hard" reset is performed in which
        the environment's entire state is destroyed and a fresh instance of
        the server is created from scratch. Note that the "hard" reset is
        expected to take more time. In both of the reset modes ,the state/
        pose and configuration of all actors (including the sensor actors) are
        (re)initialized as per the actor configuration.

        Returns:
            dict: Dictionaries of observations for actors.

        Raises:
            RuntimeError: If spawning an actor at its initial state as per its'
            configuration fails (eg.: Due to collision with an existing object
            on that spot). This Error will be handled by the caller
            `self.reset()` which will perform a "hard" reset by creating
            a new server instance
        """

        self._done_dict["__all__"] = False
        self._lateral_distance = defaultdict(list)
        self._initial_speed_by_actor = {}
        # 🎯 重置 NGSIM 背景车辆轨迹误差累积变量（RMSE 计算）
        self._ep_bg_total_error = 0.0  # 误差平方和
        self._ep_bg_total_count = 0  # 有效车辆数
        self._ep_bg_candidate_frames = 0
        self._ep_bg_valid_frames = 0
        self._ep_bg_pending_frames = 0
        self._episode_cosim_available = bool(
            self._cosim_available and self._sumo_bridge is not None)
        self._episode_bg_metric_available = False
        self._cosim_last_status = None

        if clean_world:
            # Reject old LiDAR frames before changing background traffic too.
            self._sensor_callback_gate.begin_close()
            self._sensor_callback_gate.wait_for_idle(timeout=10.0)

        # ✅ 关键修复3：先重置SUMO桥接器，再清理CARLA world
        # 这样可以确保SUMO映射的车辆（包括NGSIM背景车辆）被正确清理
        if self._sumo_bridge is not None:
            try:
                print("🔄 重置SUMO桥接器（清理SUMO和映射车辆）...")
                with self._cleanup_rpc_timeout():
                    self._sumo_bridge.reset(carla_client=self._client)
                    self._check_cleanup_server()
                self._cosim_last_error = None
                self._cosim_last_status = {"ok": True, "phase": "reset"}
                print("✅ SUMO桥接器已重置")
            except Exception as e:
                self._episode_cosim_available = False
                self._episode_bg_metric_available = False
                self._cosim_last_error = str(e)
                self._cosim_last_status = {
                    "ok": False,
                    "phase": "reset",
                    "error": str(e),
                }
                print(f"⚠️ 重置SUMO桥接器时发生错误: {e}")
                # Do not turn a failed reset into an invalid training episode.
                raise

        if clean_world:
            self._clean_world()
            # 注意：物理引擎同步在_clean_world()结尾统一执行

        # A hard reset can follow an interrupted cleanup. Do not let a fresh
        # sensor stream start until callbacks from the old world have drained.
        self._sensor_callback_gate.wait_for_idle(timeout=10.0)
        self._sensor_callback_gate.reopen()

        weather_num = 0
        if "weather_distribution" in self._scenario_map:
            weather_num = random.choice(self._scenario_map["weather_distribution"])
            if weather_num not in WEATHERS:
                weather_num = 0

        self.world.set_weather(WEATHERS[weather_num])

        self._weather = [
            self.world.get_weather().cloudiness,
            self.world.get_weather().precipitation,
            self.world.get_weather().precipitation_deposits,
            self.world.get_weather().wind_intensity,
        ]

        for actor_id, actor_config in self._actor_configs.items():
            # if self._done_dict.get(actor_id, True):
            if True:
                self._measurements_file_dict[actor_id] = None
                self._episode_id_dict[actor_id] = datetime.today().strftime(
                    "%Y-%m-%d_%H-%M-%S_%f"
                )
                actor_config = self._actor_configs[actor_id]

                # Try to spawn actor (soft reset) or fail and reinitialize the server before get back here
                try:
                    self._actors[actor_id] = self._spawn_new_actor(actor_id)
                    actor_id_in_carla = self._actors[actor_id].id
                    self._actor_id_in_carla[actor_id_in_carla] = actor_id
                except RuntimeError as spawn_err:
                    del self._done_dict[actor_id]
                    # Chain the exception & re-raise to be handled by the caller `self.reset()`
                    raise spawn_err from RuntimeError(
                        "Unable to spawn actor:{}".format(actor_id)
                    )

                if self._env_config["enable_planner"]:
                    self._path_trackers[actor_id] = PathTracker(
                        self.world,
                        self.planner,
                        (
                            self._start_pos[actor_id][0],
                            self._start_pos[actor_id][1],
                            self._start_pos[actor_id][2],
                        ),
                        (
                            self._end_pos[actor_id][0],
                            self._end_pos[actor_id][1],
                            self._end_pos[actor_id][2],
                        ),
                        self._actors[actor_id],
                    )

                    # print("pathtracker ",actor_id)
                # if actor_config["auto_control"]:
                self._behavior_agents[actor_id] = BehaviorAgent(
                    self._actors[actor_id], behavior="aggressive", ignore_speed_limits=True)
                endloc = carla.Location(self._end_pos[actor_id][0], self._end_pos[actor_id][1],
                                        self._end_pos[actor_id][2])
                self._behavior_agents[actor_id].set_destination(endloc)

                if self._behavior_agents[actor_id] is not None:
                    self._initial_distance_to_end[actor_id] = get_distance_to_end(self._actors[actor_id].get_location(),
                                                                                  self._behavior_agents[
                                                                                      actor_id]._local_planner._waypoints_queue)

                # Spawn collision and lane sensors if necessary
                if actor_config["collision_sensor"] == "on":
                    collision_sensor = CollisionSensor(self._actors[actor_id], hud=0, ego=self._actors["ego"])
                    self._collisions.update({actor_id: collision_sensor})

                if actor_config["lane_sensor"] == "on":
                    lane_sensor = LaneInvasionSensor(self._actors[actor_id], 0)
                    self._lane_invasions.update({actor_id: lane_sensor})

                # Spawn cameras
                pygame.font.init()  # for HUD
                hud = HUD(self._env_config["x_res"], self._env_config["y_res"])
                camera_manager = CameraManager(self._actors[actor_id], hud)
                if actor_config["log_images"]:
                    # TODO: The recording option should be part of config
                    # 1: Save to disk during runtime
                    # 2: save to memory first, dump to disk on exit
                    camera_manager.set_recording_option(1)

                # in CameraManger's._sensors
                camera_type = self._actor_configs[actor_id]["camera_type"]
                camera_pos = getattr(
                    self._actor_configs[actor_id], "camera_position", 0
                )
                camera_types = [ct.name for ct in CAMERA_TYPES]
                assert (
                        camera_type in camera_types
                ), "Camera type `{}` not available. Choose in {}.".format(
                    camera_type, camera_types
                )
                camera_manager.set_sensor(
                    CAMERA_TYPES[camera_type].value - 1, int(camera_pos), notify=False
                )
                assert camera_manager.sensor.is_listening
                self._cameras.update({actor_id: camera_manager})

                if actor_config.get("lidar_sensor", False):
                    lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
                    lidar_bp.set_attribute('channels', '16')
                    lidar_bp.set_attribute('points_per_second', '10000')
                    lidar_bp.set_attribute('range', str(self._lidar_range))
                    lidar_bp.set_attribute('rotation_frequency', '20')
                    lidar_bp.set_attribute('horizontal_fov', '360')
                    transform = carla.Transform(carla.Location(x=0, y=0, z=2.5),
                                                carla.Rotation(pitch=0, yaw=0, roll=0))
                    lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=self._actors[actor_id])
                    self._radars[actor_id] = lidar
                    print("_reset actor_id:", actor_id, "lidar:", lidar)

                # Manual Control
                if actor_config["manual_control"]:
                    self._control_clock = pygame.time.Clock()

                    self._manual_controller = KeyboardControl(
                        self, actor_config["auto_control"]
                    )
                    self._manual_controller.actor_id = actor_id

                    self.world.on_tick(self._hud.on_world_tick)
                    self._manual_control_camera_manager = CameraManager(
                        self._actors[actor_id], self._hud
                    )
                    self._manual_control_camera_manager.set_sensor(
                        CAMERA_TYPES["rgb"].value - 1, pos=2, notify=False
                    )

                self._start_coord.update(
                    {
                        actor_id: [
                            self._start_pos[actor_id][0] // 100,
                            self._start_pos[actor_id][1] // 100,
                        ]
                    }
                )
                self._end_coord.update(
                    {
                        actor_id: [
                            self._end_pos[actor_id][0] // 100,
                            self._end_pos[actor_id][1] // 100,
                        ]
                    }
                )

                print(
                    "Actor: {} start_pos_xyz(coordID): {} ({}), "
                    "end_pos_xyz(coordID) {} ({})".format(
                        actor_id,
                        self._start_pos[actor_id],
                        self._start_coord[actor_id],
                        self._end_pos[actor_id],
                        self._end_coord[actor_id],
                    )
                )
        if self._external_ego_controller is not None:
            self._external_ego_controller.bind_episode(self)
        time.sleep(2)

        print("New episode initialized with actors:{}".format(self._actors.keys()))

        # set red trafficlight when egovehicle is spawned
        if self._actors["ego"].is_at_traffic_light():
            traffic_light = self._actors["ego"].get_traffic_light()
            # if traffic_light.get_state() == carla.TrafficLightState.Green:
            #     traffic_light.set_state(carla.TrafficLightState.Red)
            # traffic_light.set_red_time(1)
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)

        self._npc_vehicles, self._npc_pedestrians = apply_traffic(
            self.world,
            self._traffic_manager,
            self._scenario_map.get("num_vehicles", 0),
            self._scenario_map.get("num_pedestrians", 0),
        )

        # SUMO桥接器已在_reset()开头重置，这里不需要重复调用
        print("✅ 新回合初始化完成，准备开始仿真")

    def _load_scenario(self, scenario_parameter):
        self._scenario_map = {}
        # If config contains a single scenario, then use it,
        # if it's an array of scenarios,randomly choose one and init
        if isinstance(scenario_parameter, dict):
            scenario = scenario_parameter
        else:  # instance array of dict
            scenario = random.choice(scenario_parameter)

        # if map_name not in (town, "OpenDriveMap"):  TODO
        #     print("The CARLA server uses the wrong map: {}".format(map_name))
        #     print("This scenario requires to use map: {}".format(town))
        #     return False

        self._scenario_map = scenario
        self._validate_traffic_manager_configuration()
        for actor_id, actor in scenario["actors"].items():
            if isinstance(actor["start"], int):
                self._start_pos[actor_id] = self.pos_coor_map[str(actor["start"])]
            else:
                self._start_pos[actor_id] = actor["start"]

            if isinstance(actor["end"], int):
                self._end_pos[actor_id] = self.pos_coor_map[str(actor["end"])]
            else:
                self._end_pos[actor_id] = actor["end"]

    def _decode_obs(self, actor_id, obs):
        """Decode actor observation into original image reversing the pre_process() operation.
        Args:
            actor_id (str): Actor identifier
            obs (dict): Properly encoded observation data of an actor

        Returns:
            image (array): Original actor camera view
        """
        if self._env_config["using_imitation_model"]:
            # return self._prev_image[actor_id]
            obs = self._prev_image[actor_id]
            obs = np.multiply(obs.astype(np.float32), 255.0)
            # print(obs.shape)
            return obs.swapaxes(0, 1)
        if self._actor_configs[actor_id]["send_measurements"]:
            obs = obs[0]
        # Reverse the processing operation
        if self._actor_configs[actor_id]["use_depth_camera"]:
            img = np.tile(obs.swapaxes(0, 1), 3)
        else:
            # print("obs shape",obs.shape)
            img = obs.swapaxes(0, 1)
            # print("img render shape",img.shape)
        img = np.multiply(img.astype(np.float32), 255.0)
        return img

    def _encode_obs(self, actor_id, image, lidar, py_measurements):
        """Encode sensor and measurements into obs based on state-space config.

        Args:
            actor_id (str): Actor identifier
            image (array): original unprocessed image
            py_measurements (dict): measurement file

        Returns:
            obs (dict): properly encoded observation data for each actor
        """
        assert self._framestack in [1, 2]
        # Apply preprocessing
        config = self._actor_configs[actor_id]
        image = preprocess_image(image, config)
        # Stack frames
        prev_image = self._prev_image.get(actor_id, None)
        self._prev_image[actor_id] = image
        if prev_image is None:
            prev_image = image
        if self._framestack == 2:
            # image = np.concatenate([prev_image, image], axis=2)
            image = np.concatenate([prev_image, image], axis=2)
        # Structure the observation
        if not self._actor_configs[actor_id]["send_measurements"]:
            return image
        ego_location = self._actors["ego"].get_location()
        location = self._actors[actor_id].get_location()
        goal = self._end_pos[actor_id]
        ego_speed = self._actors["ego"].get_velocity()
        speed = self._actors[actor_id].get_velocity()

        rel_loc = np.array([ego_location.x - location.x, ego_location.y - location.y])
        rel_speed = np.array([ego_speed.x - speed.x, ego_speed.y - speed.y])
        rel_loc = np.clip(rel_loc / 5, -20, 20)
        rel_speed = np.clip(rel_speed, -20, 20)

        speed = np.array([speed.x, speed.y])
        norm = np.linalg.norm(speed)
        if (norm != 0):
            speed = speed / norm
        # command_index = COMMAND_ORDINAL[py_measurements["next_command"]]
        # command_onehot = np.zeros(5)
        # command_onehot[command_index]=1
        rel_to_goal = np.array([goal[0] - location.x, goal[1] - location.y])
        norm = np.linalg.norm(rel_to_goal)
        if (norm != 0):
            rel_to_goal /= norm
        command_onehot = np.zeros(5)
        command_onehot[0] = rel_to_goal[0]
        command_onehot[1] = rel_to_goal[1]
        if py_measurements.get("autocontrol", None) is not None:
            control = py_measurements["autocontrol"]
            command_onehot[2] = control["steer"]
            command_onehot[3] = control["throttle"]
            command_onehot[4] = control["brake"]
        # norm = np.linalg.norm(rel_speed)
        #
        # if (norm != 0):
        #     rel_speed = rel_speed / norm
        # speed = math.sqrt(pow(self._actors[actor_id].get_velocity().x,2) + pow(self._actors[actor_id].get_velocity().y,2) + pow(self._actors[actor_id].get_velocity().z,2))
        # direction_to_ego = np.array([ego_location.x-location.x, ego_location.y-location.y])
        # norm = np.linalg.norm(direction_to_ego)
        # if(norm != 0):
        #     direction_to_ego = direction_to_ego/norm
        # distance_to_ego = calculate_distance(location.x,location.y,location.z, ego_location.x,ego_location.y,ego_location.z)
        if self._env_config["using_imitation_model"]:
            if self._use_custom_multimodal_encoder:
                with self._multimodal_encoder_lock:
                    feature = self._multimodal_feature_extractor.encode_image(
                        image
                    )
            else:
                # Legacy image encoder path remains unchanged by default.
                image = cv2.resize(
                    image, (88, 200), interpolation=cv2.INTER_AREA
                )
                feature = self._image_agent.compute_feature(image)
                feature = np.clip(feature, 0, 1)
            # lidar_feature = self._lidar_features.get(actor_id, None)
            obs = (
                feature,
                # direction, relative speed and relative position, and command
                [speed[0], speed[1], rel_speed[0], rel_speed[1], rel_loc[0], rel_loc[1], command_onehot[0],
                 command_onehot[1], command_onehot[2], command_onehot[3], command_onehot[4]],
                # lidar_feature,
                lidar
            )
        else:
            # lidar_feature = self._lidar_features.get(actor_id, None)
            obs = (
                image,
                # direction, relative speed and relative position, and command
                [speed[0], speed[1], rel_speed[0], rel_speed[1], rel_loc[0], rel_loc[1], command_onehot[0],
                 command_onehot[1], command_onehot[2], command_onehot[3], command_onehot[4]],
                # lidar_feature,
                lidar
            )

        self._last_obs = obs
        return obs

    def _step_sumo_and_collect_metrics(self):
        """Advance SUMO once and propagate metric validity without aborting CARLA."""
        if self._sumo_bridge is None:
            return

        try:
            status = self._sumo_bridge.step_and_sync_from_carla(self._actors)
        except Exception as exc:
            status = {
                "ok": False,
                "error": "SUMO synchronization raised an exception: {}".format(exc),
            }

        if not isinstance(status, dict):
            status = {
                "ok": False,
                "error": "SUMO bridge returned no validity status",
            }
        self._cosim_last_status = status

        if not status.get("ok", False):
            # CARLA training/inference continues, but co-simulation-dependent
            # metrics from this episode must not be reported as valid results.
            self._episode_cosim_available = False
            self._episode_bg_metric_available = False
            self._cosim_last_error = status.get("error") or str(status)
            if self._verbose:
                print("SUMO同步局部失败，本回合协同指标记为无效: {}".format(
                    self._cosim_last_error))
            return

        ngsim_metrics = self._sumo_bridge.get_current_deviation_metrics(
            require_carla_actor=True)
        active_count = int(ngsim_metrics.get("active_count", 0))
        has_background = int(status.get("ngsim_background_count", 0)) > 0
        pending_spawns = (int(status.get("sumo_to_carla_pending", 0))
                          + int(status.get("sumo_to_carla_spawn_skipped", 0)))
        if has_background:
            self._ep_bg_candidate_frames += 1
        if pending_spawns:
            self._ep_bg_pending_frames += 1
        # The bridge filters measurements per vehicle. Missing background
        # spawns do not discard measurements from other successfully synced cars.
        if self._episode_cosim_available and active_count > 0:
            frame_error = float(ngsim_metrics.get("frame_total_error", 0.0))
            if math.isfinite(frame_error) and frame_error >= 0.0:
                self._ep_bg_total_error += frame_error
                self._ep_bg_total_count += active_count
                self._episode_bg_metric_available = True
                self._ep_bg_valid_frames += 1
            else:
                self._episode_cosim_available = False
                self._episode_bg_metric_available = False
                self._cosim_last_error = "NGSIM frame error is non-finite or negative"

        if self._verbose:
            sumo_info = self._sumo_bridge.get_sumo_vehicles_info()
            if sumo_info:
                print(f"SUMO 中有 {len(sumo_info)} 輛車輛正在仿真")

    def step(self, action_dict):
        """Execute one environment step for the specified actors.

        Executes the provided action for the corresponding actors in the
        environment and returns the resulting environment observation, reward,
        done and info (measurements) for each of the actors. The step is
        performed asynchronously i.e. only for the specified actors and not
        necessarily for all actors in the environment.

        Args:
            action_dict (dict): Actions to be executed for each actor. Keys are
                agent_id strings, values are corresponding actions.

        Returns
            obs (dict): Observations for each actor.
            rewards (dict): Reward values for each actor. None for first step
            dones (dict): Done values for each actor. Special key "__all__" is
            set when all actors are done and the env terminates
            info (dict): Info for each actor.

        Raises
            RuntimeError: If `step(...)` is called before calling `reset()`
            ValueError: If `action_dict` is not a dictionary of actions
            ValueError: If `action_dict` contains actions for nonexistent actor
        """
        if (not self._server_process) or (not self._client):
            raise RuntimeError("Cannot call step(...) before calling reset()")

        assert len(self._actors), (
            "No actors exist in the environment. Either"
            " the environment was not properly "
            "initialized using`reset()` or all the "
            "actors have exited. Cannot execute `step()`"
        )

        if not isinstance(action_dict, dict):
            raise ValueError(
                "`step(action_dict)` expected dict of actions. "
                "Got {}".format(type(action_dict))
            )
        # Make sure the action_dict contains actions only for actors that
        # exist in the environment
        if not set(action_dict).issubset(set(self._actors)):
            raise ValueError(
                "Cannot execute actions for non-existent actors."
                " Received unexpected actor ids:{}".format(
                    set(action_dict).difference(set(self._actors))
                )
            )

        try:
            obs_dict = {}
            reward_dict = {}
            info_dict = {}

            # ============================================================================
            # 🎯 关键修复：三阶段步进，每个训练步只tick一次
            # ============================================================================

            # --- 第一阶段：应用所有车辆的动作 ---
            # 此时物理世界还没动
            controls = {}
            for actor_id, action in action_dict.items():
                control = self._apply_actor_control(actor_id, action)
                controls[actor_id] = control

            # --- 第二阶段：世界推演 (单次Tick) ---
            # ✅ 关键修改：整个回合只 Tick 一次！
            if self._sync_server:
                self.world.tick()
                # 同步 SUMO (只走一步)
                self._step_sumo_and_collect_metrics()
            else:
                # 非同步模式下也進行 SUMO 同步
                self._step_sumo_and_collect_metrics()

            # --- 第三阶段：读取所有车辆的结果 ---
            # 此时物理世界已经更新到了下一帧
            for actor_id, action in action_dict.items():
                obs, reward, done, info = self._process_actor_result(
                    actor_id, action, controls[actor_id]
                )
                obs_dict[actor_id] = obs
                reward_dict[actor_id] = reward
                if not self._done_dict.get(actor_id, False):
                    self._done_dict[actor_id] = done
                info_dict[actor_id] = info

            # ============================================================================
            # ✅ 修改：以 Ego 状态为核心的重置逻辑
            # ============================================================================

            # 1. 检查 Ego 是否结束（包含：碰撞、超时、到达终点）
            # 注意：StraightRoads3NPC.py 中 ego 的 early_terminate_on_collision 必须为 True
            if self._done_dict.get("ego", False):
                self._done_dict["__all__"] = True

            # 2. 保底逻辑：如果 Ego 没结束，但其他所有 NPC 竟然都结束了（极少见），也重置
            elif sum(self._done_dict.values()) >= len(self._actors):
                self._done_dict["__all__"] = True

            # 3. 否则继续
            else:
                self._done_dict["__all__"] = False

            if (self._done_dict["__all__"]):
                # 🎯 计算 episode 级别的 NGSIM 背景车辆平均轨迹误差（标准 RMSE）
                # RMSE = √(Σ(偏差²) / N)
                bg_lde_available = bool(
                    self._episode_cosim_available
                    and self._episode_bg_metric_available
                    and self._ep_bg_total_count > 0
                )
                if bg_lde_available:
                    avg_ep_bg_lde = math.sqrt(self._ep_bg_total_error / self._ep_bg_total_count)
                    if self._verbose:
                        print(f"📊 全程NGSIM轨迹回放统计:")
                        print(f"   - 总误差平方和: {self._ep_bg_total_error:.2f}")
                        print(f"   - 总有效测量点数: {self._ep_bg_total_count}")
                        print(f"   - 全程平均RMSE: {avg_ep_bg_lde:.4f}m")
                else:
                    avg_ep_bg_lde = float("nan")
                    if self._verbose:
                        print("⚠️ 本episode无有效NGSIM轨迹误差数据，BG_LDE记为NaN")

                # 🎯 将 bg_lde 存入 info_dict（所有 actor 共享该指标）
                for actor_id in self._actors.keys():
                    if actor_id not in info_dict:
                        info_dict[actor_id] = {}  # 确保键存在
                    info_dict[actor_id]["bg_lde"] = avg_ep_bg_lde
                    info_dict[actor_id]["bg_lde_available"] = bg_lde_available
                    info_dict[actor_id]["cosim_available"] = self._episode_cosim_available
                    info_dict[actor_id]["bg_sample_count"] = self._ep_bg_total_count
                    info_dict[actor_id]["bg_candidate_frames"] = self._ep_bg_candidate_frames
                    info_dict[actor_id]["bg_valid_frames"] = self._ep_bg_valid_frames
                    info_dict[actor_id]["bg_pending_frames"] = self._ep_bg_pending_frames
                    info_dict[actor_id]["bg_metric_coverage"] = (
                        float(self._ep_bg_valid_frames) / self._ep_bg_candidate_frames
                        if self._ep_bg_candidate_frames else 0.0)
                    info_dict[actor_id]["cosim_error"] = self._cosim_last_error
                    info_dict[actor_id]["cosim_status"] = self._cosim_last_status

                for actor_id in self._actors.keys():
                    end_dist = get_distance_to_end(
                        self._actors[actor_id].get_location(),
                        self._behavior_agents[actor_id]._local_planner._waypoints_queue
                    )
                    self._end_distance_to_end[actor_id] = end_dist

                    initial_dist = self._initial_distance_to_end.get(actor_id, 0.0)
                    if initial_dist > 1e-6:
                        self._rc_normalized[actor_id] = max(
                            0.0, (initial_dist - end_dist) / initial_dist
                        )
                    else:
                        self._rc_normalized[actor_id] = 0.0

                    info_dict[actor_id]["rc_normalized"] = self._rc_normalized[actor_id]
                    self._rc_segment_initial[actor_id] = None
                if self._env_config.get("cummulated_reward", False) == True:
                    cummulate_reward = 0
                    for actor_id, total_reward in self._total_reward.items():
                        if actor_id != "ego":
                            cummulate_reward += total_reward
                    file_name = self._env_config.get("cummulated_reward_filename")
                    self._reward_episode_global += 1
                    with open(file_name, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([self._reward_episode_global, cummulate_reward])
                        # for row in zip(self._episode, cummulate_reward):
                        #     writer.writerow(row)
                    print("write into cummulate_reward successfully!")
                    # print(self._total_reward)
                    # for actor_id, total_reward in self._total_reward.items():
                    #     self._episode_reward[actor_id].append(total_reward)
                    # print("episode_reward:",self._episode_reward)

            # Find if any actor's config has render=True & render only for
            # that actor. NOTE: with async server stepping, enabling rendering
            # affects the step time & therefore MAX_STEPS needs adjustments
            render_required = [
                k for k, v in self._actor_configs.items() if v.get("render", False)
            ]
            if render_required:
                images = {
                    k: self._decode_obs(k, v)
                    for k, v in obs_dict.items()
                    if self._actor_configs[k]["render"]
                }

                Render.multi_view_render(images, self._camera_poses)
                if self._manual_controller is None:
                    Render.dummy_event_handler()

            return obs_dict, reward_dict, self._done_dict, info_dict
        except Exception:
            print(
                "Error during step, terminating episode early.", traceback.format_exc()
            )
            self._clear_server_state()
            if self._actor_configs.get("ego", {}).get("required_external_controller") == "wor":
                raise

    def _record_behavior_control(self, actor_id, control):
        """Append pre-tick diagnostics only for actually applied agent control.

        This is a separate trace, not the episode reward/checkpoint ledger.
        Process/server-specific files also separate retries and parallel envs.
        """
        diagnostics = self._behavior_agents[actor_id].last_control_diagnostics
        reward_file = self._env_config.get("cummulated_reward_filename")
        default_root = (os.path.dirname(os.path.abspath(reward_file))
                        if reward_file else CARLA_OUT_PATH)
        output_dir = self._env_config.get(
            "behavior_control_log_dir",
            os.path.join(default_root, "behavior_control_diagnostics") if default_root else None)
        if not output_dir:
            return
        fields = ["episode_id", "step", "actor_id", "carla_actor_id",
                  "controller_version", "speed_kmh", "road_speed_limit_kmh",
                  "ignore_speed_limits", "lead_actor_id", "lead_speed_kmh",
                  "gap_m", "ttc_s", "target_speed_kmh", "reason",
                  "throttle", "brake", "steer"]
        row = {key: diagnostics.get(key) for key in fields}
        row.update({"episode_id": self._episode_id_dict[actor_id],
                    "step": self._num_steps[actor_id] + 1,
                    "actor_id": actor_id, "carla_actor_id": self._actors[actor_id].id,
                    "throttle": float(control.throttle), "brake": float(control.brake),
                    "steer": float(control.steer)})
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, "behavior_control_{}_{}.csv".format(
                os.getpid(), self._server_port))
            needs_header = not os.path.isfile(filename) or os.path.getsize(filename) == 0
            with open(filename, "a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            if not getattr(self, "_behavior_control_log_warning", False):
                logger.warning("Could not write behavior control diagnostics", exc_info=True)
                self._behavior_control_log_warning = True

    def _apply_actor_control(self, actor_id, action):
        """
        只负责应用控制命令，不执行world.tick()

        Args:
            actor_id(str): Actor identifier
            action: Actions to be executed for the actor.

        Returns:
            control: BehaviorAgent生成的控制命令（用于后续记录）
        """
        external_control = self._external_control_overrides.pop(actor_id, None)
        external_source = self._external_control_sources.pop(actor_id, None)
        required = self._actor_configs[actor_id].get("required_external_controller")
        if required and (external_control is None or external_source != required):
            raise RuntimeError("{} requires {} control every step; fallback is forbidden".format(actor_id, required))
        if external_control is not None:
            if required == "wor":
                self._advance_external_route(actor_id)
            self._last_actor_control_sources[actor_id] = external_source or "external"
            external_control.manual_gear_shift = False
            self._actors[actor_id].apply_control(external_control)
            behavior_agent = self._behavior_agents.get(actor_id)
            if behavior_agent is not None:
                self._target_waypoint[actor_id] = (
                    behavior_agent._local_planner.target_waypoint)
                if self._target_waypoint[actor_id] is not None:
                    self._lateral_distance[actor_id].append(
                        calculate_lateral_distance(
                            self._target_waypoint[actor_id],
                            self._actors[actor_id].get_location()))
            return external_control

        if self._discrete_actions:
            # 🎯 MultiDiscrete 动作解析: action = [longitudinal_idx, lateral_idx]
            val_long = LONGITUDINAL_ACTIONS[int(action[0])]  # 纵向控制值 [-1.0, 1.0]
            val_lat = LATERAL_ACTIONS[int(action[1])]  # 横向控制值 [-0.7, 0.7]

            throttle = float(np.clip(val_long, 0, 1))  # 正值 -> 油门
            brake = float(np.abs(np.clip(val_long, -1, 0)))  # 负值 -> 刹车
            steer = float(np.clip(val_lat, -1, 1))  # 横向 -> 转向
        else:
            # 连续动作空间
            assert len(action) == 2, "Invalid action {}".format(action)
            if self._squash_action_logits:
                forward = 2 * float(sigmoid(action[0]) - 0.5)
                throttle = float(np.clip(forward, 0, 1))
                brake = float(np.abs(np.clip(forward, -1, 0)))
                steer = 2 * float(sigmoid(action[1]) - 0.5)
            else:
                throttle = float(np.clip(action[0], 0, 1))
                brake = float(np.abs(np.clip(action[0], -1, 0)))
                steer = float(np.clip(action[1], -1, 1))

        reverse = False
        hand_brake = False

        if self._verbose:
            print("steer", steer, "throttle", throttle, "brake", brake, "reverse", reverse)

        config = self._actor_configs[actor_id]

        def _distance_to_end_for_actor(_actor_id):
            return get_distance_to_end(
                self._actors[_actor_id].get_location(),
                self._behavior_agents[_actor_id]._local_planner._waypoints_queue
            )

        def _accumulate_rc_segment(_actor_id):
            return

        # BehaviorAgent初始化和控制生成
        if self._behavior_agents[actor_id] is None:
            self._behavior_agents[actor_id] = BehaviorAgent(
                self._actors[actor_id], behavior="aggressive", ignore_speed_limits=True)
            endloc = carla.Location(self._end_pos[actor_id][0],
                                    self._end_pos[actor_id][1],
                                    self._end_pos[actor_id][2])
            self._behavior_agents[actor_id].set_destination(endloc)
            initial_dist = _distance_to_end_for_actor(actor_id)
            self._initial_distance_to_end[actor_id] = initial_dist
            self._rc_segment_initial[actor_id] = initial_dist
            self._rc_progress_sum[actor_id] = 0.0
            self._rc_segment_count[actor_id] = 0
            self._rc_normalized[actor_id] = 0.0

        if self._behavior_agents[actor_id].done():
            if self._time_to_goal[actor_id] is None:
                self._time_to_goal[actor_id] = self._num_steps[actor_id]
            print("ego target reached, searching another endpoint")
            self._behavior_agents[actor_id].set_destination(random.choice(self.world.get_map().get_spawn_points()).location)
            self._rc_segment_initial[actor_id] = _distance_to_end_for_actor(actor_id)

        control = self._behavior_agents[actor_id].run_step()
        control.manual_gear_shift = False

        # 计算侧向距离到规划航点
        self._target_waypoint[actor_id] = self._behavior_agents[actor_id]._local_planner.target_waypoint
        draw_waypoints(self.world, [self._target_waypoint[actor_id]], 1.0)
        self._lateral_distance[actor_id].append(
            calculate_lateral_distance(self._target_waypoint[actor_id], self._actors[actor_id].get_location()))

        # 应用控制命令
        if config["manual_control"]:
            self._control_clock.tick(60)
            self._manual_control_camera_manager._hud.tick(
                self.world,
                self._actors[actor_id],
                self._collisions[actor_id],
                self._control_clock,
            )
            self._manual_controller.parse_events(self, self._control_clock)
            self._manual_control_camera_manager.render(
                Render.get_screen(), self._manual_control_render_pose
            )
            self._manual_control_camera_manager._hud.render(
                Render.get_screen(), self._manual_control_render_pose
            )
            pygame.display.flip()
        elif config["auto_control"]:
            self._actors[actor_id].apply_control(control)
            self._last_actor_control_sources[actor_id] = "behavior"
            self._record_behavior_control(actor_id, control)
        else:
            self._last_actor_control_sources[actor_id] = "ppo"
            if config.get("frame_skip", False) == False:
                agent_type = config.get("type", "vehicle")
                if agent_type == "pedestrian":
                    rotation = self._actors[actor_id].get_transform().rotation
                    rotation.yaw += steer * 180.0
                    x_dir = math.cos(math.radians(rotation.yaw))
                    y_dir = math.sin(math.radians(rotation.yaw))
                    self._actors[actor_id].apply_control(
                        carla.WalkerControl(
                            speed=3.0 * throttle,
                            direction=carla.Vector3D(x_dir, y_dir, 0.0),
                        )
                    )
                elif "vehicle" in agent_type:
                    self._actors[actor_id].apply_control(
                        carla.VehicleControl(
                            throttle=throttle,
                            steer=steer,
                            brake=brake,
                            hand_brake=hand_brake,
                            reverse=reverse,
                        )
                    )
            else:  # frame_skip mode
                if self._num_steps[actor_id] % 10 < 5:
                    agent_type = config.get("type", "vehicle")
                    if agent_type == "pedestrian":
                        rotation = self._actors[actor_id].get_transform().rotation
                        rotation.yaw += steer * 180.0
                        x_dir = math.cos(math.radians(rotation.yaw))
                        y_dir = math.sin(math.radians(rotation.yaw))
                        self._actors[actor_id].apply_control(
                            carla.WalkerControl(
                                speed=3.0 * throttle,
                                direction=carla.Vector3D(x_dir, y_dir, 0.0),
                            )
                        )
                    elif "vehicle" in agent_type:
                        self._actors[actor_id].apply_control(
                            carla.VehicleControl(
                                throttle=throttle,
                                steer=steer,
                                brake=brake,
                                hand_brake=hand_brake,
                                reverse=reverse,
                            )
                        )
                else:
                    self._actors[actor_id].apply_control(control)

        # 返回控制命令用于后续记录
        return control

    def _advance_external_route(self, actor_id):
        """Update the same local-planner queue without computing PID/agent control.

        This is the purge rule in CARLA LocalPlanner.run_step. Keeping the
        original queue and distance formula preserves RC/LDE bookkeeping.
        WOR alone produces throttle/brake/steer; no agent/PID output is used.
        """
        planner = self._behavior_agents[actor_id]._local_planner
        vehicle = self._actors[actor_id]
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        planner._min_distance = planner._base_min_distance + 0.5 * speed
        count = 0
        for waypoint, _ in planner._waypoints_queue:
            threshold = 1 if len(planner._waypoints_queue) - count == 1 else planner._min_distance
            if location.distance(waypoint.transform.location) < threshold:
                count += 1
            else:
                break
        for _ in range(count):
            planner._waypoints_queue.popleft()
        if planner._waypoints_queue:
            planner.target_waypoint, planner.target_road_option = planner._waypoints_queue[0]
        elif self._time_to_goal[actor_id] is None:
            self._time_to_goal[actor_id] = self._num_steps[actor_id]

    def set_external_actor_control(self, actor_id, control, source="external"):
        """Override one actor's control for the next environment transition.

        This narrow hook preserves the regular action interface while allowing
        a reference controller to replay the exact recorded NPC actions and
        replace only AVUT control.  The override is consumed once in
        ``_apply_actor_control``.
        """

        if actor_id not in self._actors:
            raise KeyError("Unknown actor_id: {}".format(actor_id))
        values = [control.throttle, control.steer, control.brake]
        if not np.isfinite(values).all():
            raise ValueError("External actor control is non-finite")
        self._external_control_overrides[actor_id] = control
        self._external_control_sources[actor_id] = source

    def _process_actor_result(self, actor_id, action, control):
        """
        读取观测数据并计算奖励（在world.tick()之后调用）

        Args:
            actor_id(str): Actor identifier
            action: Actions that were executed for the actor.
            control: BehaviorAgent生成的控制命令

        Returns:
            obs: Encoded observation
            reward: Reward value
            done: Done flag
            info: Info dictionary
        """
        config = self._actor_configs[actor_id]

        # 读取观测
        py_measurements = self._read_observation(actor_id)
        if self._verbose:
            print("Next command", py_measurements["next_command"])

        # 存储上一个动作
        self._previous_actions[actor_id] = action
        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action

        py_measurements["autocontrol"] = {
            "steer": control.steer,
            "throttle": control.throttle,
            "brake": control.brake,
        }
        py_measurements["control_source"] = self._last_actor_control_sources.get(actor_id, "unknown")

        # 计算Done
        current_step = self._num_steps[actor_id] + 1
        py_measurements["step"] = current_step
        done = (
                current_step >= self._scenario_map["max_steps"]
                or py_measurements["next_command"] == "REACH_GOAL"
                or (
                        config["early_terminate_on_collision"]
                        and collided_done(py_measurements)
                )
                or (
                        self._env_config.get("terminate_on_collision_to_ego", False)
                        and py_measurements["collision_to_ego"] > 0
                )
        )
        py_measurements["done"] = done

        # 计算奖励
        flag = config["reward_function"]
        ego_curr_measurement = self._read_observation("ego")
        reward = self._reward_policy.compute_reward(
            self._prev_measurement[actor_id],
            py_measurements,
            ego_prev_measurement=self._prev_measurement["ego"],
            ego_curr_measurement=ego_curr_measurement,
            flag=flag
        )

        self._previous_rewards[actor_id] = reward
        if self._total_reward[actor_id] is None:
            self._total_reward[actor_id] = reward
        else:
            self._total_reward[actor_id] += reward

        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self._total_reward[actor_id]

        # 更新状态
        self._prev_measurement[actor_id] = py_measurements
        self._num_steps[actor_id] = current_step

        # 日志记录
        if config["log_measurements"] and CARLA_OUT_PATH:
            if not self._measurements_file_dict[actor_id]:
                self._measurements_file_dict[actor_id] = open(
                    os.path.join(
                        CARLA_OUT_PATH,
                        "measurements_{}.json".format(self._episode_id_dict[actor_id]),
                    ),
                    "w",
                )
            self._measurements_file_dict[actor_id].write(json.dumps(py_measurements))
            self._measurements_file_dict[actor_id].write("\n")
            if done:
                self._measurements_file_dict[actor_id].close()
                self._measurements_file_dict[actor_id] = None

        # 编码观测
        original_image = self._cameras[actor_id].image
        with self._lidar_lock:
            lidar_feature = self._lidar_features.get(
                actor_id, np.zeros(256, dtype=np.float32)).copy()

        return (
            self._encode_obs(actor_id, original_image, lidar_feature, py_measurements),
            reward,
            done,
            py_measurements,
        )

    def _read_observation(self, actor_id):
        """Read observation and return measurement.

        Args:
            actor_id (str): Actor identifier

        Returns:
            dict: measurement data.

        """
        cur = self._actors[actor_id]
        cur_config = self._actor_configs[actor_id]
        planner_enabled = cur_config["enable_planner"]
        planner_enabled = False
        next_command = "LANE_FOLLOW"
        if planner_enabled:
            dist = self._path_trackers[actor_id].get_distance_to_end()
            orientation_diff = self._path_trackers[
                actor_id
            ].get_orientation_difference_to_end_in_radians()
            commands = self.planner.plan_route(
                (cur.get_location().x, cur.get_location().y),
                (self._end_pos[actor_id][0], self._end_pos[actor_id][1]),
            )
            if len(commands) > 0:
                next_command = ROAD_OPTION_TO_COMMANDS_MAPPING.get(
                    commands[0], "LANE_FOLLOW"
                )
            elif (
                    dist <= DISTANCE_TO_GOAL_THRESHOLD
                    and orientation_diff <= ORIENTATION_TO_GOAL_THRESHOLD
            ):
                next_command = "REACH_GOAL"
            else:
                next_command = "LANE_FOLLOW"

            # DEBUG
            # self.path_trackers[actor_id].draw()
        else:
            next_command = "LANE_FOLLOW"

        collision_vehicles = self._collisions[actor_id].collision_vehicles
        collision_pedestrians = self._collisions[actor_id].collision_pedestrians
        collision_other = self._collisions[actor_id].collision_other
        intersection_otherlane = self._lane_invasions[actor_id].offlane
        intersection_offroad = self._lane_invasions[actor_id].offroad
        collision_to_ego = self._collisions[actor_id].collision_to_ego

        distance_to_goal = 0.0
        if planner_enabled:
            distance_to_goal = self._path_trackers[actor_id].get_distance_to_end()
        else:
            distance_to_goal = float(
                np.linalg.norm(
                    [
                        self._actors[actor_id].get_location().x
                        - self._end_pos[actor_id][0],
                        self._actors[actor_id].get_location().y
                        - self._end_pos[actor_id][1],
                    ]
                )
            )
        ego_location = self._actors["ego"].get_location()
        location = self._actors[actor_id].get_location()
        distance_to_ego = calculate_distance(location.x, location.y, location.z, ego_location.x, ego_location.y,
                                             ego_location.z)

        if (collision_vehicles + collision_other + collision_pedestrians + collision_to_ego) > 0 and (
                self._time_to_firstcollision[actor_id] is None):
            # _read_observation() is called after the current physics tick during step().
            # Store human-readable 1-based transition indices: first-step collision == 1.
            self._time_to_firstcollision[actor_id] = self._num_steps[actor_id] + 1

        # ✅ 获取 SUMO/NGSIM 的统计数据（传给奖励函数）
        ngsim_stats = {"avg_deviation": 0.0, "max_deviation": 0.0, "active_count": 0, "new_detached": 0}
        if self._sumo_bridge is not None:
            try:
                ngsim_stats = self._sumo_bridge.get_current_deviation_metrics()
            except Exception:
                pass

        py_measurements = {
            "actor_id": actor_id,
            # "episode_id": self._episode_id_dict[actor_id],
            "episode": self._episode,
            "step": self._num_steps[actor_id],
            # "location": self._actors[actor_id].get_location(),
            # "rotation": self._actors[actor_id].get_transform().rotation,
            # "velocity": self._actors[actor_id].get_velocity(),
            "x": self._actors[actor_id].get_location().x,
            "y": self._actors[actor_id].get_location().y,
            "z": self._actors[actor_id].get_location().z,
            "pitch": self._actors[actor_id].get_transform().rotation.pitch,
            "yaw": self._actors[actor_id].get_transform().rotation.yaw,
            "roll": self._actors[actor_id].get_transform().rotation.roll,
            "forward_speed": math.sqrt(
                pow(self._actors[actor_id].get_velocity().x, 2) + pow(self._actors[actor_id].get_velocity().y, 2) + pow(
                    self._actors[actor_id].get_velocity().z, 2)),
            "distance_to_ego": distance_to_ego,
            "distance_to_goal": distance_to_goal,
            # "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_to_ego": collision_to_ego,
            "time_to_firstcollision": self._time_to_firstcollision,
            "time_to_goal": self._time_to_goal,
            "collision_vehicles": collision_vehicles,
            "collision_pedestrians": collision_pedestrians,
            "collision_other": collision_other,
            "intersection_offroad": intersection_offroad,
            "intersection_otherlane": intersection_otherlane,
            # "weather": self._weather,
            # "map": self._server_map,
            # "start_coord": self._start_coord[actor_id],
            # "end_coord": self._end_coord[actor_id],
            # "current_scenario": self._scenario_map,
            # "x_res": self._x_res,
            # "y_res": self._y_res,
            # "max_steps": self._scenario_map["max_steps"],
            "next_command": next_command,
            "previous_action": self._previous_actions.get(actor_id, None),
            "previous_reward": self._previous_rewards.get(actor_id, None),

            # ✅ NGSIM背景车辆统计数据
            "ngsim_avg_deviation": ngsim_stats.get("avg_deviation", 0.0),  # 仅供参考
            "ngsim_max_deviation": ngsim_stats.get("max_deviation", 0.0),  # 仅供参考
            "ngsim_active_count": ngsim_stats.get("active_count", 0),

            # 🎯 本帧新被SUMO接管的车辆数量（用于一次性惩罚）
            "ngsim_new_detached": ngsim_stats.get("new_detached", 0),
        }

        return py_measurements

    @staticmethod
    def _on_lidar_frame(weak_self, actor_id, point_cloud, generation):
        """Keep inference unchanged, but drain/reject callbacks across resets."""
        env = weak_self()
        if env is None:
            return
        gate = env._sensor_callback_gate
        if not gate.enter(generation):
            return
        try:
            env._process_lidar_data(weak_self, actor_id, point_cloud)
        finally:
            gate.leave()

    def _process_lidar_data(self, weak_self, actor_id, point_cloud):
        self = weak_self()
        if not self:
            return

        try:
            raw_points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
            if raw_points.size == 0 or raw_points.size % 4 != 0:
                raise ValueError(
                    "invalid LiDAR buffer length: {}".format(raw_points.size))

            points = raw_points.reshape((-1, 4))[:, :3]
            finite_mask = np.isfinite(points).all(axis=1)
            points = points[finite_mask]
            if points.shape[0] == 0:
                raise ValueError("LiDAR frame contains no finite XYZ points")

            if self._encoder_collection_only:
                with self._lidar_lock:
                    self._lidar_features[actor_id] = np.zeros(
                        256, dtype=np.float32
                    )
                return

            if self._use_custom_multimodal_encoder:
                with self._multimodal_encoder_lock:
                    feature = self._multimodal_feature_extractor.encode_lidar(
                        points
                    )
                with self._lidar_lock:
                    self._lidar_features[actor_id] = feature.copy()
                return

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.1)
            downsampled_points = np.asarray(
                downsampled_pcd.points, dtype=np.float32)
            if downsampled_points.shape[0] == 0:
                raise ValueError("LiDAR frame is empty after voxel downsampling")

            # Fixed physical-range normalization preserves inter-frame geometry;
            # per-frame min/max scaling distorted equal metric distances.
            downsampled_points = np.nan_to_num(
                downsampled_points, nan=0.0, posinf=self._lidar_range,
                neginf=-self._lidar_range)
            downsampled_points = np.clip(
                downsampled_points / self._lidar_range, -1.0, 1.0)
            model_input = torch.from_numpy(downsampled_points).unsqueeze(
                0).float().transpose(1, 2)

            # CARLA can invoke callbacks concurrently for different actors.
            # Serialize access to the shared PointNet instance and disable
            # autograd to keep inference deterministic and lightweight.
            with self._lidar_model_lock, torch.inference_mode():
                _, _, feature_tensor = self.lidar_model(model_input)

            feature = feature_tensor.detach().cpu().numpy().reshape(-1).astype(
                np.float32, copy=False)
            if feature.size != 256:
                raise ValueError(
                    "PointNet feature size {} does not match observation size 256".format(
                        feature.size))
            feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

            # Keep the existing [0, 1] observation-space contract.
            min_val = float(np.min(feature))
            max_val = float(np.max(feature))
            if max_val - min_val > 1e-6:
                feature = (feature - min_val) / (max_val - min_val)
            else:
                feature = np.zeros(256, dtype=np.float32)

            with self._lidar_lock:
                self._lidar_features[actor_id] = feature.copy()
        except Exception as exc:
            # A malformed sensor frame must not terminate training. Preserve the
            # last valid feature, or use a zero vector until one is available.
            with self._lidar_lock:
                self._lidar_features.setdefault(
                    actor_id, np.zeros(256, dtype=np.float32))
            if self._verbose:
                print("LiDAR frame processing failed for {}: {}".format(
                    actor_id, exc))

    def close(self):
        """Clean-up the world, clear server state & close the Env"""
        try:
            self._clean_world()
        finally:
            self._clear_server_state()


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player macad_agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents,
    )
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    """Define the main episode termination criteria"""
    m = py_measurements
    collided = (
            m["collision_vehicles"] > 0
            or m["collision_pedestrians"] > 0
            or m["collision_other"] > 0
            or m["collision_to_ego"] > 0
    )
    return bool(collided)  # or m["total_reward"] < -100)


def get_next_actions(measurements, is_discrete_actions):
    """Get/Update next action, work with way_point based planner.

    Args:
        measurements (dict): measurement data.
        is_discrete_actions (bool): whether use discrete actions

    Returns:
        dict: action_dict, dict of len-two integer lists.
    """
    # action_dict = {}
    # for actor_id, meas in measurements.items():
    #
    # return action_dict


if __name__ == "__main__":
    env = MultiCarlaEnv()
    # get_next_actions
    print(env.observation_space)
    print(env.observation_space.spaces['vehicle1'])
