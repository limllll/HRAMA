import importlib
import importlib.util
import logging
import math
import os
import sys
from typing import Any, Dict, Optional

import carla
import cv2
import numpy as np

from macad_gym.carla.PythonAPI.agents.navigation.behavior_agent import BehaviorAgent

logger = logging.getLogger(__name__)
EARTH_RADIUS = 6371e3


def _ensure_sys_path(paths):
    for path in paths:
        if not path:
            continue
        normalized = os.path.abspath(path)
        if normalized not in sys.path:
            sys.path.insert(0, normalized)


def _load_module(module_or_file: str):
    if os.path.isfile(module_or_file):
        module_name = os.path.splitext(os.path.basename(module_or_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, module_or_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from file: {module_or_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_or_file)


def _coerce_vehicle_control(control_like: Any) -> carla.VehicleControl:
    if isinstance(control_like, carla.VehicleControl):
        return control_like

    if isinstance(control_like, dict):
        return carla.VehicleControl(
            throttle=float(control_like.get("throttle", 0.0)),
            steer=float(control_like.get("steer", 0.0)),
            brake=float(control_like.get("brake", 0.0)),
            hand_brake=bool(control_like.get("hand_brake", False)),
            reverse=bool(control_like.get("reverse", False)),
        )

    if isinstance(control_like, (tuple, list)) and len(control_like) >= 3:
        return carla.VehicleControl(
            throttle=float(control_like[0]),
            steer=float(control_like[1]),
            brake=float(control_like[2]),
        )

    raise TypeError(f"Unsupported control type: {type(control_like)!r}")


def _location_to_fake_gps(location: carla.Location):
    lat = location.x / EARTH_RADIUS * (180.0 / math.pi)
    lon = location.y / EARTH_RADIUS * (180.0 / math.pi)
    return np.array([lat, lon, location.z], dtype=np.float32)


def _resize_rgb(image, width, height):
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


class BaseAutonomousAgentAdapter:
    def __init__(
        self,
        vehicle,
        actor_id: str,
        world=None,
        destination: Optional[carla.Location] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._vehicle = vehicle
        self._actor_id = actor_id
        self._world = world
        self._config = config or {}
        self._destination = destination

    def set_destination(self, destination: carla.Location):
        self._destination = destination

    def done(self) -> bool:
        return False

    def run_step(self, env, actor_id: str) -> carla.VehicleControl:
        raise NotImplementedError

    def get_waypoints_queue(self):
        return []

    def get_target_waypoint(self):
        return None

    def set_max_speed(self, speed_kmh: float) -> bool:
        return False

    def destroy(self):
        return


class BehaviorAgentAdapter(BaseAutonomousAgentAdapter):
    def __init__(self, vehicle, actor_id, world=None, destination=None, config=None):
        super().__init__(vehicle, actor_id, world, destination, config)
        behavior = self._config.get("behavior", "aggressive")
        self._agent = BehaviorAgent(vehicle, behavior=behavior)
        self._behavior = self._agent._behavior
        self._local_planner = self._agent._local_planner
        if destination is not None:
            self.set_destination(destination)

    def set_destination(self, destination: carla.Location):
        super().set_destination(destination)
        self._agent.set_destination(destination)

    def done(self) -> bool:
        return self._agent.done()

    def run_step(self, env, actor_id: str) -> carla.VehicleControl:
        return self._agent.run_step()

    def get_waypoints_queue(self):
        return self._agent._local_planner._waypoints_queue

    def get_target_waypoint(self):
        return self._agent._local_planner.target_waypoint

    def set_max_speed(self, speed_kmh: float) -> bool:
        self._agent._behavior.max_speed = speed_kmh
        return True


class WorldOnRailsAdapter(BaseAutonomousAgentAdapter):
    def __init__(self, vehicle, actor_id, world=None, destination=None, config=None):
        super().__init__(vehicle, actor_id, world, destination, config)
        planner_behavior = self._config.get("planner_behavior", "aggressive")
        self._planner_agent = BehaviorAgent(vehicle, behavior=planner_behavior)
        self._local_planner = self._planner_agent._local_planner
        self._behavior = self._planner_agent._behavior
        self._wrapped_agent = None
        self._wrapped_agent_ready = False
        self._root_dir = self._config.get("root_dir") or self._config.get("repo_root")
        if destination is not None:
            self.set_destination(destination)
        self._try_initialize_wrapped_agent()

    def _try_initialize_wrapped_agent(self):
        module_name = (
            self._config.get("module")
            or self._config.get("agent_module")
            or os.environ.get("WORLD_ON_RAILS_AGENT_MODULE")
        )
        if not module_name:
            logger.warning(
                "World on Rails backend selected for actor `%s`, but no module was configured. "
                "Falling back to planner control.",
                self._actor_id,
            )
            return

        class_name = (
            self._config.get("class_name")
            or self._config.get("agent_class")
            or os.environ.get("WORLD_ON_RAILS_AGENT_CLASS", "WorldOnRailsAgent")
        )
        try:
            python_paths = list(self._config.get("python_paths", []))
            if self._root_dir:
                python_paths.extend(
                    [
                        self._root_dir,
                        os.path.join(self._root_dir, "leaderboard"),
                        os.path.join(self._root_dir, "scenario_runner"),
                    ]
                )
            module_dir = (
                os.path.dirname(module_name) if os.path.isfile(module_name) else None
            )
            if module_dir:
                python_paths.append(module_dir)
                python_paths.append(os.path.dirname(module_dir))
            _ensure_sys_path(python_paths)

            module = _load_module(module_name)
            agent_cls = getattr(module, class_name)
            self._wrapped_agent = agent_cls()

            setup_arg = (
                self._config.get("setup_path")
                or self._config.get("config_path")
                or self._config.get("checkpoint_path")
            )
            if hasattr(self._wrapped_agent, "setup"):
                if setup_arg is not None:
                    self._wrapped_agent.setup(setup_arg)
                else:
                    try:
                        self._wrapped_agent.setup(None)
                    except TypeError:
                        self._wrapped_agent.setup()

            if hasattr(self._wrapped_agent, "set_vehicle"):
                self._wrapped_agent.set_vehicle(self._vehicle)
            elif hasattr(self._wrapped_agent, "bind_vehicle"):
                self._wrapped_agent.bind_vehicle(self._vehicle)

            if hasattr(self._wrapped_agent, "track") and getattr(
                self._wrapped_agent, "track", None
            ) is None:
                pass

            self._wrapped_agent_ready = True
        except Exception as exc:
            logger.warning(
                "Failed to initialize World on Rails agent for actor `%s`: %s. "
                "Falling back to planner control.",
                self._actor_id,
                exc,
            )
            self._wrapped_agent = None
            self._wrapped_agent_ready = False

    def _build_global_plan(self):
        global_plan = []
        for item in list(self.get_waypoints_queue()):
            if isinstance(item, tuple) and len(item) >= 2:
                waypoint, road_option = item[0], item[1]
            else:
                waypoint, road_option = item, None
            if waypoint is None:
                continue
            location = waypoint.transform.location
            gps = _location_to_fake_gps(location)
            global_plan.append(
                (
                    {"lat": float(gps[0]), "lon": float(gps[1]), "z": float(gps[2])},
                    road_option,
                )
            )
        return global_plan

    def _prepare_wrapped_agent_state(self):
        if self._wrapped_agent is None:
            return
        if not hasattr(self._wrapped_agent, "_global_plan"):
            self._wrapped_agent._global_plan = self._build_global_plan()
        elif not self._wrapped_agent._global_plan:
            self._wrapped_agent._global_plan = self._build_global_plan()

    def _build_input_data(self, env, actor_id: str) -> Dict[str, Any]:
        timestamp = float(getattr(env, "_elapsed_time", 0.0))
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        location = transform.location
        rotation = transform.rotation
        gps = _location_to_fake_gps(location)
        rgb_image = None
        camera_manager = env._cameras.get(actor_id)
        if camera_manager is not None:
            rgb_image = camera_manager.image
        wide_rgb = _resize_rgb(rgb_image, 160, 240)
        narrow_rgb = _resize_rgb(rgb_image, 384, 240)
        left_rgb = _resize_rgb(rgb_image, 160, 240)
        center_rgb = _resize_rgb(rgb_image, 160, 240)
        right_rgb = _resize_rgb(rgb_image, 160, 240)

        input_data = {
            "rgb": (timestamp, rgb_image),
            "speed": (timestamp, {"speed": speed}),
            "gps": (timestamp, np.array([location.x, location.y], dtype=np.float32)),
            "imu": (
                timestamp,
                np.array([0.0, 0.0, math.radians(rotation.yaw)], dtype=np.float32),
            ),
            "transform": (timestamp, transform),
            "route": (timestamp, list(self.get_waypoints_queue())),
            "EGO": (timestamp, {"spd": speed, "speed": speed}),
            "GPS": (timestamp, gps),
            "Wide_RGB": (timestamp, wide_rgb),
            "Narrow_RGB": (timestamp, narrow_rgb),
            "RGB_0": (timestamp, left_rgb),
            "RGB_1": (timestamp, center_rgb),
            "RGB_2": (timestamp, right_rgb),
        }
        return input_data

    def _run_wrapped_agent(self, env, actor_id: str) -> Optional[carla.VehicleControl]:
        if not self._wrapped_agent_ready or self._wrapped_agent is None:
            return None

        self._prepare_wrapped_agent_state()
        input_data = self._build_input_data(env, actor_id)
        timestamp = float(getattr(env, "_elapsed_time", 0.0))

        try:
            if hasattr(self._wrapped_agent, "run_step"):
                try:
                    control_like = self._wrapped_agent.run_step(input_data, timestamp)
                except TypeError:
                    try:
                        control_like = self._wrapped_agent.run_step(input_data)
                    except TypeError:
                        control_like = self._wrapped_agent.run_step()
                return _coerce_vehicle_control(control_like)
        except Exception as exc:
            logger.warning(
                "World on Rails run_step failed for actor `%s`: %s. Falling back to planner control.",
                actor_id,
                exc,
            )
        return None

    def set_destination(self, destination: carla.Location):
        super().set_destination(destination)
        self._planner_agent.set_destination(destination)
        if self._wrapped_agent is None:
            return
        if hasattr(self._wrapped_agent, "_global_plan"):
            self._wrapped_agent._global_plan = self._build_global_plan()
        if hasattr(self._wrapped_agent, "waypointer"):
            self._wrapped_agent.waypointer = None
        if hasattr(self._wrapped_agent, "set_destination"):
            try:
                self._wrapped_agent.set_destination(destination)
            except Exception:
                pass

    def done(self) -> bool:
        return self._planner_agent.done()

    def run_step(self, env, actor_id: str) -> carla.VehicleControl:
        control = self._run_wrapped_agent(env, actor_id)
        if control is not None:
            return control
        return self._planner_agent.run_step()

    def get_waypoints_queue(self):
        return self._planner_agent._local_planner._waypoints_queue

    def get_target_waypoint(self):
        return self._planner_agent._local_planner.target_waypoint

    def set_max_speed(self, speed_kmh: float) -> bool:
        self._planner_agent._behavior.max_speed = speed_kmh
        return True

    def destroy(self):
        if self._wrapped_agent is None:
            return
        if hasattr(self._wrapped_agent, "destroy"):
            try:
                self._wrapped_agent.destroy()
            except Exception:
                pass
        elif hasattr(self._wrapped_agent, "cleanup"):
            try:
                self._wrapped_agent.cleanup()
            except Exception:
                pass


def create_autonomous_agent(
    backend_name: str,
    vehicle,
    actor_id: str,
    world=None,
    destination: Optional[carla.Location] = None,
    config: Optional[Dict[str, Any]] = None,
):
    backend = (backend_name or "behavior").strip().lower()
    if backend in ("behavior", "behavior_agent", "carla_behavior"):
        return BehaviorAgentAdapter(vehicle, actor_id, world, destination, config)
    if backend in ("wor", "world_on_rails", "world-on-rails"):
        return WorldOnRailsAdapter(vehicle, actor_id, world, destination, config)
    raise ValueError(f"Unsupported autonomous control backend: {backend_name}")
