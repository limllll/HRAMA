"""Fail-closed World-on-Rails inference with the pinned official policy code.

Only the model, Waypointer, command handling and action post-processing are
reused. Leaderboard never owns the CARLA clock, actors, TrafficManager or logs.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import pickle
import select
import struct
import subprocess
import sys
import time
import types
from enum import Enum

import carla
import numpy as np
import torch
import yaml

# The private CUDA worker loads this file by path with an empty PYTHONPATH.
# Keep the pure sensor helper local; importing macad_gym here would pull the
# training package (and its incompatible dependencies) into the WOR runtime.
_buffer_spec = importlib.util.spec_from_file_location(
    "_wor_sensor_buffer", os.path.join(os.path.dirname(__file__), "wor_sensor_buffer.py"))
_buffer_module = importlib.util.module_from_spec(_buffer_spec)
_buffer_spec.loader.exec_module(_buffer_module)
WORFrameBuffer = _buffer_module.WORFrameBuffer
WORSensorFailure = _buffer_module.WORSensorFailure


class WORFailure(RuntimeError):
    """A failed WOR episode must not be replaced by another controller."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree(root, relative):
    path = os.path.join(root, relative)
    with open(path, "r") as stream:
        return ast.parse(stream.read(), filename=path), path


def _official_components(root, road_option):
    # Namespace the official model's three absolute `common.*` imports. This
    # avoids shadowing modules in RLlib or in the user's project. Vendor files
    # and all network/control operations remain unchanged.
    prefix = "_macad_wor_official"
    package = types.ModuleType(prefix)
    package.__path__ = [root]
    sys.modules[prefix] = package
    common = types.ModuleType(prefix + ".common")
    common.__path__ = [os.path.join(root, "common")]
    sys.modules[common.__name__] = common
    for name in ("resnet", "normalize", "segmentation"):
        qualified = prefix + ".common." + name
        spec = importlib.util.spec_from_file_location(
            qualified, os.path.join(root, "common", name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    tree, path = _source_tree(root, "rails/models/main_model.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("common."):
            node.module = prefix + "." + node.module
    scope = {"__name__": prefix + ".model"}
    exec(compile(tree, path, "exec"), scope)
    camera_model = scope["CameraModel"]

    def load_route_code(relative, class_name=None, function_name=None):
        tree, path = _source_tree(root, relative)
        # Route conversion uses CARLA's map API below. Reuse only the official
        # pure route/command functions, without importing ScenarioRunner.
        if class_name:
            nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        else:
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]
        scope = {"__name__": prefix, "math": math, "np": np, "RoadOption": road_option}
        exec(compile(ast.Module(body=nodes), path, "exec"), scope)
        return scope[class_name or function_name]

    waypointer = load_route_code("autoagents/waypointer.py", class_name="Waypointer")
    downsample = load_route_code(
        "leaderboard/leaderboard/utils/route_manipulation.py", function_name="downsample_route")
    tree, path = _source_tree(root, "autoagents/image_agent.py")
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ImageAgent")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in
               ("run_step", "_lerp", "action_prob", "post_process")]
    if len(methods) != 4:
        raise WORFailure("Unsupported official ImageAgent interface")
    logic_tree = ast.Module(body=[ast.ClassDef(
        name="OfficialImageLogic", bases=[], keywords=[], body=methods, decorator_list=[])])
    ast.fix_missing_locations(logic_tree)
    scope = {"__name__": prefix, "math": math, "np": np, "torch": torch,
             "carla": carla, "Waypointer": waypointer,
             # Visualization is not an input to the controller. Do not allocate
             # videos or start wandb during 500-episode evaluation.
             "visualize_obs": lambda *args, **kwargs: None}
    exec(compile(logic_tree, path, "exec"), scope)
    logic_class = scope["OfficialImageLogic"]
    tree, path = _source_tree(root, "leaderboard/leaderboard/envs/sensor_interface.py")
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpeedometerReader")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_get_forward_speed"]
    speed_tree = ast.Module(body=[ast.ClassDef(
        name="OfficialSpeedReader", bases=[], keywords=[], body=methods, decorator_list=[])])
    ast.fix_missing_locations(speed_tree)
    speed_scope = {"__name__": prefix, "np": np}
    exec(compile(speed_tree, path, "exec"), speed_scope)
    return camera_model, logic_class, downsample, speed_scope["OfficialSpeedReader"]


class _NoVideo:
    def append(self, item):
        pass

    def __len__(self):
        return 0


def resolve_speed_caps(config):
    """Optional post-processing caps, not changes to the learned speed grid."""
    caps = {}
    for kind, official in (("turn", 10.0 / 3.6), ("cruise", 20.0 / 3.6)):
        requested = config.get(kind + "_speed_cap_mps")
        value = official if requested is None else float(requested)
        if not math.isfinite(value) or value <= 0:
            raise WORFailure("WOR {} speed cap must be finite and positive (m/s)".format(kind))
        caps[kind + "_mps"] = value
    caps["adapted"] = (caps["turn_mps"] != 10.0 / 3.6 or
                       caps["cruise_mps"] != 20.0 / 3.6)
    return caps


def _install_control_audit(logic, root, caps):
    """Preserve official control code; optionally substitute its one speed cap.

    The vendor file, neural logits, brake threshold, minimum throttle, steering
    and speed interpolation stay unchanged. Unsupported source fails closed.
    """
    post_process = logic.post_process
    if caps["adapted"]:
        tree, path = _source_tree(root, "autoagents/image_agent.py")
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ImageAgent")
        method = copy.deepcopy(next(n for n in original.body
                                   if isinstance(n, ast.FunctionDef) and n.name == "post_process"))
        expected = ast.dump(ast.parse("{0:10,1:10}.get(cmd,20)/3.6", mode="eval").body)
        replacements = 0
        for node in ast.walk(method):
            if (isinstance(node, ast.Compare) and len(node.ops) == 1 and
                    isinstance(node.ops[0], ast.Gt) and
                    isinstance(node.left, ast.Name) and node.left.id == "spd" and
                    ast.dump(node.comparators[0]) == expected):
                node.comparators[0] = ast.parse(
                    "self.wor_turn_speed_cap_mps if cmd in (0, 1) else self.wor_cruise_speed_cap_mps",
                    mode="eval").body
                replacements += 1
        if replacements != 1:
            raise WORFailure("Official WOR speed-cap code changed; refusing unverified adaptation")
        compiled = ast.Module(body=[method])
        ast.fix_missing_locations(compiled)
        scope = {}
        exec(compile(compiled, path, "exec"), scope)
        logic.wor_turn_speed_cap_mps = caps["turn_mps"]
        logic.wor_cruise_speed_cap_mps = caps["cruise_mps"]
        post_process = types.MethodType(scope["post_process"], logic)

    def audited(self, steer, throt, brake_prob, spd, cmd):
        effective_cap = caps["turn_mps"] if cmd in (0, 1) else caps["cruise_mps"]
        result = post_process(steer, throt, brake_prob, spd, cmd)
        self.wor_last_post_process = {
            "effective_command": int(cmd + 1), "speed_cap_mps": effective_cap,
            "speed_cap_exceeded": bool(spd > effective_cap),
            "neural_brake_probability": float(brake_prob),
            "neural_brake_selected": bool(brake_prob > 0.5),
            "pre_postprocess_steer": float(steer),
            "pre_postprocess_throttle": float(throt),
            "throttle_suppressed_by_cap": bool(spd > effective_cap and brake_prob <= 0.5),
        }
        return result

    logic.post_process = types.MethodType(audited, logic)
    logic.wor_last_post_process = None


class WORRoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class WORRuntime:
    """Official neural/control logic; no MACAD, Ray, TF or simulator lifecycle."""
    def __init__(self, config):
        self.config = dict(config)
        self.speed_caps = resolve_speed_caps(config)
        self.root = os.path.abspath(config["root"])
        self.timeout = float(config.get("sensor_timeout", 10.0))
        if not 0 < self.timeout <= 60:
            raise WORFailure("WOR sensor timeout must be in (0, 60]")
        self.device = torch.device(config.get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise WORFailure("WOR requested CUDA but it is unavailable; no CPU/agent fallback")
        with open(config["config_path"], "r") as stream:
            profile = yaml.safe_load(stream)
        if not isinstance(profile, dict):
            raise WORFailure("Invalid WOR YAML profile")
        camera_model, logic_class, self._downsample, _ = _official_components(self.root, WORRoadOption)
        # The complete pretrained state contains both backbones. Disable the
        # redundant ImageNet download, not any network layers or WOR weights.
        model_config = dict(profile, imagenet_pretrained=False)
        model = camera_model(model_config)
        state = torch.load(config["weights"], map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(self.device).eval()
        self.logic = logic_class()
        for key, value in profile.items():
            setattr(self.logic, key, value)
        self.logic.device = self.device
        self.logic.image_model = model
        self.logic.steers = torch.tensor(np.linspace(
            -profile["max_steers"], profile["max_steers"], profile["num_steers"]),
            dtype=torch.float32, device=self.device)
        self.logic.throts = torch.tensor(np.linspace(
            0, profile["max_throts"], profile["num_throts"]),
            dtype=torch.float32, device=self.device)
        self.logic.vizs = _NoVideo()
        self.logic.num_frames = 0
        self.logic.waypointer = None
        self.logic.lane_changed = None
        self.logic.lane_change_counter = 0
        self.logic.prev_steer = 0
        self.logic.stop_counter = 0
        self.profile = profile
        _install_control_audit(self.logic, self.root, self.speed_caps)
        self.control_metadata = {
            "mode": "speed_cap_adapted" if self.speed_caps["adapted"] else "official",
            "speed_caps": self.speed_caps,
            "learned_speed_grid_mps": [profile["min_speeds"], profile["max_speeds"]],
            "speed_grid_modified": False,
            "target_speeds_modified": False,
            "constant_speed_override": False,
            "note": "Raising throttle cutoff does not retrain or extend the learned speed branches",
        }

    def reset(self, route):
        self.logic._global_plan = [(gps, WORRoadOption(option)) for gps, option in route]
        self.logic.num_frames = 0
        self.logic.waypointer = None
        self.logic.lane_changed = None
        self.logic.lane_change_counter = 0
        self.logic.prev_steer = 0
        self.logic.stop_counter = 0
        self.logic.wor_last_post_process = None

    def infer(self, inputs, timestamp):
        with torch.no_grad():
            control = self.logic.run_step(inputs, timestamp)
        return {"control": {"throttle": float(control.throttle), "steer": float(control.steer),
                            "brake": float(control.brake)},
                "command": int(self.logic.waypointer.checkpoint[2].value),
                "postprocessing": dict(self.logic.wor_last_post_process)}

    def close(self):
        # In-process unit tests only. Normal evaluation uses WORWorker.close.
        self.logic.image_model = None


class WORWorker:
    """One persistent private pipe worker, explicitly using the CUDA-capable env."""
    def __init__(self, config):
        import atexit
        self.timeout = max(10., float(config.get("sensor_timeout", 10.)))
        environment = os.environ.copy()
        environment["PYTHONPATH"] = ""
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = config.get("cuda_visible_devices", "0")
        log_path = config.get("worker_log", os.path.join(os.path.dirname(config["weights"]), "wor_worker.log"))
        self._log = open(log_path, "a")
        self.process = subprocess.Popen(
            [config["worker_python"], "-u", os.path.join(os.path.dirname(__file__), "wor_worker.py"),
             "--parent-pid", str(os.getpid()), "--config-json", json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            bufsize=0, env=environment, start_new_session=True)
        atexit.register(self.close)
        try:
            hello = self._read(time.monotonic() + 120.)
            if not hello.get("ok"):
                raise WORFailure(hello.get("error", "WOR worker initialization failed"))
            self.profile = hello["profile"]
            self.runtime_metadata = hello["runtime"]
            self.control_metadata = hello["control_metadata"]
        except BaseException:
            self.close()
            raise

    def _read_exact(self, count, deadline):
        result = bytearray()
        while len(result) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise WORFailure("WOR worker response timeout; no fallback")
            part = os.read(self.process.stdout.fileno(), count - len(result))
            if not part:
                raise WORFailure("WOR worker exited; inspect its separate log; no fallback")
            result.extend(part)
        return bytes(result)

    def _read(self, deadline):
        length = struct.unpack("!I", self._read_exact(4, deadline))[0]
        if length > 4 * 1024 * 1024:
            raise WORFailure("Invalid WOR private-pipe response size")
        return pickle.loads(self._read_exact(length, deadline))

    def call(self, request):
        payload = pickle.dumps(request, protocol=4)
        message = struct.pack("!I", len(payload)) + payload
        deadline = time.monotonic() + self.timeout
        offset = 0
        try:
            while offset < len(message):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise WORFailure("WOR worker input timeout; no fallback")
                offset += os.write(self.process.stdin.fileno(), message[offset:offset + 4096])
            reply = self._read(deadline)
            if not reply.get("ok"):
                raise WORFailure(reply.get("error", "WOR inference failed"))
            return reply
        except (OSError, EOFError) as exc:
            raise WORFailure("WOR worker pipe failed; no fallback: {}".format(exc)) from exc

    def reset(self, route):
        self.call({"operation": "reset", "route": route})

    def infer(self, inputs, timestamp):
        return self.call({"operation": "infer", "inputs": inputs, "timestamp": timestamp})

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            try:
                process.wait(timeout=3.)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.)
            if process.stdout is not None:
                process.stdout.close()
            self.process = None
        if getattr(self, "_log", None) is not None:
            self._log.close()
            self._log = None


class WORController:
    def __init__(self, config):
        from macad_gym.carla.PythonAPI.agents.navigation.local_planner import RoadOption
        self.config = dict(config)
        self.root = os.path.abspath(config["root"])
        self.timeout = float(config.get("sensor_timeout", 10.))
        if not 0 < self.timeout <= 60:
            raise WORFailure("WOR sensor timeout must be in (0, 60]")
        self.sensor_workers = int(config.get("sensor_workers", 4))
        if not 1 <= self.sensor_workers <= 32:
            raise WORFailure("WOR sensor workers must be in [1, 32]")
        self._listener_client = None
        self._listener_world = None
        self._listener_port = None
        self._sensor_buffer = WORFrameBuffer()
        self._diagnostic_path = config.get("sensor_diagnostic_log")
        self._last_sensor_wait_ms = None
        if config.get("worker_python"):
            self.runtime = WORWorker(config)
            self.logic = types.SimpleNamespace(**self.runtime.profile)
            runtime_metadata = self.runtime.runtime_metadata
            self.device = torch.device(config.get("device", "cuda:0"))
        else:
            self.runtime = WORRuntime(config)
            self.logic = self.runtime.logic
            self.device = self.runtime.device
            runtime_metadata = {"python": sys.executable, "torch": torch.__version__}
        _, _, self._downsample, speed_class = _official_components(self.root, RoadOption)
        self._speed_reader = speed_class()
        self._generation = 0
        self._sensors = []
        self._vehicle = None
        self._env = None
        self.last_record = None
        self.episode_steps = 0
        self.metadata = {
            "backend": "wor", "fallback_allowed": False, "device": str(self.device),
            "root": self.root, "weights": config["weights"],
            "weights_sha256": sha256(config["weights"]),
            "config_path": config["config_path"], "config_sha256": sha256(config["config_path"]),
            "official_source_sha256": {p: sha256(os.path.join(self.root, p)) for p in (
                "autoagents/image_agent.py", "autoagents/waypointer.py", "rails/models/main_model.py",
                "common/resnet.py", "common/normalize.py", "common/segmentation.py",
                "leaderboard/leaderboard/envs/sensor_interface.py",
                "leaderboard/leaderboard/utils/route_manipulation.py")},
            "raw_wide_shape": [240, 480, 3], "raw_narrow_shape": [240, 384, 3],
            "initial_speed_protocol": "unchanged scenario speeds; no constant-speed override",
            "control_adaptation": self.runtime.control_metadata,
            "speed_input": "official SpeedometerReader signed forward speed in m/s",
            "inference_runtime": runtime_metadata,
            "sensor_transport": {"client": "dedicated WOR listener",
                                 "worker_threads": self.sensor_workers,
                                 "owns_world_tick": False, "exact_frame_join": True,
                                 "retained_frame_span": self._sensor_buffer.keep_frames,
                                 "failure_log": self._diagnostic_path},
        }
        print("[WOR] strict model state loaded on {}: {}".format(self.device, config["weights"]), flush=True)

    def detach_sensors(self):
        self._generation += 1
        self._sensor_buffer.close(self._generation)
        sensors, self._sensors = self._sensors, []
        # Keep the listener client alive until the environment has stopped and
        # destroyed these sensor handles. It is reused on the same server port.
        self._vehicle = None
        self._env = None
        return sensors

    def abandon_episode(self):
        # Only after the owned server has been stopped: no RPC to dead actors.
        self.detach_sensors()
        self._listener_world = None
        self._listener_client = None
        self._listener_port = None

    def _sensor_world(self, env):
        """Separate callback pool, never a second simulator clock."""
        port = int(env._server_port)
        if self._listener_client is None or self._listener_port != port:
            self._listener_client = carla.Client("localhost", port, self.sensor_workers)
            self._listener_port = port
        # Sensor RPCs do not load a map or own its clock. Bound this client too:
        # sensor.stop() uses its owning client, not env's temporary cleanup RPC
        # timeout, so inheriting the environment's 120 s timeout could hang reset.
        self._listener_client.set_timeout(min(
            self.timeout, float(getattr(env, "_rpc_timeout", 20.))))
        world = self._listener_client.get_world()
        if world.id != env.world.id:
            raise WORFailure("WOR sensor client is connected to a different CARLA world")
        vehicle = world.get_actor(self._vehicle.id)
        if vehicle is None or not vehicle.is_alive:
            raise WORFailure("WOR sensor client cannot find the live ego actor")
        self._listener_world = world
        return world, vehicle

    def bind_episode(self, env):
        if self._sensors:
            raise WORFailure("Old WOR sensors were not detached before a new episode")
        self._env, self._vehicle = env, env._actors["ego"]
        self.episode_steps = 0
        self.last_record = None
        self.logic.waypointer = None
        self.logic.prev_steer = 0
        self.logic.lane_change_counter = 0
        self.logic.stop_counter = 0
        self.logic.lane_changed = None
        route = [(wp.transform, option) for wp, option in
                 env._behavior_agents["ego"]._local_planner._waypoints_queue]
        if not route:
            raise WORFailure("No planned ego route for WOR")
        route_ids = self._downsample(route, 50)
        world_map = env.world.get_map()
        self.logic._global_plan = []
        for index in route_ids:
            transform, option = route[index]
            gps = world_map.transform_to_geolocation(transform.location)
            self.logic._global_plan.append((
                    {"lat": gps.latitude, "lon": gps.longitude, "z": gps.altitude}, option))
        self.runtime.reset([(gps, int(option.value)) for gps, option in self.logic._global_plan])
        self._generation += 1
        generation = self._generation
        self._sensor_buffer.start(generation)
        gate_generation = env._sensor_callback_gate.generation
        specs = [("left", 160, 240, 60, -55), ("center", 160, 240, 60, 0),
                 ("right", 160, 240, 60, 55), ("narrow", 384, 240, 50, 0)]
        try:
            sensor_world, attachment = self._sensor_world(env)
            library = sensor_world.get_blueprint_library()
            for name, width, height, fov, yaw in specs:
                blueprint = library.find("sensor.camera.rgb")
                attributes = {"image_size_x": width, "image_size_y": height, "fov": fov,
                              "sensor_tick": 0.0, "lens_circle_multiplier": 3.0,
                              "lens_circle_falloff": 3.0, "chromatic_aberration_intensity": 0.5,
                              "chromatic_aberration_offset": 0}
                for key, value in attributes.items():
                    if blueprint.has_attribute(key):
                        blueprint.set_attribute(key, str(value))
                transform = carla.Transform(carla.Location(x=self.logic.camera_x, z=self.logic.camera_z),
                                            carla.Rotation(yaw=yaw))
                sensor = sensor_world.spawn_actor(blueprint, transform, attach_to=attachment)
                self._sensors.append(sensor)
                sensor.listen(self._callback(name, generation, env._sensor_callback_gate, gate_generation))
            blueprint = library.find("sensor.other.gnss")
            for key in ("noise_alt_stddev", "noise_lat_stddev", "noise_lon_stddev",
                        "noise_alt_bias", "noise_lat_bias", "noise_lon_bias", "sensor_tick"):
                if blueprint.has_attribute(key):
                    blueprint.set_attribute(key, "0")
            sensor = sensor_world.spawn_actor(blueprint, carla.Transform(
                carla.Location(z=self.logic.camera_z)), attach_to=attachment)
            self._sensors.append(sensor)
            sensor.listen(self._callback("gps", generation, env._sensor_callback_gate, gate_generation))
        except Exception as exc:
            raise WORFailure("Could not create official WOR sensor suite: {}".format(exc)) from exc

    def _callback(self, name, generation, gate, gate_generation):
        def receive(event):
            if not gate.enter(gate_generation):
                return
            try:
                if not self._sensor_buffer.is_current(generation):
                    return
                if name == "gps":
                    value = np.array([event.latitude, event.longitude, event.altitude], dtype=np.float64)
                else:
                    value = np.frombuffer(event.raw_data, dtype=np.uint8).reshape(
                        event.height, event.width, 4)[..., :3].copy()  # Official input is BGR.
                self._sensor_buffer.store(name, generation, event.frame, value)
            except Exception as exc:
                self._sensor_buffer.fail_callback(name, generation, repr(exc))
            finally:
                gate.leave()
        return receive

    def ready(self, frame):
        return self._sensor_buffer.ready(frame)

    def _record_sensor_failure(self, diagnostic):
        record = dict(diagnostic, wall_time=time.time(), server_port=self._listener_port,
                      world_id=getattr(self._listener_world, "id", None),
                      sensor_actor_ids=[int(sensor.id) for sensor in self._sensors],
                      completed_wor_steps=self.episode_steps)
        if self._diagnostic_path:
            try:
                parent = os.path.dirname(os.path.abspath(self._diagnostic_path))
                os.makedirs(parent, exist_ok=True)
                with open(self._diagnostic_path, "a") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            except OSError as exc:
                # Report a disk error, but never suppress the original failure.
                print("[WOR] could not write sensor diagnostic: {}".format(exc), flush=True)
        print("[WOR] sensor diagnostic: {}".format(json.dumps(record, sort_keys=True)), flush=True)

    def _wait_frame(self, frame):
        started = time.monotonic()
        try:
            return self._sensor_buffer.wait(frame, self.timeout)
        except WORSensorFailure as exc:
            self._record_sensor_failure(exc.diagnostic)
            raise WORFailure(str(exc)) from exc
        finally:
            self._last_sensor_wait_ms = (time.monotonic() - started) * 1000.

    def compute_control(self):
        if self._env is None or self._vehicle is None:
            raise WORFailure("WOR is not bound to a live episode")
        snapshot = self._env.world.get_snapshot()
        frame = int(snapshot.frame)
        data = self._wait_frame(frame)
        wide = np.concatenate([data[name] for name in ("left", "center", "right")], axis=1)
        if wide.shape != (240, 480, 3) or data["narrow"].shape != (240, 384, 3):
            raise WORFailure("WOR camera shape does not match the official model")
        velocity = self._vehicle.get_velocity()
        transform = self._vehicle.get_transform()
        speed = float(self._speed_reader._get_forward_speed(transform=transform, velocity=velocity))
        if not np.isfinite(speed) or not np.isfinite(data["gps"]).all():
            raise WORFailure("Non-finite WOR speed/GNSS")
        inputs = {"Wide_RGB": (frame, wide), "Narrow_RGB": (frame, data["narrow"]),
                  "EGO": (frame, {"spd": speed}), "GPS": (frame, data["gps"])}
        started = time.monotonic()
        try:
            result = self.runtime.infer(inputs, snapshot.timestamp.elapsed_seconds)
            control = carla.VehicleControl(**result["control"])
        except Exception as exc:
            raise WORFailure("WOR inference failed; no BehaviorAgent fallback: {}".format(exc)) from exc
        values = [control.throttle, control.steer, control.brake]
        if not np.isfinite(values).all() or not (0 <= values[0] <= 1 and -1 <= values[1] <= 1 and 0 <= values[2] <= 1):
            raise WORFailure("Invalid WOR control {}".format(values))
        self.episode_steps += 1
        location = self._vehicle.get_location()
        self.last_record = {
            "control_source": "wor", "frame": frame, "sensor_frames": {key: frame for key in data},
            "sim_time": snapshot.timestamp.elapsed_seconds, "speed_mps": speed,
            "position": [location.x, location.y, location.z],
            "command": result["command"],
            "postprocessing": result["postprocessing"],
            "throttle": float(control.throttle), "steer": float(control.steer), "brake": float(control.brake),
            "inference_ms": (time.monotonic() - started) * 1000,
            "sensor_wait_ms": self._last_sensor_wait_ms,
            "route_waypoint_count": len(self._env._behavior_agents["ego"]._local_planner._waypoints_queue),
        }
        return control
