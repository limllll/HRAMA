#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect 10,000 diverse CARLA RGB--LiDAR pairs from Town01--Town05.

Version 2 keeps the PPO sensor geometry unchanged while reducing temporal
correlation and increasing appearance diversity: samples are taken every ten
simulation frames and one seeded-random weather preset is used per route.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import queue
import random
import sys
import time

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import carla

from macad_gym.agents.multimodal_encoder.dataset import SynchronizedPairWriter


# ---------------------------------------------------------------------------
# IDE / mouse-run configuration
# Edit this block and click Run. Every value can still be overridden by a
# command-line argument when running on a server.
# ---------------------------------------------------------------------------
RUN_CONFIG = {
    "output_dir": os.path.join(PROJECT_ROOT, "artifacts", "multimodal_dataset_v2"),
    "host": "127.0.0.1",
    "port": 2000,
    "traffic_manager_port": 8000,
    "timeout": 20.0,
    "seed": 2026,
    "stride": 10,
    "minimum_route_distance": 150.0,
    "behavior": "normal",
    "towns": list(("Town01", "Town02", "Town03", "Town04", "Town05")),
    "weather": [
        "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
        "SoftRainNoon", "MidRainyNoon", "ClearSunset", "CloudySunset",
    ],
    "vehicle_filter": "vehicle.tesla.model3",
}


PYTHON_API_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "carla", "PythonAPI")
)
if PYTHON_API_DIR not in sys.path:
    sys.path.insert(0, PYTHON_API_DIR)
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402


TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05")
WEATHER_PRESETS = (
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "SoftRainNoon", "MidRainyNoon", "ClearSunset", "CloudySunset",
)
TRAFFIC_TIERS = (
    ("low", 400, 0, 5),
    ("medium", 800, 10, 15),
    ("high", 800, 20, 30),
)

# Identical to CameraManager and MultiCarlaEnv PPO sensor configuration.
SENSOR_CONFIG = {
    "camera": {
        "blueprint": "sensor.camera.rgb",
        "image_size_x": 300,
        "image_size_y": 120,
        "fov": 90.0,
        "transform": {"x": 1.8, "y": 0.0, "z": 1.7,
                      "pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    },
    "lidar": {
        "blueprint": "sensor.lidar.ray_cast",
        "channels": 16,
        "points_per_second": 10000,
        "range": 20.0,
        "rotation_frequency": 20.0,
        "horizontal_fov": 360.0,
        "transform": {"x": 0.0, "y": 0.0, "z": 2.5,
                      "pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect 10k diverse Town01--Town05 RGB--LiDAR pairs"
    )
    parser.add_argument("--output-dir", default=RUN_CONFIG["output_dir"])
    parser.add_argument("--host", default=RUN_CONFIG["host"])
    parser.add_argument("--port", type=int, default=RUN_CONFIG["port"])
    parser.add_argument(
        "--traffic-manager-port", type=int,
        default=RUN_CONFIG["traffic_manager_port"]
    )
    parser.add_argument("--timeout", type=float, default=RUN_CONFIG["timeout"])
    parser.add_argument("--seed", type=int, default=RUN_CONFIG["seed"])
    parser.add_argument("--stride", type=int, default=RUN_CONFIG["stride"])
    parser.add_argument(
        "--minimum-route-distance", type=float,
        default=RUN_CONFIG["minimum_route_distance"]
    )
    parser.add_argument(
        "--behavior", choices=["cautious", "normal", "aggressive"],
        default=RUN_CONFIG["behavior"]
    )
    parser.add_argument(
        "--towns", nargs="+", choices=TOWNS, default=RUN_CONFIG["towns"]
    )
    parser.add_argument(
        "--weather", nargs="+", choices=WEATHER_PRESETS,
        default=RUN_CONFIG["weather"]
    )
    parser.add_argument(
        "--vehicle-filter", default=RUN_CONFIG["vehicle_filter"]
    )
    return parser.parse_args()


def as_transform(spec):
    return carla.Transform(
        carla.Location(x=spec["x"], y=spec["y"], z=spec["z"]),
        carla.Rotation(
            pitch=spec["pitch"], yaw=spec["yaw"], roll=spec["roll"]
        ),
    )


def sensor_blueprints(world):
    library = world.get_blueprint_library()
    camera = library.find(SENSOR_CONFIG["camera"]["blueprint"])
    camera.set_attribute("image_size_x", str(SENSOR_CONFIG["camera"]["image_size_x"]))
    camera.set_attribute("image_size_y", str(SENSOR_CONFIG["camera"]["image_size_y"]))
    camera.set_attribute("fov", str(SENSOR_CONFIG["camera"]["fov"]))
    lidar = library.find(SENSOR_CONFIG["lidar"]["blueprint"])
    for key in ("channels", "points_per_second", "range",
                "rotation_frequency", "horizontal_fov"):
        lidar.set_attribute(key, str(SENSOR_CONFIG["lidar"][key]))
    return camera, lidar


def four_wheel_blueprints(world, pattern="vehicle.*"):
    return [
        bp for bp in world.get_blueprint_library().filter(pattern)
        if not bp.has_attribute("number_of_wheels")
        or int(bp.get_attribute("number_of_wheels")) == 4
    ]


def choose_vehicle_blueprint(world, pattern, rng):
    candidates = four_wheel_blueprints(world, pattern)
    if not candidates:
        candidates = four_wheel_blueprints(world)
    if not candidates:
        raise RuntimeError("No compatible vehicle blueprint is available")
    blueprint = rng.choice(candidates)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "multimodal_collector")
    return blueprint


def choose_route(spawn_points, rng, minimum_distance):
    if len(spawn_points) < 2:
        raise RuntimeError("Map has fewer than two spawn points")
    for _ in range(200):
        start, destination = rng.sample(spawn_points, 2)
        if start.location.distance(destination.location) >= minimum_distance:
            return start, destination
    return tuple(rng.sample(spawn_points, 2))


def spawn_sensor_vehicle(world, spawn_points, args, rng):
    blueprint = choose_vehicle_blueprint(world, args.vehicle_filter, rng)
    # Match PythonAPI/examples/automatic_control.py World.restart(): choose
    # official map spawn points and retry try_spawn_actor until one succeeds.
    candidates = list(spawn_points)
    rng.shuffle(candidates)
    for start in candidates:
        vehicle = world.try_spawn_actor(blueprint, start)
        if vehicle is not None:
            return vehicle
    raise RuntimeError("Unable to spawn the sensor vehicle")


def spawn_background_traffic(world, traffic_manager, count, ego_transform, rng):
    blueprints = four_wheel_blueprints(world)
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    actors = []
    for transform in spawn_points:
        if len(actors) >= count:
            break
        if transform.location.distance(ego_transform.location) < 12.0:
            continue
        blueprint = rng.choice(blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "background")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        actors.append(actor)
    return actors


def wait_for_frame(sensor_queue, frame, sensor_name, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        measurement = sensor_queue.get(timeout=max(0.01, deadline - time.time()))
        measurement_frame = getattr(measurement, "frame", None)
        if measurement_frame is None:
            measurement_frame = getattr(measurement, "frame_number", None)
        if int(measurement_frame) == int(frame):
            return measurement
        if int(measurement_frame) > int(frame):
            raise RuntimeError(
                "{} advanced past frame {} to {}".format(
                    sensor_name, frame, measurement_frame
                )
            )
    raise RuntimeError("Timed out waiting for {} frame {}".format(sensor_name, frame))


def lidar_xyz(measurement):
    values = np.frombuffer(measurement.raw_data, dtype=np.float32)
    if values.size == 0 or values.size % 4 != 0:
        raise ValueError("Invalid LiDAR buffer length: {}".format(values.size))
    points = values.reshape((-1, 4))[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        raise ValueError("LiDAR frame contains no finite XYZ points")
    return points


def destroy_actors(actors):
    for actor in reversed(actors):
        if actor is None:
            continue
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except Exception:
            pass
        try:
            actor.destroy()
        except Exception:
            pass


def update_spectator_chase_view(world, vehicle):
    """Place the visible CARLA spectator behind and above the ego vehicle."""
    vehicle_transform = vehicle.get_transform()
    chase_location = vehicle_transform.transform(
        carla.Location(x=-8.0, y=0.0, z=4.0)
    )
    chase_rotation = carla.Rotation(
        pitch=-15.0,
        yaw=vehicle_transform.rotation.yaw,
        roll=0.0,
    )
    world.get_spectator().set_transform(
        carla.Transform(chase_location, chase_rotation)
    )


def waiting_at_traffic_light(vehicle):
    """Return whether a stopped ego vehicle is legitimately waiting at a light."""
    try:
        if not vehicle.is_at_traffic_light():
            return False
        return vehicle.get_traffic_light_state() in (
            carla.TrafficLightState.Red,
            carla.TrafficLightState.Yellow,
        )
    except (AttributeError, RuntimeError):
        return False


def collect_route(world, traffic_manager, town, tier_name, route_index,
                  background_count, target_samples, weather_name, args, rng):
    spawn_points = list(world.get_map().get_spawn_points())
    actors = []
    camera_queue = queue.Queue()
    lidar_queue = queue.Queue()
    collision_queue = queue.Queue()
    scenario = "{}_{}_route{:02d}".format(town, tier_name, route_index)
    writer = SynchronizedPairWriter(
        args.output_dir, scenario=scenario, stride=1, max_samples=0
    )
    route_saved = 0
    try:
        vehicle = spawn_sensor_vehicle(world, spawn_points, args, rng)
        actors.append(vehicle)

        # CARLA automatic_control.py advances the world immediately after the
        # player is spawned, before sensors and the navigation agent are made.
        world.tick()

        background = spawn_background_traffic(
            world, traffic_manager, background_count, vehicle.get_transform(), rng
        )
        actors.extend(background)
        if len(background) != background_count:
            raise RuntimeError(
                "Requested {} background vehicles but spawned only {}".format(
                    background_count, len(background)
                )
            )
        camera_bp, lidar_bp = sensor_blueprints(world)
        camera = world.spawn_actor(
            camera_bp, as_transform(SENSOR_CONFIG["camera"]["transform"]),
            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        lidar = world.spawn_actor(
            lidar_bp, as_transform(SENSOR_CONFIG["lidar"]["transform"]),
            attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        actors.extend([camera, lidar])
        camera.listen(camera_queue.put)
        lidar.listen(lidar_queue.put)
        collision = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(), attach_to=vehicle,
        )
        actors.append(collision)
        collision.listen(collision_queue.put)
        agent = BehaviorAgent(vehicle, behavior=args.behavior)
        destination = rng.choice(spawn_points).location
        agent.set_destination(destination)
        print("{} {} route {}: destination set with BehaviorAgent".format(
            town, tier_name, route_index
        ))

        stationary_ticks = 0
        tick_index = 0
        while route_saved < target_samples:
            # Follow CARLA PythonAPI/examples/automatic_control.py: advance the
            # synchronous world first, then plan and apply control for the next
            # simulation frame.
            update_spectator_chase_view(world, vehicle)
            frame = world.tick()
            tick_index += 1
            image = wait_for_frame(camera_queue, frame, "camera", args.timeout)
            point_cloud = wait_for_frame(lidar_queue, frame, "lidar", args.timeout)

            if agent.done():
                print("{} {} route {}: target reached; starting a new route".format(
                    town, tier_name, route_index
                ))
                break
            control = agent.run_step()
            control.manual_gear_shift = False
            vehicle.apply_control(control)
            if not collision_queue.empty():
                collision_event = collision_queue.get_nowait()
                print("{} {} route {}: collision at frame {}; restarting route".format(
                    town, tier_name, route_index, collision_event.frame
                ))
                break

            velocity = vehicle.get_velocity()
            speed = float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))
            # Never save stationary frames. A red/yellow-light wait is valid
            # route behavior and must not trigger the stuck-vehicle watchdog.
            if speed < 0.2 and not waiting_at_traffic_light(vehicle):
                stationary_ticks += 1
            else:
                stationary_ticks = 0
            if stationary_ticks >= 400:
                print("{} {} route {}: stationary for 20s; restarting route".format(
                    town, tier_name, route_index
                ))
                break

            if tick_index % args.stride != 0:
                continue
            if speed < 0.2:
                continue
            if writer.write(
                image=image, points=lidar_xyz(point_cloud),
                actor_id="sensor_vehicle", frame=frame, episode=route_index,
                metadata={
                    "weather": weather_name,
                    "traffic_tier": tier_name,
                    "background_vehicles": int(len(background)),
                    "sampling_stride": int(args.stride),
                },
            ):
                route_saved += 1
                if route_saved % 25 == 0:
                    print("{} {} route {}: {}/{} pairs".format(
                        town, tier_name, route_index, route_saved, target_samples
                    ))
        # There is no fixed per-route sample cap. A route ends only after
        # reaching its destination, collision/stationary protection, or after
        # the whole traffic-tier remainder has been collected. The caller
        # creates a fresh ego vehicle and a fresh destination when needed.
        return route_saved, len(background)
    finally:
        destroy_actors(actors)
        for _ in range(5):
            world.tick()


def write_collection_config(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    path = os.path.join(args.output_dir, "collection_config.json")
    payload = {
        "collection_version": 2,
        "towns": list(args.towns),
        "traffic_tiers_per_town": [
            {"name": name, "samples": samples,
             "background_vehicle_range": [minimum, maximum]}
            for name, samples, minimum, maximum in TRAFFIC_TIERS
        ],
        "samples_per_town": sum(item[1] for item in TRAFFIC_TIERS),
        "total_target_samples": len(args.towns) * sum(
            item[1] for item in TRAFFIC_TIERS
        ),
        "simulation_hz": 20,
        "stride": args.stride,
        "sampling_interval_seconds": args.stride / 20.0,
        "weather_presets": list(args.weather),
        "weather_schedule": "seeded random per route",
        "behavior_agent": args.behavior,
        "sensor_config": SENSOR_CONFIG,
        "seed": args.seed,
    }
    if os.path.isfile(path):
        with open(path, "r") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise RuntimeError(
                "Existing dataset configuration differs from V2 settings. "
                "Use a new --output-dir instead of mixing collection rules: {}"
                .format(path)
            )
        return
    manifest_path = os.path.join(args.output_dir, "manifest.jsonl")
    if os.path.isfile(manifest_path):
        raise RuntimeError(
            "Dataset manifest exists without a matching V2 configuration: {}"
            .format(manifest_path)
        )
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def existing_tier_progress(output_dir, town, tier_name):
    """Return existing sample count and next route id for safe resume."""
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    prefix = "{}_{}_route".format(town, tier_name)
    count = 0
    route_ids = []
    if not os.path.isfile(manifest_path):
        return count, 0
    with open(manifest_path, "r") as manifest:
        for line in manifest:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            scenario = str(entry.get("scenario", ""))
            if not scenario.startswith(prefix):
                continue
            count += 1
            try:
                route_ids.append(int(scenario.rsplit("route", 1)[1]))
            except (IndexError, ValueError):
                pass
    return count, (max(route_ids) + 1 if route_ids else 0)


def main():
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    args.output_dir = os.path.abspath(args.output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    write_collection_config(args)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)
    world = None
    original_settings = None
    try:
        for town in args.towns:
            world = client.load_world(town)
            original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            for tier_name, tier_samples, minimum_bg, maximum_bg in TRAFFIC_TIERS:
                existing, route_index = existing_tier_progress(
                    args.output_dir, town, tier_name
                )
                remaining = max(0, tier_samples - existing)
                if existing:
                    print(
                        "resume {} {}: {}/{} pairs already exist".format(
                            town, tier_name, existing, tier_samples
                        )
                    )
                while remaining > 0:
                    target = remaining
                    # Derive route randomness from stable identifiers so a
                    # resumed collection selects the same weather, traffic
                    # count, spawn order and destination for that route id.
                    route_seed = "{}:{}:{}:{}".format(
                        args.seed, town, tier_name, route_index
                    )
                    route_rng = random.Random(route_seed)
                    weather_name = route_rng.choice(args.weather)
                    world.set_weather(
                        getattr(carla.WeatherParameters, weather_name)
                    )
                    world.tick()
                    requested_background = route_rng.randint(
                        minimum_bg, maximum_bg
                    )
                    print(
                        "starting {} {} route {}: weather={}, background={}".format(
                            town, tier_name, route_index, weather_name,
                            requested_background,
                        )
                    )
                    saved, actual_background = collect_route(
                        world, traffic_manager, town, tier_name, route_index,
                        requested_background, target, weather_name, args,
                        route_rng,
                    )
                    print(
                        "completed {} {} route {}: {} pairs, background {}/{}".format(
                            town, tier_name, route_index, saved,
                            actual_background, requested_background
                        )
                    )
                    remaining -= saved
                    route_index += 1
            print(
                "completed {}: {} synchronized pairs".format(
                    town, sum(item[1] for item in TRAFFIC_TIERS)
                )
            )
        print(
            "Dataset complete: {} synchronized pairs".format(
                len(args.towns) * sum(item[1] for item in TRAFFIC_TIERS)
            )
        )
        print("Dataset directory: {}".format(args.output_dir))
    finally:
        traffic_manager.set_synchronous_mode(False)
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass


if __name__ == "__main__":
    main()
