"""Offline WOR transport regressions: no simulator, model, GPU or Ray needed.

Run from the project: python tests/test_wor_sensor_transport.py
WOR_TEST_SOURCE_DIR can point at a staged ads/ directory before deployment.
"""
import argparse
import ast
import importlib.util
import json
import os
import tempfile
import threading
import time
import types
import unittest

import numpy as np


SOURCE = os.environ.get("WOR_TEST_SOURCE_DIR", os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "src", "macad_gym", "ads")))


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SOURCE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


buffer_module = load_file("wor_buffer_test", "wor_sensor_buffer.py")
Buffer = buffer_module.WORFrameBuffer
Failure = buffer_module.WORSensorFailure
NAMES = buffer_module.SENSOR_NAMES


def fake_controller():
    with open(os.path.join(SOURCE, "wor.py")) as stream:
        tree = ast.parse(stream.read())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and
             node.name in ("WORFailure", "WORController")]
    scope = {"np": np, "time": time, "os": os, "json": json,
             "WORFrameBuffer": Buffer, "WORSensorFailure": Failure,
             "carla": types.SimpleNamespace(VehicleControl=lambda **kw: types.SimpleNamespace(**kw))}
    exec(compile(ast.Module(body=nodes), "wor.py", "exec"), scope)
    controller = object.__new__(scope["WORController"])
    controller.sensor_workers = 4
    controller.timeout = .1
    controller._listener_client = None
    controller._listener_world = None
    controller._listener_port = None
    controller._sensor_buffer = Buffer()
    controller._generation = 1
    controller._sensor_buffer.start(1)
    controller._sensors = []
    controller._diagnostic_path = None
    controller._env = None
    controller._vehicle = None
    controller._last_sensor_wait_ms = None
    controller.episode_steps = 0
    return controller, scope


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.buffer = Buffer()
        self.buffer.start(1)

    def fill(self, frame, generation=1):
        for name in NAMES:
            self.buffer.store(name, generation, frame, (name, frame))

    def start_wait(self, frame=10, timeout=.3):
        result = {}
        def run():
            try:
                result["data"] = self.buffer.wait(frame, timeout)
            except Exception as exc:
                result["error"] = exc
        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 1.
        while thread.is_alive():
            with self.buffer.condition:
                if (1, frame) in self.buffer.waiters:
                    break
            if time.monotonic() > deadline:
                self.fail("test waiter did not start")
            time.sleep(.001)
        return thread, result

    def test_complete_exact_frame(self):
        self.fill(10)
        self.assertEqual(set(self.buffer.wait(10, .01)), set(NAMES))

    def test_neighbors_never_replace_missing_frame(self):
        self.fill(9)
        self.fill(11)
        with self.assertRaises(Failure) as caught:
            self.buffer.wait(10, .01)
        self.assertEqual(caught.exception.diagnostic["missing_sensors"], list(NAMES))

    def test_newer_gps_never_replaces_requested_gps(self):
        for name in NAMES[:-1]:
            self.buffer.store(name, 1, 10, name)
        self.buffer.store("gps", 1, 11, "gps-new")
        with self.assertRaises(Failure) as caught:
            self.buffer.wait(10, .01)
        self.assertEqual(caught.exception.diagnostic["missing_sensors"], ["gps"])

    def test_reordered_delayed_callbacks(self):
        self.fill(11)
        thread, result = self.start_wait()
        self.fill(10)
        thread.join(1.)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["data"]["gps"], ("gps", 10))
        self.assertEqual(self.buffer.stats["gps"]["out_of_order"], 1)

    def test_waiting_frame_pinned_while_streams_run_ahead(self):
        self.buffer.store("left", 1, 10, "original-left")
        thread, result = self.start_wait()
        for frame in range(11, 80):
            self.fill(frame)
        self.assertIn(10, self.buffer.frames)
        for name in NAMES[1:]:
            self.buffer.store(name, 1, 10, name)
        thread.join(1.)
        self.assertEqual(result["data"]["left"], "original-left")
        self.assertLessEqual(len(self.buffer.frames), 17)
        self.assertNotIn(10, self.buffer.frames)

    def test_cache_bounded_with_late_callbacks(self):
        for frame in range(100):
            self.fill(frame)
        self.fill(0)
        self.assertLessEqual(len(self.buffer.frames), 17)
        self.assertNotIn(0, self.buffer.frames)

    def test_detach_cancels_wait_without_timeout(self):
        thread, result = self.start_wait(timeout=5.)
        self.buffer.close(2)
        thread.join(.5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["error"].diagnostic["reason"], "episode_detached_or_replaced")

    def test_new_episode_cannot_complete_old_wait(self):
        thread, result = self.start_wait(timeout=5.)
        self.buffer.start(2)
        self.fill(10, generation=2)
        thread.join(.5)
        self.assertFalse(thread.is_alive())
        self.assertIn("error", result)
        self.assertEqual(result["error"].diagnostic["requested_generation"], 1)
        self.assertEqual(set(self.buffer.wait(10, .01)), set(NAMES))

    def test_old_callback_and_old_error_ignored(self):
        self.buffer.start(2)
        self.assertFalse(self.buffer.store("gps", 1, 10, "old"))
        self.assertFalse(self.buffer.fail_callback("gps", 1, "old failure"))
        self.fill(10, generation=2)
        self.assertEqual(self.buffer.wait(10, .01)["gps"], ("gps", 10))

    def test_current_callback_error_fails_even_with_full_frame(self):
        self.fill(10)
        self.buffer.fail_callback("gps", 1, "malformed data")
        with self.assertRaises(Failure) as caught:
            self.buffer.wait(10, .01)
        self.assertEqual(caught.exception.diagnostic["reason"], "callback_error")

    def test_timeout_diagnostic_is_serializable_and_actionable(self):
        self.buffer.store("left", 1, 7, "a")
        self.buffer.store("left", 1, 10, "b")
        self.buffer.store("left", 1, 10, "b-again")
        with self.assertRaises(Failure) as caught:
            self.buffer.wait(10, .01)
        record = json.loads(json.dumps(caught.exception.diagnostic))
        self.assertEqual(record["sensor_stats"]["left"]["largest_forward_gap"], 2)
        self.assertEqual(record["sensor_stats"]["left"]["duplicate"], 1)
        self.assertEqual(record["sensor_stats"]["gps"]["received"], 0)
        self.assertFalse(record["fallback_allowed"])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller, self.scope = fake_controller()

    def test_client_is_separate_reused_and_never_ticks(self):
        controller = self.controller
        vehicle = types.SimpleNamespace(id=42, is_alive=True)
        world = types.SimpleNamespace(id=8, get_actor=lambda ident: vehicle)
        calls = []
        class Client:
            def __init__(self, host, port, workers):
                calls.append((host, port, workers))
            def set_timeout(self, value):
                pass
            def get_world(self):
                return world
        self.scope["carla"].Client = Client
        controller._vehicle = vehicle
        env = types.SimpleNamespace(_server_port=4000, _rpc_timeout=20., world=world)
        self.assertEqual(controller._sensor_world(env), (world, vehicle))
        controller._sensor_world(env)
        self.assertEqual(calls, [("localhost", 4000, 4)])
        env._server_port = 4001
        controller._sensor_world(env)
        self.assertEqual(len(calls), 2)
        # No tick/settings/reload API exists on this stub; any use would fail.

    def test_wrong_world_rejected(self):
        client = types.SimpleNamespace(set_timeout=lambda value: None,
                                       get_world=lambda: types.SimpleNamespace(id=9))
        self.scope["carla"].Client = lambda *args: client
        controller = self.controller
        controller._vehicle = types.SimpleNamespace(id=42)
        with self.assertRaises(self.scope["WORFailure"]):
            controller._sensor_world(types.SimpleNamespace(_server_port=4000,
                                                          world=types.SimpleNamespace(id=8)))

    def test_detach_keeps_client_alive_for_sensor_cleanup(self):
        controller = self.controller
        client = object()
        controller._listener_client = client
        sensors = [object(), object()]
        controller._sensors = sensors
        self.assertEqual(controller.detach_sensors(), sensors)
        self.assertIs(controller._listener_client, client)
        self.assertFalse(controller.ready(10))
        controller.abandon_episode()
        self.assertIsNone(controller._listener_client)

    def test_gate_leave_on_bad_payload_and_old_generation(self):
        counts = []
        gate = types.SimpleNamespace(enter=lambda generation: True,
                                     leave=lambda: counts.append("leave"))
        callback = self.controller._callback("left", 1, gate, 1)
        callback(types.SimpleNamespace(frame=10, raw_data=b"bad", height=240, width=160))
        self.assertEqual(counts, ["leave"])
        self.assertEqual(self.controller._sensor_buffer.callback_error["sensor"], "left")
        self.controller._sensor_buffer.start(2)
        callback(types.SimpleNamespace())
        self.assertEqual(counts, ["leave", "leave"])
        self.assertIsNone(self.controller._sensor_buffer.callback_error)

    def test_error_during_old_decode_cannot_poison_new_episode(self):
        controller = self.controller
        counts = []
        gate = types.SimpleNamespace(enter=lambda generation: True,
                                     leave=lambda: counts.append("leave"))
        def interrupted_decode(*args, **kwargs):
            controller._sensor_buffer.start(2)
            raise ValueError("old frame became malformed during teardown")
        self.scope["np"] = types.SimpleNamespace(frombuffer=interrupted_decode, uint8=np.uint8)
        controller._callback("left", 1, gate, 1)(
            types.SimpleNamespace(frame=10, raw_data=b"old"))
        self.assertIsNone(controller._sensor_buffer.callback_error)
        self.assertEqual(counts, ["leave"])

    def test_bind_spawns_five_official_sensors_on_listener_only(self):
        controller = self.controller
        controller.logic = types.SimpleNamespace(camera_x=1.5, camera_z=2.4)
        controller._downsample = lambda route, distance: [0]
        controller.runtime = types.SimpleNamespace(reset=lambda route: None)
        records = []
        vehicle = types.SimpleNamespace(id=42, is_alive=True)
        class Blueprint:
            def __init__(self, kind):
                self.kind, self.attributes = kind, {}
            def has_attribute(self, name):
                return True
            def set_attribute(self, name, value):
                self.attributes[name] = value
        class Sensor:
            def __init__(self, ident):
                self.id = ident
            def listen(self, callback):
                self.callback = callback
        def spawn(blueprint, transform, attach_to):
            self.assertIs(attach_to, vehicle)
            records.append((blueprint, transform))
            return Sensor(len(records))
        listener_world = types.SimpleNamespace(id=8, get_actor=lambda ident: vehicle,
            get_blueprint_library=lambda: types.SimpleNamespace(find=Blueprint), spawn_actor=spawn)
        client = types.SimpleNamespace(set_timeout=lambda value: None, get_world=lambda: listener_world)
        self.scope["carla"] = types.SimpleNamespace(Client=lambda *args: client,
            Location=lambda **kw: types.SimpleNamespace(**kw),
            Rotation=lambda **kw: types.SimpleNamespace(**kw),
            Transform=lambda location, rotation=None: types.SimpleNamespace(location=location, rotation=rotation))
        world_map = types.SimpleNamespace(transform_to_geolocation=lambda loc:
            types.SimpleNamespace(latitude=1., longitude=2., altitude=3.))
        # Environment world has no spawn/tick/settings method: sensors must be
        # owned by the separate listener, and no hidden warm-up tick is allowed.
        world = types.SimpleNamespace(id=8, get_map=lambda: world_map)
        gate = types.SimpleNamespace(generation=1, enter=lambda gen: True, leave=lambda: None)
        waypoint = types.SimpleNamespace(transform=types.SimpleNamespace(location=object()))
        agent = types.SimpleNamespace(_local_planner=types.SimpleNamespace(
            _waypoints_queue=[(waypoint, types.SimpleNamespace(value=4))]))
        env = types.SimpleNamespace(_server_port=4000, world=world,
            _actors={"ego": vehicle}, _behavior_agents={"ego": agent}, _sensor_callback_gate=gate)
        controller.bind_episode(env)
        self.assertEqual(len(controller._sensors), 5)
        self.assertEqual([record[0].attributes["image_size_x"] for record in records[:4]],
                         ["160", "160", "160", "384"])
        self.assertEqual([record[1].rotation.yaw for record in records[:4]], [-55, 0, 55, 0])
        self.assertEqual([record[0].attributes["fov"] for record in records[:4]], ["60", "60", "60", "50"])
        self.assertEqual(records[-1][0].kind, "sensor.other.gnss")
        self.assertEqual(set(records[-1][0].attributes.values()), {"0"})
        self.assertEqual(records[-1][1].location.z, 2.4)

    def test_failure_log_and_exception_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            self.controller._diagnostic_path = os.path.join(directory, "failures.jsonl")
            with self.assertRaises(self.scope["WORFailure"]):
                self.controller._wait_frame(10)
            with open(self.controller._diagnostic_path) as stream:
                record = json.loads(stream.readline())
            self.assertEqual(record["reason"], "same_frame_timeout")
            self.assertFalse(record["fallback_allowed"])

    def test_compute_uses_only_requested_frame_and_unchanged_control(self):
        controller = self.controller
        for name in NAMES:
            width = 384 if name == "narrow" else 160
            value = np.array([1., 2., 3.]) if name == "gps" else np.full((240, width, 3), 7, np.uint8)
            controller._sensor_buffer.store(name, 1, 10, value)
        snapshot = types.SimpleNamespace(frame=10, timestamp=types.SimpleNamespace(elapsed_seconds=4.5))
        location = types.SimpleNamespace(x=1., y=2., z=3.)
        controller._vehicle = types.SimpleNamespace(get_velocity=lambda: 14.5,
            get_transform=lambda: object(), get_location=lambda: location)
        controller._speed_reader = types.SimpleNamespace(_get_forward_speed=lambda **kw: kw["velocity"])
        controller._env = types.SimpleNamespace(world=types.SimpleNamespace(get_snapshot=lambda: snapshot),
            _behavior_agents={"ego": types.SimpleNamespace(_local_planner=types.SimpleNamespace(_waypoints_queue=[1]))})
        seen = []
        def infer(inputs, timestamp):
            seen.append((inputs, timestamp))
            return {"control": {"throttle": .5, "steer": .1, "brake": 0.},
                    "command": 4, "postprocessing": {"neural_brake_probability": .2}}
        controller.runtime = types.SimpleNamespace(infer=infer)
        control = controller.compute_control()
        self.assertEqual((control.throttle, control.steer, control.brake), (.5, .1, 0.))
        self.assertEqual([item[0] for item in seen[0][0].values()], [10] * 4)
        self.assertEqual(seen[0][0]["Wide_RGB"][1].shape, (240, 480, 3))
        self.assertEqual(seen[0][0]["EGO"][1]["spd"], 14.5)
        self.assertEqual(controller.last_record["control_source"], "wor")
        self.assertGreaterEqual(controller.last_record["sensor_wait_ms"], 0.)

    def test_cli_workers_default_and_behavior_unchanged(self):
        module = load_file("wor_args_test", "wor_evaluation.py")
        parser = argparse.ArgumentParser()
        module.add_wor_arguments(parser, "/project")
        args = parser.parse_args([])
        self.assertEqual(args.ego_ads, "behavior")
        self.assertEqual(args.wor_sensor_workers, 4)
        self.assertEqual(args.wor_sensor_timeout, 10.)
        self.assertIsNone(args.wor_cruise_speed_cap_mps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
