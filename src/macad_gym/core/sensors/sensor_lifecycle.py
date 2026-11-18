"""Explicit, ordered sensor teardown for CARLA's asynchronous data streams."""
import threading
import time


def unique_sensors(sensors):
    """An actor can be referenced by several managers; close it only once."""
    result, seen = [], set()
    for sensor in sensors:
        if sensor is None:
            continue
        key = getattr(sensor, "id", id(sensor))
        if key not in seen:
            seen.add(key)
            result.append(sensor)
    return result


def stop_sensors(sensors, ensure_server_alive=None):
    """Stop every live listener before any sensor or parent is destroyed."""
    for sensor in unique_sensors(sensors):
        if ensure_server_alive is not None:
            ensure_server_alive()
        try:
            if sensor.is_alive and sensor.is_listening:
                sensor.stop()
        except Exception as exc:
            # A server-side deletion racing with our cleanup is harmless.
            try:
                already_dead = not sensor.is_alive
            except Exception:
                already_dead = False
            if not already_dead:
                # A transport failure must not be repeated for every actor.
                raise RuntimeError("Could not stop sensor {}: {}".format(
                    getattr(sensor, "id", "?"), exc)) from exc


def destroy_actor_ids(client, actor_ids, ensure_server_alive=None):
    """One bounded CARLA batch RPC, with no implicit simulation tick.

    Uses the command API demonstrated by CARLA's generate_traffic.py. The
    caller owns the client's timeout and the explicit physics-step barrier.
    Never fall back to per-actor RPCs after an unsuccessful batch.
    """
    actor_ids = list(dict.fromkeys(int(actor_id) for actor_id in actor_ids))
    if not actor_ids:
        return
    if ensure_server_alive is not None:
        ensure_server_alive()
    import carla
    responses = client.apply_batch_sync(
        [carla.command.DestroyActor(actor_id) for actor_id in actor_ids], False)
    if len(responses) != len(actor_ids):
        raise RuntimeError("Incomplete actor destruction batch response")
    errors = []
    for actor_id, response in zip(actor_ids, responses):
        error = response.error
        if error and "not found" not in error.lower():
            errors.append("{}: {}".format(actor_id, error))
    if errors:
        raise RuntimeError("Could not destroy actors: " + "; ".join(errors))
    if ensure_server_alive is not None:
        ensure_server_alive()


def destroy_sensors(sensors, client=None, ensure_server_alive=None):
    """Destroy stopped sensors, tolerating actors that are already dead."""
    sensors = unique_sensors(sensors)
    if client is not None:
        live_sensors = [sensor for sensor in sensors if sensor.is_alive]
        if any(sensor.is_listening for sensor in live_sensors):
            raise RuntimeError("Cannot destroy sensors while listeners are active")
        destroy_actor_ids(client, [sensor.id for sensor in live_sensors],
                          ensure_server_alive)
        return
    for sensor in sensors:
        if ensure_server_alive is not None:
            ensure_server_alive()
        try:
            if sensor.is_alive:
                if sensor.is_listening:
                    raise RuntimeError("listener is still active")
                if sensor.destroy() is False and sensor.is_alive:
                    raise RuntimeError("destroy returned False for a live sensor")
        except Exception as exc:
            try:
                already_dead = not sensor.is_alive
            except Exception:
                already_dead = False
            if not already_dead:
                raise RuntimeError("Could not destroy sensor {}: {}".format(
                    getattr(sensor, "id", "?"), exc)) from exc


class SensorCallbackGate(object):
    """Drain in-flight callbacks and reject frames from previous episodes."""

    def __init__(self):
        self._condition = threading.Condition()
        self._generation = 0
        self._closing = False
        self._active = 0

    @property
    def generation(self):
        with self._condition:
            return self._generation

    def enter(self, generation):
        with self._condition:
            if self._closing or generation != self._generation:
                return False
            self._active += 1
            return True

    def leave(self):
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def begin_close(self):
        with self._condition:
            if not self._closing:
                self._closing = True
                self._generation += 1

    def wait_for_idle(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._active:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("Timed out draining {} LiDAR callback(s)".format(
                        self._active))
                self._condition.wait(remaining)

    def reopen(self):
        with self._condition:
            if self._active:
                raise RuntimeError("Cannot start sensors while old callbacks are active")
            self._closing = False
