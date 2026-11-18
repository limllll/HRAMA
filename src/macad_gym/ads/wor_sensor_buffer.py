"""Bounded, generation-aware WOR sensor inbox (no CARLA clock or inference).

Exact-frame joins only: delayed or reordered callbacks are allowed, but missing
data never resolves to a newer/older frame or a synthetic observation.
"""
import threading
import time


SENSOR_NAMES = ("left", "center", "right", "narrow", "gps")


class WORSensorFailure(RuntimeError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


class WORFrameBuffer:
    def __init__(self, keep_frames=16):
        if keep_frames < 1:
            raise ValueError("WOR frame retention must be positive")
        self.condition = threading.Condition()
        self.keep_frames = int(keep_frames)
        self.generation = 0
        self.closed = True
        self.frames = {}
        self.waiters = {}
        self.highest_frame = None
        self.callback_error = None
        self.stats = {}

    def start(self, generation):
        with self.condition:
            self.generation = int(generation)
            self.closed = False
            self.frames.clear()
            self.highest_frame = None
            self.callback_error = None
            self.stats = {name: {"received": 0, "latest_frame": None,
                                 "last_arrival": None, "out_of_order": 0,
                                 "duplicate": 0, "largest_forward_gap": 0}
                          for name in SENSOR_NAMES}
            self.condition.notify_all()

    def close(self, generation):
        with self.condition:
            self.generation = int(generation)
            self.closed = True
            self.frames.clear()
            self.callback_error = None
            self.condition.notify_all()

    def is_current(self, generation):
        with self.condition:
            return not self.closed and int(generation) == self.generation

    def _prune_locked(self):
        if self.highest_frame is None:
            return
        cutoff = self.highest_frame - self.keep_frames
        for frame in list(self.frames):
            # Waiting frames are protected even if another stream runs ahead.
            if frame < cutoff and (self.generation, frame) not in self.waiters:
                del self.frames[frame]

    def store(self, name, generation, frame, value):
        with self.condition:
            if self.closed or int(generation) != self.generation:
                return False
            if name not in self.stats:
                raise ValueError("Unknown WOR sensor: " + str(name))
            frame = int(frame)
            stats = self.stats[name]
            previous = stats["latest_frame"]
            stats["received"] += 1
            stats["last_arrival"] = time.monotonic()
            if previous is not None:
                stats["out_of_order"] += int(frame < previous)
                stats["largest_forward_gap"] = max(
                    stats["largest_forward_gap"], max(0, frame - previous - 1))
            stats["latest_frame"] = frame if previous is None else max(frame, previous)
            self.highest_frame = frame if self.highest_frame is None else max(frame, self.highest_frame)
            pair = self.frames.setdefault(frame, {})
            stats["duplicate"] += int(name in pair)
            pair[name] = value
            self._prune_locked()
            self.condition.notify_all()
            return True

    def fail_callback(self, name, generation, error):
        with self.condition:
            if self.closed or int(generation) != self.generation:
                return False
            self.callback_error = {"sensor": name, "error": str(error)}
            self.condition.notify_all()
            return True

    def ready(self, frame):
        with self.condition:
            return not self.closed and all(name in self.frames.get(int(frame), {})
                                           for name in SENSOR_NAMES)

    def _diagnostic_locked(self, frame, reason, wait_started):
        now = time.monotonic()
        stats = {}
        for name, values in self.stats.items():
            values = dict(values)
            arrival = values.pop("last_arrival")
            values["last_arrival_age_seconds"] = None if arrival is None else now - arrival
            stats[name] = values
        cached = sorted(self.frames)
        return {"reason": reason, "frame": int(frame), "generation": self.generation,
                "closed": self.closed, "missing_sensors": [name for name in SENSOR_NAMES
                    if name not in self.frames.get(int(frame), {})],
                "cached_frame_min": cached[0] if cached else None,
                "cached_frame_max": cached[-1] if cached else None,
                "cached_frame_count": len(cached), "highest_received_frame": self.highest_frame,
                "wait_seconds": now - wait_started, "sensor_stats": stats,
                "callback_error": self.callback_error, "fallback_allowed": False}

    def wait(self, frame, timeout):
        frame = int(frame)
        started = time.monotonic()
        deadline = started + float(timeout)
        with self.condition:
            generation = self.generation
            key = (generation, frame)
            self.waiters[key] = self.waiters.get(key, 0) + 1
            try:
                while True:
                    reason = None
                    if self.closed or generation != self.generation:
                        reason = "episode_detached_or_replaced"
                    elif self.callback_error:
                        reason = "callback_error"
                    elif all(name in self.frames.get(frame, {}) for name in SENSOR_NAMES):
                        return dict(self.frames[frame])
                    elif time.monotonic() >= deadline:
                        reason = "same_frame_timeout"
                    if reason:
                        diagnostic = self._diagnostic_locked(frame, reason, started)
                        diagnostic["requested_generation"] = generation
                        raise WORSensorFailure(
                            "WOR frame {}: {}, missing {}; no stale/black-frame fallback".format(
                                frame, reason, diagnostic["missing_sensors"]), diagnostic)
                    self.condition.wait(max(0., deadline - time.monotonic()))
            finally:
                self.waiters[key] -= 1
                if not self.waiters[key]:
                    del self.waiters[key]
                self._prune_locked()
