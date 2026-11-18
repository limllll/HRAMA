"""Own simulator process groups even when a Ray worker exits without atexit.

Linux servers are launched by a small, standard-library-only supervisor. A pipe
held only by the worker detects SIGKILL/worker crashes; EOF terminates the whole
server group, including CarlaUE4.sh's descendants. PID start times, boot IDs and
an inherited random token protect unrelated processes during stale-lease cleanup.
"""
import argparse
import atexit
import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid

TOKEN_KEY = "MACAD_SIMULATOR_LEASE_TOKEN"
DEFAULT_REGISTRY = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "artifacts", "simulator_processes"))
_OWNED = {}


def _stat(pid):
    try:
        with open("/proc/{}/stat".format(pid)) as stream:
            fields = stream.read().rsplit(") ", 1)[1].split()
        return {"state": fields[0], "pgid": int(fields[2]), "start": fields[19]}
    except (IOError, OSError, IndexError, ValueError):
        return None


def _boot_id():
    with open("/proc/sys/kernel/random/boot_id") as stream:
        return stream.read().strip()


def _alive(pid, start):
    stat = _stat(pid)
    return bool(stat and stat["state"] not in ("Z", "X") and stat["start"] == start)


def _members(pgid):
    result = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            stat = _stat(int(name))
            if stat and stat["pgid"] == pgid and stat["state"] not in ("Z", "X"):
                result.append(int(name))
    return result


def _has_token(pid, token):
    try:
        with open("/proc/{}/environ".format(pid), "rb") as stream:
            return (TOKEN_KEY + "=" + token).encode() in stream.read().split(b"\0")
    except (IOError, OSError):
        return False


def _event(registry, event, **values):
    record = dict(values, event=event, time=time.time(), pid=os.getpid())
    fd = os.open(os.path.join(registry, "events.jsonl"),
                 os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def _signal_lease(lease, sig):
    pgid = int(lease["pgid"])
    if lease["boot_id"] != _boot_id() or pgid <= 1 or pgid == os.getpgrp():
        return
    members = _members(pgid)
    if not members:
        return
    if not all(_has_token(pid, lease["token"]) for pid in members):
        raise RuntimeError("Refusing to signal an unverified simulator group {}".format(pgid))
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def cleanup_orphaned_processes(registry=None):
    """Clean only dead-owner groups bearing our lease token; fail closed."""
    if not sys.platform.startswith("linux"):
        return 0
    registry = registry or DEFAULT_REGISTRY
    if not os.path.isdir(registry):
        return 0
    cleaned = 0
    for name in os.listdir(registry):
        if not name.endswith(".lease.json"):
            continue
        path = os.path.join(registry, name)
        try:
            with open(path) as stream:
                lease = json.load(stream)
            if lease["boot_id"] != _boot_id():
                os.unlink(path)
                continue
            if _alive(lease["owner_pid"], lease["owner_start"]):
                continue
            _signal_lease(lease, signal.SIGTERM)
            deadline = time.monotonic() + 3.0
            while _members(lease["pgid"]) and time.monotonic() < deadline:
                time.sleep(0.1)
            if _members(lease["pgid"]):
                _signal_lease(lease, signal.SIGKILL)
                deadline = time.monotonic() + 2.0
                while _members(lease["pgid"]) and time.monotonic() < deadline:
                    time.sleep(0.1)
            if _members(lease["pgid"]):
                raise RuntimeError("Simulator group {} cannot be stopped; refusing another launch".format(
                    lease["pgid"]))
            if os.path.exists(path):
                os.unlink(path)
            _event(registry, "orphan_cleaned", pgid=lease["pgid"])
            cleaned += 1
        except FileNotFoundError:
            # The supervisor can finish cleanup concurrently.
            continue
    return cleaned


def launch_owned_process(command, stdout=None, stderr=None, registry=None, name="simulator"):
    if not sys.platform.startswith("linux"):
        return subprocess.Popen(command, stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                                if sys.platform == "win32" else 0)
    registry = registry or DEFAULT_REGISTRY
    os.makedirs(registry, exist_ok=True)
    if cleanup_orphaned_processes(registry):
        # Give the graphics driver time to release a failed server's resources.
        time.sleep(5.0)
    read_fd, write_fd = os.pipe()
    token = uuid.uuid4().hex
    lease_path = os.path.join(registry, token + ".lease.json")
    env = os.environ.copy()
    env[TOKEN_KEY] = token
    args = [sys.executable, os.path.abspath(__file__), "--owner-fd", str(read_fd),
            "--owner-pid", str(os.getpid()), "--owner-start", _stat(os.getpid())["start"],
            "--lease", lease_path, "--name", name, "--"] + list(command)
    try:
        process = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=env,
                                   start_new_session=True, pass_fds=(read_fd,))
    except BaseException:
        os.close(write_fd)
        raise
    finally:
        os.close(read_fd)
    process._macad_owner_fd = write_fd
    process._macad_lease_path = lease_path
    _OWNED[process.pid] = process
    return process


def stop_owned_process(process, timeout=8.0):
    if process is None:
        return
    if hasattr(process, "_macad_owner_fd"):
        fd = process._macad_owner_fd
        process._macad_owner_fd = None
        if fd is not None:
            os.close(fd)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with open(process._macad_lease_path) as stream:
                lease = json.load(stream)
            _signal_lease(lease, signal.SIGKILL)
            process.wait(timeout=3.0)
        finally:
            _OWNED.pop(process.pid, None)
        if os.path.isfile(process._macad_lease_path):
            with open(process._macad_lease_path) as stream:
                lease = json.load(stream)
            if lease["boot_id"] == _boot_id() and _members(lease["pgid"]):
                raise RuntimeError("Simulator descendants are still alive; refusing another launch")
    else:
        if process.poll() is None:
            if sys.platform == "win32":
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(process.pid)])
            else:
                process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)


def stop_all_owned_processes():
    pids = list(_OWNED)
    for pid in pids:
        try:
            stop_owned_process(_OWNED[pid])
        except Exception as exc:
            print("Simulator cleanup failed for {}: {}".format(pid, exc), file=sys.stderr)
    return pids


def _supervise(args):
    registry = os.path.dirname(args.lease)
    token = os.environ[TOKEN_KEY]
    pgid = os.getpgrp()
    assert pgid == os.getpid(), "Supervisor must have a dedicated process group"
    # Adopt/reap grandchildren after the CARLA shell or an intermediate child dies.
    try:
        ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
    except (AttributeError, OSError):
        pass
    stopping = [False]
    def stop_signal(*unused):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    lease = dict(owner_pid=args.owner_pid, owner_start=args.owner_start,
                 pgid=pgid, token=token, boot_id=_boot_id(), name=args.name)
    with open(args.lease + ".tmp", "w") as stream:
        json.dump(lease, stream)
    os.replace(args.lease + ".tmp", args.lease)
    child = None
    reason = "command_failed"
    try:
        child = subprocess.Popen(args.command, close_fds=True)
        _event(registry, "started", pgid=pgid, child_pid=child.pid, name=args.name)
        while not stopping[0] and child.poll() is None:
            ready, _, _ = select.select([args.owner_fd], [], [], 0.25)
            if ready and not os.read(args.owner_fd, 1):
                reason = "owner_pipe_closed"
                break
            if not _alive(args.owner_pid, args.owner_start):
                reason = "owner_died"
                break
        else:
            reason = "stop_signal" if stopping[0] else "command_exited"
    finally:
        # Do not include this supervisor in TERM/KILL; it must reap the children.
        for sig, grace in [(signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)]:
            for pid in _members(pgid):
                if pid != os.getpid():
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                if child is not None:
                    child.poll()
                try:
                    while os.waitpid(-1, os.WNOHANG)[0]:
                        pass
                except ChildProcessError:
                    pass
                if not [pid for pid in _members(pgid) if pid != os.getpid()]:
                    break
                time.sleep(0.05)
        remaining = [pid for pid in _members(pgid) if pid != os.getpid()]
        _event(registry, "stopped", pgid=pgid, reason=reason, remaining=remaining)
        if not remaining and os.path.exists(args.lease):
            os.unlink(args.lease)
        os.close(args.owner_fd)
    return 0 if child is not None and child.returncode is None else (child.returncode or 0) if child else 1


atexit.register(stop_all_owned_processes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-fd", type=int, required=True)
    parser.add_argument("--owner-pid", type=int, required=True)
    parser.add_argument("--owner-start", required=True)
    parser.add_argument("--lease", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ["--"]:
        arguments.command = arguments.command[1:]
    sys.exit(_supervise(arguments))
