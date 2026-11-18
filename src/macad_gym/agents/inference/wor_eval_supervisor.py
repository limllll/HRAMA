"""Bounded restart of the calibrated lane evaluation; no global pkill."""
import argparse
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from evaluation_resume import inspect, atomic_json


def arguments(project, tokens, actuator_gain=1.25):
    recovery = argparse.ArgumentParser(add_help=False)
    recovery.add_argument('--resume', choices=['auto', 'never'], default='auto')
    recovery.add_argument('--max-restarts', type=int, default=5)
    recovery.add_argument('--retry-delay', type=float, default=15)
    recovery.add_argument('--stall-timeout', type=float, default=600)
    recovery.add_argument('--resume-check', action='store_true')
    controls, child_tokens = recovery.parse_known_args(tokens)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--multimodal-encoder-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--npc-count', type=int, choices=[2, 3], default=3)
    parser.add_argument('--env', default='Town04C3-v0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=80)
    parser.add_argument('--structural-ablation', default='full')
    parser.add_argument('--ego-ads', default='behavior')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--envs-per-worker', type=int, default=1)
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--multimodal-encoder-device', default='auto')
    parser.add_argument('--wor-device', default='cuda:0')
    parser.add_argument('--wor-sensor-timeout', type=float, default=10)
    parser.add_argument('--wor-sensor-workers', type=int, default=4)
    parser.add_argument('--wor-cruise-speed-cap-mps', type=float, default=None)
    parser.add_argument('--wor-turn-speed-cap-mps', type=float, default=None)
    parser.add_argument('--wor-weights', default=os.path.join(project, 'artifacts/wor_pretrained/main_model_10.th'))
    parser.add_argument('--wor-config', default=os.path.join(project, 'artifacts/wor_pretrained/config_leaderboard.yaml'))
    args, _ = parser.parse_known_args(child_tokens)
    args._wor_actuator_gain = actuator_gain
    if args.ego_ads != 'wor' or args.num_workers != 0 or args.envs_per_worker != 1:
        parser.error('This launcher requires --ego-ads wor --num-workers 0 --envs-per-worker 1')
    if args.episodes <= 0 or controls.max_restarts < 0 or controls.retry_delay < 0 or controls.stall_timeout < 60:
        parser.error('Invalid episode/restart/delay/timeout setting')
    return controls, args, child_tokens


def event(path, **record):
    record['time'] = time.strftime('%Y-%m-%d %H:%M:%S%z')
    with open(path, 'a') as stream:
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def stop_child(process):
    # This group belongs only to this subprocess. CARLA has its existing
    # token-verified process_guard and owner-pipe cleanup, not a broad pkill.
    for sig, timeout in ((signal.SIGTERM, 12), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # Also remove same-group descendants after their parent exited.
        if sig == signal.SIGTERM:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        break


def run_attempt(command, environment, output_dir, log_path, stall_timeout):
    process = subprocess.Popen(command, env=environment, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    progress_path = os.path.join(output_dir, 'wor_controls.jsonl')
    last_progress = time.monotonic()
    last_stamp = None
    timed_out = False
    try:
        with open(log_path, 'ab', buffering=0) as log:
            while True:
                ready, _, _ = select.select([process.stdout], [], [], 1)
                if ready:
                    data = os.read(process.stdout.fileno(), 65536)
                    if data:
                        log.write(data)
                        try:
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        except BrokenPipeError:
                            pass
                try:
                    stamp = os.stat(progress_path).st_mtime_ns
                except FileNotFoundError:
                    stamp = None
                if stamp != last_stamp:
                    last_stamp, last_progress = stamp, time.monotonic()
                if process.poll() is not None:
                    # Drain buffered traceback even when exit precedes the next select.
                    while True:
                        ready, _, _ = select.select([process.stdout], [], [], 0)
                        if not ready:
                            break
                        data = os.read(process.stdout.fileno(), 65536)
                        if not data:
                            break
                        log.write(data)
                        try:
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                        except BrokenPipeError:
                            pass
                    break
                if time.monotonic() - last_progress > stall_timeout:
                    timed_out = True
                    log.write(b'\nSupervisor: no WOR step progress before stall timeout.\n')
                    break
        return process.poll(), timed_out
    finally:
        stop_child(process)
        process.stdout.close()


def supervise(project, script, tokens, actuator_gain=1.25):
    if '--help' in tokens or '-h' in tokens:
        environment = dict(os.environ, HRAMA_WOR_GAIN_CHILD='1', HRAMA_WOR_ACTUATOR_GAIN=str(actuator_gain))
        code = subprocess.call([sys.executable, '-u', script, '--help'], env=environment)
        print('Recovery: --resume auto|never (default auto), --max-restarts 5, '
              '--retry-delay 15, --stall-timeout 600, --resume-check (read-only).')
        return code
    os.chdir(project)
    recovery, args, child_tokens = arguments(project, tokens, actuator_gain)
    output_dir = os.path.join(os.path.abspath(args.output_dir), 'seed{}'.format(args.seed))
    state = inspect(args, output_dir)
    count = len(state['rows'])
    print('Verified {}/{} committed episodes; next = {}; FINAL = {}'.format(
        count, args.episodes, count + 1, state['final']), flush=True)
    if recovery.resume_check:
        print('Read-only check passed; no CARLA started and no files changed.', flush=True)
        return 0
    if recovery.resume == 'never' and (state['rows'] or os.path.exists(os.path.join(output_dir, 'wor_status.json'))):
        raise ValueError('Existing evaluation found; use --resume auto or a new directory')
    if state['final']:
        if count != args.episodes:
            raise ValueError('Already-finalized evaluation cannot be extended in place')
        print('Already complete; no simulator launch.', flush=True)
        return 0
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, '.evaluation.lock'), 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another evaluation already holds this output directory')
        events = os.path.join(output_dir, 'recovery_events.jsonl')
        environment = dict(os.environ, HRAMA_WOR_GAIN_CHILD='1',
                           HRAMA_WOR_ACTUATOR_GAIN=str(actuator_gain), PYTHONHASHSEED=str(args.seed))
        command = [sys.executable, '-u', script] + child_tokens + ['--resume', 'auto']
        for attempt in range(recovery.max_restarts + 1):
            fd, log_path = tempfile.mkstemp(prefix='attempt-', suffix='.log', dir=output_dir)
            os.close(fd)
            event(events, event='attempt_started', completed=count, attempt=attempt + 1, log=log_path)
            print('Attempt {}: continue episode {}; log {}'.format(attempt + 1, count + 1, log_path), flush=True)
            try:
                code, stalled = run_attempt(command, environment, output_dir, log_path, recovery.stall_timeout)
            except KeyboardInterrupt:
                event(events, event='user_interrupted', attempt=attempt + 1)
                print('Stopped by user; completed episode results remain resumable.', flush=True)
                return 130
            state = inspect(args, output_dir)
            count = len(state['rows'])
            event(events, event='attempt_exited', attempt=attempt + 1, exit_code=code,
                  stalled=stalled, completed=count, final=state['final'])
            if state['final'] and count == args.episodes:
                print('Completed {}/{} episodes with FINAL row.'.format(count, args.episodes), flush=True)
                return 0
            # Usage/config/audit integrity failures are not transient CARLA failures.
            with open(log_path, 'rb') as stream:
                tail = stream.read()[-16000:].decode('utf-8', errors='replace')
            if code == 2 or any(message in tail for message in (
                    'Resume configuration/weights differ', 'resume provenance differs',
                    'Applied ego control differs', 'Ego was not actually controlled',
                    'WOR control count differs', 'CSV schema changed')):
                print('Non-transient configuration/audit error; see ' + log_path, flush=True)
                return 1
            if attempt == recovery.max_restarts:
                print('Restart limit reached at {} episodes; rerun the same command to continue. Log: {}'.format(count, log_path), flush=True)
                return 1
            print('Evaluation exited; retry in {}s from episode {}.'.format(recovery.retry_delay, count + 1), flush=True)
            try:
                time.sleep(recovery.retry_delay)
            except KeyboardInterrupt:
                return 130
    return 1
