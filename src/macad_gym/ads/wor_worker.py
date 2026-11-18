"""Private inference worker: EOF or parent death stops it; never owns CARLA."""
import argparse
import ctypes
import importlib.util
import json
import os
import pickle
import signal
import struct
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--parent-pid", type=int, required=True)
parser.add_argument("--config-json", required=True)
args = parser.parse_args()
if sys.platform.startswith("linux"):
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
if os.getppid() != args.parent_pid:
    sys.exit(1)

def send(record):
    data = pickle.dumps(record, protocol=4)
    sys.stdout.buffer.write(struct.pack("!I", len(data)) + data)
    sys.stdout.buffer.flush()

def read_exact(length):
    result = bytearray()
    while len(result) < length:
        data = sys.stdin.buffer.read(length - len(result))
        if not data:
            raise EOFError()
        result.extend(data)
    return bytes(result)

try:
    config = json.loads(args.config_json)
    spec = importlib.util.spec_from_file_location("wor_private_runtime", os.path.join(os.path.dirname(__file__), "wor.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch.set_num_threads(2)
    module.torch.manual_seed(int(config.get("seed", 2026)))
    module.torch.cuda.manual_seed_all(int(config.get("seed", 2026)))
    module.torch.backends.cudnn.deterministic = True
    module.torch.backends.cudnn.benchmark = False
    module.np.random.seed(int(config.get("seed", 2026)))
    runtime = module.WORRuntime(config)
    send({"ok": True, "profile": runtime.profile, "control_metadata": runtime.control_metadata,
          "runtime": {"python": sys.executable, "torch": module.torch.__version__,
                      "device": str(runtime.device), "pid": os.getpid()}})
except Exception:
    send({"ok": False, "error": traceback.format_exc()})
    sys.exit(1)

while True:
    try:
        length = struct.unpack("!I", read_exact(4))[0]
        if length > 4 * 1024 * 1024:
            raise RuntimeError("Invalid private-pipe request size")
        request = pickle.loads(read_exact(length))
        if request["operation"] == "reset":
            runtime.reset(request["route"])
            send({"ok": True})
        elif request["operation"] == "infer":
            send(dict(runtime.infer(request["inputs"], request["timestamp"]), ok=True))
        else:
            raise RuntimeError("Unknown WOR request")
    except EOFError:
        break
    except Exception:
        send({"ok": False, "error": traceback.format_exc()})
        break
