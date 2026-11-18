"""Exercise the private worker's real module loading, without starting a model."""
import importlib.util
import json
import os
import sys

assert not os.environ.get("PYTHONPATH"), "Use the private worker's empty PYTHONPATH"
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("wor_private_runtime", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
buffer = module.WORFrameBuffer()
buffer.start(1)
buffer.close(1)
assert issubclass(module.WORSensorFailure, RuntimeError)
for forbidden in ("macad_gym", "ray", "tensorflow"):
    assert not any(name == forbidden or name.startswith(forbidden + ".")
                   for name in sys.modules), forbidden + " leaked into isolated runtime"
assert not module.torch.cuda.is_initialized(), "Import must not initialize CUDA"
print(json.dumps({"passed": True, "python": sys.executable,
                  "module": path, "torch": module.torch.__version__,
                  "training_packages_loaded": False, "model_started": False}))
