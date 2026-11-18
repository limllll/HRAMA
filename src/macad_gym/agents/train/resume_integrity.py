"""Ray 0.8 recovery guards; no changes to PPO optimization or simulator control."""
import copy
import glob
import json
import os
import pickle

from ray import cloudpickle
from ray.tune.checkpoint_manager import Checkpoint, CheckpointManager
from ray.tune.ray_trial_executor import RayTrialExecutor

RECOVERY_ENV_KEYS = (
    "reward_resume_checkpoint", "reward_resume_episodes", "reward_resume_iteration",
    "reward_resume_lineage", "reward_resume_required", "reward_checkpoint_dir",
    "reward_checkpoint_scan_after",
)


def checkpoint_metadata(path):
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError("Missing/empty PPO checkpoint: {}".format(path))
    with open(path + ".tune_metadata", "rb") as stream:
        metadata = pickle.load(stream)
    for key in ("iteration", "episodes_total", "timesteps_total"):
        if metadata.get(key) is None or int(metadata[key]) < 0:
            raise RuntimeError("Invalid checkpoint metadata {}: {}".format(path, key))
    if not metadata.get("experiment_id"):
        raise RuntimeError("Checkpoint has no policy lineage: " + path)
    return metadata


def select_checkpoint(logdir, explicit=None, lineage=None):
    """Refuse ambiguous policy branches instead of taking the largest reward number."""
    if explicit:
        explicit = os.path.realpath(os.path.expanduser(explicit))
        if os.path.commonpath([explicit, os.path.realpath(logdir)]) != os.path.realpath(logdir):
            raise RuntimeError("Resume checkpoint must belong to the original trial directory")
        metadata = checkpoint_metadata(explicit)
        if lineage and metadata["experiment_id"] != lineage:
            raise RuntimeError("Explicit checkpoint belongs to a different policy lineage")
        return explicit, metadata
    candidates = []
    for path in glob.glob(os.path.join(logdir, "checkpoint_*", "*.tune_metadata")):
        checkpoint = path[:-len(".tune_metadata")]
        try:
            metadata = checkpoint_metadata(checkpoint)
        except (IOError, ValueError, EOFError, RuntimeError, pickle.UnpicklingError):
            continue
        if not lineage or metadata["experiment_id"] == lineage:
            candidates.append((checkpoint, metadata))
    if not candidates:
        raise RuntimeError("No valid PPO checkpoint for this policy lineage under " + logdir)
    if len(set(value["experiment_id"] for _, value in candidates)) != 1:
        raise RuntimeError("Multiple policy branches found; specify --resume-checkpoint explicitly")
    return max(candidates, key=lambda item: (
        int(item[1]["iteration"]), int(item[1]["timesteps_total"]),
        os.path.getmtime(item[0] + ".tune_metadata")))


def result_for_checkpoint(logdir, metadata):
    result = {}
    path = os.path.join(logdir, "result.json")
    if os.path.isfile(path):
        with open(path) as stream:
            for line in stream:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if (value.get("experiment_id") == metadata["experiment_id"] and
                        value.get("training_iteration") == metadata["iteration"] and
                        value.get("episodes_total") == metadata["episodes_total"]):
                    result = value
    result.update(training_iteration=int(metadata["iteration"]),
                  episodes_total=int(metadata["episodes_total"]),
                  timesteps_total=int(metadata["timesteps_total"]),
                  experiment_id=metadata["experiment_id"])
    return result


def pin_trial_checkpoint(trial_state, path, metadata):
    """Reconstruct the actual Ray manager, not just the visible trial status."""
    result = result_for_checkpoint(trial_state["logdir"], metadata)
    manager = CheckpointManager(None, "training_iteration", None)
    manager.on_checkpoint(Checkpoint(Checkpoint.PERSISTENT, path, result))
    trial_state["checkpoint_manager"] = {
        "_type": "CLOUDPICKLE_FALLBACK",
        "value": cloudpickle.dumps(manager).hex(),
    }
    trial_state["restore_path"] = path
    trial_state["resume_lineage_id"] = metadata["experiment_id"]
    trial_state["resume_required"] = True
    trial_state["restoring_from"] = None
    trial_state["resuming_from"] = None
    trial_state["saving_to"] = None
    trial_state["last_result"] = result


def metadata_for_ray_checkpoint(checkpoint):
    if checkpoint.storage == Checkpoint.PERSISTENT:
        return checkpoint_metadata(checkpoint.value)
    if checkpoint.storage != Checkpoint.MEMORY or not isinstance(checkpoint.value, bytes):
        raise RuntimeError("Unsupported/empty Ray recovery checkpoint")
    payload = pickle.loads(checkpoint.value)
    name = payload["checkpoint_name"]
    metadata = pickle.loads(payload["data"][name + ".tune_metadata"])
    for key in ("iteration", "episodes_total", "timesteps_total"):
        if metadata.get(key) is None or int(metadata[key]) < 0:
            raise RuntimeError("Invalid in-memory checkpoint metadata: " + key)
    if not metadata.get("experiment_id"):
        raise RuntimeError("In-memory checkpoint has no policy lineage")
    return metadata


class VerifiedResumeExecutor(RayTrialExecutor):
    """Supply the exact restore boundary to envs BEFORE worker construction.

    Old Ray may clear a checkpoint after restore fails and silently start from
    zero. A required resume must instead select a valid persistent checkpoint
    of the same lineage or fail closed. No Ray installation is modified.
    """
    def __init__(self, require_resume=False, **kwargs):
        super(VerifiedResumeExecutor, self).__init__(**kwargs)
        self._require_resume = require_resume
        self._expected_restore = {}

    def _start_trial(self, trial, checkpoint=None, runner=None):
        selected = checkpoint if checkpoint is not None and checkpoint.value is not None else trial.checkpoint
        required = self._require_resume or getattr(trial, "resume_required", False)
        lineage = getattr(trial, "resume_lineage_id", None)
        if selected.value is None:
            disk_metadata = glob.glob(os.path.join(trial.logdir, "checkpoint_*", "*.tune_metadata"))
            if required or disk_metadata:
                path, metadata = select_checkpoint(trial.logdir, lineage=lineage)
                selected = Checkpoint(Checkpoint.PERSISTENT, path,
                                      result_for_checkpoint(trial.logdir, metadata))
                trial.on_checkpoint(selected)

        recovery = trial.config.setdefault("env_config", {}).setdefault("env", {})
        if selected.value is not None:
            metadata = metadata_for_ray_checkpoint(selected)
            if lineage and metadata["experiment_id"] != lineage:
                raise RuntimeError("Ray attempted to restore the wrong policy lineage")
            trial.resume_required = True
            trial.resume_lineage_id = metadata["experiment_id"]
            self._expected_restore[trial.trial_id] = metadata
            recovery.update(
                reward_resume_checkpoint=(selected.value if selected.storage == Checkpoint.PERSISTENT else None),
                reward_resume_episodes=int(metadata["episodes_total"]),
                reward_resume_iteration=int(metadata["iteration"]),
                reward_resume_lineage=metadata["experiment_id"],
                reward_resume_required=True,
                reward_checkpoint_dir=trial.logdir,
                reward_checkpoint_scan_after=0.0,
            )
            print("Verified PPO restore target: iteration {} / completed episodes {} / lineage {}".format(
                metadata["iteration"], metadata["episodes_total"], metadata["experiment_id"]), flush=True)
        else:
            # A truly new experiment has no committed episode boundary.
            recovery.update(reward_resume_episodes=0, reward_resume_required=False)
            self._expected_restore.pop(trial.trial_id, None)
        return super(VerifiedResumeExecutor, self)._start_trial(trial, selected, runner)

    def fetch_result(self, trial):
        result = super(VerifiedResumeExecutor, self).fetch_result(trial)
        expected = self._expected_restore.get(trial.trial_id)
        if expected and isinstance(result, dict) and "training_iteration" in result:
            if (int(result["training_iteration"]) <= int(expected["iteration"]) or
                    int(result.get("episodes_total", -1)) < int(expected["episodes_total"]) or
                    result.get("experiment_id") != expected["experiment_id"]):
                raise RuntimeError("Restored PPO counters/lineage do not match the selected checkpoint; refusing fresh training")
            print("PPO continuation confirmed: iteration {} / completed episodes {}".format(
                result["training_iteration"], result["episodes_total"]), flush=True)
            self._expected_restore.pop(trial.trial_id, None)
        return result
