"""Shared opt-in WOR arguments, strict control audit and failure reporting."""
import atexit
import json
import os


def add_wor_arguments(parser, project_root):
    parser.add_argument("--ego-ads", choices=["behavior", "wor"], default="behavior")
    parser.add_argument("--wor-root", default=os.path.join(project_root, "third_party/WorldOnRails"))
    parser.add_argument("--wor-config", default=os.path.join(project_root, "artifacts/wor_pretrained/config_leaderboard.yaml"))
    parser.add_argument("--wor-weights", default=os.path.join(project_root, "artifacts/wor_pretrained/main_model_10.th"))
    parser.add_argument("--wor-device", default="cuda:0", help="Explicit WOR device; never automatically fall back")
    parser.add_argument("--wor-python", default="/home/anaconda3/envs/world_on_rails02/bin/python",
                        help="Dedicated WOR inference interpreter; PPO remains in macad-gym")
    parser.add_argument("--wor-sensor-timeout", type=float, default=10.0)
    parser.add_argument("--wor-sensor-workers", type=int, default=4,
                        help="WOR-only CARLA callback threads (1-32); does not advance the world")
    parser.add_argument("--wor-cruise-speed-cap-mps", type=float, default=None,
                        help="Straight/lane-change throttle cutoff (m/s); omitted keeps official 20km/h")
    parser.add_argument("--wor-turn-speed-cap-mps", type=float, default=None,
                        help="Turn throttle cutoff (m/s); omitted keeps official 10km/h")


def configure_wor(args, parser, actor_configs, env_settings):
    if args.ego_ads != "wor":
        return
    from macad_gym.ads.wor import resolve_speed_caps, WORFailure
    if not 1 <= args.wor_sensor_workers <= 32:
        parser.error("--wor-sensor-workers must be in [1, 32]")
    if not 0 < args.wor_sensor_timeout <= 60:
        parser.error("--wor-sensor-timeout must be in (0, 60]")
    try:
        resolve_speed_caps({"cruise_speed_cap_mps": args.wor_cruise_speed_cap_mps,
                            "turn_speed_cap_mps": args.wor_turn_speed_cap_mps})
    except (WORFailure, ValueError, TypeError) as exc:
        parser.error(str(exc))
    if args.num_workers != 0 or args.envs_per_worker != 1:
        parser.error("WOR evaluation requires --num-workers 0 --envs-per-worker 1; only one CARLA clock")
    if args.rss_counterfactual:
        parser.error("Run WOR and RSS counterfactual evaluations separately")
    if args.output_dir == parser.get_default("output_dir"):
        parser.error("WOR requires a separate --output-dir; existing BehaviorAgent results are preserved")
    result_dir = os.path.join(os.path.abspath(args.output_dir), "seed{}".format(args.seed))
    if os.path.isdir(result_dir):
        existing = [name for name in os.listdir(result_dir) if
                    (name.startswith("inference_") and name.endswith("_Results.csv")) or
                    name in ("wor_controls.jsonl", "wor_status.json", "wor_model_provenance.json",
                             "wor_sensor_failures.jsonl")]
        if existing and not getattr(args, '_wor_resume_validated', False):
            parser.error("WOR output already contains results/audit {}; choose a new --output-dir".format(existing))
    for key in ("wor_root", "wor_config", "wor_weights"):
        setattr(args, key, os.path.abspath(os.path.expanduser(getattr(args, key))))
    for path in (os.path.join(args.wor_root, "autoagents/image_agent.py"), args.wor_config, args.wor_weights):
        if not os.path.isfile(path):
            raise IOError("Required WOR asset not found: {}".format(path))
    if not os.path.isfile(args.wor_python):
        raise IOError("Dedicated WOR Python not found: {}".format(args.wor_python))
    actor_configs["actors"]["ego"]["required_external_controller"] = "wor"
    actor_configs["actors"]["ego"]["auto_control"] = True
    env_settings["wor_controller"] = {
        "root": args.wor_root, "config_path": args.wor_config, "weights": args.wor_weights,
        "device": args.wor_device, "sensor_timeout": args.wor_sensor_timeout,
        "sensor_workers": args.wor_sensor_workers,
        "sensor_diagnostic_log": os.path.join(result_dir, "wor_sensor_failures.jsonl"),
        "worker_python": args.wor_python, "seed": args.seed,
        "cruise_speed_cap_mps": args.wor_cruise_speed_cap_mps,
        "turn_speed_cap_mps": args.wor_turn_speed_cap_mps,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "worker_log": os.path.join(os.path.abspath(args.output_dir), "seed{}".format(args.seed), "wor_worker.log"),
    }
    env_settings["carla_extra_args"] = ["-vulkan"]
    env_settings["cummulated_reward"] = False  # Never append evaluation to PPO training ledgers.
    # Recovery is initialized even when appending is disabled. Do not create
    # or trim a legacy reward file during evaluation initialization either.
    env_settings["cummulated_reward_filename"] = None


class WORAudit:
    def __init__(self, args, output_dir):
        self.enabled = args.ego_ads == "wor"
        self.controller = None
        self.env = None
        self.pending_record = None
        self.path = os.path.join(output_dir, "wor_status.json")
        self.output_dir = output_dir
        self.complete = False
        self.episode_counts = list(getattr(args, '_wor_resume_counts', []))
        self.count = sum(self.episode_counts)
        self.resuming = getattr(args, '_wor_resume_validated', False)
        self.trace = None
        if self.enabled:
            self.trace = open(os.path.join(output_dir, "wor_controls.jsonl"), "a" if self.resuming else "x")
            self._save_status("initializing")
            atexit.register(self.close)

    def _save_status(self, status):
        with open(self.path, "w") as stream:
            json.dump({"status": status, "ego_ads": "wor", "fallback_allowed": False,
                       "fallback_steps": 0, "wor_steps": self.count,
                       "completed_episodes": len(self.episode_counts),
                       "episode_wor_steps": self.episode_counts}, stream, indent=2)

    def attach(self, env):
        if not self.enabled:
            return
        self.controller = env._external_ego_controller
        self.env = env
        if self.controller is None:
            raise RuntimeError("WOR controller not created; refusing BehaviorAgent evaluation")
        provenance_path = os.path.join(self.output_dir, 'wor_model_provenance.json')
        if self.resuming and os.path.isfile(provenance_path):
            from evaluation_resume import validate_provenance
            with open(provenance_path) as stream:
                validate_provenance(json.load(stream), self.controller.metadata)
        else:
            with open(provenance_path, 'x') as stream:
                json.dump(self.controller.metadata, stream, indent=2)
        self._save_status("running")

    def before_step(self, env, episode, step):
        if not self.enabled:
            return
        control = self.controller.compute_control()
        env.set_external_actor_control("ego", control, source="wor")
        self.pending_record = dict(self.controller.last_record, episode=episode, step=step)

    def after_step(self, info):
        if self.enabled:
            if info.get("ego", {}).get("control_source") != "wor":
                raise RuntimeError("Ego was not actually controlled by WOR; episode rejected")
            applied = info["ego"]["autocontrol"]
            for name in ("throttle", "steer", "brake"):
                if abs(applied[name] - self.pending_record[name]) > 1e-6:
                    raise RuntimeError("Applied ego control differs from WOR output")
            self.trace.write(json.dumps(dict(self.pending_record, phase="applied")) + "\n")
            self.trace.flush()
            self.pending_record = None
            self.count += 1

    def finish_episode(self, step):
        if self.enabled:
            if self.controller.episode_steps != step:
                raise RuntimeError("WOR control count differs from episode steps")
            self.episode_counts.append(step)
            self.trace.flush()
            os.fsync(self.trace.fileno())
            self._save_status("running")

    def finish(self):
        if self.enabled:
            self.complete = True
            self._save_status("completed")
            self.trace.close()
            self.controller.runtime.close()

    def close(self):
        if self.enabled and not self.complete:
            self._save_status("aborted; no controller fallback and no FINAL success claim")
            if self.trace is not None:
                self.trace.close()
            try:
                if self.env is not None:
                    self.env.close()
            finally:
                if self.controller is not None:
                    self.controller.runtime.close()
