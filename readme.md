# HRAMA

Source-code snapshot of HRAMA for adversarial multi-agent simulation testing of autonomous driving systems.

## Main components

- `src/macad_gym/carla/`: CARLA environment, SUMO bridge, reward and trajectory-processing code.
- `src/macad_gym/agents/train/LaneChange_ppo_train.py`: lane-changing PPO training.
- `src/macad_gym/agents/train/Following_ppo_train.py`: car-following PPO training.
- `src/macad_gym/agents/inference/`: evaluation scripts, including RS, OS and policy inference.
- `src/macad_gym/agents/train/models.py`: policy-network implementation.
- `src/macad_gym/PointNet2/`: point-cloud encoder code.
- `src/macad_gym/ads/` and `third_party/WorldOnRails/`: ADS integration and third-party source.
- `tests/` and `test/`: source tests.

## Environment and entry points

See `conda_env.yml`, `requirement.txt` and `setup.py` for dependency configuration. CARLA, SUMO and compatible CUDA/framework versions must be installed separately. Review script configuration and adapt local simulator, dataset and checkpoint paths before running.

```bash
pip install -e .
python src/macad_gym/agents/train/LaneChange_ppo_train.py
python src/macad_gym/agents/train/Following_ppo_train.py
```

These commands identify the training entry points; this source snapshot has not been validated as a self-contained reproduction package.

## Availability

This repository contains source code and lightweight configuration only. Raw/processed trajectories, experimental CSV files, training logs, images, videos, simulator/map assets, compiled binaries and trained model weights are not included. Obtain required external assets separately and configure their paths. Newer experimental utilities are included in this snapshot and should not be assumed to have been used in every reported paper experiment.

Existing Git history retains earlier repository versions. Third-party source retains its accompanying notices; consult upstream projects for their licenses and dependencies.
