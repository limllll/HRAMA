#!/bin/env python
"""Fine-tune a lane-change HRAMA checkpoint for attributable failure diversity."""

from macad_gym.agents.train.diversity_finetune_common import main_lane_change


if __name__ == "__main__":
    main_lane_change()

