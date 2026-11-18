#!/usr/bin/env python
"""Supervised lane WOR evaluation without any throttle multiplication.

Omitting the speed-cap arguments uses official WOR cutoffs. Passing explicit
cutoffs changes only those existing post-processing options, never the weights.
Vehicle initialization and the frozen NPC PPO policies remain unchanged.
"""
import os
import runpy
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path[:0] = [os.path.dirname(__file__), os.path.join(PROJECT, 'src'),
               os.path.join(PROJECT, 'src/macad_gym/agents')]


def main():
    if os.environ.get('HRAMA_WOR_GAIN_CHILD') != '1':
        from wor_eval_supervisor import supervise
        return supervise(PROJECT, os.path.abspath(__file__), sys.argv[1:], actuator_gain=1.0)
    # No WORController patch: use the original controller compute_control.
    os.environ['HRAMA_WOR_ACTUATOR_GAIN'] = '1.0'
    script = os.path.join(PROJECT, 'src/macad_gym/agents/inference/LaneChange_ppo_inference_structural.py')
    sys.argv[0] = script
    os.chdir(PROJECT)
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    sys.exit(main())
