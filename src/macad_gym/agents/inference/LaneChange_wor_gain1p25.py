#!/usr/bin/env python
"""Lane-change evaluation using the same process-local WOR gain as following.

Only WOR throttle is multiplied by 1.25 and clipped; steering, braking,
speed cutoff, vehicle initialization, NPC policies and metric calculations
remain unchanged. This wrapper does not change production WOR defaults.
"""
import os
import runpy
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path[:0] = [os.path.dirname(__file__), os.path.join(PROJECT, 'src'), os.path.join(PROJECT, 'src/macad_gym/agents')]


def main():
    if os.environ.get('HRAMA_WOR_GAIN_CHILD') != '1':
        from wor_eval_supervisor import supervise
        return supervise(PROJECT, os.path.abspath(__file__), sys.argv[1:])
    if '--help' not in sys.argv and '-h' not in sys.argv:
        tokens = list(sys.argv[1:])
        if not ('--ego-ads=wor' in tokens or any(
                t == '--ego-ads' and i + 1 < len(tokens) and tokens[i + 1] == 'wor'
                for i, t in enumerate(tokens))):
            raise ValueError('This calibrated launcher requires --ego-ads wor')
    from macad_gym.ads.wor import WORController
    original_init = WORController.__init__
    original_compute = WORController.compute_control

    def initialize(self, config):
        original_init(self, config)
        self.metadata['diagnostic_actuator_adaptation'] = {
            'gain': 1.25, 'method': 'multiply_postprocessed_throttle_then_clip',
            'brake_steering_unchanged': True, 'speed_cutoff_preserved': True,
            'initialization_unchanged': True, 'diagnostic_only': True,
        }

    def compute(self):
        control = original_compute(self)
        native = {name: float(getattr(control, name)) for name in ('throttle', 'steer', 'brake')}
        control.throttle = min(1.0, 1.25 * native['throttle'])
        assert float(control.brake) == native['brake']
        assert float(control.steer) == native['steer']
        if native['throttle'] == 0:
            assert float(control.throttle) == 0
        self.last_record.update(
            native_wor_throttle=native['throttle'], native_wor_steer=native['steer'],
            native_wor_brake=native['brake'], throttle=float(control.throttle),
            diagnostic_only=True, actuator_gain=1.25)
        return control

    WORController.__init__ = initialize
    WORController.compute_control = compute
    script = os.path.join(PROJECT, 'src/macad_gym/agents/inference/LaneChange_ppo_inference_structural.py')
    sys.argv[0] = script
    os.chdir(PROJECT)
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    sys.exit(main())
