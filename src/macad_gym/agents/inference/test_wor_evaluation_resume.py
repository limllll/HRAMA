"""Recovery tests use synthetic files/processes only, never CARLA or real results."""
import argparse
import ast
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import evaluation_resume as recovery
import wor_eval_supervisor as supervisor
from lane_evaluation_layout import apply_lane_npc_count


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='wor-resume-test-')
        self.root = self.tmp.name
        self.out = os.path.join(self.root, 'seed2026')
        os.mkdir(self.out)
        fields = dict(env='Town04C3-v0', seed=2026, max_steps=80,
                      structural_ablation='full', ego_ads='wor', num_workers=0,
                      envs_per_worker=1, num_gpus=0, multimodal_encoder_device='cpu',
                      wor_device='cuda:0', wor_sensor_timeout=10, wor_sensor_workers=4,
                      wor_cruise_speed_cap_mps=15., wor_turn_speed_cap_mps=None, episodes=5)
        for name in ('checkpoint_path', 'multimodal_encoder_checkpoint', 'wor_weights', 'wor_config'):
            path = os.path.join(self.root, name)
            with open(path, 'wb') as stream:
                stream.write(name.encode())
            fields[name] = path
        self.args = argparse.Namespace(**fields)
        with open(os.path.join(os.path.dirname(__file__), 'LaneChange_ppo_inference_structural.py')) as stream:
            tree = ast.parse(stream.read())
        self.header = next(ast.literal_eval(node.value) for node in tree.body
                           if isinstance(node, ast.Assign) and
                           any(isinstance(target, ast.Name) and target.id == 'csv_columns'
                               for target in node.targets))
        self.csv_path = os.path.join(self.out, 'inference_LaneChange_full_Results.csv')
        self.provenance = dict(weights_sha256=recovery.digest(self.args.wor_weights),
                               config_sha256=recovery.digest(self.args.wor_config),
                               device='cuda:0', diagnostic_actuator_adaptation={'gain': 1.25},
                               control_adaptation={'speed_caps': {'cruise_mps': 15.}})
        recovery.atomic_json(os.path.join(self.out, 'wor_model_provenance.json'), self.provenance)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, episodes=2, final=False, extra=False):
        with open(self.csv_path, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.header)
            writer.writeheader()
            for ep in range(1, episodes + 1):
                row = dict((name, '1') for name in self.header)
                row.update(record_type='EPISODE', episode=ep, RSS_Label='',
                           RSS_Reference_Collision='', RSS_Reference_Road_Violation='',
                           RSS_Replay_Succeeded='', BG_LDE_Available='True', CoSim_Available='True')
                writer.writerow(row)
            if final:
                writer.writerow(dict(record_type='FINAL', episode=episodes))
        with open(os.path.join(self.out, 'wor_controls.jsonl'), 'w') as stream:
            for ep in range(1, episodes + 1 + int(extra)):
                for step in range(1, 3):
                    stream.write(json.dumps(dict(episode=ep, step=step, phase='applied')) + '\n')
            if extra:
                stream.write('{torn')
        with open(os.path.join(self.out, 'cosim_diagnostics.jsonl'), 'w') as stream:
            for ep in range(1, episodes + 1):
                stream.write(json.dumps(dict(episode=ep)) + '\n')

    def test_read_only_preserves_bytes(self):
        self.fixture(extra=True)
        before = {name: recovery.digest(os.path.join(self.out, name)) for name in os.listdir(self.out)}
        state = recovery.inspect(self.args, self.out)
        self.assertEqual(len(state['rows']), 2)
        self.assertEqual(state['episode_counts'], [2, 2])
        self.assertEqual(before, {name: recovery.digest(os.path.join(self.out, name)) for name in os.listdir(self.out)})

    def test_prepare_keeps_committed_prefix_and_backs_up_tail(self):
        self.fixture(extra=True)
        csv_before = recovery.digest(self.csv_path)
        state = recovery.prepare(self.args, self.out)
        self.assertEqual(recovery.digest(self.csv_path), csv_before)
        with open(os.path.join(self.out, 'wor_controls.jsonl')) as stream:
            self.assertEqual(len(stream.readlines()), 4)
        self.assertEqual(len(state['rows']), 2)
        self.assertTrue(os.listdir(os.path.join(self.out, 'resume_backups')))

    def test_first_episode_failure_trimmed(self):
        self.fixture(episodes=0, extra=True)
        recovery.prepare(self.args, self.out)
        self.assertEqual(os.path.getsize(os.path.join(self.out, 'wor_controls.jsonl')), 0)

    def test_duplicates_rejected(self):
        self.fixture()
        with open(self.csv_path) as stream:
            line = stream.readlines()[-1]
        with open(self.csv_path, 'a') as stream:
            stream.write(line)
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_torn_last_csv_rolled_back(self):
        self.fixture(extra=True)
        with open(self.csv_path, 'a') as stream:
            stream.write('EPISODE,3,incomplete')
        recovery.prepare(self.args, self.out)
        self.assertEqual(len(recovery.read_ledger(self.csv_path)[1]), 2)

    def test_missing_committed_audit_rejected(self):
        self.fixture()
        with open(os.path.join(self.out, 'cosim_diagnostics.jsonl'), 'w') as stream:
            stream.write('{"episode":1}\n')
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_noncontiguous_control_rejected(self):
        self.fixture()
        path = os.path.join(self.out, 'wor_controls.jsonl')
        with open(path) as stream:
            lines = stream.readlines()
        with open(path, 'w') as stream:
            stream.writelines(lines[1:])
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_config_change_rejected(self):
        self.fixture()
        recovery.prepare(self.args, self.out)
        self.args.seed = 999
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_npc_count_resume_mismatch_rejected(self):
        self.fixture()
        recovery.prepare(self.args, self.out)
        self.args.npc_count = 2
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_legacy_three_npc_manifest_still_readable(self):
        self.fixture()
        signature = recovery.signature(self.args)
        del signature['npc_count']
        recovery.atomic_json(os.path.join(self.out, 'resume_manifest.json'), {'signature': signature})
        self.assertEqual(len(recovery.inspect(self.args, self.out)['rows']), 2)

    def test_two_npc_changes_only_car2_in_evaluation_copy(self):
        import copy
        config = {'actors': {actor: {'initial_speed': speed} for actor, speed in
                            [('ego', 14.5), ('car1', 14.5), ('car2', 12.5), ('car3', 12.5)]},
                  'scenarios': {'map': 'Town04', 'max_steps': 80, 'weather_distribution': [0],
                      'actors': {'ego': {'start': [-349, 33.4, .8], 'end': [-268, 30.7, .8]},
                                 'car1': {'start': [-360, 30.7, .8], 'end': [-280, 30.7, .8]},
                                 'car2': {'start': [-329, 30.7, .8], 'end': [-269, 30.7, .8]},
                                 'car3': {'start': [-331, 33.4, .8], 'end': [-251, 33.4, .8]}}}}
        original = copy.deepcopy(config)
        evaluation = copy.deepcopy(config)
        self.assertEqual(apply_lane_npc_count(evaluation, 2), ['car2'])
        self.assertEqual(set(evaluation['actors']), {'ego', 'car1', 'car3'})
        for actor in ('ego', 'car1', 'car3'):
            self.assertEqual(evaluation['actors'][actor], original['actors'][actor])
            self.assertEqual(evaluation['scenarios']['actors'][actor], original['scenarios']['actors'][actor])
        self.assertEqual(config, original)
        self.assertEqual(apply_lane_npc_count(config, 3), [])
        self.assertEqual(config, original)

    def test_supervisor_parses_two_npc(self):
        _, args, tokens = supervisor.arguments(self.root, ['--checkpoint-path', self.args.checkpoint_path,
                    '--multimodal-encoder-checkpoint', self.args.multimodal_encoder_checkpoint,
                    '--output-dir', self.root, '--ego-ads', 'wor', '--num-workers', '0', '--npc-count', '2'])
        self.assertEqual(args.npc_count, 2)
        self.assertIn('--npc-count', tokens)

    def test_weights_change_rejected(self):
        self.fixture()
        recovery.prepare(self.args, self.out)
        with open(self.args.checkpoint_path, 'ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_gain_change_rejected(self):
        self.fixture()
        self.provenance['diagnostic_actuator_adaptation']['gain'] = 1.4
        recovery.atomic_json(os.path.join(self.out, 'wor_model_provenance.json'), self.provenance)
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_native_no_gain_provenance_is_resumable(self):
        self.fixture()
        self.args._wor_actuator_gain = 1.0
        self.args.wor_cruise_speed_cap_mps = None
        del self.provenance['diagnostic_actuator_adaptation']
        self.provenance['control_adaptation']['speed_caps']['cruise_mps'] = 20.0 / 3.6
        recovery.atomic_json(os.path.join(self.out, 'wor_model_provenance.json'), self.provenance)
        state = recovery.prepare(self.args, self.out)
        self.assertEqual(state['signature']['wor_gain'], 1.0)
        self.assertEqual(len(state['rows']), 2)
        self.args._wor_actuator_gain = 1.25
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_native_with_explicit_15_cutoff(self):
        self.fixture()
        self.args._wor_actuator_gain = 1.0
        del self.provenance['diagnostic_actuator_adaptation']
        recovery.atomic_json(os.path.join(self.out, 'wor_model_provenance.json'), self.provenance)
        self.assertEqual(recovery.inspect(self.args, self.out)['signature']['wor_gain'], 1.0)
        self.args.wor_cruise_speed_cap_mps = None
        with self.assertRaises(ValueError):
            recovery.inspect(self.args, self.out)

    def test_native_supervisor_profile(self):
        _, args, _ = supervisor.arguments(self.root, ['--checkpoint-path', self.args.checkpoint_path,
                    '--multimodal-encoder-checkpoint', self.args.multimodal_encoder_checkpoint,
                    '--output-dir', self.root, '--ego-ads', 'wor', '--num-workers', '0'], actuator_gain=1.0)
        self.assertEqual(args._wor_actuator_gain, 1.0)
        self.assertIsNone(args.wor_cruise_speed_cap_mps)

    def test_final_no_overwrite(self):
        self.fixture(episodes=5, final=True)
        before = recovery.digest(self.csv_path)
        with self.assertRaises(SystemExit) as raised:
            recovery.prepare(self.args, self.out)
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(recovery.digest(self.csv_path), before)

    def test_cumulative_metrics(self):
        self.fixture()
        metrics = recovery.restored_metrics(recovery.inspect(self.args, self.out)['rows'])
        metrics['ego_collision_flag'].append(0)
        metrics['ego_rc'].append(0.25)
        self.assertAlmostEqual(sum(metrics['ego_collision_flag']) / 3, 2 / 3)
        self.assertAlmostEqual(sum(metrics['ego_rc']) / 3, .75)
        self.assertEqual(metrics['bg_lde_available'], [1, 1])

    def test_actual_lane_loop_appends_and_reports_all_episodes(self):
        # Execute the actual metric/episode loop with a fake one-step simulator,
        # verifying integration, not a second reimplementation of aggregation.
        self.fixture()
        state = recovery.prepare(self.args, self.out)
        self.args.episodes = 3
        with open(os.path.join(os.path.dirname(__file__), 'LaneChange_ppo_inference_structural.py')) as stream:
            source = stream.read()
        block = source[source.index('metrics = {'):source.index('save_json(os.path.join(output_dir, "evaluation_config.json")')]
        fake_env = mock.Mock()
        fake_env._sumo_bridge = object()
        fake_env._cosim_available = fake_env._episode_cosim_available = True
        fake_env._cosim_last_error = None
        fake_env._cosim_last_status = {'ok': True}
        fake_env._initial_distance_to_end = {'ego': 80}
        fake_env._end_distance_to_end = {'ego': 60}
        fake_env._time_to_goal = {}
        fake_env._time_to_firstcollision = {}
        fake_env._lateral_distance = {'car1': [1.]}
        fake_env._collisions = {actor: argparse.Namespace(collision_vehicles=0, collision_other=0)
                                for actor in ('ego', 'car1')}
        fake_env._lane_invasions = {'ego': argparse.Namespace(offlane=0)}
        fake_env.reset.return_value = {'ego': [], 'car1': []}
        fake_env.step.return_value = ({'ego': [], 'car1': []}, {'ego': 0, 'car1': 0},
                                     {'__all__': True}, {'ego': {'rc_normalized': .25,
                                         'bg_lde_available': True, 'cosim_available': True,
                                         'bg_sample_count': 1, 'bg_lde': 2.}})
        trainer = mock.Mock()
        trainer.get_policy.return_value.get_initial_state.return_value = []
        trainer.compute_action.return_value = ([2, 2], [], {})
        import numpy as np
        import copy
        import math
        import time
        namespace = dict(output_dir=self.out, STRUCTURAL_ABLATION='full', args=self.args,
                         resume_state=state, start_episode=2, eps=3, max_steps=80,
                         restored_metrics=recovery.restored_metrics, actor_ids=['ego', 'car1'],
                         env=fake_env, trainer=trainer, wor_audit=mock.Mock(), rss_evaluation=None,
                         os=os, csv=csv, json=json, copy=copy, math=math, time=time, np=np)
        exec(compile(block, '<actual-lane-loop>', 'exec'), namespace)
        _, rows, final, _ = recovery.read_ledger(self.csv_path)
        self.assertEqual([row['episode'] for row in rows], ['1', '2', '3'])
        self.assertTrue(final)
        with open(self.csv_path, newline='') as stream:
            result = list(csv.DictReader(stream))[-1]
        self.assertAlmostEqual(float(result['R_collision']), 2 / 3)
        self.assertAlmostEqual(float(result['RC']), .75)
        self.assertAlmostEqual(float(result['BG_LDE']), 4 / 3)

    def test_traceback_logged(self):
        log = os.path.join(self.root, 'attempt.log')
        code, stalled = supervisor.run_attempt(
            [sys.executable, '-u', '-c', 'raise RuntimeError("simulated failure")'],
            os.environ.copy(), self.out, log, 10)
        self.assertNotEqual(code, 0)
        self.assertFalse(stalled)
        with open(log) as stream:
            self.assertIn('RuntimeError: simulated failure', stream.read())

    def test_stalled_child_stopped(self):
        log = os.path.join(self.root, 'stall.log')
        _, stalled = supervisor.run_attempt([sys.executable, '-c', 'import time; time.sleep(20)'],
                                             os.environ.copy(), self.out, log, .05)
        self.assertTrue(stalled)

    def test_provenance_runtime_can_change_but_model_cannot(self):
        recovery.validate_provenance({'device': 'cuda:0', 'inference_runtime': {'pid': 1}},
                                     {'device': 'cuda:0', 'inference_runtime': {'pid': 2}})
        with self.assertRaises(ValueError):
            recovery.validate_provenance({'weights_sha256': 'a'}, {'weights_sha256': 'b'})

    def test_supervisor_restarts_then_completes(self):
        tokens = ['--checkpoint-path', self.args.checkpoint_path,
                  '--multimodal-encoder-checkpoint', self.args.multimodal_encoder_checkpoint,
                  '--output-dir', self.root, '--ego-ads', 'wor', '--num-workers', '0',
                  '--episodes', '5', '--retry-delay', '0', '--max-restarts', '1']
        first = dict(rows=[{}] * 2, final=False)
        second = dict(rows=[{}] * 5, final=True)
        cwd = os.getcwd()
        try:
            with mock.patch.object(supervisor, 'inspect', side_effect=[first, first, second]), \
                    mock.patch.object(supervisor, 'run_attempt', side_effect=[(9, False), (0, False)]) as run:
                self.assertEqual(supervisor.supervise(self.root, 'dummy.py', tokens), 0)
                self.assertEqual(run.call_count, 2)
        finally:
            os.chdir(cwd)

    def test_restart_limit_leaves_results(self):
        tokens = ['--checkpoint-path', self.args.checkpoint_path,
                  '--multimodal-encoder-checkpoint', self.args.multimodal_encoder_checkpoint,
                  '--output-dir', self.root, '--ego-ads', 'wor', '--num-workers', '0',
                  '--max-restarts', '0']
        cwd = os.getcwd()
        try:
            with mock.patch.object(supervisor, 'inspect', return_value=dict(rows=[{}] * 2, final=False)), \
                    mock.patch.object(supervisor, 'run_attempt', return_value=(9, False)) as run:
                self.assertEqual(supervisor.supervise(self.root, 'dummy.py', tokens), 1)
                self.assertEqual(run.call_count, 1)
        finally:
            os.chdir(cwd)

    def test_audit_append_restores_counts(self):
        self.fixture()
        recovery.prepare(self.args, self.out)
        audit_path = os.path.join(os.path.dirname(__file__), 'wor_evaluation.py')
        if not os.path.isfile(audit_path):
            audit_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ads', 'wor_evaluation.py')
        spec = importlib.util.spec_from_file_location('audit_test', audit_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        audit = module.WORAudit(self.args, self.out)
        self.assertEqual(audit.count, 4)
        self.assertEqual(audit.episode_counts, [2, 2])
        controller = mock.Mock(metadata=self.provenance, episode_steps=2)
        env = mock.Mock(_external_ego_controller=controller)
        audit.attach(env)
        audit.pending_record = dict(throttle=.1, steer=0, brake=0, episode=3, step=1)
        audit.after_step({'ego': {'control_source': 'wor', 'autocontrol': dict(throttle=.1, steer=0, brake=0)}})
        audit.close()
        audit.complete = True  # Prevent later atexit mutation of the deleted fixture.
        with open(os.path.join(self.out, 'wor_controls.jsonl')) as stream:
            self.assertEqual(len(stream.readlines()), 5)


if __name__ == '__main__':
    unittest.main()
