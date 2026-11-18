"""Opt-in, CSV-committed WOR evaluation recovery (standard library only)."""
import csv
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter


METRIC_COLUMNS = {
    'ego_rc': 'RC', 'ego_ol': 'OL', 'ego_ttd': 'TTD', 'ego_ttfc': 'TTFC',
    'ego_collision_flag': 'R_collision', 'global_ego_cols': 'Ego_Cols',
    'global_total_cols': 'Total_Cols', 'npc_lde': 'NPC_LDE', 'bg_lde': 'BG_LDE',
    'bg_lde_available': 'BG_LDE_Available', 'cosim_available': 'CoSim_Available',
    'bg_sample_count': 'BG_Sample_Count', 'rss_label': 'RSS_Label',
    'rss_reference_collision': 'RSS_Reference_Collision',
    'rss_reference_road_violation': 'RSS_Reference_Road_Violation',
    'rss_replay_succeeded': 'RSS_Replay_Succeeded',
    'bg_candidate_frames': 'BG_Candidate_Frames', 'bg_valid_frames': 'BG_Valid_Frames',
    'bg_pending_frames': 'BG_Pending_Frames', 'bg_metric_coverage': 'BG_Metric_Coverage',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.resume-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_ledger(path):
    if not os.path.isfile(path):
        return [], [], False, []
    with open(path, newline='') as stream:
        lines = stream.readlines()
    if not lines:
        raise ValueError('Empty existing evaluation CSV; refusing overwrite')
    header = next(csv.reader([lines[0]]))
    rows, kept, final = [], [lines[0]], False
    for index, line in enumerate(lines[1:], 1):
        cells = next(csv.reader([line]))
        # A process/power failure may leave only the very last row torn.
        if len(cells) != len(header) or not line.endswith('\n'):
            if index != len(lines) - 1:
                raise ValueError('Malformed non-trailing CSV row {}'.format(index))
            break
        row = dict(zip(header, cells))
        if row['record_type'] == 'FINAL':
            if index != len(lines) - 1 or int(row['episode']) != len(rows):
                raise ValueError('Invalid FINAL evaluation row')
            final = True
        elif row['record_type'] == 'EPISODE' and int(row['episode']) == len(rows) + 1:
            rows.append(row)
            kept.append(line)
        else:
            raise ValueError('Non-contiguous or duplicate episode in CSV')
    return header, rows, final, kept


def jsonl_prefix(path, completed, controls=False):
    if not os.path.isfile(path):
        if completed:
            raise ValueError('Missing committed audit file: ' + path)
        return [], Counter()
    kept, counts = [], Counter()
    with open(path) as stream:
        lines = stream.readlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError('Corrupt non-trailing audit line: ' + path)
            break
        episode = int(row['episode'])
        if episode > completed:
            continue
        if not 1 <= episode <= completed:
            raise ValueError('Invalid audit episode')
        if controls:
            if int(row['step']) != counts[episode] + 1 or row.get('phase') != 'applied':
                raise ValueError('Non-contiguous committed WOR controls')
        elif counts[episode]:
            raise ValueError('Duplicate committed diagnostic')
        counts[episode] += 1
        kept.append(line)
    if set(counts) != set(range(1, completed + 1)):
        raise ValueError('Missing controls/diagnostics for completed episodes')
    return kept, counts


def restored_metrics(rows):
    values = {key: [] for key in METRIC_COLUMNS}
    for row in rows:
        for key, column in METRIC_COLUMNS.items():
            value = row[column]
            if key.startswith('rss_'):
                pass
            elif value.lower() in ('true', 'false'):
                value = int(value.lower() == 'true')
            else:
                value = float(value) if value else float('nan')
            values[key].append(value)
    return values


def signature(args):
    # Freeze evaluated model identities and experimental settings, not restart limits.
    names = ('env', 'seed', 'max_steps', 'structural_ablation', 'ego_ads', 'num_workers',
             'envs_per_worker', 'num_gpus', 'multimodal_encoder_device', 'wor_device',
             'wor_sensor_timeout', 'wor_sensor_workers', 'wor_cruise_speed_cap_mps',
             'wor_turn_speed_cap_mps')
    result = {name: getattr(args, name) for name in names}
    result['npc_count'] = getattr(args, 'npc_count', 3)
    for name in ('checkpoint_path', 'multimodal_encoder_checkpoint', 'wor_weights', 'wor_config'):
        path = os.path.abspath(os.path.expanduser(getattr(args, name)))
        result[name] = {'path': path, 'sha256': digest(path)}
    result['wor_gain'] = getattr(args, '_wor_actuator_gain', 1.25)
    return result


def inspect(args, output_dir):
    path = os.path.join(output_dir, 'inference_LaneChange_{}_Results.csv'.format(args.structural_ablation))
    header, rows, final, kept = read_ledger(path)
    if len(rows) > args.episodes:
        raise ValueError('Requested total is less than completed episodes')
    manifest_path = os.path.join(output_dir, 'resume_manifest.json')
    expected = signature(args)
    if os.path.isfile(manifest_path):
        with open(manifest_path) as stream:
            saved = json.load(stream)['signature']
            # All pre-option calibrated lane evaluations necessarily used 3NPC.
            # Normalize in memory; never rewrite an existing experiment manifest.
            saved.setdefault('npc_count', 3)
            if saved != expected:
                raise ValueError('Resume configuration/weights differ from the saved experiment')
    elif rows and expected['npc_count'] != 3:
        raise ValueError('Legacy lane results have no NPC-count manifest; refusing to adopt as 2NPC')
    if rows:
        with open(os.path.join(output_dir, 'wor_model_provenance.json')) as stream:
            provenance = json.load(stream)
        if (provenance.get('weights_sha256') != expected['wor_weights']['sha256'] or
                provenance.get('config_sha256') != expected['wor_config']['sha256'] or
                provenance.get('device') != args.wor_device or
                provenance.get('diagnostic_actuator_adaptation', {}).get('gain', 1.0) != expected['wor_gain']):
            raise ValueError('Existing WOR provenance does not match this calibrated run')
        caps = provenance.get('control_adaptation', {}).get('speed_caps', {})
        expected_cruise = args.wor_cruise_speed_cap_mps if args.wor_cruise_speed_cap_mps is not None else 20.0 / 3.6
        if caps.get('cruise_mps') != expected_cruise:
            raise ValueError('WOR cruise cutoff differs from the original evaluation')
    controls, counts = jsonl_prefix(os.path.join(output_dir, 'wor_controls.jsonl'), len(rows), True)
    diagnostics, _ = jsonl_prefix(os.path.join(output_dir, 'cosim_diagnostics.jsonl'), len(rows))
    restored_metrics(rows)  # Reject invalid metric cells before launching CARLA.
    return dict(path=path, header=header, rows=rows, final=final, kept=kept,
                controls=controls, diagnostics=diagnostics,
                episode_counts=[counts[i] for i in range(1, len(rows) + 1)], signature=expected)


def replace_lines(path, lines):
    fd, temporary = tempfile.mkstemp(prefix='.resume-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'w', newline='') as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(args, output_dir):
    state = inspect(args, output_dir)
    if state['final']:
        if len(state['rows']) != args.episodes:
            raise ValueError('Completed experiment cannot be extended in-place; use a new output directory')
        print('Already completed {}/{} episodes; nothing to run.'.format(len(state['rows']), args.episodes))
        raise SystemExit(0)
    if any(os.path.isfile(os.path.join(output_dir, name)) for name in (
            os.path.basename(state['path']), 'wor_controls.jsonl', 'wor_status.json')):
        backup = tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S-'),
                                  dir=_backup_dir(output_dir))
        for name in ('wor_status.json', 'wor_model_provenance.json', 'wor_controls.jsonl',
                     'cosim_diagnostics.jsonl', os.path.basename(state['path'])):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                shutil.copy2(path, os.path.join(backup, name))
        if state['header']:
            replace_lines(state['path'], state['kept'])
        for name, lines in (('wor_controls.jsonl', state['controls']),
                            ('cosim_diagnostics.jsonl', state['diagnostics'])):
            if os.path.exists(os.path.join(output_dir, name)):
                replace_lines(os.path.join(output_dir, name), lines)
        print('Preserved {} committed episodes; backup: {}'.format(len(state['rows']), backup))
    manifest = os.path.join(output_dir, 'resume_manifest.json')
    if not os.path.exists(manifest):
        atomic_json(manifest, {'signature': state['signature'],
                              'legacy_adopted_episodes': len(state['rows']),
                              'note': 'Legacy PPO identity/settings were not logged at start; supplied '
                                      'original command is adopted. Restarted stochastic TF inference '
                                      'is not bitwise identical to an uninterrupted run.'})
    args._wor_resume_counts = state['episode_counts']
    args._wor_resume_validated = True
    return state


def _backup_dir(output_dir):
    path = os.path.join(output_dir, 'resume_backups')
    os.makedirs(path, exist_ok=True)
    return path


def validate_provenance(old, new):
    for key in ('backend', 'fallback_allowed', 'device', 'root', 'weights_sha256',
                'config_sha256', 'official_source_sha256', 'raw_wide_shape',
                'raw_narrow_shape', 'initial_speed_protocol', 'control_adaptation',
                'diagnostic_actuator_adaptation'):
        if old.get(key) != new.get(key):
            raise ValueError('WOR resume provenance differs: ' + key)
