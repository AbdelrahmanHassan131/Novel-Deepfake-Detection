"""Generate an UNEXECUTED notebook and its single-source command plan; stdlib only."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def make_plan():
    common = ['--no-pretrained', '--dataroot', '{data}', '--manifest', '{manifest}',
              '--manifest_split', 'train', '--val_manifest', '{manifest}', '--val_manifest_split', 'dev',
              '--run_id', '{run_id}', '--batch_size', '8', '--grad_accum_steps', '1',
              '--num_workers', '0', '--epochs_decay', '0', '--optim', 'adam', '--lr', '0.0001',
              '--monitor_metric', 'auc', '--seed', '42', '--wavelet_type', 'haar',
              '--wavelet_level', '3', '--wavelet_log_mode', 'signed_log1p']
    def train(arch, stage, extra=()):
        return ['{python}', '{repo}/train.py', '--arch', arch, *common,
                '--name', 'smoke_' + stage, '--checkpoints_dir', '{output}/' + stage, *extra]
    fusion = train('MHA_128', 'fusion', ['--fusion_type', 'token_attention', '--freeze_base_models',
                   '--rgb_model_path', '{rgb_best}', '--wavelet_model_path', '{wavelet_best}'])
    commands = {
        'bound_manifest': ['{python}', '{repo}/tools/prepare_smoke_manifest.py', '--manifest', '{input_manifest}',
                           '--root', '{data}', '--output', '{manifest}'],
        'audit': ['{python}', '{repo}/prepare_dataset.py', 'audit', '--manifest', '{manifest}',
                  '--root', '{data}', '--output', '{output}/smoke_audit.json', '--hashes'],
        'rgb': train('Wang2020_128', 'rgb') + ['--epochs', '1'],
        'wavelet': train('WolterWavelet2021_128', 'wavelet') + ['--epochs', '1'],
        'fusion': fusion + ['--epochs', '1'],
        'evaluate': ['{python}', '{repo}/evaluate.py', '--checkpoint', '{fusion_best}', '--arch', 'MHA_128',
                     '--dataroot', '{data}', '--manifest', '{manifest}', '--split', 'dev',
                     '--rgb_model_path', '{rgb_best}', '--wavelet_model_path', '{wavelet_best}',
                     '--output_dir', '{output}/evaluation', '--batch_size', '8', '--bootstrap', '20',
                     '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling'],
        'calibrate': ['{python}', '{repo}/analyze_predictions.py', 'calibrate', '--predictions', '{predictions}',
                      '--output', '{threshold}'],
        'resume': fusion + ['--epochs', '2', '--continue_train', '--resume_checkpoint', '{fusion_last}'],
    }
    return {
        'plan_type': 'declarative_smoke_test_plan', 'consumer': 'colab_smoke_pipeline.ipynb',
        'status': 'RUNTIME NOT RUN', 'hardware_target': 'single GPU',
        'dataset': {'input': 'separate preassigned smoke CSV; never full pool', 'max_total': 300,
                    'split_limits': {'train': 100, 'dev': 100, 'internal_test': 100},
                    'max_train_per_class': 50, 'require_both_classes': ['train', 'dev']},
        'training': {'batch_size': 8, 'grad_accum_steps': 1, 'num_workers': 0, 'use_amp': False,
                     'amp_policy': 'omit --use_amp; statically verify TRAINING_DEFAULTS.use_amp is false',
                     'pretrained': False, 'epochs': 1, 'resume_total_epochs': 2,
                     'optimizer': 'adam', 'learning_rate': 0.0001, 'monitor_metric': 'auc',
                     'wavelet_type': 'haar', 'wavelet_level': 3, 'wavelet_log_mode': 'signed_log1p'},
        'command_templates': commands,
    }


def cell(source, kind='code'):
    result = {'cell_type': kind, 'metadata': {}, 'source': dedent(source).strip().splitlines(True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build_smoke_notebook():
    cells = [cell('''
        # Single-GPU Colab smoke workflow — UNEXECUTED

        Checks the data audit, fresh experts, fusion, development prediction export/calibration,
        and resume progression. This is a wiring check, not a generalization benchmark.
        Prepare a separate CSV with verified metadata: at most 50 real + 50 fake train rows,
        100 dev rows, and optionally 100 internal_test rows. Both train and dev need both classes.
        Preserve connected identities/videos/originals within one split. Never use the full pool here.
        See COLAB_RUN_GUIDE.md. No historical checkpoint is required.
        Run cells in order. Re-running setup starts a fresh isolated run; rerunning a training cell
        in the same run intentionally fails on collisions. The last cell resumes that run explicitly.
        /content output is ephemeral: set OUTPUT_BASE to mounted Drive to retain artifacts.
    ''', 'markdown'), cell('''
        import csv
        import hashlib
        import json
        import os
        import subprocess
        import sys
        from datetime import datetime, timezone
        from pathlib import Path
        from uuid import uuid4

        REPO_ROOT = Path('/content/Novel-Deepfake-Detection').resolve()
        DATA_ROOT = Path('/content/data').resolve()
        SMOKE_INPUT_MANIFEST = DATA_ROOT / 'smoke_input.csv'
        OUTPUT_BASE = Path('/content/experiments/smoke').resolve()
        # To persist outputs, mount Drive first and choose a directory under /content/drive/MyDrive.
        RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:10]
        SMOKE_OUTPUT = OUTPUT_BASE / RUN_ID
        if not (REPO_ROOT / 'train.py').is_file():
            raise FileNotFoundError(f'Set REPO_ROOT to your uploaded repository: {REPO_ROOT}')
        if not SMOKE_INPUT_MANIFEST.is_file():
            raise FileNotFoundError(f'Supply a bounded, verified-metadata smoke CSV: {SMOKE_INPUT_MANIFEST}')
        SMOKE_OUTPUT.mkdir(parents=True, exist_ok=False)
        os.chdir(REPO_ROOT)
        plan = json.loads((REPO_ROOT / 'config/experiments/smoke_test_run.json').read_text())
        def checkpoint(stage, kind):
            return SMOKE_OUTPUT / stage / f'smoke_{stage}_{RUN_ID}' / 'checkpoints' / f'{kind}.pth'
        values = {key: str(value) for key, value in {
            'python': sys.executable, 'repo': REPO_ROOT, 'data': DATA_ROOT,
            'input_manifest': SMOKE_INPUT_MANIFEST, 'manifest': SMOKE_OUTPUT / 'smoke_manifest.csv',
            'output': SMOKE_OUTPUT, 'run_id': RUN_ID, 'rgb_best': checkpoint('rgb', 'best'),
            'wavelet_best': checkpoint('wavelet', 'best'), 'fusion_best': checkpoint('fusion', 'best'),
            'fusion_last': checkpoint('fusion', 'last'),
            'predictions': SMOKE_OUTPUT / 'evaluation/MHA_128/predictions.csv',
            'threshold': SMOKE_OUTPUT / 'evaluation/dev_threshold.json',
        }.items()}
        commands = {name: [part.format_map(values) for part in argv]
                    for name, argv in plan['command_templates'].items()}
        (SMOKE_OUTPUT / 'resolved_commands.json').write_text(json.dumps(commands, indent=2))
        def run(name, capture=False):
            print(f'Running {name}; outputs: {SMOKE_OUTPUT}')
            if capture:
                log_path = SMOKE_OUTPUT / f'{name}.log'
                with log_path.open('w', encoding='utf-8') as log:
                    result = subprocess.run(commands[name], stdout=log, stderr=subprocess.STDOUT)
                text = log_path.read_text(encoding='utf-8')
                print(text[-8000:])
                result.check_returncode()
                return text
            subprocess.run(commands[name], check=True)
        print(f'Run: {RUN_ID}\\nOutput: {SMOKE_OUTPUT}')
    '''), cell('''
        # Standard-library checks and CSV bounds before dependencies, image I/O, or model work.
        subprocess.run([sys.executable, 'tools/check_smoke_static.py'], check=True)
        run('bound_manifest')
    '''), cell('''
        # Install only when YOU execute this notebook on Colab.
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'timm', 'pytorch-wavelets', 'scikit-learn', 'matplotlib', 'seaborn'], check=True)
        run('audit')
        from data.manifest import verify_manifest_gate
        gate = verify_manifest_gate(values['manifest'], enforce_class_coverage=True, require_hashes=True)
        print(f"Audit passed for {gate['samples']} bounded smoke rows.")
    '''), cell('''
        run('rgb')
        if not Path(values['rgb_best']).is_file():
            raise RuntimeError('RGB best checkpoint was not written.')
    '''), cell('''
        run('wavelet')
        if not Path(values['wavelet_best']).is_file():
            raise RuntimeError('Wavelet best checkpoint was not written.')
    '''), cell('''
        run('fusion')
        for key in ('fusion_best', 'fusion_last'):
            if not Path(values[key]).is_file():
                raise RuntimeError(f'Missing fresh checkpoint: {key}')
    '''), cell('''
        run('evaluate')
        with Path(values['predictions']).open(newline='') as stream:
            predictions = list(csv.DictReader(stream))
        with Path(values['manifest']).open(newline='') as stream:
            dev = [row for row in csv.DictReader(stream) if row['split'] == 'dev']
        if len(predictions) != len(dev) or {r['sample_id'] for r in predictions} != {r['sample_id'] for r in dev}:
            raise RuntimeError('Prediction export does not cover exactly the smoke dev split.')
        run('calibrate')
        calibration = json.loads(Path(values['threshold']).read_text())
        if not 0 <= calibration['threshold'] <= 1:
            raise RuntimeError('Invalid calibrated threshold.')
        print('Development predictions and threshold exported. Smoke metrics are not benchmark results.')
    '''), cell('''
        # Read only checkpoints produced by the preceding cells in this fresh run.
        import torch
        def snapshot():
            saved = torch.load(values['fusion_last'], map_location='cpu', weights_only=False)
            if not saved.get('optimizer_state_dict', {}).get('state'):
                raise RuntimeError('Missing optimizer state in fresh fusion checkpoint.')
            if not saved.get('rng_state'):
                raise RuntimeError('Missing RNG state in fresh fusion checkpoint.')
            digest = hashlib.sha256(Path(values['manifest']).read_bytes()).hexdigest()
            for name in ('manifest', 'val_manifest'):
                if saved['protocol']['manifests'][name]['sha256'] != digest:
                    raise RuntimeError(f'Checkpoint {name} digest mismatch.')
            return {'epoch': saved['epoch'], 'global_step': saved['global_step']}
        before = snapshot()
        if before['epoch'] != 1 or before['global_step'] <= 0:
            raise RuntimeError(f'Expected a completed first epoch; got {before}')
        resume_log = run('resume', capture=True)
        after = snapshot()
        expected = f"[BaseTrainer] Resumed: epoch={before['epoch']}, global_step={before['global_step']},"
        if expected not in resume_log or after['epoch'] != 2 or after['global_step'] <= before['global_step']:
            raise RuntimeError(f'Resume progression check failed: {before} -> {after}')
        evidence = {'before': before, 'after': after, 'resume_log': 'resume.log',
                    'verified': 'saved state exists, manifest identity, logged resume counters, subsequent progression',
                    'not_verified': 'bitwise optimizer/RNG continuation or generalization accuracy'}
        (SMOKE_OUTPUT / 'resume_evidence.json').write_text(json.dumps(evidence, indent=2))
        print('Resume counters advanced from epoch 1 to 2. Numerical equivalence remains untested.')
        print(f'Preserve all artifacts in {SMOKE_OUTPUT} before disconnecting Colab.')
    ''')]
    notebook = {'cells': cells, 'metadata': {'accelerator': 'GPU',
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 0}
    (ROOT / 'colab_smoke_pipeline.ipynb').write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'config/experiments/smoke_test_run.json').write_text(json.dumps(make_plan(), indent=2) + '\n', encoding='utf-8')
    print('Generated notebook and command plan; no notebook cells executed.')


if __name__ == '__main__':
    build_smoke_notebook()
