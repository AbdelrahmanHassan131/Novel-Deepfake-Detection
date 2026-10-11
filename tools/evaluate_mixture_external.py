"""Evaluate the two downloaded pilot checkpoints once each at frozen thresholds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_calibration(arm, entry, checkpoint):
    actual = digest(checkpoint)
    calibration = entry['dev_calibration_artifact']
    if actual != entry['checkpoint_sha256'] or actual != calibration['checkpoint_sha256']:
        raise ValueError(f'{arm}: checkpoint does not match development report')
    if calibration['source_splits'] != ['dev']:
        raise ValueError(f'{arm}: expected calibration on shared dev only')
    threshold = float(calibration['threshold'])
    if not 0 <= threshold <= 1 or threshold != float(entry['dev_selected']['threshold']):
        raise ValueError(f'{arm}: invalid or inconsistent development threshold')
    return dict(calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--arm_a', type=Path, required=True)
    parser.add_argument('--arm_b', type=Path, required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda:0', 'cpu'])
    parser.add_argument('--check_only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    for label in ('real', 'fake'):
        if not (args.data_root / label).is_dir():
            raise FileNotFoundError(args.data_root / label)
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    calibrations = {}
    for arm in ('arm_a', 'arm_b'):
        calibrations[arm] = verify_calibration(arm, report[arm], getattr(args, arm))
        print(f'{arm}: checkpoint verified; dev threshold = {calibrations[arm]["threshold"]}', flush=True)
    if args.check_only:
        print('Preflight passed. No image inference performed.')
        return

    from evaluation.generalization import binary_metrics, read_predictions

    tag = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    output = REPO / 'output' / 'external_evaluation' / f'mixture_pilot_{tag}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source_development_report.json').write_bytes(args.report.read_bytes())
    comparison = {'dataset': str(args.data_root.resolve()), 'split': 'external_dev',
                  'development_report_sha256': digest(args.report),
                  'note': 'Thresholds frozen on shared development data; no external threshold fitting.',
                  'arms': {}}
    lines = ['# External RGB pilot comparison', '', comparison['note'], '',
             '| Arm | Threshold rule | Threshold | Accuracy | Balanced accuracy | AUC | Real recall | Fake recall | Macro F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    reference_cohort = None
    print(f'Results: {output}', flush=True)
    for arm in ('arm_a', 'arm_b'):
        arm_dir = output / arm
        arm_dir.mkdir()
        threshold_file = arm_dir / 'threshold.json'
        threshold_file.write_text(json.dumps(calibrations[arm], indent=2), encoding='utf-8')
        command = [sys.executable, '-u', str(REPO / 'evaluate.py'),
                   '--checkpoint', str(getattr(args, arm).resolve()), '--arch', 'Wang2020_128',
                   '--val_root', str(args.data_root.resolve()), '--split', 'external_dev',
                   '--threshold_file', str(threshold_file), '--output_dir', str(arm_dir),
                   '--batch_size', str(args.batch_size), '--num_workers', '0',
                   '--eval_precision', 'fp32', '--bootstrap', '0', '--no_plots',
                   '--no_tsne', '--no_gradcam', '--no_profiling']
        if args.device != 'auto':
            command += ['--device', args.device]
        print(f'\nEvaluating {arm} (one inference pass)...', flush=True)
        subprocess.run(command, cwd=REPO, check=True)
        rows = read_predictions(arm_dir / 'Wang2020_128' / 'predictions.csv')
        if {r['checkpoint_sha256'] for r in rows} != {calibrations[arm]['checkpoint_sha256']}:
            raise ValueError(f'{arm}: exported predictions have incorrect checkpoint linkage')
        cohort = {(r['sample_id'], r['path'], r['label']) for r in rows}
        if reference_cohort is not None and cohort != reference_cohort:
            raise ValueError('Arms evaluated different images or labels; comparison aborted')
        reference_cohort = cohort
        labels = [r['label'] for r in rows]
        if set(labels) != {0, 1}:
            raise ValueError('External dataset must contain both real and fake images')
        scores = [r['probability'] for r in rows]
        result = {'checkpoint_sha256': calibrations[arm]['checkpoint_sha256']}
        for rule, threshold in [('fixed_0_5', 0.5), ('frozen_dev', calibrations[arm]['threshold'])]:
            metrics = binary_metrics(labels, scores, threshold)
            result[rule] = metrics
            lines.append(f'| {arm} | {rule} | {threshold:.8f} | {metrics["accuracy"]:.2%} | '
                         f'{metrics["balanced_accuracy"]:.2%} | {metrics["roc_auc"]:.4f} | '
                         f'{metrics["real_recall"]:.2%} | {metrics["fake_recall"]:.2%} | '
                         f'{metrics["macro_f1"]:.4f} |')
        (arm_dir / 'two_threshold_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        comparison['arms'][arm] = result
    (output / 'external_comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'external_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(lines), flush=True)
    print(f'\nSend back: {output / "external_comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
