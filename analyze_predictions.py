"""Analyze saved predictions without retraining or tuning on final test sets."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluation.generalization import (read_predictions, summarize, calibrate,
                                       group_intervals, export_predictions, summarize_seeds,
                                       ALLOWED_CALIBRATION_SPLITS, FORBIDDEN_CALIBRATION_SPLITS)
from data.manifest import known


def load_threshold(threshold_arg, threshold_file_arg, default=0.5, expected_checkpoint_hash=None):
    """Load threshold from JSON artifact or float argument, validating schema, splits, and checkpoint hash."""
    if threshold_file_arg:
        data = json.loads(Path(threshold_file_arg).read_text(encoding='utf-8'))
        if 'threshold' not in data:
            raise ValueError(f"Threshold artifact {threshold_file_arg} missing 'threshold' field")
        val = float(data['threshold'])
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Invalid threshold {val} in {threshold_file_arg}; must be in [0, 1]")

        # Validate calibration split provenance
        source_splits = set(data.get('source_splits', []))
        forbidden = source_splits.intersection(FORBIDDEN_CALIBRATION_SPLITS)
        if not source_splits or forbidden or (source_splits - ALLOWED_CALIBRATION_SPLITS):
            raise ValueError(
                f"Threshold artifact {threshold_file_arg} was calibrated on forbidden evaluation split(s) / invalid split(s). "
                f"Allowed: {sorted(ALLOWED_CALIBRATION_SPLITS)}, forbidden: {sorted(FORBIDDEN_CALIBRATION_SPLITS)}, got: {sorted(source_splits)}"
            )

        # Validate checkpoint hash binding
        if expected_checkpoint_hash is not None:
            artifact_ckpt = data.get('checkpoint_sha256')
            if not known(artifact_ckpt) or artifact_ckpt != expected_checkpoint_hash:
                raise ValueError(
                    f"Threshold artifact {threshold_file_arg} was calibrated for checkpoint '{artifact_ckpt}', "
                    f"which does not match evaluation checkpoint '{expected_checkpoint_hash}'."
                )

        return val
    if threshold_arg is not None:
        val = float(threshold_arg)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Invalid threshold {val}; must be in [0, 1]")
        return val
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['summarize', 'calibrate', 'compare', 'ensemble', 'seeds'])
    parser.add_argument('--predictions', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.5, help='Fixed or development-selected threshold for first model')
    parser.add_argument('--threshold_file', type=str, default=None,
                        help='Path to JSON threshold artifact for first model (e.g. from calibrate)')
    parser.add_argument('--threshold_other', type=float, default=None,
                        help='Development-selected threshold for second model in paired comparison')
    parser.add_argument('--threshold_file_other', type=str, default=None,
                        help='Path to JSON threshold artifact for second model in paired comparison')
    parser.add_argument('--thresholds', type=float, nargs='+', default=None,
                        help='Independently development-selected thresholds for each run in multi-seed analysis')
    parser.add_argument('--threshold_files', type=str, nargs='+', default=None,
                        help='Paths to JSON threshold artifacts for each run in multi-seed analysis')
    parser.add_argument('--run_ids', type=str, nargs='+', default=None,
                        help='Optional explicit independent training run IDs for multi-seed verification')
    parser.add_argument('--eval_precision', type=str, default=None,
                        help='Optional evaluation precision (e.g. fp32, amp, fp16) recorded in threshold artifact')
    parser.add_argument('--selection_predictions', type=str, default=None,
                        help='Path to selection development predictions to verify calibration cohort independence')
    parser.add_argument('--selection_manifest', type=str, default=None,
                        help='Path to selection development manifest to verify calibration cohort independence')
    parser.add_argument('--train_manifest', type=str, default=None,
                        help='Path to training manifest to verify calibration cohort independence')
    parser.add_argument('--allow_external_dev', action='store_true', default=False,
                        help='Permit threshold calibration on external development split (external_dev)')
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.action == 'calibrate':
        if len(args.predictions) != 1:
            raise ValueError('Calibrate one checkpoint at a time')
        calibrate(args.predictions[0], args.output, eval_precision=args.eval_precision,
                  allow_external_dev=args.allow_external_dev,
                  selection_predictions_path=args.selection_predictions,
                  selection_manifest=args.selection_manifest,
                  train_manifest=args.train_manifest)
        print(f"Calibrated threshold artifact saved to {args.output}")
        return

    if args.action == 'seeds':
        all_seed_rows = [read_predictions(p) for p in args.predictions]
        seed_hashes = [r[0].get('checkpoint_sha256') if r else None for r in all_seed_rows]
        thresholds = None
        if args.threshold_files:
            if len(args.threshold_files) != len(args.predictions):
                raise ValueError(f"Expected {len(args.predictions)} threshold_files, got {len(args.threshold_files)}")
            thresholds = [load_threshold(None, tf, expected_checkpoint_hash=h) for tf, h in zip(args.threshold_files, seed_hashes)]
        elif args.thresholds:
            thresholds = [load_threshold(t, None) for t in args.thresholds]
        t_base = load_threshold(args.threshold, args.threshold_file, default=0.5, expected_checkpoint_hash=seed_hashes[0] if seed_hashes else None)
        report = summarize_seeds(args.predictions, threshold=t_base, thresholds=thresholds,
                                 repeats=args.bootstrap, seed=args.seed, run_ids=args.run_ids)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f"Multi-seed summary report saved to {args.output}")
        return

    sets = [read_predictions(p) for p in args.predictions]
    rows = sets[0]
    ids = [r['sample_id'] for r in rows]
    aligned = []
    for data in sets:
        by_id = {r['sample_id']: r for r in data}
        if set(ids) != set(by_id):
            raise ValueError('All methods must be evaluated on exactly the same sample IDs')
        for row in rows:
            other = by_id[row['sample_id']]
            for key in ('label', 'group_id', 'split', 'path'):
                if row.get(key) != other.get(key):
                    raise ValueError(f'Prediction metadata mismatch for {row["sample_id"]}: {key}')
        aligned.append([by_id[i]['probability'] for i in ids])

    if args.action == 'ensemble':
        import numpy as np
        probabilities = np.mean(aligned, axis=0)
        fingerprint = hashlib.sha256('|'.join(str(d[0].get('checkpoint_sha256')) for d in sets).encode()).hexdigest()
        export_predictions(args.output, rows, probabilities, fingerprint)
        return

    m1_hash = sets[0][0].get('checkpoint_sha256') if sets and sets[0] else None
    t1 = load_threshold(args.threshold, args.threshold_file, default=0.5, expected_checkpoint_hash=m1_hash)
    if args.action == 'compare':
        if len(aligned) != 2:
            raise ValueError('Paired comparison takes exactly two prediction files')
        m2_hash = sets[1][0].get('checkpoint_sha256') if len(sets) > 1 and sets[1] else None
        t2 = load_threshold(args.threshold_other, args.threshold_file_other, default=t1, expected_checkpoint_hash=m2_hash)
        m1_summary = summarize(rows, aligned[0], threshold=t1, repeats=args.bootstrap, seed=args.seed)
        m2_summary = summarize(rows, aligned[1], threshold=t2, repeats=args.bootstrap, seed=args.seed)
        diff_report = group_intervals(
            [r['label'] for r in rows],
            aligned[0],
            [r.get('group_id', '') for r in rows],
            threshold=t1,
            repeats=args.bootstrap,
            seed=args.seed,
            other=aligned[1],
            other_threshold=t2,
        )
        report = {
            'models': [m1_summary, m2_summary],
            'model_thresholds': [t1, t2],
            'first_minus_second': diff_report,
        }
    else:
        report = {'models': [summarize(rows, p, t1, args.bootstrap, args.seed) for p in aligned]}

    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Analysis saved to {args.output}")


if __name__ == '__main__':
    main()
