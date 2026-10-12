"""Analyze saved predictions without retraining or tuning on final test sets."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluation.generalization import (read_predictions, summarize, calibrate,
                                       group_intervals, export_predictions, summarize_seeds,
                                       ALLOWED_CALIBRATION_SPLITS, FORBIDDEN_CALIBRATION_SPLITS,
                                       load_validated_threshold, compare_evaluation_runs)
from data.manifest import known


def load_threshold(threshold_arg, threshold_file_arg, default=0.5, expected_checkpoint_hash=None,
                   return_score_space=False, score_space_arg=None, eval_precision=None):
    """Load threshold from JSON artifact or float argument, validating schema, splits, and checkpoint hash."""
    import numpy as np
    if threshold_file_arg:
        val, score_space = load_validated_threshold(
            threshold_file_arg,
            expected_checkpoint_hash=expected_checkpoint_hash,
            eval_precision=eval_precision
        )
        if return_score_space:
            return val, score_space
        return val

    if threshold_arg is not None:
        val = float(threshold_arg)
        score_space = score_space_arg or 'probability'
        if score_space not in ('logit', 'probability'):
            raise ValueError(f"Unknown score_space '{score_space}'; expected 'logit' or 'probability'.")
        if score_space == 'probability' and not (0.0 <= val <= 1.0):
            raise ValueError(f"Explicit probability threshold must be in [0, 1], got {val}")
        if score_space == 'logit' and not np.isfinite(val):
            raise ValueError(f"Explicit logit threshold must be finite, got {val}")
        if return_score_space:
            return val, score_space
        return val

    if return_score_space:
        return default, score_space_arg or 'probability'
    return default


def main():
    import numpy as np
    from evaluation.generalization import binary_metrics
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['summarize', 'calibrate', 'compare', 'compare_ab', 'ensemble', 'seeds', 'rescore'])
    parser.add_argument('--predictions', nargs='*', default=None)
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
    parser.add_argument('--control_predictions', type=str, default=None, help='Control prediction file for compare_ab')
    parser.add_argument('--smooth_predictions', type=str, default=None, help='Smoothing prediction file for compare_ab')
    parser.add_argument('--control_threshold', type=str, default=None, help='Control threshold file for compare_ab')
    parser.add_argument('--smooth_threshold', type=str, default=None, help='Smoothing threshold file for compare_ab')
    parser.add_argument('--control_checkpoint', type=str, default=None, help='Path to control best.pth')
    parser.add_argument('--smooth_checkpoint', type=str, default=None, help='Path to smoothing best.pth')
    parser.add_argument('--control_name', type=str, default='control', help='Control arm name')
    parser.add_argument('--smooth_name', type=str, default='smoothing', help='Smoothing arm name')
    parser.add_argument('--output_md', type=str, default=None, help='Path to output markdown table')
    parser.add_argument('--score_space', choices=['probability', 'logit'], default='probability',
                        help='Score space for calibration or evaluation (default: probability)')
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

    if args.action == 'compare_ab':
        ctrl_p = args.control_predictions or (args.predictions[0] if args.predictions and len(args.predictions) >= 1 else None)
        smth_p = args.smooth_predictions or (args.predictions[1] if args.predictions and len(args.predictions) >= 2 else None)
        ctrl_t = args.control_threshold or args.threshold_file
        smth_t = args.smooth_threshold or args.threshold_file_other
        if not ctrl_p or not smth_p or not ctrl_t or not smth_t:
            raise ValueError("compare_ab requires control and smoothing predictions and threshold artifacts")
        res = compare_evaluation_runs(
            control_pred_path=ctrl_p,
            smooth_pred_path=smth_p,
            control_thresh_path=ctrl_t,
            smooth_thresh_path=smth_t,
            control_checkpoint_path=args.control_checkpoint,
            smooth_checkpoint_path=args.smooth_checkpoint,
            output_json_path=args.output,
            output_md_path=args.output_md,
            control_name=args.control_name,
            smooth_name=args.smooth_name,
            bootstrap_repeats=args.bootstrap,
            seed=args.seed,
            eval_precision=args.eval_precision or 'fp32',
        )
        print(f"A/B comparison saved to {args.output}")
        return

    if not args.predictions and args.action != 'compare_ab':
        raise ValueError(f"Action '{args.action}' requires --predictions")

    if args.action == 'calibrate':
        if len(args.predictions) != 1:
            raise ValueError('Calibrate one checkpoint at a time')
        calibrate(args.predictions[0], args.output, eval_precision=args.eval_precision,
                  allow_external_dev=args.allow_external_dev,
                  selection_predictions_path=args.selection_predictions,
                  selection_manifest=args.selection_manifest,
                  train_manifest=args.train_manifest,
                  score_space=args.score_space)
        print(f"Calibrated threshold artifact saved to {args.output}")
        return

    if args.action == 'rescore':
        if len(args.predictions) != 1:
            raise ValueError('Rescore takes exactly one prediction file')
        rows = read_predictions(args.predictions[0])
        ckpt_hashes = {r.get('checkpoint_sha256') for r in rows}
        if len(ckpt_hashes) != 1 or not all(known(h) for h in ckpt_hashes):
            raise ValueError("Rescore requires predictions from one identified checkpoint across all rows")
        ckpt_hash = next(iter(ckpt_hashes))
        y = np.array([r['label'] for r in rows])
        p = np.array([r['probability'] for r in rows])
        has_logits = all('logit' in r and r.get('logit') is not None for r in rows) and len(rows) > 0
        logits = np.array([float(r['logit']) for r in rows]) if has_logits else None

        fixed_05_metrics = binary_metrics(y, p, threshold=0.5, logits=logits, score_space='probability')
        frozen_thresh, score_space = load_threshold(
            args.threshold if (args.threshold != 0.5 or args.threshold_file) else None,
            args.threshold_file,
            default=0.5,
            expected_checkpoint_hash=ckpt_hash,
            return_score_space=True,
            score_space_arg=args.score_space
        )
        frozen_metrics = binary_metrics(y, p, threshold=frozen_thresh, logits=logits, score_space=score_space)

        rescore_report = {
            'schema_version': '2.0',
            'predictions_file': args.predictions[0],
            'checkpoint_sha256': ckpt_hash,
            'samples': len(rows),
            'fixed_05': fixed_05_metrics,
            'frozen_threshold': {
                'threshold': frozen_thresh,
                'score_space': score_space,
                'threshold_file': args.threshold_file,
                'metrics': frozen_metrics,
            }
        }
        Path(args.output).write_text(json.dumps(rescore_report, indent=2, allow_nan=False), encoding='utf-8')
        print(f"Rescore report saved to {args.output}")
        auc_05 = fixed_05_metrics['roc_auc']
        auc_str = f"{auc_05:.4f}" if auc_05 is not None else "None"
        print(f"  Fixed 0.5: acc={fixed_05_metrics['accuracy']:.4f}, AUC={auc_str}")
        auc_frz = frozen_metrics['roc_auc']
        auc_frz_str = f"{auc_frz:.4f}" if auc_frz is not None else "None"
        print(f"  Frozen ({score_space}={frozen_thresh:.4f}): acc={frozen_metrics['accuracy']:.4f}, AUC={auc_frz_str}")
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
    aligned_logits = []
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
        if all('logit' in r and r.get('logit') is not None for r in data):
            aligned_logits.append([by_id[i]['logit'] for i in ids])
        else:
            aligned_logits.append(None)

    if args.action == 'ensemble':
        probabilities = np.mean(aligned, axis=0)
        fingerprint = hashlib.sha256('|'.join(str(d[0].get('checkpoint_sha256')) for d in sets).encode()).hexdigest()
        export_predictions(args.output, rows, probabilities, fingerprint)
        return

    m1_hash = sets[0][0].get('checkpoint_sha256') if sets and sets[0] else None
    t1, s1 = load_threshold(args.threshold, args.threshold_file, default=0.5, expected_checkpoint_hash=m1_hash, return_score_space=True, score_space_arg=args.score_space)
    if args.action == 'compare':
        if len(aligned) != 2:
            raise ValueError('Paired comparison takes exactly two prediction files')
        m2_hash = sets[1][0].get('checkpoint_sha256') if len(sets) > 1 and sets[1] else None
        t2, s2 = load_threshold(args.threshold_other, args.threshold_file_other, default=t1, expected_checkpoint_hash=m2_hash, return_score_space=True, score_space_arg=args.score_space)
        m1_summary = summarize(rows, aligned[0], threshold=t1, repeats=args.bootstrap, seed=args.seed, logits=aligned_logits[0], score_space=s1)
        m2_summary = summarize(rows, aligned[1], threshold=t2, repeats=args.bootstrap, seed=args.seed, logits=aligned_logits[1], score_space=s2)
        diff_report = group_intervals(
            [r['label'] for r in rows],
            aligned[0],
            [r.get('group_id', '') for r in rows],
            threshold=t1,
            repeats=args.bootstrap,
            seed=args.seed,
            other=aligned[1],
            other_threshold=t2,
            logits=aligned_logits[0],
            other_logits=aligned_logits[1],
            score_space=s1,
            other_score_space=s2,
        )
        report = {
            'models': [m1_summary, m2_summary],
            'model_thresholds': [t1, t2],
            'first_minus_second': diff_report,
        }
    else:
        report = {'models': [summarize(rows, p, t1, args.bootstrap, args.seed, logits=l, score_space=s1) for p, l in zip(aligned, aligned_logits)]}

    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f"Analysis saved to {args.output}")


if __name__ == '__main__':
    main()
