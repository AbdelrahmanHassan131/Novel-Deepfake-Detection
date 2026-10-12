"""Traceable predictions, source/subgroup metrics and group-level uncertainty."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_fscore_support)
from data.manifest import known, sha256

ALLOWED_CALIBRATION_SPLITS = frozenset({'dev', 'val', 'dev_calibration', 'external_dev'})
FORBIDDEN_CALIBRATION_SPLITS = frozenset({'test', 'internal_test', 'external_test', 'final_test', 'challenge_test', 'unseen_test'})


def compute_interpolated_eer(labels, scores):
    """Compute Equal Error Rate with continuous linear interpolation at FPR/FNR crossing."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return None, None, None, None
    fpr, tpr, thresholds = roc_curve(y, s)
    fnr = 1.0 - tpr
    gap = fpr - fnr
    j = int(np.searchsorted(gap, 0.0))
    if j == 0:
        eer = float(fpr[0])
    elif j >= len(gap):
        eer = float(fpr[-1])
    elif gap[j] == 0.0:
        eer = float(fpr[j])
    else:
        denom = gap[j] - gap[j - 1]
        weight = -gap[j - 1] / denom if denom > 0 else 0.5
        eer = float(fpr[j - 1] + weight * (fpr[j] - fpr[j - 1]))
    return float(eer), fpr, tpr, thresholds


def compute_binary_predictions(probabilities, logits=None, threshold=0.5, score_space=None):
    """
    Compute binary predictions (0=real, 1=fake) using unified decision logic.

    Args:
        probabilities (array-like): Predicted probabilities in [0, 1].
        logits (array-like, optional): Raw unscaled logits.
        threshold (float): Decision threshold. Defaults to 0.5.
        score_space (str, optional): 'probability' (default if None) or 'logit'.
            Invalid strings will raise ValueError.

    Returns:
        tuple: (prediction_boolean_array, decision_score_space_str)
    """
    if score_space is not None and score_space not in ('probability', 'logit'):
        raise ValueError(f"Unknown score_space '{score_space}'; expected 'logit' or 'probability'.")

    p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError('Probabilities must be finite and in [0,1]')

    if logits is not None:
        z = np.asarray(logits, dtype=float)
        if z.shape != p.shape:
            raise ValueError('Logits shape must match probabilities')
        if not np.isfinite(z).all():
            raise ValueError('Logits must be finite floats')
    else:
        z = None

    effective_space = score_space or 'probability'

    if effective_space == 'logit':
        t = float(threshold)
        if not np.isfinite(t):
            raise ValueError(f'Logit threshold must be a finite float, got {threshold}')
        if z is None:
            raise ValueError("Score space is 'logit' but no logits provided.")
        prediction = (z >= t)
        return prediction, 'logit'
    else:
        # probability score space
        t = float(threshold)
        if not (0.0 <= t <= 1.0) or not np.isfinite(t):
            raise ValueError(f'Probability threshold must be finite and in [0,1], got {threshold}')
        if z is not None:
            if t == 0.5:
                prediction = (z >= 0.0)
                return prediction, 'logit'
            elif 0.0 < t < 1.0:
                import math
                tau_logit = float(math.log(t / (1.0 - t)))
                prediction = (z >= tau_logit)
                return prediction, 'logit'
            elif t == 0.0:
                prediction = np.ones(len(p), dtype=bool)
                return prediction, 'probability'
            else:  # t == 1.0
                # Mathematical probability boundary 1.0 corresponds to logit +infinity.
                # For finite logits, no sample can reach +inf, so all samples are predicted real (0).
                prediction = np.zeros(len(p), dtype=bool)
                return prediction, 'logit'
        else:
            prediction = (p >= t)
            return prediction, 'probability'


def binary_metrics(labels, probabilities, threshold=0.5, logits=None, score_space=None):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if not len(y) or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected binary labels with canonical mapping: real=0, fake=1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0,1]')

    if logits is not None:
        logits = np.asarray(logits, dtype=float)
        if logits.shape != p.shape:
            raise ValueError('Logits shape must match probabilities')
        if not np.isfinite(logits).all():
            raise ValueError('Logits must be finite floats')

    prediction, decision_score_space = compute_binary_predictions(
        probabilities=p, logits=logits, threshold=threshold, score_space=score_space
    )

    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    prec, rec, f1, counts = precision_recall_fscore_support(y, prediction, labels=[0, 1], zero_division=0)

    real_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    fake_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None

    quantiles_probs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    real_scores = p[y == 0]
    fake_scores = p[y == 1]

    score_quantiles = {
        'real': {f'q{int(q*100):02d}': float(np.quantile(real_scores, q)) for q in quantiles_probs} if len(real_scores) > 0 else None,
        'fake': {f'q{int(q*100):02d}': float(np.quantile(fake_scores, q)) for q in quantiles_probs} if len(fake_scores) > 0 else None,
    }

    logit_quantiles = None
    if logits is not None:
        real_logits = logits[y == 0]
        fake_logits = logits[y == 1]
        logit_quantiles = {
            'real': {f'q{int(q*100):02d}': float(np.quantile(real_logits, q)) for q in quantiles_probs} if len(real_logits) > 0 else None,
            'fake': {f'q{int(q*100):02d}': float(np.quantile(fake_logits, q)) for q in quantiles_probs} if len(fake_logits) > 0 else None,
        }

    saturation_counts = {
        'real_low_p0001': int(np.sum(real_scores <= 1e-4)),
        'real_high_p9999': int(np.sum(real_scores >= 1.0 - 1e-4)),
        'fake_low_p0001': int(np.sum(fake_scores <= 1e-4)),
        'fake_high_p9999': int(np.sum(fake_scores >= 1.0 - 1e-4)),
        'total_saturated': int(np.sum((p <= 1e-4) | (p >= 1.0 - 1e-4))),
    }

    # Authoritative ranking scores
    if logits is not None:
        ranking_scores = logits
        ranking_score_space = 'logit'
    else:
        ranking_scores = p
        ranking_score_space = 'probability'

    roc_auc = float(roc_auc_score(y, ranking_scores)) if both else None
    avg_prec = float(average_precision_score(y, ranking_scores)) if both else None
    roc_auc_logit = float(roc_auc_score(y, logits)) if (both and logits is not None) else None
    roc_auc_probability = float(roc_auc_score(y, p)) if both else None

    eer = None
    fpr_at_tpr95 = None
    tpr_at_fpr01 = None
    if both:
        eer, fpr, tpr, _ = compute_interpolated_eer(y, ranking_scores)
        tpr95_indices = np.flatnonzero(tpr >= .95)
        fpr_at_tpr95 = float(fpr[tpr95_indices].min()) if len(tpr95_indices) else None
        fpr01_indices = np.flatnonzero(fpr <= .01)
        tpr_at_fpr01 = float(tpr[fpr01_indices].max()) if len(fpr01_indices) else None

    result = dict(
        schema_version='2.0',
        samples=len(y),
        real=int(tn + fp),
        fake=int(tp + fn),
        real_recall=real_recall,
        fake_recall=fake_recall,
        accuracy=float((tn + tp) / len(y)),
        balanced_accuracy=float(rec.mean()) if both else None,
        roc_auc=roc_auc,
        roc_auc_logit=roc_auc_logit,
        roc_auc_probability=roc_auc_probability,
        average_precision=avg_prec,
        fpr=float(fp / (tn + fp)) if tn + fp else None,
        fnr=float(fn / (tp + fn)) if tp + fn else None,
        threshold=float(threshold),
        score_space=score_space or 'probability',
        ranking_score_space=ranking_score_space,
        decision_score_space=decision_score_space,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        macro_f1=float(f1.mean()) if both else None,
        per_class={
            name: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]), support=int(counts[i]))
            for i, name in enumerate(('real', 'fake'))
        },
        score_quantiles=score_quantiles,
        logit_quantiles=logit_quantiles,
        saturation_counts=saturation_counts,
        eer=eer,
        interpolated_eer=eer,
        fpr_at_tpr95=fpr_at_tpr95,
        tpr_at_fpr01=tpr_at_fpr01,
        status='computed' if both else 'single_class_evaluated_ranking_metrics_undefined',
    )
    if not both:
        result['note'] = 'Single-class evaluation: ROC AUC, balanced accuracy, and EER are undefined and marked None.'
    if logits is None:
        result['score_space_fallback'] = True
    return result


def group_intervals(labels, probabilities, groups, threshold=.5, repeats=200, seed=42, other=None, other_threshold=None,
                    logits=None, other_logits=None, score_space=None, other_score_space=None):
    """Cluster percentile bootstrap; paired differences share each resample.

    Supports independent thresholds per model in paired comparison.
    """
    if repeats < 0:
        raise ValueError('Bootstrap repeats must be nonnegative')
    if not repeats:
        return {'status': 'not_requested'}
    if len(groups) != len(labels) or not all(known(g) for g in groups):
        return {'status': 'unavailable: missing independent group IDs'}
    y, p, groups = np.asarray(labels), np.asarray(probabilities), np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {'status': 'unavailable: fewer than two independent groups'}
    grouped = defaultdict(list)
    for i, group in enumerate(groups):
        grouped[group].append(i)
    indices = [np.asarray(grouped[g]) for g in unique]
    rng = np.random.default_rng(seed)
    metrics = {k: [] for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'eer', 'fpr', 'fnr')}
    t2 = threshold if other_threshold is None else other_threshold
    z1 = np.asarray(logits) if logits is not None else None
    z2 = np.asarray(other_logits) if other_logits is not None else None

    for _ in range(repeats):
        selected = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        first = binary_metrics(y[selected], p[selected], threshold,
                               logits=z1[selected] if z1 is not None else None,
                               score_space=score_space)
        second = binary_metrics(y[selected], np.asarray(other)[selected], t2,
                                logits=z2[selected] if z2 is not None else None,
                                score_space=other_score_space) if other is not None else None
        for key in metrics:
            if first[key] is not None and (second is None or second[key] is not None):
                metrics[key].append(first[key] - second[key] if second else first[key])

    # Compute actual aligned full-sample difference on unresampled data
    first_full = binary_metrics(y, p, threshold, logits=z1, score_space=score_space)
    second_full = binary_metrics(y, np.asarray(other), t2, logits=z2, score_space=other_score_space) if other is not None else None
    if second_full is not None:
        observed_difference = {
            k: float(first_full[k] - second_full[k])
            for k in metrics
            if first_full.get(k) is not None and second_full.get(k) is not None
        }
    else:
        observed_difference = {
            k: float(first_full[k])
            for k in metrics
            if first_full.get(k) is not None
        }

    intervals_95 = {}
    for k, v in metrics.items():
        if len(v) > 0:
            low = float(np.quantile(v, .025))
            high = float(np.quantile(v, .975))
            obs = observed_difference.get(k)
            intervals_95[k] = {
                'observed': obs,
                'low': low,
                'high': high,
                'valid_repeats': len(v),
                'ci_excludes_zero': bool(low > 0 or high < 0),
            }

    return {
        'status': 'computed',
        'method': 'paired cluster percentile bootstrap' if other is not None else 'cluster percentile bootstrap',
        'group_count': len(unique),
        'requested_repeats': repeats,
        'seed': seed,
        'observed_difference': observed_difference,
        'intervals_95': intervals_95,
    }


def export_predictions(path, records, probabilities, checkpoint_hash=None, logits=None, eval_precision=None):
    if len(records) != len(probabilities):
        raise ValueError('Prediction count does not match ordered dataset records')
    if logits is not None and len(logits) != len(probabilities):
        raise ValueError('Logits count does not match ordered dataset records')
    rows = []
    for i, (r, p) in enumerate(zip(records, probabilities)):
        row = dict(r, probability=float(p), checkpoint_sha256=checkpoint_hash or 'unknown')
        if logits is not None:
            row['logit'] = float(logits[i])
        if eval_precision is not None:
            row['eval_precision'] = str(eval_precision)
        rows.append(row)
    from data.manifest import write_manifest
    write_manifest(path, rows)


def read_predictions(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['label'], row['probability'] = int(row['label']), float(row['probability'])
        if 'logit' in row and row['logit'] not in ('', None):
            row['logit'] = float(row['logit'])
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Prediction sample IDs must be unique')
    logits = [r['logit'] for r in rows] if all('logit' in r and r['logit'] is not None for r in rows) and len(rows) > 0 else None
    binary_metrics([r['label'] for r in rows], [r['probability'] for r in rows], logits=logits)
    return rows


def summarize(records, probabilities, threshold=.5, repeats=200, seed=42, logits=None, score_space=None):
    y, p = np.array([r['label'] for r in records]), np.asarray(probabilities)
    if logits is None and len(records) > 0 and all('logit' in r and r.get('logit') is not None for r in records):
        logits = np.array([float(r['logit']) for r in records])
    report = {
        'schema_version': '2.0',
        'overall': binary_metrics(y, p, threshold, logits=logits, score_space=score_space),
        'uncertainty': group_intervals(y, p, [r.get('group_id', '') for r in records], threshold, repeats, seed,
                                       logits=logits, score_space=score_space),
        'breakdowns': {},
    }
    fields = ('dataset_source', 'generator', 'manipulation_type', 'age_group', 'gender', 'ethnicity', 'skin_tone')
    for field in fields:
        groups = defaultdict(list)
        for i, row in enumerate(records):
            groups[row.get(field) or 'unknown'].append(i)
        if any(known(g) for g in groups):
            breakdowns = {}
            for g, ix in groups.items():
                ix_arr = np.asarray(ix)
                sub_logits = logits[ix_arr] if logits is not None else None
                res = binary_metrics(y[ix_arr], p[ix_arr], threshold, logits=sub_logits, score_space=score_space)
                res['subgroup_sample_count'] = len(ix)
                res['subgroup_share'] = float(len(ix) / len(records))
                if len(ix) < 30:
                    res['caution_small_subgroup'] = True
                breakdowns[g] = res
            report['breakdowns'][field] = breakdowns
    # Aggregate video frames by video AND class; group_id can link real and fake.
    videos = defaultdict(list)
    for i, row in enumerate(records):
        if known(row.get('source_video_id')):
            videos[(row.get('dataset_source'), row['source_video_id'], row['label'])].append(i)
    if videos:
        if score_space == 'logit':
            report['video_level'] = {
                'status': 'unavailable: image-level logit threshold cannot be applied to mean-probability video aggregates without video-level calibration'
            }
            report['video_aggregation'] = 'mean frame probability by dataset_source, source_video_id, label; known IDs only'
            report['video_aggregation_score_space'] = 'probability'
        else:
            report['video_level'] = binary_metrics([key[2] for key in videos], [float(p[ix].mean()) for ix in videos.values()], threshold, score_space='probability')
            report['video_aggregation'] = 'mean frame probability by dataset_source, source_video_id, label; known IDs only'
            report['video_aggregation_score_space'] = 'probability'
    report['fairness_note'] = 'Only supplied annotations are analyzed; missing annotations are not inferred. Small groups require cautious interpretation.'
    return report


def calibrate(prediction_path, output_path, eval_precision=None, allow_external_dev=False,
              selection_predictions_path=None, selection_manifest=None, train_manifest=None,
              score_space='probability'):
    import os
    rows = read_predictions(prediction_path)
    splits = {r['split'] for r in rows}
    if splits & FORBIDDEN_CALIBRATION_SPLITS:
        raise ValueError(f'Forbidden: cannot fit threshold or calibrate on test split(s): {splits & FORBIDDEN_CALIBRATION_SPLITS}')
    if ('external_dev' in splits or 'external_val' in splits) and not allow_external_dev:
        raise ValueError("Operating threshold calibration on external development data ('external_dev') requires explicit permission (pass allow_external_dev=True or --allow_external_dev).")
    if splits - ALLOWED_CALIBRATION_SPLITS:
        raise ValueError(f'Threshold calibration accepts only development splits {sorted(ALLOWED_CALIBRATION_SPLITS)}, got {sorted(splits)}')
    y, p = np.array([r['label'] for r in rows]), np.array([r['probability'] for r in rows])
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration requires both classes')

    if score_space == 'logit':
        if not all('logit' in r and r.get('logit') is not None for r in rows):
            raise ValueError("Logit calibration requires 'logit' column in prediction file.")
        logits = np.array([float(r['logit']) for r in rows])
        if not np.all(np.isfinite(logits)):
            raise ValueError("Logit calibration requires finite logits.")
        fpr, tpr, thresholds = roc_curve(y, logits)
        eligible = np.flatnonzero(np.isfinite(thresholds))
        if len(eligible) == 0:
            raise ValueError("No finite ROC thresholds found for logit calibration.")
        j_scores = (tpr - fpr)[eligible]
        max_j = np.max(j_scores)
        tie_candidates = eligible[np.flatnonzero(j_scores == max_j)]
        selected = tie_candidates[0]
        selected_threshold = float(thresholds[selected])
    elif score_space == 'probability':
        fpr, tpr, thresholds = roc_curve(y, p)
        eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= 1) & (thresholds >= 0))
        if len(eligible) == 0:
            raise ValueError("No finite ROC thresholds in [0, 1] found for probability calibration.")
        j_scores = (tpr - fpr)[eligible]
        max_j = np.max(j_scores)
        tie_candidates = eligible[np.flatnonzero(j_scores == max_j)]
        selected = tie_candidates[0]
        selected_threshold = float(thresholds[selected])
    else:
        raise ValueError(f"Unknown score_space '{score_space}'; expected 'logit' or 'probability'.")

    hashes = {r.get('checkpoint_sha256') for r in rows}
    if len(hashes) != 1 or not all(known(h) for h in hashes):
        raise ValueError('Calibration requires predictions from one identified checkpoint')

    # Validate precision provenance
    file_precisions = {r.get('eval_precision') for r in rows if known(r.get('eval_precision'))}
    if len(file_precisions) == 1:
        recorded_prec = next(iter(file_precisions))
        if eval_precision is not None and str(eval_precision).lower() != recorded_prec.lower():
            raise ValueError(
                f"Evaluation precision mismatch: prediction records state '{recorded_prec}', "
                f"but caller specified '{eval_precision}'."
            )
        prec = recorded_prec
    elif len(file_precisions) > 1:
        raise ValueError(f"Mixed prediction precisions detected in calibration file: {file_precisions}")
    else:
        if eval_precision is not None:
            raise ValueError('Prediction file has no recorded precision; a caller flag cannot certify it.')
        prec = 'unknown'
    if file_precisions and any(not known(r.get('eval_precision')) for r in rows):
        raise ValueError('Precision provenance is missing from some prediction rows.')

    # Verify sample/group/hash separation before claiming independence
    overlap_reasons = []
    reference_records = []
    reference_evidence = {}
    for role, path in (('selection_predictions', selection_predictions_path),
                       ('selection_manifest', selection_manifest), ('train_manifest', train_manifest)):
        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f'{role} does not exist: {path}')
            reference_evidence[role] = {'path': str(path), 'sha256': sha256(path)}
    if selection_predictions_path and os.path.isfile(selection_predictions_path):
        reference_records.extend(read_predictions(selection_predictions_path))
    if selection_manifest and os.path.isfile(selection_manifest):
        from data.manifest import read_manifest
        reference_records.extend(read_manifest(selection_manifest, check_files=False))
    if train_manifest and os.path.isfile(train_manifest):
        from data.manifest import read_manifest
        reference_records.extend(read_manifest(train_manifest, check_files=False))

    if reference_records:
        from data.manifest import find_connected_components
        linked = ([dict(r, _calibration_role='reference') for r in reference_records] +
                  [dict(r, _calibration_role='calibration') for r in rows])
        if any(len({r['_calibration_role'] for r in component}) > 1
               for component in find_connected_components(linked).values()):
            overlap_reasons.append('connected group/video/original/identity/content component overlap')
        ref_samples = {r['sample_id'] for r in reference_records if known(r.get('sample_id'))}
        ref_groups = {r['group_id'] for r in reference_records if known(r.get('group_id'))}
        ref_videos = {r['source_video_id'] for r in reference_records if known(r.get('source_video_id'))}
        ref_hashes = {r['sha256'] for r in reference_records if known(r.get('sha256'))}
        ref_paths = {r.get('relative_path', r.get('path')) for r in reference_records if known(r.get('relative_path', r.get('path')))}

        cal_samples = {r['sample_id'] for r in rows if known(r.get('sample_id'))}
        cal_groups = {r['group_id'] for r in rows if known(r.get('group_id'))}
        cal_videos = {r['source_video_id'] for r in rows if known(r.get('source_video_id'))}
        cal_hashes = {r['sha256'] for r in rows if known(r.get('sha256'))}
        cal_paths = {r.get('relative_path', r.get('path')) for r in rows if known(r.get('relative_path', r.get('path')))}

        if cal_samples & ref_samples:
            overlap_reasons.append(f"sample_id overlap ({len(cal_samples & ref_samples)} samples)")
        if cal_groups & ref_groups:
            overlap_reasons.append(f"group_id overlap ({len(cal_groups & ref_groups)} groups)")
        if cal_videos & ref_videos:
            overlap_reasons.append(f"source_video_id overlap ({len(cal_videos & ref_videos)} videos)")
        if cal_hashes & ref_hashes:
            overlap_reasons.append(f"image content sha256 overlap ({len(cal_hashes & ref_hashes)} images)")
        if cal_paths & ref_paths:
            overlap_reasons.append(f"file path overlap ({len(cal_paths & ref_paths)} files)")

        if overlap_reasons:
            raise ValueError(
                f"Data leakage detected: calibration cohort overlaps with training/selection data: {'; '.join(overlap_reasons)}"
            )

    if 'external_dev' in splits:
        independence_status = 'external_dev_calibration'
    elif 'dev_calibration' in splits and not ('dev' in splits or 'val' in splits):
        if reference_records:
            # Supplied reference files are not yet bound to the checkpoint's
            # actual training/selection cohorts. Do not certify independence.
            independence_status = 'reference_overlap_checked_independence_unverified'
        else:
            independence_status = 'unverified_dev_calibration'
    else:
        independence_status = 'same_dev_reused_selection_and_calibration'

    artifact = dict(
        schema_version='2.0',
        score_space=score_space,
        threshold=selected_threshold,
        criterion='development Youden J',
        label_mapping={'real': 0, 'fake': 1},
        decision_rule='fake if logit >= logit_threshold' if score_space == 'logit' else 'fake if probability >= threshold',
        checkpoint_sha256=next(iter(hashes)),
        prediction_sha256=sha256(prediction_path),
        source_splits=sorted(splits),
        eval_precision=str(prec),
        independence_status=independence_status,
        reference_evidence=reference_evidence,
    )
    if score_space == 'logit':
        artifact['logit_threshold'] = selected_threshold

    Path(output_path).write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    return artifact


def summarize_seeds(seed_prediction_paths: List[str], threshold: float = 0.5,
                    thresholds: Optional[List[float]] = None,
                    repeats: int = 200, seed: int = 42,
                    run_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Summarize performance across at least three independently trained runs.

    Requires identical evaluation cohorts (same sample IDs, labels, groups, and splits)
    and verifies independent training run/seed provenance. Repeated inference from the
    same checkpoint or different epochs from the same training run are strictly rejected.
    Supports individual development-selected thresholds per seed.
    """
    if len(seed_prediction_paths) < 3:
        raise ValueError(
            f"Multi-seed summary requires at least 3 independently trained runs; got {len(seed_prediction_paths)}. "
            "Do not report seed variability without sufficient independent trials."
        )
    if thresholds is not None and len(thresholds) != len(seed_prediction_paths):
        raise ValueError(
            f"Expected {len(seed_prediction_paths)} thresholds for multi-seed summary, got {len(thresholds)}."
        )
    if run_ids is not None:
        if len(run_ids) != len(seed_prediction_paths):
            raise ValueError(f"Expected {len(seed_prediction_paths)} run_ids, got {len(run_ids)}.")
        if len(set(run_ids)) != len(run_ids):
            raise ValueError(f"All run_ids in multi-seed summary must be distinct; got duplicate in {run_ids}.")

    all_predictions = [read_predictions(p) for p in seed_prediction_paths]

    # 1. Enforce identical evaluation cohorts across all seeds
    base_rows = all_predictions[0]
    base_ids = [r['sample_id'] for r in base_rows]
    base_id_set = set(base_ids)
    if len(base_ids) != len(base_id_set):
        raise ValueError("Prediction sample IDs must be unique within file.")

    for idx, rows in enumerate(all_predictions[1:], start=2):
        if len(rows) != len(base_rows):
            raise ValueError(
                f"Cohort size mismatch in seed file {idx} ({len(rows)} vs {len(base_rows)}). "
                "All seeds must be evaluated on exactly the same evaluation cohort."
            )
        by_id = {r['sample_id']: r for r in rows}
        if set(by_id.keys()) != base_id_set:
            raise ValueError(
                f"Cohort sample ID mismatch in seed file {idx}. "
                "All seeds must be evaluated on exactly the same evaluation cohort."
            )
        for base_r in base_rows:
            other_r = by_id[base_r['sample_id']]
            for key in ('label', 'group_id', 'split'):
                if base_r.get(key) != other_r.get(key):
                    raise ValueError(
                        f"Evaluation cohort metadata mismatch for sample {base_r['sample_id']} in seed file {idx}: "
                        f"'{key}' differs ('{base_r.get(key)}' vs '{other_r.get(key)}'). "
                        "All seeds must share identical evaluation labels, groups, and splits."
                    )

    # 2. Enforce independent training run provenance and unique checkpoints
    checkpoint_hashes = []
    for idx, rows in enumerate(all_predictions, start=1):
        hashes = {r.get('checkpoint_sha256') for r in rows if known(r.get('checkpoint_sha256'))}
        if len(hashes) != 1:
            raise ValueError(f"Seed file {idx} must originate from a single identifiable checkpoint.")
        checkpoint_hashes.append(next(iter(hashes)))

    # Reject duplicate checkpoints first
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError(
            "Repeated inference from the same checkpoint is not an independent training run. "
            "Found duplicate checkpoint hash in multi-seed evaluation."
        )

    has_any_explicit_prov = any(
        any(known(r.get('run_id')) or known(r.get('training_seed')) or known(r.get('seed')) for r in rows)
        for rows in all_predictions
    )

    detected_runs = []
    detected_seeds = []
    for idx, rows in enumerate(all_predictions, start=1):
        run_ids_found = {str(r.get('run_id')) for r in rows if known(r.get('run_id'))}
        seeds_found = {int(r.get('training_seed')) for r in rows if known(r.get('training_seed'))}
        if not seeds_found:
            seeds_found = {int(r.get('seed')) for r in rows if known(r.get('seed'))}

        if has_any_explicit_prov:
            if not run_ids_found and seeds_found:
                if run_ids is not None and len(run_ids) >= idx:
                    run_ids_found = {str(run_ids[idx - 1])}
                else:
                    run_ids_found = {f"run_seed_{next(iter(seeds_found))}"}

            if not run_ids_found or not seeds_found:
                raise ValueError(
                    f"Seed file {idx} missing required run provenance ('run_id' and 'training_seed'). "
                    "Independent seed reporting requires complete provenance bound to checkpoint identity."
                )
        else:
            if not run_ids_found:
                run_ids_found = {f"run_{checkpoint_hashes[idx - 1]}"}
            if not seeds_found:
                seeds_found = {idx}

        if len(run_ids_found) > 1:
            raise ValueError(f"Seed file {idx} contains mixed run_ids: {run_ids_found}")
        if len(seeds_found) > 1:
            raise ValueError(f"Seed file {idx} contains mixed training_seeds: {seeds_found}")

        detected_runs.append(next(iter(run_ids_found)))
        detected_seeds.append(next(iter(seeds_found)))

    # Reject duplicate run IDs (e.g. multiple epochs from the same training run)
    if len(set(detected_runs)) != len(detected_runs):
        raise ValueError(
            "Different epochs or outputs from the same training run/seed cannot be passed as independent seeds. "
            f"Found duplicate run_id: {detected_runs}"
        )

    # Reject duplicate training seeds (e.g. distinct runs using the exact same random seed)
    if len(set(detected_seeds)) != len(detected_seeds):
        raise ValueError(
            "Different epochs or outputs from the same training run/seed cannot be passed as independent seeds. "
            f"Found duplicate training_seed: {detected_seeds}"
        )

    # Validate caller-supplied run_ids against detected provenance
    if run_ids is not None:
        for idx, (caller_id, det_id) in enumerate(zip(run_ids, detected_runs), start=1):
            if str(caller_id) != str(det_id):
                raise ValueError(
                    f"Caller-supplied run_id '{caller_id}' conflicts with detected run_id '{det_id}' in seed file {idx}. "
                    "Caller annotations cannot override recorded run provenance."
                )

    seed_summaries = []
    metrics_to_track = ('roc_auc', 'balanced_accuracy', 'accuracy', 'eer', 'fpr_at_tpr95')
    metric_values = {k: [] for k in metrics_to_track}
    used_thresholds = []

    for i, rows in enumerate(all_predictions):
        probs = [r['probability'] for r in rows]
        seed_thresh = thresholds[i] if thresholds is not None else threshold
        used_thresholds.append(float(seed_thresh))
        summary = summarize(rows, probs, threshold=seed_thresh, repeats=repeats, seed=seed)
        seed_summaries.append({
            'seed_index': i + 1,
            'prediction_file': seed_prediction_paths[i],
            'checkpoint_sha256': checkpoint_hashes[i],
            'threshold_used': float(seed_thresh),
            'overall': summary['overall'],
        })
        for k in metrics_to_track:
            v = summary['overall'].get(k)
            if v is not None:
                metric_values[k].append(v)

    aggregated = {}
    for k, vals in metric_values.items():
        if len(vals) == len(seed_prediction_paths):
            aggregated[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'values': [float(v) for v in vals],
            }
        else:
            aggregated[k] = {'status': 'unavailable: metric undefined for some seeds'}

    res = {
        'num_independent_seeds': len(seed_prediction_paths),
        'checkpoint_hashes': checkpoint_hashes,
        'thresholds_used': used_thresholds,
        'aggregated_metrics': aggregated,
        'per_seed_runs': seed_summaries,
        'note': 'Seed variability measures training stochasticity across distinct checkpoints, not sample bootstrap error.',
    }
    res.update(aggregated)
    return res


def load_validated_threshold(threshold_file, expected_checkpoint_hash=None, eval_precision=None):
    """
    Load and strictly validate a threshold artifact JSON file.

    Supports:
      - New logit artifacts ('score_space': 'logit')
      - Probability artifacts ('score_space': 'probability')
      - Narrowly validated legacy Colab external calibration schema
        ('calibration_status': 'previously_inspected_external_data_not_independent_test')

    Enforces:
      - Forbidden splits can NEVER be bypassed, even with legacy tags.
      - Checkpoint hash binding against expected_checkpoint_hash.
      - Floating-point precision matching against eval_precision.
      - Conflicting threshold fields rejected.
      - Canonical label mapping and decision rules.
      - Finite thresholds.

    Returns:
        tuple: (threshold_float, score_space_str)
    """
    t_path = Path(threshold_file)
    if not t_path.is_file():
        raise FileNotFoundError(f"Threshold artifact not found: {threshold_file}")

    data = json.loads(t_path.read_text(encoding='utf-8'))

    # 1. Forbidden split checks MUST always run first
    source_splits = data.get('source_splits')
    if source_splits is not None:
        splits = set(source_splits)
        forbidden = splits.intersection(FORBIDDEN_CALIBRATION_SPLITS)
        if forbidden:
            raise ValueError(f"Forbidden: threshold was calibrated on forbidden evaluation split(s): {sorted(forbidden)}")
    else:
        splits = None

    # 2. Conflicting threshold fields
    has_logit_thresh = 'logit_threshold' in data
    has_thresh = 'threshold' in data
    if has_logit_thresh and has_thresh:
        if float(data['logit_threshold']) != float(data['threshold']):
            raise ValueError(
                f"Conflicting threshold fields in {threshold_file}: "
                f"logit_threshold={data['logit_threshold']} vs threshold={data['threshold']}"
            )

    # 3. Score space validation
    score_space = data.get('score_space', 'probability')
    if score_space not in ('logit', 'probability'):
        raise ValueError(f"Unknown score_space '{score_space}' in {threshold_file}; expected 'logit' or 'probability'.")

    # 4. Legacy vs standard schema validation
    legacy_tag = data.get('calibration_status')
    if legacy_tag == 'previously_inspected_external_data_not_independent_test':
        if score_space != 'logit':
            raise ValueError(f"Legacy external calibration artifact in {threshold_file} must have score_space='logit'")
        if not has_logit_thresh and not has_thresh:
            raise ValueError(f"Legacy external calibration artifact missing 'logit_threshold' in {threshold_file}")
        if not known(data.get('calibration_prediction_sha256')):
            raise ValueError(f"Legacy external calibration artifact missing valid 'calibration_prediction_sha256' in {threshold_file}")
        label_map = data.get('label_mapping')
        if label_map != {'real': 0, 'fake': 1} and label_map != ['real', 'fake']:
            raise ValueError(f"Legacy artifact label_mapping must be {{'real': 0, 'fake': 1}}, got {label_map}")
        decision_rule = data.get('decision_rule')
        if decision_rule != 'fake if logit >= logit_threshold':
            raise ValueError(f"Legacy artifact decision_rule mismatch in {threshold_file}: got '{decision_rule}'")
    elif legacy_tag is not None:
        raise ValueError(f"Unknown calibration_status '{legacy_tag}' in {threshold_file}")
    else:
        # Standard calibration artifact requires source_splits from allowed development sets
        if not source_splits:
            raise ValueError(f"Threshold artifact {threshold_file} missing 'source_splits'")
        if not splits or (splits - ALLOWED_CALIBRATION_SPLITS):
            raise ValueError(
                f"Threshold artifact {threshold_file} must come from development data {sorted(ALLOWED_CALIBRATION_SPLITS)}, got {sorted(splits)}"
            )

    # 5. Extract and validate threshold value
    if score_space == 'logit':
        if has_logit_thresh:
            val = float(data['logit_threshold'])
        elif has_thresh:
            val = float(data['threshold'])
        else:
            raise ValueError(f"Logit threshold artifact {threshold_file} missing 'logit_threshold' or 'threshold'")
        if not np.isfinite(val):
            raise ValueError(f"Logit threshold in {threshold_file} must be finite")
    else:
        if has_logit_thresh:
            raise ValueError(f"Probability threshold artifact {threshold_file} cannot contain 'logit_threshold'")
        if not has_thresh:
            raise ValueError(f"Probability threshold artifact {threshold_file} missing 'threshold'")
        val = float(data['threshold'])
        if not (0.0 <= val <= 1.0) or not np.isfinite(val):
            raise ValueError(f"Invalid threshold {val} in {threshold_file}; must be in [0, 1]")

    # 6. Label mapping if present
    if 'label_mapping' in data:
        lm = data['label_mapping']
        if lm != {'real': 0, 'fake': 1} and lm != ['real', 'fake']:
            raise ValueError(f"Threshold artifact label_mapping must be {{'real': 0, 'fake': 1}}, got {lm}")

    # 7. Checkpoint hash binding
    artifact_ckpt = data.get('checkpoint_sha256')
    if expected_checkpoint_hash is not None:
        if not known(artifact_ckpt) or artifact_ckpt != expected_checkpoint_hash:
            raise ValueError(
                f"Threshold artifact {threshold_file} was calibrated for checkpoint '{artifact_ckpt}', "
                f"which does not match evaluation checkpoint '{expected_checkpoint_hash}'."
            )

    # 8. Eval precision validation
    recorded_precision = data.get('eval_precision', 'unknown')
    if eval_precision is not None:
        if recorded_precision not in ('unknown', eval_precision):
            raise ValueError(
                f"Evaluation requires {eval_precision} precision, but threshold artifact {threshold_file} was calibrated on {recorded_precision}."
            )

    return val, score_space


def compare_evaluation_runs(
    control_pred_path: str,
    smooth_pred_path: str,
    control_thresh_path: str,
    smooth_thresh_path: str,
    control_checkpoint_path: Optional[str] = None,
    smooth_checkpoint_path: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_md_path: Optional[str] = None,
    control_name: str = 'control',
    smooth_name: str = 'smoothing',
    bootstrap_repeats: int = 500,
    seed: int = 42,
    eval_precision: str = 'fp32',
) -> Dict[str, Any]:
    """
    Compare two matched evaluation runs (e.g. Arm A Control vs Arm B Smoothing).

    Verifies:
      - Whole-file checkpoint digest consistency across ALL prediction rows.
      - Checkpoint binding against physical best.pth files if provided.
      - Threshold artifact binding against matched checkpoints.
      - Strict sample alignment (sample ID, ground-truth label, and canonical path).
      - FP32 evaluation precision across all prediction rows.
      - Generates paired cluster percentile bootstrap difference and SHA256 digests.
    """
    p_ctrl_file = Path(control_pred_path)
    p_smth_file = Path(smooth_pred_path)
    t_ctrl_file = Path(control_thresh_path)
    t_smth_file = Path(smooth_thresh_path)

    if not p_ctrl_file.is_file():
        raise FileNotFoundError(f"Missing control prediction file: {control_pred_path}")
    if not p_smth_file.is_file():
        raise FileNotFoundError(f"Missing smoothing prediction file: {smooth_pred_path}")
    if not t_ctrl_file.is_file():
        raise FileNotFoundError(f"Missing control threshold file: {control_thresh_path}")
    if not t_smth_file.is_file():
        raise FileNotFoundError(f"Missing smoothing threshold file: {smooth_thresh_path}")

    ctrl_pred_sha = sha256(str(p_ctrl_file))
    smth_pred_sha = sha256(str(p_smth_file))
    ctrl_thresh_sha = sha256(str(t_ctrl_file))
    smth_thresh_sha = sha256(str(t_smth_file))

    # Read predictions
    rows_ctrl = read_predictions(str(p_ctrl_file))
    rows_smth = read_predictions(str(p_smth_file))

    if len(rows_ctrl) == 0:
        raise ValueError("Control predictions file is empty")
    if len(rows_ctrl) != len(rows_smth):
        raise ValueError(f"Prediction row count mismatch: control has {len(rows_ctrl)}, smoothing has {len(rows_smth)}")

    # Checkpoint digest validation across ALL rows
    ctrl_hashes = {r.get('checkpoint_sha256') for r in rows_ctrl}
    if len(ctrl_hashes) != 1 or not all(known(h) for h in ctrl_hashes):
        raise ValueError("Arm A (Control) predictions must have exactly one consistent, known checkpoint_sha256 across all rows")
    ctrl_ckpt_hash = next(iter(ctrl_hashes))

    smth_hashes = {r.get('checkpoint_sha256') for r in rows_smth}
    if len(smth_hashes) != 1 or not all(known(h) for h in smth_hashes):
        raise ValueError("Arm B (Smoothing) predictions must have exactly one consistent, known checkpoint_sha256 across all rows")
    smth_ckpt_hash = next(iter(smth_hashes))

    # If physical checkpoints are provided, bind against their actual hashes
    if control_checkpoint_path is not None:
        p_ctrl_ckpt = Path(control_checkpoint_path)
        if not p_ctrl_ckpt.is_file():
            raise FileNotFoundError(f"Control checkpoint file not found: {control_checkpoint_path}")
        expected_ctrl_ckpt = sha256(str(p_ctrl_ckpt))
        if ctrl_ckpt_hash != expected_ctrl_ckpt:
            raise ValueError(f"Control predictions checkpoint hash '{ctrl_ckpt_hash}' does not match best.pth '{expected_ctrl_ckpt}'")

    if smooth_checkpoint_path is not None:
        p_smth_ckpt = Path(smooth_checkpoint_path)
        if not p_smth_ckpt.is_file():
            raise FileNotFoundError(f"Smoothing checkpoint file not found: {smooth_checkpoint_path}")
        expected_smth_ckpt = sha256(str(p_smth_ckpt))
        if smth_ckpt_hash != expected_smth_ckpt:
            raise ValueError(f"Smoothing predictions checkpoint hash '{smth_ckpt_hash}' does not match best.pth '{expected_smth_ckpt}'")

    # Precision validation across ALL rows
    if eval_precision is not None:
        ctrl_precs = {r.get('eval_precision') for r in rows_ctrl}
        if ctrl_precs != {eval_precision}:
            raise ValueError(f"Control predictions eval_precision must be '{eval_precision}' across all rows, got {ctrl_precs}")
        smth_precs = {r.get('eval_precision') for r in rows_smth}
        if smth_precs != {eval_precision}:
            raise ValueError(f"Smoothing predictions eval_precision must be '{eval_precision}' across all rows, got {smth_precs}")

    # Load and validate frozen thresholds against the respective checkpoint hash
    tau_ctrl, s_ctrl = load_validated_threshold(
        str(t_ctrl_file),
        expected_checkpoint_hash=ctrl_ckpt_hash,
        eval_precision=eval_precision
    )
    tau_smth, s_smth = load_validated_threshold(
        str(t_smth_file),
        expected_checkpoint_hash=smth_ckpt_hash,
        eval_precision=eval_precision
    )

    # Check sample-by-sample alignment: ID, label, and path
    for i, (rc, rs) in enumerate(zip(rows_ctrl, rows_smth)):
        if rc.get('sample_id') != rs.get('sample_id'):
            raise ValueError(f"Sample ID alignment mismatch at index {i}: '{rc.get('sample_id')}' != '{rs.get('sample_id')}'")
        if int(rc['label']) != int(rs['label']):
            raise ValueError(f"Ground-truth label mismatch at index {i}: {rc['label']} != {rs['label']}")
        if rc.get('path') != rs.get('path'):
            raise ValueError(f"Sample path mismatch at index {i}: '{rc.get('path')}' != '{rs.get('path')}'")

    y = np.array([int(r['label']) for r in rows_ctrl])
    p_ctrl = np.array([float(r['probability']) for r in rows_ctrl])
    z_ctrl = np.array([float(r['logit']) for r in rows_ctrl]) if all('logit' in r and r.get('logit') is not None for r in rows_ctrl) else None
    p_smth = np.array([float(r['probability']) for r in rows_smth])
    z_smth = np.array([float(r['logit']) for r in rows_smth]) if all('logit' in r and r.get('logit') is not None for r in rows_smth) else None

    # Compute metrics
    m_ctrl_05 = binary_metrics(y, p_ctrl, threshold=0.5, logits=z_ctrl, score_space='probability')
    m_smth_05 = binary_metrics(y, p_smth, threshold=0.5, logits=z_smth, score_space='probability')

    m_ctrl_frz = binary_metrics(y, p_ctrl, threshold=tau_ctrl, logits=z_ctrl, score_space=s_ctrl)
    m_smth_frz = binary_metrics(y, p_smth, threshold=tau_smth, logits=z_smth, score_space=s_smth)

    groups = [r.get('group_id', '') for r in rows_ctrl]
    paired_diff = group_intervals(
        y, p_smth, groups, threshold=tau_smth, repeats=bootstrap_repeats, seed=seed,
        other=p_ctrl, other_threshold=tau_ctrl,
        logits=z_smth, other_logits=z_ctrl,
        score_space=s_smth, other_score_space=s_ctrl
    )

    comparison = {
        'schema_version': '2.0',
        'samples': len(y),
        'file_digests': {
            'control_predictions_sha256': ctrl_pred_sha,
            'smooth_predictions_sha256': smth_pred_sha,
            'control_threshold_sha256': ctrl_thresh_sha,
            'smooth_threshold_sha256': smth_thresh_sha,
            'control_checkpoint_sha256': ctrl_ckpt_hash,
            'smooth_checkpoint_sha256': smth_ckpt_hash,
        },
        'control': {
            'name': control_name,
            'checkpoint_sha256': ctrl_ckpt_hash,
            'frozen_threshold': tau_ctrl,
            'score_space': s_ctrl,
            'fixed_05': m_ctrl_05,
            'frozen_calibrated': m_ctrl_frz,
        },
        'smoothing': {
            'name': smooth_name,
            'checkpoint_sha256': smth_ckpt_hash,
            'frozen_threshold': tau_smth,
            'score_space': s_smth,
            'fixed_05': m_smth_05,
            'frozen_calibrated': m_smth_frz,
        },
        'paired_difference_smoothing_minus_control': paired_diff,
    }

    def sat_pct(m):
        return (m['saturation_counts']['total_saturated'] / m['samples']) * 100

    md_table = fr"""# A/B Evaluation Summary (External Development Dataset: {len(y):,} images)

| Metric | Arm A: Control ({control_name}) | Arm B: Smoothing ({smooth_name}) | Absolute Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Logit ROC AUC** | {m_ctrl_frz['roc_auc_logit']:.5f} | {m_smth_frz['roc_auc_logit']:.5f} | {m_smth_frz['roc_auc_logit'] - m_ctrl_frz['roc_auc_logit']:+.5f} |
| **Probability ROC AUC (Diagnostic)** | {m_ctrl_frz['roc_auc_probability']:.5f} | {m_smth_frz['roc_auc_probability']:.5f} | {m_smth_frz['roc_auc_probability'] - m_ctrl_frz['roc_auc_probability']:+.5f} |
| **Continuous Logit EER** | {m_ctrl_frz['interpolated_eer']:.4f} | {m_smth_frz['interpolated_eer']:.4f} | {m_smth_frz['interpolated_eer'] - m_ctrl_frz['interpolated_eer']:+.4f} |
| **Total Saturated ($p \le 10^{{-4}}$ or $\ge 1-10^{{-4}}$)** | {sat_pct(m_ctrl_frz):.2f}% | {sat_pct(m_smth_frz):.2f}% | {sat_pct(m_smth_frz) - sat_pct(m_ctrl_frz):+.2f}% |
| **Frozen Logit Threshold ($\tau_z$)** | `{tau_ctrl:.4f}` | `{tau_smth:.4f}` | — |
| **Accuracy @ Frozen Logit $\tau_z$** | {m_ctrl_frz['accuracy']*100:.2f}% | {m_smth_frz['accuracy']*100:.2f}% | {(m_smth_frz['accuracy'] - m_ctrl_frz['accuracy'])*100:+.2f}% |
| **Balanced Accuracy @ Frozen $\tau_z$** | {m_ctrl_frz['balanced_accuracy']*100:.2f}% | {m_smth_frz['balanced_accuracy']*100:.2f}% | {(m_smth_frz['balanced_accuracy'] - m_ctrl_frz['balanced_accuracy'])*100:+.2f}% |
| **Real Recall @ Frozen $\tau_z$** | {m_ctrl_frz['real_recall']*100:.2f}% | {m_smth_frz['real_recall']*100:.2f}% | {(m_smth_frz['real_recall'] - m_ctrl_frz['real_recall'])*100:+.2f}% |
| **Fake Recall @ Frozen $\tau_z$** | {m_ctrl_frz['fake_recall']*100:.2f}% | {m_smth_frz['fake_recall']*100:.2f}% | {(m_smth_frz['fake_recall'] - m_ctrl_frz['fake_recall'])*100:+.2f}% |
| **Macro F1 @ Frozen $\tau_z$** | {m_ctrl_frz['macro_f1']:.4f} | {m_smth_frz['macro_f1']:.4f} | {m_smth_frz['macro_f1'] - m_ctrl_frz['macro_f1']:+.4f} |
| **Accuracy @ Default 0.5** | {m_ctrl_05['accuracy']*100:.2f}% | {m_smth_05['accuracy']*100:.2f}% | {(m_smth_05['accuracy'] - m_ctrl_05['accuracy'])*100:+.2f}% |
| **Real / Fake Recall @ Default 0.5** | {m_ctrl_05['real_recall']*100:.1f}% / {m_ctrl_05['fake_recall']*100:.1f}% | {m_smth_05['real_recall']*100:.1f}% / {m_smth_05['fake_recall']*100:.1f}% | — |
"""

    if output_json_path is not None:
        p_out = Path(output_json_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        p_out.write_text(json.dumps(comparison, indent=2), encoding='utf-8')

    if output_md_path is not None:
        p_md = Path(output_md_path)
        p_md.parent.mkdir(parents=True, exist_ok=True)
        p_md.write_text(md_table, encoding='utf-8')

    comparison['markdown_summary'] = md_table
    return comparison
