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


def binary_metrics(labels, probabilities, threshold=0.5, logits=None):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if not len(y) or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected binary labels with canonical mapping: real=0, fake=1')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 <= threshold <= 1:
        raise ValueError('Probabilities and threshold must be finite and in [0,1]')
    if logits is not None:
        logits = np.asarray(logits, dtype=float)
        if logits.shape != p.shape:
            raise ValueError('Logits shape must match probabilities')
    prediction = p >= threshold
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

    result = dict(
        schema_version='2.0',
        samples=len(y),
        real=int(tn + fp),
        fake=int(tp + fn),
        real_recall=real_recall,
        fake_recall=fake_recall,
        accuracy=float((tn + tp) / len(y)),
        balanced_accuracy=float(rec.mean()) if both else None,
        roc_auc=float(roc_auc_score(y, p)) if both else None,
        average_precision=float(average_precision_score(y, p)) if both else None,
        fpr=float(fp / (tn + fp)) if tn + fp else None,
        fnr=float(fn / (tp + fn)) if tp + fn else None,
        threshold=float(threshold),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        macro_f1=float(f1.mean()) if both else None,
        per_class={
            name: dict(precision=float(prec[i]), recall=float(rec[i]), f1=float(f1[i]), support=int(counts[i]))
            for i, name in enumerate(('real', 'fake'))
        },
        score_quantiles=score_quantiles,
        logit_quantiles=logit_quantiles,
        saturation_counts=saturation_counts,
        eer=None,
        fpr_at_tpr95=None,
        tpr_at_fpr01=None,
        status='computed' if both else 'single_class_evaluated_ranking_metrics_undefined',
    )
    if not both:
        result['note'] = 'Single-class evaluation: ROC AUC, balanced accuracy, and EER are undefined and marked None.'
    if both:
        fpr, tpr, _ = roc_curve(y, p)
        gap = fpr - (1 - tpr)
        j = int(np.searchsorted(gap, 0))
        if j == 0:
            eer = fpr[0]
        else:
            weight = -gap[j - 1] / (gap[j] - gap[j - 1])
            eer = fpr[j - 1] + weight * (fpr[j] - fpr[j - 1])
        tpr95_indices = np.flatnonzero(tpr >= .95)
        fpr_at_tpr95 = float(fpr[tpr95_indices].min()) if len(tpr95_indices) else None
        fpr01_indices = np.flatnonzero(fpr <= .01)
        tpr_at_fpr01 = float(tpr[fpr01_indices].max()) if len(fpr01_indices) else None
        result.update(eer=float(eer), fpr_at_tpr95=fpr_at_tpr95, tpr_at_fpr01=tpr_at_fpr01)
    return result


def group_intervals(labels, probabilities, groups, threshold=.5, repeats=200, seed=42, other=None, other_threshold=None):
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

    for _ in range(repeats):
        selected = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        first = binary_metrics(y[selected], p[selected], threshold)
        second = binary_metrics(y[selected], np.asarray(other)[selected], t2) if other is not None else None
        for key in metrics:
            if first[key] is not None and (second is None or second[key] is not None):
                metrics[key].append(first[key] - second[key] if second else first[key])

    # Compute actual aligned full-sample difference on unresampled data
    first_full = binary_metrics(y, p, threshold)
    second_full = binary_metrics(y, np.asarray(other), t2) if other is not None else None
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


def summarize(records, probabilities, threshold=.5, repeats=200, seed=42, logits=None):
    y, p = np.array([r['label'] for r in records]), np.asarray(probabilities)
    if logits is None and len(records) > 0 and all('logit' in r and r.get('logit') is not None for r in records):
        logits = np.array([float(r['logit']) for r in records])
    report = {
        'schema_version': '2.0',
        'overall': binary_metrics(y, p, threshold, logits=logits),
        'uncertainty': group_intervals(y, p, [r.get('group_id', '') for r in records], threshold, repeats, seed),
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
                res = binary_metrics(y[ix], p[ix], threshold)
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
        report['video_level'] = binary_metrics([key[2] for key in videos], [float(p[ix].mean()) for ix in videos.values()], threshold)
        report['video_aggregation'] = 'mean frame probability by dataset_source, source_video_id, label; known IDs only'
    report['fairness_note'] = 'Only supplied annotations are analyzed; missing annotations are not inferred. Small groups require cautious interpretation.'
    return report


def calibrate(prediction_path, output_path, eval_precision=None, allow_external_dev=False,
              selection_predictions_path=None, selection_manifest=None, train_manifest=None):
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
    fpr, tpr, thresholds = roc_curve(y, p)
    eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= 1) & (thresholds >= 0))
    selected = eligible[np.argmax((tpr - fpr)[eligible])]
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
        threshold=float(thresholds[selected]),
        criterion='development Youden J',
        checkpoint_sha256=next(iter(hashes)),
        prediction_sha256=sha256(prediction_path),
        source_splits=sorted(splits),
        eval_precision=str(prec),
        independence_status=independence_status,
        reference_evidence=reference_evidence,
    )
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
