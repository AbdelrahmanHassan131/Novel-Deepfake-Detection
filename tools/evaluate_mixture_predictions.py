"""Per-stratum development evaluation and comparative reporting for RGB mixture pilot.

Implements Prompt 3 from RGB_NEXT_CONTROLLED_PILOT_HANDOFF.md & MIXTURE_PILOT_BLOCKING_FIXES.md:
- Requires valid shared-dev manifest and verified cryptographic gate (fail-closed).
- Enforces single homogeneous checkpoint hash per arm (rejects mixed or unknown checkpoint hashes).
- Validates sample_id, label, image sha256, group_id, and full paths against shared-dev manifest.
- Rejects unknown filename patterns immediately.
- Calibrates dev threshold using unclipped development Youden's J statistic with deterministic tie-breaking.
  Explicitly labels metrics as development fit, not independent generalization.
- Reports per-stratum denominator, errors, real/fake recall, probability/logit summaries.
- Reports both-class numeric/video AUC where defined; keeps AUC undefined for one-class strata.
- Reports overall balanced accuracy and minimum of the 4 applicable stratum recalls as diagnostics.
- Compares Arm A vs Arm B on exactly identical dev IDs:
  - Descriptive paired contingency matrix (both correct, A right / B wrong, A wrong / B right, both wrong)
  - Per-stratum net error changes (B - A)
  - Omits invalid asymptotic p-values that violate frame independence.
- Enforces empty destination directory and atomic report writes.
"""

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import sklearn.metrics

from data.manifest import verify_manifest_gate

FAMILY_RE = re.compile(r"^(vid_[0-9a-fA-F]{64})_face_\d+_\d+\.[a-zA-Z0-9]+$")
NUMERIC_RE = re.compile(r"^\d+\.[a-zA-Z0-9]+$")


def classify_stratum(path: str, label: int) -> str:
    fn = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    if NUMERIC_RE.match(fn):
        return "numeric_real" if label == 0 else "numeric_fake"
    elif FAMILY_RE.match(fn):
        return "video_real" if label == 0 else "video_fake"
    return "unknown"


def compute_roc_auc(labels: np.ndarray, probs: np.ndarray):
    """Compute tie-aware ROC AUC or return 'undefined_single_class'."""
    if len(labels) == 0:
        return "undefined_single_class"
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise ValueError("Non-finite probability values encountered in AUC computation.")
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return "undefined_single_class"
    return float(sklearn.metrics.roc_auc_score(labels, probs))


def summarize_distribution(vals: np.ndarray) -> Dict[str, Any]:
    if len(vals) == 0:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "p25": float(np.percentile(vals, 25)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "max": float(np.max(vals)),
    }


def select_best_threshold(labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Select threshold on dev set maximizing balanced accuracy (Youden's J) without artificial clipping."""
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, probs)
    eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= 1.0) & (thresholds >= 0.0))
    if len(eligible) == 0:
        eligible = np.flatnonzero(np.isfinite(thresholds))
    selected = eligible[np.argmax((tpr - fpr)[eligible])]
    chosen_threshold = float(thresholds[selected])

    return {
        "threshold": chosen_threshold,
        "criterion": "development Youden J",
        "source_splits": ["dev"],
        "independence_status": "development_fit_only_not_independent",
        "note": "Optimized on development partition; diagnostic fit only, not independent performance."
    }


def read_and_validate_predictions(
    pred_path: Path,
    manifest_rows: List[Dict[str, Any]],
    allowed_path_prefix: Optional[Tuple[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Read prediction CSV and strictly validate schema, integrity, and manifest linkage.

    Fails closed if:
    - File does not exist or is empty
    - Any sample_id is missing, duplicate, or not in manifest
    - Label does not match manifest
    - Image sha256 or group_id mismatches manifest
    - Checkpoint hash is missing, 'unknown', or mixed within the arm
    - Total prediction count does not match manifest
    """
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    with open(pred_path, mode="r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty predictions file: {pred_path}")

    man_map = {m["sample_id"]: m for m in manifest_rows}
    if len(rows) != len(man_map):
        raise ValueError(
            f"Prediction count ({len(rows)}) does not match dev manifest count ({len(man_map)}) in {pred_path}."
        )

    seen_ids = set()
    ckpt_hashes = set()
    parsed_rows = []

    for i, r in enumerate(rows):
        sid = r.get("sample_id", "")
        if not sid:
            raise ValueError(f"Row {i} in {pred_path} missing 'sample_id'.")
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample_id '{sid}' in {pred_path} at row {i}.")
        seen_ids.add(sid)

        if sid not in man_map:
            raise ValueError(f"Prediction sample_id '{sid}' at row {i} not present in shared-dev manifest.")
        mr = man_map[sid]

        lbl_str = str(r.get("label", "")).strip()
        if lbl_str not in ("0", "1"):
            raise ValueError(f"Invalid label '{lbl_str}' in {pred_path} at row {i}.")
        lbl = int(lbl_str)
        if lbl != int(mr["label"]):
            raise ValueError(
                f"Label mismatch for sample '{sid}': pred label {lbl} vs manifest label {mr['label']}."
            )

        try:
            prob = float(r["probability"])
            if math.isnan(prob) or math.isinf(prob) or prob < 0.0 or prob > 1.0:
                raise ValueError
        except Exception:
            raise ValueError(f"Invalid probability '{r.get('probability')}' in {pred_path} at row {i}.")

        logit = None
        if "logit" in r and r["logit"] not in ("", None):
            try:
                logit = float(r["logit"])
                if math.isnan(logit) or math.isinf(logit):
                    raise ValueError
            except Exception:
                raise ValueError(f"Invalid logit '{r.get('logit')}' in {pred_path} at row {i}.")

        ckpt_hash = str(r.get("checkpoint_sha256", "")).strip()
        if not ckpt_hash or ckpt_hash.lower() in ("unknown", "none", "nan") or not re.fullmatch(r"[0-9a-fA-F]{64}", ckpt_hash):
            raise ValueError(
                f"Missing or invalid checkpoint_sha256 at row {i} in {pred_path}: '{ckpt_hash}'. "
                f"Predictions must be cryptographically bound to a single verified checkpoint."
            )
        ckpt_hashes.add(ckpt_hash)

        # Cross-reference content hash and group if present in prediction
        if "sha256" in r and r["sha256"] and r["sha256"] != mr["sha256"]:
            raise ValueError(
                f"Content sha256 mismatch for sample '{sid}': pred '{r['sha256']}' vs manifest '{mr['sha256']}'."
            )
        if "group_id" in r and r["group_id"] and r["group_id"] != mr["group_id"]:
            raise ValueError(
                f"group_id mismatch for sample '{sid}': pred '{r['group_id']}' vs manifest '{mr['group_id']}'."
            )

        # Path validation: canonical path or allowed relocation prefix
        pred_p_norm = str(r.get("path", "")).replace("\\", "/")
        man_p_norm = str(mr["path"]).replace("\\", "/")
        if pred_p_norm and pred_p_norm != man_p_norm:
            if allowed_path_prefix:
                old_pre, new_pre = allowed_path_prefix
                expected_p = man_p_norm.replace(old_pre, new_pre)
                if pred_p_norm != expected_p:
                    raise ValueError(
                        f"Path mismatch for sample '{sid}': '{pred_p_norm}' does not match expected relocated '{expected_p}'."
                    )
            else:
                # Disallow arbitrary basename fallback
                raise ValueError(
                    f"Path mismatch for sample '{sid}': pred '{pred_p_norm}' vs manifest '{man_p_norm}'. "
                    f"Silent fallback and basename joins are prohibited."
                )

        parsed_rows.append({
            "sample_id": sid,
            "path": mr["path"],
            "label": lbl,
            "probability": prob,
            "logit": logit,
            "group_id": mr["group_id"],
            "sha256": mr["sha256"],
            "checkpoint_sha256": ckpt_hash,
        })

    if len(ckpt_hashes) > 1:
        raise ValueError(
            f"Mixed checkpoint hashes detected in {pred_path}: {sorted(list(ckpt_hashes))}. "
            f"All predictions within an arm must originate from the same verified checkpoint."
        )

    homogeneous_ckpt = next(iter(ckpt_hashes))
    return parsed_rows, homogeneous_ckpt


def evaluate_cohort(
    records: List[Dict[str, Any]],
    threshold: float,
    threshold_label: str,
) -> Dict[str, Any]:
    """Compute overall, subcollection, and per-stratum metrics at a specified threshold."""
    labels = np.array([r["label"] for r in records], dtype=int)
    probs = np.array([r["probability"] for r in records], dtype=float)
    preds = (probs >= threshold).astype(int)

    has_logits = all(r.get("logit") is not None for r in records)
    logits = np.array([r["logit"] for r in records], dtype=float) if has_logits else None

    # Overall metrics
    overall_correct = int(np.sum(preds == labels))
    overall_errors = len(labels) - overall_correct
    overall_acc = float(np.mean(preds == labels))
    overall_auc = compute_roc_auc(labels, probs)

    real_mask = (labels == 0)
    fake_mask = (labels == 1)
    real_recall = float(np.mean(preds[real_mask] == 0)) if np.any(real_mask) else None
    fake_recall = float(np.mean(preds[fake_mask] == 1)) if np.any(fake_mask) else None
    balanced_acc = float(0.5 * (real_recall + fake_recall)) if (real_recall is not None and fake_recall is not None) else None

    # Per-stratum evaluations
    strata_records = defaultdict(list)
    for r in records:
        st = classify_stratum(r["path"], r["label"])
        if st == "unknown":
            raise ValueError(
                f"Unknown filename pattern encountered for sample '{r['sample_id']}': '{r['path']}'. "
                f"Unknown patterns must be rejected, not silently assigned or omitted."
            )
        strata_records[st].append(r)

    strata_results = {}
    stratum_recalls = []

    for stratum_name in ("numeric_real", "video_real", "numeric_fake", "video_fake"):
        items = strata_records.get(stratum_name, [])
        if not items:
            strata_results[stratum_name] = {"denominator": 0, "errors": 0, "recall": None}
            continue

        st_labels = np.array([x["label"] for x in items], dtype=int)
        st_probs = np.array([x["probability"] for x in items], dtype=float)
        st_preds = (st_probs >= threshold).astype(int)
        st_errors = int(np.sum(st_preds != st_labels))
        st_denom = len(items)
        st_recall = float(1.0 - (st_errors / st_denom))
        stratum_recalls.append(st_recall)

        st_logits = np.array([x["logit"] for x in items], dtype=float) if has_logits else None

        strata_results[stratum_name] = {
            "denominator": st_denom,
            "errors": st_errors,
            "recall": st_recall,
            "roc_auc": "undefined_single_class",
            "probabilities": summarize_distribution(st_probs),
            "logits": summarize_distribution(st_logits) if has_logits else None,
        }

    # Combined subcollections: numeric_both and video_both
    subcollections = {}
    for sub_name, (r_name, f_name) in [("numeric_both", ("numeric_real", "numeric_fake")), ("video_both", ("video_real", "video_fake"))]:
        sub_items = strata_records.get(r_name, []) + strata_records.get(f_name, [])
        if not sub_items:
            continue
        sub_y = np.array([x["label"] for x in sub_items], dtype=int)
        sub_p = np.array([x["probability"] for x in sub_items], dtype=float)
        sub_pred = (sub_p >= threshold).astype(int)
        sub_rmask = (sub_y == 0)
        sub_fmask = (sub_y == 1)
        sub_r_rec = float(np.mean(sub_pred[sub_rmask] == 0)) if np.any(sub_rmask) else None
        sub_f_rec = float(np.mean(sub_pred[sub_fmask] == 1)) if np.any(sub_fmask) else None
        sub_bacc = float(0.5 * (sub_r_rec + sub_f_rec)) if (sub_r_rec is not None and sub_f_rec is not None) else None
        sub_auc = compute_roc_auc(sub_y, sub_p)

        subcollections[sub_name] = {
            "denominator": len(sub_items),
            "errors": int(np.sum(sub_pred != sub_y)),
            "accuracy": float(np.mean(sub_pred == sub_y)),
            "balanced_accuracy": sub_bacc,
            "real_recall": sub_r_rec,
            "fake_recall": sub_f_rec,
            "roc_auc": sub_auc,
            "probabilities": summarize_distribution(sub_p),
        }

    min_stratum_recall = float(min(stratum_recalls)) if stratum_recalls else None

    return {
        "threshold": float(threshold),
        "threshold_label": threshold_label,
        "overall": {
            "denominator": len(records),
            "errors": overall_errors,
            "accuracy": overall_acc,
            "balanced_accuracy": balanced_acc,
            "real_recall": real_recall,
            "fake_recall": fake_recall,
            "roc_auc": overall_auc,
            "min_stratum_recall": min_stratum_recall,
            "probabilities": summarize_distribution(probs),
            "logits": summarize_distribution(logits) if has_logits else None,
        },
        "strata": strata_results,
        "subcollections": subcollections,
    }


def compare_arms(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    thresh_a: float,
    thresh_b: float,
    thresh_label: str,
) -> Dict[str, Any]:
    """Descriptive paired comparison of Arm A vs Arm B on exactly identical sample IDs."""
    id_map_b = {r["sample_id"]: r for r in records_b}
    if set(r["sample_id"] for r in records_a) != set(id_map_b.keys()):
        raise ValueError("Cannot perform paired comparison: sample ID sets between Arm A and B differ!")

    n_both_correct = 0
    n_a_correct_b_wrong = 0
    n_a_wrong_b_correct = 0
    n_both_wrong = 0

    strata_paired = defaultdict(lambda: {"a_errors": 0, "b_errors": 0, "a_corr_b_wrong": 0, "a_wrong_b_corr": 0, "net_b_minus_a": 0})

    for ra in records_a:
        sid = ra["sample_id"]
        rb = id_map_b[sid]
        lbl = ra["label"]

        if ra["label"] != rb["label"]:
            raise ValueError(f"Label mismatch for sample '{sid}' between Arm A ({ra['label']}) and Arm B ({rb['label']})!")

        pred_a = int(ra["probability"] >= thresh_a)
        pred_b = int(rb["probability"] >= thresh_b)

        a_corr = (pred_a == lbl)
        b_corr = (pred_b == lbl)

        if a_corr and b_corr:
            n_both_correct += 1
        elif a_corr and not b_corr:
            n_a_correct_b_wrong += 1
        elif not a_corr and b_corr:
            n_a_wrong_b_correct += 1
        else:
            n_both_wrong += 1

        st = classify_stratum(ra["path"], lbl)
        strata_paired[st]["a_errors"] += int(not a_corr)
        strata_paired[st]["b_errors"] += int(not b_corr)
        if a_corr and not b_corr:
            strata_paired[st]["a_corr_b_wrong"] += 1
        elif not a_corr and b_corr:
            strata_paired[st]["a_wrong_b_corr"] += 1

    for st in strata_paired:
        strata_paired[st]["net_b_minus_a"] = strata_paired[st]["b_errors"] - strata_paired[st]["a_errors"]

    return {
        "threshold_label": thresh_label,
        "threshold_a": float(thresh_a),
        "threshold_b": float(thresh_b),
        "contingency_matrix": {
            "both_correct": n_both_correct,
            "a_correct_b_wrong": n_a_correct_b_wrong,
            "a_wrong_b_correct": n_a_wrong_b_correct,
            "both_wrong": n_both_wrong,
            "net_b_minus_a_errors": n_a_correct_b_wrong - n_a_wrong_b_correct,
        },
        "per_stratum_shifts": dict(strata_paired),
        "notes": "Descriptive paired counts only; asymptotic p-values omitted due to video-frame clustering."
    }


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """Format evaluation results into a clear Markdown comparison table with dynamic denominators."""
    lines = [
        "# Controlled RGB Mixture Pilot: Shared Development Evaluation Report",
        "",
        "> **Note:** All metrics evaluated on the frozen shared development partition.",
        "> Stratum recalls and balanced accuracies are diagnostic measurements of engineering data representation,",
        "> not claims of out-of-distribution generalization or certified source performance.",
        "> Metrics at dev-selected thresholds represent development fit only.",
        "",
    ]

    has_b = "arm_b" in report_data
    arm_names = ["arm_a", "arm_b"] if has_b else ["arm_a"]

    # Table 1: Overall metrics
    lines.extend([
        "## Overall Development Performance",
        "",
        "| Arm | Checkpoint Digest | Threshold Mode | Threshold Value | Balanced Accuracy | ROC AUC | Real Recall | Fake Recall | Min Stratum Recall |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])

    for arm in arm_names:
        c_hash = report_data[arm]["checkpoint_sha256"][:12]
        for mode in ("fixed_0_5", "dev_selected"):
            res = report_data[arm][mode]
            ov = res["overall"]
            auc_str = f"{ov['roc_auc']:.4f}" if isinstance(ov['roc_auc'], (int, float)) else str(ov['roc_auc'])
            bacc_str = f"{ov['balanced_accuracy'] * 100:.2f}%" if ov['balanced_accuracy'] is not None else "N/A"
            rrec_str = f"{ov['real_recall'] * 100:.2f}%" if ov['real_recall'] is not None else "N/A"
            frec_str = f"{ov['fake_recall'] * 100:.2f}%" if ov['fake_recall'] is not None else "N/A"
            min_str = f"{ov['min_stratum_recall'] * 100:.2f}%" if ov['min_stratum_recall'] is not None else "N/A"
            mode_lbl = res["threshold_label"] + (" (dev fit)" if mode == "dev_selected" else "")
            lines.append(
                f"| {arm.upper()} | `{c_hash}` | {mode_lbl} | {res['threshold']:.4f} | "
                f"{bacc_str} | {auc_str} | {rrec_str} | {frec_str} | {min_str} |"
            )

    # Table 2: Per-Stratum Recalls and Errors
    lines.extend([
        "",
        "## Per-Stratum Diagnostic Recalls (Fixed 0.5)",
        "",
        "| Stratum | Denominator | Arm A Recall | Arm A Errors | " + ("Arm B Recall | Arm B Errors | Net Error Shift (B - A) |" if has_b else ""),
        "|---|---:|---:|---:|---:|---:|---:|" if has_b else "|---|---:|---:|---:|",
    ])

    for st in ("numeric_real", "video_real", "numeric_fake", "video_fake"):
        st_a = report_data["arm_a"]["fixed_0_5"]["strata"][st]
        denom = st_a["denominator"]
        rec_a = f"{st_a['recall'] * 100:.2f}%" if st_a['recall'] is not None else "N/A"
        err_a = st_a["errors"]
        if has_b:
            st_b = report_data["arm_b"]["fixed_0_5"]["strata"][st]
            rec_b = f"{st_b['recall'] * 100:.2f}%" if st_b['recall'] is not None else "N/A"
            err_b = st_b["errors"]
            net = err_b - err_a
            net_str = f"{'+' if net > 0 else ''}{net}"
            lines.append(f"| `{st}` | {denom} | {rec_a} | {err_a} | {rec_b} | {err_b} | {net_str} |")
        else:
            lines.append(f"| `{st}` | {denom} | {rec_a} | {err_a} |")

    # Table 3: Paired contingency analysis if both arms are present
    if has_b and "paired_comparison" in report_data:
        pair_05 = report_data["paired_comparison"]["fixed_0_5"]
        cm = pair_05["contingency_matrix"]
        total = cm["both_correct"] + cm["a_correct_b_wrong"] + cm["a_wrong_b_correct"] + cm["both_wrong"]
        lines.extend([
            "",
            "## Descriptive Paired Contingency Analysis (Fixed 0.5)",
            "",
            f"- **Both models correct:** {cm['both_correct']} / {total}",
            f"- **Arm A correct, Arm B incorrect:** {cm['a_correct_b_wrong']}",
            f"- **Arm A incorrect, Arm B correct:** {cm['a_wrong_b_correct']}",
            f"- **Both models incorrect:** {cm['both_wrong']}",
            f"- **Net error shift (B error count minus A error count):** {cm['net_b_minus_a_errors']}",
            "",
            "> Note: Video frames share identity clusters; paired error changes are reported descriptively without assuming independent sampling.",
        ])

    return "\n".join(lines)


def run_evaluation(
    pred_a_path: Path,
    manifest_path: Path,
    output_dir: Path,
    pred_b_path: Optional[Path] = None,
    threshold_a: Optional[float] = None,
    threshold_b: Optional[float] = None,
    allowed_path_prefix: Optional[Tuple[str, str]] = None,
) -> Dict[str, Any]:
    out_p = Path(output_dir).resolve()
    if out_p.exists() and any(out_p.iterdir()):
        raise ValueError(
            f"Use an empty NEW output directory; old evidence is never overwritten. "
            f"Destination '{out_p}' already exists and contains files."
        )

    # 1. Require and validate shared-dev manifest and verified gate
    man_p = Path(manifest_path).resolve()
    if not man_p.is_file():
        raise FileNotFoundError(f"Shared development manifest not found: {man_p}")

    # Verify manifest gate sidecar
    verify_manifest_gate(str(man_p), require_hashes=True)

    with open(man_p, mode="r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        man_rows = list(reader)

    if not man_rows:
        raise ValueError(f"Shared development manifest '{man_p}' contains 0 rows.")

    for mr in man_rows:
        if mr.get("split") not in ("dev", None, ""):
            raise ValueError(f"Shared dev manifest must contain only split='dev', got '{mr.get('split')}'.")
        if not mr.get("sample_id") or not mr.get("sha256") or not mr.get("group_id"):
            raise ValueError(f"Shared dev manifest missing required fields at sample '{mr.get('sample_id')}'.")

    # 2. Load and validate Arm A predictions
    records_a, ckpt_a = read_and_validate_predictions(
        pred_path=Path(pred_a_path),
        manifest_rows=man_rows,
        allowed_path_prefix=allowed_path_prefix,
    )
    labels_a = np.array([r["label"] for r in records_a], dtype=int)
    probs_a = np.array([r["probability"] for r in records_a], dtype=float)

    dev_cal_a = select_best_threshold(labels_a, probs_a)
    dev_cal_a["checkpoint_sha256"] = ckpt_a
    dev_cal_a["prediction_file"] = str(pred_a_path)
    dev_cal_a["manifest_path"] = str(man_p)

    dev_thresh_a = dev_cal_a["threshold"]
    fix_thresh_a = threshold_a if threshold_a is not None else 0.5

    arm_a_fixed = evaluate_cohort(records_a, fix_thresh_a, "fixed_0_5")
    arm_a_dev = evaluate_cohort(records_a, dev_thresh_a, "dev_selected")

    report: Dict[str, Any] = {
        "shared_dev_manifest": {
            "path": str(man_p),
            "samples": len(man_rows),
        },
        "arm_a": {
            "checkpoint_sha256": ckpt_a,
            "prediction_file": str(pred_a_path),
            "fixed_0_5": arm_a_fixed,
            "dev_selected": arm_a_dev,
            "dev_calibration_artifact": dev_cal_a,
        }
    }

    # 3. Load and validate Arm B if provided
    if pred_b_path:
        records_b, ckpt_b = read_and_validate_predictions(
            pred_path=Path(pred_b_path),
            manifest_rows=man_rows,
            allowed_path_prefix=allowed_path_prefix,
        )
        labels_b = np.array([r["label"] for r in records_b], dtype=int)
        probs_b = np.array([r["probability"] for r in records_b], dtype=float)

        dev_cal_b = select_best_threshold(labels_b, probs_b)
        dev_cal_b["checkpoint_sha256"] = ckpt_b
        dev_cal_b["prediction_file"] = str(pred_b_path)
        dev_cal_b["manifest_path"] = str(man_p)

        dev_thresh_b = dev_cal_b["threshold"]
        fix_thresh_b = threshold_b if threshold_b is not None else 0.5

        arm_b_fixed = evaluate_cohort(records_b, fix_thresh_b, "fixed_0_5")
        arm_b_dev = evaluate_cohort(records_b, dev_thresh_b, "dev_selected")

        report["arm_b"] = {
            "checkpoint_sha256": ckpt_b,
            "prediction_file": str(pred_b_path),
            "fixed_0_5": arm_b_fixed,
            "dev_selected": arm_b_dev,
            "dev_calibration_artifact": dev_cal_b,
        }

        # Paired descriptive comparisons
        report["paired_comparison"] = {
            "fixed_0_5": compare_arms(records_a, records_b, fix_thresh_a, fix_thresh_b, "fixed_0_5"),
            "dev_selected": compare_arms(records_a, records_b, dev_thresh_a, dev_thresh_b, "dev_selected"),
        }

    # 4. Save JSON and MD atomically
    out_p.mkdir(parents=True, exist_ok=True)
    json_path = out_p / "mixture_evaluation_report.json"
    json_tmp = out_p / "mixture_evaluation_report.json.tmp"
    md_path = out_p / "mixture_evaluation_report.md"
    md_tmp = out_p / "mixture_evaluation_report.md.tmp"

    with open(json_tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_content = generate_markdown_report(report)
    with open(md_tmp, "w", encoding="utf-8") as f:
        f.write(md_content)

    if json_path.exists():
        json_path.unlink()
    json_tmp.rename(json_path)

    if md_path.exists():
        md_path.unlink()
    md_tmp.rename(md_path)

    print(f"Report saved to:\n  JSON: {json_path}\n  MD:   {md_path}")
    return report


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate RGB mixture predictions on shared development partition.")
    parser.add_argument("--pred_a", type=Path, required=True, help="Path to Arm A predictions CSV")
    parser.add_argument("--shared_dev_manifest", type=Path, required=True, help="Path to shared dev manifest CSV")
    parser.add_argument("--pred_b", type=Path, default=None, help="Path to Arm B predictions CSV (optional)")
    parser.add_argument("--output_dir", type=Path, default=Path("output/controlled_mixture/evaluation_v2"))
    parser.add_argument("--threshold_a", type=float, default=None)
    parser.add_argument("--threshold_b", type=float, default=None)
    return parser.parse_args(args)


def main():
    args = parse_args()
    run_evaluation(
        pred_a_path=args.pred_a,
        manifest_path=args.shared_dev_manifest,
        output_dir=args.output_dir,
        pred_b_path=args.pred_b,
        threshold_a=args.threshold_a,
        threshold_b=args.threshold_b,
    )


if __name__ == "__main__":
    main()
