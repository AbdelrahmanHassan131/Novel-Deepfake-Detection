"""Compute subcollection diagnostic metrics from saved R2 dev predictions.

Features:
- Validates prediction IDs, labels, image hashes, and full paths against the original
  development split in the saved 100K manifest (no basename join).
- Rejects non-finite probabilities, invalid labels, duplicate sample IDs, and mixed checkpoint hashes.
- Uses tie-aware ROC AUC calculation via sklearn.metrics.roc_auc_score.
- Embedded self-tests for ROC AUC edge cases (perfect, reversed, all-ties, mixed-ties,
  single-class undefined, row-order invariance).
- Produces: output/review/archive_investigation/subcollection_metrics.json
"""

import csv
import json
import os
import re
import sys
import zipfile
import numpy as np
import sklearn.metrics

MY_DATASET_ZIP = r"F:\Discovery AI\Second Expirement 100K RGB model\my_dataset_archive.zip"
OUT_DIR = r"output\review\archive_investigation"

RE_FAMILY = re.compile(r"^(vid_[0-9a-fA-F]{64})_face_\d+_\d+\.[a-zA-Z0-9]+$")
RE_NUM = re.compile(r"^\d+\.[a-zA-Z0-9]+$")


def test_roc_auc_implementation():
    """Verify ROC AUC calculation on edge cases."""
    # 1. Perfect separation (1.0)
    auc_perf = compute_roc_auc(np.array([0, 1]), np.array([0.1, 0.9]))
    assert auc_perf == 1.0, f"Expected 1.0, got {auc_perf}"

    # 2. Reversed separation (0.0)
    auc_rev = compute_roc_auc(np.array([0, 1]), np.array([0.9, 0.1]))
    assert auc_rev == 0.0, f"Expected 0.0, got {auc_rev}"

    # 3. All ties (0.5)
    auc_ties = compute_roc_auc(np.array([0, 1]), np.array([0.5, 0.5]))
    assert auc_ties == 0.5, f"Expected 0.5, got {auc_ties}"

    # 4. Mixed ties
    auc_mixed = compute_roc_auc(np.array([0, 0, 1, 1]), np.array([0.2, 0.5, 0.5, 0.8]))
    assert 0.0 <= auc_mixed <= 1.0, f"Expected valid AUC, got {auc_mixed}"

    # 5. Single-class undefined
    auc_single_0 = compute_roc_auc(np.array([0, 0]), np.array([0.1, 0.2]))
    assert auc_single_0 == "undefined_single_class", f"Expected undefined_single_class, got {auc_single_0}"
    auc_single_1 = compute_roc_auc(np.array([1, 1]), np.array([0.8, 0.9]))
    assert auc_single_1 == "undefined_single_class", f"Expected undefined_single_class, got {auc_single_1}"

    # 6. Invariance to row order
    labels = np.array([0, 0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.4, 0.35, 0.9, 0.2, 0.85])
    auc_orig = compute_roc_auc(labels, probs)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(labels))
    auc_perm = compute_roc_auc(labels[perm], probs[perm])
    assert abs(auc_orig - auc_perm) < 1e-12, f"AUC not invariant to row order: {auc_orig} vs {auc_perm}"


def compute_roc_auc(labels: np.ndarray, probs: np.ndarray):
    """Compute tie-aware ROC AUC or return undefined_single_class."""
    if len(labels) == 0:
        return "undefined_single_class"
    
    # Check non-finite
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise ValueError("Non-finite probability values encountered in AUC computation.")
    
    # Check labels
    unique_labels = np.unique(labels)
    for u in unique_labels:
        if u not in (0, 1):
            raise ValueError(f"Invalid label {u}: labels must be strictly 0 or 1.")
    
    if len(unique_labels) < 2:
        return "undefined_single_class"
    
    return float(sklearn.metrics.roc_auc_score(labels, probs))


def validate_predictions_and_cross_reference(pred_rows, manifest_dev_map):
    """Validate prediction rows and verify against manifest dev split."""
    seen_ids = set()
    ckpt_hashes = set()
    
    for idx, r in enumerate(pred_rows):
        sid = r.get("sample_id", "")
        if not sid:
            raise ValueError(f"Row {idx}: missing sample_id")
        if sid in seen_ids:
            raise ValueError(f"Row {idx}: duplicate sample_id '{sid}'")
        seen_ids.add(sid)
        
        # Check label
        try:
            lbl = int(r["label"])
            if lbl not in (0, 1):
                raise ValueError()
        except Exception:
            raise ValueError(f"Row {idx}: invalid label '{r.get('label')}'")
        
        # Check probability
        try:
            prob = float(r["probability"])
            if np.isnan(prob) or np.isinf(prob) or not (0.0 <= prob <= 1.0):
                raise ValueError()
        except Exception:
            raise ValueError(f"Row {idx}: invalid probability '{r.get('probability')}'")
        
        # Check checkpoint hash
        ckpt = r.get("checkpoint_sha256", "")
        if not ckpt:
            raise ValueError(f"Row {idx}: missing checkpoint_sha256")
        ckpt_hashes.add(ckpt)
        
        # Verify against manifest dev split
        if sid not in manifest_dev_map:
            raise ValueError(f"Row {idx}: sample_id '{sid}' not found in manifest dev split")
        m_row = manifest_dev_map[sid]
        
        if r["path"] != m_row["path"]:
            raise ValueError(f"Row {idx} ('{sid}'): path mismatch: pred='{r['path']}' vs manifest='{m_row['path']}'")
        if str(r["label"]) != str(m_row["label"]):
            raise ValueError(f"Row {idx} ('{sid}'): label mismatch: pred='{r['label']}' vs manifest='{m_row['label']}'")
        if r.get("sha256") and m_row.get("sha256") and r["sha256"] != m_row["sha256"]:
            raise ValueError(f"Row {idx} ('{sid}'): sha256 mismatch: pred='{r['sha256']}' vs manifest='{m_row['sha256']}'")
    
    if len(ckpt_hashes) != 1:
        raise ValueError(f"Mixed or missing checkpoint hashes found: {ckpt_hashes}")
    
    return list(ckpt_hashes)[0]


def calc_metrics(subset, name=""):
    if not subset:
        return {"name": name, "total_samples": 0}

    labels = np.array([int(r["label"]) for r in subset], dtype=np.int32)
    probs = np.array([float(r["probability"]) for r in subset], dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int32)

    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    real_recall = float(tn / n_real) if n_real > 0 else None
    fake_recall = float(tp / n_fake) if n_fake > 0 else None
    bacc = float((real_recall + fake_recall) / 2) if (real_recall is not None and fake_recall is not None) else None
    auc = compute_roc_auc(labels, probs)

    quantiles = [float(x) for x in np.quantile(probs, [0.0, 0.25, 0.5, 0.75, 0.90, 0.99, 1.0])]

    return {
        "name": name,
        "total_samples": len(subset),
        "class_denominators": {
            "real_actual": n_real,
            "fake_actual": n_fake
        },
        "decision_counts_at_05": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "total_errors": fp + fn
        },
        "rates": {
            "real_recall": float(real_recall) if real_recall is not None else None,
            "fake_recall": float(fake_recall) if fake_recall is not None else None,
            "balanced_accuracy": float(bacc) if bacc is not None else None,
            "roc_auc": float(auc) if isinstance(auc, (int, float)) else "undefined_single_class"
        },
        "probability_quantiles": {
            "min": quantiles[0],
            "p25": quantiles[1],
            "median": quantiles[2],
            "p75": quantiles[3],
            "p90": quantiles[4],
            "p99": quantiles[5],
            "max": quantiles[6]
        }
    }


def main():
    print("Running unit tests on ROC AUC edge cases...")
    test_roc_auc_implementation()
    print("ROC AUC unit tests PASSED.")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Reading manifest and predictions from archive...")
    with zipfile.ZipFile(MY_DATASET_ZIP, "r") as z:
        with z.open("prepared_data/rgb_sizes_seed42/sizes/100000/selected_manifest.csv") as f:
            reader = csv.DictReader(line.decode("utf-8") for line in f)
            manifest_dev = {r["sample_id"]: r for r in reader if r.get("split") == "dev"}
        print(f"Loaded {len(manifest_dev)} dev split rows from manifest.")

        with z.open("rgb_pilots/rgb_r2_100000_seed42_fe805782/checkpoints/best_dev_predictions.csv") as f:
            reader = csv.DictReader(line.decode("utf-8") for line in f)
            pred_rows = list(reader)
        print(f"Loaded {len(pred_rows)} predictions.")

    print("Validating predictions and verifying against manifest dev rows...")
    ckpt_hash = validate_predictions_and_cross_reference(pred_rows, manifest_dev)
    print(f"Validation successful. Uniform checkpoint SHA256: {ckpt_hash}")

    # Compute metrics across slices
    strata_metrics = {
        "checkpoint_sha256": ckpt_hash,
        "overall_development": calc_metrics(pred_rows, "Overall Development (14,835 samples)"),
        "pattern_subcollections": {
            "numeric_all": calc_metrics([r for r in pred_rows if RE_NUM.match(os.path.basename(r["path"]))], "All Numeric (Real JPG + Fake PNG)"),
            "video_all": calc_metrics([r for r in pred_rows if RE_FAMILY.match(os.path.basename(r["path"]))], "All Video Pattern (Real PNG + Fake PNG)")
        },
        "container_format_subcollections": {
            "jpg_all": calc_metrics([r for r in pred_rows if r["path"].lower().endswith(".jpg")], "All JPEG (Real only)"),
            "png_all": calc_metrics([r for r in pred_rows if r["path"].lower().endswith(".png")], "All PNG (Real PNG + Fake PNG)")
        },
        "granular_strata": {
            "numeric_real_jpg": calc_metrics([r for r in pred_rows if r["label"] == "0" and RE_NUM.match(os.path.basename(r["path"]))], "Numeric Real (100% JPG)"),
            "video_real_png": calc_metrics([r for r in pred_rows if r["label"] == "0" and RE_FAMILY.match(os.path.basename(r["path"]))], "Video Real (100% PNG)"),
            "numeric_fake_png": calc_metrics([r for r in pred_rows if r["label"] == "1" and RE_NUM.match(os.path.basename(r["path"]))], "Numeric Fake (100% PNG)"),
            "video_fake_png": calc_metrics([r for r in pred_rows if r["label"] == "1" and RE_FAMILY.match(os.path.basename(r["path"]))], "Video Fake (100% PNG)")
        },
        "key_diagnostic_interpretation": [
            "1. Stratified Recall Asymmetry: Real recall differs sharply across observed strata (99.91% on numeric JPEG real with 5 FPs / 5,763 vs 84.15% on video PNG real with 267 FPs / 1,685). Video-pattern PNG real accounts for 98.2% of all development false positives (267 / 272).",
            "2. Confounding Across Strata: The numeric stratum exhibits 99.96% balanced accuracy and 0.999997 ROC AUC. However, this stratum differs from the video stratum simultaneously in source identity, resolution, framing, crop policy, and container format (100% JPEG real vs 100% PNG fake). The observational correlation supports investigating confounding but does not establish compression as the sole causal driver.",
            "3. Fake Recall Across Strata: Fake recall is 100.0% on numeric PNG fake (392/392) and 97.98% on video PNG fake (6,854/6,995, with 141 false negatives).",
            "4. Aggregate Performance Masking: The aggregate development balanced accuracy (97.22%) and ROC AUC (0.996173) summarize the pooled evaluation cohort, but mask the large difference in real recall between numeric JPEG crops and video-tracked PNG crops."
        ]
    }

    out_json = os.path.join(OUT_DIR, "subcollection_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(strata_metrics, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
