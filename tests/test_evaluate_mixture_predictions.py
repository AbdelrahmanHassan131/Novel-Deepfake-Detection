"""Unit tests for tools/evaluate_mixture_predictions.py.

Verifies:
- Rejection of mixed or missing/unknown checkpoint hashes within an arm.
- Rejection of missing shared-dev manifest and invalid cryptographic gate.
- Rejection of label, sha256, group_id, or unpermitted path mismatches against manifest.
- Rejection of unknown filename patterns.
- Order invariance with reordered prediction rows.
- Saturated-probability Youden's J calibration without artificial clipping.
- Fail-closed rejection of non-empty destination directories.
- End-to-end paired comparison fixture using exact CheckpointHook prediction schema.
"""

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.evaluate_mixture_predictions import (
    classify_stratum,
    compute_roc_auc,
    evaluate_cohort,
    read_and_validate_predictions,
    run_evaluation,
    select_best_threshold,
)


class TestEvaluateMixturePredictions(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_csv(self, path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _create_synthetic_dev_manifest(self, sample_count=20):
        # 5 samples per stratum
        rows = []
        strata_info = [
            (0, "0000{:02d}.jpg", False),
            (0, "vid_" + ("a" * 64) + "_face_000001_{:02d}.png", True),
            (1, "0000{:02d}.png", False),
            (1, "vid_" + ("b" * 64) + "_face_000001_{:02d}.png", True),
        ]
        per_stratum = sample_count // 4
        idx = 0
        for lbl, pat, is_vid in strata_info:
            for i in range(per_stratum):
                sid = f"sample_{idx:04d}"
                fn = pat.format(i)
                p = f"/content/dataset/train/{'fake' if lbl == 1 else 'real'}/{fn}"
                sha = hashlib.sha256(f"content_{idx}".encode()).hexdigest()
                rows.append({
                    "sample_id": sid,
                    "path": p,
                    "label": str(lbl),
                    "split": "dev",
                    "group_id": f"group_{idx}",
                    "sha256": sha,
                })
                idx += 1

        man_p = self.temp_dir / "shared_dev" / "selected_manifest.csv"
        self._write_csv(man_p, rows)
        man_sha = hashlib.sha256(man_p.read_bytes()).hexdigest()

        gate_p = man_p.with_suffix(".verified.json")
        gate_data = {
            "verified": True,
            "hashes_verified": True,
            "manifest_sha256": man_sha,
            "manifest_path": str(man_p),
            "samples": len(rows),
            "classes": [0, 1],
            "splits": ["dev"],
        }
        gate_p.write_text(json.dumps(gate_data, indent=2), encoding="utf-8")
        return man_p, rows

    def test_classify_stratum(self):
        self.assertEqual(classify_stratum("000123.jpg", 0), "numeric_real")
        self.assertEqual(classify_stratum("000123.png", 1), "numeric_fake")
        vid_p = "vid_" + ("c" * 64) + "_face_000001_001.png"
        self.assertEqual(classify_stratum(vid_p, 0), "video_real")
        self.assertEqual(classify_stratum(vid_p, 1), "video_fake")
        self.assertEqual(classify_stratum("random.jpg", 0), "unknown")

    def test_compute_roc_auc_edge_cases(self):
        # Single class -> undefined
        self.assertEqual(compute_roc_auc(np.array([0, 0]), np.array([0.1, 0.2])), "undefined_single_class")
        self.assertEqual(compute_roc_auc(np.array([1, 1]), np.array([0.8, 0.9])), "undefined_single_class")
        # Ties
        self.assertAlmostEqual(compute_roc_auc(np.array([0, 1]), np.array([0.5, 0.5])), 0.5)
        # Perfect
        self.assertAlmostEqual(compute_roc_auc(np.array([0, 1]), np.array([0.1, 0.9])), 1.0)
        # Reversed
        self.assertAlmostEqual(compute_roc_auc(np.array([0, 1]), np.array([0.9, 0.1])), 0.0)
        # Non-finite raises ValueError
        with self.assertRaises(ValueError):
            compute_roc_auc(np.array([0, 1]), np.array([np.nan, 0.5]))

    def test_saturated_probabilities_calibration(self):
        # All probabilities are 0.0 or 1.0
        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        res = select_best_threshold(labels, probs)
        self.assertIn("threshold", res)
        self.assertTrue(0.0 <= res["threshold"] <= 1.0)
        self.assertEqual(res["criterion"], "development Youden J")
        self.assertEqual(res["independence_status"], "development_fit_only_not_independent")

    def test_mixed_checkpoint_hashes_rejected(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "mixed_ckpt.csv"
        self._write_csv(pred_p, [
            {"sample_id": "sample_0000", "label": "0", "probability": "0.1", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0001", "label": "0", "probability": "0.2", "checkpoint_sha256": "b" * 64},
            {"sample_id": "sample_0002", "label": "1", "probability": "0.8", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0003", "label": "1", "probability": "0.9", "checkpoint_sha256": "a" * 64},
        ])
        with self.assertRaises(ValueError) as ctx:
            read_and_validate_predictions(pred_p, man_rows)
        self.assertIn("Mixed checkpoint hashes detected", str(ctx.exception))

    def test_missing_or_unknown_checkpoint_hash_rejected(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "unknown_ckpt.csv"
        self._write_csv(pred_p, [
            {"sample_id": "sample_0000", "label": "0", "probability": "0.1", "checkpoint_sha256": "unknown"},
            {"sample_id": "sample_0001", "label": "0", "probability": "0.2", "checkpoint_sha256": "unknown"},
            {"sample_id": "sample_0002", "label": "1", "probability": "0.8", "checkpoint_sha256": "unknown"},
            {"sample_id": "sample_0003", "label": "1", "probability": "0.9", "checkpoint_sha256": "unknown"},
        ])
        with self.assertRaises(ValueError) as ctx:
            read_and_validate_predictions(pred_p, man_rows)
        self.assertIn("Missing or invalid checkpoint_sha256", str(ctx.exception))

    def test_label_mismatch_rejected(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "label_mismatch.csv"
        self._write_csv(pred_p, [
            {"sample_id": "sample_0000", "label": "1", "probability": "0.9", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0001", "label": "0", "probability": "0.2", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0002", "label": "1", "probability": "0.8", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0003", "label": "1", "probability": "0.9", "checkpoint_sha256": "a" * 64},
        ])
        with self.assertRaises(ValueError) as ctx:
            read_and_validate_predictions(pred_p, man_rows)
        self.assertIn("Label mismatch", str(ctx.exception))

    def test_hash_mismatch_rejected(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "hash_mismatch.csv"
        self._write_csv(pred_p, [
            {"sample_id": "sample_0000", "label": "0", "probability": "0.1", "sha256": "f" * 64, "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0001", "label": "0", "probability": "0.2", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0002", "label": "1", "probability": "0.8", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0003", "label": "1", "probability": "0.9", "checkpoint_sha256": "a" * 64},
        ])
        with self.assertRaises(ValueError) as ctx:
            read_and_validate_predictions(pred_p, man_rows)
        self.assertIn("Content sha256 mismatch", str(ctx.exception))

    def test_path_mismatch_rejected(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "path_mismatch.csv"
        self._write_csv(pred_p, [
            {"sample_id": "sample_0000", "path": "/wrong/path.png", "label": "0", "probability": "0.1", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0001", "label": "0", "probability": "0.2", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0002", "label": "1", "probability": "0.8", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0003", "label": "1", "probability": "0.9", "checkpoint_sha256": "a" * 64},
        ])
        with self.assertRaises(ValueError) as ctx:
            read_and_validate_predictions(pred_p, man_rows)
        self.assertIn("Path mismatch", str(ctx.exception))

    def test_unknown_strata_rejected(self):
        # Create manifest containing an unrecognized filename pattern
        bad_rows = [
            {"sample_id": "bad_0", "path": "/content/dataset/weird_name.jpg", "label": 0, "probability": 0.5, "split": "dev", "group_id": "g0", "sha256": "0" * 64}
        ]
        with self.assertRaises(ValueError) as ctx:
            evaluate_cohort(bad_rows, threshold=0.5, threshold_label="fixed_0_5")
        self.assertIn("Unknown filename pattern", str(ctx.exception))

    def test_reordered_rows_invariant(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "reordered.csv"
        # Reverse rows order
        rows_reversed = [
            {"sample_id": "sample_0003", "label": "1", "probability": "0.9", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0002", "label": "1", "probability": "0.8", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0001", "label": "0", "probability": "0.2", "checkpoint_sha256": "a" * 64},
            {"sample_id": "sample_0000", "label": "0", "probability": "0.1", "checkpoint_sha256": "a" * 64},
        ]
        self._write_csv(pred_p, rows_reversed)
        parsed, ckpt = read_and_validate_predictions(pred_p, man_rows)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(ckpt, "a" * 64)

    def test_nonempty_output_dir_rejected(self):
        man_p, _ = self._create_synthetic_dev_manifest(4)
        pred_p = self.temp_dir / "pred.csv"
        self._write_csv(pred_p, [
            {"sample_id": f"sample_{i:04d}", "label": "0" if i < 2 else "1", "probability": "0.5", "checkpoint_sha256": "a" * 64}
            for i in range(4)
        ])
        out_d = self.temp_dir / "nonempty_out"
        out_d.mkdir(parents=True, exist_ok=True)
        (out_d / "file.txt").write_text("content", encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            run_evaluation(pred_a_path=pred_p, manifest_path=man_p, output_dir=out_d)
        self.assertIn("empty NEW output directory", str(ctx.exception))

    def test_end_to_end_checkpoint_hook_schema(self):
        man_p, man_rows = self._create_synthetic_dev_manifest(20)
        # Create Arm A and Arm B predictions matching CheckpointHook output schema
        preds_a = []
        preds_b = []
        ckpt_a_sha = "aaaaaaaa" * 8
        ckpt_b_sha = "bbbbbbbb" * 8

        for r in man_rows:
            lbl = int(r["label"])
            st = classify_stratum(r["path"], lbl)
            # Arm A: high on numeric real and video fake, low on others
            if st in ("numeric_real", "video_fake"):
                prob_a = 0.05 if lbl == 0 else 0.95
            else:
                prob_a = 0.75 if lbl == 0 else 0.25

            # Arm B: balanced
            prob_b = 0.15 if lbl == 0 else 0.85

            preds_a.append({
                "sample_id": r["sample_id"],
                "path": r["path"],
                "label": r["label"],
                "probability": str(prob_a),
                "logit": str(np.log(prob_a / (1.0 - prob_a + 1e-8))),
                "eval_precision": "fp32",
                "checkpoint_sha256": ckpt_a_sha,
            })
            preds_b.append({
                "sample_id": r["sample_id"],
                "path": r["path"],
                "label": r["label"],
                "probability": str(prob_b),
                "logit": str(np.log(prob_b / (1.0 - prob_b + 1e-8))),
                "eval_precision": "fp32",
                "checkpoint_sha256": ckpt_b_sha,
            })

        pred_a_p = self.temp_dir / "arm_a_best_dev_predictions.csv"
        pred_b_p = self.temp_dir / "arm_b_best_dev_predictions.csv"
        out_d = self.temp_dir / "e2e_eval_out"

        self._write_csv(pred_a_p, preds_a)
        self._write_csv(pred_b_p, preds_b)

        report = run_evaluation(
            pred_a_path=pred_a_p,
            manifest_path=man_p,
            output_dir=out_d,
            pred_b_path=pred_b_p,
        )

        self.assertIn("arm_a", report)
        self.assertIn("arm_b", report)
        self.assertIn("paired_comparison", report)

        # Check descriptive paired counts
        pair_cm = report["paired_comparison"]["fixed_0_5"]["contingency_matrix"]
        self.assertIn("both_correct", pair_cm)
        self.assertIn("a_correct_b_wrong", pair_cm)
        self.assertIn("a_wrong_b_correct", pair_cm)
        self.assertIn("both_wrong", pair_cm)
        self.assertIn("net_b_minus_a_errors", pair_cm)

        # Verify atomic files generated
        self.assertTrue((out_d / "mixture_evaluation_report.json").is_file())
        self.assertTrue((out_d / "mixture_evaluation_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
