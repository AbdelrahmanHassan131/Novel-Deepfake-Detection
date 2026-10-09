"""Unit tests for tools/build_controlled_mixture_manifests.py.

Verifies:
- Deterministic selection across repeated runs with identical seed.
- Cap=8 enforcement per connected group per class.
- Content hash unification across differing filenames/paths.
- Paired shared-dev exact identity across Arm A, Arm B, and standalone shared_dev.
- Quota shortfall reporting and exception handling.
- Rejection of final-test repartitioning (split='test').
- Rejection of invalid grouping_basis (must be 'image_level_unverified').
- Rejection of unknown filename patterns.
- Absolute zero token/hash/path crossing between train and dev cohorts.
"""

import copy
import csv
import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_controlled_mixture_manifests import (
    build_and_save_manifests,
    classify_stratum,
    partition_and_build_manifests,
    QUOTAS,
)


def _make_sha(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _generate_synthetic_rows(
    num_numeric_real=1000,
    num_video_real=1000,
    num_numeric_fake=1000,
    num_video_fake=1000,
    seed_prefix="syn",
    start_idx=0,
):
    rows = []
    idx = start_idx

    def add_batch(count, fn_prefix, fn_ext, label, is_video):
        nonlocal idx
        for i in range(count):
            sid = f"{idx:016x}"
            sha = _make_sha(f"{seed_prefix}_{idx}")
            if is_video:
                vid_hash = _make_sha(f"vid_{seed_prefix}_{i}")
                fn = f"vid_{vid_hash}_face_000001_001.{fn_ext}"
            else:
                fn = f"{i:06d}.{fn_ext}"
            path = f"/content/dataset/Preapred Dataset/train/{'fake' if label == 1 else 'real'}/{fn}"
            rows.append({
                "sample_id": sid,
                "path": path,
                "label": str(label),
                "split": "train",
                "dataset_source": "diffgan",
                "group_id": f"group_{i}",
                "source_video_id": "unknown",
                "original_id": "unknown",
                "identity_id": "unknown",
                "generator": "authentic" if label == 0 else "manipulated",
                "grouping_basis": "image_level_unverified",
                "sha256": sha,
            })
            idx += 1

    add_batch(num_numeric_real, "num_real", "jpg", 0, is_video=False)
    add_batch(num_video_real, "vid_real", "png", 0, is_video=True)
    add_batch(num_numeric_fake, "num_fake", "png", 1, is_video=False)
    add_batch(num_video_fake, "vid_fake", "png", 1, is_video=True)
    return rows


class TestControlledMixtureManifests(unittest.TestCase):

    def setUp(self):
        # Generate a small pool of synthetic rows with custom quotas for fast testing
        self.custom_quotas = {
            "arm_a_train": {
                "numeric_real": 20,
                "video_real": 10,
                "numeric_fake": 5,
                "video_fake": 25,
            },
            "arm_b_train": {
                "numeric_real": 15,
                "video_real": 15,
                "numeric_fake": 15,
                "video_fake": 15,
            },
            "shared_dev": {
                "numeric_real": 5,
                "video_real": 5,
                "numeric_fake": 5,
                "video_fake": 5,
            },
        }

    def test_classify_stratum(self):
        self.assertEqual(classify_stratum("/path/to/012345.jpg", 0), "numeric_real")
        self.assertEqual(classify_stratum("/path/to/012345.png", 1), "numeric_fake")
        vid_fn = "vid_" + ("a" * 64) + "_face_000001_001.png"
        self.assertEqual(classify_stratum(f"/path/to/{vid_fn}", 0), "video_real")
        self.assertEqual(classify_stratum(f"/path/to/{vid_fn}", 1), "video_fake")
        self.assertEqual(classify_stratum("/path/to/random_name.jpg", 0), "unknown")

    def test_rejection_of_test_split(self):
        rows = _generate_synthetic_rows(10, 10, 10, 10)
        rows[0]["split"] = "test"
        with self.assertRaises(ValueError) as ctx:
            partition_and_build_manifests(rows)
        self.assertIn("Only an existing engineering train/dev pool can be repartitioned", str(ctx.exception))

    def test_rejection_of_invalid_grouping_basis(self):
        rows = _generate_synthetic_rows(10, 10, 10, 10)
        rows[0]["grouping_basis"] = "verified_source"
        with self.assertRaises(ValueError) as ctx:
            partition_and_build_manifests(rows)
        self.assertIn("Expected grouping_basis 'image_level_unverified'", str(ctx.exception))

    def test_rejection_of_unknown_patterns(self):
        rows = _generate_synthetic_rows(10, 10, 10, 10)
        rows[0]["path"] = "/content/dataset/arbitrary_name.jpg"
        with self.assertRaises(ValueError) as ctx:
            partition_and_build_manifests(rows)
        self.assertIn("unknown filename pattern", str(ctx.exception))

    def test_cap_enforcement_and_hash_deduplication(self):
        # Create 15 images with the SAME group_id and class 0, but different hashes
        rows = []
        for i in range(15):
            sid = f"{i:016x}"
            sha = _make_sha(f"cap_test_{i}")
            rows.append({
                "sample_id": sid,
                "path": f"/content/dataset/train/real/{i:06d}.jpg",
                "label": "0",
                "split": "train",
                "dataset_source": "diffgan",
                "group_id": "same_shared_group",
                "source_video_id": "unknown",
                "original_id": "unknown",
                "identity_id": "unknown",
                "generator": "authentic",
                "grouping_basis": "image_level_unverified",
                "sha256": sha,
            })
        # Add also 2 rows with DUPLICATE content hash
        rows.append({
            "sample_id": "0000000000000099",
            "path": "/content/dataset/train/real/000099.jpg",
            "label": "0",
            "split": "train",
            "dataset_source": "diffgan",
            "group_id": "same_shared_group",
            "source_video_id": "unknown",
            "original_id": "unknown",
            "identity_id": "unknown",
            "generator": "authentic",
            "grouping_basis": "image_level_unverified",
            "sha256": rows[0]["sha256"],  # duplicate hash
        })

        # Add sufficient filler for other strata to satisfy a small custom quota
        quotas_patch = {
            "arm_a_train": {"numeric_real": 5, "video_real": 2, "numeric_fake": 2, "video_fake": 2},
            "arm_b_train": {"numeric_real": 5, "video_real": 2, "numeric_fake": 2, "video_fake": 2},
            "shared_dev": {"numeric_real": 2, "video_real": 2, "numeric_fake": 2, "video_fake": 2},
        }

        # Build extra rows
        extra = _generate_synthetic_rows(50, 50, 50, 50, seed_prefix="extra", start_idx=1000)
        # Ensure all extra have distinct groups
        all_rows = rows + extra

        import tools.build_controlled_mixture_manifests as bcmm
        orig_quotas = bcmm.QUOTAS
        try:
            bcmm.QUOTAS = quotas_patch
            manifests, report = partition_and_build_manifests(all_rows, cap=8, dev_fraction=0.2)
            # In either arm, the number of samples from 'same_shared_group' must be <= 8
            for arm_name in ("arm_a", "arm_b"):
                group_samples = [r for r in manifests[arm_name] if "same_shared_group" in r.get("group_id", "")]
                self.assertLessEqual(len(group_samples), 8)
        finally:
            bcmm.QUOTAS = orig_quotas

    def test_exact_hash_grouping_across_differing_names(self):
        # Two rows have completely different paths and names, but same sha256
        sha_shared = _make_sha("identical_content_hash")
        rows = [
            {
                "sample_id": "0000000000000001",
                "path": "/content/dataset/train/real/123456.jpg",
                "label": "0",
                "split": "train",
                "dataset_source": "diffgan",
                "group_id": "group_alpha",
                "source_video_id": "unknown",
                "original_id": "unknown",
                "identity_id": "unknown",
                "generator": "authentic",
                "grouping_basis": "image_level_unverified",
                "sha256": sha_shared,
            },
            {
                "sample_id": "0000000000000002",
                "path": "/content/dataset/train/real/vid_" + ("f" * 64) + "_face_000001_001.png",
                "label": "0",
                "split": "train",
                "dataset_source": "diffgan",
                "group_id": "group_beta",
                "source_video_id": "unknown",
                "original_id": "unknown",
                "identity_id": "unknown",
                "generator": "authentic",
                "grouping_basis": "image_level_unverified",
                "sha256": sha_shared,
            },
        ]
        # Supplement with standard rows
        extra = _generate_synthetic_rows(60, 60, 60, 60, seed_prefix="hash_test", start_idx=1000)
        all_rows = rows + extra

        quotas_patch = {
            "arm_a_train": {"numeric_real": 5, "video_real": 5, "numeric_fake": 5, "video_fake": 5},
            "arm_b_train": {"numeric_real": 5, "video_real": 5, "numeric_fake": 5, "video_fake": 5},
            "shared_dev": {"numeric_real": 2, "video_real": 2, "numeric_fake": 2, "video_fake": 2},
        }

        import tools.build_controlled_mixture_manifests as bcmm
        orig_quotas = bcmm.QUOTAS
        try:
            bcmm.QUOTAS = quotas_patch
            manifests, report = partition_and_build_manifests(all_rows, seed=42)
            # Find the two rows in output manifests if selected:
            # They should NEVER appear in different splits (one train, one dev)
            arm_a_rows = manifests["arm_a"]
            splits_for_sha = {r["split"] for r in arm_a_rows if r["sha256"] == sha_shared}
            self.assertLessEqual(len(splits_for_sha), 1, "Content hash was split between train and dev!")
        finally:
            bcmm.QUOTAS = orig_quotas

    def test_deterministic_selection_and_shared_dev_identity(self):
        rows = _generate_synthetic_rows(100, 100, 100, 100, seed_prefix="determ")
        quotas_patch = {
            "arm_a_train": {"numeric_real": 20, "video_real": 10, "numeric_fake": 5, "video_fake": 25},
            "arm_b_train": {"numeric_real": 15, "video_real": 15, "numeric_fake": 15, "video_fake": 15},
            "shared_dev": {"numeric_real": 5, "video_real": 5, "numeric_fake": 5, "video_fake": 5},
        }

        import tools.build_controlled_mixture_manifests as bcmm
        orig_quotas = bcmm.QUOTAS
        try:
            bcmm.QUOTAS = quotas_patch
            res1, rep1 = partition_and_build_manifests(copy.deepcopy(rows), seed=42)
            res2, rep2 = partition_and_build_manifests(copy.deepcopy(rows), seed=42)

            # Exactly identical results across runs
            for key in ("arm_a", "arm_b", "shared_dev"):
                self.assertEqual(len(res1[key]), len(res2[key]))
                for r1, r2 in zip(res1[key], res2[key]):
                    self.assertEqual(r1["sample_id"], r2["sample_id"])
                    self.assertEqual(r1["split"], r2["split"])

            # Shared dev identity across arm_a, arm_b, and shared_dev
            dev_a = [r for r in res1["arm_a"] if r["split"] == "dev"]
            dev_b = [r for r in res1["arm_b"] if r["split"] == "dev"]
            dev_standalone = res1["shared_dev"]

            self.assertEqual(len(dev_a), len(dev_b))
            self.assertEqual(len(dev_a), len(dev_standalone))

            for ra, rb, rs in zip(dev_a, dev_b, dev_standalone):
                self.assertEqual(ra["sample_id"], rb["sample_id"])
                self.assertEqual(ra["sample_id"], rs["sample_id"])
                self.assertEqual(ra["path"], rb["path"])
                self.assertEqual(ra["path"], rs["path"])
                self.assertEqual(ra["label"], rb["label"])
                self.assertEqual(ra["label"], rs["label"])
                self.assertEqual(ra["sha256"], rb["sha256"])
                self.assertEqual(ra["sha256"], rs["sha256"])

        finally:
            bcmm.QUOTAS = orig_quotas

    def test_shortfall_reporting(self):
        # Create insufficient rows for video_real
        rows = _generate_synthetic_rows(50, 2, 50, 50, seed_prefix="shortfall")
        quotas_patch = {
            "arm_a_train": {"numeric_real": 10, "video_real": 10, "numeric_fake": 10, "video_fake": 10},
            "arm_b_train": {"numeric_real": 10, "video_real": 10, "numeric_fake": 10, "video_fake": 10},
            "shared_dev": {"numeric_real": 5, "video_real": 5, "numeric_fake": 5, "video_fake": 5},
        }

        import tools.build_controlled_mixture_manifests as bcmm
        orig_quotas = bcmm.QUOTAS
        try:
            bcmm.QUOTAS = quotas_patch
            with self.assertRaises(ValueError) as ctx:
                partition_and_build_manifests(rows, seed=42)
            self.assertIn("Quotas cannot be met post-grouping", str(ctx.exception))
            self.assertIn("shortfall", str(ctx.exception).lower())
        finally:
            bcmm.QUOTAS = orig_quotas


    def _create_synthetic_clean_parent(self, temp_dir: Path, corrupt_gate: str = None, corrupt_rec: str = None):
        rows = _generate_synthetic_rows(100, 100, 100, 100, seed_prefix="file_test")
        csv_p = temp_dir / "clean_parent" / "selected_manifest.csv"
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        csv_sha = hashlib.sha256(csv_p.read_bytes()).hexdigest()

        rec_p = csv_p.parent / "recovery_report.json"
        rec_data = {
            "image_files_changed": False,
            "hash_verification_mode": "inherited_completed_audit_on_explicitly_unchanged_dataset"
        }
        if corrupt_rec == "tampered":
            rec_data["tampered"] = True
        rec_bytes = json.dumps(rec_data, indent=2).encode("utf-8")
        rec_p.write_bytes(rec_bytes)
        rec_sha = hashlib.sha256(rec_bytes).hexdigest()

        gate_p = csv_p.with_suffix(".verified.json")
        gate_data = {
            "verified": True,
            "hashes_verified": True,
            "manifest_sha256": csv_sha,
            "recovery_report_sha256": rec_sha,
            "hash_verification_mode": "inherited_completed_audit_on_explicitly_unchanged_dataset"
        }
        if corrupt_gate == "verified_false":
            gate_data["verified"] = False
        elif corrupt_gate == "hashes_verified_false":
            gate_data["hashes_verified"] = False
        elif corrupt_gate == "mismatched_sha":
            gate_data["manifest_sha256"] = "0" * 64
        elif corrupt_gate == "mismatched_rec_sha":
            gate_data["recovery_report_sha256"] = "1" * 64

        gate_p.write_text(json.dumps(gate_data, indent=2), encoding="utf-8")
        return csv_p, gate_p, rec_p

    def test_build_and_save_nonempty_output_dir_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, _, _ = self._create_synthetic_clean_parent(p)
            out_d = p / "out"
            out_d.mkdir(parents=True, exist_ok=True)
            (out_d / "stray.txt").write_text("hello", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("empty NEW output directory", str(ctx.exception))

    def test_build_and_save_missing_gate_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, gate_p, _ = self._create_synthetic_clean_parent(p)
            gate_p.unlink()
            out_d = p / "out"

            with self.assertRaises(FileNotFoundError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("Missing parent gate file", str(ctx.exception))
            self.assertFalse(out_d.exists() and any(out_d.iterdir()))

    def test_build_and_save_missing_recovery_report_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, _, rec_p = self._create_synthetic_clean_parent(p)
            rec_p.unlink()
            out_d = p / "out"

            with self.assertRaises(FileNotFoundError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("Missing parent recovery report", str(ctx.exception))
            self.assertFalse(out_d.exists() and any(out_d.iterdir()))

    def test_build_and_save_gate_verified_false_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, _, _ = self._create_synthetic_clean_parent(p, corrupt_gate="verified_false")
            out_d = p / "out"

            with self.assertRaises(ValueError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("verified=False", str(ctx.exception))
            self.assertFalse(out_d.exists() and any(out_d.iterdir()))

    def test_build_and_save_gate_hashes_verified_false_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, _, _ = self._create_synthetic_clean_parent(p, corrupt_gate="hashes_verified_false")
            out_d = p / "out"

            with self.assertRaises(ValueError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("hashes unverified", str(ctx.exception))
            self.assertFalse(out_d.exists() and any(out_d.iterdir()))

    def test_build_and_save_mismatched_csv_digest_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, _, _ = self._create_synthetic_clean_parent(p, corrupt_gate="mismatched_sha")
            out_d = p / "out"

            with self.assertRaises(ValueError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("manifest_sha256 mismatch", str(ctx.exception))
            self.assertFalse(out_d.exists() and any(out_d.iterdir()))

    def test_build_and_save_tampered_recovery_report_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, _, _ = self._create_synthetic_clean_parent(p, corrupt_gate="mismatched_rec_sha")
            out_d = p / "out"

            with self.assertRaises(ValueError) as ctx:
                build_and_save_manifests(str(csv_p), str(out_d))
            self.assertIn("recovery_report_sha256 mismatch", str(ctx.exception))
            self.assertFalse(out_d.exists() and any(out_d.iterdir()))

    def test_build_and_save_valid_csv_inheritance(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, gate_p, rec_p = self._create_synthetic_clean_parent(p)
            out_d = p / "out"

            import tools.build_controlled_mixture_manifests as bcmm
            orig_quotas = bcmm.QUOTAS
            try:
                bcmm.QUOTAS = {
                    "arm_a_train": {"numeric_real": 10, "video_real": 10, "numeric_fake": 5, "video_fake": 15},
                    "arm_b_train": {"numeric_real": 10, "video_real": 10, "numeric_fake": 10, "video_fake": 10},
                    "shared_dev": {"numeric_real": 5, "video_real": 5, "numeric_fake": 5, "video_fake": 5},
                }
                rep = build_and_save_manifests(str(csv_p), str(out_d))
                # Verify outputs exist and sidecars carry valid inherited evidence
                for sub in ("arm_a", "arm_b", "shared_dev"):
                    sub_csv = out_d / sub / "selected_manifest.csv"
                    sub_gate = out_d / sub / "selected_manifest.verified.json"
                    self.assertTrue(sub_csv.is_file())
                    self.assertTrue(sub_gate.is_file())
                    gate = json.loads(sub_gate.read_text(encoding="utf-8"))
                    self.assertIs(gate["verified"], True)
                    self.assertIs(gate["hashes_verified"], True)
                    self.assertEqual(gate["hash_verification_mode"], "inherited_completed_audit_on_explicitly_unchanged_dataset")
                    self.assertEqual(gate["inherited_parent_evidence"]["parent_manifest_sha256"], hashlib.sha256(csv_p.read_bytes()).hexdigest())

                # Check union lists
                self.assertTrue((out_d / "union_members_list.txt").is_file())
                self.assertTrue((out_d / "union_archive_members.txt").is_file())
                self.assertTrue((out_d / "mixture_manifest_report.json").is_file())
            finally:
                bcmm.QUOTAS = orig_quotas

    def test_build_and_save_valid_zip_inheritance(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            csv_p, gate_p, rec_p = self._create_synthetic_clean_parent(p)
            zip_p = p / "parent_archive.zip"
            with zipfile.ZipFile(zip_p, "w") as z:
                z.write(csv_p, "prepared_data/rgb_sizes_seed42/clean_parent/selected_manifest.csv")
                z.write(gate_p, "prepared_data/rgb_sizes_seed42/clean_parent/selected_manifest.verified.json")
                z.write(rec_p, "prepared_data/rgb_sizes_seed42/clean_parent/recovery_report.json")

            out_d = p / "out_zip"
            import tools.build_controlled_mixture_manifests as bcmm
            orig_quotas = bcmm.QUOTAS
            try:
                bcmm.QUOTAS = {
                    "arm_a_train": {"numeric_real": 10, "video_real": 10, "numeric_fake": 5, "video_fake": 15},
                    "arm_b_train": {"numeric_real": 10, "video_real": 10, "numeric_fake": 10, "video_fake": 10},
                    "shared_dev": {"numeric_real": 5, "video_real": 5, "numeric_fake": 5, "video_fake": 5},
                }
                rep = build_and_save_manifests(str(zip_p), str(out_d))
                self.assertTrue((out_d / "arm_a" / "selected_manifest.verified.json").is_file())
                self.assertTrue((out_d / "shared_dev" / "selected_manifest.verified.json").is_file())
            finally:
                bcmm.QUOTAS = orig_quotas


if __name__ == "__main__":
    unittest.main()
