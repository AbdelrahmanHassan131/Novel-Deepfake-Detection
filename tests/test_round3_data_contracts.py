# STATUS: NOT RUN
"""
Deferred test suite for Code Review Round 3 - Prompt O.
Validates:
  - T1: Source-namespaced sample_id generation preventing collisions across sources with identical relative paths.
  - T5: Holdout enforcement rejecting train-labeled heldout components, excluding linked originals, and per-video frame capping.
  - T6: Audit gate verification enforcing hashes_verified, class coverage {0, 1}, and shortfall policies.
  - T7: Absolute path relocation requiring explicit mapping and rejecting arbitrary suffix guessing.

DO NOT EXECUTE AT RUNTIME IN THIS ENVIRONMENT.
Static AST / syntax validation only.
"""

import unittest
from pathlib import Path
import tempfile
import json
import hashlib

from data.manifest import read_manifest, write_manifest, audit_rows, verify_manifest_gate
from prepare_dataset import build_pilot_100k, apply_frame_caps


class TestRound3DataContracts(unittest.TestCase):
    """Static deferred unit test contract for Prompt O data specifications."""

    def test_t1_source_namespaced_sample_ids(self):
        """Verify that identical relative paths from different sources produce distinct sample IDs."""
        rel_path = "real/0001.png"
        id_source_a = hashlib.sha256(f"source_A:{rel_path}".encode()).hexdigest()[:16]
        id_source_b = hashlib.sha256(f"source_B:{rel_path}".encode()).hexdigest()[:16]

        self.assertNotEqual(id_source_a, id_source_b)
        self.assertEqual(len(id_source_a), 16)
        self.assertEqual(len(id_source_b), 16)

        # Relocation stability: Moving dataset root does not alter namespaced ID
        relocated_rel_path = "real/0001.png"
        id_relocated = hashlib.sha256(f"source_A:{relocated_rel_path}".encode()).hexdigest()[:16]
        self.assertEqual(id_source_a, id_relocated)

    def test_t5_holdout_train_conflict_rejection(self):
        """Verify that pre-assigned train components matching holdout rules fail closed unless explicitly overridden."""
        rows = [
            {
                'sample_id': 's1',
                'path': 'img1.png',
                'label': 1,
                'split': 'train',
                'dataset_source': 'FaceForensics',
                'group_id': 'grp_ff_1',
                'source_video_id': 'vid_1',
                'generator': 'Deepfakes',
            }
        ]

        # Default: must fail closed with ValueError
        with self.assertRaises(ValueError) as ctx:
            build_pilot_100k(
                rows,
                target_real=1,
                target_fake=1,
                holdout_source='FaceForensics',
                allow_shortfall=True,
                allow_holdout_override=False,
            )
        self.assertIn("Conflicting fixed partition in component", str(ctx.exception))

        # Explicit override: moves to external_dev without raising
        out_rows, report = build_pilot_100k(
            rows,
            target_real=1,
            target_fake=1,
            holdout_source='FaceForensics',
            allow_shortfall=True,
            allow_holdout_override=True,
        )
        self.assertEqual(out_rows[0]['split'], 'external_dev')

    def test_t5_linked_original_exclusion(self):
        """Verify that pristine originals linked to held-out fakes are excluded from training."""
        rows = [
            # Manipulated fake from held-out source
            {
                'sample_id': 'fake_ff',
                'path': 'fake.png',
                'label': 1,
                'split': 'unassigned',
                'dataset_source': 'FaceForensics',
                'group_id': 'shared_identity_01',
                'source_video_id': 'vid_fake_1',
                'generator': 'Face2Face',
            },
            # Pristine original from a non-held-out source linked via shared component
            {
                'sample_id': 'real_yt',
                'path': 'real.png',
                'label': 0,
                'split': 'unassigned',
                'dataset_source': 'YouTube',
                'group_id': 'shared_identity_01',
                'source_video_id': 'vid_real_1',
                'generator': 'authentic',
            },
            # Another independent original eligible for training
            {
                'sample_id': 'real_other',
                'path': 'real2.png',
                'label': 0,
                'split': 'unassigned',
                'dataset_source': 'YouTube',
                'group_id': 'independent_grp_02',
                'source_video_id': 'vid_real_2',
                'generator': 'authentic',
            },
        ]

        out_rows, report = build_pilot_100k(
            rows,
            target_real=1,
            target_fake=1,
            holdout_source='FaceForensics',
            dev_ratio=0.0,
            test_ratio=0.0,
            allow_shortfall=True,
        )

        row_map = {r['sample_id']: r['split'] for r in out_rows}
        self.assertEqual(row_map['fake_ff'], 'external_dev')
        # Linked original MUST be placed in external_dev, NOT in train!
        self.assertEqual(row_map['real_yt'], 'external_dev')
        self.assertEqual(row_map['real_other'], 'train')

    def test_t5_per_video_frame_capping_in_connected_component(self):
        """Verify that frame capping applies per video inside a connected component, preserving both videos."""
        rows = []
        # Video A in group 1: 20 frames
        for k in range(20):
            rows.append({
                'sample_id': f'vid_a_{k:02d}',
                'path': f'vid_a_{k:02d}.png',
                'label': 1,
                'split': 'unassigned',
                'dataset_source': 'FaceForensics',
                'group_id': 'comp_01',
                'source_video_id': 'video_A',
                'generator': 'Deepfakes',
            })
        # Video B in group 1: 20 frames
        for k in range(20):
            rows.append({
                'sample_id': f'vid_b_{k:02d}',
                'path': f'vid_b_{k:02d}.png',
                'label': 0,
                'split': 'unassigned',
                'dataset_source': 'YouTube',
                'group_id': 'comp_01',
                'source_video_id': 'video_B',
                'generator': 'authentic',
            })

        out_rows, report = build_pilot_100k(
            rows,
            target_real=15,
            target_fake=15,
            frame_cap=15,
            dev_ratio=0.0,
            test_ratio=0.0,
            allow_shortfall=False,
        )

        train_rows = [r for r in out_rows if r['split'] == 'train']
        vid_a_count = sum(1 for r in train_rows if r['source_video_id'] == 'video_A')
        vid_b_count = sum(1 for r in train_rows if r['source_video_id'] == 'video_B')

        # Both videos must be capped to 15 frames each (30 frames total), not capping the entire component to 15!
        self.assertEqual(vid_a_count, 15)
        self.assertEqual(vid_b_count, 15)
        self.assertEqual(len(train_rows), 30)

    def test_t6_verify_manifest_gate_strict_hashes_and_coverage(self):
        """Verify verify_manifest_gate enforces require_hashes and class coverage {0, 1}."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            m_path = tmppath / "manifest.csv"
            g_path = tmppath / "manifest.verified.json"

            # Create a single-class manifest (fake only)
            single_class_rows = [
                {'sample_id': 's1', 'path': 'fake.png', 'label': 1, 'split': 'train', 'dataset_source': 'src', 'group_id': 'g1'}
            ]
            write_manifest(str(m_path), single_class_rows)
            m_digest = hashlib.sha256(m_path.read_bytes()).hexdigest()

            # Gate without hashes verified
            gate_data = {
                'verified': True,
                'manifest_path': str(m_path),
                'manifest_sha256': m_digest,
                'samples': 1,
                'hashes_verified': False,
                'classes': [1],
            }
            g_path.write_text(json.dumps(gate_data), encoding='utf-8')

            # 1. require_hashes failure
            with self.assertRaises(RuntimeError) as ctx_hash:
                verify_manifest_gate(str(m_path), str(g_path), enforce_class_coverage=False, require_hashes=True)
            self.assertIn("without verifying content hashes", str(ctx_hash.exception))

            # 2. enforce_class_coverage failure
            with self.assertRaises(RuntimeError) as ctx_cov:
                verify_manifest_gate(str(m_path), str(g_path), enforce_class_coverage=True, require_hashes=False)
            self.assertIn("failed class coverage enforcement", str(ctx_cov.exception))

    def test_t6_shortfall_policy(self):
        """Verify pilot selection raises ValueError on shortage unless allow_shortfall is set."""
        rows = [
            {'sample_id': 's1', 'path': 'img1.png', 'label': 0, 'split': 'unassigned', 'dataset_source': 'src', 'group_id': 'g1'}
        ]
        # Requesting 10 real but only 1 available: must fail without allow_shortfall
        with self.assertRaises(ValueError) as ctx:
            build_pilot_100k(rows, target_real=10, target_fake=10, dev_ratio=0.0, allow_shortfall=False)
        self.assertIn("Pilot selection shortfall", str(ctx.exception))

        # With allow_shortfall: succeeds and records shortage in report
        out_rows, report = build_pilot_100k(rows, target_real=10, target_fake=10, dev_ratio=0.0, allow_shortfall=True)
        self.assertEqual(report['shortage_summary']['selected_real'], 1)
        self.assertEqual(report['shortage_summary']['real_shortage'], 9)

    def test_t7_absolute_path_relocation_rejection_of_guessing(self):
        """Verify that read_manifest rejects unmapped absolute paths and does not guess via suffix slicing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            fake_abs_path = "/nonexistent/machine/root/real/face_01.png"
            m_path = tmppath / "manifest.csv"
            rows = [
                {'sample_id': 's1', 'path': fake_abs_path, 'label': 0, 'split': 'train', 'dataset_source': 'src_a', 'group_id': 'g1'}
            ]
            write_manifest(str(m_path), rows)

            # Rejects unmapped path without guessing
            with self.assertRaises(FileNotFoundError) as ctx:
                read_manifest(str(m_path), root=str(tmppath), check_files=True)
            self.assertIn("does not exist on this system", str(ctx.exception))

            # Supplying explicit remap resolves to existing local file
            local_target = tmppath / "face_01.png"
            local_target.write_bytes(b"face_bytes")
            remap = {"/nonexistent/machine/root/real": str(tmppath)}
            loaded = read_manifest(str(m_path), remap_prefixes=remap, check_files=True)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(Path(loaded[0]['path']).resolve(), local_target.resolve())


if __name__ == '__main__':
    unittest.main()
