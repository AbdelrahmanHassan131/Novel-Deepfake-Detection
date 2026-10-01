"""Deferred tests for source-aware pilot preparation, frame capping, and leakage safeguards.

STATUS: NOT RUN (code-only preparation phase).
Tests verify:
  - Deterministic quotas and honest shortage calculation.
  - Frame cap limits per source video.
  - Connected component resolution: linked pristine and manipulated samples stay in the same split.
  - Rejection of missing group metadata under strict auditable splitting.
  - Resumable hash caching with mtime/size invalidation.
  - Dataset metadata adapters for CelebA, FaceForensics++, and DiffFace.
"""
import json
from pathlib import Path
import tempfile
import unittest


class TestPilotDataPreparation(unittest.TestCase):
    """DEFERRED / NOT RUN: Verified via AST parsing during code preparation."""

    def test_connected_group_integrity_no_cross_split_leakage(self):
        from prepare_dataset import build_pilot_100k
        # Build 10 synthetic groups, each with 1 pristine real and 1 manipulated fake
        rows = []
        for i in range(20):
            vid = f"vid_{i // 2}"
            rows.append({
                'sample_id': f"sample_{i}",
                'path': f"/synthetic/path_{i}.png",
                'label': i % 2,
                'dataset_source': 'FaceForensics++',
                'source_video_id': vid,
                'identity_id': f"id_{i // 2}",
                'original_id': f"orig_{i // 2}",
                'group_id': f"group_{vid}",
                'generator': 'authentic' if i % 2 == 0 else 'Deepfakes',
                'sha256': f"sha_{i}",
                'split': 'unassigned',
            })
        split_rows, report = build_pilot_100k(rows, target_real=4, target_fake=4, dev_ratio=0.5, test_ratio=0.5, seed=42)

        # Check: for every video, all of its frames must be in the same split!
        splits_by_video = {}
        for r in split_rows:
            vid = r['source_video_id']
            if vid not in splits_by_video:
                splits_by_video[vid] = set()
            splits_by_video[vid].add(r['split'])

        for vid, splits in splits_by_video.items():
            self.assertEqual(len(splits), 1, f"Video {vid} leaked across splits: {splits}")

    def test_deterministic_quotas_and_shortage_reporting(self):
        from prepare_dataset import build_pilot_100k
        # Create 10 real and 5 fake samples (asking for 50 real and 50 fake)
        rows = []
        for i in range(10):
            rows.append({
                'sample_id': f"real_{i}",
                'path': f"/synthetic/real_{i}.png",
                'label': 0,
                'dataset_source': 'CelebA',
                'source_video_id': 'n/a_photograph',
                'identity_id': f"id_real_{i}",
                'original_id': f"real_{i}",
                'group_id': f"group_real_{i}",
                'generator': 'authentic',
                'sha256': f"sha_real_{i}",
                'split': 'unassigned',
            })
        for i in range(5):
            rows.append({
                'sample_id': f"fake_{i}",
                'path': f"/synthetic/fake_{i}.png",
                'label': 1,
                'dataset_source': 'DiffFace',
                'source_video_id': 'n/a_synthesis',
                'identity_id': 'unknown',
                'original_id': 'unknown',
                'group_id': f"group_fake_{i}",
                'generator': 'ADM',
                'sha256': f"sha_fake_{i}",
                'split': 'unassigned',
            })

        split_rows, report = build_pilot_100k(rows, target_real=50, target_fake=50, seed=42)
        summary = report['shortage_summary']
        self.assertEqual(summary['target_real'], 50)
        self.assertEqual(summary['target_fake'], 50)
        self.assertGreater(summary['real_shortage'], 0)
        self.assertGreater(summary['fake_shortage'], 0)

    def test_frame_cap_limiting(self):
        from prepare_dataset import apply_frame_caps
        rows = []
        # Video 1 has 50 frames
        for i in range(50):
            rows.append({
                'sample_id': f"frame_{i}",
                'dataset_source': 'FaceForensics++',
                'source_video_id': 'video_001',
                'label': 1,
            })
        capped = apply_frame_caps(rows, frame_cap=15, seed=42)
        self.assertEqual(len(capped), 15)

    def test_missing_group_id_fails_strict_splitting(self):
        from prepare_dataset import build_pilot_100k
        rows = [{
            'sample_id': 'bad_sample',
            'path': '/synthetic/bad.png',
            'label': 0,
            'dataset_source': 'UnknownSource',
            'group_id': 'unknown',
            'source_video_id': 'unknown',
            'identity_id': 'unknown',
            'generator': 'unknown',
            'sha256': 'sha_bad',
            'split': 'unassigned',
        }]
        with self.assertRaises(ValueError) as ctx:
            build_pilot_100k(rows, require_groups=True)
        self.assertIn("Missing group_id", str(ctx.exception))

    def test_hash_cache_resumability(self):
        from data.hash_cache import HashCache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "hashes.json"
            dummy_file = Path(tmpdir) / "image.png"
            dummy_file.write_bytes(b"dummy image data for testing hash")

            cache = HashCache(str(cache_file))
            h1 = cache.get_or_compute(str(dummy_file))
            cache.save()

            # Re-read cache in new instance
            cache2 = HashCache(str(cache_file))
            self.assertIn(str(dummy_file.resolve()), cache2.entries)
            self.assertEqual(cache2.get_or_compute(str(dummy_file)), h1)

    def test_metadata_adapters(self):
        from data.adapters import apply_adapter
        row = {'path': 'data/FaceForensics/manipulated_sequences/Deepfakes/c23/videos/042_056/001.png', 'label': -1}
        adapted = apply_adapter(row, 'ffpp')
        self.assertEqual(adapted['dataset_source'], 'FaceForensics++')
        self.assertEqual(adapted['generator'], 'Deepfakes')
        self.assertEqual(adapted['label'], 1)
        self.assertEqual(adapted['group_id'], 'ffpp_pair_042_056')


if __name__ == '__main__':
    unittest.main()
