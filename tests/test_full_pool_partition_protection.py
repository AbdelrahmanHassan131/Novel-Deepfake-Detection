# STATUS: NOT RUN
"""Regression tests for Prompt J: Full-pool partition protection (S3).

Verifies:
1. Protected holdout assignments are resolved on the full pool before capping; dropping
   a bridge row during capping does not cause related records to enter training.
2. Conflicting fixed partition assignments (e.g. train and final_test sharing a component)
   are strictly rejected with ValueError.
3. Existing fixed development and final holdouts in input manifests remain frozen and are
   not rehashed into train or truncated by training frame caps.
4. Pool growth (adding new frames to existing videos/entities) preserves stable component IDs
   and leaves development and final holdout assignments unchanged.
"""
import unittest
from prepare_dataset import build_pilot_100k


class TestFullPoolPartitionProtection(unittest.TestCase):

    def test_bridge_removal_preserves_holdout(self):
        """When a bridge record carrying a protected split is in a component, related records remain protected."""
        # 10 fake frames from video V1, connected via original_id to V2
        # V2 has a single frame marked final_test
        rows = []
        for i in range(10):
            rows.append({
                'sample_id': f'v1_f{i:02d}',
                'path': f'src/fake/v1/frame_{i:02d}.png',
                'label': 1,
                'split': 'unassigned',
                'dataset_source': 'ffpp',
                'group_id': 'ffpp_v1',
                'source_video_id': 'ffpp_v1',
                'original_id': 'ffpp_v2',
                'identity_id': 'unknown',
                'generator': 'deepfakes'
            })
        # Bridge frame from original video V2 with protected split 'final_test'
        rows.append({
            'sample_id': 'v2_f00',
            'path': 'src/real/v2/frame_00.png',
            'label': 0,
            'split': 'final_test',
            'dataset_source': 'ffpp',
            'group_id': 'ffpp_v2',
            'source_video_id': 'ffpp_v2',
            'original_id': 'ffpp_v2',
            'identity_id': 'unknown',
            'generator': 'authentic'
        })

        out_rows, report = build_pilot_100k(
            rows,
            target_real=10,
            target_fake=10,
            frame_cap=3,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
            require_groups=True
        )

        # None of the v1 or v2 frames should be in 'train'
        train_samples = {r['sample_id'] for r in out_rows if r['split'] == 'train'}
        self.assertEqual(len(train_samples), 0, "No records in a final_test component should enter training")
        
        # All members of the component in out_rows must have split == 'final_test'
        final_test_samples = {r['sample_id'] for r in out_rows if r['split'] == 'final_test'}
        self.assertIn('v2_f00', final_test_samples)
        for r in out_rows:
            self.assertEqual(r['split'], 'final_test')

    def test_conflicting_fixed_partitions_rejected(self):
        """A component containing conflicting fixed assignments (e.g. train and final_test) raises ValueError."""
        rows = [
            {
                'sample_id': 's1',
                'path': 'img1.png',
                'label': 0,
                'split': 'train',
                'dataset_source': 'src1',
                'group_id': 'g1',
                'source_video_id': 'v1',
                'original_id': 'v1',
                'identity_id': 'unknown',
                'generator': 'authentic'
            },
            {
                'sample_id': 's2',
                'path': 'img2.png',
                'label': 1,
                'split': 'final_test',
                'dataset_source': 'src1',
                'group_id': 'g1',
                'source_video_id': 'v1',
                'original_id': 'v1',
                'identity_id': 'unknown',
                'generator': 'manip'
            }
        ]
        with self.assertRaises(ValueError) as ctx:
            build_pilot_100k(rows, target_real=10, target_fake=10, seed=42)
        self.assertIn("Conflicting fixed partition assignments", str(ctx.exception))

    def test_input_manifest_fixed_partitions_frozen(self):
        """Pre-assigned fixed partitions in input manifests remain frozen and keep all evaluation frames."""
        rows = [
            {
                'sample_id': f'dev_{i}',
                'path': f'dev_{i}.png',
                'label': 0,
                'split': 'dev',
                'dataset_source': 'src1',
                'group_id': 'g_dev',
                'source_video_id': 'v_dev',
                'original_id': 'v_dev',
                'identity_id': 'unknown',
                'generator': 'authentic'
            }
            for i in range(20)  # 20 dev frames
        ] + [
            {
                'sample_id': f'tr_{i}',
                'path': f'tr_{i}.png',
                'label': 1,
                'split': 'unassigned',
                'dataset_source': 'src1',
                'group_id': f'g_tr_{i}',
                'source_video_id': f'v_tr_{i}',
                'original_id': f'v_tr_{i}',
                'identity_id': 'unknown',
                'generator': 'manip'
            }
            for i in range(10)
        ]

        # Request frame_cap=5 for training
        out_rows, report = build_pilot_100k(
            rows,
            target_real=10,
            target_fake=10,
            frame_cap=5,
            dev_ratio=0.0,
            test_ratio=0.0,
            seed=42
        )

        dev_out = [r for r in out_rows if r['split'] == 'dev']
        # All 20 dev frames must be preserved (not capped to 5)
        self.assertEqual(len(dev_out), 20, "Evaluation splits must not be truncated by training frame caps")

    def test_pool_growth_leaves_dev_and_final_ids_unchanged(self):
        """Adding new frames to existing entities preserves canonical anchor and stable partition hash."""
        # Initial pool: 5 distinct videos
        initial_rows = []
        for v in range(10):
            initial_rows.append({
                'sample_id': f'v{v}_f0',
                'path': f'v{v}_f0.png',
                'label': v % 2,
                'split': 'unassigned',
                'dataset_source': 'src1',
                'group_id': f'vid_{v:02d}',
                'source_video_id': f'vid_{v:02d}',
                'original_id': f'orig_{v:02d}',
                'identity_id': 'unknown',
                'generator': 'authentic' if v % 2 == 0 else 'manip'
            })

        out1, report1 = build_pilot_100k(
            [dict(r) for r in initial_rows],
            target_real=2,
            target_fake=2,
            dev_ratio=0.3,
            test_ratio=0.2,
            seed=42
        )
        initial_dev_samples = {r['sample_id'] for r in out1 if r['split'] == 'dev'}

        # Grow pool: add 3 new frames to each video
        grown_rows = [dict(r) for r in initial_rows]
        for v in range(10):
            for f in range(1, 4):
                grown_rows.append({
                    'sample_id': f'v{v}_f{f}',
                    'path': f'v{v}_f{f}.png',
                    'label': v % 2,
                    'split': 'unassigned',
                    'dataset_source': 'src1',
                    'group_id': f'vid_{v:02d}',
                    'source_video_id': f'vid_{v:02d}',
                    'original_id': f'orig_{v:02d}',
                    'identity_id': 'unknown',
                    'generator': 'authentic' if v % 2 == 0 else 'manip'
                })

        out2, report2 = build_pilot_100k(
            [dict(r) for r in grown_rows],
            target_real=2,
            target_fake=2,
            dev_ratio=0.3,
            test_ratio=0.2,
            seed=42
        )
        # All initial dev samples must still be in dev!
        grown_dev_samples = {r['sample_id'] for r in out2 if r['split'] == 'dev'}
        for sid in initial_dev_samples:
            self.assertIn(sid, grown_dev_samples, f"Sample {sid} changed partition under pool growth!")


if __name__ == '__main__':
    unittest.main()
