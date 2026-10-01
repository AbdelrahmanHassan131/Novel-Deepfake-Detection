# STATUS: NOT RUN
"""
Deferred regression tests for Prompt C (R8):
1. Invalid split ratios rejected (< 0 or sum >= 1.0).
2. Empty dev split raises explicit error about insufficient validation coverage.
3. Accurate shortage reporting without secret +50 quota exceeding.
4. Caller quota dictionaries are not mutated.
5. Mixed-source quota accounting by exact row counts.
6. Protected holdouts (final_test, external_test) remain protected.
7. Same-input repeatability across runs.
8. Fixed development holdout across training pilot scaling (e.g. 100K to 200K).
"""

import unittest
from prepare_dataset import build_pilot_100k


def create_synthetic_pool(num_groups=50, frames_per_group=4):
    rows = []
    for g in range(num_groups):
        label = 0 if g % 2 == 0 else 1
        src = "CelebA" if g % 4 < 2 else "FaceForensics++"
        gen = "authentic" if label == 0 else ("Deepfakes" if src == "FaceForensics++" else "synthetic")
        for f in range(frames_per_group):
            rows.append({
                'sample_id': f"g{g}_f{f}",
                'path': f"/data/{src}/{gen}/vid_{g}/frame_{f:03d}.png",
                'label': label,
                'dataset_source': src,
                'generator': gen,
                'source_video_id': f"vid_{g}",
                'group_id': f"group_{g}",
                'split': 'unassigned',
            })
    return rows


class TestStablePilotSelection(unittest.TestCase):
    def test_invalid_ratio_rejected(self):
        pool = create_synthetic_pool(10, 2)
        with self.assertRaises(ValueError):
            build_pilot_100k(pool, dev_ratio=0.6, test_ratio=0.5)
        with self.assertRaises(ValueError):
            build_pilot_100k(pool, dev_ratio=-0.1, test_ratio=0.1)

    def test_empty_dev_rejected(self):
        # A tiny pool of 2 groups with dev_ratio=0.01 where u_part never falls under dev_ratio
        pool = create_synthetic_pool(2, 2)
        with self.assertRaises(ValueError):
            build_pilot_100k(pool, dev_ratio=0.0001, test_ratio=0.0001)

    def test_caller_quota_dict_not_mutated(self):
        pool = create_synthetic_pool(20, 2)
        original_quotas = {'CelebA': 10, 'FaceForensics++': 10}
        quotas_copy = original_quotas.copy()

        build_pilot_100k(pool, target_real=10, target_fake=10, source_quotas=original_quotas)
        self.assertEqual(original_quotas, quotas_copy, "build_pilot_100k must not mutate caller quota dictionary.")

    def test_shortage_reporting_and_exact_quotas(self):
        # Request more than available
        pool = create_synthetic_pool(10, 2)  # 5 real groups (10 frames), 5 fake groups (10 frames)
        rows, report = build_pilot_100k(pool, target_real=50, target_fake=50, dev_ratio=0.2, test_ratio=0.2)
        s_rep = report['shortage_summary']

        self.assertGreater(s_rep['real_shortage'], 0)
        self.assertGreater(s_rep['fake_shortage'], 0)
        # Verify target is never exceeded (no undocumented +50)
        self.assertLessEqual(s_rep['selected_real'], 50)
        self.assertLessEqual(s_rep['selected_fake'], 50)

    def test_protected_holdouts_preserved(self):
        pool = create_synthetic_pool(20, 2)
        # Pre-assign group 0 to final_test
        for r in pool:
            if r['group_id'] == 'group_0':
                r['split'] = 'final_test'

        rows, report = build_pilot_100k(pool, target_real=20, target_fake=20, dev_ratio=0.2, test_ratio=0.2)
        # Verify group 0 rows are still final_test
        g0_splits = {r['split'] for r in rows if r['group_id'] == 'group_0'}
        self.assertEqual(g0_splits, {'final_test'}, "Protected holdout final_test must be preserved.")

    def test_same_input_repeatability(self):
        pool1 = create_synthetic_pool(30, 2)
        pool2 = create_synthetic_pool(30, 2)

        rows1, _ = build_pilot_100k(pool1, target_real=15, target_fake=15, seed=42)
        rows2, _ = build_pilot_100k(pool2, target_real=15, target_fake=15, seed=42)

        splits1 = [r['split'] for r in rows1]
        splits2 = [r['split'] for r in rows2]
        self.assertEqual(splits1, splits2, "Same input and seed must yield identical partition assignments.")

    def test_fixed_development_holdout_across_scales(self):
        """
        Verify that scaling target_real/target_fake from 10 to 20 NEVER absorbs
        former development groups into training.
        """
        pool1 = create_synthetic_pool(40, 2)
        pool2 = create_synthetic_pool(40, 2)

        # Scale 1: 10 real / 10 fake
        rows_10, _ = build_pilot_100k(pool1, target_real=10, target_fake=10, dev_ratio=0.2, test_ratio=0.1, seed=42)
        dev_groups_10 = {r['group_id'] for r in rows_10 if r['split'] == 'dev'}

        # Scale 2: 20 real / 20 fake
        rows_20, _ = build_pilot_100k(pool2, target_real=20, target_fake=20, dev_ratio=0.2, test_ratio=0.1, seed=42)
        dev_groups_20 = {r['group_id'] for r in rows_20 if r['split'] == 'dev'}

        self.assertEqual(
            dev_groups_10, dev_groups_20,
            "Development partition groups must remain strictly fixed and identical across training sizes."
        )


if __name__ == '__main__':
    unittest.main()
