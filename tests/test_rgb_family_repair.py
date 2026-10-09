"""Deferred metadata-only regression tests; no image files or model loads."""
import hashlib
import unittest

from tools.repair_rgb_family_split import build_rows, family_id, tokens


class FamilyRepairTests(unittest.TestCase):
    def make_rows(self):
        rows = []
        for family in range(160):
            for label in (0, 1):
                for frame in range(5):
                    sid = f'{family}_{label}_{frame}'
                    rows.append(dict(sample_id=sid, label=str(label),
                        path=f'/data/{label}/vid_{family:064x}_face_{frame:06d}_000.png',
                        split='train' if frame % 2 else 'dev',
                        sha256=hashlib.sha256(sid.encode()).hexdigest(), group_id=sid,
                        dataset_source='diffgan', grouping_basis='image_level_unverified'))
        return rows

    def test_family_parse_is_specific_and_platform_independent(self):
        name = f'vid_{1:064x}_face_000020_003.png'
        self.assertEqual(family_id('F:\\data\\' + name), family_id('/data/' + name))
        self.assertIsNone(family_id('/data/video_1.png'))
        self.assertIsNone(family_id('/data/vid_short_face_1_2.png'))

    def test_no_family_or_content_leakage_and_cap_and_determinism(self):
        rows = self.make_rows()
        original = [dict(r) for r in rows]
        selected, report = build_rows(rows, 100, 40, cap=2, dev_fraction=.3)
        self.assertEqual(rows, original)
        reverse, _ = build_rows(list(reversed(rows)), 100, 40, cap=2, dev_fraction=.3)
        self.assertEqual(selected, reverse)
        self.assertEqual(report['parent_filename_families_crossing_splits'], 160)
        self.assertEqual(report['repaired_filename_families_crossing_splits'], 0)
        self.assertEqual(report['splits']['train']['actual'], 100)
        self.assertEqual(report['splits']['dev']['actual'], 40)
        memberships, counts = {}, {}
        for row in selected:
            for token in tokens(row):
                self.assertEqual(memberships.setdefault(token, row['split']), row['split'])
            key = row['group_id'], row['label']
            counts[key] = counts.get(key, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_rejects_test_repartition_and_conflicting_content(self):
        rows = self.make_rows()
        rows[0]['split'] = 'external_test'
        with self.assertRaisesRegex(ValueError, 'never final/test'):
            build_rows(rows)
        rows[0]['split'] = 'train'
        rows[5]['sha256'] = rows[0]['sha256']
        with self.assertRaisesRegex(ValueError, 'contradictory'):
            build_rows(rows)

    def test_training_guard_cannot_be_bypassed_with_engineering_flags(self):
        from training.validator import verify_source_readiness
        rows = self.make_rows()
        with self.assertRaisesRegex(ValueError, 'filename frame families'):
            verify_source_readiness([rows[0]], [rows[1]],
                                    allow_aggregate_sources=True, allow_source_overlap=True)


if __name__ == '__main__':
    unittest.main()
