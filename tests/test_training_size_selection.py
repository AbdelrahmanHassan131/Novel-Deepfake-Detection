"""Deferred selection tests; NOT RUN during code-only implementation."""
import copy
import unittest

from prepare_dataset import build_pilot_100k


def pool():
    rows = []
    for split, count in [('train', 8), ('dev', 2), ('internal_test', 2)]:
        for label in (0, 1):
            for i in range(count):
                sid = f'{split}_{label}_{i}'
                rows.append(dict(sample_id=sid, path=f'/same_folder/{sid}.png',
                                 label=label, split=split, dataset_source='fixture',
                                 group_id=sid, source_video_id=sid))
    return rows


class TrainingSizeSelection(unittest.TestCase):
    def select(self, rows, size, **kwargs):
        return build_pilot_100k(copy.deepcopy(rows), train_size=size, **kwargs)

    def test_sizes_share_original_paths_and_preserve_evaluation(self):
        original = pool()
        eval_ids = {r['sample_id'] for r in original if r['split'] != 'train'}
        for size in ('4', '8', 'all'):
            with self.subTest(size=size):
                rows, report = self.select(original, size)
                train = [r for r in rows if r['split'] == 'train']
                self.assertEqual(len(train), 16 if size == 'all' else int(size))
                self.assertEqual({r['sample_id'] for r in rows if r['split'] != 'train'}, eval_ids)
                self.assertTrue({r['path'] for r in rows} <= {r['path'] for r in original})
                self.assertFalse(report['training_selection']['images_copied'])

    def test_full_mode_keeps_all_frames_and_natural_imbalance(self):
        rows = pool()
        for i in range(5):
            rows.append(dict(rows[0], sample_id=f'extra_{i}', path=f'/same_folder/extra_{i}.png'))
        selected, report = self.select(rows, 'all', frame_cap=1)
        self.assertEqual(sum(r['split'] == 'train' for r in selected), 21)
        self.assertIsNone(report['shortage_summary']['frame_cap_applied'])
        self.assertEqual(report['shortage_summary']['selected_real'], 13)
        self.assertEqual(report['shortage_summary']['selected_fake'], 8)

    def test_subset_repeatable_and_shortage_explicit(self):
        first, _ = self.select(pool(), '8')
        second, _ = self.select(list(reversed(pool())), '8')
        self.assertEqual({r['sample_id'] for r in first}, {r['sample_id'] for r in second})
        with self.assertRaisesRegex(ValueError, 'shortfall'):
            self.select(pool(), '100')
        _, report = self.select(pool(), '100', allow_shortfall=True)
        self.assertEqual(report['training_selection']['selected_train_size'], 16)

    def test_full_mode_respects_holdouts_and_rejects_conflicts(self):
        rows = pool()
        for row in rows[:2]:
            row['split'] = 'external_test'
            row['dataset_source'] = 'heldout'
        selected, _ = self.select(rows, 'all', holdout_source='heldout')
        self.assertFalse(any(r['split'] == 'train' and r['dataset_source'] == 'heldout' for r in selected))
        with self.assertRaises(ValueError):
            self.select(pool(), 'all', source_quotas={'fixture': 2})

    def test_invalid_sizes_rejected(self):
        for size in ('0', '-2', '3', '100K', '1.5', ''):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.select(pool(), size)


if __name__ == '__main__':
    unittest.main()
