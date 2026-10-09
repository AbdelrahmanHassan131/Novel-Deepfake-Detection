"""Deferred metadata-only checks for Colab multi-size preparation."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.colab_manifest_sizes import digest, partition, subsets
from data.manifest import write_manifest
from tools.recover_prepared_manifest import recover


class ColabSizesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def read(self, path):
        with Path(path).open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    def row(self, name, label, split):
        return dict(sample_id=name, path=str(self.root / (name + '.png')), label=str(label),
                    split=split, dataset_source='diffgan', group_id=name,
                    sha256=hashlib.sha256(name.encode()).hexdigest(),
                    grouping_basis='image_level_unverified')

    def parent(self):
        rows = [self.row(f'{label}_{i}', label, 'train') for label in (0, 1) for i in range(6)]
        rows += [self.row(f'dev_{label}', label, 'dev') for label in (0, 1)]
        # Same-class duplicates form an indivisible recovered group.
        duplicate = dict(rows[0], sample_id='duplicate', group_id='duplicate',
                         path=str(self.root / 'duplicate.png'))
        rows.append(duplicate)
        manifest, audit = self.root / 'audited.csv', self.root / 'audit.json'
        write_manifest(manifest, rows)
        audit.write_text(json.dumps(dict(hashes_verified=True, manifest_sha256=digest(manifest),
                                         errors=[], samples=len(rows))))
        recover(manifest, audit, self.root / 'clean', True, True, True)
        return self.root / 'clean/selected_manifest.csv'

    def test_shared_dev_nested_training_groups_shortfalls_and_reuse(self):
        parent = self.parent()
        out = self.root / 'sizes'
        subsets(parent, out, ['4', '8', '100', 'all'], 42, True)
        records = {s: self.read(out / s / 'selected_manifest.csv') for s in ('4', '8', '100', 'all')}
        ids = lambda rs, split: {r['sample_id'] for r in rs if r['split'] == split}
        self.assertLessEqual(ids(records['4'], 'train'), ids(records['8'], 'train'))
        self.assertEqual(ids(records['100'], 'train'), ids(records['all'], 'train'))
        parent_train = [r for r in self.read(parent) if r['split'] == 'train']
        for rows in records.values():
            self.assertEqual(ids(rows, 'dev'), {'dev_0', 'dev_1'})
            for group in {r['group_id'] for r in rows if r['split'] == 'train'}:
                self.assertEqual({r['sample_id'] for r in rows if r['group_id'] == group},
                                 {r['sample_id'] for r in parent_train if r['group_id'] == group})
        self.assertGreater(json.loads((out / '100/selection_report.json').read_text())['shortfall'], 0)
        subsets(parent, out, ['4', '8'], 42, True)
        self.assertFalse(list(self.root.glob('*.png')))  # No actual images needed or read.

    def test_requires_immutable_data_and_rejects_changed_parent(self):
        parent = self.parent()
        with self.assertRaisesRegex(ValueError, 'immutable_dataset'):
            subsets(parent, self.root / 'sizes', ['4'], 42, False)
        parent.write_text(parent.read_text() + '\n')
        with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
            subsets(parent, self.root / 'sizes', ['4'], 42, True)

    def test_partition_is_balanced_and_does_not_overwrite_fixed_splits(self):
        source, target = self.root / 'pool.csv', self.root / 'candidate.csv'
        rows = [self.row(f'{label}_{i}', label, 'unassigned') for label in (0, 1) for i in range(5)]
        write_manifest(source, rows)
        partition(source, target, 4, 42)
        dev = [r for r in self.read(target) if r['split'] == 'dev']
        self.assertEqual([sum(r['label'] == str(k) for r in dev) for k in (0, 1)], [2, 2])
        with self.assertRaisesRegex(ValueError, 'existing split'):
            partition(target, self.root / 'other.csv', 4, 42)


if __name__ == '__main__':
    unittest.main()
