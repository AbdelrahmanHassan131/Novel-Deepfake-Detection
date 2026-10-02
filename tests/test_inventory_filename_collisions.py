"""Regression cases for independent-image inventory; deferred, NOT RUN locally."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data.manifest import audit_rows, read_manifest
from prepare_dataset import build_pilot_100k, connected_components, main


class IndependentInventoryCollisions(unittest.TestCase):
    def inventory(self, root, source='fixture', independent=True):
        output = root / 'pool.csv'
        argv = ['prepare_dataset.py', 'inventory', '--root', str(root),
                '--source', source, '--output', str(output)]
        if independent:
            argv.append('--independent_images')
        with patch('sys.argv', argv):
            main()
        return read_manifest(output, root=root)

    def populate(self, root):
        for relative in ('train/real/0001.png', 'train/fake/0001.png',
                         'val/real/0001.png', 'val/fake/0001.jpg'):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            # Inventory/hash checks do not decode image data.
            path.write_bytes(relative.encode())

    def test_same_stems_do_not_create_cross_split_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            rows = self.inventory(root)
            self.assertEqual(len({r['group_id'] for r in rows}), 4)
            self.assertEqual(len(connected_components(rows)), 4)
            selected, _ = build_pilot_100k(copy.deepcopy(rows), train_size='all', frame_cap=0)
            self.assertEqual(sum(r['split'] == 'train' for r in selected), 2)
            self.assertEqual(sum(r['split'] == 'dev' for r in selected), 2)
            self.assertTrue(audit_rows(selected, hash_files=True)['passed'])

    def test_ids_stable_under_root_relocation_but_namespaced_by_source(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            roots = [Path(a), Path(b)]
            for root in roots:
                self.populate(root)
            first, second = [self.inventory(root) for root in roots]
            self.assertEqual({r['group_id'] for r in first}, {r['group_id'] for r in second})
            other = self.inventory(roots[1], source='other_source')
            self.assertFalse({r['group_id'] for r in first} & {r['group_id'] for r in other})

    def test_real_duplicates_and_identity_links_remain_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            rows = self.inventory(root)
            train = next(r for r in rows if r['split'] == 'train')
            dev = next(r for r in rows if r['split'] == 'dev')
            Path(dev['path']).write_bytes(Path(train['path']).read_bytes())
            self.assertFalse(audit_rows(rows, hash_files=True)['passed'])
            with self.assertRaisesRegex(ValueError, 'Conflicting fixed partition'):
                build_pilot_100k(copy.deepcopy(rows), train_size='all')
            # Also reject a verified shared identity even without hash evidence.
            for row in rows:
                row['sha256'] = 'unknown'
            train['identity_id'] = dev['identity_id'] = 'verified_same_person'
            with self.assertRaisesRegex(ValueError, 'Representative records'):
                build_pilot_100k(copy.deepcopy(rows), train_size='all')

    def test_default_inventory_still_requires_verified_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root)
            rows = self.inventory(root, independent=False)
            self.assertTrue(all(r['group_id'] == 'unknown' for r in rows))
            with self.assertRaisesRegex(ValueError, 'Missing group_id'):
                connected_components(rows)


if __name__ == '__main__':
    unittest.main()
