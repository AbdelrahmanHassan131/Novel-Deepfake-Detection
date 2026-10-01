"""Deferred runtime checks for smoke bounds; STATUS: NOT RUN during code-only work."""
import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_smoke_manifest import prepare


class SmokeManifestBounds(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.csv'
        self.output = self.root / 'output.csv'
        self.rows = [dict(sample_id=f'{split}_{label}', path=f'{split}/{label}.png',
                          label=str(label), split=split, dataset_source='fixture', group_id=f'{split}_{label}')
                     for split in ('train', 'dev') for label in (0, 1)]

    def write(self, rows):
        with self.source.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_bounded_copy_preserves_metadata_without_opening_images(self):
        self.write(self.rows)
        report = prepare(self.source, self.root, self.output)
        self.assertEqual(report['samples'], 4)
        with self.output.open(newline='') as stream:
            actual = list(csv.DictReader(stream))
        self.assertEqual([r['group_id'] for r in actual], [r['group_id'] for r in self.rows])
        self.assertTrue(all(Path(r['path']).is_absolute() for r in actual))
        self.assertFalse(self.output.with_suffix('.verified.json').exists())

    def test_oversized_pool_rejected_before_output(self):
        self.write(self.rows * 76)
        with self.assertRaisesRegex(ValueError, 'exceeds 300'):
            prepare(self.source, self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_class_and_oversized_dev_rejected(self):
        for rows in (self.rows[:-1], self.rows + [dict(self.rows[-1], sample_id=f'extra_{i}') for i in range(100)]):
            with self.subTest(size=len(rows)):
                self.write(rows)
                with self.assertRaises(ValueError):
                    prepare(self.source, self.root, self.output)
                self.assertFalse(self.output.exists())

    def test_output_collision_and_unassigned_input_rejected(self):
        self.write(self.rows)
        with self.assertRaises(ValueError):
            prepare(self.source, self.root, self.source)
        self.write([dict(row, split='unassigned') for row in self.rows])
        with self.assertRaises(ValueError):
            prepare(self.source, self.root, self.output)


if __name__ == '__main__':
    unittest.main()
