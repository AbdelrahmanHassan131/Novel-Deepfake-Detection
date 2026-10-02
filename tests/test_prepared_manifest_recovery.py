"""Deferred stdlib recovery tests. No model imports or image reads are needed."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.recover_prepared_manifest import digest, recover


class PreparedRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.audit = self.root / 'source.audit.json'
        self.output = self.root / 'prepared'
        self.rows = []
        for split in ('train', 'dev'):
            for label in ('0', '1'):
                sid = f'{split}_{label}'
                self.rows.append(dict(sample_id=sid, path=str(self.root / (sid + '.png')),
                                      label=label, split=split, dataset_source='fixture', group_id=sid,
                                      identity_id='unknown', sha256=hashlib.sha256(sid.encode()).hexdigest()))

    def save_evidence(self):
        with self.source.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        self.audit.write_text(json.dumps(dict(passed=False, hashes_verified=True,
                                              samples=len(self.rows), errors=['sha256 overlaps train and dev: fixture'],
                                              manifest_sha256=digest(self.source))))

    def test_removes_whole_linked_training_component_and_reuses_result(self):
        duplicate = dict(self.rows[2], sample_id='duplicate', split='train', group_id='duplicate',
                         identity_id='shared', path=str(self.root / 'duplicate.png'))
        related = dict(self.rows[0], sample_id='related', group_id='related', identity_id='shared',
                       sha256=hashlib.sha256(b'related').hexdigest(), path=str(self.root / 'related.png'))
        self.rows += [duplicate, related]
        self.save_evidence()
        report = recover(self.source, self.audit, self.output, True, True)
        self.assertEqual(report['removed_training_samples'], 2)
        with (self.output / 'selected_manifest.csv').open(newline='') as stream:
            kept = list(csv.DictReader(stream))
        self.assertEqual({r['sample_id'] for r in kept}, {'train_0', 'train_1', 'dev_0', 'dev_1'})
        self.assertFalse(list(self.root.glob('*.png')))  # No image files existed or were needed.
        before = (self.output / 'selected_manifest.csv').stat().st_mtime_ns
        recover(self.source, self.audit, self.output, True, True)
        self.assertEqual((self.output / 'selected_manifest.csv').stat().st_mtime_ns, before)

    def test_requires_unchanged_dataset_and_bound_hash_evidence(self):
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, 'immutable_dataset'):
            recover(self.source, self.audit, self.output)
        self.source.write_text(self.source.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'digest'):
            recover(self.source, self.audit, self.output, True, True)

    def test_shortfall_requires_explicit_acceptance(self):
        self.rows.append(dict(self.rows[2], sample_id='duplicate', split='train',
                              path=str(self.root / 'duplicate.png'), group_id='duplicate'))
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, 'allow_shortfall'):
            recover(self.source, self.audit, self.output, True)
        self.assertFalse((self.output / 'selected_manifest.verified.json').exists())
        recover(self.source, self.audit, self.output, True, True)

    def test_eval_overlap_is_not_silently_repartitioned(self):
        self.rows.append(dict(self.rows[2], sample_id='external', split='external_test',
                              path=str(self.root / 'external.png'), group_id='external'))
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, 'evaluation partitions'):
            recover(self.source, self.audit, self.output, True, True)
        self.assertFalse((self.output / 'selected_manifest.verified.json').exists())

    def test_conflicting_evaluation_labels_require_review(self):
        self.rows[3]['sha256'] = self.rows[2]['sha256']
        self.save_evidence()
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            recover(self.source, self.audit, self.output, True, True)

    def test_changed_output_is_not_reused(self):
        self.save_evidence()
        recover(self.source, self.audit, self.output, True, True)
        target = self.output / 'selected_manifest.csv'
        target.write_text(target.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'changed files'):
            recover(self.source, self.audit, self.output, True, True)


if __name__ == '__main__':
    unittest.main()
