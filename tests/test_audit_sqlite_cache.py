"""Deferred audit cache integration tests; no model construction."""
import tempfile
import unittest
from pathlib import Path
from data.hash_cache import SQLiteHashCache
from data.manifest import audit_rows


class AuditCache(unittest.TestCase):
    def test_reuses_completed_hashes_and_invalidates_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'image.bin'
            path.write_bytes(b'first')
            rows = [dict(sample_id='one', path=str(path), label=0, split='train',
                         dataset_source='fixture', group_id='one', sha256='unknown')]
            cache = SQLiteHashCache(root / 'hashes.sqlite')
            try:
                first = audit_rows(rows, hash_files=True, hash_cache=cache)
                self.assertTrue(first['passed'])
                self.assertEqual(cache.misses, 1)
            finally:
                cache.close()
            cache = SQLiteHashCache(root / 'hashes.sqlite')
            try:
                self.assertTrue(audit_rows(rows, hash_files=True, hash_cache=cache)['passed'])
                self.assertEqual(cache.hits, 1)
                path.write_bytes(b'changed size and contents')
                self.assertFalse(audit_rows(rows, hash_files=True, hash_cache=cache)['passed'])
                self.assertEqual(cache.misses, 1)
            finally:
                cache.close()


if __name__ == '__main__':
    unittest.main()
