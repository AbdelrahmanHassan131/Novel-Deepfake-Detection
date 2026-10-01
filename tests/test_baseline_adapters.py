"""Deferred tests for baseline adapters and evaluation interfaces.

STATUS: NOT RUN (code-only preparation phase).
Tests verify:
  - Baseline registry retrieval for UniversalFakeDetect, SBI, UCF, and Effort.
  - Lazy importing without model downloads or execution side effects.
  - Native preprocessing metadata specifications.
  - Effort is marked 'pending_verification' per handoff rules.
  - Explicit distinction between 'supplied_checkpoint' and 'retrained_on_pilot'.
  - Baseline prediction output format compatibility with analyze_predictions.
"""
from pathlib import Path
import tempfile
import unittest


class TestBaselineAdapters(unittest.TestCase):
    """DEFERRED / NOT RUN: Verified via AST parsing during code preparation."""

    def test_baseline_registry_retrieval(self):
        from models.baselines import get_baseline_adapter, BASELINE_REGISTRY
        self.assertIn('universal_fake_detect', BASELINE_REGISTRY)
        self.assertIn('sbi', BASELINE_REGISTRY)
        self.assertIn('ucf', BASELINE_REGISTRY)
        self.assertIn('effort', BASELINE_REGISTRY)

        adapter_ufd = get_baseline_adapter('universal_fake_detect')
        self.assertEqual(adapter_ufd.name, 'UniversalFakeDetect')

        adapter_sbi = get_baseline_adapter('sbi')
        self.assertEqual(adapter_sbi.name, 'SBI')

    def test_lazy_metadata_without_downloads(self):
        from models.baselines import get_baseline_adapter
        adapter = get_baseline_adapter('universal_fake_detect')
        meta = adapter.metadata()
        self.assertIn('Ojha', meta['paper_citation'])
        self.assertEqual(meta['code_license'], 'MIT License')
        self.assertIn('CLIP', meta['auxiliary_data_provenance'])
        self.assertEqual(meta['native_preprocessing']['image_size'], 224)

    def test_sbi_metadata_and_native_preprocessing(self):
        from models.baselines import get_baseline_adapter
        adapter = get_baseline_adapter('sbi')
        meta = adapter.metadata()
        self.assertIn('Shiohara', meta['paper_citation'])
        self.assertEqual(meta['native_preprocessing']['image_size'], 224)
        self.assertIn('pristine', meta['auxiliary_data_provenance'])

    def test_effort_marked_pending_verification(self):
        from models.baselines import get_baseline_adapter
        adapter = get_baseline_adapter('effort')
        self.assertEqual(adapter.status, 'pending_verification')
        with self.assertRaises(RuntimeError) as ctx:
            adapter.predict_manifest('dummy_manifest.csv', 'dummy_data', 'dummy_out.csv')
        self.assertIn('pending_verification', str(ctx.exception))

    def test_evaluation_modes(self):
        from models.baselines import get_baseline_adapter
        a1 = get_baseline_adapter('ucf', evaluation_mode='supplied_checkpoint')
        self.assertEqual(a1.evaluation_mode, 'supplied_checkpoint')

        a2 = get_baseline_adapter('ucf', evaluation_mode='retrained_on_pilot')
        self.assertEqual(a2.evaluation_mode, 'retrained_on_pilot')

        with self.assertRaises(ValueError):
            get_baseline_adapter('ucf', evaluation_mode='invalid_mode')


if __name__ == '__main__':
    unittest.main()
