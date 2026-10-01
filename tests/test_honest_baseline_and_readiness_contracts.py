# STATUS: NOT RUN
"""
Deferred tests for honest baseline status and readiness contracts (Prompt H).
Verifies:
1. Baseline adapters (UniversalFakeDetect, SBI, UCF) report 'incomplete_stub_future_work' status and are not falsely marked ready for runtime.
2. Calling load_model or predict_manifest raises NotImplementedError explaining they are architectural specification stubs.
3. Declarative experiment study plans in config/experiments/ are explicitly annotated with plan_type and consumer.
4. CODE_READY_STATUS.md maintains strict distinctions between code readiness, incomplete baseline stubs, and pending empirical Colab runs.
"""
import json
import unittest
from pathlib import Path


class TestHonestBaselineAndReadinessContracts(unittest.TestCase):

    def test_baseline_adapters_status_incomplete(self):
        from models.baselines.universal_fake_detect import UniversalFakeDetectAdapter
        from models.baselines.sbi import SBIAdapter
        from models.baselines.ucf import UCFAdapter

        adapters = [
            UniversalFakeDetectAdapter(),
            SBIAdapter(),
            UCFAdapter(),
        ]

        for adapter in adapters:
            self.assertEqual(
                adapter.status,
                'incomplete_stub_future_work',
                f"Adapter {adapter.name} must not be marked 'ready_for_runtime'"
            )
            # Verify load_model raises NotImplementedError
            with self.assertRaises(NotImplementedError) as ctx:
                adapter.load_model()
            self.assertIn("stub", str(ctx.exception).lower())

            # Verify predict_manifest raises NotImplementedError
            with self.assertRaises(NotImplementedError) as ctx:
                adapter.predict_manifest(
                    manifest_path="dummy_manifest.csv",
                    dataroot="/dummy/data",
                    output_predictions_path="/dummy/preds.csv"
                )
            self.assertIn("stub", str(ctx.exception).lower())

    def test_declarative_study_plans_annotated(self):
        config_dir = Path("config/experiments")
        self.assertTrue(config_dir.is_dir())
        json_files = list(config_dir.glob("*.json"))
        self.assertGreaterEqual(len(json_files), 4)

        for jf in json_files:
            data = json.loads(jf.read_text(encoding='utf-8'))
            self.assertEqual(
                data.get('plan_type'),
                'declarative_study_plan',
                f"Study plan {jf.name} must be explicitly labeled as 'declarative_study_plan'"
            )
            self.assertIn('consumer', data)

    def test_code_ready_status_honesty(self):
        status_file = Path("CODE_READY_STATUS.md")
        self.assertTrue(status_file.exists())
        content = status_file.read_text(encoding='utf-8')

        # Verify baselines are classified as stubs
        self.assertIn("incomplete_stub_future_work", content)
        # Verify no false zero-gap assertion ignores baselines
        self.assertIn("Architectural stubs", content)
        # Verify Colab execution is listed as pending
        self.assertIn("CODE READY FOR MANUAL COLAB EXECUTION", content)


if __name__ == '__main__':
    unittest.main()
