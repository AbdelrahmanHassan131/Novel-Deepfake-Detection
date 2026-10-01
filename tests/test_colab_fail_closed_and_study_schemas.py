# STATUS: NOT RUN
"""
Deferred regression test suite for Prompt M / Findings S7 & S9.
Verifies fail-closed preparation audit gate verification, stale/missing gate blocking,
checked runner generation in notebook, and schema-aware study plan dispatch.

DO NOT EXECUTE AT RUNTIME IN THIS ENVIRONMENT.
Static validation only.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data.manifest import verify_manifest_gate
from tools.run_study_plan import (
    parse_study_plan,
    identify_plan_schema,
    generate_pipeline_commands,
)


class TestAuditGateAndFailClosed(unittest.TestCase):
    """Verifies that unverified, modified, or failed data preparation halts execution before training."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.manifest_path = self.base_dir / "pilot_manifest.csv"
        self.manifest_content = "sample_id,path,label,split,group_id,dataset_source\n1,1.png,0,train,g1,s1\n2,2.png,1,train,g2,s1\n"
        self.manifest_path.write_text(self.manifest_content, encoding='utf-8')
        self.manifest_digest = hashlib.sha256(self.manifest_content.encode('utf-8')).hexdigest()
        self.gate_path = self.manifest_path.with_suffix('.verified.json')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_verify_manifest_gate_success(self):
        """When gate file exists with verified=True and matching digest, verification succeeds."""
        gate_data = {
            'verified': True,
            'manifest_path': str(self.manifest_path),
            'manifest_sha256': self.manifest_digest,
            'samples': 2,
        }
        self.gate_path.write_text(json.dumps(gate_data), encoding='utf-8')

        result = verify_manifest_gate(self.manifest_path)
        self.assertTrue(result['verified'])
        self.assertEqual(result['manifest_sha256'], self.manifest_digest)

    def test_verify_manifest_gate_missing_gate_raises(self):
        """If audit was never run, gate file does not exist, raising RuntimeError."""
        if self.gate_path.is_file():
            self.gate_path.unlink()

        with self.assertRaises(RuntimeError) as ctx:
            verify_manifest_gate(self.manifest_path)
        self.assertIn("Preparation gate not found", str(ctx.exception))
        self.assertIn("audit", str(ctx.exception))

    def test_verify_manifest_gate_audit_failed_raises(self):
        """If audit recorded verified=False, verification raises RuntimeError."""
        gate_data = {
            'verified': False,
            'manifest_path': str(self.manifest_path),
            'manifest_sha256': self.manifest_digest,
        }
        self.gate_path.write_text(json.dumps(gate_data), encoding='utf-8')

        with self.assertRaises(RuntimeError) as ctx:
            verify_manifest_gate(self.manifest_path)
        self.assertIn("indicates audit failed or is unverified", str(ctx.exception))

    def test_verify_manifest_gate_stale_digest_raises(self):
        """If manifest was altered after the audit, digest mismatch raises RuntimeError."""
        gate_data = {
            'verified': True,
            'manifest_path': str(self.manifest_path),
            'manifest_sha256': self.manifest_digest,
        }
        self.gate_path.write_text(json.dumps(gate_data), encoding='utf-8')

        # Modify manifest content
        self.manifest_path.write_text(self.manifest_content + "3,3.png,0,train,g3,s1\n", encoding='utf-8')

        with self.assertRaises(RuntimeError) as ctx:
            verify_manifest_gate(self.manifest_path)
        self.assertIn("Manifest digest mismatch", str(ctx.exception))
        self.assertIn("modified after the audit passed", str(ctx.exception))


class TestStudyPlanSchemasAndCommandContracts(unittest.TestCase):
    """Verifies schema identification, mandatory data audits, and parameter wiring across study plans."""

    def test_identify_plan_schemas(self):
        # Multi-stage pipeline
        p_l4 = {"plan_type": "declarative_study_plan", "stages": {}}
        self.assertEqual(identify_plan_schema(p_l4), 'multi_stage_pipeline')

        # Multi-seed comparison
        p_seeds = {"plan_type": "declarative_study_plan", "models_to_compare": []}
        self.assertEqual(identify_plan_schema(p_seeds), 'multi_seed_comparison')

        # Targeted ablations
        p_abl = {"plan_type": "declarative_study_plan", "ablations": []}
        self.assertEqual(identify_plan_schema(p_abl), 'targeted_ablations')

    def test_unsupported_schema_rejected(self):
        """Unknown or unimplemented study plans must be rejected, not replaced with defaults."""
        unknown_plan = {
            "plan_type": "declarative_study_plan",
            "study_name": "unimplemented_synthetic_experiment",
        }
        with self.assertRaises(NotImplementedError) as ctx:
            identify_plan_schema(unknown_plan)
        self.assertIn("refusing to silently substitute a default", str(ctx.exception))

    def test_mandatory_audit_command_generated_with_dataset(self):
        """When dataset preparation is specified, mandatory audit command with --hashes is generated."""
        plan = {
            "plan_type": "declarative_study_plan",
            "dataset": {"target_real_train": 1000, "target_fake_train": 1000},
            "stages": {"stage1_rgb": {}, "stage2_wavelet": {}, "stage3_fusion": {}},
            "seeds": [42],
        }
        cmds = generate_pipeline_commands(plan)
        self.assertGreater(len(cmds), 2)
        # Check first command is pilot preparation
        self.assertIn("prepare_dataset.py pilot", cmds[0])
        # Check second command is mandatory audit with --hashes
        self.assertIn("prepare_dataset.py audit", cmds[1])
        self.assertIn("--hashes", cmds[1])

    def test_multi_stage_parameter_wiring_contract(self):
        """Verifies embed_dim, pretrained, and wavelet_level are forwarded across all relevant stages."""
        plan = {
            "plan_type": "declarative_study_plan",
            "training": {
                "batch_size": 24,
                "grad_accum_steps": 3,
                "epochs": 5,
                "learning_rate": 0.0002,
                "num_workers": 2,
                "use_amp": True,
            },
            "stages": {
                "stage1_rgb": {"arch": "Wang2020_128", "pretrained": False, "embed_dim": 256},
                "stage2_wavelet": {"arch": "WolterWavelet2021_128", "wavelet_level": 4, "embed_dim": 256},
                "stage3_fusion": {"arch": "MHA_128", "fusion_candidates": ["token_attention"], "embed_dim": 256},
            },
            "seeds": [99],
        }
        cmds = generate_pipeline_commands(plan)
        full_text = "\n".join(cmds)

        # Pretrained=False forwards --no-pretrained
        self.assertIn("--no-pretrained", full_text)
        # embed_dim=256 forwarded to all stages
        self.assertEqual(full_text.count("--embed_dim 256"), 3)
        # wavelet_level=4 forwarded to stage 2 and stage 3
        self.assertIn("--wavelet_level 4", full_text)
        self.assertEqual(full_text.count("--wavelet_level 4"), 2)
        # training args forwarded
        self.assertIn("--batch_size 24", full_text)
        self.assertIn("--grad_accum_steps 3", full_text)
        self.assertIn("--num_workers 2", full_text)
        self.assertIn("--seed 99", full_text)

    def test_multi_seed_comparison_schema_wiring(self):
        """multi_seed_comparison.json must generate shared experts and comparison heads per seed."""
        plan = {
            "plan_type": "declarative_study_plan",
            "seeds": [42, 43],
            "models_to_compare": [
                {"model_id": "token_attention", "arch": "MHA_128", "fusion_type": "token_attention"},
                {"model_id": "late_ensemble", "arch": "Ensemble", "fusion_type": "ensemble"},
            ],
        }
        cmds = generate_pipeline_commands(plan)
        full_text = "\n".join(cmds)

        # Stage 1 and Stage 2 experts generated for both seed42 and seed43
        self.assertIn("stage1_rgb_expert_seed42", full_text)
        self.assertIn("stage1_rgb_expert_seed43", full_text)
        self.assertIn("stage2_wavelet_expert_seed42", full_text)
        self.assertIn("stage2_wavelet_expert_seed43", full_text)

        # Token attention head generated for both seeds
        self.assertIn("stage3_token_attention", full_text)
        # Late ensemble evaluation command generated for both seeds
        self.assertIn("ensemble_seed42_dev_predictions.csv", full_text)
        self.assertIn("ensemble_seed43_dev_predictions.csv", full_text)

    def test_targeted_ablations_schema_wiring(self):
        """ablations_plan.json dimensions must generate targeted parameter sweeps."""
        plan = {
            "plan_type": "declarative_study_plan",
            "ablations": [
                {"dimension": "wavelet_level", "values": [2, 4], "fixed_settings": {"embed_dim": 128}},
                {"dimension": "embedding_dimension", "values": [64, 256]},
            ],
        }
        cmds = generate_pipeline_commands(plan)
        full_text = "\n".join(cmds)

        # Wavelet level sweep generates level 2 and level 4 commands
        self.assertIn("--wavelet_level 2", full_text)
        self.assertIn("--wavelet_level 4", full_text)
        self.assertIn("stage2_wavelet_wav_level2", full_text)
        self.assertIn("stage2_wavelet_wav_level4", full_text)

        # Embedding dimension sweep generates dim 64 and dim 256 commands
        self.assertIn("--embed_dim 64", full_text)
        self.assertIn("--embed_dim 256", full_text)
        self.assertIn("stage1_rgb_dim64", full_text)
        self.assertIn("stage1_rgb_dim256", full_text)


if __name__ == '__main__':
    unittest.main()
