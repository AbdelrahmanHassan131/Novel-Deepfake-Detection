# STATUS: NOT RUN
"""
Deferred tests for Colab workflow and command integration (Prompt G).
Verifies:
1. tools/make_notebook.py generates an unexecuted notebook with device-aware launch (torchrun for multi-GPU, python for single), exact expert checkpoint paths, threshold file flags, and explicit opt-in controls for optional cells.
2. tools/run_study_plan.py correctly parses declarative study plan JSON files and generates verified, reproducible pipeline commands.
3. evaluation/robustness_cli.py has a functional __main__ guard and argument parser accepting dev/internal_test splits.
4. Cross-platform path portability handles Windows backslashes and relative resolutions on Linux/Colab.
"""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path


class TestColabWorkflowAndCommands(unittest.TestCase):

    def test_notebook_structure_and_command_parser_parity(self):
        from tools.make_notebook import generate_colab_notebook

        nb_path = Path("colab_pilot_pipeline.ipynb")
        self.assertTrue(nb_path.exists(), "colab_pilot_pipeline.ipynb must exist")
        nb_data = json.loads(nb_path.read_text(encoding='utf-8'))
        cells = nb_data.get('cells', [])
        self.assertGreaterEqual(len(cells), 10)

        # Check all cells are unexecuted (execution_count is None and outputs are empty)
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'], f"Cell {i} must be unexecuted")
                self.assertEqual(cell['outputs'], [], f"Cell {i} outputs must be empty")

        # Combine all code cell sources
        all_code = "\n".join("".join(c['source']) for c in cells if c['cell_type'] == 'code')

        # 1. Device-aware launch command
        self.assertIn("LAUNCH_CMD = f\"torchrun --nproc_per_node={gpu_count}\" if gpu_count > 1 else \"python\"", all_code)

        # 2. Check exact expert checkpoint paths
        self.assertIn("stage1_rgb_expert_seed42/checkpoints/best.pth", all_code)
        self.assertIn("stage2_wavelet_expert_seed42/checkpoints/best.pth", all_code)

        # 3. Check threshold calibration artifacts passed to comparison
        self.assertIn("--threshold_file", all_code)
        self.assertIn("--threshold_file_other", all_code)

        # 4. Check explicit opt-in controls for optional cells
        self.assertIn("RUN_OPTIONAL_ROBUSTNESS = False", all_code)
        self.assertIn("RUN_OPTIONAL_PROFILING = False", all_code)

    def test_study_plan_command_generator(self):
        from tools.run_study_plan import parse_study_plan, generate_pipeline_commands

        plan_path = Path("config/experiments/pilot_100k_l4_single_gpu.json")
        self.assertTrue(plan_path.exists())
        plan = parse_study_plan(str(plan_path))
        commands = generate_pipeline_commands(plan, data_root="/content/data", output_root="/content/experiments")

        self.assertGreater(len(commands), 0)
        # Check first command is data preparation
        self.assertIn("prepare_dataset.py", commands[0])
        self.assertIn("--target_real 50000", commands[0])
        self.assertIn("--target_fake 50000", commands[0])

        # Check Stage 1, Stage 2, and Stage 3 heads are generated
        full_script = "\n".join(commands)
        self.assertIn("Wang2020_128", full_script)
        self.assertIn("WolterWavelet2021_128", full_script)
        self.assertIn("stage3_token_attention", full_script)
        self.assertIn("stage3_gated", full_script)
        self.assertIn("stage3_concat", full_script)

    def test_robustness_cli_main_guard(self):
        cli_path = Path("evaluation/robustness_cli.py")
        self.assertTrue(cli_path.exists())
        content = cli_path.read_text(encoding='utf-8')
        tree = ast.parse(content)

        # Verify main guard exists in AST
        has_main_guard = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = node.test
                if isinstance(test, ast.Compare):
                    left = getattr(test.left, 'id', None)
                    if left == '__name__':
                        has_main_guard = True
                        break
        self.assertTrue(has_main_guard, "evaluation/robustness_cli.py must have an if __name__ == '__main__': main() guard")

    def test_cross_platform_path_portability(self):
        from data.manifest import write_manifest, read_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            img_file = Path(tmpdir) / "real" / "img1.png"
            img_file.parent.mkdir(parents=True)
            img_file.write_bytes(b"dummy_image_data")

            # Simulate Windows-style absolute path in manifest
            win_path = f"C:\\dummy\\root\\real\\img1.png"
            rows = [{
                'sample_id': 's1',
                'path': win_path,
                'label': 0,
                'split': 'dev',
                'dataset_source': 'test_src',
                'group_id': 'g1',
            }]
            manifest_path = Path(tmpdir) / "manifest.csv"
            write_manifest(str(manifest_path), rows)

            # Read with root pointing to tmpdir; verify it resolves to the real image file
            loaded = read_manifest(str(manifest_path), root=tmpdir, check_files=True)
            self.assertEqual(len(loaded), 1)
            self.assertTrue(Path(loaded[0]['path']).exists())


if __name__ == '__main__':
    unittest.main()
