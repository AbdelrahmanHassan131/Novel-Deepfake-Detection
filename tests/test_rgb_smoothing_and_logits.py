import os
import json
import math
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace

from config.defaults import MODEL_DEFAULTS
from config.configuration import ModelConfig, Config
from config.compatibility import config_from_opt, config_to_opt
from config.validator import ConfigValidator
from models.wang2020_128.trainer import Wang2020_128Trainer
from evaluation.generalization import (
    binary_metrics, compute_interpolated_eer, calibrate,
    export_predictions, read_predictions, group_intervals, summarize,
    compute_binary_predictions, load_validated_threshold, compare_evaluation_runs
)
from evaluation.metrics import ClassificationMetrics
from evaluation.visualization.confusion_matrix import plot_evaluation_confusion_matrix
from analyze_predictions import load_threshold
from training.hooks.checkpoint_hook import CheckpointHook
from PIL import Image


class TestRgbSmoothingConfigAndValidation(unittest.TestCase):
    def test_default_smoothing(self):
        self.assertEqual(MODEL_DEFAULTS.get('rgb_label_smoothing'), 0.0)
        cfg = ModelConfig()
        self.assertEqual(cfg.rgb_label_smoothing, 0.0)

    def test_opt_round_trip(self):
        class MockOpt:
            arch = 'Wang2020_128'
            rgb_label_smoothing = 0.05
            name = 'test'
            checkpoints_dir = './checkpoints'

        cfg = config_from_opt(MockOpt())
        self.assertEqual(cfg.model.rgb_label_smoothing, 0.05)
        opt_dict = config_to_opt(cfg)
        self.assertEqual(opt_dict.rgb_label_smoothing, 0.05)

    def test_validation_rules(self):
        from config.validator import ValidationReport
        # Valid smoothing
        cfg = Config()
        cfg.model.architecture = 'Wang2020_128'
        cfg.model.rgb_label_smoothing = 0.05
        report = ValidationReport()
        ConfigValidator._check_model(cfg, report)
        self.assertTrue(report.is_valid)

        # Negative smoothing
        cfg.model.rgb_label_smoothing = -0.01
        report = ValidationReport()
        ConfigValidator._check_model(cfg, report)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("must be in [0.0, 0.5)" in e for e in report.errors))

        # Smoothing >= 0.5
        cfg.model.rgb_label_smoothing = 0.5
        report = ValidationReport()
        ConfigValidator._check_model(cfg, report)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("must be in [0.0, 0.5)" in e for e in report.errors))

        # NaN smoothing
        cfg.model.rgb_label_smoothing = float('nan')
        report = ValidationReport()
        ConfigValidator._check_model(cfg, report)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("finite" in e for e in report.errors))

        # Non-Wang architecture with nonzero smoothing
        cfg.model.architecture = 'CNNDetection'
        cfg.model.rgb_label_smoothing = 0.05
        report = ValidationReport()
        ConfigValidator._check_model(cfg, report)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("only supported for architecture 'Wang2020_128'" in e for e in report.errors))


class TestWangTrainerSmoothingMath(unittest.TestCase):
    def _create_mock_trainer(self, smoothing=0.0):
        class MockOpt:
            isTrain = True
            arch = 'Wang2020_128'
            rgb_head_type = '128d'
            rgb_dropout = 0.0
            rgb_label_smoothing = smoothing
            fine_tune_policy = 'full'
            bn_policy = 'train'
            backbone_lr_mult = 1.0
            lr = 0.001
            beta1 = 0.9
            weight_decay = 0.0
            optim = 'adam'
            pretrained = False
            continue_train = False
            _loading_checkpoint = True

        trainer = Wang2020_128Trainer(MockOpt())
        return trainer

    def test_smoothing_math_training_mode(self):
        # Asymmetric non-zero logits: 4 samples
        z = torch.tensor([[2.0], [-1.0], [0.5], [-3.0]], dtype=torch.float32)
        y = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

        # 1. Test s = 0.0 matches standard BCE and analytical gradient
        trainer_s0 = self._create_mock_trainer(smoothing=0.0)
        trainer_s0.train()
        trainer_s0.output = z.clone().detach().requires_grad_(True)
        trainer_s0.label = y.clone()
        loss_s0 = trainer_s0._compute_loss()
        loss_s0.backward()

        expected_loss_s0 = nn.BCEWithLogitsLoss()(z, y)
        self.assertAlmostEqual(loss_s0.item(), expected_loss_s0.item(), places=5)
        # Analytical gradient dL/dz = (sigma(z) - y) / N
        expected_grad_s0 = (torch.sigmoid(z) - y) / 4.0
        self.assertTrue(torch.allclose(trainer_s0.output.grad, expected_grad_s0, atol=1e-5))

        # 2. Test s = 0.05
        trainer_s05 = self._create_mock_trainer(smoothing=0.05)
        trainer_s05.train()
        trainer_s05.output = z.clone().detach().requires_grad_(True)
        original_label_clone = y.clone()
        trainer_s05.label = y.clone()
        loss_s05 = trainer_s05._compute_loss()
        loss_s05.backward()

        # Smoothed targets: y_smooth = y * (1 - 2*0.05) + 0.05 = y * 0.9 + 0.05
        y_smooth = y * 0.9 + 0.05
        expected_loss_s05 = nn.BCEWithLogitsLoss()(z, y_smooth)
        self.assertAlmostEqual(loss_s05.item(), expected_loss_s05.item(), places=5)
        # Analytical gradient dL/dz = (sigma(z) - y_smooth) / N
        expected_grad_s05 = (torch.sigmoid(z) - y_smooth) / 4.0
        self.assertTrue(torch.allclose(trainer_s05.output.grad, expected_grad_s05, atol=1e-5))

        # Label tensor immutability: trainer.label was not modified in-place
        self.assertTrue(torch.equal(trainer_s05.label, original_label_clone))

        # 3. Active train/eval mode transitions on same instance
        trainer_s05.eval()
        trainer_s05.output = z.clone()
        trainer_s05.label = y.clone()
        eval_loss = trainer_s05._compute_loss()
        self.assertAlmostEqual(eval_loss.item(), expected_loss_s0.item(), places=5)

        # Switch back to train mode
        trainer_s05.train()
        train_loss_again = trainer_s05._compute_loss()
        self.assertAlmostEqual(train_loss_again.item(), expected_loss_s05.item(), places=5)

    def test_optimizer_step_updates_head_parameters(self):
        trainer = self._create_mock_trainer(smoothing=0.05)
        trainer.train()
        initial_params = [p.clone().detach() for p in trainer.model.fc.parameters()]
        dummy_input = torch.randn(2, 3, 128, 128)
        dummy_label = torch.tensor([[1.0], [0.0]])
        trainer.set_input((dummy_input, dummy_label))
        trainer.optimize_parameters()
        updated_params = list(trainer.model.fc.parameters())
        any_changed = any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, updated_params))
        self.assertTrue(any_changed, "Head parameters should update after optimizer step")


class TestScoreAwareMetricsAndSaturationFixture(unittest.TestCase):
    def test_saturated_probability_vs_logit_ranking(self):
        # Prompt 4 regression fixture:
        # real logits [19, 20, 21], fake logits [27, 28, 29]
        real_z = [19.0, 20.0, 21.0]
        fake_z = [27.0, 28.0, 29.0]
        logits = np.array(real_z + fake_z, dtype=np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

        # In float32 sigmoid:
        # 1 / (1 + exp(-19)) = 1.0 (in float32 precision)
        # 1 / (1 + exp(-27)) = 1.0 (in float32 precision)
        probs = 1.0 / (1.0 + np.exp(-logits))
        self.assertTrue(np.all(probs >= 1.0 - 1e-7))

        # Under probability space, ties make AUC 0.5
        metrics_p = binary_metrics(labels, probs, threshold=0.5, score_space='probability')
        self.assertAlmostEqual(metrics_p['roc_auc_probability'], 0.5, places=5)

        # Under logit space, real < fake strictly: logit AUC must be 1.0!
        metrics_z = binary_metrics(labels, probs, threshold=0.5, logits=logits, score_space='probability')
        self.assertAlmostEqual(metrics_z['roc_auc_logit'], 1.0, places=5)
        self.assertAlmostEqual(metrics_z['roc_auc'], 1.0, places=5)
        self.assertEqual(metrics_z['ranking_score_space'], 'logit')

        # Logit threshold 27 gives perfect separation:
        # real logits < 27 -> pred 0
        # fake logits >= 27 -> pred 1
        metrics_thresh27 = binary_metrics(labels, probs, threshold=27.0, logits=logits, score_space='logit')
        self.assertEqual(metrics_thresh27['accuracy'], 1.0)
        self.assertEqual(metrics_thresh27['real_recall'], 1.0)
        self.assertEqual(metrics_thresh27['fake_recall'], 1.0)

        # Logit threshold 0 gives all fake (since all logits >= 19 > 0)
        metrics_thresh0 = binary_metrics(labels, probs, threshold=0.0, logits=logits, score_space='logit')
        self.assertEqual(metrics_thresh0['accuracy'], 0.5)
        self.assertEqual(metrics_thresh0['real_recall'], 0.0)
        self.assertEqual(metrics_thresh0['fake_recall'], 1.0)

    def test_continuous_eer_interpolation(self):
        # Construct synthetic curve with known crossing
        # y: 10 real, 10 fake
        y = np.array([0]*10 + [1]*10)
        # scores slightly overlapping
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
        eer, fpr, tpr, _ = compute_interpolated_eer(y, scores)
        self.assertIsNotNone(eer)
        self.assertGreater(eer, 0.0)
        self.assertLess(eer, 1.0)

    def test_single_class_metrics_undefined(self):
        y = np.array([1, 1, 1, 1])
        probs = np.array([0.8, 0.9, 0.7, 0.85])
        logits = np.array([1.3, 2.1, 0.8, 1.7])
        m = binary_metrics(y, probs, threshold=0.5, logits=logits)
        self.assertIsNone(m['roc_auc'])
        self.assertIsNone(m['balanced_accuracy'])
        self.assertIsNone(m['eer'])
        self.assertEqual(m['status'], 'single_class_evaluated_ranking_metrics_undefined')

    def test_saturation_fixture_endpoint_agreement(self):
        # Saturated predictions fixture from review
        real_z = [20.0]
        fake_z = [30.0]
        logits = np.array(real_z + fake_z, dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        probs = np.array([1.0, 1.0], dtype=np.float32)

        # 1. binary_metrics at probability threshold 1.0
        m_bin = binary_metrics(labels, probs, threshold=1.0, logits=logits)
        self.assertEqual(m_bin['confusion_matrix'], [[1, 0], [1, 0]])

        # 2. ClassificationMetrics at probability threshold 1.0
        cm_calc = ClassificationMetrics(threshold=1.0, score_space='probability')
        m_cm = cm_calc.compute_all(probabilities=probs, labels=labels, logits=logits)
        self.assertEqual(m_cm['confusion_matrix'], [[1, 0], [1, 0]])

        # 3. Decision routine direct check
        pred, space = compute_binary_predictions(probabilities=probs, logits=logits, threshold=1.0, score_space='probability')
        self.assertTrue(np.all(pred == 0))
        self.assertEqual(space, 'logit')

        # 4. Evaluator binarization agreement
        self.assertEqual(m_bin['confusion_matrix'], m_cm['confusion_matrix'])

    def test_logit_threshold_inside_zero_one(self):
        logits = np.array([-0.2, 0.4, 0.6, 1.2], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        probs = 1.0 / (1.0 + np.exp(-logits))

        # threshold 0.5 with score_space='logit' evaluates logits >= 0.5
        m = binary_metrics(labels, probs, threshold=0.5, logits=logits, score_space='logit')
        self.assertEqual(m['accuracy'], 1.0)
        self.assertEqual(m['decision_score_space'], 'logit')

    def test_video_aggregation_with_logit_threshold(self):
        records = [
            {'sample_id': '0', 'label': 0, 'source_video_id': 'v1', 'dataset_source': 'ds'},
            {'sample_id': '1', 'label': 0, 'source_video_id': 'v1', 'dataset_source': 'ds'},
            {'sample_id': '2', 'label': 1, 'source_video_id': 'v2', 'dataset_source': 'ds'},
            {'sample_id': '3', 'label': 1, 'source_video_id': 'v2', 'dataset_source': 'ds'},
        ]
        logits = np.array([24.0, 26.0, 28.0, 30.0], dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))
        rep = summarize(records, probs, threshold=25.5, repeats=0, logits=logits, score_space='logit')
        self.assertIn('video_level', rep)
        self.assertEqual(rep['video_level']['status'], 'unavailable: image-level logit threshold cannot be applied to mean-probability video aggregates without video-level calibration')

    def test_invalid_score_space_string_rejected(self):
        y = np.array([0, 1])
        p = np.array([0.2, 0.8])
        with self.assertRaises(ValueError):
            binary_metrics(y, p, threshold=0.5, score_space='invalid_space')
        with self.assertRaises(ValueError):
            compute_binary_predictions(p, threshold=0.5, score_space='invalid_space')


class TestCalibrationAndArtifactLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_logit_calibration_and_loading(self):
        pred_csv = self.dir_path / 'dev_preds.csv'
        thresh_json = self.dir_path / 'threshold_logit.json'

        # Create valid dev predictions with logits
        records = [
            {'sample_id': f's{i}', 'label': 0 if i < 10 else 1, 'split': 'dev', 'group_id': f'g{i}'}
            for i in range(20)
        ]
        # Real logits around -2.0, Fake logits around +2.0
        logits = np.array([-2.5 + 0.1*i for i in range(10)] + [1.5 + 0.1*i for i in range(10)])
        probs = 1.0 / (1.0 + np.exp(-logits))
        ckpt_hash = 'a' * 64
        export_predictions(str(pred_csv), records, probs, checkpoint_hash=ckpt_hash, logits=logits, eval_precision='fp32')

        # Calibrate in logit space
        artifact = calibrate(str(pred_csv), str(thresh_json), score_space='logit', eval_precision='fp32')
        self.assertEqual(artifact['score_space'], 'logit')
        self.assertIn('logit_threshold', artifact)
        self.assertEqual(artifact['checkpoint_sha256'], ckpt_hash)

        # Load threshold
        val, space = load_threshold(None, str(thresh_json), expected_checkpoint_hash=ckpt_hash, return_score_space=True)
        self.assertEqual(space, 'logit')
        self.assertEqual(val, artifact['logit_threshold'])

        # Mismatch checkpoint hash
        with self.assertRaises(ValueError):
            load_threshold(None, str(thresh_json), expected_checkpoint_hash='b' * 64)

    def test_colab_logit_calibration_legacy_artifact(self):
        # Schema from COLAB_LOGIT_CALIBRATION.md
        legacy_json = self.dir_path / 'colab_legacy.json'
        ckpt_hash = 'c' * 64
        content = {
            'checkpoint_sha256': ckpt_hash,
            'calibration_prediction_sha256': 'd' * 64,
            'score_space': 'logit',
            'criterion': 'external calibration Youden J; first maximum on descending ROC thresholds',
            'label_mapping': {'real': 0, 'fake': 1},
            'decision_rule': 'fake if logit >= logit_threshold',
            'calibration_status': 'previously_inspected_external_data_not_independent_test',
            'logit_threshold': 25.587146759033203,
        }
        legacy_json.write_text(json.dumps(content), encoding='utf-8')

        val, space = load_threshold(None, str(legacy_json), expected_checkpoint_hash=ckpt_hash, return_score_space=True)
        self.assertEqual(space, 'logit')
        self.assertAlmostEqual(val, 25.587146759033203, places=6)

    def test_forbidden_calibration_split(self):
        pred_csv = self.dir_path / 'test_preds.csv'
        thresh_json = self.dir_path / 'thresh_forbidden.json'
        records = [
            {'sample_id': f's{i}', 'label': 0 if i < 5 else 1, 'split': 'test', 'group_id': f'g{i}'}
            for i in range(10)
        ]
        probs = [0.1]*5 + [0.9]*5
        export_predictions(str(pred_csv), records, probs, checkpoint_hash='e'*64)

        with self.assertRaises(ValueError) as ctx:
            calibrate(str(pred_csv), str(thresh_json), score_space='probability')
        self.assertIn("Forbidden", str(ctx.exception))

    def test_forbidden_split_cannot_be_bypassed_by_legacy_tag(self):
        bad_json = self.dir_path / 'bad_bypass.json'
        content = {
            'checkpoint_sha256': 'a' * 64,
            'calibration_status': 'previously_inspected_external_data_not_independent_test',
            'source_splits': ['external_test'],
            'score_space': 'logit',
            'logit_threshold': 25.0,
            'threshold': -999.0,
        }
        bad_json.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaises(ValueError) as ctx:
            load_validated_threshold(str(bad_json), expected_checkpoint_hash='a' * 64)
        self.assertIn("Forbidden", str(ctx.exception))

    def test_conflicting_threshold_fields_rejected(self):
        conflict_json = self.dir_path / 'conflict.json'
        content = {
            'checkpoint_sha256': 'a' * 64,
            'source_splits': ['dev'],
            'score_space': 'logit',
            'logit_threshold': 25.0,
            'threshold': 10.0,
        }
        conflict_json.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaises(ValueError) as ctx:
            load_validated_threshold(str(conflict_json), expected_checkpoint_hash='a' * 64)
        self.assertIn("Conflicting threshold fields", str(ctx.exception))

    def test_invalid_label_mapping_rejected(self):
        bad_map_json = self.dir_path / 'bad_map.json'
        content = {
            'checkpoint_sha256': 'a' * 64,
            'source_splits': ['dev'],
            'score_space': 'logit',
            'logit_threshold': 0.5,
            'label_mapping': {'real': 1, 'fake': 0},
        }
        bad_map_json.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaises(ValueError) as ctx:
            load_validated_threshold(str(bad_map_json), expected_checkpoint_hash='a' * 64)
        self.assertIn("label_mapping", str(ctx.exception))

    def test_unknown_score_space_rejected(self):
        bad_space_json = self.dir_path / 'bad_space.json'
        content = {
            'checkpoint_sha256': 'a' * 64,
            'source_splits': ['dev'],
            'score_space': 'logits',
            'threshold': 0.5,
        }
        bad_space_json.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaises(ValueError) as ctx:
            load_validated_threshold(str(bad_space_json), expected_checkpoint_hash='a' * 64)
        self.assertIn("Unknown score_space", str(ctx.exception))

    def test_precision_mismatch_rejected(self):
        prec_json = self.dir_path / 'prec_mismatch.json'
        content = {
            'checkpoint_sha256': 'a' * 64,
            'source_splits': ['dev'],
            'score_space': 'logit',
            'logit_threshold': 0.5,
            'eval_precision': 'fp16',
        }
        prec_json.write_text(json.dumps(content), encoding='utf-8')
        with self.assertRaises(ValueError) as ctx:
            load_validated_threshold(str(prec_json), expected_checkpoint_hash='a' * 64, eval_precision='fp32')
        self.assertIn("fp32", str(ctx.exception))


class TestRescoreAndComparison(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_analyze_predictions_rescore_cli(self):
        import subprocess
        pred_csv = self.dir_path / 'ext_preds.csv'
        thresh_json = self.dir_path / 'frozen_thresh.json'
        output_json = self.dir_path / 'rescore.json'

        ckpt_hash = 'f' * 64
        records = [
            {'sample_id': f'img_{i}', 'label': 0 if i < 10 else 1, 'split': 'external_dev', 'group_id': f'grp_{i}'}
            for i in range(20)
        ]
        logits = np.array([-3.0 + 0.2*i for i in range(10)] + [1.0 + 0.2*i for i in range(10)])
        probs = 1.0 / (1.0 + np.exp(-logits))
        export_predictions(str(pred_csv), records, probs, checkpoint_hash=ckpt_hash, logits=logits, eval_precision='fp32')

        # Create calibrated threshold
        thresh_data = {
            'schema_version': '2.0',
            'score_space': 'logit',
            'logit_threshold': 0.5,
            'threshold': 0.5,
            'checkpoint_sha256': ckpt_hash,
            'source_splits': ['dev'],
            'eval_precision': 'fp32',
        }
        thresh_json.write_text(json.dumps(thresh_data), encoding='utf-8')

        # Run rescore action via command line
        cmd = [
            'python', 'analyze_predictions.py', 'rescore',
            '--predictions', str(pred_csv),
            '--threshold_file', str(thresh_json),
            '--output', str(output_json)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertTrue(output_json.is_file())

        data = json.loads(output_json.read_text(encoding='utf-8'))
        self.assertIn('fixed_05', data)
        self.assertIn('frozen_threshold', data)
        self.assertEqual(data['frozen_threshold']['score_space'], 'logit')
        self.assertEqual(data['frozen_threshold']['threshold'], 0.5)


class TestFolderValidationAndCalibrationIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_folder_validation_export_and_logit_calibration(self):
        # 1. Create a minimal folder dataset structure
        val_root = self.dir_path / 'val'
        (val_root / 'real').mkdir(parents=True)
        (val_root / 'fake').mkdir(parents=True)

        for i in range(5):
            img = Image.new('RGB', (16, 16), color=(i * 10, i * 10, i * 10))
            img.save(val_root / 'real' / f'real_{i}.png')
        for i in range(5):
            img = Image.new('RGB', (16, 16), color=(200, i * 10, i * 10))
            img.save(val_root / 'fake' / f'fake_{i}.png')

        from data.datasets.rgb_dataset import RGBDataset
        from types import SimpleNamespace
        opt = SimpleNamespace(isTrain=False, arch='Wang2020_128', input_size=16)
        dataset = RGBDataset(opt, str(val_root))

        # In folder mode, base_dataset records start with split='unknown'
        self.assertEqual(dataset.records[0]['split'], 'unknown')
        self.assertEqual(len(dataset.records), 10)

        # 2. Setup mock trainer with CheckpointHook
        ckpt_dir = self.dir_path / 'checkpoints'
        ckpt_dir.mkdir(parents=True)
        dummy_best = ckpt_dir / 'best.pth'
        dummy_best.write_bytes(b'fake_weights_for_test')
        from data.manifest import sha256
        best_hash = sha256(str(dummy_best))

        class MockCheckpointManager:
            save_dir = str(ckpt_dir)
            rank = 0
            def save_best(self, **kwargs):
                pass
            def save_last(self, **kwargs):
                pass
            def save_epoch(self, **kwargs):
                pass

        hook = CheckpointHook(checkpoint_manager=MockCheckpointManager(), monitor_metric='auc')

        labels = np.array([r['label'] for r in dataset.records], dtype=np.int64)
        logits = np.where(labels == 1, 2.0, -2.0).astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))

        result = SimpleNamespace(
            accuracy=1.0,
            auc=1.0,
            predictions=probs,
            logits=logits,
            labels=labels,
            indices=list(range(10)),
            eval_precision='fp32',
        )

        trainer = SimpleNamespace(
            val_loader=SimpleNamespace(dataset=dataset),
            best_metric=None,
            current_epoch=1,
            global_step=10,
            scheduler=None,
            opt=SimpleNamespace(run_id='folder_test_run', seed=42, name='folder_test'),
        )

        # Trigger hook validation recording and epoch save
        hook.on_epoch_start(trainer)
        hook.on_validation_end(trainer, result)
        hook.on_epoch_end(trainer)

        # 3. Verify authoritative dev predictions exported with role='dev' and unverified status
        exported_csv = ckpt_dir / 'best_dev_predictions.csv'
        self.assertTrue(exported_csv.is_file())
        rows = read_predictions(str(exported_csv))
        self.assertEqual(len(rows), 10)
        for r in rows:
            self.assertEqual(r['split'], 'dev')
            self.assertEqual(r['folder_role'], 'dev')
            self.assertEqual(r['independence_status'], 'unverified_dev_folder')
            self.assertEqual(r['checkpoint_sha256'], best_hash)
            self.assertEqual(r['eval_precision'], 'fp32')
            self.assertIn('logit', r)

        # 4. Calibrate logit threshold on exported file - must succeed without split errors!
        thresh_json = ckpt_dir / 'threshold_logit.json'
        cal_art = calibrate(str(exported_csv), str(thresh_json), score_space='logit', eval_precision='fp32')
        self.assertEqual(cal_art['score_space'], 'logit')
        self.assertEqual(cal_art['checkpoint_sha256'], best_hash)
        self.assertEqual(cal_art['eval_precision'], 'fp32')

    def test_negative_manifest_test_split_preserved_and_rejected(self):
        # A test-split manifest dataset passed to checkpoint hook must retain split='test'
        ckpt_dir = self.dir_path / 'checkpoints_neg'
        ckpt_dir.mkdir(parents=True)
        dummy_best = ckpt_dir / 'best.pth'
        dummy_best.write_bytes(b'fake_weights_neg')
        from data.manifest import sha256
        best_hash = sha256(str(dummy_best))

        class MockCheckpointManager:
            save_dir = str(ckpt_dir)
            rank = 0
            def save_best(self, **kwargs):
                pass
            def save_last(self, **kwargs):
                pass
            def save_epoch(self, **kwargs):
                pass

        hook = CheckpointHook(checkpoint_manager=MockCheckpointManager(), monitor_metric='auc')

        records = [
            {'sample_id': str(i), 'path': f'test_{i}.png', 'label': 0 if i < 5 else 1, 'split': 'test'}
            for i in range(10)
        ]
        class MockManifestDataset:
            pass
        ds = MockManifestDataset()
        ds.records = records

        logits = np.array([-1.0]*5 + [1.0]*5, dtype=np.float32)
        probs = 1.0 / (1.0 + np.exp(-logits))
        labels = np.array([0]*5 + [1]*5, dtype=np.int64)

        result = SimpleNamespace(
            accuracy=1.0,
            auc=1.0,
            predictions=probs,
            logits=logits,
            labels=labels,
            indices=list(range(10)),
            eval_precision='fp32',
        )

        trainer = SimpleNamespace(
            val_loader=SimpleNamespace(dataset=ds),
            best_metric=None,
            current_epoch=1,
            global_step=10,
            scheduler=None,
            opt=SimpleNamespace(run_id='test_neg_run', seed=42, name='test_neg'),
        )

        hook.on_epoch_start(trainer)
        hook.on_validation_end(trainer, result)
        hook.on_epoch_end(trainer)

        exported_csv = ckpt_dir / 'best_dev_predictions.csv'
        self.assertTrue(exported_csv.is_file())
        rows = read_predictions(str(exported_csv))
        # Ensure test split was NOT mutated into dev
        for r in rows:
            self.assertEqual(r['split'], 'test')

        # Calibrate must reject test split
        thresh_json = ckpt_dir / 'thresh.json'
        with self.assertRaises(ValueError) as ctx:
            calibrate(str(exported_csv), str(thresh_json), score_space='logit')
        self.assertIn("Forbidden", str(ctx.exception))


class TestComparisonAndIdentitySecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def _create_matched_runs(self):
        ctrl_ckpt = self.dir_path / 'ctrl_best.pth'
        smth_ckpt = self.dir_path / 'smth_best.pth'
        ctrl_ckpt.write_bytes(b'ctrl_model_bytes_12345')
        smth_ckpt.write_bytes(b'smth_model_bytes_67890')
        from data.manifest import sha256
        h_ctrl = sha256(str(ctrl_ckpt))
        h_smth = sha256(str(smth_ckpt))

        records_ctrl = [
            {'sample_id': f'img_{i}', 'path': f'/data/img_{i}.png', 'label': 0 if i < 10 else 1,
             'group_id': f'grp_{i}', 'split': 'external_dev', 'eval_precision': 'fp32', 'checkpoint_sha256': h_ctrl}
            for i in range(20)
        ]
        records_smth = [
            {'sample_id': f'img_{i}', 'path': f'/data/img_{i}.png', 'label': 0 if i < 10 else 1,
             'group_id': f'grp_{i}', 'split': 'external_dev', 'eval_precision': 'fp32', 'checkpoint_sha256': h_smth}
            for i in range(20)
        ]

        z_ctrl = np.array([-2.0 + 0.1*i for i in range(10)] + [1.0 + 0.1*i for i in range(10)], dtype=np.float32)
        p_ctrl = 1.0 / (1.0 + np.exp(-z_ctrl))

        z_smth = np.array([-2.5 + 0.1*i for i in range(10)] + [1.5 + 0.1*i for i in range(10)], dtype=np.float32)
        p_smth = 1.0 / (1.0 + np.exp(-z_smth))

        pred_ctrl_path = self.dir_path / 'pred_ctrl.csv'
        pred_smth_path = self.dir_path / 'pred_smth.csv'
        export_predictions(str(pred_ctrl_path), records_ctrl, p_ctrl, checkpoint_hash=h_ctrl, logits=z_ctrl, eval_precision='fp32')
        export_predictions(str(pred_smth_path), records_smth, p_smth, checkpoint_hash=h_smth, logits=z_smth, eval_precision='fp32')

        thresh_ctrl_path = self.dir_path / 'thresh_ctrl.json'
        thresh_smth_path = self.dir_path / 'thresh_smth.json'

        thresh_ctrl_data = {
            'schema_version': '2.0',
            'score_space': 'logit',
            'logit_threshold': 0.1,
            'threshold': 0.1,
            'checkpoint_sha256': h_ctrl,
            'source_splits': ['dev'],
            'eval_precision': 'fp32',
        }
        thresh_smth_data = {
            'schema_version': '2.0',
            'score_space': 'logit',
            'logit_threshold': 0.2,
            'threshold': 0.2,
            'checkpoint_sha256': h_smth,
            'source_splits': ['dev'],
            'eval_precision': 'fp32',
        }
        thresh_ctrl_path.write_text(json.dumps(thresh_ctrl_data), encoding='utf-8')
        thresh_smth_path.write_text(json.dumps(thresh_smth_data), encoding='utf-8')

        return (pred_ctrl_path, pred_smth_path, thresh_ctrl_path, thresh_smth_path,
                ctrl_ckpt, smth_ckpt, h_ctrl, h_smth)

    def test_swapped_thresholds_rejected(self):
        (p_ctrl, p_smth, t_ctrl, t_smth, c_ctrl, c_smth, h_ctrl, h_smth) = self._create_matched_runs()
        # Pass t_smth as control threshold (hash mismatch against control predictions)
        with self.assertRaises(ValueError) as ctx:
            compare_evaluation_runs(
                control_pred_path=str(p_ctrl),
                smooth_pred_path=str(p_smth),
                control_thresh_path=str(t_smth),  # Swapped!
                smooth_thresh_path=str(t_ctrl),
                control_checkpoint_path=str(c_ctrl),
                smooth_checkpoint_path=str(c_smth),
            )
        self.assertIn("does not match evaluation checkpoint", str(ctx.exception))

    def test_mixed_checkpoint_rows_rejected(self):
        (p_ctrl, p_smth, t_ctrl, t_smth, c_ctrl, c_smth, h_ctrl, h_smth) = self._create_matched_runs()
        # Tamper control predictions to have mixed checkpoint hashes
        rows = read_predictions(str(p_ctrl))
        rows[5]['checkpoint_sha256'] = 'different_hash_' + '0'*50
        export_predictions(str(p_ctrl), rows, [r['probability'] for r in rows],
                           checkpoint_hash=None, logits=[r['logit'] for r in rows], eval_precision='fp32')

        with self.assertRaises(ValueError) as ctx:
            compare_evaluation_runs(
                control_pred_path=str(p_ctrl),
                smooth_pred_path=str(p_smth),
                control_thresh_path=str(t_ctrl),
                smooth_thresh_path=str(t_smth),
            )
        self.assertIn("exactly one consistent, known checkpoint_sha256", str(ctx.exception))

    def test_different_paths_with_matching_sample_ids_rejected(self):
        (p_ctrl, p_smth, t_ctrl, t_smth, c_ctrl, c_smth, h_ctrl, h_smth) = self._create_matched_runs()
        # Tamper smoothing predictions to have a different path for sample_id 'img_3'
        rows = read_predictions(str(p_smth))
        rows[3]['path'] = '/other_folder/different_image.png'
        export_predictions(str(p_smth), rows, [r['probability'] for r in rows],
                           checkpoint_hash=h_smth, logits=[r['logit'] for r in rows], eval_precision='fp32')

        with self.assertRaises(ValueError) as ctx:
            compare_evaluation_runs(
                control_pred_path=str(p_ctrl),
                smooth_pred_path=str(p_smth),
                control_thresh_path=str(t_ctrl),
                smooth_thresh_path=str(t_smth),
            )
        self.assertIn("Sample path mismatch", str(ctx.exception))

    def test_successful_ab_comparison_with_digests(self):
        (p_ctrl, p_smth, t_ctrl, t_smth, c_ctrl, c_smth, h_ctrl, h_smth) = self._create_matched_runs()
        out_json = self.dir_path / 'comp.json'
        out_md = self.dir_path / 'comp.md'

        comp = compare_evaluation_runs(
            control_pred_path=str(p_ctrl),
            smooth_pred_path=str(p_smth),
            control_thresh_path=str(t_ctrl),
            smooth_thresh_path=str(t_smth),
            control_checkpoint_path=str(c_ctrl),
            smooth_checkpoint_path=str(c_smth),
            output_json_path=str(out_json),
            output_md_path=str(out_md),
            control_name='Arm A Control',
            smooth_name='Arm B Smoothing',
            bootstrap_repeats=50,
            seed=42,
            eval_precision='fp32',
        )

        self.assertEqual(comp['samples'], 20)
        self.assertIn('file_digests', comp)
        self.assertEqual(comp['file_digests']['control_checkpoint_sha256'], h_ctrl)
        self.assertEqual(comp['file_digests']['smooth_checkpoint_sha256'], h_smth)
        self.assertIn('paired_difference_smoothing_minus_control', comp)
        self.assertTrue(out_json.is_file())
        self.assertTrue(out_md.is_file())


class TestDistributedGatheringAlignment(unittest.TestCase):
    def test_padded_gather_with_empty_and_uneven_shards_preserves_logits_and_labels(self):
        # Simulate validator._gather_predictions padded gather across 3 ranks
        # Rank 0 has 3 samples: indices [0, 2, 4], logits [-1.5, 0.5, 2.5], labels [0, 1, 1]
        # Rank 1 has 0 samples: empty shard
        # Rank 2 has 2 samples: indices [1, 3], logits [-0.5, 1.5], labels [0, 1]
        lengths = [3, 0, 2]
        world_size = 3
        max_len = 3

        gathered_p = [torch.zeros(max_len), torch.zeros(max_len), torch.zeros(max_len)]
        gathered_z = [torch.zeros(max_len), torch.zeros(max_len), torch.zeros(max_len)]
        gathered_l = [torch.zeros(max_len), torch.zeros(max_len), torch.zeros(max_len)]
        gathered_i = [torch.zeros(max_len, dtype=torch.long), torch.zeros(max_len, dtype=torch.long), torch.zeros(max_len, dtype=torch.long)]

        # Fill Rank 0
        gathered_z[0][:3] = torch.tensor([-1.5, 0.5, 2.5])
        gathered_p[0][:3] = torch.sigmoid(gathered_z[0][:3])
        gathered_l[0][:3] = torch.tensor([0.0, 1.0, 1.0])
        gathered_i[0][:3] = torch.tensor([0, 2, 4])

        # Rank 1 is empty (lengths[1] = 0)

        # Fill Rank 2
        gathered_z[2][:2] = torch.tensor([-0.5, 1.5])
        gathered_p[2][:2] = torch.sigmoid(gathered_z[2][:2])
        gathered_l[2][:2] = torch.tensor([0.0, 1.0])
        gathered_i[2][:2] = torch.tensor([1, 3])

        # Slicing and concatenation on rank 0
        valid_p = [gathered_p[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]
        valid_z = [gathered_z[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]
        valid_l = [gathered_l[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]
        valid_i = [gathered_i[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]

        all_preds = np.concatenate(valid_p)
        all_logits = np.concatenate(valid_z)
        all_labels = np.concatenate(valid_l)
        all_indices = np.concatenate(valid_i)

        sort_order = np.argsort(all_indices)
        all_preds = all_preds[sort_order]
        all_logits = all_logits[sort_order]
        all_labels = all_labels[sort_order]
        all_indices = all_indices[sort_order]

        # Verify exact alignment
        np.testing.assert_array_equal(all_indices, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(all_logits, [-1.5, -0.5, 0.5, 1.5, 2.5], atol=1e-5)
        np.testing.assert_array_equal(all_labels, [0, 0, 1, 1, 1])
        np.testing.assert_allclose(all_preds, 1.0 / (1.0 + np.exp(-all_logits)), atol=1e-5)


class TestColabNotebookOrchestration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.test_dir.name)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_colab_markdown_python_cells_syntax(self):
        md_file = Path('COLAB_RGB_LABEL_SMOOTHING_AB.md')
        self.assertTrue(md_file.is_file())
        text = md_file.read_text(encoding='utf-8')

        lines = text.splitlines()
        in_py_block = False
        current_block = []
        py_blocks = []

        for line in lines:
            if line.strip() == '```python':
                in_py_block = True
                current_block = []
            elif line.strip() == '```' and in_py_block:
                in_py_block = False
                py_blocks.append('\n'.join(current_block))
            elif in_py_block:
                current_block.append(line)

        self.assertGreater(len(py_blocks), 0, 'No python code blocks found in notebook guide')
        for idx, block in enumerate(py_blocks):
            try:
                compile(block, f'<colab_py_cell_{idx}>', 'exec')
            except SyntaxError as e:
                self.fail(f"Syntax error in Colab cell {idx}: {e}")

    def test_colab_guide_flags_and_paths(self):
        md_file = Path('COLAB_RGB_LABEL_SMOOTHING_AB.md')
        text = md_file.read_text(encoding='utf-8')

        # Cell 1 must check --use_amp and --amp_dtype, and must NOT check --amp_train
        self.assertIn('--use_amp', text)
        self.assertIn('--amp_dtype', text)
        self.assertNotIn('--amp_train', text)

        # Cell 1 must insert REPO_ROOT into sys.path
        self.assertIn('sys.path.insert(0, str(REPO_ROOT))', text)

        # Cell 4 must calibrate from checkpoints/best_dev_predictions.csv
        self.assertIn('$RUN_ROOT_CONTROL/checkpoints/best_dev_predictions.csv', text)
        self.assertIn('$RUN_ROOT_SMOOTH/checkpoints/best_dev_predictions.csv', text)

    def test_colab_end_to_end_mocked_orchestration(self):
        # Execute the actual guide orchestration with mocked models
        output_base = self.dir_path / 'output'
        output_base.mkdir(parents=True)
        (output_base / 'launch_logs').mkdir()

        ctrl_root = output_base / 'ctrl_run'
        smth_root = output_base / 'smth_run'
        (ctrl_root / 'checkpoints').mkdir(parents=True)
        (smth_root / 'checkpoints').mkdir(parents=True)
        (ctrl_root / 'external_eval/Wang2020_128').mkdir(parents=True)
        (smth_root / 'external_eval/Wang2020_128').mkdir(parents=True)

        ctrl_best = ctrl_root / 'checkpoints/best.pth'
        smth_best = smth_root / 'checkpoints/best.pth'
        ctrl_best.write_bytes(b'ctrl_best_checkpoint_weights')
        smth_best.write_bytes(b'smth_best_checkpoint_weights')
        from data.manifest import sha256
        h_ctrl = sha256(str(ctrl_best))
        h_smth = sha256(str(smth_best))

        # 1. Mock exported best_dev_predictions.csv
        dev_records_ctrl = [
            {'sample_id': f'd{i}', 'path': f'/data/dev_{i}.png', 'label': 0 if i < 5 else 1,
             'group_id': f'g{i}', 'split': 'dev', 'folder_role': 'dev', 'independence_status': 'unverified_dev_folder'}
            for i in range(10)
        ]
        dev_records_smth = [
            {'sample_id': f'd{i}', 'path': f'/data/dev_{i}.png', 'label': 0 if i < 5 else 1,
             'group_id': f'g{i}', 'split': 'dev', 'folder_role': 'dev', 'independence_status': 'unverified_dev_folder'}
            for i in range(10)
        ]
        dev_z_ctrl = np.array([-2.0]*5 + [2.0]*5, dtype=np.float32)
        dev_p_ctrl = 1.0 / (1.0 + np.exp(-dev_z_ctrl))
        dev_z_smth = np.array([-2.5]*5 + [2.5]*5, dtype=np.float32)
        dev_p_smth = 1.0 / (1.0 + np.exp(-dev_z_smth))

        export_predictions(str(ctrl_root / 'checkpoints/best_dev_predictions.csv'), dev_records_ctrl,
                           dev_p_ctrl, checkpoint_hash=h_ctrl, logits=dev_z_ctrl, eval_precision='fp32')
        export_predictions(str(smth_root / 'checkpoints/best_dev_predictions.csv'), dev_records_smth,
                           dev_p_smth, checkpoint_hash=h_smth, logits=dev_z_smth, eval_precision='fp32')

        # 2. Mock external evaluation predictions.csv
        ext_records_ctrl = [
            {'sample_id': f'e{i}', 'path': f'/data/ext_{i}.png', 'label': 0 if i < 10 else 1,
             'group_id': f'eg{i}', 'split': 'external_dev', 'eval_precision': 'fp32', 'checkpoint_sha256': h_ctrl}
            for i in range(20)
        ]
        ext_records_smth = [
            {'sample_id': f'e{i}', 'path': f'/data/ext_{i}.png', 'label': 0 if i < 10 else 1,
             'group_id': f'eg{i}', 'split': 'external_dev', 'eval_precision': 'fp32', 'checkpoint_sha256': h_smth}
            for i in range(20)
        ]
        ext_z_ctrl = np.array([-1.5]*10 + [1.5]*10, dtype=np.float32)
        ext_p_ctrl = 1.0 / (1.0 + np.exp(-ext_z_ctrl))
        ext_z_smth = np.array([-1.8]*10 + [1.8]*10, dtype=np.float32)
        ext_p_smth = 1.0 / (1.0 + np.exp(-ext_z_smth))

        export_predictions(str(ctrl_root / 'external_eval/Wang2020_128/predictions.csv'), ext_records_ctrl,
                           ext_p_ctrl, checkpoint_hash=h_ctrl, logits=ext_z_ctrl, eval_precision='fp32')
        export_predictions(str(smth_root / 'external_eval/Wang2020_128/predictions.csv'), ext_records_smth,
                           ext_p_smth, checkpoint_hash=h_smth, logits=ext_z_smth, eval_precision='fp32')

        # 3. Execute Cell 4 Calibration
        import subprocess
        for run_root in (ctrl_root, smth_root):
            cmd = [
                'python', 'analyze_predictions.py', 'calibrate',
                '--predictions', str(run_root / 'checkpoints/best_dev_predictions.csv'),
                '--output', str(run_root / 'checkpoints/threshold_logit.json'),
                '--score_space', 'logit',
                '--eval_precision', 'fp32',
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)

        self.assertTrue((ctrl_root / 'checkpoints/threshold_logit.json').is_file())
        self.assertTrue((smth_root / 'checkpoints/threshold_logit.json').is_file())

        # 4. Execute Cell 6 Comparison
        timestamp = '20261012_test'
        comp_json = output_base / f'comparison_{timestamp}.json'
        comp_md = output_base / f'comparison_{timestamp}.md'

        comp = compare_evaluation_runs(
            control_pred_path=str(ctrl_root / 'external_eval/Wang2020_128/predictions.csv'),
            smooth_pred_path=str(smth_root / 'external_eval/Wang2020_128/predictions.csv'),
            control_thresh_path=str(ctrl_root / 'checkpoints/threshold_logit.json'),
            smooth_thresh_path=str(smth_root / 'checkpoints/threshold_logit.json'),
            control_checkpoint_path=str(ctrl_root / 'checkpoints/best.pth'),
            smooth_checkpoint_path=str(smth_root / 'checkpoints/best.pth'),
            output_json_path=str(comp_json),
            output_md_path=str(comp_md),
            control_name='ctrl_run',
            smooth_name='smth_run',
            bootstrap_repeats=50,
            seed=42,
            eval_precision='fp32',
        )

        self.assertTrue(comp_json.is_file())
        self.assertTrue(comp_md.is_file())
        self.assertEqual(comp['samples'], 20)
        self.assertEqual(comp['control']['checkpoint_sha256'], h_ctrl)
        self.assertEqual(comp['smoothing']['checkpoint_sha256'], h_smth)


if __name__ == '__main__':
    unittest.main()
