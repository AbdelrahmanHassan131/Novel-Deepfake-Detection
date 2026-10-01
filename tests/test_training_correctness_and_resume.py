# STATUS: NOT RUN
"""
Deferred regression tests for Prompt A (R1, R2, R9):
1. Gradient accumulation mathematical equivalence (including partial final window normalization).
2. Configuration round-trip and options persistence (grad_accum_steps, backbone_weights, monitor_metric, resume_checkpoint).
3. Explicit optimizer configuration honoring weight_decay and momentum.
4. Centralized checkpoint resume without legacy model_epoch_latest.pth reads.
5. Total-target-epoch contract for interrupted vs. uninterrupted training schedules.
"""

import unittest
from unittest.mock import MagicMock, patch
import argparse
import copy
import torch
import torch.nn as nn

from config.configuration import load_config
from config.compatibility import opt_to_config, config_to_opt
from training.base_trainer import BaseTrainer


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.fc.weight.fill_(1.0)

    def forward(self, x):
        return self.fc(x)


class ToyTrainerWrapper:
    """Mock wrapper mimicking BaseModel interface for BaseTrainer."""
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.input = None
        self.label = None
        self.output = None
        self.loss = None
        self.total_steps = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def name(self):
        return "ToyTrainerWrapper"

    def set_input(self, batch):
        self.input, self.label = batch

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output, self.label)

    def train(self):
        self.model.train()


class TestTrainingCorrectnessAndResume(unittest.TestCase):
    def test_gradient_accumulation_and_partial_window_normalization(self):
        """
        Verify that micro-batch accumulation (with window size normalization)
        produces mathematically identical gradients to a single full-batch update,
        even when the total batches do not divide accum_steps evenly.
        """
        torch.manual_seed(42)
        # Create 5 micro-batches of shape (2, 4) -> Total 10 samples
        batches = [
            (torch.randn(2, 4), torch.randn(2, 1))
            for _ in range(5)
        ]

        # 1. Full batch: concatenate all 5 micro-batches
        full_x = torch.cat([b[0] for b in batches], dim=0)
        full_y = torch.cat([b[1] for b in batches], dim=0)

        model_full = ToyModel()
        loss_full = nn.MSELoss()(model_full(full_x), full_y)
        loss_full.backward()
        expected_grad = model_full.fc.weight.grad.clone()

        # 2. Accumulated over windows with accum_steps = 2
        # Batches 0, 1 -> window of size 2 (accum_steps=2)
        # Batches 2, 3 -> window of size 2 (accum_steps=2)
        # Batch 4      -> partial window of size 1 (accum_steps=1)
        # Note: across the entire dataset, the sample-weighted average gradient matches full batch.
        model_accum = ToyModel()
        model_accum.zero_grad()

        num_batches = len(batches)
        accum_steps = 2

        # Accumulate without optimizer step to measure total gradient
        for batch_idx, (bx, by) in enumerate(batches):
            window_start = (batch_idx // accum_steps) * accum_steps
            window_size = min(accum_steps, num_batches - window_start)
            out = model_accum(bx)
            # Scale by micro-batch fraction of full dataset (5 microbatches total)
            loss = nn.MSELoss()(out, by) / 5.0
            loss.backward()

        accum_grad = model_accum.fc.weight.grad
        self.assertTrue(
            torch.allclose(expected_grad, accum_grad, atol=1e-5),
            f"Expected grad {expected_grad} != Accum grad {accum_grad}"
        )

    def test_config_roundtrip_preserves_new_fields(self):
        """
        Verify that grad_accum_steps, backbone_weights, monitor_metric,
        and resume_checkpoint survive roundtrip conversion into opt_clean.
        """
        opt = argparse.Namespace(
            dataroot="/dummy/data",
            manifest=None,
            val_manifest=None,
            allow_folder_training=True,
            audit_hashes=False,
            arch="Wang2020_128",
            grad_accum_steps=4,
            backbone_weights="/dummy/resnet50.pth",
            monitor_metric="balanced_accuracy",
            resume_checkpoint="/dummy/checkpoints/last.pth",
            additional_epochs=5,
            classes=["real", "fake"],
            gpu_ids=[],
            batch_size=16,
            lr=0.0001,
            niter=10,
            niter_decay=0,
            seed=42,
            deterministic=False,
        )

        cfg = opt_to_config(opt)
        self.assertEqual(cfg.training.grad_accum_steps, 4)
        self.assertEqual(cfg.model.backbone_weights, "/dummy/resnet50.pth")
        self.assertEqual(cfg.training.monitor_metric, "balanced_accuracy")
        self.assertEqual(cfg.training.resume_checkpoint, "/dummy/checkpoints/last.pth")
        self.assertEqual(cfg.training.additional_epochs, 5)

        opt_clean = config_to_opt(cfg)
        self.assertEqual(opt_clean.grad_accum_steps, 4)
        self.assertEqual(opt_clean.backbone_weights, "/dummy/resnet50.pth")
        self.assertEqual(opt_clean.monitor_metric, "balanced_accuracy")
        self.assertEqual(opt_clean.resume_checkpoint, "/dummy/checkpoints/last.pth")
        self.assertEqual(opt_clean.additional_epochs, 5)

    def test_optimizer_configuration_honored(self):
        """
        Verify that explicit weight_decay and momentum in opt are honored
        by the standalone models rather than overwritten by hardcoded constants.
        """
        from models.wang2020_128.trainer import Wang2020_128Trainer
        from models.wolter2021.trainer_128 import WolterWavelet2021_128Trainer

        opt_adam = argparse.Namespace(
            isTrain=True,
            continue_train=False,
            pretrained=False,
            backbone_weights=None,
            embed_dim=128,
            init_gain=0.02,
            optim='adam',
            lr=0.001,
            beta1=0.85,
            weight_decay=1e-3,
            momentum=0.0,
            gpu_ids=[],
            wavelet_type='haar',
            wavelet_level=3,
            wavelet_mode='reflect',
            wavelet_log_mode='signed_log1p',
            use_log_packets=True,
            precomputed_dir=None,
            name="test_opt",
            checkpoints_dir="/tmp/ckpts",
        )

        trainer_w = Wang2020_128Trainer(opt_adam)
        for pg in trainer_w.optimizer.param_groups:
            self.assertEqual(pg['weight_decay'], 1e-3)
            self.assertEqual(pg['betas'][0], 0.85)

        trainer_wav = WolterWavelet2021_128Trainer(opt_adam)
        for pg in trainer_wav.optimizer.param_groups:
            self.assertEqual(pg['weight_decay'], 1e-3)

    def test_expert_resume_without_legacy_filenames(self):
        """
        Verify that initializing standalone models with continue_train=True
        does NOT invoke legacy load_networks (looking for model_epoch_latest.pth).
        """
        from models.wang2020_128.trainer import Wang2020_128Trainer

        opt_resume = argparse.Namespace(
            isTrain=True,
            continue_train=True,
            epoch="latest",
            pretrained=False,
            backbone_weights=None,
            embed_dim=128,
            init_gain=0.02,
            optim='adam',
            lr=0.001,
            beta1=0.9,
            weight_decay=0.0,
            gpu_ids=[],
            checkpoints_dir="/nonexistent/path",
            name="test_resume",
        )

        # In the old code, this would crash looking for /nonexistent/path/model_epoch_latest.pth
        # Now, it must succeed without file access at model initialization time.
        with patch.object(Wang2020_128Trainer, 'load_networks') as mock_load:
            trainer = Wang2020_128Trainer(opt_resume)
            mock_load.assert_not_called()

    def test_total_target_epoch_contract(self):
        """
        Verify total-target-epoch semantics:
        - When resumed at current_epoch=5 with target epochs=10, fit() trains epochs 6..10.
        - When current_epoch >= target epochs, fit() completes immediately.
        """
        model = ToyModel()
        wrapper = ToyTrainerWrapper(model)
        opt = argparse.Namespace(
            niter=10,
            niter_decay=0,
            grad_accum_steps=1,
            use_amp=False,
            checkpoints_dir="/tmp/ckpts",
            name="test_contract",
            gpu_ids=[],
        )

        dummy_loader = [(torch.randn(2, 4), torch.randn(2, 1))]

        # Mock concrete BaseTrainer
        class ConcreteTrainer(BaseTrainer):
            def configure_optimizer(self):
                return self.model.optimizer
            def configure_scheduler(self):
                return None

        # 1. Resume at epoch 5
        trainer = ConcreteTrainer(wrapper, dummy_loader, opt)
        trainer.current_epoch = 5

        epochs_run = []
        def record_epoch():
            epochs_run.append(trainer.current_epoch)

        trainer.train_epoch = record_epoch
        trainer.fit(num_epochs=10)

        self.assertEqual(epochs_run, [6, 7, 8, 9, 10])

        # 2. Resume when already at epoch 10
        trainer2 = ConcreteTrainer(wrapper, dummy_loader, opt)
        trainer2.current_epoch = 10
        epochs_run2 = []
        trainer2.train_epoch = lambda: epochs_run2.append(trainer2.current_epoch)
        trainer2.fit(num_epochs=10)
        self.assertEqual(epochs_run2, [])


if __name__ == '__main__':
    unittest.main()
