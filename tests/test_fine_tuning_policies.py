"""
Unit tests for RGB fine-tuning policies, parameter freezing, BatchNorm modes,
backbone LR multipliers, AdamW optimizer groups, and strict initialization.
STATUS: NOT RUN (Static code phase test suite)
"""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from config.configuration import Config
from config.compatibility import config_to_opt
from models.wang2020_128.trainer import Wang2020_128Trainer
from models.shared.resnet import resnet50


class TestFineTuningPolicies(unittest.TestCase):
    def _create_base_opt(self, **kwargs):
        opt = config_to_opt(Config.from_defaults())
        opt.arch = 'Wang2020_128'
        opt.pretrained = False
        opt.isTrain = True
        opt.continue_train = False
        opt.gpu_ids = []
        opt.lr = 0.001
        for k, v in kwargs.items():
            setattr(opt, k, v)
        return opt

    def test_head_only_policy_freezes_backbone(self):
        opt = self._create_base_opt(fine_tune_policy='head_only')
        trainer = Wang2020_128Trainer(opt)

        trainable_names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
        frozen_names = [name for name, p in trainer.model.named_parameters() if not p.requires_grad]

        self.assertTrue(all(name.startswith('fc.') for name in trainable_names))
        self.assertTrue(any(name.startswith('layer4.') for name in frozen_names))
        self.assertTrue(any(name.startswith('conv1.') for name in frozen_names))

        # Check optimizer parameters match only trainable
        opt_params = set()
        for group in trainer.optimizer.param_groups:
            opt_params.update(group['params'])
        self.assertEqual(len(opt_params), len(trainable_names))

    def test_layer4_and_head_policy(self):
        opt = self._create_base_opt(fine_tune_policy='layer4_and_head')
        trainer = Wang2020_128Trainer(opt)

        trainable_names = [name for name, p in trainer.model.named_parameters() if p.requires_grad]
        frozen_names = [name for name, p in trainer.model.named_parameters() if not p.requires_grad]

        self.assertTrue(all(name.startswith('fc.') or name.startswith('layer4.') for name in trainable_names))
        self.assertTrue(any(name.startswith('layer4.') for name in trainable_names))
        self.assertTrue(any(name.startswith('layer3.') for name in frozen_names))
        self.assertTrue(any(name.startswith('conv1.') for name in frozen_names))

    def test_backbone_lr_multiplier(self):
        opt = self._create_base_opt(
            fine_tune_policy='full',
            lr=0.001,
            backbone_lr_mult=0.1,
            weight_decay=1e-4,
        )
        trainer = Wang2020_128Trainer(opt)

        group_names = {g['name']: g['lr'] for g in trainer.optimizer.param_groups}
        self.assertIn('head_decay', group_names)
        self.assertAlmostEqual(group_names['head_decay'], 0.001)

        self.assertIn('backbone_decay', group_names)
        self.assertAlmostEqual(group_names['backbone_decay'], 0.0001)

    def test_frozen_bn_policy(self):
        opt = self._create_base_opt(bn_policy='frozen')
        trainer = Wang2020_128Trainer(opt)

        # Set to train mode
        trainer.train(True)

        # All BatchNorm layers must still be in eval mode
        bn_modules = [m for m in trainer.model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertGreater(len(bn_modules), 0)
        for bn in bn_modules:
            self.assertFalse(bn.training)

    def test_train_bn_policy(self):
        opt = self._create_base_opt(bn_policy='train')
        trainer = Wang2020_128Trainer(opt)

        trainer.train(True)
        bn_modules = [m for m in trainer.model.modules() if isinstance(m, nn.BatchNorm2d)]
        self.assertGreater(len(bn_modules), 0)
        for bn in bn_modules:
            self.assertTrue(bn.training)

    def test_head_types_and_dropout(self):
        # 128d MLP
        opt_128d = self._create_base_opt(rgb_head_type='128d', rgb_dropout=0.3)
        trainer_128d = Wang2020_128Trainer(opt_128d)
        self.assertEqual(len(trainer_128d.model.fc), 4)
        self.assertIsInstance(trainer_128d.model.fc[2], nn.Dropout)
        self.assertEqual(trainer_128d.model.fc[2].p, 0.3)

        # Linear head
        opt_lin = self._create_base_opt(rgb_head_type='linear', rgb_dropout=0.2)
        trainer_lin = Wang2020_128Trainer(opt_lin)
        self.assertEqual(trainer_lin.head_type, 'linear')
        # Final layer is Linear(2048, 1)
        self.assertEqual(trainer_lin.model.fc[1].in_features, 2048)
        self.assertEqual(trainer_lin.model.fc[1].out_features, 1)

    def test_adamw_and_decay_policy(self):
        opt = self._create_base_opt(
            optim='adamw',
            weight_decay=0.01,
            decay_bias_norm=False,
            fine_tune_policy='full',
        )
        trainer = Wang2020_128Trainer(opt)
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)

        # Verify that no_decay groups have weight_decay == 0.0
        for group in trainer.optimizer.param_groups:
            if 'no_decay' in group['name']:
                self.assertEqual(group['weight_decay'], 0.0)
            elif 'decay' in group['name']:
                self.assertEqual(group['weight_decay'], 0.01)

    def test_strict_backbone_validation_rejects_missing_keys(self):
        # Create a checkpoint missing layer4
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            partial_state = {'conv1.weight': torch.randn(64, 3, 7, 7)}
            torch.save(partial_state, f.name)
            corrupt_path = f.name

        try:
            with self.assertRaises(RuntimeError) as ctx:
                resnet50(pretrained=False, weights_path=corrupt_path)
            self.assertIn("missing backbone keys", str(ctx.exception))
        finally:
            Path(corrupt_path).unlink(missing_ok=True)


    def test_ddp_wrapping_preserves_head_train_mode_under_partial_freezing(self):
        """Verify R02: DDP-wrapped model keeps head/dropout in train mode when fine_tune_policy is partial."""
        opt = self._create_base_opt(fine_tune_policy='layer4_and_head', bn_policy='frozen')
        trainer = Wang2020_128Trainer(opt)

        class MockDDPWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.module = inner
            def forward(self, *a, **kw):
                return self.module(*a, **kw)

        # Wrap underlying ResNet in simulated DDP
        inner_resnet = trainer.model
        trainer.model = MockDDPWrapper(inner_resnet)

        # First simulate validation call (which puts model in eval)
        trainer.eval()
        self.assertFalse(inner_resnet.fc.training)

        # Call trainer.train(True)
        ret = trainer.train(True)
        self.assertIs(ret, trainer)

        # Head and layer4 MUST be in train mode!
        self.assertTrue(inner_resnet.fc.training)
        self.assertTrue(inner_resnet.layer4.training)

        # Earlier layers MUST be in eval mode!
        self.assertFalse(inner_resnet.conv1.training)
        self.assertFalse(inner_resnet.layer1.training)
        self.assertFalse(inner_resnet.layer2.training)
        self.assertFalse(inner_resnet.layer3.training)

        # BatchNorm modules MUST be in eval mode due to bn_policy='frozen'
        for m in inner_resnet.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                self.assertFalse(m.training)


if __name__ == '__main__':
    unittest.main()
