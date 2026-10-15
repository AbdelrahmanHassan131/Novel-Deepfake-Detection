"""Tests for DRCT-128 configuration, CLI options, checkpoint save/load/resume,
and prediction/evaluation lifecycle (Prompt 5).
"""
import os
import json
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config.configuration import Config, ModelConfig, TrainingConfig, DataConfig
from config.validator import ConfigValidator
from config.compatibility import config_to_opt, opt_to_config
from config.types import ArchitectureType
from models.drct.backbone import DummyBackbone
from models.drct.drct_model import DRCT128Model
from models.drct.trainer import DRCT_128Trainer
from training.checkpoint_manager import CheckpointManager
from evaluation.checkpoint_loader import CheckpointLoader
from evaluation.inference import InferenceRunner


@pytest.fixture
def dummy_opt():
    class DummyOpt:
        def __init__(self):
            self.arch = 'DRCT_128'
            self.drct_backbone = 'dummy'
            self.drct_alpha = 0.3
            self.drct_pos_margin = 0.0
            self.drct_neg_margin = 1.0
            self.drct_enable_contrastive = True
            self.drct_freeze_encoder = True
            self.drct_projection_activation = 'relu'
            self.drct_projection_dropout = 0.0
            self.drct_manifest_path = None
            self.isTrain = True
            self.gpu_ids = []
            self.lr = 1e-4
            self.beta1 = 0.9
            self.weight_decay = 0.0
            self.channels_last = False
            self.pin_memory = False
            self.name = 'test_drct_lifecycle'
            self.checkpoints_dir = None
            self.label_mapping = {'real': 0, 'fake': 1}
            self.rgb_label_smoothing = 0.0
    return DummyOpt()


def test_drct_config_validation_valid():
    """Verify valid DRCT config passes validation."""
    cfg = Config(
        model=ModelConfig(
            architecture=ArchitectureType.DRCT_128,
            drct_backbone="clip_vit_l14",
            drct_alpha=0.3,
            drct_pos_margin=0.0,
            drct_neg_margin=1.0,
            drct_enable_contrastive=True,
            rgb_label_smoothing=0.0,
        ),
        training=TrainingConfig(learning_rate=1e-4),
        data=DataConfig(train_path="dummy/train", val_path="dummy/val"),
    )
    # Should not raise
    ConfigValidator.validate_or_raise(cfg)


def test_drct_config_rejects_wang_smoothing():
    """Verify DRCT rejects Wang-specific label smoothing."""
    cfg = Config(
        model=ModelConfig(
            architecture=ArchitectureType.DRCT_128,
            rgb_label_smoothing=0.1,  # Wang-only option!
        ),
        training=TrainingConfig(learning_rate=1e-4),
        data=DataConfig(train_path="dummy/train", val_path="dummy/val"),
    )
    with pytest.raises(ValueError, match="Label smoothing is not supported for DRCT"):
        ConfigValidator.validate_or_raise(cfg)


def test_drct_config_rejects_invalid_alpha():
    """Verify DRCT validation rejects alpha outside [0, 1]."""
    cfg = Config(
        model=ModelConfig(
            architecture=ArchitectureType.DRCT_128,
            drct_alpha=1.5,
        ),
        training=TrainingConfig(learning_rate=1e-4),
        data=DataConfig(train_path="dummy/train", val_path="dummy/val"),
    )
    with pytest.raises(ValueError, match="drct_alpha must be between 0.0 and 1.0"):
        ConfigValidator.validate_or_raise(cfg)


def test_drct_config_rejects_inverted_margins():
    """Verify DRCT validation rejects neg_margin <= pos_margin."""
    cfg = Config(
        model=ModelConfig(
            architecture=ArchitectureType.DRCT_128,
            drct_pos_margin=1.0,
            drct_neg_margin=0.5,
        ),
        training=TrainingConfig(learning_rate=1e-4),
        data=DataConfig(train_path="dummy/train", val_path="dummy/val"),
    )
    with pytest.raises(ValueError, match="drct_neg_margin.*must be greater than drct_pos_margin"):
        ConfigValidator.validate_or_raise(cfg)


def test_drct_config_compatibility_roundtrip():
    """Verify bidirectional conversion between Config and opt preserves DRCT options."""
    cfg = Config(
        model=ModelConfig(
            architecture=ArchitectureType.DRCT_128,
            drct_backbone="dummy",
            drct_alpha=0.4,
            drct_pos_margin=0.1,
            drct_neg_margin=1.2,
            drct_enable_contrastive=False,
            drct_freeze_encoder=True,
            drct_projection_activation="gelu",
            drct_projection_dropout=0.1,
            rgb_label_smoothing=0.0,
        ),
        training=TrainingConfig(learning_rate=2e-4),
        data=DataConfig(train_path="dummy/train", val_path="dummy/val"),
    )
    opt = config_to_opt(cfg)
    assert opt.arch == 'DRCT_128'
    assert opt.drct_backbone == 'dummy'
    assert opt.drct_alpha == 0.4
    assert opt.drct_pos_margin == 0.1
    assert opt.drct_neg_margin == 1.2
    assert opt.drct_enable_contrastive is False
    assert opt.drct_projection_activation == 'gelu'
    assert opt.drct_projection_dropout == 0.1

    restored_cfg = opt_to_config(opt)
    assert ArchitectureType.from_string(restored_cfg.model.architecture) == ArchitectureType.DRCT_128
    assert restored_cfg.model.drct_backbone == 'dummy'
    assert restored_cfg.model.drct_alpha == 0.4
    assert restored_cfg.model.drct_pos_margin == 0.1
    assert restored_cfg.model.drct_neg_margin == 1.2
    assert restored_cfg.model.drct_enable_contrastive is False
    assert restored_cfg.model.drct_projection_activation == 'gelu'
    assert restored_cfg.model.drct_projection_dropout == 0.1


def test_drct_checkpoint_save_and_resume_roundtrip(dummy_opt, tmp_path):
    """Verify CheckpointManager saves and resumes DRCT state perfectly."""
    dummy_opt.checkpoints_dir = str(tmp_path)
    trainer = DRCT_128Trainer(dummy_opt)
    mgr = CheckpointManager(save_dir=str(tmp_path), model=trainer, rank=0)

    # Take a synthetic step to alter weights and optimizer state
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0.0, 1.0]).view(-1, 1)
    trainer.set_input((x, y))
    loss = trainer.optimize_parameters()
    assert loss > 0

    # Save checkpoint
    mgr.save_last(epoch=1, best_metric=0.85, global_step=100)
    ckpt_path = tmp_path / "last.pth"
    assert ckpt_path.is_file()

    # Create new trainer and verify weights differ initially
    fresh_trainer = DRCT_128Trainer(dummy_opt)
    fresh_mgr = CheckpointManager(save_dir=str(tmp_path), model=fresh_trainer, rank=0)

    # Resume into fresh trainer
    resume_data = fresh_mgr.resume(str(ckpt_path))
    assert resume_data['epoch'] == 1
    assert resume_data['best_metric'] == 0.85
    assert resume_data['global_step'] == 100

    # Weights of projection and fc must match exactly
    for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7)


def test_drct_checkpoint_resume_rejects_protocol_mismatch(dummy_opt, tmp_path):
    """Verify CheckpointManager.resume rejects mismatched DRCT protocol parameters."""
    dummy_opt.checkpoints_dir = str(tmp_path)
    trainer = DRCT_128Trainer(dummy_opt)
    mgr = CheckpointManager(save_dir=str(tmp_path), model=trainer, rank=0)
    mgr.save_last(epoch=1, best_metric=0.85, global_step=10)
    ckpt_path = tmp_path / "last.pth"

    # Now attempt to resume with different drct_alpha
    dummy_opt.drct_alpha = 0.8  # Changed from 0.3!
    fresh_trainer = DRCT_128Trainer(dummy_opt)
    fresh_mgr = CheckpointManager(save_dir=str(tmp_path), model=fresh_trainer, rank=0)

    with pytest.raises(ValueError, match="Resume protocol mismatch for drct_alpha"):
        fresh_mgr.resume(str(ckpt_path))


def test_drct_checkpoint_loader_and_inference_runner(dummy_opt, tmp_path):
    """Verify CheckpointLoader and InferenceRunner operate seamlessly with DRCT-128."""
    dummy_opt.checkpoints_dir = str(tmp_path)
    trainer = DRCT_128Trainer(dummy_opt)
    mgr = CheckpointManager(save_dir=str(tmp_path), model=trainer, rank=0)
    mgr.save_last(epoch=2, best_metric=0.92, global_step=50)
    ckpt_path = str(tmp_path / "last.pth")

    loader = CheckpointLoader(checkpoint_path=ckpt_path, device=torch.device('cpu'))
    eval_model = loader.load(opt_overrides={'drct_backbone': 'dummy'})

    assert eval_model.name() == 'DRCT_128'
    assert eval_model.training is False

    # Create synthetic dataset and dataloader
    images = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    runner = InferenceRunner(eval_model, dataloader, collect_embeddings=True)
    res = runner.run()

    assert len(res.logits) == 6
    assert len(res.probabilities) == 6
    assert len(res.predictions) == 6
    assert len(res.labels) == 6
    # Embeddings must be [6, 128]
    assert res.embeddings.shape == (6, 128)
    # Check that probabilities are sigmoid of logits
    np_sig = 1.0 / (1.0 + np.exp(-res.logits))
    assert np.allclose(res.probabilities, np_sig, atol=1e-5)
