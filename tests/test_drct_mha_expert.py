"""Tests for DRCT-128 as an MHA Expert (Prompt 6).

Verifies:
1. Expert checkpoint validation for DRCT-128 and failure modes.
2. MHA/ControlledFusion forward pass with DRCT expert producing [B, 1].
3. Frozen expert parameter isolation (zero gradients, invariant weights).
4. Feature parity between standalone DRCT and MHA RGB expert.
5. Branch-aware preprocessing in FusionDataset.
6. Embedded expert checkpoint save and reload without source expert file.
7. Backward compatibility for Wang2020 RGB expert.
"""
import os
import shutil
import tempfile
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

from models.drct.backbone import DummyBackbone
from models.drct.drct_model import DRCT128Model
from models.drct.trainer import DRCT_128Trainer
from models.shared.expert_loading import validate_expert_checkpoint, load_expert, DRCT128FeatureExtractor
from models.mha.trainer import MHAFusionTrainer
from models.fusion.trainer import ControlledFusionTrainer
from training.checkpoint_manager import CheckpointManager
from evaluation.checkpoint_loader import CheckpointLoader
from data.datasets.fusion_dataset import FusionDataset
from data.transforms.drct_transforms import DRCTValTransform


def create_dummy_drct_checkpoint(path: Path, embed_dim: int = 128, arch: str = 'DRCT_128', label_mapping=None):
    """Helper to write a valid or configurable DRCT-128 checkpoint."""
    if label_mapping is None:
        label_mapping = {'real': 0, 'fake': 1}
    encoder = DummyBackbone(in_channels=3, out_features=768)
    if embed_dim == 128:
        model = DRCT128Model(backbone_name='dummy', encoder_module=encoder, embed_dim=128)
        state_dict = model.state_dict()
    else:
        state_dict = {
            'encoder.proj.weight': encoder.proj.weight,
            'encoder.proj.bias': encoder.proj.bias,
            'projection.0.weight': torch.randn(embed_dim, 768),
            'projection.0.bias': torch.randn(embed_dim),
            'fc.weight': torch.randn(1, embed_dim),
            'fc.bias': torch.randn(1),
        }
    state = {
        'model_state_dict': state_dict,
        'model_name': arch,
        'protocol': {
            'format_version': 2,
            'label_mapping': label_mapping,
            'options': {
                'arch': arch,
                'drct_backbone': 'dummy',
                'drct_alpha': 0.3,
                'drct_pos_margin': 0.0,
                'drct_neg_margin': 1.0,
                'drct_preprocessing_version': 'drct_clip_v1',
                'embed_dim': embed_dim,
            }
        }
    }
    torch.save(state, path)
    return path


def create_dummy_wavelet_checkpoint(path: Path, embed_dim: int = 128):
    """Helper to write a valid Wavelet expert checkpoint."""
    from models.wolter2021.wavelet_cnn import WaveletPacketCNN128
    model = WaveletPacketCNN128(input_channels=3 * 4**3, embed_dim=embed_dim)
    state = {
        'model_state_dict': model.state_dict(),
        'model_name': 'WolterWavelet2021_128',
        'protocol': {
            'format_version': 2,
            'label_mapping': {'real': 0, 'fake': 1},
            'options': {
                'arch': 'WolterWavelet2021_128',
                'wavelet_type': 'haar',
                'wavelet_level': 3,
                'wavelet_mode': 'reflect',
                'use_log_packets': True,
                'wavelet_log_mode': 'signed_log1p',
                'embed_dim': embed_dim,
            }
        }
    }
    torch.save(state, path)
    return path


def create_dummy_wang_checkpoint(path: Path, embed_dim: int = 128):
    """Helper to write a valid Wang2020_128 checkpoint."""
    from models.shared.resnet import resnet50
    model = resnet50(pretrained=False, num_classes=1)
    model.fc = nn.Sequential(
        nn.Linear(2048, embed_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(embed_dim, 1)
    )
    state = {
        'model_state_dict': model.state_dict(),
        'model_name': 'Wang2020_128',
        'protocol': {
            'format_version': 2,
            'label_mapping': {'real': 0, 'fake': 1},
            'options': {
                'arch': 'Wang2020_128',
                'cropSize': 224,
                'loadSize': 256,
                'no_crop': False,
                'no_resize': False,
                'rz_interp': ['bilinear'],
                'embed_dim': embed_dim,
            }
        }
    }
    torch.save(state, path)
    return path


def test_drct_expert_checkpoint_validation(tmp_path):
    """Verify validate_expert_checkpoint accepts valid DRCT checkpoints and rejects invalid schemas."""
    valid_path = tmp_path / "drct_valid.pth"
    create_dummy_drct_checkpoint(valid_path, embed_dim=128)
    ckpt = torch.load(valid_path)
    # Should validate without error
    validate_expert_checkpoint(ckpt, expected_type='rgb', expected_embed_dim=128)

    # 1. Reject invalid embed_dim
    bad_dim_path = tmp_path / "drct_bad_dim.pth"
    create_dummy_drct_checkpoint(bad_dim_path, embed_dim=64)
    bad_ckpt = torch.load(bad_dim_path)
    with pytest.raises(ValueError, match="RGB expert embedding dim mismatch"):
        validate_expert_checkpoint(bad_ckpt, expected_type='rgb', expected_embed_dim=128)

    # 2. Reject wrong claimed arch
    bad_arch_path = tmp_path / "drct_bad_arch.pth"
    create_dummy_drct_checkpoint(bad_arch_path, arch='WrongArch_128')
    bad_ckpt = torch.load(bad_arch_path)
    with pytest.raises(ValueError, match="RGB expert claimed invalid architecture"):
        validate_expert_checkpoint(bad_ckpt, expected_type='rgb', expected_embed_dim=128)

    # 3. Reject wrong label mapping
    bad_labels_path = tmp_path / "drct_bad_labels.pth"
    create_dummy_drct_checkpoint(bad_labels_path, label_mapping={'real': 1, 'fake': 0})
    bad_ckpt = torch.load(bad_labels_path)
    with pytest.raises(ValueError, match="Incompatible label mapping"):
        validate_expert_checkpoint(bad_ckpt, expected_type='rgb', expected_embed_dim=128)


def test_mha_with_drct_expert_forward_shapes(tmp_path):
    """Verify MHA and ControlledFusion forward pass produces [B, 1] raw logit with DRCT expert."""
    rgb_path = tmp_path / "drct_expert.pth"
    create_dummy_drct_checkpoint(rgb_path, embed_dim=128)
    wav_path = tmp_path / "wav_expert.pth"
    create_dummy_wavelet_checkpoint(wav_path, embed_dim=128)

    class Opt:
        arch = 'MHA_128'
        fusion_type = 'token_attention'
        embed_dim = 128
        num_heads = 4
        dropout = 0.1
        freeze_base_models = True
        rgb_model_path = str(rgb_path)
        wavelet_model_path = str(wav_path)
        isTrain = True
        gpu_ids = []
        optim = 'adam'
        lr = 1e-4
        beta1 = 0.9
        drct_backbone = 'dummy'
        wavelet_type = 'haar'
        wavelet_level = 3
        wavelet_mode = 'reflect'
        use_log_packets = True
        wavelet_log_mode = 'signed_log1p'

    opt = Opt()
    trainer = MHAFusionTrainer(opt)

    # Test B=1
    rgb_1 = torch.randn(1, 3, 224, 224)
    wav_1 = torch.randn(1, 3 * 4**3, 28, 28)
    out_1 = trainer.forward(rgb=rgb_1, wavelet=wav_1)
    assert out_1.shape == (1, 1)

    # Test B=4
    rgb_4 = torch.randn(4, 3, 224, 224)
    wav_4 = torch.randn(4, 3 * 4**3, 28, 28)
    out_4 = trainer.forward(rgb=rgb_4, wavelet=wav_4)
    assert out_4.shape == (4, 1)


def test_drct_expert_frozen_parameters_and_isolation(tmp_path):
    """Verify expert parameters are frozen, receive no gradients, and remain invariant across training steps."""
    rgb_path = tmp_path / "drct_expert.pth"
    create_dummy_drct_checkpoint(rgb_path, embed_dim=128)
    wav_path = tmp_path / "wav_expert.pth"
    create_dummy_wavelet_checkpoint(wav_path, embed_dim=128)

    class Opt:
        arch = 'MHA_128'
        fusion_type = 'token_attention'
        embed_dim = 128
        num_heads = 4
        dropout = 0.1
        freeze_base_models = True
        rgb_model_path = str(rgb_path)
        wavelet_model_path = str(wav_path)
        isTrain = True
        gpu_ids = []
        optim = 'adam'
        lr = 1e-3
        beta1 = 0.9
        drct_backbone = 'dummy'
        wavelet_type = 'haar'
        wavelet_level = 3
        wavelet_mode = 'reflect'
        use_log_packets = True
        wavelet_log_mode = 'signed_log1p'

    opt = Opt()
    trainer = MHAFusionTrainer(opt)

    # Capture initial weights of DRCT expert
    initial_weights = [p.clone() for p in trainer.rgb_model.parameters()]
    assert len(initial_weights) > 0
    assert all(not p.requires_grad for p in trainer.rgb_model.parameters())
    assert all(not p.requires_grad for p in trainer.wavelet_model.parameters())

    # Set batch and run optimization step
    rgb_b = torch.randn(2, 3, 224, 224)
    wav_b = torch.randn(2, 3 * 4**3, 28, 28)
    labels = torch.tensor([0.0, 1.0])
    trainer.set_input((rgb_b, wav_b, labels))
    trainer.optimize_parameters()

    # Verify fusion head has gradients
    head_has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in trainer.model.parameters())
    assert head_has_grad

    # Verify DRCT expert parameters have NO gradients and did not change
    for p_init, p_curr in zip(initial_weights, trainer.rgb_model.parameters()):
        assert p_curr.grad is None
        assert torch.allclose(p_init, p_curr, atol=1e-7)


def test_rgb_feature_parity_standalone_vs_fusion(tmp_path):
    """Verify feature parity: standalone DRCT-128 forward_features equals MHA rgb_model output."""
    rgb_path = tmp_path / "drct_expert.pth"
    create_dummy_drct_checkpoint(rgb_path, embed_dim=128)
    wav_path = tmp_path / "wav_expert.pth"
    create_dummy_wavelet_checkpoint(wav_path, embed_dim=128)

    # 1. Standalone model
    class StandaloneOpt:
        arch = 'DRCT_128'
        drct_backbone = 'dummy'
        drct_alpha = 0.3
        drct_pos_margin = 0.0
        drct_neg_margin = 1.0
        drct_enable_contrastive = True
        drct_freeze_encoder = True
        isTrain = False
        gpu_ids = []
        name = 'standalone'
        checkpoints_dir = str(tmp_path)

    standalone = DRCT_128Trainer(StandaloneOpt())
    # Load the exact same weights
    ckpt = torch.load(rgb_path)
    standalone.model.load_state_dict(ckpt['model_state_dict'])
    standalone.eval()

    # 2. MHA Trainer
    class MHAOpt:
        arch = 'MHA_128'
        fusion_type = 'gated'
        embed_dim = 128
        freeze_base_models = True
        rgb_model_path = str(rgb_path)
        wavelet_model_path = str(wav_path)
        isTrain = False
        gpu_ids = []
        drct_backbone = 'dummy'
        wavelet_level = 3

    mha = MHAFusionTrainer(MHAOpt())
    mha.eval()

    # Feed identical input tensor
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        feat_standalone = standalone.forward_features(x)
        feat_mha_rgb = mha.rgb_model(x)

    assert torch.allclose(feat_standalone, feat_mha_rgb, atol=1e-6)


def test_branch_aware_transforms_fusion_dataset(tmp_path):
    """Verify FusionDataset applies DRCT transforms to RGB and preserves Wavelet pipeline."""
    # Create tiny mock dataset
    data_dir = tmp_path / "data"
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    img_path = real_dir / "img1.png"
    Image.new('RGB', (300, 150), color=(128, 64, 32)).save(img_path)
    fake_path = fake_dir / "img2.png"
    Image.new('RGB', (100, 200), color=(32, 64, 128)).save(fake_path)

    rgb_ckpt = tmp_path / "drct_expert.pth"
    create_dummy_drct_checkpoint(rgb_ckpt, embed_dim=128)

    class Opt:
        dataroot = str(data_dir)
        cropSize = 224
        loadSize = 256
        no_crop = False
        no_flip = True
        no_resize = False
        isTrain = False
        mode = 'binary'
        wavelet_backend = 'cpu'
        wavelet_type = 'haar'
        wavelet_level = 3
        wavelet_mode = 'reflect'
        use_log_packets = True
        wavelet_log_mode = 'signed_log1p'
        rgb_model_path = str(rgb_ckpt)
        rgb_expert_arch = 'DRCT_128'

    opt = Opt()
    dataset = FusionDataset(opt, str(data_dir))
    assert dataset.rgb_is_drct is True

    rgb_tensor, wav_tensor, label = dataset[0]
    # RGB branch must be exactly (3, 224, 224)
    assert rgb_tensor.shape == (3, 224, 224)
    # Wavelet branch must be (3 * 4^3, 28, 28) for 224x224 input
    assert wav_tensor.shape == (192, 28, 28)
    assert label in (0, 1)


def test_embedded_drct_fusion_checkpoint_roundtrip(tmp_path):
    """Verify fusion checkpoint with embedded DRCT expert can reload and run without source expert file."""
    expert_dir = tmp_path / "experts"
    expert_dir.mkdir()
    rgb_path = expert_dir / "drct_expert.pth"
    create_dummy_drct_checkpoint(rgb_path, embed_dim=128)
    wav_path = expert_dir / "wav_expert.pth"
    create_dummy_wavelet_checkpoint(wav_path, embed_dim=128)

    save_dir = tmp_path / "fusion_ckpts"
    save_dir.mkdir()

    class Opt:
        def __init__(self):
            self.arch = 'MHA_128'
            self.fusion_type = 'token_attention'
            self.embed_dim = 128
            self.num_heads = 4
            self.dropout = 0.1
            self.freeze_base_models = True
            self.rgb_model_path = str(rgb_path)
            self.wavelet_model_path = str(wav_path)
            self.isTrain = True
            self.gpu_ids = []
            self.optim = 'adam'
            self.lr = 1e-4
            self.beta1 = 0.9
            self.name = 'mha_run'
            self.checkpoints_dir = str(save_dir)
            self.drct_backbone = 'dummy'
            self.wavelet_type = 'haar'
            self.wavelet_level = 3
            self.wavelet_mode = 'reflect'
            self.use_log_packets = True
            self.dropout = 0.0

    opt = Opt()
    trainer = MHAFusionTrainer(opt)

    # Save checkpoint
    mgr = CheckpointManager(save_dir=str(save_dir), model=trainer, rank=0)
    mgr.save_last(epoch=1, best_metric=0.91, global_step=40)
    fusion_ckpt_path = save_dir / "last.pth"
    assert fusion_ckpt_path.is_file()

    # Now DELETE the source experts directory!
    shutil.rmtree(expert_dir)
    assert not rgb_path.exists()
    assert not wav_path.exists()

    # Load from fusion checkpoint using CheckpointLoader
    loader = CheckpointLoader(checkpoint_path=str(fusion_ckpt_path), device=torch.device('cpu'))
    restored_model = loader.load(opt_overrides={'drct_backbone': 'dummy'})

    assert restored_model.name() == 'MHA_128'
    assert restored_model.rgb_model.arch == 'DRCT_128'

    # Forward pass on restored model must work perfectly without external files
    trainer.eval()
    restored_model.eval()
    rgb_test = torch.randn(2, 3, 224, 224)
    wav_test = torch.randn(2, 3 * 4**3, 28, 28)
    with torch.no_grad():
        out_orig = trainer.forward(rgb=rgb_test, wavelet=wav_test)
        out_restored = restored_model.forward(rgb=rgb_test, wavelet=wav_test)

    assert torch.allclose(out_orig, out_restored, atol=1e-5)


def test_wang_rgb_expert_backward_compatibility(tmp_path):
    """Verify existing Wang2020_128 expert still loads and operates cleanly without regression."""
    wang_path = tmp_path / "wang_expert.pth"
    create_dummy_wang_checkpoint(wang_path, embed_dim=128)
    wav_path = tmp_path / "wav_expert.pth"
    create_dummy_wavelet_checkpoint(wav_path, embed_dim=128)

    class Opt:
        arch = 'Fusion_128'
        fusion_type = 'concat'
        embed_dim = 128
        freeze_base_models = True
        rgb_model_path = str(wang_path)
        wavelet_model_path = str(wav_path)
        isTrain = True
        gpu_ids = []
        optim = 'adam'
        lr = 1e-4
        beta1 = 0.9
        wavelet_type = 'haar'
        wavelet_level = 3
        wavelet_mode = 'reflect'
        use_log_packets = True
        wavelet_log_mode = 'signed_log1p'
        cropSize = 224
        loadSize = 256
        no_crop = False
        no_resize = False
        rz_interp = ['bilinear']

    opt = Opt()
    trainer = ControlledFusionTrainer(opt)

    rgb = torch.randn(2, 3, 224, 224)
    wav = torch.randn(2, 3 * 4**3, 28, 28)
    out = trainer.forward(rgb=rgb, wavelet=wav)
    assert out.shape == (2, 1)


def test_drct_expert_rejects_missing_encoder_weights(tmp_path):
    """Verify load_expert rejects checkpoints that lack encoder parameters (R6/R7)."""
    bad_ckpt_path = tmp_path / "incomplete_drct.pth"
    # Create checkpoint without encoder weights
    state = {
        'model_state_dict': {
            'projection.0.weight': torch.randn(128, 768),
            'projection.0.bias': torch.randn(128),
            'fc.weight': torch.randn(1, 128),
            'fc.bias': torch.randn(1),
        },
        'model_name': 'DRCT_128',
        'protocol': {
            'format_version': 2,
            'label_mapping': {'real': 0, 'fake': 1},
            'options': {
                'arch': 'DRCT_128',
                'drct_backbone': 'dummy',
                'embed_dim': 128,
            }
        }
    }
    torch.save(state, bad_ckpt_path)

    opt = SimpleNamespace(rgb_model_path=str(bad_ckpt_path), drct_backbone='dummy')
    with pytest.raises(ValueError, match="Failed to load DRCT_128 model weights"):
        load_expert(opt, 'rgb', torch.device('cpu'))


def test_wang_in_drct_named_path_not_misidentified(tmp_path):
    """Verify a Wang checkpoint located under a folder containing 'drct' is not misidentified (R8)."""
    drct_dir = tmp_path / "drct_experiments" / "wang_run"
    drct_dir.mkdir(parents=True)
    wang_path = drct_dir / "wang_model.pth"
    create_dummy_wang_checkpoint(wang_path, embed_dim=128)

    data_dir = tmp_path / "data"
    (data_dir / "real").mkdir(parents=True)
    (data_dir / "fake").mkdir(parents=True)
    Image.new('RGB', (64, 64)).save(data_dir / "real" / "1.png")
    Image.new('RGB', (64, 64)).save(data_dir / "fake" / "2.png")

    class Opt:
        dataroot = str(data_dir)
        cropSize = 224
        loadSize = 256
        no_crop = False
        no_flip = True
        no_resize = False
        isTrain = False
        mode = 'binary'
        wavelet_backend = 'cpu'
        wavelet_type = 'haar'
        wavelet_level = 3
        wavelet_mode = 'reflect'
        use_log_packets = True
        wavelet_log_mode = 'signed_log1p'
        rgb_model_path = str(wang_path)

    opt = Opt()
    dataset = FusionDataset(opt, str(data_dir))
    # Must NOT be misidentified as DRCT despite 'drct' in the path
    assert dataset.rgb_is_drct is False


def test_embedded_drct_preserves_non_default_activation_and_dropout(tmp_path):
    """Verify embedded MHA checkpoint preserves non-default projection activation (e.g. ReLU) without source file (R7)."""
    expert_dir = tmp_path / "source_experts"
    expert_dir.mkdir()
    rgb_path = expert_dir / "drct_relu_expert.pth"

    # Create DRCT with ReLU activation
    encoder = DummyBackbone(in_channels=3, out_features=768)
    model = DRCT128Model(
        backbone_name='dummy',
        encoder_module=encoder,
        embed_dim=128,
        projection_activation='relu',
        projection_dropout=0.2,
    )
    state = {
        'model_state_dict': model.state_dict(),
        'model_name': 'DRCT_128',
        'protocol': {
            'format_version': 2,
            'label_mapping': {'real': 0, 'fake': 1},
            'options': {
                'arch': 'DRCT_128',
                'drct_backbone': 'dummy',
                'drct_projection_activation': 'relu',
                'drct_projection_dropout': 0.2,
                'embed_dim': 128,
            }
        }
    }
    torch.save(state, rgb_path)

    wav_path = expert_dir / "wav_expert.pth"
    create_dummy_wavelet_checkpoint(wav_path, embed_dim=128)

    save_dir = tmp_path / "fusion_ckpts"
    save_dir.mkdir()

    class Opt:
        def __init__(self):
            self.arch = 'MHA_128'
            self.fusion_type = 'token_attention'
            self.embed_dim = 128
            self.num_heads = 4
            self.dropout = 0.0
            self.freeze_base_models = True
            self.rgb_model_path = str(rgb_path)
            self.wavelet_model_path = str(wav_path)
            self.isTrain = True
            self.gpu_ids = []
            self.optim = 'adam'
            self.lr = 1e-4
            self.beta1 = 0.9
            self.name = 'mha_relu_run'
            self.checkpoints_dir = str(save_dir)
            self.drct_backbone = 'dummy'
            self.wavelet_type = 'haar'
            self.wavelet_level = 3
            self.wavelet_mode = 'reflect'
            self.use_log_packets = True

    trainer = MHAFusionTrainer(Opt())
    mgr = CheckpointManager(save_dir=str(save_dir), model=trainer, rank=0)
    mgr.save_last(epoch=1, best_metric=0.95, global_step=10)
    ckpt_path = save_dir / "last.pth"

    # Delete source experts directory completely
    shutil.rmtree(expert_dir)
    assert not rgb_path.exists()

    # Load from embedded checkpoint
    loader = CheckpointLoader(checkpoint_path=str(ckpt_path), device=torch.device('cpu'))
    restored = loader.load(opt_overrides={'drct_backbone': 'dummy'})

    # Verify that the restored DRCT expert uses ReLU in its projection
    proj_seq = restored.rgb_model.projection
    has_relu = any(isinstance(layer, nn.ReLU) for layer in proj_seq)
    assert has_relu, "Restored DRCT expert did not preserve ReLU activation from embedded metadata!"

    # Test numerical output equivalence
    trainer.eval()
    restored.eval()
    rgb_test = torch.randn(2, 3, 224, 224)
    wav_test = torch.randn(2, 3 * 4**3, 28, 28)
    with torch.no_grad():
        out_orig = trainer.forward(rgb=rgb_test, wavelet=wav_test)
        out_restored = restored.forward(rgb=rgb_test, wavelet=wav_test)
    assert torch.allclose(out_orig, out_restored, atol=1e-5)

