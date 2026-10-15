"""Unit tests for DRCT checkpoint import, strict restoration, and versioned preprocessing."""
import numpy as np
import pytest
import torch
from PIL import Image

from data.transforms.drct_transforms import DRCTValTransform, get_drct_preprocessing_metadata
from models.drct.backbone import DummyBackbone
from models.drct.checkpoint_io import (
    initialize_drct128_from_pretrained_encoder,
    import_official_drct_detector,
    restore_drct128_checkpoint,
)
from models.drct.drct_model import DRCT128Model, DRCTNativeReference


def test_strict_restore_drct128_checkpoint():
    """Test strict round-trip restoration of a locally trained DRCT-128 checkpoint."""
    model = DRCT128Model(backbone_name='dummy', embed_dim=128)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'protocol': {
            'format_version': 2,
            'label_mapping': {'real': 0, 'fake': 1},
            'options': {'arch': 'DRCT_128', 'embed_dim': 128},
        },
        'model_name': 'DRCT_128',
    }

    fresh_model = DRCT128Model(backbone_name='dummy', embed_dim=128)
    res = restore_drct128_checkpoint(fresh_model, ckpt, strict=True)
    assert res['status'] == 'success'
    assert len(res['missing_keys']) == 0
    assert len(res['unexpected_keys']) == 0

    # Test error on invalid architecture
    bad_arch_ckpt = dict(ckpt)
    bad_arch_ckpt['protocol'] = {'options': {'arch': 'Wang2020_128'}}
    bad_arch_ckpt['model_name'] = 'Wang2020_128'
    with pytest.raises(ValueError, match="expected 'DRCT_128'"):
        restore_drct128_checkpoint(fresh_model, bad_arch_ckpt, strict=True)

    # Test error on reversed label mapping
    bad_label_ckpt = dict(ckpt)
    bad_label_ckpt['protocol'] = {'label_mapping': {'real': 1, 'fake': 0}, 'options': {'arch': 'DRCT_128'}}
    with pytest.raises(ValueError, match="Incompatible label mapping"):
        restore_drct128_checkpoint(fresh_model, bad_label_ckpt, strict=True)


def test_rejection_of_slicing_official_projection():
    """Verify that encoder-only initialization discards official 1024-wide heads without truncation or corruption (S6)."""
    model = DRCT128Model(backbone_name='dummy', embed_dim=128)
    initial_proj_weight = model.projection[0].weight.clone()
    initial_fc_weight = model.fc.weight.clone()

    # Simulate an official checkpoint with 1024-wide projection and 2-class classifier
    fake_official_ckpt = {
        'encoder.proj.weight': torch.randn(768, 3),
        'encoder.proj.bias': torch.randn(768),
        'model.fc.weight': torch.randn(1024, 768),
        'model.fc.bias': torch.randn(1024),
        'fc.weight': torch.randn(2, 1024),
        'fc.bias': torch.randn(2),
    }

    meta = initialize_drct128_from_pretrained_encoder(model, fake_official_ckpt)
    assert meta['status'] == 'initialized_for_adaptation_training'
    assert meta['trained_detector'] is False
    assert meta['head_status'] == 'freshly_initialized_requires_training'
    # 128D projection and classifier must remain untouched with fresh weights
    assert model.projection[0].weight.shape == (128, 768)
    torch.testing.assert_close(model.projection[0].weight, initial_proj_weight)
    torch.testing.assert_close(model.fc.weight, initial_fc_weight)


def test_import_official_drct_detector_and_single_logit_parity():
    """Test importing an official reference detector and checking 2-logit to 1-logit parity."""
    dummy_encoder = DummyBackbone(in_channels=3, out_features=768)

    # Synthetic official checkpoint weights matching upstream ContrastiveModels
    official_weights = {
        'model.model.visual.proj.weight': dummy_encoder.proj.weight.clone(),
        'model.model.visual.proj.bias': dummy_encoder.proj.bias.clone(),
        # Deliberately include text-only tensors to verify R6 allowlist filtering
        'model.model.token_embedding.weight': torch.randn(10, 10),
        'model.model.text_projection': torch.randn(10, 10),
        'model.model.ln_final.weight': torch.randn(10),
        'model.fc.weight': torch.randn(1024, 768),
        'model.fc.bias': torch.randn(1024),
        'fc.weight': torch.randn(2, 1024),
        'fc.bias': torch.randn(2),
    }

    ref_model, meta = import_official_drct_detector(
        checkpoint_path_or_dict=official_weights,
        backbone_name='dummy',
        embedding_size=1024,
        encoder_module=dummy_encoder,
    )
    assert meta['model_type'] == 'official_reference'
    assert meta['num_classes'] == 2

    x = torch.randn(3, 3, 224, 224)
    logits2 = ref_model(x)  # [3, 2]
    z = ref_model.forward_single_logit(x)  # [3, 1]

    # Check softmax fake prob == sigmoid(z)
    prob_softmax = torch.softmax(logits2, dim=-1)[:, 1:2]
    prob_sigmoid = torch.sigmoid(z)
    torch.testing.assert_close(prob_softmax, prob_sigmoid)


def test_import_official_drct_detector_rejects_missing_visual_weights():
    """Verify import_official_drct_detector rejects incomplete checkpoints missing visual encoder weights (R6)."""
    dummy_encoder = DummyBackbone(in_channels=3, out_features=768)
    incomplete_weights = {
        'model.fc.weight': torch.randn(1024, 768),
        'model.fc.bias': torch.randn(1024),
        'fc.weight': torch.randn(2, 1024),
        'fc.bias': torch.randn(2),
    }
    with pytest.raises(ValueError, match="Incomplete official detector checkpoint: missing required weights"):
        import_official_drct_detector(
            checkpoint_path_or_dict=incomplete_weights,
            backbone_name='dummy',
            embedding_size=1024,
            encoder_module=dummy_encoder,
        )


def test_initialize_drct128_from_pretrained_encoder():
    """Test initializing a new 128D adaptation from pretrained encoder without marking it as a trained detector."""
    model = DRCT128Model(backbone_name='dummy', embed_dim=128)
    encoder_weights = {'encoder.proj.weight': torch.randn(768, 3), 'encoder.proj.bias': torch.randn(768)}

    meta = initialize_drct128_from_pretrained_encoder(model, encoder_weights)
    assert meta['status'] == 'initialized_for_adaptation_training'
    assert meta['trained_detector'] is False
    assert meta['head_status'] == 'freshly_initialized_requires_training'


def test_drct_val_transform_various_shapes():
    """Test versioned DRCT preprocessing on square, landscape, portrait, small, and large images."""
    transform = DRCTValTransform(size=224)

    test_shapes = [
        (224, 224),   # Square nominal
        (100, 100),   # Small square
        (640, 360),   # Landscape
        (360, 640),   # Portrait
        (1920, 1080), # Large
    ]

    for w, h in test_shapes:
        # 1. PIL Image
        img_pil = Image.fromarray(np.random.randint(0, 256, (h, w, 3), dtype=np.uint8))
        out_pil = transform(img_pil)
        assert out_pil.shape == (3, 224, 224), f"Failed shape for PIL ({w}, {h}): got {out_pil.shape}"

        # 2. Numpy RGB
        img_np = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        out_np = transform(img_np)
        assert out_np.shape == (3, 224, 224), f"Failed shape for NumPy ({w}, {h}): got {out_np.shape}"

    # Verify deterministic repeated evaluation
    img_fixed = Image.fromarray(np.zeros((150, 300, 3), dtype=np.uint8))
    out1 = transform(img_fixed)
    out2 = transform(img_fixed)
    torch.testing.assert_close(out1, out2)


def test_drct_preprocessing_metadata():
    """Verify that DRCT preprocessing metadata contains expected versioned keys."""
    meta = get_drct_preprocessing_metadata(size=224)
    assert meta['preprocessing_version'] == 'drct_clip_v1'
    assert meta['input_size'] == 224
    assert meta['color_order'] == 'RGB'
    assert meta['resize_policy'] == 'longest_max_size'
