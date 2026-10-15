"""Unit tests for DRCT_128 architecture, reference contracts, and gradient isolation."""
import argparse
import pytest
import torch
import torch.nn as nn

from models.drct.backbone import DummyBackbone, build_drct_backbone
from models.drct.drct_model import DRCT128Model, DRCTNativeReference
from models.drct.trainer import DRCT_128Trainer
from models.registry import build_model


def make_dummy_opt(embed_dim=128, drct_alpha=0.3):
    opt = argparse.Namespace()
    opt.arch = 'DRCT_128'
    opt.isTrain = True
    opt.continue_train = False
    opt.pretrained = False
    opt.gpu_ids = []
    opt.drct_backbone = 'dummy'
    opt.embed_dim = embed_dim
    opt.drct_alpha = drct_alpha
    opt.drct_pos_margin = 0.0
    opt.drct_neg_margin = 1.0
    opt.drct_enable_contrastive = True
    opt.drct_freeze_encoder = True
    opt.lr = 1e-3
    opt.weight_decay = 0.0
    opt.optim = 'adam'
    opt.beta1 = 0.9
    opt.name = 'test_drct'
    opt.checkpoints_dir = './tmp_test_ckpt'
    return opt


@pytest.mark.parametrize("batch_size", [1, 4])
def test_drct_128_shapes_and_features(batch_size):
    """Test output shapes for B=1 and B>1, and check that classifier strictly uses the 128D feature."""
    model = DRCT128Model(backbone_name='dummy', embed_dim=128, freeze_encoder=True)
    model.eval()

    x = torch.randn(batch_size, 3, 224, 224)
    logits, feat = model(x, return_feature=True)

    assert logits.shape == (batch_size, 1), f"Expected logits shape ({batch_size}, 1), got {logits.shape}"
    assert feat.shape == (batch_size, 128), f"Expected feature shape ({batch_size, 128}), got {feat.shape}"

    # Classifier uses exported feature
    manual_logits = model.fc(feat)
    torch.testing.assert_close(logits, manual_logits)

    # forward_features parity in eval mode
    feat_eval = model.forward_features(x)
    torch.testing.assert_close(feat, feat_eval)


def test_embed_dim_128_enforcement():
    """Verify that specifying embed_dim != 128 raises a clear ValueError."""
    with pytest.raises(ValueError, match="strictly requires embed_dim=128"):
        DRCT128Model(backbone_name='dummy', embed_dim=256)

    opt = make_dummy_opt(embed_dim=64)
    with pytest.raises(ValueError, match="requires embed_dim=128"):
        DRCT_128Trainer(opt)


def test_gradient_isolation_and_head_updates():
    """Prove that head parameters update, frozen encoder parameters do not, and encoder stays frozen."""
    opt = make_dummy_opt()
    trainer = DRCT_128Trainer(opt)
    trainer.train(True)

    encoder_params_before = [p.clone() for p in trainer.model.encoder.parameters()]
    proj_weight_before = trainer.model.projection[0].weight.clone()
    fc_weight_before = trainer.model.fc.weight.clone()

    # Create synthetic batch
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    trainer.set_input((images, labels))

    # Single optimization step
    trainer.optimize_parameters()

    # 1. Encoder parameters must NOT have gradients
    for p in trainer.model.encoder.parameters():
        assert p.grad is None, "Frozen encoder parameter unexpectedly received a gradient!"

    # 2. Encoder parameters must NOT change
    for p_before, p_after in zip(encoder_params_before, trainer.model.encoder.parameters()):
        torch.testing.assert_close(p_before, p_after)

    # 3. Head parameters must have updated
    assert not torch.equal(trainer.model.projection[0].weight, proj_weight_before), "Projection weight did not update!"
    assert not torch.equal(trainer.model.fc.weight, fc_weight_before), "Classifier weight did not update!"


def test_input_gradient_diagnostic_contract():
    """Ensure that input-gradient diagnostics (e.g. adversarial test) can compute dLoss/dInput through frozen encoder."""
    model = DRCT128Model(backbone_name='dummy', embed_dim=128, freeze_encoder=True)
    model.train(True)

    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient was blocked by frozen encoder!"
    assert x.grad.shape == (2, 3, 224, 224)
    assert not torch.all(x.grad == 0), "Input gradient is all zero!"


def test_native_reference_adapter_numerical_parity():
    """Test native reference adapter 2-logit output and single logit numerical equivalence."""
    ref = DRCTNativeReference(backbone_name='dummy', embedding_size=1024, freeze_encoder=False)
    ref.eval()

    x = torch.randn(4, 3, 224, 224)
    logits2 = ref(x)  # [4, 2]
    assert logits2.shape == (4, 2)

    # Two-class softmax probability for fake (class 1)
    prob_fake_softmax = torch.softmax(logits2, dim=-1)[:, 1:2]

    # Single logit z = logit_fake - logit_real
    z = ref.forward_single_logit(x)  # [4, 1]
    prob_fake_sigmoid = torch.sigmoid(z)

    torch.testing.assert_close(prob_fake_softmax, prob_fake_sigmoid, rtol=1e-6, atol=1e-6)


def test_build_model_registry_resolution():
    """Test constructing DRCT_128 via the central build_model function."""
    opt = make_dummy_opt()
    trainer = build_model(opt)
    assert isinstance(trainer, DRCT_128Trainer)
    assert trainer.name() == 'DRCT_128'


def test_forward_features_preserves_autograd():
    """Verify that forward_features preserves autograd and allows backprop into input."""
    model = DRCT128Model(backbone_name='dummy', embed_dim=128, freeze_encoder=True)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    feat = model.forward_features(x)
    assert feat.requires_grad, "forward_features must not strip autograd graph!"
    loss = feat.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow back into input through forward_features!"
    assert x.grad.norm() > 0.0


def test_unfrozen_encoder_updates():
    """Verify that if drct_freeze_encoder=False, encoder receives gradients and updates."""
    opt = make_dummy_opt()
    opt.drct_freeze_encoder = False
    opt.backbone_lr_mult = 0.5
    trainer = DRCT_128Trainer(opt)
    trainer.train(True)

    encoder_params_before = [p.clone() for p in trainer.model.encoder.parameters()]
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    trainer.set_input((images, labels))
    trainer.optimize_parameters()

    # Unfrozen encoder parameters must receive gradients and update
    for p in trainer.model.encoder.parameters():
        assert p.grad is not None, "Unfrozen encoder parameters must receive gradients!"
    for p_before, p_after in zip(encoder_params_before, trainer.model.encoder.parameters()):
        assert not torch.equal(p_before, p_after), "Unfrozen encoder parameters did not update!"


def test_backbone_aliases_and_download_guard(monkeypatch):
    """Verify that aliases like clip_vit_b16 and clip_vit_l14 resolve, and allow_download=False guards."""
    from models.drct.backbone import normalize_backbone_name, build_drct_backbone

    assert normalize_backbone_name('clip_vit_b16') == 'ViT-B/16'
    assert normalize_backbone_name('clip-ViT-B/16') == 'ViT-B/16'
    assert normalize_backbone_name('clip_vit_l14') == 'ViT-L/14'
    assert normalize_backbone_name('clip-ViT-L/14') == 'ViT-L/14'
    assert normalize_backbone_name('dummy') == 'dummy'

    # Verify dummy builds without downloading
    bb, dim = build_drct_backbone('dummy', pretrained=False, allow_download=False)
    assert dim == 768

    # If clip is mocked, verify allow_download=False raises when pretrained requested without local file
    import sys
    class MockClip:
        @staticmethod
        def load(*args, **kwargs):
            raise AssertionError("clip.load should not be called when allow_download=False and weights missing!")

    monkeypatch.setitem(sys.modules, 'clip', MockClip)
    with pytest.raises((FileNotFoundError, ImportError)):
        build_drct_backbone('clip_vit_l14', pretrained=True, allow_download=False)
