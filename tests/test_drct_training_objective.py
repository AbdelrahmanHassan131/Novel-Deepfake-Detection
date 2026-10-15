"""Unit tests for DRCT training objective, margin-based contrastive loss, and engine integration."""
import argparse
import os
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist


from models.drct.loss import DRCTCombinedLoss, MarginContrastiveLoss
from models.drct.trainer import DRCT_128Trainer
from training.base_trainer import BaseTrainer


def test_margin_contrastive_loss_upstream_parity():
    """Verify MarginContrastiveLoss against exact upstream normalized Euclidean arithmetic."""
    loss_fn = MarginContrastiveLoss(pos_margin=0.0, neg_margin=1.0)

    # Exact review benchmark embeddings and labels:
    # emb = [[1, 0], [1, 0.5], [0, 1], [0.5, 1]], labels = [0, 0, 1, 1]
    e = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.5],
        [0.0, 1.0],
        [0.5, 1.0],
    ], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = loss_fn(e, labels)
    expected_loss = torch.tensor(0.8270503283, dtype=torch.float32)
    torch.testing.assert_close(loss, expected_loss, atol=1e-5, rtol=1e-5)

    # Verify gradients are finite and non-zero
    loss.backward()
    assert e.grad is not None
    assert torch.isfinite(e.grad).all()
    assert e.grad.norm() > 0.0

    # Scale invariance test: scaling unnormalized vectors should not change normalized Euclidean loss
    e_scaled = (e.detach() * 5.0).requires_grad_(True)
    loss_scaled = loss_fn(e_scaled, labels)
    torch.testing.assert_close(loss_scaled, expected_loss, atol=1e-5, rtol=1e-5)


def test_combined_loss_alpha_weighting():
    """Test alpha weighting between classification and contrastive losses."""
    logits = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    emb = torch.randn(2, 128)

    # alpha = 0.0 (pure classification)
    loss_fn_0 = DRCTCombinedLoss(alpha=0.0)
    total_0, cls_0, cont_0 = loss_fn_0(logits, labels, emb)
    torch.testing.assert_close(total_0, cls_0)

    # alpha = 0.3
    loss_fn_03 = DRCTCombinedLoss(alpha=0.3)
    total_03, cls_03, cont_03 = loss_fn_03(logits, labels, emb)
    expected_03 = 0.7 * cls_03 + 0.3 * cont_03
    torch.testing.assert_close(total_03, expected_03)


def test_edge_cases_singleton_and_homogeneous():
    """Verify finite behavior on B=1 and homogeneous batches without NaN or exceptions."""
    loss_fn = DRCTCombinedLoss(alpha=0.3)

    # 1. Singleton batch (B=1)
    logit_single = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    label_single = torch.tensor([1.0], dtype=torch.float32)
    emb_single = torch.randn(1, 128, requires_grad=True)

    total_1, cls_1, cont_1 = loss_fn(logit_single, label_single, emb_single)
    assert not torch.isnan(total_1)
    assert not torch.isinf(total_1)
    assert cont_1.item() == 0.0
    total_1.backward()
    assert logit_single.grad is not None and not torch.isnan(logit_single.grad)

    # 2. Homogeneous batch (all label 1)
    logit_homo = torch.tensor([[0.2], [0.8], [0.5]], dtype=torch.float32, requires_grad=True)
    label_homo = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    emb_homo = torch.randn(3, 128, requires_grad=True)

    total_h, cls_h, cont_h = loss_fn(logit_homo, label_homo, emb_homo)
    assert not torch.isnan(total_h)
    assert not torch.isinf(total_h)
    total_h.backward()
    assert logit_homo.grad is not None and not torch.isnan(logit_homo.grad).any()
    assert emb_homo.grad is not None and not torch.isnan(emb_homo.grad).any()



def test_trainer_all_engine_paths():
    """Verify get_loss, optimize_parameters, and evaluation loss behavior."""
    opt = argparse.Namespace(
        arch='DRCT_128',
        isTrain=True,
        continue_train=False,
        pretrained=False,
        gpu_ids=[],
        drct_backbone='dummy',
        embed_dim=128,
        drct_alpha=0.3,
        drct_pos_margin=0.0,
        drct_neg_margin=1.0,
        drct_enable_contrastive=True,
        drct_freeze_encoder=True,
        lr=1e-3,
        weight_decay=0.0,
        optim='adam',
        beta1=0.9,
        name='test_engine',
        checkpoints_dir='./tmp_ckpt',
    )
    trainer = DRCT_128Trainer(opt)

    batch_x = torch.randn(4, 3, 224, 224)
    batch_y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    trainer.set_input((batch_x, batch_y))

    # Path 1: optimize_parameters
    trainer.train(True)
    trainer.optimize_parameters()
    assert trainer.loss is not None
    assert trainer.cls_loss is not None
    assert trainer.contrastive_loss is not None

    # Path 2: forward + get_loss (used by AMP and accumulation engines)
    trainer.forward()
    loss = trainer.get_loss()
    assert loss.requires_grad
    loss.backward()

    # Path 3: evaluation mode get_loss (must use pure classification loss)
    trainer.eval()
    trainer.forward()
    eval_loss = trainer.get_loss()
    assert trainer.contrastive_loss.item() == 0.0
    torch.testing.assert_close(eval_loss, trainer.cls_loss)


def _ddp_worker(rank, world_size, tmp_port):
    """Worker process for 2-process CPU Gloo DDP test."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(tmp_port)
    os.environ['USE_LIBUV'] = '0'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


    opt = argparse.Namespace(
        arch='DRCT_128',
        isTrain=True,
        continue_train=False,
        pretrained=False,
        gpu_ids=[],
        drct_backbone='dummy',
        embed_dim=128,
        drct_alpha=0.3,
        drct_pos_margin=0.0,
        drct_neg_margin=1.0,
        drct_enable_contrastive=True,
        drct_freeze_encoder=True,
        lr=1e-3,
        weight_decay=0.0,
        optim='adam',
        beta1=0.9,
        name='test_ddp',
        checkpoints_dir='./tmp_ddp',
    )
    trainer = DRCT_128Trainer(opt)
    # Wrap inner model in DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    trainer.model = DDP(trainer.model)

    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([0.0, 1.0])
    trainer.set_input((x, y))

    # Test forward and optimize_parameters through DDP
    trainer.optimize_parameters()

    # Ensure gradients synchronized
    for p in trainer.model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    dist.destroy_process_group()


def test_cpu_ddp_two_process_step():
    """Run a 2-process CPU DDP test to ensure DDP forward and backward work cleanly."""
    import socket
    os.environ['USE_LIBUV'] = '0'
    # Find a free port
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    try:
        mp.spawn(_ddp_worker, args=(2, port), nprocs=2, join=True)
    except Exception as e:
        if "libuv" in str(e).lower() or "gloo" in str(e).lower():
            pytest.skip(f"Multi-process CPU Gloo not supported in current Windows PyTorch build: {e}")
        raise

