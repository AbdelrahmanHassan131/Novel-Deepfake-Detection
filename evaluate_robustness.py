#!/usr/bin/env python
"""
Robustness Evaluation Script for Deepfake Detection Models.

Evaluates model resilience against:
  1. Adversarial Perturbations (White-box):
     - FGSM (Fast Gradient Sign Method)
     - PGD  (Projected Gradient Descent)

  2. Anti-Forensic / Re-rendering Proxies:
     - Heavy JPEG Compression (quality 30 and 50)
     - Gaussian Blur (kernel 5x5, sigma 2.0)
     - Gaussian Noise (std 0.05 and 0.10)

Reports Accuracy, AUC, EER for Clean data and the metric drop (Delta)
under each attack condition.  Saves a full JSON + Markdown report.

This script does NOT modify the existing evaluate.py pipeline.
It reuses the same CheckpointLoader, ClassificationMetrics, and
model interface (set_input / forward / output / label).

Usage Examples:
    # Single model robustness evaluation:
    python evaluate_robustness.py --val_root ./dataset/val \
        --mha ./checkpoints/mha.pth \
        --rgb_model_path ./checkpoints/wang2020_128.pth \
        --wavelet_model_path ./checkpoints/wolter2021_128.pth

    # Choose specific attacks only:
    python evaluate_robustness.py --val_root ./dataset/val \
        --wang2020_raw ./checkpoints/wang2020_raw.pth \
        --attacks fgsm pgd jpeg_30

    # Adjust adversarial epsilon:
    python evaluate_robustness.py --val_root ./dataset/val \
        --mha_wwxc ./checkpoints/mha_wwxc.pth \
        --fgsm_eps 0.03 --pgd_eps 0.03 --pgd_steps 20
"""

import os
import sys
import io
import json
import copy
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.checkpoint_loader import CheckpointLoader
from evaluation.metrics.classification import ClassificationMetrics


# ======================================================================
# 1.  Attack Implementations
# ======================================================================

def _differentiable_forward(model_wrapper, rgb_adv, batch, loss_fn):
    """
    Build a differentiable forward pass that bypasses the model's
    internal torch.no_grad() wrappers.

    For fusion/MHA models, the base models' forward() calls are wrapped
    in torch.no_grad() (because they are frozen during training). This
    prevents gradients from flowing back to the input images, breaking
    FGSM/PGD. This function calls model components directly with
    gradients enabled so that loss.backward() produces valid input grads.

    Args:
        model_wrapper: The BaseModel subclass (e.g., MHAFusionWWXCTrainer).
        rgb_adv: The RGB image tensor with requires_grad=True [B, 3, H, W].
        batch: The original DataLoader batch tuple.
        loss_fn: Loss function (BCEWithLogitsLoss).

    Returns:
        loss: Scalar loss tensor with grad_fn attached to rgb_adv.
        label: The ground-truth label tensor on the model's device.
    """
    device = model_wrapper.device
    is_wwxc = hasattr(model_wrapper, 'xception_model') and hasattr(model_wrapper, 'convnext_model')
    is_fusion = hasattr(model_wrapper, 'rgb_model')

    if is_fusion:
        # --- Fusion / MHA / MHA_WWXC models (multi-input) ---
        # batch = (rgb_imgs, wavelet_imgs, labels)
        wavelet_imgs = batch[1]
        labels = batch[2].to(device).float()

        # Move rgb_adv to device (keeping grad chain alive)
        rgb_on_device = rgb_adv.to(device)

        # Compute wavelet input (non-differentiable path, detached)
        if wavelet_imgs.shape[1] == 3:
            backend = model_wrapper._get_gpu_wavelet_backend()
            with torch.no_grad():
                wavelet_input = backend(wavelet_imgs.to(device))
        else:
            wavelet_input = wavelet_imgs.to(device)

        # Forward through base models WITH gradients (no torch.no_grad!)
        wang_embed = model_wrapper.rgb_model(rgb_on_device)
        wolter_embed = model_wrapper.wavelet_model(wavelet_input)

        if is_wwxc:
            xception_embed = model_wrapper.xception_model(rgb_on_device)
            convnext_embed = model_wrapper.convnext_model(rgb_on_device)
            output = model_wrapper.model(wang_embed, wolter_embed,
                                         xception_embed, convnext_embed)
        else:
            output = model_wrapper.model(wang_embed, wolter_embed)

        loss = loss_fn(output.squeeze(1), labels)
        return loss, labels

    else:
        # --- Single-input models (Wang2020, Xception, ConvNeXt, Wolter) ---
        labels = batch[1].to(device).float()
        rgb_on_device = rgb_adv.to(device)
        output = model_wrapper.model(rgb_on_device)
        loss = loss_fn(output.squeeze(1), labels)
        return loss, labels


def _zero_all_grads(model_wrapper):
    """Zero gradients on all model components."""
    model_wrapper.model.zero_grad()
    if hasattr(model_wrapper, 'rgb_model'):
        model_wrapper.rgb_model.zero_grad()
    if hasattr(model_wrapper, 'wavelet_model'):
        model_wrapper.wavelet_model.zero_grad()
    if hasattr(model_wrapper, 'xception_model'):
        model_wrapper.xception_model.zero_grad()
    if hasattr(model_wrapper, 'convnext_model'):
        model_wrapper.convnext_model.zero_grad()


class AdversarialFGSM:
    """
    FGSM (Goodfellow et al., 2015) -- single-step white-box attack.

    Perturbs input x as:  x_adv = x + eps * sign(grad_x L(model(x), y))

    Uses a differentiable forward pass that bypasses the model's
    internal torch.no_grad() wrappers on frozen base models.

    Processes images in micro-batches to avoid CUDA OOM when running
    multiple large base models with gradients enabled.
    """

    def __init__(self, model_wrapper, eps=0.02, micro_bs=4):
        self.model_wrapper = model_wrapper
        self.eps = eps
        self.micro_bs = micro_bs
        self.loss_fn = nn.BCEWithLogitsLoss()

    def __repr__(self):
        return f"FGSM(eps={self.eps}, micro_bs={self.micro_bs})"

    def _slice_batch(self, batch, start, end):
        """Slice a batch tuple along the first (sample) dimension."""
        return tuple(t[start:end] for t in batch)

    def attack(self, batch):
        """
        Apply FGSM to the RGB image tensor in a batch.

        Processes in micro-batches to avoid CUDA OOM.

        Args:
            batch: Tuple from DataLoader -- (img, label) or
                   (rgb_img, wavelet_img, label) for fusion models.

        Returns:
            Perturbed batch in the same tuple format.
        """
        model = self.model_wrapper
        total = batch[0].size(0)
        all_rgb_adv = []

        for start in range(0, total, self.micro_bs):
            end = min(start + self.micro_bs, total)
            micro = self._slice_batch(batch, start, end)

            rgb_clean = micro[0].clone().detach().requires_grad_(True)

            loss, _ = _differentiable_forward(model, rgb_clean, micro, self.loss_fn)
            loss.backward()

            grad_sign = rgb_clean.grad.data.sign()
            rgb_adv = torch.clamp(rgb_clean.data + self.eps * grad_sign, 0.0, 1.0)
            all_rgb_adv.append(rgb_adv.detach().cpu())

            _zero_all_grads(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rgb_adv_full = torch.cat(all_rgb_adv, dim=0)

        # Repack into the original batch format
        if len(batch) == 2:
            return (rgb_adv_full, batch[1])
        elif len(batch) == 3:
            return (rgb_adv_full, batch[1], batch[2])
        else:
            return (rgb_adv_full,) + batch[1:]

    @staticmethod
    def _unpack(batch):
        """Separate image tensors from labels."""
        if len(batch) == 2:
            return [batch[0]], [], batch[1], False
        elif len(batch) == 3:
            return [batch[0], batch[1]], [], batch[2], True
        else:
            return [batch[0]], list(batch[1:-1]), batch[-1], False

    @staticmethod
    def _repack(images, rest, label, is_multi):
        if is_multi and len(images) == 2:
            return (images[0], images[1], label)
        elif len(images) == 1 and not rest:
            return (images[0], label)
        else:
            return tuple(images) + tuple(rest) + (label,)


class AdversarialPGD:
    """
    PGD (Madry et al., 2018) -- iterative white-box attack.

    Multi-step FGSM with random start, projected back to eps-ball.

    Uses a differentiable forward pass that bypasses the model's
    internal torch.no_grad() wrappers on frozen base models.

    Processes images in micro-batches to avoid CUDA OOM.
    """

    def __init__(self, model_wrapper, eps=0.02, alpha=None, steps=10, micro_bs=4):
        self.model_wrapper = model_wrapper
        self.eps = eps
        self.alpha = alpha or (eps / max(steps // 4, 1))
        self.steps = steps
        self.micro_bs = micro_bs
        self.loss_fn = nn.BCEWithLogitsLoss()

    def __repr__(self):
        return f"PGD(eps={self.eps}, alpha={self.alpha:.4f}, steps={self.steps}, micro_bs={self.micro_bs})"

    def _slice_batch(self, batch, start, end):
        """Slice a batch tuple along the first (sample) dimension."""
        return tuple(t[start:end] for t in batch)

    def attack(self, batch):
        model = self.model_wrapper
        total = batch[0].size(0)

        # Original clean images (anchor for projection)
        rgb_orig = batch[0].clone().detach()

        # Random start within eps-ball
        delta = torch.empty_like(rgb_orig).uniform_(-self.eps, self.eps)
        rgb_adv = torch.clamp(rgb_orig + delta, 0.0, 1.0)

        for _ in range(self.steps):
            all_grads = []

            for start in range(0, total, self.micro_bs):
                end = min(start + self.micro_bs, total)
                micro = self._slice_batch(batch, start, end)

                rgb_chunk = rgb_adv[start:end].clone().detach().requires_grad_(True)
                loss, _ = _differentiable_forward(model, rgb_chunk, micro, self.loss_fn)
                loss.backward()

                all_grads.append(rgb_chunk.grad.data.sign().cpu())

                _zero_all_grads(model)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Reassemble full gradient sign and step
            grad_sign = torch.cat(all_grads, dim=0)
            rgb_step = rgb_adv + self.alpha * grad_sign

            # Project back to eps-ball around the original
            delta = torch.clamp(rgb_step - rgb_orig, -self.eps, self.eps)
            rgb_adv = torch.clamp(rgb_orig + delta, 0.0, 1.0)

        # Repack into the original batch format
        if len(batch) == 2:
            return (rgb_adv.detach(), batch[1])
        elif len(batch) == 3:
            return (rgb_adv.detach(), batch[1], batch[2])
        else:
            return (rgb_adv.detach(),) + batch[1:]


class AntiForensicTransform:
    """
    Differentiable (or numpy-based) anti-forensic image transforms
    applied at the tensor level before model inference.

    Supported transforms:
        - jpeg_30 / jpeg_50  : DiffJPEG-style quality reduction
        - gaussian_blur      : Gaussian blur (kernel 5, sigma 2)
        - gaussian_noise_005 : Additive Gaussian noise (std 0.05)
        - gaussian_noise_010 : Additive Gaussian noise (std 0.10)
    """

    @staticmethod
    def apply(batch, transform_name):
        """
        Apply a named transform to the image tensor(s) in a batch.

        Args:
            batch: DataLoader batch tuple.
            transform_name: One of the supported transform names.

        Returns:
            Transformed batch (same tuple structure).
        """
        images, rest, label, is_multi = AdversarialFGSM._unpack(batch)

        transformed = []
        for img in images:
            t = AntiForensicTransform._apply_single(img, transform_name)
            transformed.append(t)

        return AdversarialFGSM._repack(transformed, rest, label, is_multi)

    @staticmethod
    def _apply_single(tensor, name):
        """Apply transform to a single image tensor [B, C, H, W]."""
        if name == 'jpeg_30':
            return AntiForensicTransform._jpeg_compress(tensor, quality=30)
        elif name == 'jpeg_50':
            return AntiForensicTransform._jpeg_compress(tensor, quality=50)
        elif name == 'gaussian_blur':
            return AntiForensicTransform._gaussian_blur(tensor, kernel_size=5, sigma=2.0)
        elif name == 'gaussian_noise_005':
            return AntiForensicTransform._add_gaussian_noise(tensor, std=0.05)
        elif name == 'gaussian_noise_010':
            return AntiForensicTransform._add_gaussian_noise(tensor, std=0.10)
        else:
            raise ValueError(f"Unknown transform: {name}")

    @staticmethod
    def _jpeg_compress(tensor, quality=30):
        """
        Simulate JPEG compression by encoding/decoding each image
        through PIL.  Operates on CPU, returns tensor on original device.
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        device = tensor.device
        results = []
        for i in range(tensor.size(0)):
            img = tensor[i].detach().cpu().clamp(0, 1)
            pil_img = TF.to_pil_image(img)

            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer).convert('RGB')

            t = TF.to_tensor(compressed).to(device)
            results.append(t)

        return torch.stack(results)

    @staticmethod
    def _gaussian_blur(tensor, kernel_size=5, sigma=2.0):
        """Apply Gaussian blur using torchvision functional."""
        import torchvision.transforms.functional as TF
        return TF.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=sigma)

    @staticmethod
    def _add_gaussian_noise(tensor, std=0.05):
        """Add Gaussian noise and clamp to [0, 1]."""
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


# ======================================================================
# 2.  Robustness Evaluation Loop
# ======================================================================

AVAILABLE_ATTACKS = [
    'fgsm', 'pgd',
    'jpeg_30', 'jpeg_50',
    'gaussian_blur',
    'gaussian_noise_005', 'gaussian_noise_010',
]


def evaluate_clean(model, dataloader, device):
    """
    Run clean (unperturbed) evaluation.

    Returns:
        dict with 'accuracy', 'roc_auc', 'eer', plus full metrics.
        Also returns raw probabilities and labels for reference.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Clean Evaluation'):
            model.set_input(batch)
            model.forward()

            output = model.output
            if output.dim() > 1:
                output = output.squeeze(1)

            probs = torch.sigmoid(output).cpu().numpy()
            labels = model.label.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metric_calc = ClassificationMetrics(threshold=0.5)
    metrics = metric_calc.compute_all(probs, labels)

    return metrics, probs, labels


def evaluate_adversarial(model, dataloader, attack_obj, device, attack_name):
    """
    Run adversarial evaluation with the given attack object.

    The attack needs gradients, so we cannot use torch.no_grad().
    """
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc=f'Adversarial ({attack_name})'):
        # Apply the attack (generates adversarial examples)
        adv_batch = attack_obj.attack(batch)

        # Now run inference on the adversarial batch
        with torch.no_grad():
            model.set_input(adv_batch)
            model.forward()

            output = model.output
            if output.dim() > 1:
                output = output.squeeze(1)

            probs = torch.sigmoid(output).cpu().numpy()
            labels = model.label.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metric_calc = ClassificationMetrics(threshold=0.5)
    metrics = metric_calc.compute_all(probs, labels)

    return metrics


def evaluate_antiforensic(model, dataloader, transform_name, device):
    """
    Run anti-forensic transform evaluation.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Anti-Forensic ({transform_name})'):
            transformed_batch = AntiForensicTransform.apply(batch, transform_name)
            model.set_input(transformed_batch)
            model.forward()

            output = model.output
            if output.dim() > 1:
                output = output.squeeze(1)

            probs = torch.sigmoid(output).cpu().numpy()
            labels = model.label.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    metric_calc = ClassificationMetrics(threshold=0.5)
    metrics = metric_calc.compute_all(probs, labels)

    return metrics


def run_robustness_evaluation(model, dataloader, device, attacks_to_run,
                              fgsm_eps, pgd_eps, pgd_alpha, pgd_steps,
                              skip_clean=False, clean_accuracy=None,
                              clean_auc=None, clean_eer=None,
                              adv_micro_bs=4):
    """
    Master function: runs clean + all requested attacks, computes deltas.

    Args:
        model: Loaded BaseModel subclass.
        dataloader: Evaluation DataLoader.
        device: torch device string.
        attacks_to_run: List of attack names (subset of AVAILABLE_ATTACKS).
        fgsm_eps, pgd_eps, pgd_alpha, pgd_steps: Adversarial hyperparameters.
        skip_clean: If True, skip clean evaluation and use provided baselines.
        clean_accuracy, clean_auc, clean_eer: Baseline values when skip_clean=True.

    Returns:
        dict with 'clean' metrics and per-attack results + deltas.
    """
    results = {}

    if skip_clean:
        # Use provided or default baseline values
        clean_metrics = {
            'accuracy': clean_accuracy if clean_accuracy is not None else 1.0,
            'roc_auc': clean_auc if clean_auc is not None else 1.0,
            'eer': clean_eer if clean_eer is not None else 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
        }
        print("\n" + "=" * 70)
        print("  SKIPPING Clean Evaluation (using provided baselines)")
        print("=" * 70)
        print(f"  Baseline Accuracy: {clean_metrics['accuracy']:.4f}")
        print(f"  Baseline AUC:      {clean_metrics['roc_auc']:.4f}")
        print(f"  Baseline EER:      {clean_metrics['eer']:.4f}")
    else:
        print("\n" + "=" * 70)
        print("  CLEAN (Unperturbed) Evaluation")
        print("=" * 70)
        clean_full, _, _ = evaluate_clean(model, dataloader, device)
        clean_metrics = clean_full
        print(f"  Clean Accuracy: {clean_metrics['accuracy']:.4f}")
        print(f"  Clean AUC:      {clean_metrics['roc_auc']:.4f}")
        print(f"  Clean EER:      {clean_metrics['eer']:.4f}")

    results['clean'] = {
        'accuracy': clean_metrics['accuracy'],
        'roc_auc': clean_metrics['roc_auc'],
        'eer': clean_metrics['eer'],
        'f1_score': clean_metrics.get('f1_score', 0.0),
        'precision': clean_metrics.get('precision', 0.0),
        'recall': clean_metrics.get('recall', 0.0),
    }

    # --- Adversarial Attacks ---
    adversarial_attacks = {
        'fgsm': lambda: AdversarialFGSM(model, eps=fgsm_eps, micro_bs=adv_micro_bs),
        'pgd': lambda: AdversarialPGD(model, eps=pgd_eps,
                                       alpha=pgd_alpha, steps=pgd_steps,
                                       micro_bs=adv_micro_bs),
    }

    antiforensic_transforms = [
        'jpeg_30', 'jpeg_50',
        'gaussian_blur',
        'gaussian_noise_005', 'gaussian_noise_010',
    ]

    for attack_name in attacks_to_run:
        print(f"\n{'=' * 70}")
        print(f"  Attack: {attack_name.upper()}")
        print(f"{'=' * 70}")

        if attack_name in adversarial_attacks:
            attack_obj = adversarial_attacks[attack_name]()
            print(f"  Config: {attack_obj}")
            atk_metrics = evaluate_adversarial(
                model, dataloader, attack_obj, device, attack_name)

        elif attack_name in antiforensic_transforms:
            atk_metrics = evaluate_antiforensic(
                model, dataloader, attack_name, device)

        else:
            print(f"  [WARNING] Unknown attack '{attack_name}', skipping.")
            continue

        # Compute deltas
        delta_acc = atk_metrics['accuracy'] - clean_metrics['accuracy']
        delta_auc = atk_metrics['roc_auc'] - clean_metrics['roc_auc']
        delta_eer = atk_metrics['eer'] - clean_metrics['eer']

        results[attack_name] = {
            'accuracy': atk_metrics['accuracy'],
            'roc_auc': atk_metrics['roc_auc'],
            'eer': atk_metrics['eer'],
            'f1_score': atk_metrics['f1_score'],
            'precision': atk_metrics['precision'],
            'recall': atk_metrics['recall'],
            'delta_accuracy': delta_acc,
            'delta_roc_auc': delta_auc,
            'delta_eer': delta_eer,
        }

        print(f"  Accuracy: {atk_metrics['accuracy']:.4f}  (Δ = {delta_acc:+.4f})")
        print(f"  AUC:      {atk_metrics['roc_auc']:.4f}  (Δ = {delta_auc:+.4f})")
        print(f"  EER:      {atk_metrics['eer']:.4f}  (Δ = {delta_eer:+.4f})")

    return results


# ======================================================================
# 3.  Report Generation
# ======================================================================

def save_json_report(results, model_name, output_path):
    """Save robustness results as structured JSON."""
    report = {
        'model': model_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'results': results,
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report saved: {output_path}")


def save_markdown_report(results, model_name, output_path):
    """Generate a Markdown table summarizing robustness results."""
    lines = [
        f"# Robustness Evaluation Report: {model_name}",
        f"",
        f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        f"",
        f"## Summary Table",
        f"",
        f"| Condition | Accuracy | AUC | EER | Δ Accuracy | Δ AUC | Δ EER |",
        f"|:----------|:--------:|:---:|:---:|:----------:|:-----:|:-----:|",
    ]

    clean = results.get('clean', {})
    lines.append(
        f"| **Clean (Baseline)** | {clean.get('accuracy', 0):.4f} | "
        f"{clean.get('roc_auc', 0):.4f} | {clean.get('eer', 0):.4f} | — | — | — |"
    )

    # Pretty names for display
    display_names = {
        'fgsm': 'FGSM',
        'pgd': 'PGD',
        'jpeg_30': 'JPEG Q=30',
        'jpeg_50': 'JPEG Q=50',
        'gaussian_blur': 'Gaussian Blur',
        'gaussian_noise_005': 'Gaussian Noise σ=0.05',
        'gaussian_noise_010': 'Gaussian Noise σ=0.10',
    }

    for key, vals in results.items():
        if key == 'clean':
            continue
        name = display_names.get(key, key)
        lines.append(
            f"| {name} | {vals['accuracy']:.4f} | {vals['roc_auc']:.4f} | "
            f"{vals['eer']:.4f} | {vals['delta_accuracy']:+.4f} | "
            f"{vals['delta_roc_auc']:+.4f} | {vals['delta_eer']:+.4f} |"
        )

    lines += [
        "",
        "## Attack Configurations",
        "",
    ]

    for key, vals in results.items():
        if key == 'clean':
            continue
        name = display_names.get(key, key)
        lines.append(f"- **{name}**: Accuracy = {vals['accuracy']:.4f}, "
                     f"AUC = {vals['roc_auc']:.4f}, EER = {vals['eer']:.4f}")

    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Markdown report saved: {output_path}")


# ======================================================================
# 4.  CLI & Main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Robustness Evaluation: Adversarial Perturbations & Anti-Forensic Attacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Dataset
    parser.add_argument('--val_root', type=str, required=True,
                        help="Path to validation folder with 'real' and 'fake' subfolders.")

    # Model checkpoints (same flags as evaluate.py for convenience)
    model_group = parser.add_argument_group("Model Checkpoints")
    model_group.add_argument('--wang2020_raw', type=str, default=None)
    model_group.add_argument('--wang2020_128', type=str, default=None)
    model_group.add_argument('--wolter2021_raw', type=str, default=None)
    model_group.add_argument('--wolter2021_128', type=str, default=None)
    model_group.add_argument('--xception', dest='xception', type=str, default=None)
    model_group.add_argument('--xception_128', type=str, default=None)
    model_group.add_argument('--convnext_raw', type=str, default=None)
    model_group.add_argument('--convnext_128', type=str, default=None)
    model_group.add_argument('--fusion', type=str, default=None)
    model_group.add_argument('--mha', type=str, default=None)
    model_group.add_argument('--fusion_wwxc', type=str, default=None)
    model_group.add_argument('--mha_wwxc', type=str, default=None)
    model_group.add_argument('--checkpoint', type=str, default=None,
                             help="Single checkpoint path")
    model_group.add_argument('--arch', type=str, default=None,
                             help="Architecture name when using --checkpoint")

    # WWXC / Fusion base model paths
    parser.add_argument('--rgb_model_path', type=str, default=None)
    parser.add_argument('--wavelet_model_path', type=str, default=None)
    parser.add_argument('--xception_model_path', type=str, default=None)
    parser.add_argument('--convnext_model_path', type=str, default=None)

    # Attack selection & parameters
    atk_group = parser.add_argument_group("Attack Configuration")
    atk_group.add_argument('--attacks', nargs='+', default=None,
                           choices=AVAILABLE_ATTACKS,
                           help=f"Attacks to run (default: all). Choices: {AVAILABLE_ATTACKS}")
    atk_group.add_argument('--fgsm_eps', type=float, default=0.02,
                           help="FGSM epsilon (default: 0.02)")
    atk_group.add_argument('--pgd_eps', type=float, default=0.02,
                           help="PGD epsilon (default: 0.02)")
    atk_group.add_argument('--pgd_alpha', type=float, default=None,
                           help="PGD step size (default: eps/4)")
    atk_group.add_argument('--pgd_steps', type=int, default=10,
                           help="PGD iteration count (default: 10)")
    atk_group.add_argument('--adv_batch_size', type=int, default=4,
                           help="Micro-batch size for adversarial attacks to avoid CUDA OOM (default: 4)")

    # Pipeline settings
    parser.add_argument('--output_dir', type=str, default='robustness_results',
                        help="Directory for output reports")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)

    # Skip clean evaluation
    parser.add_argument('--skip_clean', action='store_true', default=False,
                        help="Skip clean evaluation and jump to attacks. "
                             "Use --clean_accuracy/--clean_auc/--clean_eer to provide baselines.")
    parser.add_argument('--clean_accuracy', type=float, default=None,
                        help="Baseline clean accuracy (used with --skip_clean for delta computation)")
    parser.add_argument('--clean_auc', type=float, default=None,
                        help="Baseline clean AUC (used with --skip_clean)")
    parser.add_argument('--clean_eer', type=float, default=None,
                        help="Baseline clean EER (used with --skip_clean)")

    return parser.parse_args()


def collect_single_model(args):
    """
    Resolve the single model to evaluate from CLI args.

    Returns:
        (display_name, checkpoint_path, architecture_name) or None.
    """
    named = [
        ('Wang2020Raw', args.wang2020_raw, 'Wang2020Raw'),
        ('Wang2020_128', args.wang2020_128, 'Wang2020_128'),
        ('WolterWavelet2021Raw', args.wolter2021_raw, 'WolterWavelet2021Raw'),
        ('WolterWavelet2021_128', args.wolter2021_128, 'WolterWavelet2021_128'),
        ('XceptionRaw', args.xception, 'XceptionRaw'),
        ('Xception_128', args.xception_128, 'Xception_128'),
        ('ConvNeXtRaw', args.convnext_raw, 'ConvNeXtRaw'),
        ('ConvNeXt_128', args.convnext_128, 'ConvNeXt_128'),
        ('Fusion_128', args.fusion, 'Fusion_128'),
        ('MHA_128', args.mha, 'MHA_128'),
        ('Fusion_WWXC', args.fusion_wwxc, 'Fusion_WWXC'),
        ('MHA_WWXC', args.mha_wwxc, 'MHA_WWXC'),
    ]

    for display, path, arch in named:
        if path and os.path.isfile(path):
            return display, path, arch

    if args.checkpoint and os.path.isfile(args.checkpoint):
        arch = args.arch or 'DetectedArch'
        return arch, args.checkpoint, args.arch

    return None


def build_dataloader(model, dataroot, batch_size):
    """Build the correct dataloader for the model architecture."""
    import copy as _copy
    from data import create_dataloader, create_mha_dataloader

    opt = _copy.copy(model.opt)
    opt.dataroot = dataroot
    opt.batch_size = batch_size
    opt.isTrain = False

    defaults = {
        'classes': ['0_real', '1_fake'],
        'mode': 'binary',
        'serial_batches': True,
        'class_bal': False,
        'num_threads': 0,
        'no_crop': False,
        'no_flip': True,
        'no_resize': False,
        'cropSize': 224,
        'loadSize': 256,
        'rz_interp': ['bilinear'],
        'blur_prob': 0.0,
        'blur_sig': [0.5],
        'jpg_prob': 0.0,
        'jpg_method': ['cv2'],
        'jpg_qual': [75],
        'data_aug': False,
        'wavelet_backend': 'gpu' if torch.cuda.is_available() else 'cpu',
        'wavelet_type': 'haar',
        'level': 3,
        'precomputed_dir': None,
        'rgb_model_path': getattr(opt, 'rgb_model_path', ''),
        'wavelet_model_path': getattr(opt, 'wavelet_model_path', ''),
    }
    for k, v in defaults.items():
        if not hasattr(opt, k) or (k == 'classes' and not getattr(opt, k)):
            setattr(opt, k, v)

    arch = getattr(opt, 'arch', '')
    if arch in ('Fusion_128', 'MHA_128', 'Fusion_WWXC', 'MHA_WWXC'):
        return create_mha_dataloader(opt)
    else:
        return create_dataloader(opt)


def main():
    args = parse_args()

    if not os.path.exists(args.val_root):
        print(f"[ERROR] Validation directory not found: {args.val_root}")
        sys.exit(1)

    model_info = collect_single_model(args)
    if model_info is None:
        print("[ERROR] No valid checkpoint provided!")
        print("Provide a model checkpoint, e.g.:")
        print("  --wang2020_raw <path.pth>  or  --mha <path.pth>  or  --checkpoint <path.pth> --arch <ArchName>")
        sys.exit(1)

    display_name, ckpt_path, arch = model_info
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Determine which attacks to run
    attacks = args.attacks if args.attacks else AVAILABLE_ATTACKS

    print("=" * 70)
    print(f"  ROBUSTNESS EVALUATION")
    print(f"  Model:      {display_name}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Dataset:    {args.val_root}")
    print(f"  Device:     {device}")
    print(f"  Attacks:    {attacks}")
    print("=" * 70)

    # 1. Load model
    overrides = {'dataroot': args.val_root, 'batch_size': args.batch_size}
    if args.rgb_model_path:
        overrides['rgb_model_path'] = args.rgb_model_path
    if args.wavelet_model_path:
        overrides['wavelet_model_path'] = args.wavelet_model_path
    if args.xception_model_path:
        overrides['xception_model_path'] = args.xception_model_path
    if args.convnext_model_path:
        overrides['convnext_model_path'] = args.convnext_model_path

    loader = CheckpointLoader(checkpoint_path=ckpt_path, arch=arch, device=device)
    model = loader.load(opt_overrides=overrides)

    # 2. Build dataloader
    dataloader = build_dataloader(model, args.val_root, args.batch_size)

    # 3. Run robustness evaluation
    results = run_robustness_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        attacks_to_run=attacks,
        fgsm_eps=args.fgsm_eps,
        pgd_eps=args.pgd_eps,
        pgd_alpha=args.pgd_alpha,
        pgd_steps=args.pgd_steps,
        skip_clean=args.skip_clean,
        clean_accuracy=args.clean_accuracy,
        clean_auc=args.clean_auc,
        clean_eer=args.clean_eer,
        adv_micro_bs=args.adv_batch_size,
    )

    # 4. Save reports
    out_dir = os.path.join(args.output_dir, display_name)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, 'robustness_results.json')
    md_path = os.path.join(out_dir, 'robustness_report.md')

    save_json_report(results, display_name, json_path)
    save_markdown_report(results, display_name, md_path)

    # 5. Print final summary
    print("\n" + "=" * 70)
    print("  ROBUSTNESS EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n  {'Condition':<28} {'Accuracy':>9} {'AUC':>9} {'EER':>9} {'D Acc':>9} {'D AUC':>9} {'D EER':>9}")
    print(f"  {'-' * 28} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9}")

    clean = results['clean']
    print(f"  {'Clean (Baseline)':<28} {clean['accuracy']:>9.4f} {clean['roc_auc']:>9.4f} "
          f"{clean['eer']:>9.4f} {'---':>9} {'---':>9} {'---':>9}")

    for key, vals in results.items():
        if key == 'clean':
            continue
        print(f"  {key:<28} {vals['accuracy']:>9.4f} {vals['roc_auc']:>9.4f} "
              f"{vals['eer']:>9.4f} {vals['delta_accuracy']:>+9.4f} "
              f"{vals['delta_roc_auc']:>+9.4f} {vals['delta_eer']:>+9.4f}")

    print(f"\n  Reports saved to: {out_dir}")


if __name__ == '__main__':
    main()
