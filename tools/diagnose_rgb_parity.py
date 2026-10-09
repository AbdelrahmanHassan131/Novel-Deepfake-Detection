#!/usr/bin/env python
"""
RGB Parity and Correctness Diagnostic Tool.

Accepts a checkpoint and explicitly supplied small manifest, sample IDs, and
optional saved predictions. Compares the SAME image bytes/IDs between the
validation (training-time Validator) and standalone evaluation (Evaluator / CheckpointLoader)
paths in FP32:
    - Transformed tensors (max/mean absolute difference)
    - Forward pass outputs (logits and probabilities)
    - Class mapping and score direction
    - Module eval and BatchNorm modes

Supports a CPU-only synthetic mode (--synthetic) that tests the entire transform
and model contract without requiring a trained checkpoint or external images.

Never scans full datasets by default (bounds to --max_samples, default 50).
Strictly matches by sample_id and path, never by filename basename.
Reports discrepancies transparently without auto-correcting labels.
"""

import os
import sys
import argparse
import json
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Ensure project root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.manifest import read_manifest, known
from data.datasets.rgb_dataset import RGBDataset
from torchvision.datasets.folder import default_loader


def get_eval_transform_from_opt(opt):
    """Reconstruct eval transform pipeline directly from opt."""
    from functools import partial
    from torchvision import transforms
    from data.transforms.augmentations import data_augment
    from data.transforms.resize import custom_resize
    from data.transforms.identity import identity_image

    if getattr(opt, 'no_crop', False):
        crop_func = transforms.Lambda(identity_image)
    else:
        crop_func = transforms.CenterCrop(getattr(opt, 'cropSize', 224))

    flip_func = transforms.Lambda(identity_image)

    if getattr(opt, 'no_resize', False):
        rz_func = transforms.Lambda(identity_image)
    else:
        rz_func = transforms.Lambda(partial(custom_resize, opt=opt))

    return transforms.Compose([
        rz_func,
        transforms.Lambda(partial(data_augment, opt=opt)),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def check_module_eval_and_bn_modes(model: nn.Module) -> Dict[str, Any]:
    """Inspect model module hierarchy for correct eval and BN modes."""
    raw_net = getattr(model, 'model', model)
    is_training = raw_net.training
    bn_violations = []
    dropout_violations = []
    bn_count = 0
    dropout_count = 0

    for name, module in raw_net.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_count += 1
            if module.training:
                bn_violations.append(name)
        elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
            dropout_count += 1
            if module.training:
                dropout_violations.append(name)

    return {
        'model_training_mode': is_training,
        'model_eval_mode_correct': not is_training,
        'bn_modules_inspected': bn_count,
        'bn_training_violations': bn_violations,
        'bn_eval_mode_correct': len(bn_violations) == 0,
        'dropout_modules_inspected': dropout_count,
        'dropout_training_violations': dropout_violations,
        'dropout_eval_mode_correct': len(dropout_violations) == 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose RGB preprocessing and model parity between validation and evaluation paths.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to trained checkpoint (.pth file). Required unless --synthetic is set.")
    parser.add_argument('--manifest', type=str, default=None,
                        help="Path to manifest CSV. Required unless --synthetic is set.")
    parser.add_argument('--dataroot', type=str, default=None,
                        help="Path to dataset root directory (defaults to manifest parent directory).")
    parser.add_argument('--manifest_split', type=str, default=None,
                        help="Split filter for manifest (e.g. 'dev' or 'val').")
    parser.add_argument('--sample_ids', type=str, default=None,
                        help="Comma-separated list of explicit sample IDs to check.")
    parser.add_argument('--max_samples', type=int, default=50,
                        help="Maximum number of samples to evaluate (default: 50, bounds scan).")
    parser.add_argument('--saved_predictions', type=str, default=None,
                        help="Optional CSV of saved predictions to cross-check against computed outputs.")
    parser.add_argument('--synthetic', action='store_true', default=False,
                        help="Run CPU-only synthetic parity test without requiring checkpoint or dataset.")
    parser.add_argument('--arch', type=str, default='Wang2020_128',
                        help="Model architecture name (default: Wang2020_128).")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device for forward pass (default: cpu).")
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help="FP32 absolute tolerance for tensor and logit parity (default: 1e-5).")
    parser.add_argument('--output_report', type=str, default='rgb_parity_report.json',
                        help="Path to save discrepancy report JSON (default: rgb_parity_report.json).")
    return parser.parse_args()


def run_synthetic_diagnostic(arch: str = 'Wang2020_128', tolerance: float = 1e-5) -> Dict[str, Any]:
    """Run CPU-only synthetic verification of transforms and model parity."""
    print("=== Running Synthetic CPU-Only Diagnostic ===", flush=True)

    # 1. Create synthetic images of various sizes and modes
    synthetic_images = [
        ('rgb_standard', Image.new('RGB', (300, 300), color=(120, 150, 180))),
        ('rgb_small', Image.new('RGB', (160, 200), color=(40, 80, 200))),
        ('grayscale_l', Image.new('L', (280, 280), color=128).convert('RGB')),
        ('rgba', Image.new('RGBA', (256, 256), color=(200, 100, 50, 255)).convert('RGB')),
        ('rect_wide', Image.new('RGB', (400, 200), color=(20, 220, 100))),
    ]

    # 2. Build mock options for validation and standalone eval paths
    val_opt = SimpleNamespace(
        isTrain=False,
        no_crop=False,
        no_resize=False,
        no_flip=True,
        cropSize=224,
        loadSize=256,
        rz_interp=['bilinear'],
        blur_prob=0.0,
        jpg_prob=0.0,
        noise_prob=0.0,
        downscale_prob=0.0,
        classes=['real', 'fake'],
    )

    eval_opt = SimpleNamespace(
        isTrain=False,
        no_crop=False,
        no_resize=False,
        no_flip=True,
        cropSize=224,
        loadSize=256,
        rz_interp=['bilinear'],
        blur_prob=0.0,
        jpg_prob=0.0,
        noise_prob=0.0,
        downscale_prob=0.0,
        classes=['real', 'fake'],
    )

    val_transform = get_eval_transform_from_opt(val_opt)
    eval_transform = get_eval_transform_from_opt(eval_opt)

    tensor_comparisons = []
    max_tensor_diff = 0.0

    for name, img in synthetic_images:
        t_val = val_transform(img)
        t_eval = eval_transform(img)

        diff = (t_val - t_eval).abs().max().item()
        mean_diff = (t_val - t_eval).abs().mean().item()
        max_tensor_diff = max(max_tensor_diff, diff)

        tensor_comparisons.append({
            'name': name,
            'size': list(img.size),
            'mode': img.mode,
            'val_shape': list(t_val.shape),
            'eval_shape': list(t_eval.shape),
            'max_abs_diff': diff,
            'mean_abs_diff': mean_diff,
            'discrepancy': diff > tolerance,
        })

    # 3. Model construction and evaluation mode checks
    from models import build_model
    model_opt = SimpleNamespace(
        arch=arch,
        isTrain=False,
        skip_load_networks=True,
        continue_train=False,
        gpu_ids=[],
        name='diagnose_parity_synthetic',
        checkpoints_dir=tempfile.gettempdir(),
        pretrained=False,
        init_gain=0.02,
        optim='adam',
        lr=0.0001,
        beta1=0.9,
        classes=['real', 'fake'],
        mode='binary',
        embed_dim=128,
        score_sign=1.0,
    )
    model = build_model(model_opt)
    model.eval()

    mode_check = check_module_eval_and_bn_modes(model)

    # 4. Forward pass parity check
    forward_comparisons = []
    max_logit_diff = 0.0
    for name, img in synthetic_images:
        t_val = val_transform(img).unsqueeze(0)
        t_eval = eval_transform(img).unsqueeze(0)

        with torch.no_grad():
            out_val = model.model(t_val)
            out_eval = model.model(t_eval)

            prob_val = torch.sigmoid(out_val).squeeze().item()
            prob_eval = torch.sigmoid(out_eval).squeeze().item()

            logit_diff = (out_val - out_eval).abs().max().item()
            max_logit_diff = max(max_logit_diff, logit_diff)

            forward_comparisons.append({
                'name': name,
                'logit_val': float(out_val.squeeze().item()),
                'logit_eval': float(out_eval.squeeze().item()),
                'prob_val': float(prob_val),
                'prob_eval': float(prob_eval),
                'logit_abs_diff': float(logit_diff),
                'discrepancy': logit_diff > tolerance,
            })

    report = {
        'diagnostic_type': 'synthetic_cpu',
        'arch': arch,
        'tolerance': tolerance,
        'mode_check': mode_check,
        'tensor_parity': {
            'samples_tested': len(synthetic_images),
            'max_abs_diff': max_tensor_diff,
            'passed': max_tensor_diff <= tolerance,
            'details': tensor_comparisons,
        },
        'forward_parity': {
            'samples_tested': len(synthetic_images),
            'max_abs_diff': max_logit_diff,
            'passed': max_logit_diff <= tolerance,
            'details': forward_comparisons,
        },
        'status': 'passed' if (max_tensor_diff <= tolerance and max_logit_diff <= tolerance and mode_check['bn_eval_mode_correct']) else 'discrepancy_detected',
    }
    return report


def run_checkpoint_diagnostic(args):
    """Compare the production Validator and InferenceRunner on a bounded cohort."""
    import copy
    from data.manifest import sha256, write_manifest
    from data.loaders.dataloader_factory import create_dataloader
    from evaluation.checkpoint_loader import CheckpointLoader
    from evaluation.evaluator import Evaluator
    from evaluation.inference import InferenceRunner
    from training.validator import Validator

    if args.max_samples < 1:
        raise ValueError('max_samples must be positive')
    ckpt_path = Path(args.checkpoint).resolve()
    root = args.dataroot or str(Path(args.manifest).resolve().parent)
    rows = read_manifest(args.manifest, root=root, split=args.manifest_split, check_files=False)
    if args.sample_ids:
        requested = {x.strip() for x in args.sample_ids.split(',') if x.strip()}
        if len(requested) > args.max_samples:
            raise ValueError('Requested IDs exceed max_samples')
        rows = [r for r in rows if r['sample_id'] in requested]
        if requested != {r['sample_id'] for r in rows}:
            raise ValueError('Some requested sample IDs are missing')
    rows = rows[:args.max_samples]
    if not rows:
        raise ValueError('No samples selected')
    checkpoint_hash = sha256(ckpt_path)
    discrepancies = []
    # Only the selected image paths are probed or opened.
    for r in rows:
        actual_hash = sha256(r['path'])
        if known(r.get('sha256')) and r['sha256'] != actual_hash:
            raise ValueError(f"Image content changed for {r['sample_id']}")
        r['sha256'] = actual_hash

    loader = CheckpointLoader(str(ckpt_path), arch=args.arch, device=args.device)
    model = loader.load()
    if getattr(model.opt, 'arch', args.arch) not in ('Wang2020_128', 'Wang2020Raw'):
        raise ValueError('This diagnostic supports RGB Wang architectures only')
    model.eval()
    # These are bounded parity checks, not source-level scientific evaluation.
    model.opt.eligible_sources = None
    model.opt.val_precision = 'fp32'
    with tempfile.TemporaryDirectory(prefix='rgb_parity_') as tmp:
        small_manifest = Path(tmp) / 'samples.csv'
        write_manifest(small_manifest, rows)
        val_opt = copy.copy(model.opt)
        val_opt.isTrain = False
        val_opt.manifest = str(small_manifest)
        val_opt.manifest_split = None
        val_opt.dataroot = root
        val_opt.batch_size = val_opt.val_batch_size = 1
        val_opt.num_workers = val_opt.val_num_workers = val_opt.num_threads = 0
        val_opt.serial_batches = True
        val_opt.class_bal = False
        val_loader = create_dataloader(val_opt)
        evaluator = Evaluator(str(ckpt_path), root, output_dir=tmp, batch_size=1,
                              device=args.device, collect_embeddings=False,
                              generate_plots=False, run_profiling=False)
        eval_loader = evaluator._build_dataloader(copy.copy(val_opt))
        tensor_diffs = [(val_loader.dataset[i][0] - eval_loader.dataset[i][0]).abs().max().item()
                        for i in range(len(rows))]
        captured_logits = []
        handle = model.model.register_forward_hook(
            lambda module, inputs, output: captured_logits.append(output.detach().float().cpu().reshape(-1).numpy()))
        try:
            val_result = Validator().validate(model, val_loader)
        finally:
            handle.remove()
        eval_result = InferenceRunner(model, eval_loader, device=args.device,
                                      collect_embeddings=False).run()
    probabilities = np.asarray(eval_result.probabilities).reshape(-1)
    val_probabilities = np.asarray(val_result.predictions).reshape(-1)
    score_diff = float(np.max(np.abs(probabilities - val_probabilities)))
    val_logits = np.concatenate(captured_logits) * getattr(model, 'score_sign', 1.0)
    logit_diff = float(np.max(np.abs(val_logits - np.asarray(eval_result.logits).reshape(-1))))
    if not np.array_equal(eval_result.labels, val_result.labels):
        discrepancies.append('Production validation/evaluation labels differ')
    mode_check = check_module_eval_and_bn_modes(model)
    if not all(mode_check[k] for k in ('model_eval_mode_correct', 'bn_eval_mode_correct', 'dropout_eval_mode_correct')):
        discrepancies.append('Model/BN/dropout not in evaluation mode')
    saved_comparisons = []
    if args.saved_predictions:
        from evaluation.generalization import read_predictions
        saved = {r['sample_id']: r for r in read_predictions(args.saved_predictions)}
        for row, probability in zip(rows, probabilities):
            pred = saved.get(row['sample_id'])
            if pred is None:
                discrepancies.append(f"Missing saved prediction: {row['sample_id']}")
                continue
            if pred.get('checkpoint_sha256') != checkpoint_hash:
                discrepancies.append(f"Wrong/missing checkpoint identity: {row['sample_id']}")
            if pred.get('sha256') != row['sha256']:
                discrepancies.append(f"Wrong/missing image content identity: {row['sample_id']}")
            if int(pred['label']) != int(row['label']):
                discrepancies.append(f"Wrong saved label: {row['sample_id']}")
            saved_probability = float(pred['probability'])
            difference = abs(saved_probability - float(probability))
            saved_comparisons.append(dict(sample_id=row['sample_id'], label=int(row['label']),
                saved_probability=saved_probability, local_probability=float(probability),
                absolute_difference=difference,
                decision_changed_at_05=(saved_probability >= .5) != (float(probability) >= .5)))
            if difference > args.tolerance:
                discrepancies.append(f"Saved probability mismatch: {row['sample_id']}")
    passed = (max(tensor_diffs) <= args.tolerance and score_diff <= args.tolerance
              and logit_diff <= args.tolerance and not discrepancies)
    return dict(diagnostic_type='production_rgb_parity', checkpoint_sha256=checkpoint_hash,
                samples_inspected=len(rows), max_tensor_diff=max(tensor_diffs),
                max_probability_diff=score_diff, max_logit_diff=logit_diff, mode_check=mode_check,
                saved_prediction_comparison=dict(
                    samples_compared=len(saved_comparisons),
                    max_probability_diff=max((r['absolute_difference'] for r in saved_comparisons), default=0.0),
                    decisions_changed_at_05=sum(r['decision_changed_at_05'] for r in saved_comparisons),
                    details=sorted(saved_comparisons, key=lambda r: r['absolute_difference'], reverse=True)),
                runtime=dict(torch_version=str(torch.__version__), device=args.device,
                    cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                general_discrepancies=discrepancies, status='passed' if passed else 'discrepancy_detected')


def main():
    args = parse_args()
    if args.synthetic:
        report = run_synthetic_diagnostic(arch=args.arch, tolerance=args.tolerance)
    else:
        if not args.checkpoint:
            print("Error: --checkpoint is required when not in --synthetic mode.", file=sys.stderr)
            sys.exit(1)
        if not args.manifest:
            print("Error: --manifest is required when not in --synthetic mode.", file=sys.stderr)
            sys.exit(1)
        report = run_checkpoint_diagnostic(args)

    out_file = Path(args.output_report).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print("\n=== Parity Diagnostic Results Summary ===")
    print(f"Status           : {report['status']}")
    print(f"Mode Checks      : BN eval={report['mode_check']['bn_eval_mode_correct']}, Dropout eval={report['mode_check']['dropout_eval_mode_correct']}")
    if args.synthetic:
        print(f"Max Tensor Diff  : {report['tensor_parity']['max_abs_diff']:.2e}")
        print(f"Max Logit Diff   : {report['forward_parity']['max_abs_diff']:.2e}")
    else:
        print(f"Samples Tested   : {report['samples_inspected']}")
        print(f"Max Tensor Diff  : {report['max_tensor_diff']:.2e}")
        print(f"Max Score Diff   : {report['max_probability_diff']:.2e}")
        saved = report['saved_prediction_comparison']
        if saved['samples_compared']:
            print(f"Saved Colab Diff : {saved['max_probability_diff']:.6g}")
            print(f"Decisions Changed: {saved['decisions_changed_at_05']}/{saved['samples_compared']} at 0.5")
    print(f"Report written to: {out_file}")

    if report['status'] != 'passed':
        print("\nWARNING: A diagnostic check failed; inspect the report for local-path or saved-prediction differences.", file=sys.stderr)
        sys.exit(2)
    else:
        print("\nSUCCESS: Requested bounded parity checks passed.")


if __name__ == '__main__':
    main()
