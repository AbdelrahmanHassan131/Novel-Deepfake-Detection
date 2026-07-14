#!/usr/bin/env python
"""
Main Evaluation and Model Comparison Script.

This is the primary CLI command for evaluating one or multiple deepfake detection
checkpoints (.pth files) on a validation dataset folder containing 'real' and 'fake' images.

It generates:
- Comprehensive metrics: F1 Score, ROC AUC, EER, Accuracy, PR AUC, Specificity, Sensitivity, FAR, FRR
- Individual model plots: ROC curve, PR curve, Confusion Matrix, t-SNE, Grad-CAM heatmaps
- Numerical JSON reports: All metrics saved as clean numbers
- Multi-model comparisons: Overlaid ROC & PR graphs, metric bar charts, Markdown comparison report

Usage Examples:

    # Evaluate multiple models and compare them side-by-side:
    python evaluate.py --val_root ./dataset/val \
        --wang2020_raw ./checkpoints/wang2020_raw.pth \
        --wang2020_128 ./checkpoints/wang2020_128.pth \
        --wolter2021_raw ./checkpoints/wolter2021_raw.pth \
        --wolter2021_128 ./checkpoints/wolter2021_128.pth \
        --xception ./checkpoints/xception.pth \
        --xception_128 ./checkpoints/xception_128.pth \
        --fusion ./checkpoints/fusion.pth \
        --mha ./checkpoints/mha.pth \
        --fusion_wwxc ./checkpoints/fusion_wwxc.pth \
        --mha_wwxc ./checkpoints/mha_wwxc.pth

    # Or pass custom model paths:
    python evaluate.py --val_root ./dataset/val \
        --model Wang2020Raw=./ckpts/best_wang.pth \
        --model MHA_128=./ckpts/best_mha.pth
"""

import os
import sys
import argparse
import torch

from evaluation import Evaluator, ModelComparisonEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate & Compare Deepfake Detection Models from Checkpoint (.pth) Files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Dataset arguments
    parser.add_argument('--val_root', '--dataroot', '--val_dir', dest='val_root', type=str, required=False, default=None,
                        help="Path to validation folder containing 'real' and 'fake' subfolders.")
    parser.add_argument('--tsne_val_root', '--tsne_dir', '--tsne_dataroot', dest='tsne_val_root', type=str, default=None,
                        help="Path to dedicated validation dataset folder for multi-category t-SNE (containing subfolders like DDPM, DDIM, DFDC, ADM, GAN, DiffSwap, Real).")

    # Named checkpoint paths for the 7 standard architectures
    model_group = parser.add_argument_group("Model Checkpoints (.pth files)")
    model_group.add_argument('--wang2020_raw', type=str, default=None,
                             help="Path to Wang2020Raw .pth checkpoint file")
    model_group.add_argument('--wang2020_128', type=str, default=None,
                             help="Path to Wang2020_128 .pth checkpoint file")
    model_group.add_argument('--wolter2021_raw', '--waveletwolter2021_raw', dest='wolter2021_raw',
                             type=str, default=None,
                             help="Path to WolterWavelet2021Raw .pth checkpoint file")
    model_group.add_argument('--wolter2021_128', '--waveletwolter2021_128', dest='wolter2021_128',
                             type=str, default=None,
                             help="Path to WolterWavelet2021_128 .pth checkpoint file")
    model_group.add_argument('--xception', '--xception_raw', '--sception', dest='xception',
                             type=str, default=None,
                             help="Path to XceptionRaw .pth checkpoint file")
    model_group.add_argument('--xception_128', dest='xception_128',
                             type=str, default=None,
                             help="Path to Xception_128 .pth checkpoint file")
    model_group.add_argument('--convnext_raw', '--convnext', dest='convnext_raw',
                             type=str, default=None,
                             help="Path to ConvNeXtRaw .pth checkpoint file")
    model_group.add_argument('--convnext_128', dest='convnext_128',
                             type=str, default=None,
                             help="Path to ConvNeXt_128 .pth checkpoint file")
    model_group.add_argument('--fusion', '--fusion_128', dest='fusion',
                             type=str, default=None,
                             help="Path to Fusion_128 .pth checkpoint file")
    model_group.add_argument('--mha', '--mha_128', dest='mha',
                             type=str, default=None,
                             help="Path to MHA_128 .pth checkpoint file")
    model_group.add_argument('--fusion_wwxc', dest='fusion_wwxc',
                             type=str, default=None,
                             help="Path to Fusion_WWXC .pth checkpoint file")
    model_group.add_argument('--mha_wwxc', dest='mha_wwxc',
                             type=str, default=None,
                             help="Path to MHA_WWXC .pth checkpoint file")

    # Flexible repeatable model argument e.g. --model Wang2020Raw=path.pth
    model_group.add_argument('--model', action='append', default=[],
                             help="Custom model entry formatted as Architecture=PathToPth (can be repeated)")

    # Single model fallback
    model_group.add_argument('--checkpoint', type=str, default=None,
                             help="Single checkpoint path if evaluating only one model")
    model_group.add_argument('--arch', type=str, default=None,
                             help="Architecture name when using --checkpoint")

    # Pipeline & Output configuration
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help="Directory to save evaluation reports, JSON numbers, and plots")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument('--device', type=str, default=None,
                        help="Target device (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument('--no_plots', action='store_true', default=False,
                        help="Disable generating plots (ROC, PR, Confusion Matrix)")
    parser.add_argument('--no_tsne', action='store_true', default=False,
                        help="Disable t-SNE embedding visualization")
    parser.add_argument('--no_gradcam', action='store_true', default=False,
                        help="Disable Grad-CAM heatmap visualization")
    parser.add_argument('--no_profiling', action='store_true', default=False,
                        help="Disable FLOPs and latency profiling")

    # WWXC/Fusion base model paths
    parser.add_argument('--rgb_model_path', type=str, default=None,
                        help="Path to pre-trained RGB/Wang2020_128 checkpoint for fusion/MHA models")
    parser.add_argument('--wavelet_model_path', type=str, default=None,
                        help="Path to pre-trained Wavelet/Wolter_128 checkpoint for fusion/MHA models")
    parser.add_argument('--xception_model_path', type=str, default=None,
                        help="Path to pre-trained Xception_128 checkpoint for WWXC models")
    parser.add_argument('--convnext_model_path', type=str, default=None,
                        help="Path to pre-trained ConvNeXt_128 checkpoint for WWXC models")

    return parser.parse_args()


def collect_models(args):
    """
    Collect all requested models from CLI arguments into a dict:
        { display_name: (checkpoint_path, architecture_name) }
    """
    models_dict = {}

    # Standard architecture flags
    named_models = [
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

    for display_name, ckpt_path, arch in named_models:
        if ckpt_path:
            if ckpt_path.lower() in ('none', 'pretrained', 'default', 'scratch'):
                models_dict[display_name] = (ckpt_path, arch)
            elif not os.path.isfile(ckpt_path):
                print(f"[WARNING] Checkpoint file not found for {display_name}: '{ckpt_path}'")
                print(f"          Resolved path checked: '{os.path.abspath(ckpt_path)}'")
                if ckpt_path.startswith('./kaggle'):
                    print(f"          Hint: Path starts with './kaggle'. Did you mean '/kaggle/...'?")
            else:
                models_dict[display_name] = (ckpt_path, arch)

    # Custom repeatable --model flags
    for entry in args.model:
        if '=' not in entry:
            print(f"[WARNING] Ignoring malformed --model entry '{entry}'. Expected format: Arch=Path.pth")
            continue
        arch_or_name, ckpt_path = entry.split('=', 1)
        arch_or_name = arch_or_name.strip()
        ckpt_path = ckpt_path.strip()
        if ckpt_path.lower() in ('none', 'pretrained', 'default', 'scratch'):
            models_dict[arch_or_name] = (ckpt_path, arch_or_name)
        elif not os.path.isfile(ckpt_path):
            print(f"[WARNING] Checkpoint file not found for --model {arch_or_name}: '{ckpt_path}'")
            print(f"          Resolved path checked: '{os.path.abspath(ckpt_path)}'")
            if ckpt_path.startswith('./kaggle'):
                print(f"          Hint: Path starts with './kaggle'. Did you mean '/kaggle/...'?")
        else:
            models_dict[arch_or_name] = (ckpt_path, arch_or_name)

    # Fallback to single --checkpoint if no specific models given
    if not models_dict and args.checkpoint:
        if args.checkpoint.lower() in ('none', 'pretrained', 'default', 'scratch') or os.path.isfile(args.checkpoint):
            arch = args.arch or 'DetectedArch'
            models_dict[arch] = (args.checkpoint, args.arch)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    return models_dict


def main():
    args = parse_args()

    if not args.val_root and not args.tsne_val_root:
        print("[ERROR] Please provide at least --val_root or --tsne_val_root!")
        sys.exit(1)

    if args.val_root and not os.path.exists(args.val_root):
        print(f"[ERROR] Validation dataset directory not found: {args.val_root}")
        sys.exit(1)

    if args.tsne_val_root and not os.path.exists(args.tsne_val_root):
        print(f"[ERROR] t-SNE validation dataset directory not found: {args.tsne_val_root}")
        sys.exit(1)

    models_dict = collect_models(args)
    if not models_dict:
        print("[ERROR] No valid checkpoint (.pth) files provided!")
        print("Please specify at least one model checkpoint using flags such as:")
        print("  --wang2020_raw <path.pth> or --wang2020_128 <path.pth> or --model Arch=<path.pth>")
        sys.exit(1)

    generate_plots = not args.no_plots
    collect_embeddings = not args.no_tsne
    generate_gradcam = not args.no_gradcam
    run_profiling = not args.no_profiling

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Collect overrides for base model paths
    overrides = {}
    if args.rgb_model_path:
        overrides['rgb_model_path'] = args.rgb_model_path
    if args.wavelet_model_path:
        overrides['wavelet_model_path'] = args.wavelet_model_path
    if args.xception_model_path:
        overrides['xception_model_path'] = args.xception_model_path
    if args.convnext_model_path:
        overrides['convnext_model_path'] = args.convnext_model_path

    if len(models_dict) == 1:
        # Single model evaluation
        display_name, (ckpt_path, arch) = next(iter(models_dict.items()))
        print(f"=== Single Model Evaluation: {display_name} ===")
        evaluator = Evaluator(
            checkpoint_path=ckpt_path,
            dataroot=args.val_root,
            output_dir=os.path.join(args.output_dir, display_name),
            arch=arch,
            batch_size=args.batch_size,
            device=device,
            collect_embeddings=collect_embeddings,
            run_profiling=run_profiling,
            generate_plots=generate_plots,
            generate_gradcam=generate_gradcam,
            tsne_dataroot=args.tsne_val_root
        )
        res = evaluator.run(opt_overrides=overrides)
        print(f"\nEvaluation Complete for {display_name}:")
        print(f"  - Accuracy: {res.metrics.get('accuracy', 0.0):.4f}")
        print(f"  - ROC AUC:  {res.metrics.get('roc_auc', 0.0):.4f}")
        print(f"  - F1 Score: {res.metrics.get('f1_score', 0.0):.4f}")
        print(f"  - EER:      {res.metrics.get('eer', 0.0):.4f}")
        if 'json' in res.report_paths:
            print(f"  - JSON Report: {res.report_paths['json']}")
        if 'tsne' in res.plot_paths:
            print(f"  - t-SNE Plot:  {res.plot_paths['tsne']}")
    else:
        # Multi-model evaluation & comparison
        comparator = ModelComparisonEvaluator(
            models_dict=models_dict,
            dataroot=args.val_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=device,
            collect_embeddings=collect_embeddings,
            run_profiling=run_profiling,
            generate_plots=generate_plots,
            tsne_dataroot=args.tsne_val_root
        )
        # Note: Evaluators created by ModelComparisonEvaluator inherit generate_gradcam setting
        results = comparator.run(opt_overrides=overrides)
        print("\n=== All Models Evaluated & Compared Successfully ===")


if __name__ == '__main__':
    main()
