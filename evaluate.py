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
        --fusion ./checkpoints/fusion.pth \
        --mha ./checkpoints/mha.pth

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
    parser.add_argument('--val_root', '--dataroot', '--val_dir', dest='val_root', type=str, required=True,
                        help="Path to validation folder containing 'real' and 'fake' subfolders.")

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
    model_group.add_argument('--fusion', '--fusion_128', dest='fusion',
                             type=str, default=None,
                             help="Path to Fusion_128 .pth checkpoint file")
    model_group.add_argument('--mha', '--mha_128', dest='mha',
                             type=str, default=None,
                             help="Path to MHA_128 .pth checkpoint file")

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
        ('Fusion_128', args.fusion, 'Fusion_128'),
        ('MHA_128', args.mha, 'MHA_128'),
    ]

    for display_name, ckpt_path, arch in named_models:
        if ckpt_path:
            if not os.path.isfile(ckpt_path):
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
        if not os.path.isfile(ckpt_path):
            print(f"[WARNING] Checkpoint file not found for --model {arch_or_name}: '{ckpt_path}'")
            print(f"          Resolved path checked: '{os.path.abspath(ckpt_path)}'")
            if ckpt_path.startswith('./kaggle'):
                print(f"          Hint: Path starts with './kaggle'. Did you mean '/kaggle/...'?")
        else:
            models_dict[arch_or_name] = (ckpt_path, arch_or_name)

    # Fallback to single --checkpoint if no specific models given
    if not models_dict and args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        arch = args.arch or 'DetectedArch'
        models_dict[arch] = (args.checkpoint, args.arch)

    return models_dict


def main():
    args = parse_args()

    if not os.path.exists(args.val_root):
        print(f"[ERROR] Validation dataset directory not found: {args.val_root}")
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
            generate_gradcam=generate_gradcam
        )
        res = evaluator.run()
        print(f"\nEvaluation Complete for {display_name}:")
        print(f"  - Accuracy: {res.metrics.get('accuracy', 0.0):.4f}")
        print(f"  - ROC AUC:  {res.metrics.get('roc_auc', 0.0):.4f}")
        print(f"  - F1 Score: {res.metrics.get('f1_score', 0.0):.4f}")
        print(f"  - EER:      {res.metrics.get('eer', 0.0):.4f}")
        print(f"  - JSON Report: {res.report_paths['json']}")
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
            generate_plots=generate_plots
        )
        # Note: Evaluators created by ModelComparisonEvaluator inherit generate_gradcam setting
        results = comparator.run()
        print("\n=== All Models Evaluated & Compared Successfully ===")


if __name__ == '__main__':
    main()
