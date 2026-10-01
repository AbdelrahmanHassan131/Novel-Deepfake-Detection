#!/usr/bin/env python
"""
Main Multi-Stage Fresh Training CLI.

Orchestrates the 3-stage training pipeline:
  1. Train RGB Expert (Wang2020_128) on training split.
  2. Train Wavelet Expert (WolterWavelet2021_128) on training split.
  3. Train Fusion Head (MHA_128 / Fusion_128) with frozen experts.

Supports:
  --plan_only: outputs pipeline_manifest.json and prints runnable commands without execution.
  --stage: execute or plan specific stage ('rgb', 'wavelet', 'fusion', or 'all').
  --resume_stage: resume an interrupted stage from a checkpoint.
"""
import argparse
import json
from pathlib import Path
import sys

from training.pipeline import FreshTrainingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Stage Fresh Deepfake Detection Pipeline")
    parser.add_argument('--experiment_dir', required=True, help="Base directory for the 3-stage pipeline")
    parser.add_argument('--dataroot', required=True, help="Dataset root containing images")
    parser.add_argument('--manifest', required=True, help="CSV manifest with explicit labels and groups")
    parser.add_argument('--val_manifest', default=None, help="Validation CSV manifest (defaults to --manifest)")
    parser.add_argument('--val_root', default=None, help="Validation images root (defaults to --dataroot)")
    parser.add_argument('--manifest_split', default='train', help="Split name for training")
    parser.add_argument('--val_manifest_split', default='dev', help="Split name for development validation")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension for experts and fusion")
    parser.add_argument('--fusion_type', default='token_attention', choices=['token_attention', 'gated', 'concat'],
                        help="Fusion strategy for Stage 3")
    parser.add_argument('--rgb_arch', default='Wang2020_128', help="RGB expert architecture")
    parser.add_argument('--wavelet_arch', default='WolterWavelet2021_128', help="Wavelet expert architecture")
    parser.add_argument('--fusion_arch', default=None,
                        help="Fusion architecture (defaults to Fusion_128 for concat/gated, MHA_128 for token_attention)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size per stage")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="Epochs per stage")
    parser.add_argument('--gpu_ids', default='0', help="GPU IDs (e.g., '0' or '-1' for CPU)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--run_id', default=None, help="Explicit immutable run identifier (defaults to seed{seed})")
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', default=True,
                        help="Train backbone from scratch without downloading ImageNet weights")
    parser.add_argument('--backbone_weights', default=None,
                        help="Path to local backbone weights for offline initialization")
    parser.add_argument('--wavelet_log_mode', default='signed_log1p', choices=['signed_log1p', 'legacy'],
                        help="Wavelet packet log transformation mode")
    parser.add_argument('--use_amp', action='store_true', default=False, help="Enable mixed precision")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Number of gradient accumulation steps (simulates larger batch size)")
    parser.add_argument('--stage', default='all', choices=['all', 'rgb', 'wavelet', 'fusion'],
                        help="Which stage to target")
    parser.add_argument('--plan_only', action='store_true',
                        help="Generate commands and manifest without starting execution")
    parser.add_argument('--resume_checkpoint', default=None,
                        help="Checkpoint to resume from when resuming an individual stage")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = FreshTrainingPipeline(
        experiment_dir=args.experiment_dir,
        dataroot=args.dataroot,
        manifest=args.manifest,
        val_manifest=args.val_manifest,
        val_root=args.val_root,
        manifest_split=args.manifest_split,
        val_manifest_split=args.val_manifest_split,
        embed_dim=args.embed_dim,
        fusion_type=args.fusion_type,
        rgb_arch=args.rgb_arch,
        wavelet_arch=args.wavelet_arch,
        fusion_arch=args.fusion_arch,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        gpu_ids=args.gpu_ids,
        seed=args.seed,
        pretrained=args.pretrained,
        backbone_weights=args.backbone_weights,
        wavelet_log_mode=args.wavelet_log_mode,
        use_amp=args.use_amp,
        grad_accum_steps=args.grad_accum_steps,
        run_id=args.run_id,
    )

    stages_to_run = ['rgb', 'wavelet', 'fusion'] if args.stage == 'all' else [args.stage]

    print("=" * 70)
    print(f"      FRESH TRAINING PIPELINE: 3-STAGE PROTOCOL {'(PLAN ONLY)' if args.plan_only else ''}")
    print("=" * 70)

    print("\nPlanned commands:")
    for s in stages_to_run:
        spec = pipeline.stages[s]
        cmd_str = ' '.join(spec.command)
        print(f"\n[Stage {spec.stage_num}: {spec.name}]")
        print(f"Command:\n  {cmd_str}")

    if args.plan_only:
        plan_path = Path(args.experiment_dir) / "pipeline_plan.json"
        pipeline.save_manifest(path=plan_path)
        print(f"\n[PLAN ONLY] Plan written to: {plan_path}")
        print("Commands generated and verified. Ready for execution on Colab/GPU.")
        return

    manifest_path = pipeline.save_manifest()
    print(f"\nManifest written to: {manifest_path}")

    print("\nStarting multi-stage pipeline execution...")
    try:
        pipeline.run_stages(stages_to_run, resume_checkpoint=args.resume_checkpoint)
        print("\n[SUCCESS] All requested pipeline stages completed successfully.")
    except Exception as exc:
        print(f"\n[ERROR] Pipeline execution failed: {exc}", file=sys.stderr)
        sys.exit(1)



if __name__ == '__main__':
    main()
