"""CLI entry point for evaluating baseline models on benchmark manifests.

Uses lazy initialization. Does NOT download weights or run computation on import.
"""
import argparse
import json
from pathlib import Path
from models.baselines import BASELINE_REGISTRY, get_baseline_adapter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--baseline',
        required=True,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Name of baseline method to evaluate."
    )
    parser.add_argument('--checkpoint', default=None, help="Path to baseline checkpoint weights (.pth / .pt)")
    parser.add_argument('--manifest', default=None, help="CSV manifest with explicit labels and groups")
    parser.add_argument('--dataroot', default=None, help="Root folder of image dataset")
    parser.add_argument('--split', default='external_test', help="Manifest split to evaluate on")
    parser.add_argument('--evaluation_mode', default='supplied_checkpoint', choices=['supplied_checkpoint', 'retrained_on_pilot'])
    parser.add_argument('--output_predictions', default=None, help="Output CSV path for predictions")
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--info', action='store_true', help="Print baseline metadata, license, and preprocessing and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = get_baseline_adapter(
        name=args.baseline,
        checkpoint_path=args.checkpoint,
        evaluation_mode=args.evaluation_mode,
        device=args.device
    )

    if args.info:
        print("=" * 70)
        print(f"BASELINE METADATA: {adapter.name}")
        print("=" * 70)
        print(json.dumps(adapter.metadata(), indent=2))
        return

    if adapter.status != 'ready_for_runtime':
        print(f"[ERROR] Baseline {adapter.name} is in status '{adapter.status}'.")
        print("Official checkpoint or configuration details could not be verified.")
        return

    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required to evaluate baseline {adapter.name}.")

    if not args.manifest or not args.dataroot or not args.output_predictions:
        raise ValueError("--manifest, --dataroot, and --output_predictions are required for evaluation.")

    adapter.predict_manifest(
        manifest_path=args.manifest,
        dataroot=args.dataroot,
        output_predictions_path=args.output_predictions,
        split=args.split,
    )


if __name__ == '__main__':
    main()
