#!/usr/bin/env python
"""
Bounded Image Profile Tool for Forensic Dataset Cohorts.

Inspects a seeded sample (default <= 1000 total images) of a dataset manifest.
Reports:
    - Image container format (JPEG, PNG, WebP, etc.)
    - Pixel dimensions (width, height)
    - Aspect ratio (width / height)
    - Color mode (RGB, L, RGBA, CMYK)
    - File size in bytes
    - Basic pixel quality statistics (intensity mean, std, dynamic range)
    - Decode failures (unreadable or corrupted files)

Aggregates statistics per class (real vs fake) and per dataset source.
Emits potential packaging shortcut hypotheses (e.g. format or resolution imbalance).

CRITICAL CONSTRAINTS:
    - Must NOT infer demographic, identity, or source labels from appearance.
    - Findings are hypotheses regarding packaging discrepancies, NOT automated proof of label errors.
    - Strictly bounds sample size to prevent scanning unbounded image pools.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.manifest import read_manifest, known


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile a bounded sample of dataset images to inspect packaging artifacts and properties."
    )
    parser.add_argument('--manifest', type=str, required=True,
                        help="Path to manifest CSV.")
    parser.add_argument('--root', type=str, default=None,
                        help="Dataset root directory (defaults to manifest parent).")
    parser.add_argument('--sample_size', type=int, default=500,
                        help="Total number of images to profile (default: 500, capped at 1000).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for deterministic cohort sampling.")
    parser.add_argument('--output', type=str, default='image_profile_report.json',
                        help="Path to save profile JSON report.")
    parser.add_argument('--remap_path_prefix', type=str, nargs='*', default=[],
                        help="Old and new path prefixes separated by = (e.g. C:/old=/new/path).")
    return parser.parse_args()


def inspect_single_image(path: Path) -> Dict[str, Any]:
    """Inspect format, dimensions, mode, size, and basic pixel stats safely."""
    info = {
        'path': str(path),
        'exists': path.is_file(),
        'decode_success': False,
        'format': 'unknown',
        'width': 0,
        'height': 0,
        'aspect_ratio': 0.0,
        'color_mode': 'unknown',
        'file_size_bytes': 0,
        'pixel_mean': None,
        'pixel_std': None,
        'error': None,
    }
    if not info['exists']:
        info['error'] = 'file_not_found'
        return info

    info['file_size_bytes'] = os.path.getsize(path)

    try:
        with Image.open(path) as img:
            info['format'] = img.format or 'unknown'
            info['width'], info['height'] = img.size
            info['aspect_ratio'] = round(info['width'] / max(1, info['height']), 4)
            info['color_mode'] = img.mode

            # Lightweight sample of pixel statistics without loading full huge arrays
            # Resize a thumbnail copy in memory for basic intensity stats
            thumb = img.copy().convert('RGB')
            thumb.thumbnail((64, 64), Image.Resampling.NEAREST)
            arr = np.asarray(thumb, dtype=np.float32)
            info['pixel_mean'] = round(float(np.mean(arr)), 2)
            info['pixel_std'] = round(float(np.std(arr)), 2)
            info['decode_success'] = True
    except Exception as e:
        info['error'] = str(e)

    return info


def generate_hypotheses(class_stats: Dict[int, Dict[str, Any]], source_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    """Formulate testable hypotheses regarding packaging discrepancies.

    Findings are hypotheses, not automated proof of label errors.
    """
    hypotheses = []

    real_stats = class_stats.get(0, {})
    fake_stats = class_stats.get(1, {})

    if real_stats and fake_stats:
        # Check container format imbalance
        real_formats = real_stats.get('format_percentages', {})
        fake_formats = fake_stats.get('format_percentages', {})

        for fmt, p_fake in fake_formats.items():
            p_real = real_formats.get(fmt, 0.0)
            if abs(p_fake - p_real) > 40.0:
                hypotheses.append(
                    f"Hypothesis [Format Packaging Shortcut]: Format '{fmt}' accounts for {p_fake:.1f}% of fake samples "
                    f"vs {p_real:.1f}% of real samples. A detector trained without compression augmentation "
                    "may learn container/compression signatures instead of facial manipulation artifacts."
                )

        # Check dimension / resolution imbalance
        real_res = real_stats.get('mean_resolution', [0, 0])
        fake_res = fake_stats.get('mean_resolution', [0, 0])
        if real_res[0] > 0 and fake_res[0] > 0:
            res_ratio = (fake_res[0] * fake_res[1]) / (real_res[0] * real_res[1])
            if res_ratio < 0.5 or res_ratio > 2.0:
                hypotheses.append(
                    f"Hypothesis [Resolution Shortcut]: Significant mean resolution divergence between classes "
                    f"(real: {real_res[0]}x{real_res[1]}, fake: {fake_res[0]}x{fake_res[1]}). "
                    "Resizing all images to 256x256 could introduce frequency domain cues if downscaling factors differ."
                )

        # Check aspect ratio discrepancy
        real_ar = real_stats.get('mean_aspect_ratio', 1.0)
        fake_ar = fake_stats.get('mean_aspect_ratio', 1.0)
        if abs(real_ar - fake_ar) > 0.2:
            hypotheses.append(
                f"Hypothesis [Aspect Ratio Divergence]: Mean aspect ratio differs (real: {real_ar:.2f}, fake: {fake_ar:.2f}). "
                "Cropping or non-uniform scaling may introduce geometric shortcuts."
            )

    return hypotheses


def aggregate_group_metrics(inspections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate format, size, resolution and decode metrics for a group of images."""
    total = len(inspections)
    if not total:
        return {}

    successful = [item for item in inspections if item['decode_success']]
    failures = [item for item in inspections if not item['decode_success']]

    format_counts = Counter(item['format'] for item in successful)
    mode_counts = Counter(item['color_mode'] for item in successful)

    widths = [item['width'] for item in successful]
    heights = [item['height'] for item in successful]
    aspect_ratios = [item['aspect_ratio'] for item in successful]
    file_sizes = [item['file_size_bytes'] for item in inspections if item['exists']]

    return {
        'total_inspected': total,
        'decode_successes': len(successful),
        'decode_failures': len(failures),
        'failure_rate_pct': round((len(failures) / total) * 100, 2),
        'formats': dict(format_counts),
        'format_percentages': {fmt: round((c / max(1, len(successful))) * 100, 1) for fmt, c in format_counts.items()},
        'color_modes': dict(mode_counts),
        'mean_resolution': [int(np.mean(widths)), int(np.mean(heights))] if widths else [0, 0],
        'mean_aspect_ratio': round(float(np.mean(aspect_ratios)), 3) if aspect_ratios else 1.0,
        'mean_file_size_kb': round(float(np.mean(file_sizes)) / 1024, 1) if file_sizes else 0.0,
        'median_file_size_kb': round(float(np.median(file_sizes)) / 1024, 1) if file_sizes else 0.0,
    }


def main():
    args = parse_args()
    sample_bound = min(1000, max(10, args.sample_size))

    remap_prefixes = {}
    if args.remap_path_prefix:
        for item in args.remap_path_prefix:
            if '=' in item:
                old_p, new_p = item.split('=', 1)
                remap_prefixes[old_p] = new_p

    print(f"Reading manifest {args.manifest}...", flush=True)
    rows = read_manifest(args.manifest, root=args.root, check_files=False, remap_prefixes=remap_prefixes)

    # Stratified seeded sampling across (dataset_source, label)
    buckets = defaultdict(list)
    for r in rows:
        key = (r.get('dataset_source', 'unknown'), int(r['label']))
        buckets[key].append(r)

    rng = np.random.default_rng(args.seed)
    sampled_rows = []
    per_bucket_target = max(1, sample_bound // len(buckets))

    for key, b_rows in sorted(buckets.items()):
        indices = rng.choice(len(b_rows), size=min(len(b_rows), per_bucket_target), replace=False)
        for idx in indices:
            sampled_rows.append(b_rows[idx])

    # If rounding left room, fill up to sample_bound
    if len(sampled_rows) < sample_bound and len(sampled_rows) < len(rows):
        remaining = [r for r in rows if r not in sampled_rows]
        fill_idx = rng.choice(len(remaining), size=min(len(remaining), sample_bound - len(sampled_rows)), replace=False)
        for idx in fill_idx:
            sampled_rows.append(remaining[idx])

    print(f"Profiling {len(sampled_rows):,} sampled images (bound: {sample_bound})...", flush=True)

    inspections = []
    class_inspections = defaultdict(list)
    source_inspections = defaultdict(list)

    for i, row in enumerate(sampled_rows, 1):
        info = inspect_single_image(Path(row['path']))
        info['sample_id'] = row['sample_id']
        info['label'] = int(row['label'])
        info['dataset_source'] = row.get('dataset_source', 'unknown')

        inspections.append(info)
        class_inspections[info['label']].append(info)
        source_inspections[info['dataset_source']].append(info)

        if i % 100 == 0 or i == len(sampled_rows):
            print(f"Inspected {i}/{len(sampled_rows)} images...", flush=True)

    overall_metrics = aggregate_group_metrics(inspections)
    by_class = {lbl: aggregate_group_metrics(items) for lbl, items in class_inspections.items()}
    by_source = {src: aggregate_group_metrics(items) for src, items in source_inspections.items()}

    hypotheses = generate_hypotheses(by_class, by_source)

    report = {
        'manifest': str(Path(args.manifest).resolve()),
        'sample_bound_requested': sample_bound,
        'images_profiled': len(inspections),
        'seed': args.seed,
        'overall': overall_metrics,
        'by_class': {
            'real (0)': by_class.get(0, {}),
            'fake (1)': by_class.get(1, {}),
        },
        'by_source': by_source,
        'hypotheses_and_findings': hypotheses,
        'disclaimer': (
            'Findings are hypotheses regarding image packaging discrepancies, format imbalance, '
            'and resolution differences. They do not constitute automated proof of label errors '
            'and must not infer demographic or identity characteristics from appearance.'
        ),
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"\nImage profile report saved to: {out_path}")
    print(f"Decode success rate: {overall_metrics.get('decode_successes', 0)}/{len(inspections)}")
    if hypotheses:
        print("\nIdentified Packaging Hypotheses:")
        for h in hypotheses:
            print(f"  - {h}")


if __name__ == '__main__':
    main()
