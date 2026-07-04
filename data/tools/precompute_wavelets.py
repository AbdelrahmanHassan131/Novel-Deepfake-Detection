"""
Offline wavelet precomputation tool.

Computes wavelet packet coefficients for all images in a dataset
and saves them as .npy files, preserving the original folder structure.

Supports resumable execution — existing .npy files are skipped automatically.

Usage:
    python Refactored/data/tools/precompute_wavelets.py \
        --input dataset/train \
        --output wavelets/train \
        --wavelet_type haar \
        --wavelet_level 3 \
        --wavelet_mode reflect

    # With resize and crop:
    python Refactored/data/tools/precompute_wavelets.py \
        --input dataset/train \
        --output wavelets/train \
        --load_size 128 \
        --crop_size 128
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Import the existing Refactored wavelet implementation.
# Adds the project root to sys.path so that data.wavelets is reachable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.wavelets.packet_transform import compute_wavelet_packet_coeffs, log_scale_packets

# ---------------------------------------------------------------------------
# Supported image file extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def find_images(input_dir):
    """Recursively find all image files under *input_dir*."""
    images = []
    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, fname))
    return images


def get_npy_path(image_path, input_dir, output_dir):
    """Map an image path to its corresponding .npy output path."""
    rel = os.path.relpath(image_path, input_dir)
    npy_rel = os.path.splitext(rel)[0] + '.npy'
    return os.path.join(output_dir, npy_rel)


def process_image(image_path, npy_path, wavelet_type, wavelet_level,
                  wavelet_mode, use_log_packets, load_size, crop_size):
    """Load one image, compute wavelet packets, save as float32 .npy."""
    img = Image.open(image_path).convert('RGB')

    # Optional deterministic resize
    if load_size is not None:
        img = img.resize((load_size, load_size), Image.BILINEAR)

    # Optional deterministic centre crop
    if crop_size is not None:
        w, h = img.size
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))

    img_array = np.array(img)

    coeffs = compute_wavelet_packet_coeffs(
        img_array,
        wavelet=wavelet_type,
        level=wavelet_level,
        mode=wavelet_mode,
    )

    if use_log_packets:
        coeffs = log_scale_packets(coeffs)

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)

    # Save as uncompressed float32
    np.save(npy_path, coeffs.astype(np.float32))


def write_metadata(output_dir, wavelet_type, wavelet_level, wavelet_mode):
    """Write a minimal metadata.json to the output root."""
    metadata = {
        'wavelet_type': wavelet_type,
        'wavelet_level': wavelet_level,
        'wavelet_mode': wavelet_mode,
        'generation_date': datetime.now().isoformat(),
    }
    path = os.path.join(output_dir, 'metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute wavelet packet coefficients for a dataset.')
    parser.add_argument('--input', required=True,
                        help='Input dataset directory containing images')
    parser.add_argument('--output', required=True,
                        help='Output directory for .npy files')
    parser.add_argument('--wavelet_type', default='haar',
                        help='Wavelet type (default: haar)')
    parser.add_argument('--wavelet_level', type=int, default=3,
                        help='Decomposition level (default: 3)')
    parser.add_argument('--wavelet_mode', default='reflect',
                        help='Signal extension mode (default: reflect)')
    parser.add_argument('--no_log_packets', action='store_true',
                        help='Disable log scaling (enabled by default)')
    parser.add_argument('--load_size', type=int, default=None,
                        help='Resize images to this square size before '
                             'computing wavelets (optional)')
    parser.add_argument('--crop_size', type=int, default=None,
                        help='Centre-crop images to this square size after '
                             'resize (optional)')
    args = parser.parse_args()

    use_log_packets = not args.no_log_packets
    input_dir = os.path.normpath(args.input)
    output_dir = os.path.normpath(args.output)

    # ------------------------------------------------------------------
    # 1. Discover all images
    # ------------------------------------------------------------------
    print(f"Scanning {input_dir} for images...")
    all_images = find_images(input_dir)
    total = len(all_images)
    print(f"Found {total} images")

    if total == 0:
        print("No images found. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Separate already-processed files (resume support)
    # ------------------------------------------------------------------
    to_process = []
    skipped = 0
    for img_path in all_images:
        npy_path = get_npy_path(img_path, input_dir, output_dir)
        if os.path.exists(npy_path):
            skipped += 1
        else:
            to_process.append((img_path, npy_path))

    remaining = len(to_process)
    print(f"Skipping {skipped} already processed files")
    print(f"Processing {remaining} remaining files")
    print(f"Config: wavelet={args.wavelet_type}, level={args.wavelet_level}, "
          f"mode={args.wavelet_mode}, log_packets={use_log_packets}")
    if args.load_size is not None:
        print(f"  load_size={args.load_size}")
    if args.crop_size is not None:
        print(f"  crop_size={args.crop_size}")

    if remaining == 0:
        print("All files already processed.")
        write_metadata(output_dir, args.wavelet_type, args.wavelet_level,
                       args.wavelet_mode)
        return

    # ------------------------------------------------------------------
    # 3. Process remaining images
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    errors = 0
    start_time = time.time()

    for img_path, npy_path in to_process:
        try:
            process_image(
                img_path, npy_path,
                args.wavelet_type, args.wavelet_level, args.wavelet_mode,
                use_log_packets, args.load_size, args.crop_size,
            )
            processed += 1
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            errors += 1

        # Progress display
        done = skipped + processed + errors
        elapsed = time.time() - start_time
        rate = (processed + errors) / elapsed if elapsed > 0 else 0
        eta = (remaining - processed - errors) / rate if rate > 0 else 0
        print(
            f"\rProgress: {done}/{total} | "
            f"Processed: {processed} | Skipped: {skipped} | "
            f"Errors: {errors} | "
            f"{rate:.1f} img/s | ETA: {eta:.0f}s",
            end='', flush=True,
        )

    # Final summary
    print()
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")

    # ------------------------------------------------------------------
    # 4. Write metadata
    # ------------------------------------------------------------------
    write_metadata(output_dir, args.wavelet_type, args.wavelet_level,
                   args.wavelet_mode)


if __name__ == '__main__':
    main()
