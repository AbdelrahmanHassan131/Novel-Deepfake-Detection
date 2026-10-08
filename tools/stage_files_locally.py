"""
Local Dataset Staging Utility.

Stages selected images from slow or network storage (e.g. network share,
Kaggle /kaggle/input, Google Drive) to fast local storage (e.g. /kaggle/working,
local NVMe SSD, or ramdisk) before training.

Safety features:
1. Strict pre-flight disk space headroom check (>= 1.2x required bytes).
2. Content verification (bytes copied and optional SHA-256 digest checks).
3. Produces a relocated manifest CSV pointing to the staged local files.
4. Fast dry-run mode for pre-flight space auditing.

Usage::

    # Dry-run check for pilot manifest
    python tools/stage_files_locally.py \\
        --manifest manifests/pilot_wang2020.csv \\
        --dest_dir /kaggle/working/staged_pilot \\
        --output_manifest manifests/pilot_wang2020_staged.csv \\
        --dry_run

    # Stage pilot files with digest verification
    python tools/stage_files_locally.py \\
        --manifest manifests/pilot_wang2020.csv \\
        --dest_dir /kaggle/working/staged_pilot \\
        --output_manifest manifests/pilot_wang2020_staged.csv \\
        --verify_digest
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.manifest import read_manifest, write_manifest


def sha256_file(path, chunk_size=65536):
    """Compute sha256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def check_disk_space(target_dir, required_bytes, headroom_ratio=1.2):
    """
    Ensure the target filesystem has at least headroom_ratio * required_bytes free.

    Raises RuntimeError if insufficient space.
    """
    target_path = Path(target_dir).resolve()
    # Walk up until finding an existing directory
    check_dir = target_path
    while not check_dir.exists() and check_dir.parent != check_dir:
        check_dir = check_dir.parent

    usage = shutil.disk_usage(str(check_dir))
    free_bytes = usage.free
    min_required = int(required_bytes * headroom_ratio)

    if free_bytes < min_required:
        raise RuntimeError(
            f"Insufficient disk space in {check_dir}:\n"
            f"  Required for images:  {required_bytes / 1e6:,.2f} MB\n"
            f"  Headroom requirement ({headroom_ratio:.1f}x): {min_required / 1e6:,.2f} MB\n"
            f"  Available free space: {free_bytes / 1e6:,.2f} MB\n"
            f"  Deficit:              {(min_required - free_bytes) / 1e6:,.2f} MB"
        )
    return free_bytes


def stage_dataset(manifest_path, dest_dir, output_manifest=None,
                  split=None, max_samples=None, verify_digest=False,
                  dry_run=False, root=None, remap_prefixes=None):
    """
    Stage images referenced in manifest_path into dest_dir.

    Returns:
        dict summary of staging operation.
    """
    start_time = time.monotonic()
    rows = read_manifest(manifest_path, root=root, split=split, check_files=True, remap_prefixes=remap_prefixes)

    if max_samples is not None and len(rows) > max_samples:
        rows = rows[:max_samples]

    dest_path = Path(dest_dir).resolve()
    print(f"[Stage] Auditing {len(rows):,} samples for local staging to: {dest_path}")

    total_bytes = 0
    file_info = []
    for r in rows:
        p = Path(r['path'])
        sz = p.stat().st_size
        total_bytes += sz
        file_info.append((r, p, sz))

    print(f"[Stage] Total size of source files: {total_bytes / 1e6:,.2f} MB ({total_bytes:,} bytes)")

    free_bytes = check_disk_space(dest_path, total_bytes, headroom_ratio=1.2)
    print(f"[Stage] Disk space check PASSED: Free={free_bytes / 1e6:,.2f} MB (Headroom={free_bytes / max(1, total_bytes):.2f}x)")

    if dry_run:
        print("[Stage] Dry run complete. No files were copied.")
        return {
            'status': 'dry_run_success',
            'sample_count': len(rows),
            'total_bytes': total_bytes,
            'free_bytes': free_bytes,
        }

    dest_path.mkdir(parents=True, exist_ok=True)
    staged_rows = []
    copied_bytes = 0

    print("[Stage] Copying files...")
    copy_start = time.monotonic()

    for idx, (row, src_path, sz) in enumerate(file_info, 1):
        # Create deterministic subpath to avoid filename collisions across directories
        # Use sample_id prefix with sanitized original extension
        ext = src_path.suffix.lower() or '.png'
        staged_name = f"{row['sample_id']}_{src_path.name}"
        staged_file = dest_path / staged_name

        shutil.copy2(src_path, staged_file)
        copied_bytes += sz

        if verify_digest:
            actual_digest = sha256_file(staged_file)
            expected_digest = row.get('sha256')
            if expected_digest:
                if actual_digest != expected_digest:
                    raise ValueError(
                        f"Checksum mismatch on {staged_file}: "
                        f"expected {expected_digest}, got {actual_digest}"
                    )

        updated_row = dict(row)
        updated_row['path'] = str(staged_file)
        staged_rows.append(updated_row)

        if idx % 1000 == 0 or idx == len(file_info):
            elapsed = time.monotonic() - copy_start
            rate_mb = (copied_bytes / 1e6) / max(0.001, elapsed)
            print(f"[Stage] Copied {idx:,}/{len(file_info):,} ({copied_bytes / 1e6:,.1f} MB) - Rate: {rate_mb:.2f} MB/s", flush=True)

    elapsed_total = time.monotonic() - start_time
    print(f"[Stage] Staging completed in {elapsed_total:.2f}s ({copied_bytes / 1e6:,.2f} MB)")

    if output_manifest:
        out_man_p = Path(output_manifest).resolve()
        write_manifest(out_man_p, staged_rows)
        print(f"[Stage] Wrote staged manifest to: {out_man_p}")

    return {
        'status': 'success',
        'sample_count': len(staged_rows),
        'total_bytes': copied_bytes,
        'elapsed_seconds': elapsed_total,
        'rate_mb_s': (copied_bytes / 1e6) / max(0.001, elapsed_total),
        'output_manifest': str(output_manifest) if output_manifest else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage dataset files locally to fast storage.")
    parser.add_argument('--manifest', type=str, required=True, help="Input manifest CSV path")
    parser.add_argument('--dest_dir', type=str, required=True, help="Directory to stage images into")
    parser.add_argument('--output_manifest', type=str, default=None, help="Path for output manifest pointing to staged files")
    parser.add_argument('--split', type=str, default=None, help="Filter to specific split (e.g. train or val)")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum samples to stage")
    parser.add_argument('--verify_digest', action='store_true', help="Verify SHA-256 digests against manifest")
    parser.add_argument('--dry_run', action='store_true', help="Check disk space without copying files")
    parser.add_argument('--root', type=str, default=None, help="Dataset root for resolving relative paths")
    parser.add_argument('--remap_path_prefix', action='append', default=[], help="Old:New path prefix remappings")

    args = parser.parse_args()

    remap_dict = {}
    for item in args.remap_path_prefix:
        if ':' in item:
            old, new = item.split(':', 1)
            remap_dict[old] = new

    stage_dataset(
        manifest_path=args.manifest,
        dest_dir=args.dest_dir,
        output_manifest=args.output_manifest,
        split=args.split,
        max_samples=args.max_samples,
        verify_digest=args.verify_digest,
        dry_run=args.dry_run,
        root=args.root,
        remap_prefixes=remap_dict,
    )


if __name__ == '__main__':
    main()
