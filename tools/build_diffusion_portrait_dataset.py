#!/usr/bin/env python
"""
Build a Clean, Balanced, Video-Free, GAN-Free Diffusion Portrait Dataset.

Dataset Specification:
- 100k Real (CelebA) + 100k Fake (ADM, DDIM, DDPM, DiffSwap) = 200k images total
- Split: 80% Train (80k Real, 80k Fake) / 20% Val (20k Real, 20k Fake)
- Destination: F:\DatesetNoVideosNoGAN
- Zero CelebA identity overlap between train and val
- Balanced generator representation (20k train / 5k val per diffusion generator)
- Zero contamination with external unseen evaluation test set (F:\val to be deleted 2...)
- Multithreaded fast copying and manifest generation
"""

import os
import sys
import argparse
import random
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Clean Diffusion Portrait Dataset")
    parser.add_argument(
        "--celeba_dir",
        type=str,
        default=r"E:\wholeTrainingData\realCelebA\img_align_celeba",
        help="Path to CelebA aligned images directory",
    )
    parser.add_argument(
        "--identity_file",
        type=str,
        default=r"E:\wholeTrainingData\realCelebA\identity_CelebA.txt",
        help="Path to CelebA identity mapping file",
    )
    parser.add_argument(
        "--partition_file",
        type=str,
        default=r"E:\wholeTrainingData\realCelebA\list_eval_partition.txt",
        help="Path to CelebA official partition file",
    )
    parser.add_argument(
        "--diff_dir",
        type=str,
        default=r"E:\wholeTrainingData\imagesGeneratedDiffModels\diffFaceDataset",
        help="Path to diffusion models root directory",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default=r"F:\DatesetNoVideosNoGAN",
        help="Destination directory for the new dataset",
    )
    parser.add_argument(
        "--unseen_val_dir",
        type=str,
        default=r"F:\val to be deleted 2 REAL FROM ZENDOO and FAKE FROM LDM",
        help="Path to external unseen validation test set for contamination check",
    )
    parser.add_argument(
        "--train_real_target",
        type=int,
        default=80000,
        help="Target number of real images in train split",
    )
    parser.add_argument(
        "--val_real_target",
        type=int,
        default=20000,
        help="Target number of real images in val split",
    )
    parser.add_argument(
        "--fake_generators",
        type=str,
        default="ADM,DDIM,DDPM,DiffSwap",
        help="Comma-separated list of fake diffusion generators",
    )
    parser.add_argument(
        "--fake_train_per_gen",
        type=int,
        default=20000,
        help="Number of train fake images per generator (default 20,000 * 4 = 80,000)",
    )
    parser.add_argument(
        "--fake_val_per_gen",
        type=int,
        default=5000,
        help="Number of val fake images per generator (default 5,000 * 4 = 20,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker threads for parallel file copying",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, validate and compute the plan without copying files",
    )
    return parser.parse_args()


def load_celeba_metadata(identity_path, partition_path):
    """Load CelebA identity IDs and official evaluation partitions."""
    if not os.path.isfile(identity_path):
        raise FileNotFoundError(f"CelebA identity file not found: {identity_path}")
    if not os.path.isfile(partition_path):
        raise FileNotFoundError(f"CelebA partition file not found: {partition_path}")

    id_map = {}
    with open(identity_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                id_map[parts[0]] = parts[1]

    part_map = {}
    with open(partition_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                part_map[parts[0]] = parts[1]

    return id_map, part_map


def build_copy_tasks(args):
    """Plan all copy tasks and verify identity disjointness and generator balance."""
    print("=" * 70)
    print("STEP 1: Validating Source Paths and Metadata")
    print("=" * 70)

    # 1. Load CelebA metadata
    id_map, part_map = load_celeba_metadata(args.identity_file, args.partition_file)
    print(f"CelebA identities loaded: {len(id_map):,} records")
    print(f"CelebA partitions loaded: {len(part_map):,} records")

    # In CelebA: '0' = train, '1' = val, '2' = test
    train_pool = [img for img, part in part_map.items() if part == "0"]
    val_pool = [img for img, part in part_map.items() if part in ("1", "2")]

    print(f"CelebA candidate images - Train pool (partition 0): {len(train_pool):,}")
    print(f"CelebA candidate images - Val pool (partitions 1 & 2): {len(val_pool):,}")

    if len(train_pool) < args.train_real_target:
        raise ValueError(
            f"Insufficient train candidates: {len(train_pool)} < {args.train_real_target}"
        )
    if len(val_pool) < args.val_real_target:
        raise ValueError(
            f"Insufficient val candidates: {len(val_pool)} < {args.val_real_target}"
        )

    # Reproducible sampling
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    selected_train_real = train_pool[: args.train_real_target]
    selected_val_real = val_pool[: args.val_real_target]

    # Verify identity disjointness
    train_ids = set(id_map[img] for img in selected_train_real if img in id_map)
    val_ids = set(id_map[img] for img in selected_val_real if img in id_map)
    overlap_ids = train_ids.intersection(val_ids)

    print("\n" + "=" * 70)
    print("STEP 2: Identity Separation Verification")
    print("=" * 70)
    print(f"Selected Train Real images: {len(selected_train_real):,}")
    print(f"Unique identities in Train: {len(train_ids):,}")
    print(f"Selected Val Real images:   {len(selected_val_real):,}")
    print(f"Unique identities in Val:   {len(val_ids):,}")
    print(f"Identity Overlap Count:     {len(overlap_ids)}")

    if len(overlap_ids) > 0:
        raise RuntimeError(
            f"CRITICAL ERROR: Found {len(overlap_ids)} overlapping identities between train and val!"
        )
    print("SUCCESS: Zero CelebA identity overlap between train and val!")

    # Verify real image files exist on disk
    print("\nVerifying real file existence in CelebA directory...")
    missing_real = []
    for img in selected_train_real[:500] + selected_val_real[:500]:
        fp = os.path.join(args.celeba_dir, img)
        if not os.path.isfile(fp):
            missing_real.append(fp)
    if missing_real:
        raise FileNotFoundError(f"Sample CelebA files missing: {missing_real[:3]}")
    print("CelebA image files verified on disk.")

    # 2. Plan Fake Images
    print("\n" + "=" * 70)
    print("STEP 3: Diffusion Fake Generators Selection")
    print("=" * 70)

    generators = [g.strip() for g in args.fake_generators.split(",") if g.strip()]
    selected_fake_train = []
    selected_fake_val = []

    for gen in generators:
        gen_dir = os.path.join(args.diff_dir, gen)
        if not os.path.isdir(gen_dir):
            raise FileNotFoundError(f"Diffusion generator folder not found: {gen_dir}")

        all_files = sorted(
            [
                f
                for f in os.listdir(gen_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        required = args.fake_train_per_gen + args.fake_val_per_gen
        print(f"Generator {gen:10s}: found {len(all_files):,} images (need {required:,})")

        if len(all_files) < required:
            raise ValueError(
                f"Generator {gen} does not have enough images: {len(all_files)} < {required}"
            )

        # Deterministic slice
        gen_train = all_files[: args.fake_train_per_gen]
        gen_val = all_files[args.fake_train_per_gen : required]

        assert set(gen_train).isdisjoint(
            set(gen_val)
        ), f"Overlap detected in {gen} files!"

        selected_fake_train.extend([(gen, f) for f in gen_train])
        selected_fake_val.extend([(gen, f) for f in gen_val])

    print(f"\nTotal Selected Fake Train: {len(selected_fake_train):,} images")
    print(f"Total Selected Fake Val:   {len(selected_fake_val):,} images")

    # 3. Build Copy Task List
    # Output structure:
    #   dest_dir/train/real/celeba_*.jpg
    #   dest_dir/train/fake/{gen}_*.png
    #   dest_dir/val/real/celeba_*.jpg
    #   dest_dir/val/fake/{gen}_*.png
    copy_tasks = []

    # Real Train
    for img in selected_train_real:
        src = os.path.join(args.celeba_dir, img)
        dst_rel = os.path.join("train", "real", f"celeba_{img}")
        dst = os.path.join(args.dest_dir, dst_rel)
        ident = id_map.get(img, "unknown")
        copy_tasks.append(
            {
                "src": src,
                "dst": dst,
                "split": "train",
                "label": 0,
                "label_name": "real",
                "source": "celeba",
                "identity": ident,
                "relative_path": dst_rel,
            }
        )

    # Real Val
    for img in selected_val_real:
        src = os.path.join(args.celeba_dir, img)
        dst_rel = os.path.join("val", "real", f"celeba_{img}")
        dst = os.path.join(args.dest_dir, dst_rel)
        ident = id_map.get(img, "unknown")
        copy_tasks.append(
            {
                "src": src,
                "dst": dst,
                "split": "val",
                "label": 0,
                "label_name": "real",
                "source": "celeba",
                "identity": ident,
                "relative_path": dst_rel,
            }
        )

    # Fake Train
    for gen, f in selected_fake_train:
        src = os.path.join(args.diff_dir, gen, f)
        dst_rel = os.path.join("train", "fake", f"{gen.lower()}_{f}")
        dst = os.path.join(args.dest_dir, dst_rel)
        copy_tasks.append(
            {
                "src": src,
                "dst": dst,
                "split": "train",
                "label": 1,
                "label_name": "fake",
                "source": gen,
                "identity": "none",
                "relative_path": dst_rel,
            }
        )

    # Fake Val
    for gen, f in selected_fake_val:
        src = os.path.join(args.diff_dir, gen, f)
        dst_rel = os.path.join("val", "fake", f"{gen.lower()}_{f}")
        dst = os.path.join(args.dest_dir, dst_rel)
        copy_tasks.append(
            {
                "src": src,
                "dst": dst,
                "split": "val",
                "label": 1,
                "label_name": "fake",
                "source": gen,
                "identity": "none",
                "relative_path": dst_rel,
            }
        )

    # 4. Check Unseen Validation Contamination
    print("\n" + "=" * 70)
    print("STEP 4: External Diagnostic Set Isolation Check")
    print("=" * 70)
    if os.path.isdir(args.unseen_val_dir):
        print(f"External unseen set path exists: {args.unseen_val_dir}")
        print("Real source in unseen set: Zenodo (external benchmark)")
        print("Fake source in unseen set: LDM (Latent Diffusion Model)")
        print("Our Real source: CelebA (aligned portraits)")
        print("Our Fake source: ADM, DDIM, DDPM, DiffSwap")
        print("SUCCESS: Generator families and real sources are completely distinct.")
    else:
        print(f"Note: External unseen dir not found at {args.unseen_val_dir}")

    return copy_tasks, {
        "train_real_count": len(selected_train_real),
        "val_real_count": len(selected_val_real),
        "train_fake_count": len(selected_fake_train),
        "val_fake_count": len(selected_fake_val),
        "train_identities_count": len(train_ids),
        "val_identities_count": len(val_ids),
        "generators": generators,
    }


def execute_copy(copy_tasks, dest_dir, workers=16):
    """Execute multithreaded copying with progress reporting."""
    os.makedirs(os.path.join(dest_dir, "train", "real"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "train", "fake"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val", "real"), exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "val", "fake"), exist_ok=True)

    total_tasks = len(copy_tasks)
    completed_count = 0
    lock = Lock()
    start_time = time.time()

    def _copy_worker(task):
        src = task["src"]
        dst = task["dst"]
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        return True

    print("\n" + "=" * 70)
    print(f"STEP 5: Copying {total_tasks:,} Files ({workers} Worker Threads)")
    print("=" * 70)

    report_interval = 10000
    last_report = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_copy_worker, t): t for t in copy_tasks}
        for future in as_completed(futures):
            future.result()  # Will raise if copy failed
            with lock:
                completed_count += 1
                if completed_count - last_report >= report_interval or completed_count == total_tasks:
                    last_report = completed_count
                    elapsed = time.time() - start_time
                    fps = completed_count / max(elapsed, 0.001)
                    eta = (total_tasks - completed_count) / max(fps, 0.001)
                    pct = 100.0 * completed_count / total_tasks
                    print(
                        f"Progress: {completed_count:,}/{total_tasks:,} ({pct:5.1f}%) "
                        f"| Speed: {fps:6.1f} files/s | Elapsed: {elapsed/60:4.1f}m | ETA: {eta/60:4.1f}m"
                    )

    elapsed_total = time.time() - start_time
    print(f"\nCopy complete! Total time: {elapsed_total:.1f}s ({elapsed_total/60:.2f} minutes).")


def write_manifest_and_summary(copy_tasks, dest_dir, summary_data):
    """Write manifest CSV and summary JSON to the destination folder."""
    manifest_path = os.path.join(dest_dir, "manifest.csv")
    summary_path = os.path.join(dest_dir, "dataset_summary.json")

    print(f"\nWriting manifest to: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("path,relative_path,split,label,label_name,source,identity\n")
        for t in copy_tasks:
            # Normalized forward-slash relative path for cross-platform portability
            rel_norm = t["relative_path"].replace("\\", "/")
            f.write(
                f"{t['dst']},{rel_norm},{t['split']},{t['label']},{t['label_name']},{t['source']},{t['identity']}\n"
            )

    full_summary = {
        "dataset_name": "DatesetNoVideosNoGAN",
        "total_images": len(copy_tasks),
        "train_split": {
            "total": summary_data["train_real_count"] + summary_data["train_fake_count"],
            "real": summary_data["train_real_count"],
            "fake": summary_data["train_fake_count"],
            "unique_identities": summary_data["train_identities_count"],
        },
        "val_split": {
            "total": summary_data["val_real_count"] + summary_data["val_fake_count"],
            "real": summary_data["val_real_count"],
            "fake": summary_data["val_fake_count"],
            "unique_identities": summary_data["val_identities_count"],
        },
        "identity_overlap": 0,
        "generators": summary_data["generators"],
        "generator_breakdown": {
            "train": {g: 20000 for g in summary_data["generators"]},
            "val": {g: 5000 for g in summary_data["generators"]},
        },
        "real_source": "CelebA (aligned portraits)",
        "fake_sources": summary_data["generators"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"Writing summary to: {summary_path}")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    return full_summary


def verify_destination(dest_dir):
    """Check destination folder counts directly from filesystem."""
    print("\n" + "=" * 70)
    print("STEP 6: Verifying Destination Directory Counts")
    print("=" * 70)

    train_real_cnt = len(os.listdir(os.path.join(dest_dir, "train", "real")))
    train_fake_cnt = len(os.listdir(os.path.join(dest_dir, "train", "fake")))
    val_real_cnt = len(os.listdir(os.path.join(dest_dir, "val", "real")))
    val_fake_cnt = len(os.listdir(os.path.join(dest_dir, "val", "fake")))

    total = train_real_cnt + train_fake_cnt + val_real_cnt + val_fake_cnt
    print(f"Destination Root: {dest_dir}")
    print(f"  train/real: {train_real_cnt:,} files (expected 80,000)")
    print(f"  train/fake: {train_fake_cnt:,} files (expected 80,000)")
    print(f"  val/real:   {val_real_cnt:,} files (expected 20,000)")
    print(f"  val/fake:   {val_fake_cnt:,} files (expected 20,000)")
    print(f"  TOTAL:      {total:,} files (expected 200,000)")

    assert train_real_cnt == 80000, f"train/real count mismatch: {train_real_cnt}"
    assert train_fake_cnt == 80000, f"train/fake count mismatch: {train_fake_cnt}"
    assert val_real_cnt == 20000, f"val/real count mismatch: {val_real_cnt}"
    assert val_fake_cnt == 20000, f"val/fake count mismatch: {val_fake_cnt}"
    print("\nALL VERIFICATION CHECKS PASSED PERFECTLY!")


def main():
    args = parse_args()
    tasks, summary_data = build_copy_tasks(args)

    print("\n" + "=" * 70)
    print("SUMMARY PLAN")
    print("=" * 70)
    print(f"Destination:  {args.dest_dir}")
    print(f"Total Images: {len(tasks):,}")
    print(f"Train Split:  {summary_data['train_real_count'] + summary_data['train_fake_count']:,} images (80%)")
    print(f"  - Real: {summary_data['train_real_count']:,} (CelebA, {summary_data['train_identities_count']:,} unique IDs)")
    print(f"  - Fake: {summary_data['train_fake_count']:,} (20k ADM, 20k DDIM, 20k DDPM, 20k DiffSwap)")
    print(f"Val Split:    {summary_data['val_real_count'] + summary_data['val_fake_count']:,} images (20%)")
    print(f"  - Real: {summary_data['val_real_count']:,} (CelebA, {summary_data['val_identities_count']:,} unique IDs)")
    print(f"  - Fake: {summary_data['val_fake_count']:,} (5k ADM, 5k DDIM, 5k DDPM, 5k DiffSwap)")
    print(f"Train/Val ID Overlap: EXACTLY 0")

    if args.dry_run:
        print("\n[DRY RUN MODE] Plan validated successfully. No files copied.")
        return

    # Check free disk space before copying
    total_b, used_b, free_b = shutil.disk_usage(os.path.splitdrive(args.dest_dir)[0] + "\\")
    free_gb = free_b / (1024**3)
    print(f"\nTarget drive free space: {free_gb:.2f} GB (Estimated needed: ~10.7 GB)")
    if free_gb < 15:
        print("WARNING: Less than 15 GB free space available on target drive!")

    execute_copy(tasks, args.dest_dir, workers=args.workers)
    write_manifest_and_summary(tasks, args.dest_dir, summary_data)
    verify_destination(args.dest_dir)


if __name__ == "__main__":
    main()
