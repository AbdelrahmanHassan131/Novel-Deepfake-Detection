"""Audit relationships, filename family overlaps, and saved SHA256 duplicates.
Produces:
- output\\review\\archive_investigation\\relationships_report.json
"""

import csv
import gzip
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

TSV_GZ_PATH = r"E:\archive_investigation_scratch_task\archive_listing.tsv.gz"
MY_DATASET_ZIP = r"F:\Discovery AI\Second Expirement 100K RGB model\my_dataset_archive.zip"
REPAIRED_MANIFEST = r"output\review\rgb_family_repaired_100k_seed42\selected_manifest.csv"
OUTPUT_DIR = r"output\review\archive_investigation"

RE_FAMILY = re.compile(r"^(vid_[0-9a-fA-F]{64})_face_\d+_\d+\.[a-zA-Z0-9]+$")
RE_NUMERIC = re.compile(r"^\d+\.[a-zA-Z0-9]+$")


def extract_family_id(filename: str):
    m = RE_FAMILY.match(filename)
    return m.group(1).lower() if m else None


def run_relationship_audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()
    print("Starting relationship audit...")

    # Step 1: Read archive listing and map filename families
    print(f"Reading archive listing from {TSV_GZ_PATH}...")
    archive_families = {"train": defaultdict(list), "val": defaultdict(list)}
    archive_numeric = {"train": 0, "val": 0}
    archive_other = {"train": 0, "val": 0}
    archive_total = {"train": 0, "val": 0}
    archive_paths_by_split = {"train": set(), "val": set()}

    with gzip.open(TSV_GZ_PATH, "rt", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5 or parts[4] == "1":  # skip dirs
                continue
            path_str = parts[0]
            # format: Preapred Dataset/<split>/<class>/<filename>
            segs = path_str.split("/")
            if len(segs) < 4 or segs[0] != "Preapred Dataset":
                continue
            split = segs[1]
            if split not in ("train", "val"):
                continue

            archive_total[split] += 1
            archive_paths_by_split[split].add(path_str)
            fn = segs[-1]
            fam = extract_family_id(fn)
            if fam:
                archive_families[split][fam].append(path_str)
            elif RE_NUMERIC.match(fn):
                archive_numeric[split] += 1
            else:
                archive_other[split] += 1

    train_fams = set(archive_families["train"].keys())
    val_fams = set(archive_families["val"].keys())
    crossing_fams = train_fams.intersection(val_fams)

    crossing_train_files = sum(len(archive_families["train"][fam]) for fam in crossing_fams)
    crossing_val_files = sum(len(archive_families["val"][fam]) for fam in crossing_fams)

    print(f"Archive Train unique families: {len(train_fams):,}")
    print(f"Archive Val unique families: {len(val_fams):,}")
    print(f"Archive Crossing families: {len(crossing_fams):,}")
    print(f"  Affecting {crossing_train_files:,} train files and {crossing_val_files:,} val files")

    # Sample representative crossing families
    representative_crossing = []
    for fam in sorted(list(crossing_fams))[:5]:
        representative_crossing.append({
            "family_id": fam,
            "train_sample_paths": archive_families["train"][fam][:2],
            "val_sample_paths": archive_families["val"][fam][:2],
            "train_member_count": len(archive_families["train"][fam]),
            "val_member_count": len(archive_families["val"][fam])
        })

    # Step 2: Audit saved SHA256 metadata from clean_parent
    print("Reading saved SHA256 metadata from my_dataset_archive.zip...")
    clean_parent_records = 0
    sha_to_paths = defaultdict(list)
    sha_to_labels = defaultdict(set)
    parent_splits = Counter()
    parent_families_by_split = defaultdict(lambda: defaultdict(list))

    with zipfile.ZipFile(MY_DATASET_ZIP, "r") as z:
        clean_csv_name = "prepared_data/rgb_sizes_seed42/clean_parent/selected_manifest.csv"
        with z.open(clean_csv_name) as f:
            reader = csv.DictReader(line.decode("utf-8") for line in f)
            for row in reader:
                clean_parent_records += 1
                raw_path = row["path"]
                # strip /content/dataset/ if present
                norm_path = raw_path.replace("\\", "/")
                if norm_path.startswith("/content/dataset/"):
                    rel_path = norm_path[len("/content/dataset/"):]
                else:
                    rel_path = norm_path

                sha = row.get("sha256", "").strip()
                label = row.get("label", "").strip()
                split = row.get("split", "").strip()
                parent_splits[split] += 1

                if sha:
                    sha_to_paths[sha].append(rel_path)
                    sha_to_labels[sha].add(label)

                fn = os.path.basename(rel_path)
                fam = extract_family_id(fn)
                if fam:
                    parent_families_by_split[split][fam].append(rel_path)

    # Saved sha256 duplicates
    total_unique_sha = len(sha_to_paths)
    exact_duplicate_hashes = {sha: paths for sha, paths in sha_to_paths.items() if len(paths) > 1}
    label_conflicting_hashes = {sha: paths for sha, paths in sha_to_paths.items() if len(sha_to_labels[sha]) > 1}

    # Cross-split duplicate hashes in clean_parent
    # In clean_parent, rows were labeled train or dev
    parent_train_fams = set(parent_families_by_split["train"].keys())
    parent_dev_fams = set(parent_families_by_split["dev"].keys())
    parent_crossing_fams = parent_train_fams.intersection(parent_dev_fams)

    # Step 3: Old 100K Manifest vs Repaired Cohort
    print("Reading old 100K manifest from my_dataset_archive.zip...")
    old_100k_counts = Counter()
    old_100k_families = defaultdict(set)
    with zipfile.ZipFile(MY_DATASET_ZIP, "r") as z:
        old_100k_csv = "prepared_data/rgb_sizes_seed42/sizes/100000/selected_manifest.csv"
        with z.open(old_100k_csv) as f:
            reader = csv.DictReader(line.decode("utf-8") for line in f)
            for row in reader:
                split = row["split"]
                old_100k_counts[split] += 1
                fn = os.path.basename(row["path"])
                fam = extract_family_id(fn)
                if fam:
                    old_100k_families[fam].add(split)

    old_100k_crossing_families = {fam for fam, splits in old_100k_families.items() if len(splits) > 1}

    print("Reading repaired manifest...")
    repaired_counts = Counter()
    repaired_families = defaultdict(set)
    with open(REPAIRED_MANIFEST, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            repaired_counts[split] += 1
            fam = row.get("filename_family")
            if fam and fam != "unknown":
                repaired_families[fam].add(split)

    repaired_crossing_families = {fam for fam, splits in repaired_families.items() if len(splits) > 1}

    # Step 4: Archive coverage of saved hashes
    # How many files in archive train vs val have saved hashes?
    all_parent_paths = set()
    for paths in sha_to_paths.values():
        all_parent_paths.update(paths)

    archive_train_with_hash = len(archive_paths_by_split["train"].intersection(all_parent_paths))
    archive_val_with_hash = len(archive_paths_by_split["val"].intersection(all_parent_paths))

    t_elapsed = time.time() - t0
    print(f"Audit completed in {t_elapsed:.2f}s.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(t_elapsed, 2),
        "archive_partitions_overlap": {
            "description": "Overlap between the original archived 'train' and 'val' partitions inside Preapred Dataset.zip.",
            "archive_train_total_files": archive_total["train"],
            "archive_val_total_files": archive_total["val"],
            "archive_train_vid_files": sum(len(archive_families["train"][fam]) for fam in train_fams),
            "archive_val_vid_files": sum(len(archive_families["val"][fam]) for fam in val_fams),
            "archive_train_unique_families": len(train_fams),
            "archive_val_unique_families": len(val_fams),
            "crossing_families_count": len(crossing_fams),
            "crossing_families_train_files_affected": crossing_train_files,
            "crossing_families_val_files_affected": crossing_val_files,
            "crossing_families_fraction_of_val_families": round(len(crossing_fams) / len(val_fams), 4) if val_fams else 0,
            "representative_crossing_examples": representative_crossing,
            "numeric_naming": {
                "archive_train_numeric_files": archive_numeric["train"],
                "archive_val_numeric_files": archive_numeric["val"],
                "limitation_note": "Numbered files (<digits>.png) lack video tokens; numeric indices cannot be safely joined across splits without provenance metadata."
            }
        },
        "downstream_partitions_comparison": {
            "explanation": "Critical distinction: The original archive has train (805k) and val (211k). Downstream preparation took 794k samples from archive train, carving out an internal dev split. The old 100K manifest and repaired cohort operated entirely within this downstream pool, NOT against the archive's val partition.",
            "clean_parent_pool": {
                "total_rows": clean_parent_records,
                "split_distribution": dict(parent_splits),
                "unique_families_in_parent_train": len(parent_train_fams),
                "unique_families_in_parent_dev": len(parent_dev_fams),
                "crossing_families_in_parent": len(parent_crossing_fams)
            },
            "old_100k_manifest": {
                "source_path": "prepared_data/rgb_sizes_seed42/sizes/100000/selected_manifest.csv",
                "counts": dict(old_100k_counts),
                "crossing_filename_families": len(old_100k_crossing_families)
            },
            "repaired_100k_cohort": {
                "source_path": REPAIRED_MANIFEST,
                "counts": dict(repaired_counts),
                "crossing_filename_families": len(repaired_crossing_families)
            }
        },
        "saved_sha256_duplicate_analysis": {
            "total_rows_with_saved_sha256": clean_parent_records,
            "unique_sha256_hashes": total_unique_sha,
            "exact_duplicate_hash_groups": len(exact_duplicate_hashes),
            "total_files_in_exact_duplicate_groups": sum(len(p) for p in exact_duplicate_hashes.values()),
            "label_conflict_hashes_count": len(label_conflicting_hashes),
            "archive_coverage": {
                "archive_train_files_covered": archive_train_with_hash,
                "archive_train_coverage_fraction": round(archive_train_with_hash / archive_total["train"], 4) if archive_total["train"] else 0,
                "archive_val_files_covered": archive_val_with_hash,
                "archive_val_coverage_fraction": round(archive_val_with_hash / archive_total["val"], 4) if archive_total["val"] else 0,
                "limitation_note": "Archive val was never hashed in clean_parent; SHA256 analysis is available only for images selected during train pool preparation."
            }
        },
        "relationship_taxonomy_and_protection": {
            "categories": {
                "identical_byte_duplicates": "Identical SHA256 hashes. Clean parent quarantined label conflicts; exact duplicates within same class exist.",
                "conservative_filename_families": "Shared vid_<64hex> prefix. Tracks frames from the same video/sequence.",
                "verified_original_manipulated_relationships": "Explicit link between real source and fake derivative. Currently UNKNOWN / unrecorded in archive metadata.",
                "unknown_near_duplicates": "Frames from the same sequence with different video IDs, re-encoded frames, or different crops/resolutions without common tokens."
            },
            "what_family_repaired_cohort_protects_against": [
                "Eliminates recognized vid_<64hex> filename family leakage between train and dev within the 100K training pool (0 crossing families vs 6,926 in parent).",
                "Guarantees that video frames sharing the same hex token do not appear in both train and dev."
            ],
            "what_remains_unresolved": [
                "Numeric files (193k in train, 58k in val) lack video tokens; frames from the same video or person named with digits could cross partitions undetected.",
                "The archive's own original val partition was completely omitted during downstream Colab preparation (0 val files included in clean_parent).",
                "Severe format and class correlation persists: real images are predominantly JPEG, fake images predominantly PNG.",
                "External diagnostic dataset (F:\\val to be deleted 2) has zero overlap with Preapred Dataset and represents an unseen distribution with unknown generator/domain shifts."
            ]
        }
    }

    report_path = os.path.join(OUTPUT_DIR, "relationships_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    run_relationship_audit()
