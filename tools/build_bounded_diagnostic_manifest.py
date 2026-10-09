"""Build a strictly bounded 128-image manifest for format sensitivity diagnostic.

Composition:
- 64 archived training samples:
  * 16 numeric-real
  * 16 numeric-fake
  * 16 video-pattern-real
  * 16 video-pattern-fake
- 64 external-development samples:
  * 32 external-real
  * 32 external-fake

Deterministic sampling: Random seed 42.
Output: output/review/archive_investigation/bounded_format_diagnostic_manifest.csv
"""

import csv
import hashlib
import os
import random
from pathlib import Path

SEED = 42
BASE_EXTRACTED = Path(r"E:\archive_investigation_scratch_task\extracted_samples")
BASE_EXTERNAL = Path(r"F:\val to be deleted 2")
OUT_MANIFEST = Path(r"output\review\archive_investigation\bounded_format_diagnostic_manifest.csv")
SAMPLE_MEMBERS_CSV = Path(r"output\review\archive_investigation\sample_members.csv")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Loading candidate sample members from {SAMPLE_MEMBERS_CSV}...")
    with open(SAMPLE_MEMBERS_CSV, "r", encoding="utf-8") as f:
        members = list(csv.DictReader(f))

    # Group into the 6 targeted strata
    arch_train_num_real = []
    arch_train_num_fake = []
    arch_train_vid_real = []
    arch_train_vid_fake = []
    ext_dev_real = []
    ext_dev_fake = []

    for m in members:
        src = m["source"]
        cls = m["class"]
        fn = m["filename"]
        if src == "archived_train":
            p = BASE_EXTRACTED / m["archive_path"]
            if not p.is_file():
                raise FileNotFoundError(f"Missing extracted sample: {p}")
            if fn.lower().endswith(".jpg"):
                arch_train_num_real.append((p, m))
            elif fn.lower().startswith("vid_"):
                if cls == "fake":
                    arch_train_vid_fake.append((p, m))
                else:
                    arch_train_vid_real.append((p, m))
            else:
                arch_train_num_fake.append((p, m))
        elif src == "external_dev":
            p = BASE_EXTERNAL / m["archive_path"]
            if not p.is_file():
                raise FileNotFoundError(f"Missing external dev sample: {p}")
            if cls == "real":
                ext_dev_real.append((p, m))
            else:
                ext_dev_fake.append((p, m))

    # Deterministic sampling with seed 42
    rng = random.Random(SEED)
    # Sort first for cross-platform deterministic ordering before sampling
    arch_train_num_real.sort(key=lambda x: x[0].name)
    arch_train_num_fake.sort(key=lambda x: x[0].name)
    arch_train_vid_real.sort(key=lambda x: x[0].name)
    arch_train_vid_fake.sort(key=lambda x: x[0].name)
    ext_dev_real.sort(key=lambda x: x[0].name)
    ext_dev_fake.sort(key=lambda x: x[0].name)

    sampled_num_real = rng.sample(arch_train_num_real, 16)
    sampled_num_fake = rng.sample(arch_train_num_fake, 16)
    sampled_vid_real = rng.sample(arch_train_vid_real, 16)
    sampled_vid_fake = rng.sample(arch_train_vid_fake, 16)
    sampled_ext_real = rng.sample(ext_dev_real, 32)
    sampled_ext_fake = rng.sample(ext_dev_fake, 32)

    all_sampled = []
    
    # 1. Archived train numeric real
    for p, m in sampled_num_real:
        all_sampled.append({
            "sample_id": f"train_num_real_{p.stem}",
            "path": str(p),
            "label": 0,
            "split": "train",
            "cohort": "training_diagnostic",
            "subcollection": "numeric_real",
            "dataset_source": "archived_training",
            "group_id": m.get("family_id", "unverified"),
            "sha256": compute_sha256(p),
        })

    # 2. Archived train numeric fake
    for p, m in sampled_num_fake:
        all_sampled.append({
            "sample_id": f"train_num_fake_{p.stem}",
            "path": str(p),
            "label": 1,
            "split": "train",
            "cohort": "training_diagnostic",
            "subcollection": "numeric_fake",
            "dataset_source": "archived_training",
            "group_id": m.get("family_id", "unverified"),
            "sha256": compute_sha256(p),
        })

    # 3. Archived train video real
    for p, m in sampled_vid_real:
        all_sampled.append({
            "sample_id": f"train_vid_real_{p.stem}",
            "path": str(p),
            "label": 0,
            "split": "train",
            "cohort": "training_diagnostic",
            "subcollection": "video_real",
            "dataset_source": "archived_training",
            "group_id": m.get("family_id", "unverified"),
            "sha256": compute_sha256(p),
        })

    # 4. Archived train video fake
    for p, m in sampled_vid_fake:
        all_sampled.append({
            "sample_id": f"train_vid_fake_{p.stem}",
            "path": str(p),
            "label": 1,
            "split": "train",
            "cohort": "training_diagnostic",
            "subcollection": "video_fake",
            "dataset_source": "archived_training",
            "group_id": m.get("family_id", "unverified"),
            "sha256": compute_sha256(p),
        })

    # 5. External dev real
    for p, m in sampled_ext_real:
        all_sampled.append({
            "sample_id": f"ext_dev_real_{p.stem}",
            "path": str(p),
            "label": 0,
            "split": "dev",
            "cohort": "external_development",
            "subcollection": "external_real",
            "dataset_source": "external_development",
            "group_id": m.get("family_id", "unverified"),
            "sha256": compute_sha256(p),
        })

    # 6. External dev fake
    for p, m in sampled_ext_fake:
        all_sampled.append({
            "sample_id": f"ext_dev_fake_{p.stem}",
            "path": str(p),
            "label": 1,
            "split": "dev",
            "cohort": "external_development",
            "subcollection": "external_fake",
            "dataset_source": "external_development",
            "group_id": m.get("family_id", "unverified"),
            "sha256": compute_sha256(p),
        })

    assert len(all_sampled) == 128, f"Expected 128 samples, got {len(all_sampled)}"

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "path", "label", "split", "cohort", "subcollection", "dataset_source", "group_id", "sha256"]
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_sampled)

    print(f"Successfully generated bounded manifest with {len(all_sampled)} rows at {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
