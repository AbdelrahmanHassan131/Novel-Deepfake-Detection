"""Build controlled RGB data-mixture manifests (Arm A and Arm B) with a shared dev cohort.

Implements Prompt 2 requirements from RGB_NEXT_CONTROLLED_PILOT_HANDOFF.md:
- Connects groups once via existing parent relationships, content hashes, full paths,
  and conservative filename families (vid_<64hex>).
- Partitions connected groups once into train and dev before quota selection.
- Drops duplicate content hashes within each group and enforces cap=8 images per group per class.
- Defines observable strata using canonical class + exact numeric/vid patterns.
- Reports unknown patterns if encountered (never silently assigns).
- Preserves explicit 'image_level_unverified' provenance; never writes strata into dataset_source.
- Stable deterministic common ranking within each stratum for shared selections.
- Targets:
    Arm A: 20K train (7,714 num real, 2,286 vid real, 551 num fake, 9,449 vid fake) + 2K shared dev
    Arm B: 20K train (5,000 num real, 5,000 vid real, 5,000 num fake, 5,000 vid fake) + 2K shared dev
    Shared Dev: 2K (500 num real, 500 vid real, 500 num fake, 500 vid fake)
- Strict validation: 0 cross-partition overlap, identical shared dev between arms, verified parent hash inheritance.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

FAMILY = re.compile(r"^(vid_[0-9a-f]{64})_face_\d+_\d+\.[a-z0-9]+$", re.I)
NUMERIC = re.compile(r"^\d+\.[a-z0-9]+$", re.I)
UNKNOWN_VALS = {"", "unknown", "none", "nan", "n/a", "null"}

QUOTAS = {
    "arm_a_train": {
        "numeric_real": 7714,
        "video_real": 2286,
        "numeric_fake": 551,
        "video_fake": 9449,
    },
    "arm_b_train": {
        "numeric_real": 5000,
        "video_real": 5000,
        "numeric_fake": 5000,
        "video_fake": 5000,
    },
    "shared_dev": {
        "numeric_real": 500,
        "video_real": 500,
        "numeric_fake": 500,
        "video_fake": 500,
    }
}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def family_id(path: str) -> str:
    m = FAMILY.fullmatch(str(path).replace("\\", "/").rsplit("/", 1)[-1])
    return m[1].lower() if m else None


def is_known(val: Any) -> bool:
    s = str(val or "").strip().lower()
    return s not in UNKNOWN_VALS and not s.startswith(("n/a_", "na_"))


def tokens(row: Dict[str, Any]):
    for field in ("group_id", "source_video_id", "original_id", "identity_id", "sha256"):
        for val in re.split(r"[,;|]", str(row.get(field, ""))):
            if is_known(val):
                yield ("video" if field in {"source_video_id", "original_id"} else field, val.strip())
    yield "path", row["path"].replace("\\", "/").casefold()
    fam = family_id(row["path"])
    if fam:
        yield "filename_family", fam


def classify_stratum(path: str, label: int) -> str:
    fn = path.replace("\\", "/").rsplit("/", 1)[-1]
    if NUMERIC.match(fn):
        return "numeric_real" if label == 0 else "numeric_fake"
    elif FAMILY.match(fn):
        return "video_real" if label == 0 else "video_fake"
    return "unknown"


def partition_and_build_manifests(
    rows: List[Dict[str, Any]],
    seed: int = 42,
    cap: int = 8,
    dev_fraction: float = 0.15,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Build Arm A, Arm B, and Shared Dev manifests with strict group isolation."""
    # 1. Validation of parent rows
    ids = set()
    content_labels = {}
    old_families = defaultdict(set)
    unknown_patterns = []

    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        if row.get("split") not in {"train", "dev"}:
            raise ValueError("Only an existing engineering train/dev pool can be repartitioned; never test data.")
        if row.get("grouping_basis") != "image_level_unverified":
            raise ValueError(f"Expected grouping_basis 'image_level_unverified', got '{row.get('grouping_basis')}'.")
        
        lbl_str = str(row.get("label", "")).strip()
        if lbl_str not in ("0", "1"):
            raise ValueError(f"Expected canonical label '0' or '1', got '{lbl_str}' at row {i}.")
        lbl = int(lbl_str)

        sid = row.get("sample_id", "")
        sha = row.get("sha256", "")
        if not sid or sid in ids or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError(f"Duplicate/missing sample_id or invalid sha256 at row {i}.")
        ids.add(sid)

        if content_labels.setdefault(sha, lbl) != lbl:
            raise ValueError(f"Contradictory labels for content hash {sha}: existing {content_labels[sha]} vs new {lbl}.")

        # Check stratum pattern
        stratum = classify_stratum(row["path"], lbl)
        if stratum == "unknown":
            unknown_patterns.append((i, row["path"], lbl))

        fam = family_id(row["path"])
        if fam:
            old_families[fam].add(row["split"])

        for token in tokens(row):
            if token in seen:
                parents[find(i)] = find(seen[token])
            else:
                seen[token] = i

    if unknown_patterns:
        raise ValueError(
            f"Encountered {len(unknown_patterns)} rows with unknown filename pattern. "
            f"Sample: {unknown_patterns[:5]}. Unknown patterns must be reported, not silently assigned."
        )

    # 2. Group into connected components
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)

    print(f"Connected {len(rows):,} rows into {len(components):,} disjoint components.")

    # 3. Partition components into dev vs train and deduplicate/cap within components
    # Pools: pools['dev'][stratum] = list of candidate rows
    #        pools['train'][stratum] = list of candidate rows
    pools = {
        "dev": defaultdict(list),
        "train": defaultdict(list)
    }
    component_groups = {}

    for comp_id, indices in components.items():
        anchor = min(rows[i]["sample_id"] for i in indices)
        group = "family_connected:" + hashlib.sha256(anchor.encode()).hexdigest()
        component_groups[comp_id] = group

        fraction = int(hashlib.sha256(f"partition:{seed}:{group}".encode()).hexdigest()[:12], 16) / (16**12)
        split = "dev" if fraction < dev_fraction else "train"

        # Deterministic sample ordering within component
        ranked = sorted(indices, key=lambda idx: hashlib.sha256(f"sample:{seed}:{rows[idx]['sample_id']}".encode()).digest())
        hashes = set()
        class_counts = Counter()

        for idx in ranked:
            r = rows[idx]
            lbl = int(r["label"])
            sha = r["sha256"]
            if sha in hashes or class_counts[lbl] >= cap:
                continue
            hashes.add(sha)
            class_counts[lbl] += 1

            stratum = classify_stratum(r["path"], lbl)
            pools[split][stratum].append((group, idx))

    # 4. Report post-grouping capacity
    capacities = {
        "dev": {s: len(pools["dev"][s]) for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")},
        "train": {s: len(pools["train"][s]) for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")}
    }
    print("Post-grouping capacities:")
    print("  Dev pool:", capacities["dev"])
    print("  Train pool:", capacities["train"])

    # Check for shortfalls against quotas
    shortfalls = {}
    for stratum, q in QUOTAS["shared_dev"].items():
        avail = capacities["dev"][stratum]
        if avail < q:
            shortfalls[f"dev_{stratum}"] = q - avail

    for stratum, q in QUOTAS["arm_a_train"].items():
        avail = capacities["train"][stratum]
        if avail < q:
            shortfalls[f"arm_a_{stratum}"] = q - avail

    for stratum, q in QUOTAS["arm_b_train"].items():
        avail = capacities["train"][stratum]
        if avail < q:
            shortfalls[f"arm_b_{stratum}"] = q - avail

    if shortfalls:
        raise ValueError(f"Quotas cannot be met post-grouping! Shortfalls: {shortfalls}")

    # 5. Deterministic common ranking within each stratum
    # Sort candidate entries in each stratum using stable deterministic ranking
    # Ranking is by (deterministic_group_order, deterministic_sample_order)
    def rank_candidates(candidate_list, split_name, stratum_name):
        return sorted(
            candidate_list,
            key=lambda item: hashlib.sha256(
                f"common_rank:{seed}:{split_name}:{stratum_name}:{item[0]}:{rows[item[1]]['sample_id']}".encode()
            ).digest()
        )

    ranked_dev = {}
    ranked_train = {}
    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake"):
        ranked_dev[s] = rank_candidates(pools["dev"][s], "dev", s)
        ranked_train[s] = rank_candidates(pools["train"][s], "train", s)

    # 6. Select shared dev rows (500 from each stratum)
    shared_dev_indices = []
    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake"):
        q = QUOTAS["shared_dev"][s]
        selected_items = ranked_dev[s][:q]
        shared_dev_indices.extend(item[1] for item in selected_items)

    shared_dev_rows = []
    for idx in sorted(shared_dev_indices):
        r = dict(rows[idx])
        r["parent_split"] = r["split"]
        r["split"] = "dev"
        r["group_id"] = component_groups[find(idx)]
        r["grouping_method"] = "content_parent_links_and_filename_family"
        r["filename_family"] = family_id(r["path"]) or "unknown"
        # Preserve image_level_unverified; do not overwrite dataset_source
        shared_dev_rows.append(r)

    # 7. Select Arm A train rows (7,714 num_real, 2,286 vid_real, 551 num_fake, 9,449 vid_fake)
    arm_a_train_indices = []
    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake"):
        q = QUOTAS["arm_a_train"][s]
        selected_items = ranked_train[s][:q]
        arm_a_train_indices.extend(item[1] for item in selected_items)

    arm_a_rows = []
    for idx in sorted(arm_a_train_indices):
        r = dict(rows[idx])
        r["parent_split"] = r["split"]
        r["split"] = "train"
        r["group_id"] = component_groups[find(idx)]
        r["grouping_method"] = "content_parent_links_and_filename_family"
        r["filename_family"] = family_id(r["path"]) or "unknown"
        arm_a_rows.append(r)
    # Append shared dev
    arm_a_rows.extend(copy.deepcopy(shared_dev_rows))
    arm_a_rows.sort(key=lambda x: x["sample_id"])

    # 8. Select Arm B train rows (5,000 each)
    arm_b_train_indices = []
    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake"):
        q = QUOTAS["arm_b_train"][s]
        selected_items = ranked_train[s][:q]
        arm_b_train_indices.extend(item[1] for item in selected_items)

    arm_b_rows = []
    for idx in sorted(arm_b_train_indices):
        r = dict(rows[idx])
        r["parent_split"] = r["split"]
        r["split"] = "train"
        r["group_id"] = component_groups[find(idx)]
        r["grouping_method"] = "content_parent_links_and_filename_family"
        r["filename_family"] = family_id(r["path"]) or "unknown"
        arm_b_rows.append(r)
    # Append shared dev
    arm_b_rows.extend(copy.deepcopy(shared_dev_rows))
    arm_b_rows.sort(key=lambda x: x["sample_id"])

    # 9. Strict post-selection validation checks
    # A. Validate Arm A: no token crosses train and dev
    def validate_partition_tokens(manifest_rows, arm_name):
        assigned_tokens = {}
        for r in manifest_rows:
            split = r["split"]
            for token in tokens(r):
                if assigned_tokens.setdefault(token, split) != split:
                    raise ValueError(
                        f"Cross-partition leak in {arm_name}! Token '{token}' assigned to both "
                        f"'{assigned_tokens[token]}' and '{split}'."
                    )

    validate_partition_tokens(arm_a_rows, "Arm A")
    validate_partition_tokens(arm_b_rows, "Arm B")

    # B. Validate shared dev identity between Arm A and Arm B
    dev_a = [r for r in arm_a_rows if r["split"] == "dev"]
    dev_b = [r for r in arm_b_rows if r["split"] == "dev"]
    expected_dev_len = sum(QUOTAS["shared_dev"].values())
    assert len(dev_a) == len(dev_b) == expected_dev_len, f"Shared dev size mismatch: {len(dev_a)} vs {len(dev_b)}"
    for ra, rb in zip(dev_a, dev_b):
        assert ra["sample_id"] == rb["sample_id"], f"Shared dev sample_id mismatch: {ra['sample_id']} vs {rb['sample_id']}"
        assert ra["path"] == rb["path"], f"Shared dev path mismatch"
        assert ra["label"] == rb["label"], f"Shared dev label mismatch"
        assert ra["sha256"] == rb["sha256"], f"Shared dev sha256 mismatch"

    # C. Validate no train/dev overlap across either arm's train and shared dev
    dev_sids = set(r["sample_id"] for r in shared_dev_rows)
    arm_a_train_sids = set(r["sample_id"] for r in arm_a_rows if r["split"] == "train")
    arm_b_train_sids = set(r["sample_id"] for r in arm_b_rows if r["split"] == "train")
    assert len(dev_sids & arm_a_train_sids) == 0, "Arm A train shares sample IDs with shared dev!"
    assert len(dev_sids & arm_b_train_sids) == 0, "Arm B train shares sample IDs with shared dev!"

    # D. Build summary report
    report = {
        "capacities_post_grouping": capacities,
        "quotas_requested": QUOTAS,
        "quotas_achieved": {
            "arm_a": {
                "train": {
                    s: sum(1 for r in arm_a_rows if r["split"] == "train" and classify_stratum(r["path"], int(r["label"])) == s)
                    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")
                },
                "dev": {
                    s: sum(1 for r in arm_a_rows if r["split"] == "dev" and classify_stratum(r["path"], int(r["label"])) == s)
                    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")
                },
                "total_samples": len(arm_a_rows),
            },
            "arm_b": {
                "train": {
                    s: sum(1 for r in arm_b_rows if r["split"] == "train" and classify_stratum(r["path"], int(r["label"])) == s)
                    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")
                },
                "dev": {
                    s: sum(1 for r in arm_b_rows if r["split"] == "dev" and classify_stratum(r["path"], int(r["label"])) == s)
                    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")
                },
                "total_samples": len(arm_b_rows),
            },
            "shared_dev": {
                "samples": len(shared_dev_rows),
                "strata_counts": {
                    s: sum(1 for r in shared_dev_rows if classify_stratum(r["path"], int(r["label"])) == s)
                    for s in ("numeric_real", "video_real", "numeric_fake", "video_fake")
                }
            }
        },
        "group_counts": {
            "total_connected_components": len(components),
            "maximum_rows_per_component_per_class": cap,
            "dev_fraction": dev_fraction,
            "seed": seed,
        },
        "cross_partition_checks": {
            "arm_a_train_dev_overlap_tokens": 0,
            "arm_b_train_dev_overlap_tokens": 0,
            "shared_dev_identity_verified": True,
        }
    }

    return {
        "arm_a": arm_a_rows,
        "arm_b": arm_b_rows,
        "shared_dev": shared_dev_rows
    }, report


def build_and_save_manifests(
    parent_source: str,
    output_dir: str,
    seed: int = 42,
    cap: int = 8,
    dev_fraction: float = 0.15,
):
    out_base = Path(output_dir).resolve()
    if out_base.exists() and any(out_base.iterdir()):
        raise ValueError(
            f"Use an empty NEW output directory; old evidence is never overwritten. "
            f"Destination '{out_base}' already exists and contains files."
        )

    # 1. Load parent rows and strictly validate inherited hash evidence
    parent_p = Path(parent_source)
    if parent_p.suffix.lower() == ".zip":
        print(f"Reading clean_parent from zip archive: {parent_p}")
        if not parent_p.is_file():
            raise FileNotFoundError(f"Parent archive not found: {parent_p}")
        with zipfile.ZipFile(parent_p, "r") as z:
            # Locate clean_parent member files
            prefix_candidates = [
                "prepared_data/rgb_sizes_seed42/clean_parent/",
                "rgb_session_lt7k3ki6/prepared_data/clean_parent/",
            ]
            prefix = None
            for cand in prefix_candidates:
                if (cand + "selected_manifest.csv") in z.namelist():
                    prefix = cand
                    break
            if not prefix:
                raise FileNotFoundError("Could not find clean_parent directory inside zip archive.")

            csv_name = prefix + "selected_manifest.csv"
            gate_name = prefix + "selected_manifest.verified.json"
            rec_name = prefix + "recovery_report.json"

            if gate_name not in z.namelist():
                raise FileNotFoundError(f"Missing parent gate file inside zip: {gate_name}")
            if rec_name not in z.namelist():
                raise FileNotFoundError(f"Missing parent recovery report inside zip: {rec_name}")

            parent_csv_bytes = z.read(csv_name)
            parent_gate_bytes = z.read(gate_name)
            parent_rec_bytes = z.read(rec_name)
    else:
        print(f"Reading clean_parent from CSV file: {parent_p}")
        if not parent_p.is_file():
            raise FileNotFoundError(f"Parent CSV file not found: {parent_p}")
        parent_gate_p = parent_p.with_suffix(".verified.json")
        parent_rec_p = parent_p.parent / "recovery_report.json"

        if not parent_gate_p.is_file():
            raise FileNotFoundError(f"Missing parent gate file: {parent_gate_p}")
        if not parent_rec_p.is_file():
            raise FileNotFoundError(f"Missing parent recovery report: {parent_rec_p}")

        parent_csv_bytes = parent_p.read_bytes()
        parent_gate_bytes = parent_gate_p.read_bytes()
        parent_rec_bytes = parent_rec_p.read_bytes()

    # Cryptographic digests
    parent_csv_sha = digest_bytes(parent_csv_bytes)
    parent_gate_sha = digest_bytes(parent_gate_bytes)
    parent_rec_sha = digest_bytes(parent_rec_bytes)

    # Validate parent gate evidence (fail closed)
    try:
        parent_gate = json.loads(parent_gate_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Corrupted parent verification gate: {exc}") from exc

    try:
        parent_rec = json.loads(parent_rec_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError(f"Corrupted parent recovery report: {exc}") from exc

    if parent_gate.get("verified") is not True:
        raise ValueError(f"Parent gate indicates verification failed or unverified (verified={parent_gate.get('verified')}).")
    if parent_gate.get("hashes_verified") is not True:
        raise ValueError(f"Parent gate indicates hashes unverified (hashes_verified={parent_gate.get('hashes_verified')}).")
    if parent_gate.get("manifest_sha256") != parent_csv_sha:
        raise ValueError(
            f"Parent gate manifest_sha256 mismatch: recorded '{parent_gate.get('manifest_sha256')}' "
            f"vs actual '{parent_csv_sha}'."
        )
    if parent_gate.get("recovery_report_sha256") != parent_rec_sha:
        raise ValueError(
            f"Parent gate recovery_report_sha256 mismatch: recorded '{parent_gate.get('recovery_report_sha256')}' "
            f"vs actual '{parent_rec_sha}'."
        )

    reader = csv.DictReader(line for line in parent_csv_bytes.decode("utf-8-sig").splitlines())
    rows = list(reader)
    print(f"Loaded {len(rows):,} parent rows (SHA256: {parent_csv_sha}). Verified parent audit chain.")

    # 2. Partition and build manifests
    manifest_dict, report = partition_and_build_manifests(
        rows, seed=seed, cap=cap, dev_fraction=dev_fraction
    )

    report["parent_evidence"] = {
        "parent_manifest_sha256": parent_csv_sha,
        "parent_gate_sha256": parent_gate_sha,
        "recovery_report_sha256": parent_rec_sha,
        "hash_verification_mode": "inherited_completed_audit_on_explicitly_unchanged_dataset",
        "unchanged_image_assumption": True,
        "limitations": [
            "Hash inheritance conditioned strictly on unchanged image bytes and paths.",
            "Numeric filename patterns do not verify independent sources or identities (image_level_unverified).",
            "Unknown near-duplicates and semantic confounders may remain.",
        ]
    }
    report["parent_manifest_sha256"] = parent_csv_sha

    # 3. Write output manifests atomically
    out_base.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "path", "label", "split", "dataset_source", "group_id",
        "source_video_id", "identity_id", "original_id", "generator", "grouping_basis",
        "sha256", "parent_split", "grouping_method", "filename_family"
    ]

    def write_csv_and_gate_atomic(sub_dir_name, manifest_rows):
        dest_dir = out_base / sub_dir_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        csv_file = dest_dir / "selected_manifest.csv"
        csv_tmp = dest_dir / "selected_manifest.csv.tmp"
        gate_file = dest_dir / "selected_manifest.verified.json"
        gate_tmp = dest_dir / "selected_manifest.verified.json.tmp"

        with open(csv_tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest_rows)

        csv_sha = digest_file(csv_tmp)
        gate_data = {
            "verified": True,
            "hashes_verified": True,
            "manifest_sha256": csv_sha,
            "manifest_path": str(csv_file),
            "samples": len(manifest_rows),
            "classes": [0, 1],
            "splits": sorted(list(set(r["split"] for r in manifest_rows))),
            "hash_verification_mode": "inherited_completed_audit_on_explicitly_unchanged_dataset",
            "inherited_parent_evidence": {
                "parent_manifest_sha256": parent_csv_sha,
                "parent_gate_sha256": parent_gate_sha,
                "recovery_report_sha256": parent_rec_sha,
                "unchanged_image_assumption": True,
                "provenance_limitation": (
                    "Hash inheritance strictly conditioned on unchanged image bytes and paths. "
                    "Original identities and sources remain image_level_unverified; unknown near-duplicates may remain."
                )
            },
            "preparation_identity": {
                "experiment": "controlled_rgb_data_mixture_pilot",
                "sub_manifest": sub_dir_name,
                "seed": seed,
                "cap": cap,
                "dev_fraction": dev_fraction,
            }
        }
        with open(gate_tmp, "w", encoding="utf-8") as f:
            json.dump(gate_data, f, indent=2)

        # Atomic replacement
        if csv_file.exists():
            csv_file.unlink()
        csv_tmp.rename(csv_file)

        if gate_file.exists():
            gate_file.unlink()
        gate_tmp.rename(gate_file)

        print(f"Saved {sub_dir_name}: {csv_file} ({len(manifest_rows):,} rows, SHA256: {csv_sha})")
        return csv_file, csv_sha

    csv_a, sha_a = write_csv_and_gate_atomic("arm_a", manifest_dict["arm_a"])
    csv_b, sha_b = write_csv_and_gate_atomic("arm_b", manifest_dict["arm_b"])
    csv_dev, sha_dev = write_csv_and_gate_atomic("shared_dev", manifest_dict["shared_dev"])

    report["manifest_digests"] = {
        "arm_a": sha_a,
        "arm_b": sha_b,
        "shared_dev": sha_dev,
    }

    # 4. Generate union members lists for Colab selective extraction
    union_paths = sorted(list(set(
        r["path"] for r in (manifest_dict["arm_a"] + manifest_dict["arm_b"])
    )))
    union_txt = out_base / "union_members_list.txt"
    union_archive_txt = out_base / "union_archive_members.txt"

    with open(union_txt, "w", encoding="utf-8") as f:
        for p in union_paths:
            f.write(p.replace("\\", "/") + "\n")

    with open(union_archive_txt, "w", encoding="utf-8") as f:
        for p in union_paths:
            rel = p.replace("\\", "/").replace("/content/dataset/", "")
            f.write(rel + "\n")

    print(f"Saved union members list ({len(union_paths):,} unique paths) at {union_txt} and {union_archive_txt}")

    # 5. Save report atomically
    rep_file = out_base / "mixture_manifest_report.json"
    rep_tmp = out_base / "mixture_manifest_report.json.tmp"
    with open(rep_tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if rep_file.exists():
        rep_file.unlink()
    rep_tmp.rename(rep_file)

    print(f"Saved mixture manifest report at {rep_file}")
    return report


def main():
    p = argparse.ArgumentParser(description="Build controlled RGB mixture manifests")
    p.add_argument("--parent", default=r"F:\Discovery AI\Second Expirement 100K RGB model\my_dataset_archive.zip",
                   help="Path to clean_parent CSV or my_dataset_archive.zip")
    p.add_argument("--output_dir", default="output/controlled_mixture/manifests_v2",
                   help="Destination directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap", type=int, default=8)
    p.add_argument("--dev_fraction", type=float, default=0.15)
    args = p.parse_args()

    build_and_save_manifests(
        parent_source=args.parent,
        output_dir=args.output_dir,
        seed=args.seed,
        cap=args.cap,
        dev_fraction=args.dev_fraction,
    )


if __name__ == "__main__":
    main()
