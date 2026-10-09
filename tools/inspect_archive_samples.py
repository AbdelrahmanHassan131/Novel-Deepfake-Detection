"""Deterministic sample selection, bounded extraction, profiling, and contact sheets.
Produces:
- output\\review\\archive_investigation\\sample_members.csv
- output\\review\\archive_investigation\\sample_profile.json
- output\\review\\archive_investigation\\contact_sheet_train_real.png
- output\\review\\archive_investigation\\contact_sheet_train_fake.png
- output\\review\\archive_investigation\\contact_sheet_external_real.png
- output\\review\\archive_investigation\\contact_sheet_external_fake.png
"""

import csv
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SEED = 42
READER_EXE = r"C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe"
ARCHIVE_ZIP = r"E:\PreparedDataset\Preapred Dataset.zip"
SCRATCH_DIR = r"E:\archive_investigation_scratch_task"
TSV_GZ_PATH = os.path.join(SCRATCH_DIR, "archive_listing.tsv.gz")
OUTPUT_DIR = r"output\review\archive_investigation"
EXTERNAL_DIR = r"F:\val to be deleted 2"
MY_DATASET_ZIP = r"F:\Discovery AI\Second Expirement 100K RGB model\my_dataset_archive.zip"

RE_FAMILY = re.compile(r"^(vid_[0-9a-fA-F]{64})_face_\d+_\d+\.[a-zA-Z0-9]+$")
RE_NUMERIC = re.compile(r"^\d+\.[a-zA-Z0-9]+$")


def extract_family_id(filename: str):
    m = RE_FAMILY.match(filename)
    return m.group(1).lower() if m else None


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def select_archived_samples():
    random.seed(SEED)
    # Stratified pools reflecting actual archive composition:
    # fake has only PNG: vid (563k) and numeric (31k)
    # real has vid PNG (49k) and numeric JPG (162k)
    pools = {
        ("fake", "png", "vid"): defaultdict(list),
        ("fake", "png", "numeric"): [],
        ("real", "png", "vid"): defaultdict(list),
        ("real", "jpg", "numeric"): []
    }

    print("Loading archive listing for sampling...")
    with gzip.open(TSV_GZ_PATH, "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5 or parts[4] == "1":
                continue
            path_str = parts[0]
            segs = path_str.split("/")
            if len(segs) < 4 or segs[0] != "Preapred Dataset" or segs[1] != "train":
                continue
            cls = segs[2]
            if cls not in ("fake", "real"):
                continue
            fn = segs[-1]
            ext = os.path.splitext(fn)[1].lower().lstrip(".")
            fam = extract_family_id(fn)

            if cls == "fake" and ext == "png":
                if fam:
                    pools[("fake", "png", "vid")][fam].append(path_str)
                elif RE_NUMERIC.match(fn):
                    pools[("fake", "png", "numeric")].append(path_str)
            elif cls == "real":
                if ext == "png" and fam:
                    pools[("real", "png", "vid")][fam].append(path_str)
                elif ext == "jpg" and RE_NUMERIC.match(fn):
                    pools[("real", "jpg", "numeric")].append(path_str)

    selected = []
    quotas = {
        "shortages_and_absences": {
            "fake_jpg": "0 available in entire archive (fake is 100% PNG)",
            "real_png_numeric": "0 available in entire archive (all real numeric are JPG)",
            "real_jpg_vid": "0 available in entire archive (all real vid are PNG)"
        },
        "selected_quotas": {}
    }

    # Fake: 64 vid PNG (distinct families) + 64 numeric PNG = 128
    fake_vid_fams = list(pools[("fake", "png", "vid")].keys())
    random.shuffle(fake_vid_fams)
    chosen_fake_vid = [random.choice(pools[("fake", "png", "vid")][fam]) for fam in fake_vid_fams[:64]]

    fake_num = pools[("fake", "png", "numeric")][:]
    random.shuffle(fake_num)
    chosen_fake_num = fake_num[:64]

    # Real: 64 vid PNG (distinct families) + 64 numeric JPG = 128
    real_vid_fams = list(pools[("real", "png", "vid")].keys())
    random.shuffle(real_vid_fams)
    chosen_real_vid = [random.choice(pools[("real", "png", "vid")][fam]) for fam in real_vid_fams[:64]]

    real_num = pools[("real", "jpg", "numeric")][:]
    random.shuffle(real_num)
    chosen_real_num = real_num[:64]

    quotas["selected_quotas"] = {
        "fake_png_vid_distinct_families": len(chosen_fake_vid),
        "fake_png_numeric": len(chosen_fake_num),
        "real_png_vid_distinct_families": len(chosen_real_vid),
        "real_jpg_numeric": len(chosen_real_num),
        "total_archived_samples": len(chosen_fake_vid) + len(chosen_fake_num) + len(chosen_real_vid) + len(chosen_real_num)
    }

    selected.extend(chosen_fake_vid)
    selected.extend(chosen_fake_num)
    selected.extend(chosen_real_vid)
    selected.extend(chosen_real_num)

    print(f"Selected {len(selected)} archived training samples. Quotas: {quotas['selected_quotas']}")
    return selected, quotas


def select_external_samples():
    random.seed(SEED)
    selected = []
    quotas = {}
    for cls in ("real", "fake"):
        cls_dir = os.path.join(EXTERNAL_DIR, cls)
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith((".png", ".jpg"))])
        random.shuffle(files)
        chosen = files[:64]
        quotas[cls] = len(chosen)
        for f in chosen:
            selected.append({
                "source": "external_dev",
                "class": cls,
                "rel_path": f"{cls}/{f}",
                "abs_path": os.path.join(cls_dir, f),
                "filename": f
            })
    print(f"Selected {len(selected)} external-development samples. Quotas: {quotas}")
    return selected, quotas


def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples_dir = os.path.join(SCRATCH_DIR, "extracted_samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Step 1: Select samples
    archived_paths, train_quotas = select_archived_samples()
    ext_samples, ext_quotas = select_external_samples()

    # Step 2: Extract selected archived images
    list_file_path = os.path.join(SCRATCH_DIR, "selected_train_members.txt")
    with open(list_file_path, "w", encoding="utf-8") as f:
        for p in archived_paths:
            f.write(p.replace("/", "\\") + "\n")

    print(f"Extracting {len(archived_paths)} members using 7z list extraction...")
    extract_cmd = [READER_EXE, "x", ARCHIVE_ZIP, f"-o{samples_dir}", f"@{list_file_path}", "-y"]
    t0 = time.time()
    res = subprocess.run(extract_cmd, capture_output=True, text=True, errors="replace")
    t_extract = time.time() - t0
    print(f"Extraction finished in {t_extract:.2f}s. Returncode: {res.returncode}")

    # Step 3: Load saved SHA256 hashes from clean_parent
    print("Loading saved SHA256 hashes from clean_parent...")
    saved_hashes = {}
    with zipfile.ZipFile(MY_DATASET_ZIP, "r") as z:
        clean_csv_name = "prepared_data/rgb_sizes_seed42/clean_parent/selected_manifest.csv"
        with z.open(clean_csv_name) as f:
            reader = csv.DictReader(line.decode("utf-8") for line in f)
            for row in reader:
                norm_p = row["path"].replace("\\", "/")
                if norm_p.startswith("/content/dataset/"):
                    norm_p = norm_p[len("/content/dataset/"):]
                saved_hashes[norm_p] = row["sha256"].strip()

    # Step 4: Profile archived samples
    all_sample_records = []
    total_extracted_bytes = 0
    sha_matches = 0
    sha_missing = 0
    sha_mismatches = 0

    for member_p in archived_paths:
        local_p = os.path.join(samples_dir, member_p.replace("/", os.sep))
        segs = member_p.split("/")
        cls = segs[2]
        fn = segs[-1]
        file_ext = os.path.splitext(fn)[1].lower().lstrip(".")
        fam = extract_family_id(fn)

        if not os.path.exists(local_p):
            all_sample_records.append({
                "source": "archived_train",
                "class": cls,
                "archive_path": member_p,
                "local_path": local_p,
                "filename": fn,
                "extension": file_ext,
                "family_id": fam or "none",
                "file_size": 0,
                "sha256": "decode_failure",
                "saved_sha256": saved_hashes.get(member_p, "none"),
                "sha_status": "file_missing",
                "decoded_format": "NONE",
                "width": 0, "height": 0, "mode": "NONE", "aspect_ratio": 0.0,
                "has_jpeg_markers": False
            })
            continue

        st = os.stat(local_p)
        fsize = st.st_size
        total_extracted_bytes += fsize
        actual_sha = compute_sha256(local_p)
        saved_sha = saved_hashes.get(member_p)

        if saved_sha:
            if saved_sha.lower() == actual_sha.lower():
                sha_status = "matched"
                sha_matches += 1
            else:
                sha_status = "mismatch"
                sha_mismatches += 1
        else:
            sha_status = "missing_in_clean_parent"
            sha_missing += 1

        # PIL decode
        try:
            with Image.open(local_p) as img:
                fmt = img.format or "UNKNOWN"
                w, h = img.size
                mode = img.mode
                ar = round(w / h, 4) if h else 0.0
                info = img.info or {}
                has_jpeg = ("jfif" in info or "exif" in info or "photoshop" in info or fmt == "JPEG")
        except Exception as e:
            fmt = "DECODE_ERROR"
            w, h = 0, 0
            mode = "ERROR"
            ar = 0.0
            has_jpeg = False

        all_sample_records.append({
            "source": "archived_train",
            "class": cls,
            "archive_path": member_p,
            "local_path": local_p,
            "filename": fn,
            "extension": file_ext,
            "family_id": fam or "none",
            "file_size": fsize,
            "sha256": actual_sha,
            "saved_sha256": saved_sha or "none",
            "sha_status": sha_status,
            "decoded_format": fmt,
            "width": w, "height": h, "mode": mode, "aspect_ratio": ar,
            "has_jpeg_markers": has_jpeg
        })

    # Step 5: Profile external samples
    for item in ext_samples:
        p = item["abs_path"]
        cls = item["class"]
        fn = item["filename"]
        file_ext = os.path.splitext(fn)[1].lower().lstrip(".")
        st = os.stat(p)
        fsize = st.st_size
        actual_sha = compute_sha256(p)

        try:
            with Image.open(p) as img:
                fmt = img.format or "UNKNOWN"
                w, h = img.size
                mode = img.mode
                ar = round(w / h, 4) if h else 0.0
                info = img.info or {}
                has_jpeg = ("jfif" in info or "exif" in info or "photoshop" in info or fmt == "JPEG")
        except Exception as e:
            fmt = "DECODE_ERROR"
            w, h = 0, 0
            mode = "ERROR"
            ar = 0.0
            has_jpeg = False

        all_sample_records.append({
            "source": "external_dev",
            "class": cls,
            "archive_path": item["rel_path"],
            "local_path": p,
            "filename": fn,
            "extension": file_ext,
            "family_id": "none",
            "file_size": fsize,
            "sha256": actual_sha,
            "saved_sha256": "none",
            "sha_status": "external_not_in_parent",
            "decoded_format": fmt,
            "width": w, "height": h, "mode": mode, "aspect_ratio": ar,
            "has_jpeg_markers": has_jpeg
        })

    print(f"Total extracted bytes: {total_extracted_bytes / (1024**2):.2f} MiB (limit 512 MiB)")
    print(f"SHA check: {sha_matches} matched, {sha_missing} missing, {sha_mismatches} mismatches.")

    # Write sample_members.csv
    csv_path = os.path.join(OUTPUT_DIR, "sample_members.csv")
    fieldnames = [
        "source", "class", "archive_path", "filename", "extension", "family_id",
        "file_size", "sha256", "saved_sha256", "sha_status", "decoded_format",
        "width", "height", "mode", "aspect_ratio", "has_jpeg_markers"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_sample_records)
    print(f"Wrote {csv_path}")

    # Generate aggregated sample_profile.json
    def stats_for_group(records):
        sizes = [r["file_size"] for r in records if r["file_size"] > 0]
        widths = [r["width"] for r in records if r["width"] > 0]
        heights = [r["height"] for r in records if r["height"] > 0]
        ars = [r["aspect_ratio"] for r in records if r["aspect_ratio"] > 0]
        formats = Counter(r["decoded_format"] for r in records)
        extensions = Counter(r["extension"] for r in records)
        modes = Counter(r["mode"] for r in records)
        jpeg_markers = sum(1 for r in records if r["has_jpeg_markers"])

        return {
            "count": len(records),
            "file_size_bytes": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "mean": round(sum(sizes)/len(sizes), 1) if sizes else 0
            },
            "dimensions": {
                "min_width": min(widths) if widths else 0,
                "max_width": max(widths) if widths else 0,
                "min_height": min(heights) if heights else 0,
                "max_height": max(heights) if heights else 0,
                "most_common_resolutions": Counter((r["width"], r["height"]) for r in records).most_common(5)
            },
            "aspect_ratios": {
                "min": min(ars) if ars else 0,
                "max": max(ars) if ars else 0,
                "mean": round(sum(ars)/len(ars), 4) if ars else 0
            },
            "decoded_formats": dict(formats),
            "extensions": dict(extensions),
            "modes": dict(modes),
            "has_jpeg_markers_count": jpeg_markers
        }

    arch_records = [r for r in all_sample_records if r["source"] == "archived_train"]
    ext_records = [r for r in all_sample_records if r["source"] == "external_dev"]

    profile_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_archived_samples": len(arch_records),
        "total_external_samples": len(ext_records),
        "total_extracted_scratch_bytes": total_extracted_bytes,
        "scratch_bytes_mib": round(total_extracted_bytes / (1024**2), 2),
        "sampling_quotas": {
            "archived_train": train_quotas,
            "external_dev": ext_quotas
        },
        "sha256_hash_verification": {
            "tested_samples": len(arch_records),
            "matched_clean_parent_hash": sha_matches,
            "missing_in_clean_parent": sha_missing,
            "hash_mismatch": sha_mismatches,
            "scope_caveat": "Sample-level validation confirms exact byte fidelity with saved clean_parent metadata for matching paths. Not a full-dataset CRC."
        },
        "profiles": {
            "archived_train_overall": stats_for_group(arch_records),
            "archived_train_real": stats_for_group([r for r in arch_records if r["class"] == "real"]),
            "archived_train_fake": stats_for_group([r for r in arch_records if r["class"] == "fake"]),
            "external_dev_overall": stats_for_group(ext_records),
            "external_dev_real": stats_for_group([r for r in ext_records if r["class"] == "real"]),
            "external_dev_fake": stats_for_group([r for r in ext_records if r["class"] == "fake"])
        },
        "rgb_preprocessing_transforms_documented": {
            "pipeline": "RGBDataset (data/datasets/rgb_dataset.py) -> custom_resize -> data_augment -> crop_func -> flip_func -> ToTensor -> Normalize",
            "resize_logic": "custom_resize scales the smaller dimension to opt.loadSize (default 256) maintaining aspect ratio using bilinear interpolation by default.",
            "crop_logic": "CenterCrop(crop_size=224 or 128) during eval; RandomCrop during train.",
            "data_augment": "JPEG compression artifacts and blur applied dynamically according to augmentation recipe.",
            "resolution_normalization": "All models evaluate on square crops (e.g., 224x224 or 128x128), but input aspect ratios and initial scales vary significantly."
        },
        "sample_versus_population_warning": "Reported sample distributions reflect deterministic stratified sample cohorts, not unbiased population estimates. PNG encoding does not imply absence of prior lossy compression; JPEG augmentation means format correlations alone cannot prove the detector learned a pure compression shortcut."
    }

    json_profile_path = os.path.join(OUTPUT_DIR, "sample_profile.json")
    with open(json_profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)
    print(f"Wrote {json_profile_path}")

    # Step 6: Create contact sheets
    print("Generating contact sheets...")
    def make_contact_sheet(records, title, out_path, grid_cols=6, grid_rows=4, thumb_size=(120, 120)):
        # up to 24 images per sheet
        subset = records[: grid_cols * grid_rows]
        pad = 8
        cell_w = thumb_size[0] + pad * 2
        cell_h = thumb_size[1] + 36 + pad * 2  # room for text
        img_w = cell_w * grid_cols
        img_h = cell_h * grid_rows + 40  # room for title

        sheet = Image.new("RGB", (img_w, img_h), color=(25, 27, 33))
        draw = ImageDraw.Draw(sheet)

        draw.text((15, 12), title, fill=(240, 240, 240))

        for idx, r in enumerate(subset):
            row_idx = idx // grid_cols
            col_idx = idx % grid_cols
            x0 = col_idx * cell_w + pad
            y0 = row_idx * cell_h + pad + 40

            local_p = r["local_path"]
            try:
                with Image.open(local_p) as orig:
                    thumb = orig.convert("RGB")
                    thumb.thumbnail(thumb_size)
                    # paste centered
                    tx = x0 + (thumb_size[0] - thumb.width) // 2
                    ty = y0 + (thumb_size[1] - thumb.height) // 2
                    sheet.paste(thumb, (tx, ty))
            except Exception:
                draw.rectangle([x0, y0, x0 + thumb_size[0], y0 + thumb_size[1]], fill=(80, 20, 20))

            label_text = f"{r['extension'].upper()} {r['width']}x{r['height']}\n{r['filename'][:16]}"
            draw.text((x0, y0 + thumb_size[1] + 4), label_text, fill=(200, 200, 200))

        sheet.save(out_path)
        print(f"Saved contact sheet: {out_path}")

    make_contact_sheet([r for r in arch_records if r["class"] == "real"], "Archived Training - Real Sample (24 images)", os.path.join(OUTPUT_DIR, "contact_sheet_train_real.png"))
    make_contact_sheet([r for r in arch_records if r["class"] == "fake"], "Archived Training - Fake Sample (24 images)", os.path.join(OUTPUT_DIR, "contact_sheet_train_fake.png"))
    make_contact_sheet([r for r in ext_records if r["class"] == "real"], "External Dev - Real Sample (24 images)", os.path.join(OUTPUT_DIR, "contact_sheet_external_real.png"))
    make_contact_sheet([r for r in ext_records if r["class"] == "fake"], "External Dev - Fake Sample (24 images)", os.path.join(OUTPUT_DIR, "contact_sheet_external_fake.png"))

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
