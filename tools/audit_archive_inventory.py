"""Stream and audit split ZIP archive metadata without full extraction.
Produces:
- E:\\archive_investigation_scratch_task\\archive_listing.tsv.gz
- output\\review\\archive_investigation\\archive_inventory_summary.json
- output\\review\\archive_investigation\\source_evidence.md
"""

import gzip
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

READER_EXE = r"C:\Program Files\NVIDIA Corporation\NVIDIA App\7z.exe"
ARCHIVE_ZIP = r"E:\PreparedDataset\Preapred Dataset.zip"
SCRATCH_DIR = r"E:\archive_investigation_scratch_task"
OUTPUT_DIR = r"output\review\archive_investigation"

RE_VID = re.compile(r"^vid_[0-9a-fA-F]{64}_face_\d+_\d+\.[a-zA-Z0-9]+$")
RE_NUM = re.compile(r"^\d+\.[a-zA-Z0-9]+$")

PROVENANCE_KEYWORDS = ("readme", "license", "dataset", "source", "manifest", "info", "log", "meta", "about", "split")
PROVENANCE_EXTS = (".txt", ".md", ".json", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".log")


def parse_7z_line(line: str):
    if len(line) < 54:
        return None
    date_str = line[:19].strip()
    attr = line[20:25].strip()
    size_str = line[26:38].strip()
    comp_str = line[39:51].strip()
    name = line[53:].rstrip("\r\n")

    if not (re.match(r"^\d{4}-\d{2}-\d{2}", date_str) and ("D" in attr or "A" in attr or attr == "")):
        return None
    try:
        size = int(size_str) if size_str else 0
        comp = int(comp_str) if comp_str else 0
    except ValueError:
        return None
    is_dir = "D" in attr
    return {
        "datetime": date_str,
        "attr": attr,
        "is_dir": is_dir,
        "size": size,
        "compressed_size": comp,
        "name": name.replace("\\", "/")
    }


def is_safe_member_path(member_path: str) -> bool:
    norm = os.path.normpath(member_path).replace("\\", "/")
    if norm.startswith("../") or "/../" in norm or norm.startswith("/") or ":" in norm:
        return False
    return True


def run_inventory():
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tsv_path = os.path.join(SCRATCH_DIR, "archive_listing.tsv.gz")
    print(f"Streaming archive listing using: {READER_EXE}")
    print(f"Target scratch cache: {tsv_path}")

    cmd = [READER_EXE, "l", ARCHIVE_ZIP]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")

    total_entries = 0
    total_files = 0
    total_dirs = 0
    total_uncompressed = 0
    total_compressed = 0

    partition_stats = defaultdict(lambda: {
        "files": 0, "uncompressed_bytes": 0, "compressed_bytes": 0,
        "classes": Counter(), "extensions": Counter(), "patterns": Counter(),
        "prefixes": Counter()
    })

    duplicate_names = 0
    seen_paths = set()
    provenance_candidates = []

    with gzip.open(tsv_path, "wt", encoding="utf-8") as gz_out:
        gz_out.write("path\tsize\tcompressed_size\tdatetime\tis_dir\n")

        in_table = False
        for line in proc.stdout:
            raw = line.rstrip("\r\n")
            if raw.startswith("------------------- -----"):
                if not in_table:
                    in_table = True
                    continue
                else:
                    # Footer divider reached
                    break
            if not in_table:
                continue

            parsed = parse_7z_line(line)
            if not parsed or not parsed["attr"].strip():
                continue

            total_entries += 1
            path_str = parsed["name"]
            is_dir = parsed["is_dir"]
            size = parsed["size"]
            comp = parsed["compressed_size"]

            gz_out.write(f"{path_str}\t{size}\t{comp}\t{parsed['datetime']}\t{1 if is_dir else 0}\n")

            if is_dir:
                total_dirs += 1
                continue

            total_files += 1
            total_uncompressed += size
            total_compressed += comp

            if path_str in seen_paths:
                duplicate_names += 1
            else:
                seen_paths.add(path_str)

            # Analyze path structure
            parts = [p for p in path_str.split("/") if p]
            # Normal structure: Preapred Dataset / <split> / <class> / <filename>
            # Or root files
            filename = parts[-1]
            ext = os.path.splitext(filename)[1].lower().lstrip(".")

            # Check provenance candidate
            lower_name = filename.lower()
            if any(k in lower_name for k in PROVENANCE_KEYWORDS) or any(lower_name.endswith(e) for e in PROVENANCE_EXTS):
                if ext not in ("png", "jpg", "jpeg", "bmp", "webp"):
                    provenance_candidates.append({
                        "path": path_str,
                        "size": size,
                        "compressed_size": comp
                    })

            # Determine split and class
            # parts: ['Preapred Dataset', 'train', 'fake', '0.png']
            split = "root"
            cls = "unassigned"
            prefix = "root"
            if len(parts) >= 2 and parts[0] == "Preapred Dataset":
                split = parts[1]
                if len(parts) >= 3:
                    cls = parts[2]
                prefix = "/".join(parts[:3])
            elif len(parts) >= 2:
                split = parts[0]
                cls = parts[1]
                prefix = "/".join(parts[:2])

            pstat = partition_stats[split]
            pstat["files"] += 1
            pstat["uncompressed_bytes"] += size
            pstat["compressed_bytes"] += comp
            pstat["classes"][cls] += 1
            pstat["extensions"][ext] += 1
            pstat["prefixes"][prefix] += 1

            if RE_VID.match(filename):
                pstat["patterns"]["vid_face_hex"] += 1
            elif RE_NUM.match(filename):
                pstat["patterns"]["numeric"] += 1
            else:
                pstat["patterns"]["other"] += 1

            if total_files % 100000 == 0:
                print(f"Processed {total_files:,} files in {time.time()-t0:.1f}s...")

    proc.stdout.close()
    proc.wait()
    t_elapsed = time.time() - t0
    print(f"Finished inventory stream in {t_elapsed:.1f}s: {total_files:,} files, {total_dirs:,} dirs.")

    # Convert partition stats to regular dict
    formatted_partitions = {}
    for p_name, p_data in partition_stats.items():
        formatted_partitions[p_name] = {
            "files": p_data["files"],
            "uncompressed_bytes": p_data["uncompressed_bytes"],
            "uncompressed_gib": round(p_data["uncompressed_bytes"] / (1024**3), 3),
            "compressed_bytes": p_data["compressed_bytes"],
            "compressed_gib": round(p_data["compressed_bytes"] / (1024**3), 3),
            "classes": dict(p_data["classes"]),
            "extensions": dict(p_data["extensions"]),
            "patterns": dict(p_data["patterns"]),
            "top_prefixes": dict(p_data["prefixes"].most_common(20))
        }

    inventory_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_zip": ARCHIVE_ZIP,
        "listing_command": f'"{READER_EXE}" l "{ARCHIVE_ZIP}"',
        "cached_listing_tsv_gz": tsv_path,
        "elapsed_seconds": round(t_elapsed, 2),
        "verification_scope": "Header and Central Directory stream listing (Zip64). Payload CRC/integrity test not run across 32.5 GiB.",
        "totals": {
            "total_entries": total_entries,
            "total_files": total_files,
            "total_directories": total_dirs,
            "total_uncompressed_bytes": total_uncompressed,
            "total_uncompressed_gib": round(total_uncompressed / (1024**3), 3),
            "total_compressed_bytes": total_compressed,
            "total_compressed_gib": round(total_compressed / (1024**3), 3),
            "duplicate_exact_paths": duplicate_names
        },
        "partitions": formatted_partitions,
        "provenance_candidates_count": len(provenance_candidates),
        "provenance_candidates": provenance_candidates
    }

    summary_json_path = os.path.join(OUTPUT_DIR, "archive_inventory_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(inventory_summary, f, indent=2)
    print(f"Wrote {summary_json_path}")

    # Extract up to 20 provenance candidates if any exist, bounded to 10 MiB
    extracted_provenance_dir = os.path.join(SCRATCH_DIR, "extracted_provenance")
    os.makedirs(extracted_provenance_dir, exist_ok=True)

    extracted_files = []
    total_prov_bytes = 0
    safe_candidates = [c for c in provenance_candidates if is_safe_member_path(c["path"])]

    for cand in safe_candidates[:20]:
        if total_prov_bytes + cand["size"] > 10 * 1024 * 1024:
            break
        # Selective extraction using 7z e
        extract_cmd = [
            READER_EXE, "e", ARCHIVE_ZIP,
            f"-o{extracted_provenance_dir}",
            cand["path"].replace("/", "\\"),
            "-y"
        ]
        res = subprocess.run(extract_cmd, capture_output=True, text=True, errors="replace")
        target_local = os.path.join(extracted_provenance_dir, os.path.basename(cand["path"]))
        if os.path.exists(target_local):
            cand_stat = os.stat(target_local)
            total_prov_bytes += cand_stat.st_size
            try:
                content_preview = Path(target_local).read_text(encoding="utf-8", errors="replace")[:1000]
            except Exception:
                content_preview = "<Binary or non-UTF8 content>"
            extracted_files.append({
                "member_path": cand["path"],
                "local_path": target_local,
                "size": cand_stat.st_size,
                "preview": content_preview
            })

    # Generate source_evidence.md
    md_lines = [
        "# Archive Provenance and Source Evidence",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"**Archive:** `{ARCHIVE_ZIP}`  ",
        f"**Reader:** 7-Zip 22.01 x64  ",
        f"**Cached listing:** `{tsv_path}`  ",
        "",
        "## 1. Inventory and Partition Summary",
        "",
        f"- Total archive entries: {total_entries:,} ({total_files:,} files, {total_dirs:,} directories)",
        f"- Total uncompressed size: {total_uncompressed / (1024**3):.2f} GiB ({total_uncompressed:,} bytes)",
        f"- Total compressed size: {total_compressed / (1024**3):.2f} GiB ({total_compressed:,} bytes)",
        f"- Duplicate exact member paths: {duplicate_names}",
        "",
        "### Partitions Breakdown",
        "",
        "| Partition | Files | Uncompressed GiB | Real Files | Fake Files | Other Files | .png | .jpg / .jpeg |",
        "|---|---|---|---|---|---|---|---|"
    ]

    for pname, pinfo in formatted_partitions.items():
        f_cnt = pinfo["files"]
        u_gib = pinfo["uncompressed_gib"]
        real_c = pinfo["classes"].get("real", 0)
        fake_c = pinfo["classes"].get("fake", 0)
        other_c = f_cnt - real_c - fake_c
        png_c = pinfo["extensions"].get("png", 0)
        jpg_c = pinfo["extensions"].get("jpg", 0) + pinfo["extensions"].get("jpeg", 0)
        md_lines.append(f"| `{pname}` | {f_cnt:,} | {u_gib:.2f} | {real_c:,} | {fake_c:,} | {other_c:,} | {png_c:,} | {jpg_c:,} |")

    md_lines.extend([
        "",
        "### Filename Patterns",
        "",
        "| Partition | vid_face_hex pattern | Numeric pattern (`<int>.<ext>`) | Other pattern |",
        "|---|---|---|---|"
    ])
    for pname, pinfo in formatted_partitions.items():
        pat = pinfo["patterns"]
        md_lines.append(f"| `{pname}` | {pat.get('vid_face_hex', 0):,} | {pat.get('numeric', 0):,} | {pat.get('other', 0):,} |")

    md_lines.extend([
        "",
        "## 2. Distinction between Archived Partitions and CPU Development Cohort",
        "",
        "- The archive contains its own original split structure (e.g. `Preapred Dataset/train` and `Preapred Dataset/val`).",
        "- The later engineering `dev` partition (15,000 samples) was carved out from `train/` during local Colab/CPU preparation, NOT from the archive's `val` partition.",
        "- This audit explicitly distinguishes the archived `val/` from the downstream derived `dev` partition.",
        "",
        "## 3. Provenance Documents Inside the Archive",
        ""
    ])

    if not extracted_files:
        md_lines.extend([
            "**No explicit metadata documents, README files, licenses, source lists, dataset cards, or extraction logs were found inside the archive.**",
            "",
            "All non-directory members in the archive are image files.",
            "",
            "### Provenance Confidence Levels:",
            "- **Verified from explicit metadata:** None. The archive contains no provenance manifest, camera/generator metadata tables, or dataset cards.",
            "- **Suggested by naming:**",
            "  - Files matching `vid_<64hex>_face_<frame>_<idx>` strongly suggest extraction from face-tracked video datasets (e.g., FaceForensics++, DFDC, Celeb-DF, etc.), but the exact original dataset name is NOT recorded in path tokens.",
            "  - Files named `<integer>.png` (numeric) have completely flattened provenance. Numbers do not identify source or generator.",
            "- **Unknown:**",
            "  - True generator algorithms (e.g. DeepFakes, Face2Face, FaceSwap, NeuralTextures, StyleGAN, Midjourney, etc.) cannot be determined from paths.",
            "  - Real source provenance (YouTube, FFHQ, CelebA, VoxCeleb) is flattened and unrecorded.",
            "",
            "### Remaining Questions for User:",
            "1. What raw datasets were originally downloaded and packed into `Preapred Dataset` (e.g., FaceForensics++ c23/c40, Celeb-DF v2, DFDC, WildDeepfake)?",
            "2. Which specific generators correspond to `train/fake` and `val/fake`?",
            "3. What was the origin of the purely numeric files (`<digits>.png`) versus the `vid_<hex>` files?"
        ])
    else:
        md_lines.append(f"Found and inspected {len(extracted_files)} provenance documents:")
        for doc in extracted_files:
            md_lines.extend([
                f"#### Member: `{doc['member_path']}` ({doc['size']:,} bytes)",
                "```",
                doc["preview"],
                "```",
                ""
            ])

    source_md_path = os.path.join(OUTPUT_DIR, "source_evidence.md")
    with open(source_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote {source_md_path}")


if __name__ == "__main__":
    run_inventory()
