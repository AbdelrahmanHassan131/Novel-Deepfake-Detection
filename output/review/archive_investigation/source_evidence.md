# Archive Provenance and Source Evidence

**Generated:** 2026-10-09T18:37:40.592966+00:00  
**Archive:** `E:\PreparedDataset\Preapred Dataset.zip`  
**Reader:** 7-Zip 22.01 x64  
**Cached listing:** `E:\archive_investigation_scratch_task\archive_listing.tsv.gz`  

## 1. Inventory and Partition Summary

- Total archive entries: 1,016,862 (1,016,855 files, 7 directories)
- Total uncompressed size: 33.84 GiB (36,335,917,409 bytes)
- Total compressed size: 32.23 GiB (34,610,413,812 bytes)
- Duplicate exact member paths: 0

### Partitions Breakdown

| Partition | Files | Uncompressed GiB | Real Files | Fake Files | Other Files | .png | .jpg / .jpeg |
|---|---|---|---|---|---|---|---|
| `train` | 805,421 | 26.28 | 211,188 | 594,233 | 0 | 643,343 | 162,078 |
| `val` | 211,434 | 7.56 | 52,765 | 158,669 | 0 | 170,914 | 40,520 |

### Filename Patterns

| Partition | vid_face_hex pattern | Numeric pattern (`<int>.<ext>`) | Other pattern |
|---|---|---|---|
| `train` | 612,232 | 193,189 | 0 |
| `val` | 152,932 | 58,502 | 0 |

## 2. Distinction between Archived Partitions and CPU Development Cohort

- The archive contains its own original split structure (e.g. `Preapred Dataset/train` and `Preapred Dataset/val`).
- The later engineering `dev` partition (15,000 samples) was carved out from `train/` during local Colab/CPU preparation, NOT from the archive's `val` partition.
- This audit explicitly distinguishes the archived `val/` from the downstream derived `dev` partition.

## 3. Provenance Documents Inside the Archive

**No explicit metadata documents, README files, licenses, source lists, dataset cards, or extraction logs were found inside the archive.**

All non-directory members in the archive are image files.

### Provenance Confidence Levels:
- **Verified from explicit metadata:** None. The archive contains no provenance manifest, camera/generator metadata tables, or dataset cards.
- **Suggested by naming:**
  - Files matching `vid_<64hex>_face_<frame>_<idx>` strongly suggest extraction from face-tracked video datasets (e.g., FaceForensics++, DFDC, Celeb-DF, etc.), but the exact original dataset name is NOT recorded in path tokens.
  - Files named `<integer>.png` (numeric) have completely flattened provenance. Numbers do not identify source or generator.
- **Unknown:**
  - True generator algorithms (e.g. DeepFakes, Face2Face, FaceSwap, NeuralTextures, StyleGAN, Midjourney, etc.) cannot be determined from paths.
  - Real source provenance (YouTube, FFHQ, CelebA, VoxCeleb) is flattened and unrecorded.

### Remaining Questions for User:
1. What raw datasets were originally downloaded and packed into `Preapred Dataset` (e.g., FaceForensics++ c23/c40, Celeb-DF v2, DFDC, WildDeepfake)?
2. Which specific generators correspond to `train/fake` and `val/fake`?
3. What was the origin of the purely numeric files (`<digits>.png`) versus the `vid_<hex>` files?