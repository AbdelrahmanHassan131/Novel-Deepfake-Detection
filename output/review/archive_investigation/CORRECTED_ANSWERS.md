# Corrected Evidence Record and Provenance Assessment

**Document:** `CORRECTED_ANSWERS.md`  
**Supercedes:** Scientific conclusions and experiment proposals in `ANSWERS_FOR_REVIEW.md` and `NEXT_IMPLEMENTATION_PROMPTS.md` (original reports preserved as the investigation record).  
**Investigation Scope:** Split ZIP archive `E:\PreparedDataset\Preapred Dataset.zip`, R2 100K training pool, and external diagnostic set `F:\val to be deleted 2`.

---

## 1. Corrections to Prior Investigation Findings

This document corrects all nine points identified in the archive review:

1. **Repaired Dev Cohort Counts Corrected:**
   - The repaired dev cohort (`output/review/rgb_family_repaired_100k_seed42/selected_manifest.csv`) contains:
     - Real (Label 0): **6,950 JPEG + 550 PNG = 7,500 total**.
     - Fake (Label 1): **7,500 PNG and ZERO JPEG**.
   - The prior text erroneously referenced 550 fake JPG files; fake is 100% PNG with zero JPEG members.

2. **Source Identities are Unverified Leads, Not Established Findings:**
   - Explicit archive metadata contains **zero provenance documents** (no dataset cards, manifests, or camera/generator tags).
   - Names such as FaceForensics++, CelebA, FFHQ, DiffFace, StyleGAN, and ProGAN are **possible origins/hypotheses** suggested by repository adapter files and visual similarities, **NOT verified facts**.
   - Decoded facial appearance cannot scientifically confirm whether a fake image was synthesized from noise or edited from an existing photograph.

3. **Training Data is Not Exclusively Video Manipulation:**
   - The training archive contains **31,111 numeric fake files** alongside 563,122 video-pattern fake files.
   - In the actual R2 100K training manifest (`prepared_data/rgb_sizes_seed42/sizes/100000/selected_manifest.csv`), the training split contains:
     - **Fake:** 47,247 video-pattern (`vid_`) PNGs + **2,753 numeric PNGs**.
     - **Real:** 11,430 video-pattern (`vid_`) PNGs + **38,570 numeric JPEGs**.
   - The presence of 2,753 numeric fake training images demonstrates that the training collection contains non-video face portraits, whose exact generator and provenance remain unknown.

4. **Biased Visual Coverage Corrected:**
   - Prior contact sheets displayed only video-pattern face crops.
   - Separate contact sheets have been produced and visually inspected for all four strata (numeric real, numeric fake, video real, video fake) in [subcollection_review.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/subcollection_review.md).

5. **Archive Val Partition is Not a Certified Clean Benchmark:**
   - Zero intersection of recognized `vid_<64hex>` filename tokens between archive `train/` and `val/` is a useful narrow property, but does **not** prove zero content leakage, cross-video identity sharing, or original/manipulated pairing independence.
   - No full cross-partition SHA256 or near-duplicate perceptual audit has been performed across the 211,434 validation images.
   - The archive `val/` partition was previously used by the project to build a 209,871-image dev cohort; R2 omitted it only from its specific local preparation. It cannot be characterized as "pristine" or an "untouched final test."

6. **External Overlap Status is Unknown, Not Zero:**
   - Cross-dataset hashing was not executed between all 1,016,855 archive images and all 60,000 external images.
   - A sample check of 64 numeric fake training images against external fake images showed 62 matching basenames (`0.png` ... `29999.png`) with 0 exact byte matches. Full-population overlap is **unknown**, not zero.

7. **AUC Metrics Do Not Identify Learned Feature Mechanics:**
   - High discrimination (AUC > 0.90) on a validation split reflects ranking ability on that specific cohort, not proof that the detector learned manipulation-intrinsic features rather than background, resolution, or compression artifacts.
   - High performance after format re-encoding cannot prove absence of classification shortcuts.

8. **Encoding Interventions Clarified (No Container-Only Magic):**
   - PNG is a lossless compression algorithm, not an uncompressed raw format. Re-saving decoded JPEG pixels into a PNG container preserves original JPEG compression artifacts while altering the container.
   - Neural network models ingest RGB float tensors; they do not read file headers or extensions.
   - The unsupported phrase "PNG demosaicing artifacts" is retracted.

9. **Explicit Verification Scopes:**
   - Population counts: based on cached Central Directory stream listing (1,016,855 member files).
   - Format counts: extension distribution is known across the population; decoded PIL formats are verified on bounded 256-sample cohorts.
   - Resolution bounds: 40×40 to 372×372 verified on sample strata, not claimed as population bounds.

---

## 2. In-Repository Provenance Audit

A strict audit of repository scripts, notebooks, and documentation revealed the following explicit leads:

| File and Line | Code / Text Reference | Confidence & Meaning |
|---|---|---|
| [data/adapters.py:23-37](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/adapters.py#L23-L37) | `adapt_celeba`: maps numeric filenames (`path.stem`) to CelebA identities | **High Lead:** 6-digit JPEG numeric names (`000001.jpg`–`202599.jpg`) in archive real match CelebA conventions. |
| [data/adapters.py:40-91](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/adapters.py#L40-L91) | `adapt_ffpp`: maps video IDs and pairs to FaceForensics++ | **High Lead:** Video pattern `vid_<64hex>_face_<digits>_<digits>` matches video-derived face tracking pipelines. |
| [data/adapters.py:131-158](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/data/adapters.py#L131-L158) | `adapt_diffface`: references DiffFace benchmark (`ADM`, `DDIM`, `DDPM`, `DiffSwap`, `LDM`, `Real.tar`) | **Moderate Lead:** Explains the project's historical use of placeholder tag `diffgan`. |
| [output/review/REJECTION_RECOVERY_PLAN.md:42](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/REJECTION_RECOVERY_PLAN.md#L42) | *"The paper says most authentic faces come from CelebA while many fakes are video crops or other repositories."* | **Author Context:** Confirms manuscript text stated authentic faces originated from CelebA and fakes from video crops. |
| [TRAINING_COMMANDS.md:49](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/TRAINING_COMMANDS.md#L49) | `DATA_ROOT = Path('/kaggle/input/datasets/abdelrahmanhassani/prepareddatasetdiffgan/Preapred Dataset')` | **Storage Origin:** Archive was uploaded to Kaggle as `prepareddatasetdiffgan`. |

**Verdict:** The repository code contains conceptual adapters for CelebA, FaceForensics++, and DiffFace, but the physical archive `Preapred Dataset.zip` was stripped of all source provenance tables during assembly.

---

## 3. Claim-to-Evidence Matrix

| Claim | Evidence Source | Status |
|---|---|---|
| Archive contains 805,421 train and 211,434 val images | [archive_inventory_summary.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/archive_inventory_summary.json) | **Verified** |
| All fake images in archive train and val have `.png` extension | Central Directory stream listing (752,902 fake files) | **Verified** |
| All JPEG images in archive train and val are in the real class | Central Directory stream listing (202,598 real files) | **Verified** |
| Zero recognized `vid_<64hex>` filename tokens cross archive train/val | Token regex join across 765,164 video-pattern rows | **Verified (token scope only)** |
| Downstream Colab preparation introduced 6,926 crossing families | [relationships_report.json](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/relationships_report.json) | **Verified** |
| R2 100K training pool has 2,753 numeric fake and 38,570 numeric real | Manifest counts (`selected_manifest.csv`) | **Verified** |
| True real sources and fake generators are known | Archive Central Directory (0 provenance files found) | **UNKNOWN / Unverified** |
| Real and fake video frames share no common subject identities | Missing original-to-manipulated mapping metadata | **UNKNOWN / Unverified** |
| Cross-dataset overlap between external dataset and archive | Hash comparison on sample showed 0 matches, but full population unhashed | **UNKNOWN / Unverified** |

---

## 4. Section of Unknowns

1. **Constituent Generators:** The specific generative architectures (StyleGAN versions, diffusion models, FaceForensics++ manipulation algorithms) that created the fake images in `train/fake` and `val/fake` cannot be established from the files.
2. **Numeric Subcollection Origin:** The exact download source of the numeric fake files (`0.png`–`31111.png`) and whether they were generated unconditionally or from source images remains unknown.
3. **Subject Identity Links:** Identity-level grouping across real video crops and fake video crops cannot be established without original video pairing logs.
4. **External Dataset Construction:** The generator and source of `F:\val to be deleted 2` are completely unannotated in local records.
