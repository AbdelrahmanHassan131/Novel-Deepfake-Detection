# RGB Pilot Protocol and Provenance Specification

**Document Version:** 1.0  
**Status:** Protocol Specification (Static verification complete; execution commands prepared)

---

## 1. Objectives & Scope

Cross-source generalization in deepfake detection cannot be established if the training and development sets draw from the same single collection without verifiable generator provenance. This document defines the protocol for:
1. Auditing dataset source coverage and packaging shortcuts without decoding 1M images.
2. Preserving the historical 98,590-image manifest for baseline reproduction while constructing a separate, auditable new ~100K pilot manifest.
3. Defining strict partition roles (`train`, `dev_selection`, `dev_calibration`, `final_test`) that prevent data leakage and premature threshold fitting.
4. Constructing a group-aware, representative development cohort (10K–20K images) for frequent validation.
5. Clarifying the evidence role of previously inspected external data (`F:\val to be deleted 2`).

---

## 2. Partition Roles & Leakage Protection

Data splits must strictly respect functional roles:

| Split Role | Target Size | Purpose | Allowed Usage | Leakage Protections |
| :--- | :--- | :--- | :--- | :--- |
| **`train`** | ~100,000 (balanced 50k/50k) | Model parameter optimization | Loss backprop, optimizer steps | Excluded from validation metrics and threshold selection. |
| **`dev_selection`** | 10,000–20,000 (balanced 5k/5k to 10k/10k) | Frequent checkpoint selection (every epoch) | Monitored metric for `best.pth` | Connected groups kept together; never overlaps `train`. |
| **`dev_calibration`** | ~5,000–10,000 | Fitting operating decision thresholds (Youden J) | Independent threshold estimation | Calibrated on development only; NEVER on test sets. |
| **`full_dev`** | 209,871 (historical cohort) | Comprehensive evaluation & reporting | Occasional benchmark reports | Fully preserved; unused rows NEVER moved to training. |
| **`final_test` / `external_test`** | Variable | Unseen cross-source performance | Final evaluation ONLY | **PROTECTED_SPLITS**: Forbidden from training, validation, or calibration. |

### 2.1 Critical Clarification: `F:\val to be deleted 2`
`F:\val to be deleted 2` has already been inspected, evaluated, and analyzed in earlier project rounds (yielding 50.37% Wang2020 accuracy and external fusion real/fake score distributions). Therefore:
- It constitutes **external development evidence**, NOT an untouched final test set.
- Renaming folders or resplitting it does not make it an independent double-blind evaluation.
- Operating thresholds may be tested against it to diagnose failure modes, but final scientific claims of generalization must be tested on genuinely untouched cohorts.

---

## 3. Metadata Coverage & Auditing

A source coverage report must be generated before claiming generalization readiness:
```powershell
python prepare_dataset.py source_coverage --manifest manifests/pilot_manifest.csv --output reports/source_coverage_report.json
```

### 3.1 Provenance Principles
- **Collection vs. Generator:** A label like `diffgan` represents an aggregated collection folder, NOT a verified generator. Constituent generators (e.g. DDPM, LDM, StyleGAN2) must be specified.
- **Group IDs:** Strings formatted as `independent:<path hash>` are synthetic fallback anchors to prevent filename collisions, NOT evidence of independent human subjects or video capture sessions.
- **One-Class Sources:** If a source contains only real or only fake images, a classifier risks learning the camera or container compression signature of that source rather than facial manipulation artifacts.
- **Fail-Closed Gate:** If only a single dataset source is present in the training cohort, cross-source generalization experiments are marked **BLOCKED** in status reporting, though independent code tasks may proceed.

---

## 4. Reusable ~100K Pilot Manifest Creation

To isolate fine-tuning and augmentation hypotheses without confounding source mixtures:
1. **Preserve Historical Manifest:** The verified 98,590-image manifest (`kaggle_train_98k.csv`) and its audit gate must NEVER be overwritten.
2. **Create a NEW Manifest:** When adjusting source quotas, write to a separate manifest (e.g. `pilot_manifest_v2.csv`) with its own independent `.verified.json` gate.
3. **No Image Copying:** The pilot selection selects rows and assigns splits in place; original files remain intact in their source storage.

---

## 5. Group-Aware Representative Dev Cohort

Validating on the full 209,871-image dev set every epoch would consume 2–3x more time than training. Therefore, a deterministic, group-aware representative subset is created:
- **Size:** 10,000 to 20,000 samples (e.g., exactly 15,000 samples: 7,500 real / 7,500 fake).
- **Selection Command:**
  ```powershell
  python prepare_dataset.py representative_dev `
      --manifest manifests/full_dev_manifest.csv `
      --train_size 15000 `
      --output manifests/representative_dev_15k.csv `
      --seed 42
  ```
- **Integrity Guarantee:**
  - Preserves connected groups (identities/videos) in the same cohort.
  - Generates a fixed SHA-256 cohort digest.
  - Leaves the full 209,871-image development set intact for occasional final epoch validation.

---

## 6. Image Profile Diagnostic

Before launching extensive training runs, inspect container and resolution properties:
```powershell
python tools/profile_image_cohort.py `
    --manifest manifests/pilot_manifest.csv `
    --sample_size 500 `
    --seed 42 `
    --output reports/pilot_image_profile.json
```
This tool checks container formats (JPEG vs PNG), resolution distributions, and aspect ratios, emitting packaging shortcut hypotheses without decoding whole multi-gigabyte pools.
