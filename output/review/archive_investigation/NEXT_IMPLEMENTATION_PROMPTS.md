# Next Implementation Prompts (Evidence-Justified Proposals)

**Status:** PROPOSALS ONLY — UNEXECUTED  
**Context:** Based on empirical findings in [ANSWERS_FOR_REVIEW.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/output/review/archive_investigation/ANSWERS_FOR_REVIEW.md). No training or full evaluation is authorized by this document.

---

## Proposal 1: Extract In-Domain Zero-Leakage Validation Cohort from Archive `val/`

### Rationale
The archive `Preapred Dataset.zip` contains 211,434 images in `val/` with **0 crossing video families** against `train/`. The model has never been evaluated on this true in-domain held-out test split because downstream preparation discarded `val/`.

### Scope
- Extract a balanced 10,000-image evaluation manifest from `Preapred Dataset/val/` using 7-Zip selective list extraction.
- Stratify across:
  - 5,000 Real: 3,500 JPG numeric, 1,500 PNG `vid_` (distinct families).
  - 5,000 Fake: 3,500 PNG `vid_` (distinct families), 1,500 PNG numeric.
- Store manifest at `output/review/archive_investigation/indomain_val_10k_manifest.csv`.
- Extraction storage budget: ~1.1 GiB on Drive E: scratch.

---

## Proposal 2: Compression & Format Invariant Diagnostic Utility

### Rationale
Training data contains 0 JPEG fake images and 162k JPEG real images, while external diagnostic data is 100% PNG. An automated diagnostic utility is needed to test model robustness under uniform JPEG quality-95 and uniform PNG re-encoding on bounded batches (64–128 samples).

### Scope
- Create `tools/diagnose_format_sensitivity.py`.
- Evaluates existing checkpoints on paired batches: (original vs re-encoded JPEG-95 vs re-encoded PNG).
- Measures logit drift and probability shift caused purely by container re-encoding.
- Must be unit-tested on synthetic image fixtures before running on dataset samples.

---

## Proposal 3: Multi-Task Domain Benchmark Formulation

### Rationale
External diagnostic set `F:\val to be deleted 2` represents whole-face GAN/diffusion synthesis, while `Preapred Dataset` represents video face-swapping. A thesis evaluation must either:
1. Clearly delimit scope to video face manipulation (using `val/`), OR
2. Formulate a dual-domain benchmark explicitly categorizing:
   - Domain A: Video Face Manipulation (FaceForensics++)
   - Domain B: Unconditional Portrait Synthesis (StyleGAN/DiffGAN)

### Scope
- User input required on thesis evaluation targets before any training changes.
