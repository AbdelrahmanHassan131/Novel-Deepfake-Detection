# Next step: controlled RGB data-mixture pilot

## Review decision

The bounded format experiment supplies useful evidence. The reviewer independently recomputed confusion counts from the 512 saved condition rows (128 unique sample IDs) and reran the eight diagnostic unit tests successfully. No further checkpoint inference was needed for this review.

| Cohort | Original / PNG control | JPEG95 | JPEG75 |
|---|---:|---:|---:|
| Archived training diagnostic, 64 images | 96.875% balanced accuracy | 96.875% | 93.750% |
| External development diagnostic, 64 images | 51.5625% balanced accuracy | 51.5625% | 53.125% |

External real recall was 1/32 before intervention and 2/32 at JPEG75; external fake recall remained 32/32. Training real recall fell from 30/32 to 28/32. These tiny, deliberately selected samples are diagnostic results, not a new full-test accuracy estimate. PNG pixel/tensor controls were exact for this run.

**Decision:** do not adopt universal JPEG conversion as a cure. A controlled training experiment can now test the data-mixture hypothesis. More rounds of the same format diagnostic are unnecessary. No evidence yet establishes an improvement in generalization.

## Important qualifications to the latest agent report

- `F:\val to be deleted 2` is repeatedly inspected **external development**, not an untouched unseen holdout. Correct the contrary statement in section 6.2.
- The experiment does not prove that sensor noise, geometry, lighting or demographics dominate the failure. No demographic analysis was performed; remove that claim. It also does not identify which frequency components the model used.
- A one-step JPEG intervention cannot reproduce an unknown source's processing history. Its failure to repair errors does not logically exclude all compression-related explanations.
- Exact PNG controls apply to this cohort and pipeline, not every possible decoder/environment.
- R2 already used `rgb_v1` with JPEG probability 0.5 and blur probability 0.5. Adding the same JPEG augmentation is not a new intervention. Keep it unchanged in the first pilot comparison.
- The 64 archived-training diagnostic examples come from the archive train pool; do not assume all were members of R2's selected 100K training cohort unless verified by manifest membership.

## Instructions for the implementing agent

Repository: `G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection`.

Implement the preparation/reporting changes and deliver Colab commands below. **Do not launch real training, full archive extraction, full-dataset rehashing or another full external evaluation locally.** Preserve original checkpoints and results. Use existing saved hashes. User will execute training on a single Colab GPU after the implementation is reviewed.

### Prompt 1 — Freeze the protocol and correct report wording

Create a versioned protocol JSON and short Markdown description. Hypothesis: the R2 sampling mixture underrepresents numeric fake and video-pattern real examples; equal representation of four observable strata may improve robustness. This is an engineering hypothesis, not a verified-source experiment or promised accuracy gain.

Use one architecture (`Wang2020_128`), the same ImageNet initialization, head/fine-tuning/BN settings and existing `rgb_v1` recipe as the saved R2 checkpoint metadata. Use the same seed, batch size, optimizer, learning rates and a fixed five-epoch budget in both arms. Disable early stopping for this comparison so exposure budgets match. Keep image preprocessing unchanged. Do not resume R2 weights: they have seen examples that may move into the new development partition. Save final-epoch and development-selected checkpoints with explicit identities. Freeze the checkpoint selection rule before training, using existing supported development AUC initially, and report all four strata alongside it.

Correct the overclaims listed above in a new review note; preserve the original diagnostic outputs. Actual dataset origins remain unknown and should be requested as useful provenance, not fabricated or treated as a reason to forbid a clearly labeled engineering experiment.

### Prompt 2 — Build two small grouped manifests with one shared dev cohort

Extend/reuse `tools/repair_rgb_family_split.py` and the audited full `clean_parent` metadata. Do not independently repartition each arm or select only from the already reduced repaired 100K subset. Establish connected groups once using existing parent relationships, content hashes, full paths and conservative filename families, then fix their train/dev assignment before quota selection. Retain explicit `image_level_unverified` provenance; numeric names do not verify independent sources or identities.

Define observable strata using canonical class plus exact numeric/vid filename patterns. Unknown patterns must be reported, not silently assigned. Never write these strata into `dataset_source` as if they were verified sources. Deduplicate content and keep at most eight images per connected group per class. The same group may be present in both training arms, but no group/hash/path/recognized family may cross either arm's training cohort and shared dev.

Prepare these target quotas, without replacement:

| Stratum | Arm A: R2-like mixture, 20K | Arm B: equal mixture, 20K | Shared dev, 2K |
|---|---:|---:|---:|
| Numeric real | 7,714 | 5,000 | 500 |
| Video-pattern real | 2,286 | 5,000 | 500 |
| Numeric fake | 551 | 5,000 | 500 |
| Video-pattern fake | 9,449 | 5,000 | 500 |

Both training arms are balanced real/fake. Arm A approximates R2's within-class mixture at smaller scale; it is not a reproduction of the original leaky R2 experiment. Use deterministic common ranking within each stratum so shared selections are stable. Keep dev rows exactly identical between arms. Do not pad quotas with duplicate frames or weaken group caps silently. First report post-grouping capacity; if these quotas cannot be met, emit an explicit shortfall report and a smaller feasible equal-budget proposal before generating runnable training commands.

Bind each derived manifest and report to parent verification evidence and document unchanged-byte hash inheritance. Produce counts by class/stratum, group counts, cross-partition overlap checks, parent digests and actual quotas. Preserve the original CPU preparation and all saved manifests. Unit-test deterministic selection, caps, exact-hash grouping across names, paired shared-dev identity, quotas/shortfalls and rejection of final-test repartitioning using synthetic metadata.

### Prompt 3 — Add per-stratum development reports without inventing new sources

Reuse existing checkpoint-linked prediction exports and corrected tie-aware metrics. Report each stratum's denominator, errors, real/fake recall as applicable, probability/logit summaries, and both-class numeric/video AUC where defined. Report overall balanced accuracy and the minimum of the four applicable stratum recalls as diagnostics; do not label these source-held-out metrics. Keep AUC undefined for one-class strata.

Compare A versus B on exactly the same dev IDs. Include paired error-change counts and group-aware uncertainty if implemented; a single seed is exploratory. Retain fixed-0.5 metrics, plus a separately labeled threshold selected only from development predictions and bound to its checkpoint. Never tune a threshold or choose JPEG quality using external labels.

### Prompt 4 — Deliver a practical single-GPU Colab workflow

Write `COLAB_RGB_MIXTURE_PILOT.md` with complete cells, one at a time: mount Drive; restore current code and derived manifest packages to local `/content`; restore required image bytes; verify paths/gates; choose Arm A or B; run a bounded throughput check; train; save checkpoints/predictions/configuration to Drive; compare shared-development results. Use clear variables and fresh run IDs; rerunning setup must not hide earlier checkpoints. Preserve logs with unbuffered output and a stable log file.

Image paths remain rooted at `/content/dataset/Preapred Dataset/train`. If the user already restored that dataset, reuse it. Provide a selective extraction path for only the union of selected train/dev members when feasible, preserving relative paths; do not require 1M images or another complete hashing pass. Split-ZIP reader requirements and existing archive members must be handled explicitly. Validate storage and archive-member paths before extraction. Never claim a standard single-file ZIP reader supports these split volumes without checking.

Use the same batch and accumulation settings in both arms. On the selected Colab GPU, measure rather than promise saturation; prefetch/pin workers appropriately, keep images on local disk, and do not increase batch size merely to fill VRAM. No model architecture change, new backbone, fusion head or augmentation sweep belongs in this comparison.

### Prompt 5 — Complete implementation verification and return one runnable handoff

Run the new metadata/reporting tests and relevant existing family/gate regression tests. Use synthetic data only for any training-loop smoke test. Inspect notebook/cell commands against actual parser options; do not invent flags. Provide exact changed files, tests run, limitations and preparation steps still requiring the user's Colab files.

Return `output/review/archive_investigation/MIXTURE_PILOT_IMPLEMENTATION_REPORT.md` and the Colab guide. No new scientific accuracy claim is expected before training. Do not promise that equal mixtures remove format confounding: numeric real remains JPG-heavy and fake remains PNG-heavy. If the later pilot fails externally, report the failed hypothesis and prioritize provenance and better real/fake coverage before scaling to 500K/1M. If it improves, confirm with additional seeds and new held-out data before making a generalization claim.

## Diagnostic utility maintenance (does not require repeating this completed run)

Before reusing the generic format utility, reject nonempty output directories rather than overwriting reports, require explicit cohort metadata rather than treating ordinary `dev` as external, reject nonpositive batch sizes and nonfinite/wrong-length outputs, and add an actual production-transform tensor parity test (the current nondefault test only checks that output exists). These reusable-tool issues do not by themselves invalidate the supplied run with explicit cohort labels, expected model output, fresh destination and standard preprocessing. Keep maintenance separate from the data-mixture experiment.
