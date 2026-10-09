# RGB generalization failure: evidence and repair

Date: 2026-10-09. This supersedes the per-image partition recommendation for the current video-frame collection. The previous R2 recipe did NOT fix external generalization. Do not expand that recipe to 500K/1M on the same split on the basis of its 97.22% development accuracy.

## Verified findings from the saved files

The initial investigation analyzed existing CSV/JSON predictions and ZIP metadata only. The subsequent bounded local replay and image profiling are recorded below; no retraining or full external re-evaluation was performed.

- Checkpoint `e598e1b32871f460756118ba6d4a5f1ad3679a3b9727dff1afa04d4d0de96a28`: external accuracy 50.553%, real recall 1.11%, fake recall 99.997%, probability ROC AUC 0.72419. Earlier RGB external AUC was 0.82549 under a different training protocol/cohort: the new experiment regressed on this diagnostic dataset.
- External real median fake probability is 0.99999869. Saved development real median is about 0.00003215. This is a large score shift, not a minor threshold offset.
- Even an invalid hindsight search over all external labels yields maximum balanced accuracy only 67.125% using stored probabilities, or 67.195% using raw logits. These are diagnostic bounds on this particular dataset, NOT new test scores or suggested deployment thresholds. No threshold was exported. Ordinary monotone calibration cannot turn this checkpoint into an 85% detector here.
- FP32 sigmoid saturation collapses rankings: 60,000 rows have only 5,683 distinct probabilities but 59,741 distinct logits. Logit AUC is 0.73390. Reporting saturation more carefully does not rescue the model.
- There are 6,420 shared `vid_<64hex>_face_<frame>_<index>` filename families across train/dev. Affected development rows: 1,682 real + 6,392 fake = 8,074 / 14,835. Affected training rows: 7,310 real + 14,223 fake. These filenames are strong conservative dependencies; original-video/person provenance remains unverified.
- Training real: 38,570 JPG + 11,430 PNG. Training fake: 50,000 PNG. Every external path has PNG extension. Extensions are not verified encodings, and correlation does not prove causal shortcut learning. Nonetheless, this imbalance needs an actual image-format/resolution/content audit.
- Development real recall at 0.5 is 99.91% for JPG and 84.15% for PNG. Different underlying sources/content could explain part of this difference; simply converting filenames or re-encoding once cannot remove all pre-existing compression/resampling artifacts.

Reproducible evidence: `output/review/rgb_r2_failure_evidence.json`. Producer: `tools/summarize_rgb_failure.py`. The downloaded ZIP includes original dev predictions bound to the same checkpoint, enabling a real cross-machine replay.

## What was changed

1. `tools/repair_rgb_family_split.py`: a CPU metadata-only repair. It verifies the completed clean parent's manifest and recovery-report digests, connects existing groups/content hashes/full paths and explicit filename families before splitting, deduplicates content inside components, caps each component to eight samples per class, then selects new train/dev cohorts with separate groups. It leaves original images/manifests unchanged and inherits content verification only under the explicit unchanged-data assumption. It never assigns invented source/video/identity IDs. Shortfalls are reported.
2. `training/validator.py`: refuses cross-partition filename-family overlap even when aggregate-source engineering overrides are enabled. Exact-hash audit alone is no longer accepted as sufficient for this recognizable frame naming pattern.
3. `tools/check_rgb_second_run.ps1`: runs the existing production-path parity diagnostic on 64 saved development samples, checks image hashes and Colab probability agreement, then profiles 256 training + 256 external images for actual encoding, dimensions and other packaging properties. The manifests use full relative-path remapping, never basename matching. No source identity is inferred from appearance.
4. Regression tests for family parsing, leakage prevention, deterministic grouping, caps, forbidden final-test repartitioning, conflicting labels and the training guard.

The split repair is a concrete correction to a known defect. It is NOT a claim that generalization is solved. Source diversity, semantics, encoding/resizing history and pretrained feature suitability remain open questions.

## Completed local replay and image profiles (2026-10-09 follow-up)

The initial replay warning was too broad: it said validation/evaluation paths differed, but tensor, probability and logit differences between those local paths were all zero. The failed checks were nine saved Colab probability comparisons at tolerance 0.0001.

The enhanced diagnostic now records every saved/local probability and its difference, classifications at 0.5, and runtime information. The GPU replay's largest saved/local difference was 0.000845373; CPU replay's was 0.0021916. Neither replay changed any of the 64 classifications at 0.5. Image hashes, checkpoint identity and labels matched. Local BN/dropout evaluation modes were correct. The exact cross-environment numerical cause remains unverified; the strict replay status remains failed, and tolerance was not raised to manufacture a pass. These bounded differences do not explain the roughly 50% external accuracy.

Both image profiles completed, with 256/256 successful decodes each. The stratified training sample contained 64 JPEG real, 64 PNG real and 128 PNG fake images. Mean dimensions were 181 x 200 for real and 165 x 161 for fake. The external sample contained 128 real and 128 fake PNG images, with mean dimensions 256 x 256 in both classes. These are diagnostic samples, not estimates of class/format proportions in the full collection. They establish packaging/resolution differences, not their causal contribution to model errors. The model already used JPEG augmentation; the profile tool's generic hypothesis does not establish that augmentation was absent.

Reports: `output/review/rgb_diagnosis_20261009_210644_10c842f9/{dev_replay_detailed.json,dev_replay_cpu.json,train_profile.json,external_profile.json}`.

The diagnostic launcher now allows independent image profiling after saved-probability-only differences while retaining the failed replay report and warning. Identity or local-path failures still stop it. The external diagnostic CSV now explicitly uses `external_dev`, preserving unknown source/group provenance. There is no need to rerun these diagnostics merely to obtain these reports.

## Optional command: reproduce saved development predictions locally

Keep the current checkpoint. Do not retrain first. From PowerShell:

```powershell
& "G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection\tools\check_rgb_second_run.ps1"
```

This uses prepared small diagnostic CSVs under `output/review/rgb_r2_diagnostic_inputs`. They point to the original local `Preapred Dataset/train` tree and retain saved image hashes. A missing/changed file is a failure to investigate; do not substitute a similar basename or erase its saved hash. Probability tolerance is 1e-4 for this bounded cross-machine FP32 comparison. Passing these 64 samples does not prove every sample or environment is identical.

If identity or local-path checks fail, investigate before retraining. A saved-probability-only failure must be assessed by its magnitude and decision changes rather than automatically treated as an inference bug. Source shift and model representation remain the next experiment's focus; minor numerical differences remain recorded.

## Already prepared: corrected metadata package

The repair has now been run against the full saved clean-parent metadata locally. Its results are in `output/review/rgb_family_repaired_100k_seed42.zip` (about 12.1 MiB), containing exactly:

- `selected_manifest.csv`
- `selected_manifest.verified.json`
- `repair_report.json`

It contains 100,000 training rows (50,000 per class) and 15,000 development rows (7,500 per class), with no detected filename-family overlap between those partitions. CSV/report digests, ZIP integrity and counts were independently checked. Images were not reopened or rehashed; verification is inherited from the saved audited parent, conditional on unchanged image bytes. Unknown dependencies and near duplicates remain unverified.

Save this ZIP to Drive and retain all three extracted files together. Paths still refer to `/content/dataset/Preapred Dataset/train`; restore the same images there in a future Colab session. Keep the original preparation ZIP for provenance. You do **not** need to rerun the CPU repair below when using this package, and the old model must not be resumed on the new split.

**Do not treat this package as a complete generalization fix.** After the family cap, training real images are 46,186 JPG + 3,814 PNG, whereas fake images are all 50,000 PNG. This is an even stronger extension imbalance than before. These are extension counts, not decoded formats. Complete the replay and image profiles, and resolve source/format coverage, before choosing the next training cohort.

## Optional: reproduce the CPU repair using the saved hashes

Use the updated code, not the stale `prepared_rgb_code.zip`. Upload the new helper to your existing repo or push and update the repo. No GPU is needed. The original full `clean_parent` preparation must retain its CSV, verification sidecar and recovery report. This code does not open images, but inherited verification is valid only for unchanged bytes/paths when training later.

Run the metadata regression tests in Colab before repair:

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -m pytest -q tests/test_rgb_family_repair.py
```

Then repair into a NEW Drive directory:

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -u tools/repair_rgb_family_split.py \
  --parent "/content/drive/MyDrive/Generalization_First_Try_RGB/Expirments/rgb_sizes_seed42/clean_parent/selected_manifest.csv" \
  --output_dir "/content/drive/MyDrive/deepfake_rgb/family_repair_seed42_v1" \
  --train_size 100000 --dev_size 15000 --seed 42 --cap 8 --dev_fraction 0.15 \
  --immutable_dataset --allow_shortfall
```

Read `repair_report.json`, particularly actual counts/shortfalls and class-by-extension counts. The cap and whole-group quotas can reduce the available images. This intentionally creates different dev membership; old dev accuracy is not directly comparable. Existing externally inspected data is still external development, never promoted to an untouched final test.

For a LATER new training run, once image profiles/source information have been reviewed, point both `MANIFEST` and `VAL_MANIFEST` to the new `selected_manifest.csv`. Use a fresh run ID, never resume `best.pth` from the old partition. Retraining on a repaired split alone can reveal lower but more honest dev performance without improving external accuracy.

## What the next model experiment must address

1. Obtain actual real-source, fake-generator and original-video relationships from the download sources/extraction process. The aggregate name `diffgan` does not supply this. Do not assume arbitrary `vid_<hash>` prefixes reveal identities or original/fake pairings.
2. Confirm the task: aligned face crops/video frames versus full synthetic images/scenes. A face-only detector evaluated on scenes is a different coverage problem. This needs dataset provenance and image inspection; file names alone cannot answer it.
3. Build source-held-out development sets, covering both classes across supported domains. Balance contributions by real source/fake generator and independent groups. Keep a final untouched external collection out of model/threshold selection. Do not solve this by adding all 60K inspected external examples to training and calling them unseen afterward.
4. Address measured encoding/resolution/semantic confounds with matched real/fake content coverage and the SAME stochastic processing distribution for both classes. Treat format-matched subsets and compression/resize robustness tests as controlled ablations. Mere renaming PNG to JPG, universal JPEG conversion, or lowering/raising a threshold is not an established cure.
5. Compare the original RGB baseline, a frozen-backbone linear probe, and a frozen foundation-image-encoder linear probe (CLIP-style) on the SAME grouped/source-held-out protocol before investing in fusion or larger samples. The repository's UniversalFakeDetect adapter is currently a stub: it cannot be claimed as implemented or benchmarked. Integration/training is a separate next change after the data-contract checks.
6. Use held-out-source balanced accuracy/AUC, real recall and fake recall; retain source-wise results and the linked dev-selected threshold. A representative calibration set is separate from improving the feature ranking itself.

This direction is supported by the original [UniversalFakeDetect paper](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html), [B-Free data-bias study](https://openaccess.thecvf.com/content/CVPR2025/html/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.html), and [Community Forensics generator-diversity study](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html). Their published improvements are not guarantees for this collection.

## Validation status

Existing CSV/JSON/ZIP metadata analysis and the full-parent repair completed. The packaged result has 115,000 rows and passed independent digest, count and archive-integrity checks. Four metadata/guard regression tests passed using unittest (pytest is not installed in the local venv). Python AST, JSON and PowerShell syntax checks passed. The follow-up ran bounded GPU/CPU same-image replay and two image profiles as detailed above. No retraining or full external re-evaluation was run. No improved accuracy is claimed. The missing dataset-source information was requested from the user.
