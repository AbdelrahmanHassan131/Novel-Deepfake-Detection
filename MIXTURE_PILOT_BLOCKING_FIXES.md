# Mixture pilot review: fix these blockers before training

## Verified progress

The reviewer independently read the actual manifests: Arm A and B each have 20,000 train + 2,000 dev rows with the requested quotas. All shared-dev rows match exactly. Listed group/hash/path/family tokens do not cross train/dev; maximum observed group/class counts are 6 for A and 8 for B. CSV digests match their sidecars. These checks do not verify original identities, source provenance or unknown near duplicates.

The reviewer ran `python -m unittest tests.test_controlled_mixture_manifests tests.test_evaluate_mixture_predictions tests.test_diagnose_format_sensitivity tests.test_rgb_family_repair -q`: **28 tests passed**. The implementation report's breakdown is inaccurate: these four modules account for 8 + 6 + 10 + 4 tests. Do not count pytest-style functions as executed by unittest or claim unrun gate regressions passed.

**Status: not ready for Colab training.** Fix the issues below; keep the experiment, quotas and image selection unchanged. No extra diagnostic campaign, new training recipe or local real training is requested.

## Prompt 1 — Restore valid inherited hash evidence (highest priority)

`tools/build_controlled_mixture_manifests.py::build_and_save_manifests` currently reads only the parent CSV, computes its digest, then writes `verified=True` and `hashes_verified=True`. It never reads or validates the clean parent's `.verified.json` or its bound recovery report. This would certify an unaudited CSV containing arbitrary SHA strings. The existing CSV selections may be correct, but the sidecar generation does not establish the claimed verification chain.

For both ZIP and CSV inputs, require the matching clean-parent verification sidecar and `recovery_report.json`. Check parent `verified` and `hashes_verified` are literally true, sidecar manifest digest matches actual CSV bytes, and recovery-report digest matches the binding. Reuse the earlier family-repair contract. Require/document the unchanged-image assumption explicitly; do not open or rehash the full image dataset. Retain unknown near-duplicate/source-video status accurately.

Bind derived sidecars to parent CSV, parent gate, recovery evidence and preparation report hashes. Validate evidence before writing any verified outputs; use atomic writes and a new output directory. Do not silently overwrite the existing preparation. Rebuild/certify into a new versioned directory from the same parent with the same selection parameters. Confirm each selected CSV's rows and membership match the existing cohorts; changing a gate should not require new data selection. If byte ordering differs, document why and compare canonical rows.

Add file-level tests exercising `build_and_save_manifests`, not only its pure selection helper: missing gate, false hash status, mismatched parent digest, missing/tampered recovery report, valid inheritance from both ZIP and CSV, and nonempty destination rejection. Negative cases must not emit a successful gate. Correct the protocol report's verification claim. Existing saved audit evidence is enough; no million-image rehash is required.

## Prompt 2 — Make the Colab guide actually executable

Repair `COLAB_RGB_MIXTURE_PILOT.md` and test its cells against the real parser and experiment layout.

- Put `%%bash` at the beginning of each Bash cell. Put comments after it. Set shared variables once in a Python configuration cell using `os.environ`; shell `export` does not persist to later notebook cells.
- Respect the user's known Drive repository location: `/content/drive/MyDrive/Generalization_First_Try_RGB/Novel-Deepfake-Detection`. Make it editable and validate existence. Local code may live at `/content/Novel_Deepfake_Detection`. Do not print “Updating existing repository” while doing nothing; verify the expected code/package version or use a fresh destination.
- Provide an explicit way to transfer the updated code and newly certified manifest package to Drive. Copy only needed code/assets, excluding Windows `.venv`, `.git`, full historical results and checkpoints. Keep an inventory/digest of what the package contains. Do not presume local files have already reached Drive or reuse the stale preparation-code ZIP.
- Remove `p7zip-full` from the pip install list: it is being used here as an OS package. Use the appropriate system package step separately, verify an available compatible 7-Zip executable, and avoid unnecessary reinstall of Colab's working Torch/CUDA stack. Inspect actual imports/dependencies instead of inventing a short package list.
- Existing dataset reuse must check all required union paths, not merely whether two class folders contain >=29,163 files. If paths are missing, extract only the required missing members or fail with a useful list. A file count cannot establish membership.
- Validate all split archive volumes, exact selected member names, traversal/absolute/link safety, storage budget and selected sizes before extraction. Use exact-member selection with wildcard expansion disabled where supported. The Drive archive path is a configurable assumption, not a verified uploaded location. After extraction, verify every required path. No full payload hash pass.
- `ExperimentManager` constructs `<name>_<run_id>/checkpoints/`. Current guide uses the same string for name/run_id but invents `$OUTPUT_BASE/$RUN_ID`, so its log and checkpoint paths disagree. Use a stable name and unique ID, calculate/store the actual run directory, and preserve it across cells in a run registry JSON.
- `best_dev_predictions.csv` is under the actual run's `checkpoints/` directory. Step 8 currently searches the wrong level. Use recorded exact A/B run paths, never an ambiguous latest-file glob. Check each prediction file and checkpoint before comparison. Actual checkpoint names are `best.pth`, `last.pth`, `model_epoch_5.pth`; correct the protocol's invented `best_dev.pth` / `final_epoch_5.pth` names unless explicitly implementing those aliases.
- The benchmark supplies `--val_samples` without `--val_manifest`, so no validation benchmark occurs. Supply the correct shared-dev manifest, and describe its actual 10 warmup + 50 measured microbatches, not “100 batches.” Validate every CLI option with the real parser without starting training.
- Save each completed arm to Drive immediately with checked completion status; a failed/interrupted copy must not look like a completed backup. Keep logs and run registry with outputs. Warn accurately about unsaved local work during runtime loss rather than promising persistence.
- Protocol options should match the explicit commands and saved R2 metadata, including Adam beta1/weight decay, head configuration, preprocessing and augmentation. Do not replace the repo's existing ImageNet weight selection with an invented `ImageNet_Default` implementation. Arm A 551 numeric fake / 2,286 video real are 20K quota values, not R2 100K counts. Remove unsupported fixed runtime estimates.

Test the whole path/layout workflow with temporary directories and a mocked trainer producing the real checkpoint/export structure. Add a notebook-cell syntax/parser check that makes no downloads or real training calls. Deliver complete corrected cells, not scattered patches for the user to assemble.

## Prompt 3 — Make the comparison fail closed and remove invalid statistics

`tools/evaluate_mixture_predictions.py` still allows `mixed` or `unknown` checkpoint hashes and silently skips manifest validation when a supplied path does not exist. Its manifest join checks IDs/labels only, ignoring mismatched image hashes, nonempty paths and groups. Fix these issues before the resulting reports are trusted.

- Require a valid shared-dev manifest and gate for this workflow; reject a missing path, duplicate/unknown IDs, wrong splits and wrong cohort counts. Verify predictions cover exactly its rows. Each prediction row must carry the same valid checkpoint digest within its arm. Different checkpoints across A and B are expected, mixed identities within an arm are not.
- Verify labels, image SHA256, group and canonical full paths against the shared manifest. Allow only explicit path relocation; never basename matching or silent fallback over mismatched metadata. Reject unknown filename patterns instead of omitting them from the minimum-stratum metric.
- `select_best_threshold` clips the actual maximizing threshold into `[0.01,0.99]`, which can change decisions and invalidate the optimization, especially with saturated probabilities. Use the repository's existing tested development calibration logic or explicit finite candidate thresholds with deterministic tie handling. Record the rule and bind the threshold artifact to checkpoint, predictions, shared manifest and `source_splits=['dev']`. Label metrics computed on the threshold-fitting data as development fit, not independent performance.
- The report claims an exact McNemar p-value and the module docstring claims group-bootstrap intervals, but the code supplies only a chi-square approximation treating frames as independent. This cohort contains multiple frames per connected group. Keep paired error counts descriptive and omit significance p-values for this first pilot, or implement genuinely group-aware uncertainty with its assumptions and tests. Do not describe an approximation as exact. Additional statistical machinery is not required to run this engineering pilot.
- Save each comparison in a fresh directory. Validate cross-arm labels and cohort identity before paired counting. Preserve denominators in reports rather than hardcoding misleading counts for arbitrary inputs.

Add regression tests for mixed/missing hashes, invalid gate, missing manifest, path/hash/group mismatches, reordered valid rows, unknown strata, and saturated-probability calibration. Include an end-to-end comparison fixture using exactly the prediction export schema from `CheckpointHook`.

## Prompt 4 — Return the corrected runnable package

Run the relevant tests, distinguishing actual executed unittest tests from pytest tests. Add tests that directly exercise the failures above; the current 28 passing tests do not cover them. Preserve test outputs and exact commands. Validate the new manifest package and list its exact paths, sizes and digests. Preserve existing staged/user changes and do not commit or push unless separately requested.

Return `output/review/archive_investigation/MIXTURE_PILOT_READY_REPORT.md` with: addressed blockers, tests actually run, new verified manifest directory/package, corrected Colab guide, and any remaining limitation. Do not start real training or re-extract the complete dataset. This is a targeted correctness fix to the agreed pilot, not another change to the scientific experiment.
