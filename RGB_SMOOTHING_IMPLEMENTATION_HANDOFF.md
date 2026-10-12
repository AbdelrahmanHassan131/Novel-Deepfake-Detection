# Executable prompts: RGB label-smoothing experiment and reliable logit evaluation

Prepared 2026-10-12. Give this entire file to the coding agent, or execute the numbered prompts in order in the same checkout. Implement the code and produce the final Colab guide. Do not stop after proposing another plan. Do not launch real training or full-image inference locally. No existing checkpoint is required to implement or test these changes.

## Objective and evidence

Build two matched three-epoch Colab runs on the existing diffusion-portrait folders: a conservative RGB control and the same model with mild training-only label smoothing. Test whether smoothing reduces extreme confidence while retaining or improving ranking. Do not promise 85% accuracy or claim smoothing repairs source bias.

The current conservative checkpoint obtained these external results on 60,000 balanced images:

- Accuracy at probability 0.5: 0.50065.
- Probability AUC: 0.6102977; original-logit AUC: 0.9259208.
- 77.6733% of real and 99.7033% of fake probabilities rounded to exactly 1.0.
- Retrospective external calibration: logit threshold 25.587146759033203; accuracy 0.8463667.
- That threshold belongs ONLY to the existing checkpoint. Never reuse it for newly trained models.
- The external collection has repeatedly guided development. It is external development/calibration evidence, not an untouched final test. The user currently cannot supply another dataset.

The existing folder dataset contains 160,000 training and 40,000 development images. Real images are CelebA; fake folders were drawn from ADM, DDIM, DDPM and DiffSwap. Identity checks do not establish full content/group independence. Do not describe this dataset as fully audited. Do not silently rebuild it, change labels, resplit images, remove images, add external images to training, or replace source metadata with guesses.

## Fixed experiment contract

Both new runs use:

- `Wang2020_128`, ImageNet initialization, 128d head, dropout 0.5.
- `fine_tune_policy=layer4_and_head`, `bn_policy=frozen`, `backbone_lr_mult=0.1`.
- Adam, head LR 0.0001, beta1 0.9, weight decay 0, constant LR (`lr_policy=none`).
- Three epochs, no decay epochs, no early stopping, seed 42, no resume.
- Same existing 160K/40K folders; `--allow_folder_training` explicitly marks an engineering experiment with incomplete independence auditing.
- Batch size 64, accumulation 1, two loader workers, pinning, persistent workers, prefetch factor 2.
- `rgb_v1` augmentation, unchanged crop/resize settings.
- AMP FP16 training for BOTH runs; FP32 validation and external inference.
- Checkpoint selection by development logit AUC, with a fixed deterministic tie policy. Never choose checkpoints by external performance.
- Save every epoch, best, last, exact config, code commit and checkpoint-linked best-development predictions with original logits.

Control: `--rgb_label_smoothing 0.0`.
Experiment: `--rgb_label_smoothing 0.05`.

Define this parameter as target shrinkage: `target = label * (1 - 2*s) + s`. Thus 0.05 means real target 0.05 and fake target 0.95. This is NOT the alternative convention that mixes by s with a uniform distribution and produces 0.025/0.975. Document the definition explicitly.

The old one-epoch FP32 run remains historical context. The new two-run comparison is matched in duration and precision. The two changed policies must not be mixed with extra loss penalties, mixup, optimizer changes or augmentation experiments.

## Prompt 1 — Trace the current contracts and add the configuration option

Read repository instructions, inspect git status, and preserve unrelated work. Inspect the real call paths before editing; the following paths exist now but confirm their interfaces:

- `train.py`
- `config/defaults.py`, `config/configuration.py`, `config/compatibility.py`, `config/validator.py`
- `models/wang2020_128/trainer.py`
- `training/validator.py`, `training/hooks/checkpoint_hook.py`, `training/checkpoint_manager.py`
- `experiment/metrics/metrics.py`, `experiment/metrics/accumulators.py`
- `evaluation/generalization.py`, `evaluation/metrics/classification.py`, `evaluation/evaluator.py`, `evaluation/inference.py`
- Existing calibration entry point, report writers, and ROC/PR visualization call sites.

Add `--rgb_label_smoothing` as a finite float with default 0.0 and allowed range 0 <= s < 0.5. Propagate it through CLI, structured config, legacy/config conversions, saved options, and checkpoint/resume validation. Missing values in old checkpoints mean 0.0. Reject NaN, infinities, negative values and s >= 0.5. Reject nonzero use with unsupported architectures rather than silently ignoring it. Scope this new experiment to Wang2020_128.

Deliver working config round-trip and old-checkpoint fallback tests; do not alter the experiment defaults globally to enable smoothing.

## Prompt 2 — Implement training-only smoothing in the RGB loss

Modify the RGB trainer so both `get_loss()` and its direct `optimize_parameters()` path use one shared loss implementation. For active TRAIN mode only, transform targets with the exact formula above. For validation/evaluation use the original hard labels and ordinary BCEWithLogitsLoss. Do not mutate `self.label`, dataset labels, logits, or probabilities. Keep BCE on logits, with existing numerical/autocast behavior.

Important: `opt.isTrain` or a constructor-time training flag is not sufficient to distinguish a validation pass inside a training process. Use the active underlying network training mode and verify that the validator's `eval()`/`train()` transitions work. Frozen BatchNorm modules must remain frozen without disabling training-time smoothing in the classifier. Preserve gradient flow and all current fine-tuning policies.

Acceptance tests:

1. s=0 reproduces current BCE values and gradients.
2. s=0.05 training loss equals explicit BCE against [0.05, 0.95] for hard labels [0, 1].
3. `eval()` on the SAME trainer returns hard-label BCE; switching back restores smoothed training loss.
4. Original label tensor is unchanged, and frozen backbone/BN behavior is preserved.
5. Direct optimizer and engine `get_loss()` paths agree.
6. Saved/reloaded configuration preserves s; resume cannot silently change the loss setting.

No large model download is necessary: use the repository's small fixtures/mocks or pretrained=False for focused tests.

## Prompt 3 — Preserve original logits throughout validation and checkpoint export

The standalone inference runner already exports original logits, but the current training validator accumulates only sigmoid probabilities and the checkpoint hook does not pass logits to `export_predictions`. Fix that missing link.

- Capture canonical real=0/fake=1 logits, including the existing `score_sign` correction exactly once.
- Extend the validation result compatibly to carry logits alongside probabilities, labels and dataset indices.
- Gather, trim, sort, and align logits identically to other arrays under DDP, including uneven/empty local shards. Do not introduce padding duplicates or change sample ordering.
- Compute development ranking metrics from logits and preserve existing metric names consumed by checkpoint selection, plus explicit score-space metadata.
- Export a `logit` column in both authoritative `best_dev_predictions.csv` and any maintained alias, linked to the actual saved best checkpoint digest and FP32 prediction precision.
- Never reconstruct logits using log(p/(1-p)) from saturated probabilities. Legacy probability-only files may still be read for legacy behavior, but cannot support logit calibration.

Tests must verify per-sample alignment, sign handling, checkpoint linkage, and that a set of distinct large logits remains distinguishable even when every float32 sigmoid probability is 1.0. Add a CPU distributed test or reuse the repository's existing distributed fixtures for the new gathered field; do not require Colab GPUs.

## Prompt 4 — Make metrics, plots and decision thresholds explicitly score-aware

Introduce a shared, tested implementation rather than different AUC/EER formulas in training and evaluation. Preserve backward-compatible callers and ordinary unsaturated results.

- AUC, ROC, average precision/PR and interpolated EER use canonical raw logits whenever original logits are available. Preserve probability-based diagnostics separately when useful, naming score spaces explicitly.
- Legacy inputs without logits must explicitly report a probability fallback; do not claim saturation was repaired.
- EER must interpolate the FPR/FNR crossing, not return FPR at the nearest discrete ROC point. Do not fabricate an operational EER threshold when interpolation spans a tie jump. Explain diagnostic interpolation.
- Fixed 0.5 probability classification corresponds to logit >= 0. For an arbitrary probability threshold in (0,1), use a stable logit boundary when logits are available. Define and test endpoints 0/1 explicitly; never clip logits or change AUC by clipping.
- For a logit threshold, compare raw logits directly. Do not sigmoid the threshold to float32 and compare saturated probabilities.
- Confusion matrices, accuracy, recalls, F1, exported predictions, plots and JSON/Markdown reports must all reflect the SAME decision rule. Probability histograms may remain probabilities, but label them accordingly.
- Existing `binary_metrics`/`summarize`, `ClassificationMetrics`, training metrics and evaluation reports must not disagree about score space.
- Preserve any existing documented video aggregation semantics; do not silently replace mean probability aggregation with mean logits. Explicitly report score space/fallback for aggregated metrics if relevant.
- Handle single-class inputs with clearly undefined ranking metrics rather than fabricated zero/one performance. Reject nonfinite logits, mismatched lengths or inconsistent threshold artifacts.

Regression fixtures must include: real logits [19,20,21], fake logits [27,28,29]. Float32 sigmoid ties may yield probability AUC 0.5, but logit AUC must be 1.0; threshold 27 gives perfect classification and threshold 0 predicts all fake. Include ties, EER interpolation, reversed score signs, single-class data and ordinary nonsaturated behavior.

## Prompt 5 — Add checkpoint-linked logit calibration and evaluation support

Extend the existing calibration interface with an explicit score-space option, such as `--score_space logit`; retain the current probability default for legacy callers. Logit calibration requires original finite logits. Select finite ROC thresholds maximizing development Youden J with a deterministic tie policy. It must never fit on forbidden test splits.

Artifact contract:

- New artifacts identify schema version, `score_space=logit`, finite `logit_threshold`, `decision_rule='fake if logit >= logit_threshold'`, label mapping, criterion, checkpoint SHA256, prediction SHA256, source splits and precision.
- Preserve calibration provenance and overlap checks already present. External calibration must remain explicitly opted into and marked as such. Missing/unknown splits must not be silently relabeled from filenames.
- `evaluate.py --threshold_file` accepts new logit artifacts as well as old probability artifacts. A legacy artifact with no score_space remains probability-based. Reject ambiguous/conflicting fields and wrong-checkpoint artifacts before inference.
- Support the actual saved artifact from `COLAB_LOGIT_CALIBRATION.md` by a narrowly validated reader or explicit migration command. It uses `score_space`, `logit_threshold`, `checkpoint_sha256`, `calibration_prediction_sha256`, `calibration_status`, and a decision rule, but no `source_splits`. Do not infer generic missing provenance as permission: recognize its exact external-calibration schema/status and mark it external calibration, or require explicit migration acknowledgment. Never silently borrow this old checkpoint's threshold for a new run.
- Provide a CPU command to compute both fixed-0.5 metrics and frozen-development-logit-threshold metrics from ONE saved external prediction CSV. No second inference pass should be necessary.
- Preserve the current calibrated checkpoint and threshold 25.587146759033203 as baseline artifacts; do not overwrite them or fit a replacement without request.

Tests: valid old/new artifact behavior, checkpoint mismatch, malformed score spaces, finite thresholds outside [0,1] allowed only for logits, forbidden calibration splits, mixed/missing precision, unknown logits, explicit old-artifact migration, and fixed-threshold re-scoring without test-label fitting.

## Prompt 6 — Deliver the actual Colab commands and comparison script

Create `COLAB_RGB_LABEL_SMOOTHING_AB.md` with executable, cell-by-cell Python/bash commands, not placeholders for implementation. It must work on the existing data and require no new dataset. Use these paths:

```text
Repository branch: GeneralizationDeepFakeDetection
Repository URL: https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git
Preferred local checkout: /content/Novel_Deepfake_Detection
Train: /content/dataset/DatesetNoVideosNoGAN/train
Development: /content/dataset/DatesetNoVideosNoGAN/val
External development: /content/unseenDataset/val to be deleted 2
New output base: /content/drive/MyDrive/deepfake_checkpoints_Gen_Third_Try/label_smoothing_ab
```

Required workflow:

1. Mount Drive, check GPU, folders, dependencies and new CLI options. Allow a fresh clone of the USER-PUSHED code; never delete an existing checkout or dataset automatically. Log the commit and effective configuration. Clearly say local edits must be pushed before Colab can clone them; the agent must not push without authorization.
2. Establish a unique experiment ID with persistent paths. Do not create the exact run directory before ExperimentManager creates it. Both runs use distinct names and output paths. Reject accidental overwrite; do not switch to resume automatically. Save small logs/checkpoints/reports on Drive; keep image reads local.
3. Run control with s=0 for three epochs using the full fixed experiment contract above, with visible unbuffered batch/epoch output and fail-fast exit handling. Start a separate process per run so seeds reset consistently.
4. Run smoothing with s=0.05 for three epochs, otherwise identical. Do not add a third experiment or launch a hyperparameter search.
5. For each run, select best.pth by internal development logit AUC and calibrate a threshold from its checkpoint-linked best-development logits. These thresholds are development-fit values; current source bias may still prevent transfer. Do not stop the experiment because internal AUC looks perfect.
6. Infer ONCE per selected model on the 60K external development images in FP32. Produce a combined table using 0.5 and each run's OWN frozen internal-development threshold: logit AUC, probability AUC diagnostic, accuracy, balanced accuracy, class recalls, macro F1, interpolated logit EER and saturation fractions. Include checkpoint/threshold identities. Verify both models were scored on the same samples and labels.
7. No automatic external threshold optimization in the primary comparison. If providing an optional externally fitted diagnostic, keep it disabled by default, clearly mark calibration fit, and require an explicit command. Do not reuse the old threshold 25.587... for either new model.
8. Print exact result paths and persist configs, best/last/epoch checkpoints, best-dev predictions, threshold artifacts, external predictions, logs and comparison JSON/Markdown to Drive. Explain which result the user should send back.

Performance: use supported AMP FP16 for both runs, FP32 validation, batch 64/accumulation 1 and two workers initially. No GPU-memory-fill requirement or fake utilization guarantee. Provide a clearly recorded OOM fallback (batch 32, accumulation 2, validation batch 32), applied to both matched trials if needed. No hidden tuning run or full hashing rerun as part of notebook startup.

Clearly state that folder-mode data auditing remains incomplete and this experiment measures development improvement on an already inspected external collection. Do not make clean-test claims or imply an independent final result is available.

## Prompt 7 — Test the integrated workflow and finish the handoff

Run focused tests for the changed paths plus relevant existing suites. Start with the actual tests present, including `test_rgb_contract.py`, `test_rgb_review_final_repairs.py`, `test_fine_tuning_policies.py`, `test_calibration_and_source_readiness.py`, `test_source_aware_lifecycle.py` and applicable checkpoint/distributed/augmentation tests. Check whether each module uses pytest or unittest before choosing the invocation. Do not weaken tests merely to make them pass.

Validate notebook Python syntax and actual CLI arguments. Execute notebook orchestration with mocked training/inference and small real synthetic CSVs to verify path creation, two distinct configurations, best-checkpoint linkage, threshold loading, one inference per arm, stable comparisons and Drive persistence. This is necessary because earlier notebook errors were path collisions and stale checkpoint variables. No full dataset/checkpoint or cloud access is needed for these tests.

Check git diff for unintended changes. Do not rewrite training data, run real training, perform full external inference, commit, push, or delete artifacts.

Finish with `RGB_LABEL_SMOOTHING_IMPLEMENTATION_REPORT.md` containing:

- Files changed and what each change accomplishes.
- Exact tests run, outcomes and any untested GPU behavior.
- Exact default/opt-in compatibility rules for smoothing and score spaces.
- Link to the completed `COLAB_RGB_LABEL_SMOOTHING_AB.md`.
- Explicit statement whether implementation is ready for the user to push and run in Colab.
- Any material remaining defect. Do not claim training results or predicted accuracy.

The work is complete only when code, regression tests and runnable Colab commands are delivered. Do not return only another task list or request the old trained checkpoint as a prerequisite for implementation.
