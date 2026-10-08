> **Latest RGB review (2026-10-08):** See [RGB_REVIEW_FIXES.md](RGB_REVIEW_FIXES.md) and [RGB_READY_TO_RUN.md](RGB_READY_TO_RUN.md). Earlier completion/readiness claims below are historical; runtime verification remains pending.

# RGB implementation review — fixes required before training

Reviewed: 2026-10-08. Verdict: **not ready for a scientific pilot yet**.

Method: source/diff inspection and stdlib AST parsing of 59 changed/untracked Python files; no syntax errors found. No application imports, checkpoint loading, training, inference, or runtime tests were executed during this review. The previous implementation's claimed 55 passing tests were not independently verified here; even if those tests passed, they do not cover the integration failures below.

Preserve the useful work: nonblocking RGB transfers, configurable loaders, explicit fine-tuning options, deferred tests, tensor-based distributed validation, and fresh-validation scheduler ordering. Repair the integration gaps rather than replacing the project again.

P1 = fix before the next pilot or relying on its scientific results. P2 = correctness, compatibility or measurement issue that must be addressed before the associated workflow is advertised as supported. Line anchors refer to the reviewed working tree and may move after edits.

## R01 — P1: Best-checkpoint rewrite invalidates prediction/threshold provenance

Evidence: `training/hooks/checkpoint_hook.py:62` and `:141`.

`on_validation_end` saves `best.pth`, hashes it, and exports predictions/selection metadata tied to that hash. Later `on_epoch_end` saves `best.pth` again after the scheduler steps. With an active scheduler this changes serialized optimizer/scheduler state and therefore the file hash. The exported predictions still identify the first file. Calibration then produces a threshold for a different hash than the final checkpoint, which evaluation rejects.

Repair prompt:
```text
Defer the best-checkpoint save to a single finalized epoch-end operation after the
scheduler update. Keep the winning validation result pending until that save, then
hash the FINAL file and export all selection metadata/predictions against that hash.
Clear pending state on each epoch. Keep all-rank hook order safe and rank-zero I/O.
Do not weaken the evaluator's checkpoint-hash verification to hide this bug.
Add an integration regression with an active scheduler: save best -> calibrate saved
predictions -> verify threshold, metadata and final best.pth have the same hash.
```

## R02 — P1: DDP partial freezing turns the entire network into evaluation mode

Evidence: `models/wang2020_128/trainer.py:164`, especially iteration over `self.model.named_children()` in `train()`; DDP wrapping in `training/runtime/distributed_runtime.py`.

Before DDP, children are `conv1`, `layer4`, `fc`, etc. After DDP, the child is named `module`. For both `head_only` and `layer4_and_head`, the code sees `module` as a frozen stage and calls `module.eval()`. This disables head dropout as well as stage training behavior. It does not stop gradients, but makes single-GPU and two-GPU experiments use different training semantics. This directly affects planned R2/R3.

Repair prompt:
```text
Apply stage/BN policies to the unwrapped ResNet via get_model() or a safe module
unwrapper. Keep the trainable head in train mode; freeze only intended stages and
BN running statistics. Return self from train(mode) as nn.Module expects. Test
head/dropout/layer4 modes before and after DDP wrapping and after validation returns.
Use a real bounded DDP regression later, not only a plain model or MagicMock wrapper.
```

## R03 — P1: Legacy augmentation now silently overrides the user's settings

Evidence: `data/transforms/augmentations.py:29` and `data/datasets/rgb_dataset.py` constructor.

Every RGB dataset resolves the `legacy` recipe by overwriting `blur_prob`, `blur_sig`, JPEG, noise and downscale settings. Thus the original command `--blur_prob 0.5 --blur_sig 0.0,3.0` now trains without blur. Other explicit flags are also ignored. The advertised historical baseline is no longer historical. Recipe mutation happens during dataset creation, after initial configuration/experiment setup.

Repair prompt:
```text
Make legacy preserve the supplied/saved options exactly. Define explicit precedence
between a named preset and user overrides; reject ambiguous combinations or apply
documented explicit overrides. Resolve ONCE before saving the effective run config.
Dataset construction must not mutate the shared options namespace. Include a test
that the user's original blur-only command still means blur probability 0.5 and
sigma 0-3, plus a CLI/config/effective-metadata roundtrip for rgb_v1.
```

## R04 — P1: Both preflight entry points construct incomplete model options

Evidence: `tools/benchmark_training_throughput.py:87`, `tools/diagnose_rgb_parity.py:213`, and `models/base/base_model.py:33`.

The manually constructed namespaces omit `checkpoints_dir` and `name`, but BaseModel accesses both unconditionally. The benchmark and synthetic parity command therefore fail during model construction. The synthetic tool also passes `isTrain=False`, which invokes the legacy `load_networks()` path; it is supposed to work without a detector checkpoint.

Repair prompt:
```text
Build complete options using a shared supported configuration path. Construct the
synthetic model without downloading or loading a detector checkpoint, then explicitly
enter eval mode. Use isolated temporary/nonproduction paths if the constructor needs
them. Author subprocess-level tests that actually reach the bounded CPU synthetic
entry point and a minimal benchmark, rather than only importing helper functions.
Keep runtime execution deferred until authorized; do not mark unexecuted tests passed.
```

## R05 — P1: Parity diagnostic can pass without checking the real two pipelines

Evidence: `tools/diagnose_rgb_parity.py:346` and `:415`.

The real-checkpoint diagnostic creates both transforms by calling the same local helper with the same `val_opt`, then forwards both through the same model. This mostly measures repeatability of duplicated helper code, not validation-versus-evaluation parity. A difference from saved probabilities is recorded but does not contribute to the pass/fail decision. Saved checkpoint identity, labels and image-content identity are not validated before comparison.

Repair prompt:
```text
Exercise actual RGBDataset/validation construction and actual CheckpointLoader/evaluation
construction separately, preferably sharing production transform builders rather than
copying them into the tool. Compare tensors, logits, scores and modes on explicitly
matched IDs/content. Make requested saved-prediction mismatches fail with a reason,
including wrong checkpoint hash/labels; missing requested IDs/files must not silently
pass. Ensure max_samples bounds filesystem probing as well as inference. Add a
negative test: deliberately alter one path's normalization/crop and require failure.
```

## R06 — P1: The run guide is not executable end to end

Evidence: `RGB_PILOT_RUN_GUIDE.md:64`, `:238`, `:483`, and Cells 14–16.

- `torchrun ... -c "..."` is not the Python `-c` interface. Use a saved script as the torchrun entry point.
- Training commands omit mandatory `--dataroot`.
- Standalone evaluation omits the root required by evaluate.py's root/tsne-root check even though a manifest is provided.
- Fresh commands omit explicit `run_id`; ExperimentManager creates a suffixed directory. Checkpoints live under `checkpoints/`, and evaluation adds an architecture subdirectory. Later guide paths omit these parts.
- Resume omits the RGB architecture and R2 policies, so defaults can select Wang2020Raw/full fine-tuning and the wrong optimizer layout.
- The guide promises linear decay with `epochs_decay`, but uses no scheduler flag; `lr_policy` defaults to `none`, and scheduler_factory does not implement linear decay.
- The guide assumes prepared manifest files without providing a complete creation/audit/calibration-partition workflow.

Repair prompt:
```text
Rewrite the guide as real complete notebook cells with %%bash where appropriate,
set -euo pipefail, exported explicit roots, prepared paths and stable run IDs.
Provide a saved DDP sanity script. Derive all output/checkpoint/prediction paths from
actual ExperimentManager and evaluate.py behavior. Resume with the original complete
configuration (or implement verified config restoration). Document the scheduler that
actually runs. Include preparation/gate prerequisites and independent calibration
inference if claimed. Validate required arguments and values, not only flag spellings.
No stale claims of 100% command compliance. Do not hardcode a universal Kaggle CPU count.
```

## R07 — P1: Independent-calibration artifacts are rejected by evaluation

Evidence: `evaluation/generalization.py:251`, `:264`; `evaluation/evaluator.py:143`; `analyze_predictions.py::load_threshold`.

Calibration now accepts `dev_calibration`, but both threshold consumers still only allow `dev`, `val`, `external_dev`. The intended independent-calibration path therefore generates an artifact its own evaluator refuses. Conversely, merely naming a split `dev_calibration` is enough to label it independent: no comparison against training or checkpoint-selection membership is made. The guide still calibrates selection-dev predictions, not separate calibration predictions.

Repair prompt:
```text
Centralize calibration split policy across producer and consumers. Preserve explicitly
allowed external DEVELOPMENT workflows while rejecting final/test splits. Verify
calibration-vs-training/selection sample/group/hash separation using bound cohort
evidence before claiming independence; otherwise say unverified or reused dev. A name
alone is not proof. Provide a command to infer the separate calibration cohort with the
selected checkpoint, then calibrate those predictions. Validate prediction precision
provenance instead of letting an arbitrary --eval_precision string certify it. Add a
producer-to-evaluator regression for dev_calibration and a relabeled-overlap negative test.
```

## R08 — P1: Source-aware monitor does not enforce held-out-source readiness

Evidence: `training/validator.py:147–206`, `train.py` preflight.

If eligible_sources is omitted, every source—including unknown or the aggregate diffgan collection—is eligible. Missing requested sources are silently ignored, and there is no monitor-level requirement that eligible development sources be verified and absent from training. A source_macro_auc can therefore reproduce the original same-source problem while appearing to satisfy the new protocol.

Repair prompt:
```text
Add a scientific source-held-out readiness check before training: verify declared source
provenance, explicit eligible evaluation domains, their presence/class coverage, and
train/selection/calibration/final roles. Fail on missing eligible sources rather than
silently averaging the survivors. Require explicit engineering-diagnostic mode for
unknown/aggregate source metadata. Keep aggregate AUC for historical reproduction.
Add fixtures where diffgan alone, unknown, train/dev source overlap, a missing requested
source, and a one-class eligible source are rejected or explicitly handled by policy.
```

## R09 — P2: Old RGB resume and local-backbone portability regress

Evidence: `models/wang2020_128/trainer.py:17`, `:117`; `training/checkpoint_manager.py::resume`; `evaluation/checkpoint_loader.py:259`.

Old RGB optimizers have one parameter group; the new full-policy optimizer always separates head/backbone groups. Loading the old optimizer state then fails with a parameter-group-count mismatch. Also, saved `backbone_weights` is restored during evaluation, and the model passes it to resnet50 even when pretrained=False or continuing training. A self-contained checkpoint trained with a local Kaggle backbone file can fail on Windows because that old path does not exist.

Repair prompt:
```text
Version optimizer/policy metadata. Reconstruct the original optimizer grouping for legacy
resume or provide an explicit validated migration; never silently reset optimizer state.
For full-checkpoint restoration, skip initial backbone file loading/downloads and load
the checkpoint's own model weights. Enforce new head/fine-tuning/BN/augmentation policies
on resume, and preserve historical dropout defaults when old rgb_dropout is absent.
Test old one-group resume and evaluation after removing the original initialization file.
```

## R10 — P2: Standalone --batch_size can be overridden by a saved validation batch

Evidence: `data/loaders/dataloader_factory.py:34`; `evaluation/evaluator.py::_build_dataloader`; checkpoint option restoration.

Standalone evaluation sets `opt.batch_size` from its CLI, but the loader prefers `opt.val_batch_size` from the training checkpoint. A checkpoint saved with val_batch_size=64 will therefore ignore a local `--batch_size 2` OOM workaround.

Repair prompt:
```text
Make standalone evaluation explicitly override the loader's effective evaluation batch
size and worker/precision controls. Training-validation defaults must not override the
evaluation command. Print the actual effective values. Test a checkpoint config with
val_batch_size=64 and evaluate batch_size=2, asserting the created loader uses 2.
```

## R11 — P1 for performance claims: GPU timing does not measure GPU execution

Evidence: `tools/benchmark_training_throughput.py:228`, `:237`, `:293`; no PhaseProfiler integration found in train.py/BaseTrainer/Validator.

H2D/compute timings use host monotonic clocks around asynchronous CUDA launches. These are enqueue times, not GPU execution times; the end-of-window synchronize is outside the per-phase totals. nvidia-smi is sampled after work completes, so low utilization then says little about duty cycle. The benchmark disables augmentation and does not reproduce the planned fine-tuning recipe, while production training is not instrumented with the new profiler.

Other benchmark correctness issues: warmup optimizer gradients are not cleared after its final step before measured accumulation; the final partial accumulation window always divides by configured accumulation steps; validation's per-rank loop can execute unequal numbers of DDP forwards without unwrapping (buffer-broadcast hang risk); global validation count is estimated as rank-local count times world size.

Repair prompt:
```text
Use sampled CUDA events and deferred synchronization for GPU timing; name CPU enqueue
time accurately if retained. Sample utilization during measured work. Wire production
phase timing around real startup/train/validation/checkpoint paths with bounded overhead.
Benchmark the actual configured recipe/policy with the production train-step logic or
an equivalence-tested shared helper. Clear warmup gradients; weight accumulation tails.
Use the safe unpadded/unwrapped validation path, exact sample reductions and true global
sample bounds. Test unequal/empty rank validation. Report unsupported measurements as
unavailable, not zero. Do not claim two-GPU efficiency before target-hardware measurements.
```

## R12 — P2: AdamW is parsed and implemented but rejected by config validation

Evidence: `train.py --optim`, `config/validator.py:207`, unchanged `config/types.py::OptimizerType`.

The parser/model accept adamw, but the validator enum only accepts adam and sgd. Actual `train.py --optim adamw` fails before constructing the model; direct model tests miss this.

Repair prompt:
```text
Wire AdamW through enums, validation, conversion, persistence and the actual model
optimizer. Reject unsupported architecture/optimizer combinations explicitly. Add an
end-to-end configuration parse/validate/roundtrip test for every documented optimizer,
rather than testing the model constructor alone. Keep scheduler support equally explicit.
```

## R13 — P2: Representative dev selection is not connected-component/source aware

Evidence: `data/manifest.py:437` and `prepare_dataset.py:678`.

The subset helper groups only by literal group_id, ignoring shared source_video_id, identity_id, original_id, hashes and multi-token relationships used by the existing preparation graph. It also has no source quotas and shuffles insertion-order groups, so reordered input rows can change membership. This does not itself move unused rows into training, but it fails the advertised connected-group/representative contract and is unsafe as the basis for separate selection/calibration partitions.

Repair prompt:
```text
Reuse the established connected-component grouping and stable seeded ordering. Add
declared source/class coverage constraints and honest indivisible-group shortfall reports.
Keep the parent cohort immutable. Persist membership/provenance and document/reuse the
correct audit gate for the resulting subset; do not just write a CSV that training cannot
verify. Test reordered input, linked rows with different group_id values, mixed-class
components, rare sources, shortfalls and immutable parent partitions.
```

## R14 — P2: New update/early-stop state is not resumable as advertised

Evidence: `training/base_trainer.py` accumulation boundary; `training/runtime/amp.py::amp_step`; `training/hooks/early_stopping_hook.py`; checkpoint manager.

AMP counts attempted/successful updates, but train_step ignores amp_step's success return and global_step still increments at every boundary. New counters and early-stop best_score/wait_count are not persisted/restored. Resuming resets patience and these counters, so an interrupted run is not equivalent to uninterrupted training under the new policy. Rank-zero early-stop metric errors can also occur before peers receive a broadcast decision.

Repair prompt:
```text
Define attempted versus successful update counters and wire their semantics consistently
into logging/scheduling/checkpoints. Persist and restore early-stop best_score, wait_count,
policy and update counters; validate policy compatibility. Broadcast validation/status
errors before ranks diverge. Add overflow-skip and stop/resume equivalence regressions.
Keep legacy global_step semantics versioned if changing them would break old checkpoints.
```

## Completion/status corrections

`RGB_READY_TO_RUN.md` must not claim all commands and runtime contracts are verified while the above issues remain. Preserve any real test results, but identify exactly what ran and what was mocked. Several tests are labeled NOT RUN while the status document says they passed; reconcile this with actual evidence rather than replacing all statuses with a blanket success claim.

Additional unfinished items to retain on the checklist: rank-coordinated pretrained initialization, audited preparation/staging end-to-end commands, bounded/run-scoped caches rather than unlimited global manifest dictionaries, and real inference precision plumbing. A profiler class, a CLI option, or a Markdown statement is not completion of its production integration.

## How to execute these fixes

Use the shared code-only instructions from RGB_GENERALIZATION_GPU_HANDOFF.md. Give the implementing LLM R01–R05 first, then R06–R08, then R09–R14. Do not train between repair prompts. Update a repair checklist with each change and tests authored, retaining runtime tests as NOT RUN unless actually authorized and executed.

Final user-run sequence after another review: real entry-point smoke checks -> legacy/new checkpoint roundtrip -> DDP freeze/coverage/timing checks -> verified data protocol -> bounded pilot. Do not weaken hash or split gates to make broken commands succeed.
