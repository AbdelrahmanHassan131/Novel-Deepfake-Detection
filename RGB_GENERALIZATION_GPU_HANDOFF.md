# RGB generalization and GPU efficiency: implementation handoff

Date: 2026-10-07. Status: source review and implementation plan only. No training, inference, model imports, or GPU benchmarks were run for this review.

## What this plan is trying to achieve

Make the existing RGB ResNet-50 baseline a credible, measurable starting point for cross-source detection. First establish that preprocessing and evaluation reproduce the training protocol; then improve data diversity, source-held-out model selection, and fine-tuning. Separately measure and remove input/CPU/communication bottlenecks so one or two GPUs are used effectively.

Neither 85% unseen accuracy nor 100% GPU utilization can be guaranteed. Do not use utilization as a substitute for images/second, epoch wall time, or generalization. No amount of source-code optimization substitutes for missing source provenance or appropriate real-image coverage.

## Evidence from this project

All external models below evaluated 60,000 images: 30,000 real and 30,000 fake. Each threshold was selected using its own saved development predictions.

| Model | External accuracy | AUC | Real recall | Fake recall |
| --- | ---: | ---: | ---: | ---: |
| RGB Wang2020_128 | 50.37% | 0.82549 | 0.75% | 99.99% |
| Wavelet | 52.09% | 0.75070 | 4.40% | 99.78% |
| Token fusion | 50.22% | 0.81769 | 0.44% | 100.00% |

The RGB development AUC was 0.9963. External fusion median fake scores were 0.999219 for real images and 0.999263 for fake images. Its accuracy at the conventional 0.5 threshold was still only 50.38%. Therefore, lowering/raising the original development threshold slightly is not an adequate explanation or remedy. The root cause is not established: candidates include training-source shortcuts, source shift, preprocessing mismatch, and fine-tuning behavior. Do not label any hypothesis as proven.

Training cohort after recovery: 98,590 images (49,741 real / 48,849 fake). Development cohort: 209,871 (52,426 real / 157,445 fake). The available audit reports one source name, `diffgan`, and no video/identity/original annotations. A nonempty generator field or unique per-image group is not proof of true generator provenance or independence.

Observed RGB training time was about 388–399 seconds per epoch **excluding validation and startup**. Fusion took roughly 2,284–2,623 seconds per training epoch and used CPU wavelets. Do not assume RGB has the same bottleneck as fusion. No measured GPU duty-cycle or data-wait trace is available yet.

### Confirmed source findings

| Finding | Source anchor | Implication |
| --- | --- | --- |
| All backbone weights train at the same LR as the new head; no RGB fine-tuning/BN policy | `models/wang2020_128/trainer.py::__init__` | Add controlled experiments preserving pretrained features; improvement is a hypothesis. |
| RGB head dropout is hardcoded to 0.5 | Same constructor | Existing generic `--dropout` does not configure this head. New option must preserve old checkpoints. |
| RGB `.to(device)` calls omit `non_blocking` | RGB trainers' `set_input` | Pinned memory is already enabled, but transfer handling can be improved. |
| Every microbatch returns `loss.item()` | `training/base_trainer.py::train_step` | A CPU synchronization occurs each microbatch. |
| Workers default to zero; loader prefetch is fixed at 2, persistence tied to workers, pinning hardcoded | `train.py::parse_args`, `data/loaders/dataloader_factory.py` | Runtime controls need proper end-to-end wiring and measurement. |
| DDP and accumulation `no_sync()` already exist | `training/runtime/distributed_runtime.py`, `training/base_trainer.py` | Preserve them; do not replace with DataParallel or synchronize every microbatch. |
| Validation inherits training batch size and runs FP32 forward; predictions move to CPU batch by batch | `train.py` val loader, `training/validator.py` | Validation on >2x as many images as training can be expensive. Add separate batch/precision controls. |
| DDP validation uses three `all_gather_object` calls and computes metrics on each rank | `training/validator.py` | Tensor gathers and rank-zero metric computation are candidates after timing evidence. |
| Training preflight reads/audits train+dev on every rank; datasets read manifests again | `train.py` preflight, `BaseDataset`, `data/manifest.py` | Repeated parsing/path probes can delay startup. Retain audit guarantees while avoiding redundant work. |
| `loss_freq` takes precedence over `log_freq` | `training/trainer.py::_setup_hooks` | Explains why `--log_freq 10` did not necessarily produce frequent logs. |
| Only aggregate AUC/accuracy/balanced accuracy guide selection | Validator + checkpoint hook | Same-source aggregate AUC does not test cross-source transfer. |
| Prior RGB run enabled blur but JPEG/noise/downscale were disabled | Saved options and earlier command | Test realistic augmentation; do not assume implemented augmentations were enabled. |
| Resize→augmentation→crop→flip is the current order; normalization is ImageNet | `data/datasets/rgb_dataset.py` | Preserve existing evaluation semantics; new recipes need explicit versions. |
| `jpg_qual` is a discrete list, not an inclusive integer range | `data/transforms/augmentations.py::sample_discrete` | `50,95` means two qualities. Document or add a separately named range option. |
| Main calls `fit(num_epochs=opt_clean.niter)` while cosine uses `niter+niter_decay` | `train.py`, scheduler factory | Nonzero decay epochs can be omitted from actual training. Fix total-epoch semantics. |
| Plateau steps before current validation and stores accuracy regardless of monitor | Trainer hook ordering, `scheduler_hook.py` | Fix stale/wrong metric use before recommending plateau. |
| `earlystop_epoch` is carried in configuration but no active early-stop loop was found | Config compatibility + training engine | Do not promise early stopping without implementing and testing it. |
| Local ResNet weights use `strict=False` without checking missing backbone keys | `models/shared/resnet.py::resnet50` | Reject incompatible/incomplete backbone initialization rather than silently accepting it. |
| Windows transform lambdas have already been replaced | Dataset classes + `data/transforms/identity.py` | Preserve the fix and run the authored spawn regression later. |

Existing good behavior to retain: canonical real=0/fake=1, deterministic evaluation resize/crop, disabled evaluation augmentation, checkpoint-bound calibration, connected-group/hash leakage protections, quarantined contradictions, fresh-run collision protection, sample-weighted accumulation tails, and unpadded distributed evaluation.

## How to give this to the implementing LLM

Paste the following shared instructions with **one numbered prompt at a time**. Execute prompts in order. Ask the LLM to update `RGB_IMPLEMENTATION_STATUS.md` after each task with changed files, checks actually run, deferred checks, and unresolved dependencies. Do not let a missing dataset annotation block independent code tasks.

### Shared instructions for every prompt

```text
Work in this repository and read RGB_GENERALIZATION_GPU_HANDOFF.md first.
Implement only the requested task. Inspect the current code and preserve unrelated
working-tree changes. Do not redesign fusion or wavelets for the RGB pilot.

This is a code-and-documentation phase: do not run training, inference, checkpoint
deserialization, full dataset scans, GPU benchmarks, dependency installs, or runtime
application tests. You may inspect source and existing small report/config artifacts,
write tests and user-run tools, and perform stdlib AST/JSON and PowerShell syntax checks.
Mark runtime checks NOT RUN. Do not demand a previously trained detector to implement
fresh training. ImageNet backbone initialization may be a later user-run dependency.

Never promise a target accuracy or speedup. Do not disable audits, change label
direction, fit thresholds on external_test/final_test, invent source/group metadata,
or silently change old checkpoint preprocessing. All new options must survive CLI ->
structured config -> options -> saved checkpoint -> evaluation and resume as appropriate.
Keep future diagnostic/training commands separate from commands safe to execute now.
Finish with changed files, static checks, deferred tests, and the next numbered task.
```

## Prompt 1 — Establish the RGB correctness and preprocessing contract

```text
Inspect the entire Wang2020_128 path: parser/config conversion, pretrained loading,
RGBDataset, resize/augmentations, set_input, model.forward, Validator, checkpoint
metadata, CheckpointLoader and InferenceRunner. Create a compact contract describing
decode/RGB conversion, pixel scale, interpolation, resize, crop, normalization,
train/eval modes, score direction and checkpoint identity. Preserve legacy behavior.

Implement a user-run RGB diagnostic that accepts a checkpoint and explicitly supplied
small manifest, sample IDs and optional saved predictions. It must compare the SAME
image bytes/IDs between the validation and standalone evaluation paths in FP32:
transformed tensors, logits, probabilities, class mapping, and module eval/BN modes.
Report discrepancies instead of auto-correcting labels. Allow a CPU-only synthetic
test of transforms without requiring a trained checkpoint. Never scan 1M images by
default, nor infer relocated paths by basename. Different files with identical names
must not be matched. A saved dev-prediction subset is useful on Kaggle where those
images remain available, but missing images must not block implementation.

Add optional logit export to predictions, preserving current CSV consumers, and
per-class score/logit quantiles, saturation counts, and real/fake recall to reports.
If report schemas change, version them. Keep score interpretation unchanged.

Add deferred tests for RGB mode, grayscale/RGBA conversion, crop/no-crop/no-resize,
deterministic eval transforms, train-only augmentation, metadata restoration and
tensor parity. Preserve Windows serialization fixes. Deliver RGB_INPUT_CONTRACT.md
and diagnostic instructions. Do not run model/image diagnostics now.
```

Acceptance: old checkpoints reconstruct the old pipeline; identical inputs follow equivalent validation/evaluation processing; no claimed runtime parity until measured.

## Prompt 2 — Source provenance and a reusable RGB pilot protocol

```text
Extend the existing manifest/preparation tools, not a parallel incompatible format.
Add a metadata-only source coverage report: real/fake counts by dataset_source,
verified generator, parent/source collection and partition, annotation coverage,
independent group coverage, fixed-partition conflicts and one-class-only sources.
Distinguish collection names like diffgan from verified original sources. Do not
treat independent:<path hash> as evidence of independent people/videos.

Add a user-run bounded image-profile tool (seeded sample, default <=1000 total images)
that reports format, dimensions, aspect ratio, color mode, file size and basic quality
statistics per class/source, plus decode failures. It must not infer demographic or
source labels from appearance. It should help detect class/source packaging shortcuts.
Write findings as hypotheses, not automated proof of label errors.

Provide a user-filled provenance mapping template and explicit source-path remapping.
Reuse the verified 98,590-image manifest for historical reproduction. Make a NEW pilot
manifest for changed source composition; never overwrite existing manifests/gates.
Reuse safe hash-cache evidence, quarantine contradictions transparently, and do not
rerun full image hashing unnecessarily on unchanged immutable inputs.

Plan train, source-held-out development-selection, development-calibration, and final
test roles. Keep linked original/fake/video/identity/hash components together. Preserve
existing fixed assignments; require explicit new-cohort policy for changes. Reuse
existing holdout code where possible and ensure selected split roles reflect whether
they are for model selection or final reporting. Do not bypass PROTECTED_SPLITS.

Allow a deterministic, group-aware representative dev cohort around 10K-20K images
for frequent validation, with source/class counts and digest. This is an explicit
new selection cohort, never a silent reduction of the historical 209,871-image dev
set. Preserve full dev for an occasional/final report; do not move its unused rows
into training. Handle indivisible groups/shortfalls honestly.

Use source/class quotas for a ~100K pilot without copying images. If provenance is
insufficient, emit an actionable missing-metadata report and keep code work moving;
mark source-held-out experiment readiness blocked, not all implementation blocked.
Document that the already inspected F:\val to be deleted 2 is external development
evidence; it cannot become an untouched final test by renaming or random resplitting.
```

Acceptance: selection is reproducible and auditable, source-held-out claims require verified sources, source imbalance is visible, final-test data cannot enter selection/calibration/training.

## Prompt 3 — Measure where time goes before optimizing it

```text
Instrument startup, manifest validation, loader construction, pretrained initialization,
first-batch latency, steady data wait, host-to-device transfer, forward/backward,
optimizer update, validation, metric aggregation and checkpoint/export time.
Use CPU monotonic timing for host phases and sampled CUDA events for GPU work.
Do not call cuda.synchronize() every step; bound synchronization to profiling windows.
Avoid double-counting overlapping phases. Record rank-local and overall wall times,
global images/sec (total samples divided by slowest rank wall time), per-rank batch
size, effective global batch, CPU count, world size and peak GPU memory.

Fix log_freq vs loss_freq precedence explicitly, with backward-compatible documented
semantics. Log microbatches and optimizer updates distinctly. All rank collectives
must execute in the same order, even though only rank 0 prints. Remove silent
collective-error suppression; surface distributed failures cleanly.

Write a bounded user-run benchmark mode/tool: explicit opt-in, warmup ~20 and measured
~100 training microbatches plus a bounded validation sample. Never modifies production
checkpoints or counts as scientific training. Restore no production state from it.
Save a machine-readable report; measure data loading and model compute separately.
An optional nvidia-smi sampling helper should report each GPU, not only GPU 0.
Exit after the requested bound on every rank. Do not execute benchmarks now.
```

Acceptance: GPU idle during startup is distinguished from loader starvation, communication waits, validation and checkpoint I/O. A performance recommendation must cite measurements once the user runs the tool.

## Prompt 4 — Improve RGB data feeding and eliminate repeated startup work

```text
Expose and validate train/validation num_workers, prefetch_factor, persistent_workers,
pin_memory and validation batch size through all config conversion layers. State that
workers and batches are per rank. Honor zero workers without invalid prefetch/persistence
arguments. Preserve these settings when DistributedRuntime reconstructs loaders.
Keep Windows spawn compatibility and CPU-only DataLoader workers. Limit OpenCV/native
thread oversubscription via explicit worker settings, not indiscriminate CPU disabling.

Use non_blocking transfers for RGB tensors and labels when appropriate with pinned CPU
memory/CUDA, preserving CPU behavior and numerical preprocessing. Retain existing AMP
and no_sync accumulation. Expose optional channels_last for RGB with matching model
and input layouts; keep it opt-in until benchmarked and provide a normal-layout fallback.

Reduce redundant manifest parsing/path stat work. Separate metadata parsing from optional
file probing; note current absolute-path parsing probes even with check_files=False.
Use validated immutable in-memory records per process or a digest-bound prepared index,
not an unbounded global cache. Reuse them when constructing train/validation datasets.
If coordinating preflight on rank 0, initialize the runtime/device appropriately first,
broadcast status/errors before anyone proceeds, account for root/digest mismatches, and
avoid deadlocks if rank 0 fails. Do not broadcast giant Python records unnecessarily.
Keep authoritative gate checks and fail closed on changed manifests/protocols.

Optionally support staging ONLY selected train/dev files to local Kaggle working storage,
preserving relative paths and original bytes. Check free space, stage once before ranks
start, retain content identities and a relocation manifest with appropriate verification.
Do not duplicate the full 1M pool or recompress images as a speed optimization.
Make staging optional because storage/CPU budgets may make it counterproductive.

Add deferred tests for config roundtrip, DDP loader preservation, zero-worker mode,
spawn, manifest-cache invalidation, duplicate basenames, and rank-safe error propagation.
```

Acceptance: no CPU transforms are moved blindly to GPU with changed semantics; pinned/nonblocking alone is not claimed to guarantee overlap; content/audit identity survives relocation.

## Prompt 5 — Faster training/validation while preserving distributed correctness

```text
Accumulate detached loss sums/sample counts on device rather than calling loss.item()
every microbatch. Materialize only at logging/epoch boundaries; update logger contracts
together and preserve readable first-batch progress. Do not retain computation graphs.
Correct epoch reporting to be sample-weighted, including tails. Preserve accumulation
window weighting and no_sync semantics. Account for AMP-skipped optimizer updates in
step/scheduler counters; record attempted versus successful updates if necessary.

Modernize AMP using supported torch.amp APIs with compatibility handling for the installed
versions. Use FP16 on T4; allow BF16 only when runtime-supported and explicitly recorded.
Do not assume T4 supports native BF16. Add explicit validation/evaluation precision
controls; keep FP32 available as the reference. Use the larger separate validation batch.
Cast exported logits/probabilities appropriately and never fit a threshold under one
precision then silently use another materially different pipeline.

Preserve unpadded EvaluationSampler, buffer synchronization before unwrapped evaluation,
ordered sample-ID alignment and exact coverage. If replacing all_gather_object, gather
variable-length tensors with lengths, temporary padding then trim; handle ranks with
zero samples and CPU/Gloo. Compute expensive metrics on rank 0 and broadcast only what
hooks require, while maintaining all-rank collective order. Chunk when necessary.
Do not drop validation samples for speed or apply training augmentations at evaluation.

Cache immutable manifest provenance digests within a run with invalidation checks; avoid
rehashing the same manifest twice per checkpoint when train/dev reference it. Keep best
and last checkpoints; permit save_epoch_freq=0 to avoid redundant numbered copies.
Write only one authoritative best-linked dev prediction CSV and document any compatibility
alias. Preserve hash binding, optimizer/scaler/RNG metadata and resume behavior.

Author deferred comparisons of old/new FP32 metrics, mixed-precision tolerances, uneven
DDP validation (including empty rank), sample coverage, accumulated gradients/tails,
checkpoint/resume and memory growth. Speed changes must not change evaluated cohorts.
```

Acceptance: a bounded user benchmark proves the benefit; no guaranteed utilization percentage; any numerically consequential precision change is recorded and revalidated.

## Prompt 6 — Controlled RGB fine-tuning instead of an untracked architecture rewrite

```text
Keep Wang2020_128 and all old checkpoints working. Add explicit RGB-only configuration
for head type (legacy 128D MLP or direct linear classifier), RGB head dropout (legacy
default 0.5), fine-tuning policy (head_only, layer4_and_head, full), backbone LR multiplier,
and BN running-stat policy (train or frozen). New options must be saved and restored.
Prefer a clearly versioned RGB model/config if changing the existing registry contract
would break expert/fusion compatibility. A direct-linear checkpoint must not masquerade
as a 128D fusion expert.

Set requires_grad before constructing DDP/optimizer. Exclude frozen parameters from the
optimizer; use separate backbone/head parameter groups with explicit LRs. Preserve
frozen backbone/BN eval mode when BaseTrainer calls model.train() each epoch and after
validation restores mode. Specify whether BN affine parameters are frozen separately
from running statistics. Do not use no_grad around trainable layer4/head accidentally.
Avoid progressive unfreezing in the first implementation to simplify DDP and resume.

Add AdamW if used by the new recipe, wiring the ACTUAL model optimizer path: changing
training/optimizer_factory.py alone is insufficient because Trainer reuses model.optimizer.
Give decay treatment for bias/normalization an explicit policy. Retain Adam/SGD compatibility.

Validate pretrained backbone keys/shapes and reject incomplete loads except explicitly
allowed classifier mismatches. Record initialization source/checksum. Resolve initial
downloads once per node before workers build models, propagate errors, and support a
local weights path. On evaluation/full-checkpoint resume, do not reload obsolete local
backbone paths or download weights unnecessarily.

Tests to author: each policy's trainable names/gradients, BN buffers over train/eval
transitions, dropout config, optimizer groups/LR/decay, compatible legacy strict loading,
new-format reconstruction, mismatched initialization rejection, DDP wrapping and resume.
Do not require a trained detector checkpoint to implement fresh ImageNet initialization.
```

Acceptance: feature-preserving fine-tuning is an experiment, not a guaranteed cure. Initial RGB work does not require a new CLIP/ViT implementation or fusion retraining.

## Prompt 7 — Versioned, class-independent RGB augmentation experiments

```text
Retain the legacy preprocessing recipe. Add an explicit versioned RGB recipe that uses
the same augmentation probabilities and distributions for real and fake. Start with the
existing blur/JPEG operations rather than adding large augmentation dependencies.
Expose interpolation selection properly (rz_interp exists internally but was not a train
CLI option), and make JPEG choices/ranges unambiguous. Validate all ranges/probabilities,
decode success, color order and dtype. Log effective probabilities at startup.

Provide a moderate initial experimental preset: blur probability 0.5, sigma 0-3; JPEG
probability 0.5 with explicit choices 50,60,70,80,90,95. Keep noise/downscale disabled in
this first preset so it tests a clear hypothesis against the old blur-only recipe.
These are candidate settings, not proven best values. Add a separate optional modest
downscale recipe later, with operation order recorded; do not silently add it to baseline.

Provide an explicit crop policy option if needed by the input audit: legacy resize-short-
side then random/center crop remains default for old models. Native-resolution patch or
random-resized-crop strategies must be separate ablations, handle too-small images, and
have a deterministic evaluation counterpart. Do not add face cropping only to one class
or external dataset, and do not destroy forensic detail through arbitrary heavy transforms.

Use spawn-safe callables. Tests should verify deterministic evaluation, class independence,
size/range/color contracts and metadata restoration. CPU augmentation optimizations must
be benchmarked and checked against the intended operations before becoming defaults.
```

## Prompt 8 — Source-aware model selection and correct training lifecycle

```text
Extend validation records/results with source-level counts, AUC where both labels exist,
balanced accuracy, real/fake recalls and score quantiles. For single-class sources report
the defined class recall; mark AUC/BA undefined, never zero or silently invented. Preserve
ordered records through DDP. Do not let large fake-heavy sources dominate all summaries.

Add an explicit source-macro AUC selection option on verified held-out development
sources, with a declared eligible source list. Require both labels for source AUC or
explicit predeclared real+fake evaluation-domain mappings; do not fabricate such pairings.
Report worst-source/class recall at fixed 0.5 alongside it. Selection is on development
only. If source metadata is missing, refuse a source-aware monitor with a useful error.
Keep historical aggregate monitors for reproduction. Document that AUC alone does not
establish operating-threshold transfer; final calibration and recall gates still matter.

Fix total epochs/decay semantics in train.py. Fix plateau to consume the current selected
validation metric, only when new validation occurred. Ensure best/last checkpoints capture
consistent optimizer/scheduler state for resume; do not save best midway through an
inconsistent epoch state. Implement optional early stopping on the declared development
metric with minimum epochs, patience/min_delta and rank-broadcast stop decisions.
Do not allow one rank to exit while peers enter collectives. Validate missing metrics.

Add a separate development-calibration manifest role/tool integration. Persist split,
cohort/checkpoint hashes, threshold criterion and eval precision. Do not tune operating
thresholds on the 60K external folder or final tests. Existing same-dev selection and
calibration may be reproduced but must be labeled as such, not claimed independent.

Author tests for source weighting, single-class groups, protected splits, plateau ordering,
total epochs, early-stop synchronization, config roundtrips and resume-equivalent counters.
```

## Prompt 9 — Reproducible one-GPU/two-GPU commands and bounded tuning

```text
Create RGB_PILOT_RUN_GUIDE.md with complete copy/paste notebook commands for Kaggle 2xT4,
one T4 and one L4, plus Windows evaluation. Use only implemented CLI flags. Include
fresh setup, prepared-manifest reuse, bounded speed benchmark, train, resume, calibration,
external evaluation and output preservation. Do not regenerate UUID paths when reusing
expert/threshold outputs; keep stable explicit paths and separate preparation/training IDs.

For two GPUs use torchrun --standalone --nproc_per_node=2, LOCAL_RANK placement and DDP.
Include checks reporting rank/device/world-size and observed work on BOTH devices. A
plain python command with gpu_ids=0,1 must not imply two-GPU training. Keep RGB wavelets
disabled. Do not enable SyncBatchNorm merely to raise utilization.

Benchmark candidate per-GPU batch sizes 16 and 32 first, then 64 only if memory allows.
Compare a global effective batch of 64 via 16x2GPUsx2accum versus 32x2GPUsx1accum;
accumulation does not fill GPU memory like a larger actual batch and BN sees microbatches.
Larger global batches are separate optimization experiments, not automatically equivalent.
Candidate workers: 0/2/4 per rank subject to available CPU/RAM; do not exhaust small Kaggle
CPU allocations. Try prefetch 2 then 4 if data wait remains high and host memory permits.
Try larger validation batches independently. Choose channels_last/cuDNN benchmark only
from measured gains and respect deterministic mode. No blanket TF32 claim on T4.

Report warmup-excluded throughput, total epoch including validation/checkpoints, both GPUs'
memory/activity, data wait and 1-vs-2 GPU speedup at stated batch semantics. Do not promise
2x scaling or 100% utilization. Benchmarking is user-run, bounded and not silently followed
by full training. If OOM occurs, rerun from a clean process with a smaller batch; do not
silently alter production batch/precision mid-run or let one rank continue after peer OOM.

Start scientific training with the matrix below. Generate configuration artifacts and
commands, not an automatic sweep. Include all prerequisites and mark unrun commands.
```

### Initial experiment matrix (hypotheses, not optimal settings)

All experiments use the same new verified pilot and held-out development/calibration partitions, seed 42 first, fixed evaluation pipeline, and unique names. If new source coverage is unavailable, label runs as engineering diagnostics, not validated source generalization.

| Run | Backbone/head policy | Augmentation | Purpose |
| --- | --- | --- | --- |
| R0 | Historical full fine-tune, legacy 128D head, historical Adam/LR | Historical blur-only | Baseline on the new cohort; compare data/protocol effect without architecture changes. |
| R1 | Same as R0 | Blur + JPEG preset | Isolate augmentation effect. |
| R2 | Same head; layer4+head trainable, frozen BN stats; lower backbone LR | Same as R1 | Test preservation of pretrained features. |
| R3, conditional | Frozen ImageNet backbone, linear head | Same as R1 | Low-cost diagnostic if R2 still fails; separate model identity. |

Candidate R2 rates: head 1e-4, trainable backbone 1e-5. Keep optimizer constant versus R1 initially; test AdamW/weight decay only as a later separate ablation if needed. Candidate budget: up to 5 epochs with recorded validation each epoch; checkpoint choice is development-based. These values require user-run validation, not blind expansion to dozens of runs. Historical 2 epochs may be insufficient to assess a changed fine-tuning policy, but more epochs alone are not a remedy.

Do not change source mixture, crop policy, head, optimizer, augmentation and LR all at once and attribute an improvement to one change. After selecting a recipe, confirm it across at least 3 seeds and additional untouched sources before stronger claims. Consider a real frozen pretrained vision-language baseline only later if the RGB ResNet experiments remain inadequate; existing adapter placeholders are not evidence of a working baseline.

## Prompt 10 — Independent final code review and execution checklist

```text
Review all changes from prompts 1-9 against this handoff. Do not start another feature
cycle. Check old-checkpoint compatibility, label mapping, evaluation determinism,
threshold provenance, source-held-out gates, DDP collective order, accumulation/tails,
frozen parameters/BN, optimizer/scheduler restore, Windows workers and all CLI/config
roundtrips. Verify every guide command against parser definitions statically.

Create RGB_READY_TO_RUN.md with: implemented vs incomplete items, static checks actually
performed, authored runtime tests NOT RUN, required user metadata, and exact sequential
commands the user will run. Missing provenance must block scientific claims without
preventing source-code completion. Do not describe static checks as GPU validation.

Order user execution: dependency/device check -> targeted CPU/synthetic tests -> bounded
1-GPU smoke -> bounded 2-GPU DDP correctness/throughput tests -> preprocessing parity ->
data-source readiness -> selected RGB pilot -> independent dev calibration -> external
evaluation. No full training before those bounded checks pass. Preserve artifacts and
provide explicit failures instead of fallback guessing. No autonomous experiment sweep.

Review acceptance criteria: real recall and balanced accuracy improve on development
sources without sacrificing fake recall unacceptably; source-macro/worst-source results
are visible; operating thresholds are fixed before final tests; both GPUs contribute
without duplicate final evaluation samples; throughput improvements are measured with
comparable settings. Report all failed hypotheses. Never mark generalization solved
because training accuracy or source-overlapping development AUC is high.
```

## Research and implementation references

These support candidate experiments, not a forecast for this dataset:

- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html): use measured data-loader, memory-transfer, AMP and layout optimizations. It also discusses avoiding synchronization and unnecessary DDP synchronization during accumulation.
- [Wang et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html) and [official implementation](https://github.com/PeterWang512/CNNDetection): motivation for a controlled blur/JPEG augmentation experiment and careful preprocessing. Their results do not establish performance on this user's sources.
- [Ojha et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.pdf): evidence motivating a later pretrained-feature baseline. It does not prove that freezing this ResNet or its BN will solve this failure.

## Handoff completion checklist

- [x] 1: Input contract and parity/score diagnostic tools
- [x] 2: Provenance report, bounded data profile, new pilot/dev/calibration protocol
- [x] 3: Timing, correct logging and bounded performance benchmark
- [x] 4: Loader/transfer/config/startup improvements
- [x] 5: Training/validation synchronization and precision improvements
- [x] 6: Explicit RGB fine-tuning/head/BN options and safe initialization
- [x] 7: Versioned augmentation recipes
- [x] 8: Source-aware selection and corrected scheduler/epoch/stop lifecycle
- [x] 9: Complete single/dual GPU commands and bounded experiment matrix
- [x] 10: Final review and honest user-run acceptance checklist

The implementation LLM should check a box only when the corresponding code/documentation is complete, and separately record whether its runtime tests have actually passed.
