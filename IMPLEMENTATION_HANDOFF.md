# Generalization repair: code-only implementation handoff

## Current instruction: prepare code now; the user runs Colab later

Updated 2026-09-29. This revision supersedes earlier instructions to reproduce the 60% result, locate historical checkpoints, execute model tests, or run experiments during implementation. The missing historical checkpoint is NOT a blocker for any task below. Do not request it or search for it.

The objective is to prepare a reproducible fresh-training pipeline for a 100K-image pilot (50K real, 50K fake), followed by larger experiments if measured generalization supports scaling. Improving external accuracy beyond 85% is an experimental target, not a guaranteed consequence of these edits. Code alone cannot establish improvement or close empirical reviewer objections.

Execute ONE numbered prompt at a time to conserve usage. Write the required code, configurations, tests and notebook cells; do not execute model computation. After all implementation tasks are finished, hand control to the user, who will manually train on Google Colab using an L4 or two T4 GPUs. Finishing implementation does not authorize an automatic run.

### Execution boundary for every task

Allowed now:

- Read relevant source files and existing text reports; preserve all current user changes.
- Edit code, write tests, configurations, command generators and unexecuted Colab notebook cells.
- Perform static checks only: parse Python source with AST, validate JSON/notebook syntax, inspect diffs. Do not import project modules for validation.
- Record missing future runtime inputs as documented configuration fields; continue independent implementation.

Deferred until the user explicitly starts the later Colab runtime phase:

- Training, validation passes, inference, evaluation, diagnostic predictions, profiling and benchmarks, including on synthetic images.
- Model construction, forward/backward calls, optimizer steps, checkpoint loading, reading weight files, pretrained-weight downloads and model smoke tests.
- Running test suites or train/evaluate/robustness entry points, including help/dry-run commands that could import or initialize models. Write runtime tests now and label them NOT RUN.
- Dataset scans, sampling actual images, hashing the million-image collection, generating real split manifests, installations and GPU launches.

These restrictions govern what the implementing agent executes now. The generated training code must still contain proper validation, checkpoint saving/resume, later evaluation and runtime safety checks. Do not remove these features to satisfy the code-only boundary. All file I/O and model initialization must be behind explicit runtime entry points, never triggered by importing a module or displaying configuration.

## Repository and data references

Repository:

`G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection`

| Purpose | Local reference path | Previously observed structure |
|---|---|---|
| Approximately 1M images; count supplied by user | `G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Preapred Dataset` | `train/fake`, `train/real`, `val/fake`, `val/real` |
| Previously failed external fake set | `F:\LDM` | Images under `F:\LDM\LDM`; exclude archives and incomplete downloads |
| Previously failed mixed external set | `F:\val to be deleted 2` | `fake`, `real` |

These are read-only source references, not portable Colab paths. Make Colab data/output locations configurable; do not assume access to Windows drives. Do not inspect these image folders during code preparation.

Treat the already-inspected failed external sets as development data if they influence model decisions. Reserve independently held-out external sources for final evaluation. LDM is fake-only: later report fake recall/miss rate, not binary AUC or balanced detection accuracy. Do not invent missing source, identity, video, generator or demographic metadata.

Paper: `C:\Users\SIGMA\Downloads\After_Third_Review.pdf`.
Review history: `C:\Users\SIGMA\.codex\attachments\45a3d3e5-97d8-47bf-b0e1-9cb3b90a7bca\pasted-text.txt`.
Reviewer mapping: `output/review/REJECTION_RECOVERY_PLAN.md`. Read the existing mapping first; avoid repeatedly reading the entire paper.

## Existing work to preserve

Inspect only relevant diffs before editing. Do not rerun scratch editing scripts in `tmp/` or overwrite existing work.

- `data/manifest.py`, `prepare_dataset.py`: explicit labels, provenance/group audits, inventory, connected-group splits, optional dHash candidate search.
- Dataset loading: real=0/fake=1, manifests, deterministic validation transforms and no balanced validation resampling.
- `models/mha/token_fusion.py`: multiple-key token attention and gated fusion; legacy single-key attention remains for compatibility.
- `models/shared/two_stream_trainer.py`, `expert_loading.py`: frozen-expert protocol and shape/preprocessing checks.
- Checkpoint metadata: options, label mapping, manifest/expert hashes and embedded expert weights; legacy settings are explicit.
- Variable embedding sizes, safe level-4 wavelet pooling and explicit stable `signed_log1p` versus legacy transforms.
- `evaluation/generalization.py`, `analyze_predictions.py`: prediction export, subgroup metrics, group bootstrap, calibration and late ensemble.
- `evaluation/pixel_pipeline.py`, `robustness_cli.py`, `end_to_end_profile.py`: robustness and profiling infrastructure, still needing integration review.
- Early initialization seeding, development-AUC checkpoint selection, distributed sampling and unpadded validation.

Historical integration evidence is recorded at the end. It does not establish that subsequent edits work or that generalization improved. Runtime verification of new changes stays pending until the later Colab phase.

## Shared completion rules

Work only on the selected prompt. No agents, extra experiments or broad rewrites. Keep pure configuration validation separate from checks that require datasets, models or checkpoints. Do not weaken runtime integrity checks merely because the corresponding files do not exist during code preparation.

For each task, write code and deferred tests, perform static checks, then update this checklist and a brief handoff note listing changed files, checks actually performed, tests NOT RUN and the next prompt. A checked box means implementation is prepared, not experimentally validated. Record runtime and reviewer-evidence status separately. Never fabricate metrics or mark reviewer experiments complete from code alone.

## Task checklist

- [x] 1. Historical integration work completed; preserve it (see historical note)
- [x] 2. Implement a fresh-training workflow without historical-checkpoint dependencies
- [x] 3. Implement source-aware 100K pilot preparation and leakage safeguards
- [x] 4. Complete training-time selection and later reporting code
- [x] 5. Complete wavelet, augmentation and GPU reliability code
- [x] 6. Prepare unexecuted Colab training and ablation jobs
- [x] 7. Prepare controlled baseline interfaces and protocols
- [x] 8. Prepare reviewer evidence templates and final code handoff

## Prompt 1: preserve completed integration

Read the historical Task 1 note and output/review/INTEGRATION_STATUS.md. This task was completed previously; do not rerun its optimizer-step, checkpoint-reload or inference tests. Preserve the resume/protocol checks and self-contained checkpoint design while completing subsequent tasks. New runtime regression tests may be written but must remain unexecuted during this code-only phase.

## Prompt 2: fresh-training workflow, no old trained model required

Read IMPLEMENTATION_HANDOFF.md and follow its code-only execution boundary. Inspect train.py, configuration, expert trainers and shared fusion initialization. Implement explicit stages: train the RGB expert on the new training split; train the wavelet expert on that split; then train the selected fusion head using the newly produced expert artifacts. Keep both experts frozen for this fusion protocol. Make initialization and execution lazy, with no import-time model creation, checkpoint reads or downloads.

Fresh standalone expert training must not require any historical deepfake checkpoint. Provide explicit fresh/resume modes; resume is optional and only loads a user-selected artifact when a later run starts. General-purpose pretrained initialization may remain a clearly documented configuration option, downloaded only during the user-started runtime; distinguish it from resuming a trained detector. Also provide a documented initialization path without downloads where supported.

Fusion runtime must still require compatible expert checkpoints produced by the earlier stages. Generate their paths through a shared run manifest and validate them at the start of that future stage. Never silently substitute random experts. Missing future artifacts must not block writing code or performing static checks. Preserve label direction, preprocessing, dimensions, resume state and checkpoint metadata safeguards. Write deferred tests for fresh/resume initialization, stage ordering and missing/incompatible experts; do not execute them. Do not reproduce the old 60% result or ask for its checkpoint.

## Prompt 3: pilot data preparation code only

Extend prepare_dataset.py and manifest/configuration interfaces to support a seeded 100K TRAINING subset with 50K real and 50K fake plus separate development/test groups. Write commands/configs for later execution; do not scan images or create actual dataset splits now.

Implement configurable source/generator quotas and frame caps, source-by-class summaries, honest shortage reporting and documented metadata adapters. Balance sources within class where feasible and expose source/class confounding; equal real/fake totals alone do not solve it. Keep originals, manipulations, identities, duplicate/recompressed images and related frames together via connected groups. Never assign every frame an independent group to conceal missing metadata. Unknown provenance must remain visible, and strict auditable splitting must fail clearly at runtime if required grouping is absent.

Support scalable, resumable hash caching and duplicate-candidate reports; do not equate approximate search with proof of no leakage. Preserve input images and separate development domains from untouched final benchmarks. Write fixture-based tests for deterministic quotas, shortages and cross-split group leakage; leave them unexecuted. Provide example manifests with clearly synthetic placeholders only.

## Prompt 4: development selection and reporting code

Finish the training validator, checkpoint selection and prediction/report interfaces. Use deterministic development transforms, preserve natural development prevalence, and fit thresholds only on designated development data. Keep final-test samples out of selection and calibration. Save sample IDs, canonical labels, group/source metadata, checkpoint identity and threshold provenance in later run artifacts.

Complete metrics and grouped bootstrap code with explicit unavailable results for single-class/undefined metrics, aligned paired comparisons, independent-group units and separate seed variability. Classification comparisons must support independently development-selected model thresholds. Summarize three independently trained seeds only when those runs exist; repeated inference is not a new seed. Fairness reporting requires supplied legitimate annotations and group counts. Write deferred tests for label direction, incompatible prediction IDs, missing groups/classes and forbidden test calibration. No checkpoint loads, predictions or result generation now.

## Prompt 5: reliable preprocessing and GPU code

Inspect augmentation, CPU/tensor wavelets, DataLoader setup and distributed training. Propagate wavelet_log_mode through every supported standalone/fusion path. Derive both streams from the same transformed image; cached wavelets must not silently diverge from augmented RGB. Apply label-independent, configurable resizing/JPEG/blur augmentations and keep validation deterministic. Preserve legacy transforms only for explicitly legacy artifacts.

Write deferred tests for constant/low-texture inputs, levels 2/3/4, CPU/tensor agreement, aligned streams, model gradients and checkpoint parity. Do not construct models or execute these tests. Prevent worker-side CUDA initialization, repeated padded validation samples and inconsistent distributed run identities. Review correct rank devices, weighted sampling, buffer synchronization and exception-safe restoration.

Finish optional raw-pixel robustness and end-to-end profiling interfaces needed by reviewers, but defer all attacks, forward passes and profiling. Keep corruption and adversarial protocols distinct. Write resource-aware mixed-precision and gradient-accumulation configuration for future L4/T4 execution; never treat two T4 devices as one shared-memory GPU. Mark multi-GPU behavior and throughput unverified until measured.

## Prompt 6: Colab notebook and experiment configurations

Prepare an UNEXECUTED Colab notebook and concise run guide. Clear all cell outputs and execution counts. Use configurable repository/data/output roots, documented dependencies and persistent artifact locations. Installation, mounting, device detection, dataset preparation and model execution belong in explicit cells that the user runs later; do not execute them now.

Make the sequence clear: environment setup; data audit and 100K preparation; RGB expert training; wavelet expert training; fusion training; development comparison. Validation within training is required when the user launches training later. Put external inference, robustness and profiling in a separate optional post-training section with no automatic execution. A missing historical detector checkpoint must never appear as an initial prerequisite.

Provide a single-GPU configuration for L4/T4 and an optional two-GPU distributed configuration only when the future runtime exposes two devices. Include memory-adjustable batch sizes, accumulation, effective batch calculation, seeds and resume-from-new-artifacts instructions. Do not promise batch capacity or runtime before measurement.

Prepare one-seed screening and then at least three full-pipeline seeds for shortlisted comparisons: concatenation, gated fusion, equal-weight late ensemble, corrected token attention and a capacity-matched control. Within each seed, reuse the same selected experts for fair fusion comparisons. Define targeted embedding/wavelet/augmentation/frame-cap ablations without a full Cartesian grid. Leave-one-source/generator-out jobs must retrain experts without the held-out domain; prevent leakage through detector pretraining. Plan 100K then 200K/400K and eventually approximately 1M only after development evidence supports scaling, with fixed development groups. Validate configuration syntax statically; do not run generated commands or a training matrix.

## Prompt 7: baseline code and protocol preparation

Consult the reviewer mapping and prepare a small relevant baseline set, with candidates such as UniversalFakeDetect, UCF/SBI and Effort. Verify official documentation, code/license and preprocessing requirements before coding a concrete adapter; do not download weights or instantiate a baseline. If details cannot be verified, mark that adapter pending rather than pretending integration is complete.

Use a common manifest/prediction interface and lazy optional imports. Keep each method's native training/preprocessing requirements and auxiliary-data provenance explicit. Separate supplied-checkpoint evaluation from retraining on the pilot. Missing optional baseline weights/dependencies must not block the main training code. Write deferred adapter alignment tests and reproducible future commands. No smoke inference now, no random-weight substitution and no comparison of published numbers as if locally measured.

## Prompt 8: reviewer evidence scaffolding and code-ready handoff

Create or update the reviewer closure matrix against the existing mapping, including final rejection concerns: novelty/necessity, external generalization, strong baselines, verified splits, robustness, fairness and complete cost reporting. For each concern record the supporting code, required future experiment, expected artifact and status. Code prepared is distinct from runtime validated and evidence complete.

Prepare table-generation scripts and empty, clearly labeled templates for dataset counts, seed summaries, paired differences, ablations, robustness, subgroup coverage and full detector cost. Do not generate supposed empirical results or overwrite the old paper's results with placeholders. Mark inconsistent historical confusion counts/accuracy as unreproduced until supported evidence exists. Prepare a list of architecture/equation/claim corrections; single-key attention must not be described as dynamic routing, and concatenation MLPs must not be described as inherently nonadaptive.

Finish with static code/config/notebook checks and a compact CODE_READY_STATUS.md listing implemented tasks, unresolved implementation gaps, all deferred runtime tests, future data requirements and exact manual Colab stages. Do not declare full code readiness if implementation gaps remain. Do not ask for old trained models, launch Colab, train, infer or benchmark. Reviewer empirical claims remain pending until the user completes the later runs.

## Recommended next user prompt

Read IMPLEMENTATION_HANDOFF.md and execute Prompt 2 only. Write the fresh-training pipeline code and deferred tests. Do not load or search for trained models, initialize models, run inference/training/tests, scan datasets or download weights. Use static checks only. Keep the summary brief and update the checklist. I will run the completed pipeline on Google Colab later.

## Historical notes (preserved; not instructions to execute)

### Task 1, 2026-09-17

The earlier integration task was completed. output/review/INTEGRATION_STATUS.md records CPU optimizer-step and self-contained reload parity checks for three fusion heads, resume/protocol checks, 17 passing tests, compilation and CLI help checks. These are historical results from before the current no-execution instruction. They are not authorization to repeat those checks now and do not validate future edits. No full dataset training job was launched.

### Original Task 2, 2026-09-17; dependency superseded 2026-09-29

Earlier directory inspection found fake-only F:\LDM\LDM and the mixed external folder, with representative 256x256 PNGs. No historical checkpoint/run configuration was found. output/review/diagnostic_manifest_v1.csv and DIAGNOSTIC_STATUS.md record that limited diagnostic work; reproducing the historical 60% remains unperformed.

The old requirement to stop until that checkpoint is supplied is explicitly removed. Any conflicting checkpoint-request or execution directions in earlier plans/status files are historical and do not apply to this code-only handoff. Continue with fresh-training implementation. Revisiting the historical diagnostic is outside the current task.
