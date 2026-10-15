# DRCT-128 code review: required repairs before Colab training

Review date: 2026-10-14.

## Verdict

**Not ready for the documented Colab training workflow.** The basic one-logit / 128D model and frozen MHA interface exist, but preparation, method fidelity, checkpoint completeness and executable instructions still have material defects. These are implementation issues, not evidence that DRCT cannot generalize.

Keep the user's required architecture: one binary logit with the actual penultimate feature dimension equal to 128. Repair the integration; do not replace the requirement with a native-width classifier or remove safeguards to make commands run.

This review only adds this document. The reviewed Python implementation is unchanged.

## What was verified

Using `C:\Users\SIGMA\anaconda3\python.exe`:

- Seven DRCT test files: **41 passed**, with the two-process test excluded from that invocation.
- The two-process CPU DDP test, separately: **1 passed**.
- Relevant existing tests: **59 passed** across `test_staged_pipeline_and_fusion_heads.py`, `test_integration_checkpoint.py`, `test_rgb_contract.py`, `test_rgb_smoothing_and_logits.py`, and `test_preprocessing_and_gpu_reliability.py`.
- Total: **101 passing tests**, not a complete repository-wide suite.
- Additional small CPU reproductions exposed defects absent from these tests; see the evidence table below.

The repository `.venv` lacks pytest. The first sandboxed attempt encountered temporary-file rename restrictions and stalled in process communication; it was stopped. The tests above were rerun outside those restrictions. Do not count the environmental failures as code failures. No real image dataset, pretrained weight download, reconstruction job, full inference or GPU training was run.

### Reproduced defects despite passing tests

| Small CPU check | Actual observed result |
|---|---|
| Request absent `external_test` split from manifest containing train/train/dev | Dataset returned all three records: `['train', 'train', 'dev']`. |
| Create a training dataset with `drct_alpha=0.3` and originals only | Accepted two originals; no full-DRCT readiness check. |
| Validate contradictory same-hash labels and a group spanning train/dev | `passed=True`, no errors. |
| Add a reconstruction metadata row but leave its image file absent | `is_ready_for_full_drct=True`. |
| Reload a ReLU-projection RGB expert from embedded MHA weights/metadata | ReLU disappeared; max feature difference `0.008839010260999203` on the seeded dummy example. |
| Remove all encoder weights from an otherwise valid DRCT expert checkpoint | Expert loader accepted it. |
| Call `forward_features` with an input requiring gradients | Returned feature had `requires_grad=False`. |
| Build the notebook backbone `clip_vit_b16` | `Unsupported DRCT backbone`. `clip_vit_l14` is also unsupported. |
| Mock `clip.load`, request `pretrained=False, allow_download=False` | Factory still called `clip.load('ViT-L/14', download_root=None, ...)`. |
| Compare fallback loss with upstream default semantics on the same embeddings | Fallback `0.2714466155`; normalized, unsquared, nonzero-reduced reference `0.8270503283`. |

For the loss example, embeddings were `[[1,0],[1,0.5],[0,1],[0.5,1]]`, labels `[0,0,1,1]`, positive margin 0, negative margin 1. The comparison used direct arithmetic from upstream semantics, not an installed PML package; PML is absent from this Python environment.

## How to execute these repair prompts

Read `DRCT_128_IMPLEMENTATION_HANDOFF.md` and this review. Execute R1–R12, grouping related changes as needed. Add tests that fail on the reviewed implementation and pass after the repair. Run local synthetic tests only. Do not request a trained checkpoint to implement missing functionality. Do not run real-data training or reconstruction. Keep optional real-weight/GPU verification honestly pending.

Preserve user changes already present in the working tree and index. Do not commit, push, reset, or rewrite unrelated notebooks. Produce `DRCT_128_REVIEW_RESPONSE.md` mapping each finding to changed files, actual tests and remaining limitations. Correct the implementation report and Colab guide after the code is fixed.

## R1 — [P1] Missing splits must fail instead of loading all rows

**Location:** `data/datasets/drct_dataset.py:36–63`.

When `get_split(split)` returns no rows, the implementation replaces the selection with the entire manifest. A typo, `val`/`dev` mismatch or an external split absent from an internal manifest can consequently evaluate on training images. The broad `except Exception` also hides malformed DRCT input and routes to another loader. `DRCTManifest.load()` does not validate record integrity before selection.

**Repair prompt:**

> Make manifest type and split selection explicit. An absent/empty requested split must raise a useful error listing available splits. Validate a DRCT manifest before use, preserve canonical record/path/label types, and do not recover from corruption by changing dataset type or reading all records. If standard manifests are supported, select their parser by an explicit schema check, not an exception blanket. A checkpoint's old `drct_manifest_path` must not override a newly requested evaluation dataset; clear training-only data references when loading for evaluation.
>
> Add tests through the actual dataset factory/evaluation loader for missing split, malformed schema, wrong path, and evaluation with an old training-manifest option. Assert exact sample IDs and counts, not just tensor shapes. Never relax the existing train/dev separation checks.

## R2 — [P1] Full DRCT preparation and training are not connected

**Locations:** `data/drct_preparation.py:89–205, 270–302`; `data/datasets/drct_dataset.py`; `train.py:353–399`; `data/loaders/dataloader_factory.py`.

The preparation CLI never supplies a `reconstruction_fn`; there is no real reconstruction backend or explicit precomputed-pair importer. With an empty output cache, `prepare` generates nothing, counts missing images and still returns success. Default preparation reconstructs only real originals, and the CLI does not expose fake reconstruction selection. `check_drct_readiness()` is never called by training. The dataset is a flat list of independently transformed images, without the promised original/reconstruction batch assembly. Ordinary folder training can therefore run with alpha 0.3 and be presented as full DRCT.

New preparation also does not issue the verified-manifest artifact required by `train.py`. Fixing the notebook's argument names alone will not make the pipeline complete.

**Repair prompt:**

> Implement a runnable, explicitly configured reconstruction backend and/or a real importer of precomputed pairs with verified parent mappings. Expose model/revision, root, settings, seed, output and resume controls. Match the pinned upstream construction, including real and fake reconstructions for the chosen full recipe. If a deliberately reduced recipe is supported, name and report it as an ablation. Do not leave a callback-only stub described as a completed GPU preparation feature.
>
> Connect full-DRCT readiness to training before model construction. Distinguish full DRCT, reconstruction-free contrastive control and BCE-only control explicitly; alpha alone is insufficient provenance. Reject missing required kinds/pairs. Implement the intended batch sampling and augmentation coupling from upstream, or document/test a deliberate adaptation with actual class/kind mixture. Preserve parent IDs in batch metadata where needed. Avoid silently multiplying fake-class prevalence when derivatives are appended.
>
> Integrate preparation with the repository's verified manifest gate; issue evidence only after a completed audit and preserve it across Drive restore. Include tiny end-to-end CLI tests from original inventory through prepared manifest, audit, dataset selection and one training step. Mock the expensive reconstruction model only in tests; the production backend must exist. Do not generate the user's images during implementation.

## R3 — [P1] The readiness audit and resume cache accept invalid evidence

**Locations:** `data/drct_manifest.py:67–157`; `data/drct_preparation.py:38–76, 133–170, 209–259`.

Validation checks parent split equality but not conflicting content hashes or groups shared by different originals across splits. Readiness does not check image existence, decodability or stale hashes. Preparation skips an existing reconstruction record by ID without checking its output; otherwise it accepts any nonempty file under the expected name. Generated images are passed directly to the final filename, despite the atomic-output claim. The stored seed is not passed to the reconstruction callback. Inventory drops existing identity/video/original/generator metadata and substitutes sample ID for missing groups. Relative source paths are not consistently resolved during reconstruction.

**Repair prompt:**

> Reuse the canonical relationship/content audit rather than maintaining a weaker parallel validator. Preserve original identity, video, generator, original IDs and source fields, and keep unknown values unknown. Check duplicate IDs, parent kind consistency, connected groups spanning splits, conflicting same-content labels, paths and reconstruction lineage. The existing inspected external-development collection must not be eligible as reconstruction training parents.
>
> Bind cache evidence to verified parent content, the complete reconstruction configuration and applied seed. Validate output identity before reuse; stale metadata, deleted output and partial/corrupt images must be repaired or rejected. Write an image to a temporary file, validate it, then atomically publish it and its record. Persist progress so an interruption cannot turn a partial file into a completed cache entry. Add explicit dataset root handling and fail with a nonzero status when full readiness is false.
>
> Tests must cover a missing reconstruction file, corrupt nonempty output, content change, same-hash contradictory labels, same group on different original rows across splits, mismatched reconstruction kind/parent, external-development parent exclusion and interruption before final rename. A cache-key unit test alone is not a resume-integrity test.

## R4 — [P1] The fallback loss is not the upstream contrastive loss

**Location:** `models/drct/loss.py:28–100`; `tests/test_drct_training_objective.py:15–47`.

The fallback uses unnormalized embeddings, squared hinge violations and means including zero violations. The upstream default uses normalized Euclidean embeddings, unsquared hinge penalties and an average-over-nonzero reducer. Installing PML therefore changes the objective. Catching every PML exception can change it again mid-run. The current manual-math test verifies the fallback's different objective, not upstream agreement.

Primary references checked during review:

- https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/contrastive_loss.py
- https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/distances/base_distance.py
- https://github.com/beibuwandeluori/DRCT/blob/main/utils/losses.py

**Repair prompt:**

> Pin and use the required PML version, or implement a genuinely equivalent fallback with numerical loss and gradient parity. Do not catch arbitrary runtime errors and switch objectives. Record the loss implementation/version in checkpoints. Keep edge cases finite with graph-connected zeros and stable FP32 calculations under AMP. Replace the manual fallback test with independent PML parity tests for mixed pairs, zero/nonzero violations, scale changes, singleton/homogeneous inputs and non-default margins. Verify identical behavior in environments with and without the optional fallback path. Preserve the intended BCE/contrastive weighting.

## R5 — [P1] Offline restore still invokes CLIP's weight download path

**Location:** `models/drct/backbone.py:53–59, 78–122`.

`download_if_missing=False` enters a block containing only `pass`, then calls `clip.load(model_name)` anyway. `pretrained` is not used for the real-backbone construction branch. Evaluation, MHA loading and resume can contact the network or fail on an empty cache even when the checkpoint contains the entire encoder. Tests only use the dummy backbone and miss this.

**Repair prompt:**

> Separate architecture-only construction, explicit pretrained initialization and checkpoint restoration. Construct the real visual architecture without fetching weights when restoring a complete checkpoint. Honor `pretrained`, local weights and `allow_download` consistently, and fail clearly for unavailable explicitly requested initialization. Validate actual backbone name/feature width instead of accepting incompatible files based on a string. Implement only supported aliases and use the handoff's ViT-L/14 default consistently.
>
> Add tests with CLIP loader/network access mocked to raise: complete standalone and embedded checkpoints must load without invoking any download-capable path. Explicit pretrained initialization may download only under its documented option. Include an architecture-only real-backbone construction test where dependencies are installed; avoid a full pretrained download.

## R6 — [P1] Partial/mismapped checkpoint imports are accepted

**Locations:** `models/drct/checkpoint_io.py:135–180, 208–247`; `models/shared/expert_loading.py:181–186`.

Official import maps every `model.model.*` tensor into `encoder.*`. The official CLIP wrapper stores the full CLIP model, including text-side tensors; this implementation retains only its visual module. Unmatched/missing keys are returned under `strict=False` without rejecting an incomplete detector. The encoder-initialization mapping also strips the `visual.` wrapper incorrectly for this local encoder and accepts missing weights. It rejects an official 1024D projection even when the operation is supposed to import only the encoder and freshly initialize the head.

Separately, the MHA loader explicitly permits missing encoder weights and can construct an unrelated initialized encoder. The current official-import test supplies only projection/classifier weights and merely verifies the algebraic softmax/sigmoid identity; it cannot establish checkpoint completeness or upstream parity.

**Repair prompt:**

> Implement exact, explicit upstream visual/projection/classifier mapping for the pinned architecture. Deliberately exclude known text-only tensors with a documented allowlist; reject all missing required visual/head weights and unknown unexpected tensors. Validate original class order. Encoder-only initialization may ignore the incompatible official head explicitly, but must strictly restore the complete encoder and report a fresh 128D head. Remove the MHA path that accepts missing encoder tensors. Checkpoints that claim to be complete must be self-contained.
>
> Validate native-reference preprocessing from the official checkpoint's recipe, not the adaptation default. Expose the reference adapter through an actual supported evaluation entry point. Add fixtures matching official state-dict structure, including visual and text keys, missing-layer rejection and reference-forward score comparison. Keep tests with real official weights pending when unavailable; complete structural tests still run now.

## R7 — [P1] Embedded MHA checkpoints lose the DRCT projection definition

**Locations:** `training/checkpoint_manager.py:127–134`; `config/protocol.py:24–36`; `models/shared/expert_loading.py:158–177`.

Only architecture and backbone name are embedded as expert metadata. Projection activation/dropout and full preprocessing come from external saved options, which disappear on embedded reload. For an expert trained with ReLU, reloading the embedded expert reconstructs a linear-only projection. ReLU has no state-dict tensors, so strict weight loading passes while predictions change. This was reproduced.

**Repair prompt:**

> Persist each expert's complete architecture and preprocessing specification independently of the fusion head's options. Include activation, projection layout/dropout, input geometry, normalization/version, feature dimension and label mapping. Require and validate this specification for new embedded DRCT experts. Never infer new DRCT architecture from weight-name substrings or fall back to a dummy encoder.
>
> Test fresh-process save/load through CheckpointManager and CheckpointLoader using non-default ReLU and GELU projection experts. Remove source expert files before reload and compare both extracted features and final fusion logits. Include altered/missing metadata rejection. Test resume as well as evaluation, and keep legacy Wang checkpoints working.

## R8 — [P1] Preprocessing is not checkpoint-driven and is mislabeled as upstream-equivalent

**Locations:** `data/transforms/drct_transforms.py:1–7, 123–143`; `data/datasets/drct_dataset.py:23–29`; `data/datasets/fusion_dataset.py:45–66`; `config/protocol.py:6–10`.

The implementation calls its transforms upstream-matching, but training is only deterministic resize/pad plus horizontal flip. The official training recipe has additional transforms, and its `is_crop` reference route differs from this always-resize route. The implementation report even claims RandomResizedCrop although that is absent.

Fusion decides RGB architecture from `'drct'` anywhere in the checkpoint path before inspecting metadata, so a Wang checkpoint under a folder named `drct` gets the wrong image transform. DRCT input size is taken from global `cropSize`, not its saved expert contract. A wavelet expert's crop setting can therefore force a different size into CLIP. Saved metadata is not sufficient to restore each branch independently. Adding DRCT-specific names to global `PREPROCESSING_KEYS` also affects legacy non-DRCT config requirements.

**Repair prompt:**

> Resolve branch architecture/preprocessing from validated checkpoint metadata before constructing data loaders. Remove filename inference. Give each RGB expert its own saved input/normalization/resize/crop specification; do not substitute wavelet-global dimensions. For shared augmentations, make the alignment policy explicit; independent random flips currently leave branch alignment uncontrolled. Preserve existing wavelet processing and canonical label ordering.
>
> Implement the verified reference transform for official weights. Implement and clearly name the chosen adaptation training recipe, documenting deviations rather than claiming identity. Persist and enforce its version. Scope required preprocessing keys by architecture/schema so old Wang legacy configs do not acquire unrelated DRCT requirements.
>
> Tests must compare actual transformed tensors for standalone and fusion use, not just output shapes. Include non-square patterned images, a Wang file in a folder containing `drct`, DRCT/wavelet experts with different image sizes, and legacy config loading. Pin upstream code/dependencies before asserting equivalence.

## R9 — [P1] The Colab guide contains commands that cannot execute and incorrect backup paths

**Location:** `COLAB_DRCT_128_TRAINING.md`, especially Cells 2, 4–12.

Concrete mismatches in the reviewed checkout:

| Guide instruction | Actual interface/problem |
|---|---|
| Backbone `clip_vit_b16` throughout | Factory does not support it. |
| Install `timm`, sklearn, scipy | Required OpenAI CLIP and the intended PML loss dependency are not installed/pinned. |
| `prepare --dataset_root ... --output ... --reconstruction_root ...` | Actual prepare CLI requires `--manifest`, `--output_manifest`, `--rec_dir`. It does not inventory folders. |
| Manifest filename `*.json` | DRCT reader/writer implement CSV only. |
| `train.py --n_epochs 5` | Actual option is `--epochs` / `--niter`. |
| Resume with `--epoch latest` | Training requires `--continue_train --resume_checkpoint <full checkpoint>`. |
| Evaluation with `--drct_backbone`, `--manifest_split` | Neither is an evaluate.py argument; use checkpoint metadata and its supported `--split`/manifest interface. |
| Evaluate internal root containing both train and val | This is not a real/fake class root; no valid internal manifest is supplied. |
| Cell 9 says it calibrates | It only launches evaluation, with no calibration command/artifact. |
| Read/copy `/content/checkpoints/drct_128_run1/*.pth` | ExperimentManager uses its experiment directory and a nested `checkpoints` folder. The glob can copy zero files while printing success. |
| Resume after interrupted Colab | Only a post-training copy exists; no periodic/epoch backup, manifest restore or reconstruction restore is supplied. |
| `TARGET_COMMIT` and existing-checkout update | Variable is unused; fetch + checkout does not advance an already-existing branch to fetched changes. |

The MHA command also lacks a valid manifest or explicit permitted folder-mode setup and development selection. Do not bypass these requirements to make a command parse.

**Repair prompt:**

> Rewrite the notebook from the actual completed interfaces after R1–R8. Use the correct paths/splits and persist a run descriptor containing exact output/checkpoint paths. Add tested preparation, audit, restore, training, explicit resume, internal calibration, frozen-threshold external-development scoring and MHA cells. Keep code/data local and persist complete preparation artifacts and checkpoints to Drive. Backup must check files exist and report the number copied, use atomic writes, and have an interrupted-session recovery path. Do not describe post-success-only copying as interruption recovery.
>
> Test notebook command construction against real parsers. Run bounded CPU end-to-end lifecycle checks with temporary images and a dummy encoder, including calibration and a restored session. Do not run the user's training. Keep optional official-reference evaluation a separate runnable command. Record the actual application commit and upstream commit rather than an unused constant. Restore existing prepared artifacts instead of recreating them on each session.

## R10 — [P2] The benchmark's real-data path fails, and accumulation is ignored

**Location:** `tools/benchmark_drct.py:55, 155–176, 215–241, 300–320`.

The benchmark creates loader options without `mode`, but `dataset_factory.dataset_folder` reads `opt.mode` directly. Its documented real-data path will fail even after correcting the backbone. The CLI accepts `grad_accum_steps` but never propagates or applies it: every batch steps the optimizer. Synthetic data is silently selected whenever a manifest is missing, making a worker sweep appear relevant despite no image loading. A selected `cuda:1` is also not consistently honored by model options that hardcode GPU 0.

**Repair prompt:**

> Build benchmark options through the real validated configuration path. Test real temporary image loading as well as synthetic tensors. Honor requested accumulation/device, or reject unsupported settings rather than reporting them as used. Require an explicit synthetic mode and label its limited scope. Report actual encoder, data source, optimizer steps and expanded batch counts. Do not use a synthetic worker sweep to select real-dataset worker count. Add a command smoke test for a tiny manifest and both workers=0 and workers>0.

## R11 — [P2] Feature gradients and encoder-unfreeze options contradict their contracts

**Locations:** `models/drct/drct_model.py:125–141, 162–170`; `models/drct/trainer.py:80–84`; `train.py` DRCT freeze option.

`forward_features()` unconditionally uses `no_grad()` and temporarily switches global module mode. It detaches inputs even when gradients are requested. Separately, `--no-drct_freeze_encoder` makes encoder parameters require gradients, but `trainable_parameters()` only yields projection/classifier parameters, so the optimizer never updates or clears the encoder gradients.

**Repair prompt:**

> Make feature extraction preserve autograd when requested, with deterministic evaluation supplied by the caller's mode rather than hidden global mode toggling. Share the pre-dropout feature computation with the classifier as specified in the handoff. Either implement unfreezing with correct optimizer groups/gradient handling and saved policy or reject it until supported. Test input-gradient propagation, projection/classifier updates and frozen/unfrozen encoder behavior explicitly.

## R12 — [P2] Replace completion claims with evidence that covers the real workflow

**Location:** `DRCT_128_IMPLEMENTATION_REPORT.md` and the new DRCT test files.

Passing dummy-model tests is useful, but does not validate missing preparation implementation or the notebook. The report claims supported backbone aliases, RandomResizedCrop, complete reconstruction readiness, strict reference loading and resumable preparation beyond what the code provides. It also confuses visual encoder width and contrastive projection width and has no actual upstream commit/dependency pin under its pinned-dependencies section.

**Repair prompt:**

> After fixing R1–R11, update the report from actual code and command outputs. Provide a finding-by-finding response with file locations and named tests. Separate implemented, CPU-tested, mocked, official-weight-pending and GPU-pending work. Do not mark an acceptance checkbox complete merely because a schema or placeholder function exists. Do not strengthen claimed generalization from synthetic tests.
>
> Keep the existing 42 DRCT tests and the relevant regression coverage, updating incorrect expectations rather than deleting tests. Add tests for the reproduced failures in this document. Require strict reference import tests, real loader/CLI lifecycle tests, non-default embedded projection round trips, and backend-present/backend-absent loss behavior. Record exact test counts and interpreter/dependency versions. Make the first real Colab action clear only once that action has a tested executable path.

## Suggested repair order and completion gate

1. R1–R4: trustworthy data selection, preparation and objective.
2. R5–R8: real backbone, checkpoint fidelity and fusion/preprocessing.
3. R10–R11: benchmark and feature/optimizer contracts.
4. R9 and R12: executable Colab workflow and evidence-based report.

Do not start the user's expensive training until the full preparation -> audited data -> adaptation training -> checkpoint -> standalone evaluation -> MHA reload lifecycle has passed bounded tests. Real GPU throughput and performance remain later experiments; do not guarantee 75–85% accuracy.

### Exact message to give the implementing LLM

Read `DRCT_128_CODE_REVIEW_FIXES.md` and fix R1–R12 while preserving `DRCT_128_IMPLEMENTATION_HANDOFF.md`'s one-logit/128D-feature/MHA contract. Add regression tests for the reproduced defects, repair the actual full DRCT preparation/training path, rewrite the Colab guide using verified commands, and return `DRCT_128_REVIEW_RESPONSE.md` with evidence for each finding. Do not launch real dataset training, full inference, large downloads or reconstruction generation. Missing trained weights must not block implementation; clearly mark only the corresponding real-weight/GPU checks pending.
