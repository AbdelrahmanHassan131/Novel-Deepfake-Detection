# DRCT with a 128-dimensional RGB expert: implementation prompts

Prepared: 2026-10-14. Execute prompts in order in this repository. This is an implementation specification, not evidence of trained model performance.

## User's required outcome

Add a new `DRCT_128` model alongside `Wang2020_128`:

```text
RGB image -> DRCT-trained visual representation -> 128-dimensional embedding
                                                    |
                                                    +-> Linear(128, 1) -> binary logit
                                                    |
                                                    +-> RGB input to existing MHA fusion
```

Labels remain `real=0`, `fake=1`. The standalone model returns one raw logit per image, shape `[B,1]`. Its feature method returns `[B,128]`. Sigmoid belongs in scoring, not before the training loss. The same learned 128D representation must feed the binary classifier and MHA; do not add an unused 128D branch merely to satisfy a shape check.

Keep Wang, wavelet, and existing fusion checkpoints usable. The requested 128 refers to feature width, **not image resolution**. Use the selected backbone's verified input resolution, initially 224 for the CLIP route.

## Scope and scientific distinction

DRCT is a training framework, not just a named backbone. The official method combines reconstruction-generated hard examples with a contrastive objective. Original real images are real; original generated images and reconstructed images are fake. See the [official repository](https://github.com/beibuwandeluori/DRCT) and [ICML 2024 publication](https://proceedings.mlr.press/v235/chen24ay.html).

The user's 128D head is a deliberate adaptation. Call resulting experiments **DRCT-128 adaptation**. Do not claim the published accuracy or exact paper reproduction. Keep an official-width reference path separate so a published checkpoint can later be evaluated without silently changing its architecture.

Implement and run synthetic/local CPU tests now. Do not launch real dataset training, reconstruction generation, full inference, large downloads, or dataset-wide hashing. Missing trained weights must not block implementation. Mark unavailable real-weight parity and GPU checks as pending, never as passed. The user runs those steps in Colab after implementation.

Do not overwrite existing notebooks, manifests, reports, or unrelated local changes. Complete routine design decisions independently. Maintain `DRCT_128_IMPLEMENTATION_REPORT.md` with completed tasks, actual tests, limitations, and exact next commands. Do not invent measurements.

## Inspected integration points

The following findings are based on the current local code. Recheck them if another agent changes the checkout.

| Location | Current behavior / required work |
|---|---|
| `models/wang2020_128/trainer.py` | ResNet head is `2048 -> embed_dim -> ReLU -> Dropout -> 1`; trainer uses `set_input`, `forward`, `get_loss`, `optimize_parameters`. |
| `models/shared/expert_loading.py` | RGB validation requires `fc.0.weight`; loader constructs ResNet and truncates its head. This will not load DRCT by adding a registry entry alone. |
| `models/shared/two_stream_trainer.py` | Both expert outputs must be `[B,embed_dim]`; experts stay frozen/eval. |
| `models/mha/trainer.py` | New training supports token attention/gated fusion; legacy single-key attention is evaluation-only. Preserve that restriction. |
| `data/datasets/fusion_dataset.py` | Shared augmented image feeds both branches; RGB normalization is currently hardcoded. Branch-specific preprocessing needs explicit support. |
| `training/base_trainer.py` | AMP/accumulation paths call `forward()` and `get_loss()`; another path calls `optimize_parameters()`. All must use the intended DRCT objective. |
| `training/checkpoint_manager.py` | Saves main model plus embedded expert state dictionaries. Architecture/preprocessing metadata must accompany DRCT embedded weights. |
| `config/protocol.py` | Saves options, manifest hashes and expert file identities; current preprocessing keys are insufficient to identify arbitrary RGB backbones. |
| `evaluation/checkpoint_loader.py`, `evaluation/inference.py`, `training/validator.py` | Must rebuild DRCT, preserve raw logits, and provide identical validation/evaluation preprocessing. |
| `config/{types,defaults,configuration,compatibility,validator}.py`, `train.py`, `train_pipeline.py`, `training/pipeline.py` | Audit option conversion, architecture lists, staged dispatch and protocol enforcement end to end. |

## Primary sources to inspect before implementation

Pin an upstream commit in the implementation report; `main` links below are discovery links, not a reproducibility pin.

- Paper: https://proceedings.mlr.press/v235/chen24ay.html
- Official repository: https://github.com/beibuwandeluori/DRCT
- Model definitions: https://github.com/beibuwandeluori/DRCT/blob/main/network/models.py
- Training: https://github.com/beibuwandeluori/DRCT/blob/main/scripts/train_contrastive.py
- Objective: https://github.com/beibuwandeluori/DRCT/blob/main/utils/losses.py
- Dataset/pair assembly: https://github.com/beibuwandeluori/DRCT/blob/main/data/dataset.py
- Transforms: https://github.com/beibuwandeluori/DRCT/blob/main/data/transform.py
- Reconstruction: https://github.com/beibuwandeluori/DRCT/blob/main/data/diffusion_rec.py

Inspect actual call sites, not only README examples. Record dependency versions and applicable upstream licensing/attribution. Prefer a small explicit adapter to importing training scripts with global argument parsing or side effects.

## Prompt 1 — Establish the architecture and reference contracts

> Implement the architectural foundation for `DRCT_128`. Read repository instructions and the integration points above. Inspect and pin official DRCT sources. Use the official CLIP ViT-L/14 route as the initial backbone choice, with the visual encoder frozen by default. Keep backbone construction behind an injectable factory so offline CPU tests can use a tiny encoder. Do not implement several new backbone families at once.
>
> Define a native-width reference adapter separately from the requested 128D model. Determine actual backbone feature width from the selected implementation; do not confuse backbone width, contrastive projection width, and MHA width. The upstream CLIP implementation exposes a 768-wide visual feature and its contrastive wrapper commonly uses a 1024-wide projection. Verify those against the pinned source.
>
> For the new adaptation, use visual features -> trainable projection to 128 -> optional explicitly configured activation/dropout -> Linear(128,1). Start with a linear projection, no added activation/dropout, and no label smoothing; document these as adaptation choices. Do not inherit the user's 0.67 dropout or heavy augmentation automatically. Expose `forward_features(images)` returning the deterministic pre-dropout 128D feature in eval mode, and `forward(images)` returning `[B,1]`. A forward-with-features path must avoid encoding images twice during training. Contrastive normalization must follow the selected loss rather than changing the MHA feature silently.
>
> Implement the local trainer interface, `self.output`, and optional `self.embedding`. Trainable projection/classifier parameters must live inside `self.model` so DDP, checkpoints and optimizer state include them. Frozen encoder parameters remain frozen and eval after `.train()`. Use `no_grad` for the encoder only when input gradients are not requested; preserve the repository's input-gradient diagnostic contract. Enforce `embed_dim=128` for this architecture with a clear error.
>
> Deliver shape/gradient tests for B=1 and B>1. They must prove that the classifier uses the exported feature, head parameters update, encoder parameters do not, and no pretrained download happens in offline tests or checkpoint reconstruction.

## Prompt 2 — Add explicit checkpoint import and preprocessing

> Implement distinct operations: (a) restore a locally trained DRCT-128 checkpoint strictly; (b) import an official detector at its original width for reference evaluation; (c) initialize a new 128D adaptation from compatible pretrained encoder weights. Do not treat initialization as resume or as a trained 128D detector. Report all loaded/reinitialized/missing/unexpected keys. A broad `strict=False` that conceals incompatibilities is unacceptable. Validate upstream class order explicitly.
>
> For an official binary two-logit reference head, expose `z = logit_fake - logit_real`; sigmoid(z) equals its two-class softmax fake score. Test numerical parity. This conversion does not turn a native-width head into a trained 128D head. If an official projection cannot fit the new dimensions, initialize the new projection/classifier and require adaptation training. No automatic truncation, averaging or slicing of learned weights.
>
> Store backbone identity, upstream revision, initialization origin/hash, embedding width, head definition, label mapping, normalization, input size, resize/crop/interpolation/padding policy and preprocessing version. Evaluation/resume must reconstruct from metadata with network downloads disabled. Keep optional DRCT dependencies from breaking Wang-only imports/CLI help.
>
> Inspect upstream transforms rather than assuming standard CLIP normalization: the official DRCT training/testing scripts use their dataset transforms. Reference import must match those transforms. Specify and version preprocessing for the adaptation, and use it consistently in standalone validation, evaluation and MHA. Test PIL/OpenCV RGB order, small/large/non-square images and deterministic repeated evaluation. Store enough metadata to reject incompatible overrides. Real official-weight score parity is a later user-run check if weights are unavailable; architecture/import tests still run now.

## Prompt 3 — Prepare reconstruction-aware data without repeating old work

> Add a manifest-backed DRCT preparation workflow. Reuse existing audited source manifests where available. Existing real/fake folders alone do not contain the reconstruction examples required for full DRCT training. Provide a command to inventory/import originals and a separate resumable command to generate or import their reconstructions. Do not generate any now.
>
> Preserve original files and labels. Record original sample ID, parent ID, group/identity/source information where known, split, sample kind (`real_original`, `fake_original`, `real_reconstruction`, `fake_reconstruction`), content hash, reconstruction model revision, settings and seed. Derived samples inherit parent group and split. Reconstructed real images have binary label 1 but their authentic parents retain label 0. Never move derivatives independently across splits. Reject contradictory content/labels, cross-split connected groups and external evaluation parents. Unknown provenance stays unknown.
>
> Match reconstruction parameters and pair construction to the pinned upstream code, documenting every adaptation. Import precomputed reconstructions only with verifiable parent mapping and metadata. Supply explicit model/cache/output parameters; do not invent an available checkpoint or assume a raw ZIP has reconstructions. GPU reconstruction preparation is a separate optional Colab session, not work inside DataLoader workers or an inference requirement.
>
> Use stable IDs and a cache key covering input identity and reconstruction settings. Resume without regenerating completed valid outputs. Write images/records atomically and do not certify partial runs as complete. Missing/stale pairs must be counted and fail full-DRCT readiness; do not silently revert to RGB-only training. Keep any RGB-only control a clearly named separate mode. Persist manifests and reconstruction outputs to Drive with explicit restore paths. Reuse hash evidence only when file identity/settings support it; verify new derivatives once and invalidate changed evidence.
>
> Implement tests with tiny synthetic images for parent linkage, opposite labels on real parent versus its distinct reconstruction, duplicate conflict handling, interrupted preparation, deterministic IDs, cache invalidation and preservation of existing splits. Maintain adapters for folder inputs but do not waive existing audit/provenance checks.

## Prompt 4 — Implement the actual DRCT training objective and batch assembly

> Implement DRCT classification plus margin-based contrastive training, based on the pinned official training call and `CombinedLoss`. The official code uses `pytorch_metric_learning.losses.ContrastiveLoss`; its separately defined similarly named custom loss is not the default used by `CombinedLoss`. Verify normalization, distance, pair reduction and handling of absent pairs with the installed pinned dependency. Do not substitute arbitrary InfoNCE or a generic contrastive formula and call it a reproduction.
>
> For our one-logit adaptation, use hard-label BCEWithLogitsLoss and the verified contrastive objective on the 128D embeddings. Document the single-logit change from upstream cross entropy. Follow the upstream weighting convention `(1-alpha)*classification + alpha*contrastive`; verify the pinned default before setting it (currently alpha=0.3, positive margin=0, negative margin=1 in the training CLI). No label smoothing by default. Loss math on distances/reductions must be stable under AMP, with FP32 computation where needed.
>
> Audit upstream data assembly to reproduce original/reconstruction sampling and binary labels. Preserve pairing through batching/augmentation. State whether batch size counts originals, pairs or expanded images, and log the actual numbers. Both classes and usable positive/negative pairs should be represented during normal training; define finite behavior for singleton, homogeneous and final partial batches. Never create a class imbalance accidentally because derivatives multiply fake examples. Report the actual mixture and any balancing adaptation, rather than claiming balancing is automatically identical to upstream.
>
> Integrate with all existing engine paths: `get_loss`, `optimize_parameters`, AMP and accumulation. Log classification, contrastive and total training losses without a second forward pass. Ordinary validation uses hard-label classification loss and individual image scores, not a batch-composition-dependent contrastive loss for selection. Checkpoint selection and calibration use internal development data only.
>
> Support one GPU and DDP correctly. State contrastive pair scope explicitly. A first implementation may use rank-local pairs with recorded per-rank batch size; this is not equivalent to global pairs or a bigger accumulated batch. Gradient accumulation does not enlarge the contrastive pair pool. If global gathering is implemented, prove gradient, normalization and cross-rank mask correctness. Do not silently switch loss semantics with world size.
>
> Test reference loss agreement on synthetic embeddings, finite gradients, classification/contrastive contribution, all engine paths, frozen encoder mode and a two-process CPU DDP step. Ensure the features used by the loss are obtained through the DDP-wrapped forward, not a bypass of `.module`.

## Prompt 5 — Integrate CLI, configuration, evaluation and resume

> Register `DRCT_128` and its validated options throughout CLI/config conversion and architecture enums. Use scoped DRCT options for backbone, initialization, data manifest, loss and training mode. Existing Wang-only knobs must not silently control DRCT. Invalid combinations should fail before expensive data loading. Verify the effective saved config round-trips all new values.
>
> Integrate dataset routing, train/validation construction, checkpoints, inference, raw-logit prediction export, development calibration and `analyze_predictions.py rescore`. Preserve label identity and checkpoint/prediction hashes. A trained DRCT-128 checkpoint must evaluate without reconstruction data, original training paths, upstream classifier files or an internet connection. Resume restores optimizer, scheduler, AMP, counters and relevant protocol settings; initial pretrained import must never override resumed weights.
>
> Persist both raw logits and probabilities. Use raw logits for ranking diagnostics to avoid saturation artifacts. Use the existing validated threshold loader, whose logit field is `logit_threshold`; never silently replace a missing threshold with zero. Report real/fake recall, balanced accuracy, macro F1 and AUC at fixed and frozen internal-development thresholds. External threshold fitting, if requested later, remains separately labeled calibration fit.
>
> Ensure training startup prints actual backbone, trainable parameters, feature width, initialization source, data kinds/counts, loss settings, batch interpretation and preprocessing. Progress must be visible with unbuffered output. Add offline round-trip tests for standalone save/load/resume, predictions, config and incompatible artifacts. Keep existing RGB tests passing.

## Prompt 6 — Make DRCT-128 a real MHA expert

> Replace RGB-expert architecture assumptions with explicit checkpoint-metadata dispatch while preserving the existing Wang path. Update `validate_expert_checkpoint`, `load_expert`, expert inspection utilities and staged pipeline validation. Do not spoof a DRCT checkpoint as Wang or rely on `fc.0.weight` being present in every RGB model. New DRCT checkpoints require an explicit validated schema; never infer their architecture from loosely matching names.
>
> Load the DRCT feature extractor via `forward_features`, returning `[B,128]`, with encoder and projection frozen/eval for MHA training. Feed these features plus the existing wavelet expert's 128D feature into token attention/gated fusion. Do not feed probabilities/logits to MHA. Do not train the standalone DRCT head during frozen-expert fusion. New RGB embeddings require a newly trained fusion head; an old Wang-based MHA head is not compatible merely because shapes match.
>
> Make fusion preprocessing branch-aware. Do not blindly feed ImageNet-normalized tensors into a differently normalized DRCT branch or force the wavelet branch to adopt DRCT input size. Decode once, maintain aligned image geometry according to a documented deterministic evaluation policy, then apply each expert's saved branch transforms. If original expert geometry cannot be preserved by a shared crop, implement separate branch transforms from the decoded image and record that policy. Keep wavelet value ranges and packet settings intact. Reject unsupported combinations early. Preserve old Wang-wavelet behavior and checkpoints.
>
> Embed DRCT expert architecture, preprocessing, projection weights and state in fusion checkpoints. Audit both external-expert and embedded-expert loading; current embedded state bypasses some metadata checks. The completed fusion checkpoint must reload after source expert files are removed, without network access. Use strict state loading and versioned backward compatibility.
>
> Tests: DRCT+wavelet produces `[B,1]`; frozen expert parameters and statistics do not change; fusion-head parameters receive gradients; standalone versus fusion RGB feature parity holds for identical effective input; branch transforms match each expert's evaluation contract; fresh-process embedded reload reproduces logits; old Wang fusion still reloads. Cover B=1, incompatible dimensions, class mappings, preprocessing and missing metadata.

## Prompt 7 — Make Colab training practical and observable

> Add a DRCT throughput benchmark and integrate it with the existing profiling tools where possible. Do not infer utilization from VRAM allocation alone. Record images/sec, expanded batch size, data-wait time, step time and peak GPU memory. Supply a short user-run sweep of feasible batch/worker settings for one T4/L4/A100; do not promise a speed or hardcode a large batch that can OOM. Compare worker counts against available CPUs. Persistent workers/prefetch apply only when workers>0. Use pinned memory/nonblocking transfer where valid.
>
> Train from local `/content` storage; persist completed checkpoints, logs and preparation artifacts to Drive. Keep reconstruction precomputation separate from detector training. Frozen-backbone feature caching is optional and disabled by default: stochastic training augmentation makes a naive cache change the experiment. Never claim high memory usage itself proves efficiency. Explain that contrastive pair counts depend on actual batch size even when gradient accumulation matches effective optimizer batch size.
>
> Preserve AMP, deterministic validation, safe checkpoint writes, interrupted-session resume and Windows worker picklability. Optional dependencies must not be imported in worker-unsafe closures. CPU tests use temporary local files; GPU benchmarks remain runnable commands with honest pending status.

## Prompt 8 — Deliver executable Colab instructions and review the whole integration

> Write `COLAB_DRCT_128_TRAINING.md` with ordered, copyable cells using only commands/options actually implemented and tested for parsing. Do not publish aspirational commands as working commands. Provide a clean-clone route and a resume route; do not delete user directories automatically or rerun setup with a new random run ID when resuming.
>
> Use these known paths as editable defaults:
>
> - Repository URL: `https://github.com/AbdelrahmanHassan131/Novel-Deepfake-Detection.git`
> - Branch: `GeneralizationDeepFakeDetection` (record the resolved commit)
> - Local repo: `/content/Novel-Deepfake-Detection`
> - Original dataset root: `/content/dataset/DatesetNoVideosNoGAN` with `train/real`, `train/fake`, `val/real`, `val/fake`
> - External development collection: `/content/unseenDataset/val to be deleted 2`
> - New persistent workspace: `/content/drive/MyDrive/deepfake_drct_128`
>
> Cover: dependencies and revision checks; local dataset validation; reconstruction import/generation prerequisites; preparation persistence/restore; a tiny readiness check; optional unmodified official reference evaluation; single-GPU DRCT-128 adaptation training; interrupted resume; internal calibration; external development scoring; backup/download; later MHA training with the new expert plus an existing compatible wavelet checkpoint. Missing official detector weights must not block fresh training from valid backbone initialization. Missing reconstruction data must produce a concrete preparation instruction, not a misleading successful DRCT training command. Separate expensive preparation cells from normal session restore.
>
> Give an optional two-GPU command only after DDP is tested, explaining per-rank batch and pair scope. Mark the reference versus adapted model distinction in outputs. Use a separately named control for no-reconstruction/no-contrastive ablations. Do not choose checkpoint, augmentation or thresholds by external metrics in automated scripts. The repeatedly inspected external collection is development data, not a fresh final test. MHA performance is not assumed to improve just because DRCT standalone improves.
>
> Run the targeted new tests plus the relevant existing RGB, fusion, checkpoint, prediction, configuration and training tests. Review for regression and fix issues. In `DRCT_128_IMPLEMENTATION_REPORT.md`, list changed files, tests and actual results, the pinned upstream method/dependencies, deviations, data/weights required for each command, pending GPU/official-weight checks, and the first exact Colab cell the user should run. Do not claim generalization has been established by tests or synthetic data.

## Final acceptance checklist

- [ ] `DRCT_128` outputs `[B,1]`; its actual penultimate learned feature is `[B,128]`.
- [ ] Full DRCT mode uses reconstruction-derived examples and the verified contrastive objective.
- [ ] Official reference weights and 128D adaptation weights cannot be silently confused.
- [ ] All training paths optimize the same intended loss; validation uses the correct score contract.
- [ ] Preprocessing/labels/options survive checkpoint export, evaluation and fusion reload.
- [ ] MHA accepts DRCT features through metadata dispatch; old Wang behavior remains intact.
- [ ] Embedded fusion checkpoints are self-contained and tested without original expert paths.
- [ ] Preparation can resume and be reused across Colab sessions without regenerating valid artifacts.
- [ ] CPU tests pass; real data/GPU work not executed is explicitly pending.
- [ ] Colab instructions state the extra reconstruction data requirement and contain real implemented commands.

## Short instruction to give the implementing agent

Read `DRCT_128_IMPLEMENTATION_HANDOFF.md` and execute Prompts 1–8 sequentially. Implement code and tests, fix the issues you find, and deliver `DRCT_128_IMPLEMENTATION_REPORT.md` plus `COLAB_DRCT_128_TRAINING.md`. Do not run full training, reconstruction generation or real-data evaluation. Do not require a finished checkpoint to implement and test the architecture. Preserve the one-logit / 128D-feature contract and actual DRCT training requirements. Continue through all independent implementation work if an optional real-weight or GPU check is unavailable, and report that check as pending.
