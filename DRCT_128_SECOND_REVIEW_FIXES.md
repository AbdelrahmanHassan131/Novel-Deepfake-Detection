# DRCT-128 second review: remaining fixes before training

Review date: 2026-10-14. Reviewed the implementation accompanying `DRCT_128_REVIEW_RESPONSE.md`.

## Verdict

**Improved, but the documented Colab workflow is not ready.** Preserve the one-logit binary output and learned 128-dimensional feature for MHA. These findings concern implementation correctness; they do not predict achievable unseen-domain accuracy.

This review adds only this Markdown document. Do not treat the previous response's blanket “all resolved” statement as acceptance.

## Verification and progress

- Independently reran all seven DRCT suites plus `test_staged_pipeline_and_fusion_heads.py`, `test_resume_protocol_and_accumulation.py`, and `test_rgb_contract.py`: **84 passed, 5 warnings** (including CPU DDP).
- The initial sandbox invocation had temporary-file rename permission failures; the successful run was outside those restrictions. Those permission failures are not code defects.
- Missing requested splits now fail, basic contradictory labels and cross-group splits are rejected, embedded projection activation/dropout is preserved, expert state loading is stricter, and input gradients/unfrozen encoder optimization are improved.
- Additional small CPU checks reproduced the failures below. No real dataset inference, downloads, diffusion generation, or GPU training was performed. This was not the full repository test suite.

| Additional check | Observed result |
|---|---|
| Folder evaluation with a saved `val_manifest` and a matching requested split | Loaded images from the saved manifest instead of the requested folder. With the usual train/val manifest and default `external_test`, it instead fails because that split is absent. |
| Legacy JSON supplies all required preprocessing keys under `options` | Loader still reports all those keys missing. |
| Same content hash and label in train/val, different group IDs | `validate_drct_records` returns `passed=True`. The canonical audit may reject it later; DRCT validation alone does not. |
| Native reference import missing encoder bias | Accepted with `missing_keys=['encoder.proj.bias']`. |
| Encoder initialization from `visual.*` into a wrapper holding `.visual` | Fails: missing `visual.proj.weight` and `visual.proj.bias`. |
| Fresh pretrained CLIP construction with the documented defaults | Raises: pretrained weights requested with `allow_download=False` and no local weights path. |
| Contrastive loss for embeddings `[[1,0],[1,0],[0,1]]`, all labels zero | Returns `0.9428093433`; exact normalized Euclidean/nonzero-pair mean is `1.4142135624`. |

## Instructions for the implementing agent

Repair the following tasks in order. Preserve unrelated working changes. Add focused tests that fail before the fixes. Use tiny local images, dummy or injected encoders, and mocked downloads. Do not download large weights, generate real reconstructions, launch GPU training, or evaluate the user's external collection during implementation.

### S1 — P1: Make preparation output consumable by the actual training entry point

**Locations:** `data/drct_preparation.py:27-98, main/audit`; `train.py:354-412`; `COLAB_DRCT_128_TRAINING.md`, Cells 4/7/11.

Directory inventory creates `train` and `val`, whereas training defaults to `val_manifest_split=dev`. Preparation's audit prints readiness but does not write the `.verified.json` evidence required by the default training gate. Even after fixing those, inventory assigns the same dataset source to train/val while the documented training command does not explicitly allow that engineering source overlap. Fusion Cell 11 supplies neither a manifest nor the folder-training opt-in and stops immediately.

**Prompt:** Connect DRCT preparation to the existing canonical hash/group audit and its manifest-digest-bound verification artifact. Do not disable the gate or manufacture success evidence. Standardize or explicitly pass split names. Preserve genuine source metadata; for the user's same-source pilot, explicitly declare the engineering overlap flags instead of claiming held-out-source development. Supply valid train/dev manifests to standalone and fusion commands. Show counts and refuse empty splits.

**Acceptance:** A CPU subprocess test prepares a tiny train/dev cohort with authentic different images, runs the real audit, and reaches a bounded dummy-model training/validation/checkpoint cycle using the documented parser arguments. Changed manifest bytes, cross-split duplicates, and missing dev data must fail. Also exercise the full-DRCT readiness rejection through `train.py`, not only the helper.

### S2 — P1: Provide a working pretrained CLIP initialization path

**Locations:** `models/drct/trainer.py:29-53`; active `CLIPViTBackbone` in `models/drct/backbone.py:93-142`; training/configuration options; benchmark model construction.

Training defaults to `pretrained=True`. The trainer reads `allow_download`, which is not exposed/propagated through the training configuration, and the guide supplies no `backbone_weights`. Thus construction fails even after preparation is fixed. The benchmark currently omits pretrained settings and measures a randomly initialized frozen encoder, unlike intended training.

**Prompt:** Choose and document a complete initialization route: explicitly download/cache the authentic weights and pass their verified local path, or add an explicit download option through parser/config/trainer. Preserve architecture-only offline checkpoint restoration. Propagate the same initialization settings into the real-backbone benchmark and record their provenance. Do not solve this by silently training a frozen random encoder. Remove the shadowed earlier `CLIPViTBackbone` definition to avoid maintaining two contradictory implementations.

**Acceptance:** Mock the CLIP module with both `clip.load` and `clip.model.VisionTransformer`. Verify fresh pretrained initialization, local cached initialization, forbidden download failure, and complete-checkpoint restore with no download. Test configuration round-tripping. Existing tests that accept `ImportError` from an incomplete mock do not establish this behavior.

### S3 — P1: Stop saved training manifests from overriding evaluation data

**Locations:** `evaluation/checkpoint_loader.py:246-280`; `data/datasets/drct_dataset.py:31-36`; `evaluation/evaluator.py:_build_dataloader`.

The loader clears `opt.manifest` but retains saved `val_manifest` and `drct_manifest_path`. DRCTDataset now falls back to those during evaluation. An explicit external folder can therefore fail on old paths/splits or evaluate the old cohort when the split exists. Printed dataset-root text does not prove which files were read.

**Prompt:** Separate persisted training provenance from active evaluation input. Clear all active training/validation manifest references during checkpoint restoration, then apply only explicit evaluation overrides. Folder evaluation must read the requested folder; manifest evaluation must use only the requested manifest/split. Preserve saved provenance separately for reporting. Do not silently fall back between these modes.

**Acceptance:** Save a dummy checkpoint with valid internal manifest paths, evaluate a separate folder with disjoint image IDs, and assert every returned path/ID belongs to the external folder. Repeat with unavailable old paths, with explicit external manifest, and with an absent requested split. No real user images are needed.

### S4 — P1: Fix the legacy-loader regression

**Location:** `evaluation/checkpoint_loader.py:247-260`.

`saved = protocol.get('options', {})` executes before reading legacy JSON. After `protocol = json.load(stream)`, `saved` is never refreshed. Both key validation and option restoration use the old empty dictionary, rejecting valid legacy configurations for Wang and other models.

**Prompt:** Read the legacy protocol, validate its structure, refresh the saved options, and apply architecture-specific requirements to that dictionary. Retain original label mapping and preprocessing without guessed defaults.

**Acceptance:** A checkpoint without protocol loads with a complete matching legacy config, restores nondefault preprocessing, and fails with a missing required field or invalid mapping. Include Wang and DRCT cases without downloading encoders.

### S5 — P1: Finish reconstruction preparation and provenance checks

**Locations:** `data/drct_preparation.py:27-145,156,249-250,371,478-509`; `data/drct_manifest.py:validate_drct_records`.

There are two `import_precomputed_reconstructions` definitions: the later one shadows the earlier one and only imports real reconstructions. Readiness computes fake reconstructions but never verifies them. The CLI `prepare` still supplies no generator callback, so it cannot generate missing images. Filename-stem matching can associate a reconstruction with the wrong parent; directory IDs omit class and extension (`split_stem`), causing collisions. Inventory scans only immediate class-folder files and can silently omit nested sources. Missing split folders can fall back to the root and inventory the same files twice. The same-hash cross-split case is still accepted by the DRCT validator. Source-manifest conversion also discards video/identity/original/generator provenance.

**Prompt:** Consolidate the import implementation. Use an explicit parent-ID-to-reconstruction mapping with revision/settings/seed/content hashes; reject ambiguous matches. Make directory IDs collision-safe, scan intended nested folders, and fail on missing required partitions rather than duplicating the root. Preserve original grouping/source fields and connect records by hashes and provenance across splits. Verify reconstruction kind against parent kind and verify recorded hashes against imported files.

Define what “full DRCT” means and enforce its declared reconstruction coverage, including fake reconstructions if claiming the full four-kind recipe. Label any reduced recipe as an ablation. Either provide a usable, resumable reconstruction backend command with explicit model/settings or state an actionable precomputed-reconstruction prerequisite and fail before training when absent. A no-op `prepare` reporting success is not a generation workflow. Use nonzero CLI status for failed requested readiness.

**Acceptance:** Test duplicate stems across classes/extensions, nested folders, missing val, ambiguous imports, changed same-path reconstructions, wrong parent kind, cross-split same-content duplicates, preserved identity/video relationships, and missing/corrupt fake reconstructions. Mock generation for CPU tests; do not perform real diffusion generation. Remove duplicate function definitions.

### S6 — P1: Finish strict official checkpoint imports and encoder-only mapping

**Location:** `models/drct/checkpoint_io.py:186-189,230-260`.

Native import rejects missing keys only if they contain `weight` or belong to the heads. Encoder biases and CLIP tensors such as positional/class embeddings and projection parameters can remain randomly initialized. Unexpected keys are not rejected. Encoder-only initialization strips `visual.` even when the destination wrapper needs it, and rejects a native-width projection although encoder-only mode should deliberately ignore the head.

**Prompt:** Validate the complete required destination visual state, not substrings of tensor names. Reject all unexplained missing/unexpected keys; explicitly allow only verified text-side omissions for the selected upstream format. Map source prefixes to the actual target wrapper. In explicit encoder-only initialization, discard the official projection/classifier while retaining fresh 128D head weights; never truncate them. Keep full detector restore and encoder-only initialization distinct. Provide the promised native reference entry point or mark it pending rather than ready.

**Acceptance:** Use a CLIP-shaped tiny wrapper with a bias, class/positional embeddings, and a parameter named `proj`. Removing any one must fail full import. Valid `visual.*`, `model.model.visual.*`, and local encoder layouts must load identically. Encoder-only import from a native-width detector must restore the encoder and leave the new head unchanged. Compare imported outputs against a separate source-layout fixture, not only the algebraic sigmoid/softmax identity.

### S7 — P2: Preserve exact zero distances in the contrastive reducer

**Location:** `models/drct/loss.py:52-70`.

Clamping squared distances to `1e-12` turns identical positive embeddings into a positive violation. That adds a zero-distance pair to the nonzero reducer denominator, materially reducing loss. For the three-embedding example above the result is approximately 0.9428 instead of 1.4142; this is not just a 1e-6 rounding difference.

**Prompt:** Use a distance calculation with stable gradients and exact zero for identical vectors. Match the intended normalized Euclidean, unsquared hinge, separate nonzero positive/negative reduction. Preserve finite gradients at zero and singleton/homogeneous safety.

**Acceptance:** Add duplicate, all-identical, zero-vector, mixed-violation and ordinary random cases. Compare values and gradients to the pinned PML implementation where available, plus independent exact cases. Keep the already-correct 0.8270503283 example. State which dependency comparison actually ran; do not label a hand-written arithmetic test as library parity.

### S8 — P1: Repair the remaining Colab evaluation, calibration and recovery cells

**Location:** `COLAB_DRCT_128_TRAINING.md`, especially Cells 7-11.

- Cell 9 looks for `EVAL_DIR/predictions.csv`, but single-model evaluation writes under `EVAL_DIR/DRCT_128/`.
- It leaves the evaluation split at the default `external_test`; calibration rejects test-tagged rows. Use an explicit internal dev manifest/split, not relabeling external predictions.
- It describes logit calibration but omits `--score_space logit`, whose default is probability.
- Backup runs only after blocking `subprocess.run(..., check=True)` finishes. Runtime loss during training leaves no newly mirrored checkpoints, despite the interruption-resume instructions.
- Backup copies only PTH files, omitting authoritative dev predictions, configuration, audit evidence, logs and run identity. Resume reconstructs a partial command with defaults instead of the original configuration, and the resume cell does not back up its new output.
- Fresh-session preparation inventories again instead of restoring a saved preparation bundle. The dependency cell says “pinned” while specifying an unbounded lower limit and the CLIP default branch.

**Prompt:** Rewrite the guide only after S1-S7 are working. Use a persistent run descriptor containing exact commit, manifests and digests, model initialization, all training settings, experiment paths, and current checkpoint. Provide explicit fresh-run versus resume branches. Restore preparation from Drive without rehashing unchanged data unnecessarily, while enforcing identity checks. Implement stable periodic checkpoint/artifact copying before launching training and final copying on exit; never copy an in-progress checkpoint. Restore and validate resume configuration, and require meaningful saved outputs rather than accepting an empty glob.

Use the actual output directory returned by evaluation, explicit internal dev metadata, FP32/logit prediction exports and checkpoint-bound logit calibration. Back up the threshold alongside its matching best checkpoint. Include the corrected standalone and MHA manifest paths/source settings. Document the reconstruction prerequisite before the GPU session. Pin or record exact dependency revisions accurately.

**Acceptance:** Parse every command, then run a tiny CPU workflow through preparation, training, evaluation, calibration, interrupted backup/resume, and new-session restore with a dummy encoder. A parser-only check is insufficient. Assert checkpoint hash, prediction split/path, score space, and exact resumed settings. No real training budget is needed.

### S9 — P2: Close the remaining preprocessing/contract evidence gaps and update the report

**Locations:** `data/transforms/drct_transforms.py`; `data/datasets/fusion_dataset.py:49-67`; `DRCT_128_REVIEW_RESPONSE.md`; `DRCT_128_IMPLEMENTATION_REPORT.md`.

The transforms still claim upstream training parity without a pinned upstream tensor comparison. Fusion inspects an external checkpoint to detect its architecture but does not use that checkpoint's saved crop/preprocessing configuration when building the RGB transform; it uses embedded metadata or a default. Basic 224-shape tests cannot demonstrate equivalence. Native CLIP positional embeddings also require explicit handling/rejection of unsupported input sizes.

**Prompt:** Define and version the adaptation preprocessing and distinguish it from a verified native-reference recipe. Resolve external and embedded RGB expert preprocessing from the same saved contract, validate supported input sizes, and check standalone versus MHA tensor/feature parity using non-square images. Do not claim paper reproduction merely because the tensor has the expected shape. If an upstream recipe remains unverified, mark it pending and keep it out of readiness claims.

Update the response with an honest finding-by-finding status and exact executed test commands/results. Several named tests in the current response do not match the functions in the repository; cite actual tests. Include S1-S9 integration evidence and remaining real-CLIP/GPU verification requirements. Return `DRCT_128_SECOND_REVIEW_RESPONSE.md` and the corrected Colab guide. Do not claim a particular generalization accuracy from unit tests.

## Message to give the implementing LLM

> Execute S1-S9 in DRCT_128_SECOND_REVIEW_FIXES.md. Preserve the binary logit/128D/MHA contract. Fix actual entry-point behavior, not only helper tests. Use tiny CPU fixtures and mocked downloads; do not start real training, full inference, or reconstruction jobs. Complete the end-to-end CPU acceptance workflow, update COLAB_DRCT_128_TRAINING.md, and report evidence in DRCT_128_SECOND_REVIEW_RESPONSE.md. Distinguish verified fixes from anything still pending.
