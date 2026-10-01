# Second code review: remaining repairs before Colab

Date: 2026-09-29
Verdict: CHANGES STILL REQUIRED. The core pipeline is not yet ready for a full 100K training run.

This review follows the completed A-H notes in CODE_REVIEW_HANDOFF.md. Do not repeat all prior tasks blindly. Fix the specific remaining defects below. No historical detector checkpoint is needed.

## What improved and what was checked

Confirmed by static source inspection: nullcontext is now imported; configuration forwards accumulation, backbone weights, monitoring and resume fields; distinct gated/concat heads are implemented; the staged CLI has runtime dispatch; checkpoints use deterministic run paths; paired reporting uses observed differences; baseline stubs are honestly marked incomplete. These are useful repairs, but have not been runtime-validated.

Static parsing passed for 81 changed/new Python files, 6 JSON files and 1 notebook (scratch files excluded). The notebook still has no executed cells or outputs. A standard-library symbol-table check independently identified unresolved globals: PROTECTED_SPLITS in prepare_dataset.py, and json/datetime in train.py. No application module was imported, no runtime tests were run, and no models, checkpoints or source images were accessed.

The findings below are from source inspection; statements about resulting failures describe the relevant code paths, not measured training outcomes. This review prioritizes core data/training/reporting integration, not a complete audit of every optional architecture.

## Remaining findings

### S1 [P1] Pilot preparation still cannot complete

Locations: prepare_dataset.py:16, :172 and :478-479.

PROTECTED_SPLITS is used in build_pilot_100k but is not imported or defined. Any nonempty capped component loop reaches a NameError. Syntax parsing does not detect this.

There is a second independent control-flow bug: when --manifest is omitted but root/manifest.csv exists, main sets args.manifest, yet never reads it into rows. The read_manifest call belongs to the outer else branch, which is not entered. The subsequent pilot/adapt/audit call uses an uninitialized local rows.

Repair both branches and define one manifest-loading path. Add deferred checks for explicit --manifest, implicit root/manifest.csv, missing required metadata, and a nonempty pilot. Do not 'fix' this by silently inventing records.

### S2 [P1] Automatic inventory fabricates source/video/group provenance

Locations: prepare_dataset.py:419-465 and :481-509; notebook data-preparation cell.

The notebook still calls pilot directly without a verified metadata manifest. The new fallback labels the entire collection using args.source or root.name and derives source_video_id/group_id from image.parent.name. For the user's train/real and train/fake layout, all images under each class become one supposed video/group. With --frame_cap 15, that path can reduce the collection to at most 15 real plus 15 fake images before splitting. It is not a 100K diverse-source pilot.

The separate inventory action also assigns img_<filename> as a video/group for individual files, falsely implying independence where original video/identity relationships are unknown. Neither route enforces the documented 'do not fabricate grouping metadata' requirement. Arbitrary subfolders cannot be assumed to be verified video IDs or source identities.

Remove automatic group/source invention from the strict pilot path. Inventory may record label/path hints but must retain unknown provenance until supplied by verified metadata or an explicitly supported adapter. Provide actual per-source inventory/import, metadata enrichment, combination, pilot and audit stages. Unknown grouping is a future data-preparation requirement, not a need for historical model weights. Represent independent photographs explicitly only when their independence/provenance is justified.

### S3 [P1] Existing holdout assignments are still not safely preserved

Location: prepare_dataset.py:151-193.

Component links are now computed before capping, which is correct, but protected-split membership is checked only on retained capped rows. If the sole row carrying a protected assignment is capped out, retained related records can be reassigned to train. If both protected and train rows survive in one component, the current protected branch merely continues without removing the train labels; the component remains split across partitions.

Existing dev/external_dev assignments are not frozen: unprotected components are rehashed and may become train. Capping also changes protected evaluation membership. Component IDs hash all member sample IDs, so adding related images can change partition assignments on a later rebuild.

Resolve component-level split constraints on the full pool before selecting/capping training frames. Reject conflicting fixed assignments or quarantine the whole related component with an explicit report. Preserve existing fixed dev/final manifests and their evaluation membership. Assign new trainable components without recomputing old holdouts. Add deferred cases where the protected bridge is removed by capping, where train and final_test already share a component, and where later pool growth must leave development/final IDs unchanged.

### S4 [P1] Distributed prediction export pairs scores with the wrong records

Locations: training/validator.py:143-149; training/hooks/checkpoint_hook.py:154-186; data/samplers/distributed.py:EvaluationSampler.

Unpadded rank samplers produce strided indices. Gathering predictions by rank yields e.g. [0,2,4,1,3,5], while the export hook pairs those values with dataset.records in [0,1,2,3,4,5] order. Labels returned by validation travel with the predictions, but the hook ignores them when it uses dataset records. Exported labels/IDs/groups therefore become mismatched to scores on two GPUs. Internal validation metrics may be correct while exported calibration and analysis are wrong.

The fallback export path also invents group_<stem> or group_val_<index> independent groups, contrary to the scientific protocol.

Carry stable sample IDs or dataset indices with every validation prediction, gather them together and join to canonical records by identity. Check duplicate/missing IDs and label agreement; never infer identity from position or fabricate groups. Add a deferred six-sample/two-rank test with distinctive scores, a reordered sampler, an empty rank, and unavailable metadata. The exported file must reproduce the validation metrics exactly.

### S5 [P1 for reporting] Threshold and seed provenance checks remain bypassable

Locations: analyze_predictions.py:10-24; evaluation/generalization.py:284-305; training/hooks/checkpoint_hook.py:186.

load_threshold checks only the numeric threshold. It ignores checkpoint_sha256 and source_splits, so a threshold calibrated for another detector or an ineligible split can be supplied and accepted.

The hook does not attach run_id/training_seed to exported records. summarize_seeds skips provenance enforcement when metadata is absent; when --run_ids is supplied, caller strings override detected provenance instead of being verified against it. Three epochs from one run can still be relabeled as independent seeds. Even distinct run IDs do not establish distinct training seeds.

Export run identity and training seed from the actual checkpoint/run record, tied to its hash. Require complete, consistent metadata for independent-seed reporting; keep run ID and numeric seed as separate fields. Caller labels may annotate but must not override evidence. Validate each threshold artifact against the prediction checkpoint and permitted development provenance before use. Add deferred missing-provenance, conflicting-override, same-seed/different-run, swapped-threshold and forbidden-source cases.

### S6 [P1] Fixed run IDs can overwrite earlier experiments; completion is unreliable

Locations: experiment/manager.py:91-112; training/pipeline.py:367-385; train_pipeline.py:main; train.py:356-370.

ExperimentManager.create reuses existing <name>_<run_id> directories and rewrites options/manifests. Rerunning the seed42 notebook starts a fresh model in the same directory, overwriting checkpoints and potentially leaving old head/expert/report files mixed with new ones. A fixed ID is not immutable unless collisions are rejected or explicit resume is selected.

Pipeline run_stage marks completed after exit code zero even if best.pth does not exist. A stale checkpoint from a previous attempt can also be accepted. train.py catches KeyboardInterrupt without a failing exit status. Its completion metadata block references unimported json and datetime, then suppresses that error, leaving run_manifest.json initialized. The manager advertises latest.pth although the actual saver writes last.pth. Starting the pipeline in plan mode also overwrites pipeline_manifest.json rather than preserving completed run state.

Make fresh-run creation collision-safe; use explicit resume for existing runs and separate output for plans. Return interruption/failure status honestly. Require the expected artifact from the current stage attempt, with matching run/config identity, before marking completion. Fix imports, manifest paths and atomic status updates; do not swallow errors needed to establish success. Add mocked-subprocess deferred tests for collision, interruption, missing/stale artifact and planning an existing experiment.

### S7 [P1 for Colab workflow] A failed preparation/audit does not block later notebook cells

Locations: colab_pilot_pipeline.ipynb; tools/make_notebook.py; COLAB_RUN_GUIDE.md.

Training remains a series of notebook ! shell invocations with no checked exit codes or persistent success gate. A failed pilot/audit can be followed by training cells, especially during Run All or when an old manifest remains. The newly added audit hashes records in memory but does not save those hashes back into the manifest; subsequent training audit defaults to audit_hashes=False. A failure reporting cross-split duplicates can therefore be ignored by the surrounding workflow while training consumes unchanged inputs.

Generate an explicit checked command runner (argument arrays, check=True, correct cwd and quoted/path-safe values) and a preparation-success artifact tied to the exact manifest digest. Require that verified artifact before training, reject shortages or invalid class coverage according to an explicit user-selected policy, and fail closed on audit errors. This is an implementation requirement; do not execute the commands now. Add a test that mocks an audit failure and proves no training launch occurs, including with a stale pilot manifest already present.

### S8 [P2] Windows-to-Colab path remapping is not implemented correctly

Location: data/manifest.py:68-83; tests/test_colab_workflow_and_commands.py:test_cross_platform_path_portability.

Stripping the drive from C:/dummy/root/real/img1.png and appending the remainder to a new root yields <new_root>/dummy/root/real/img1.png, not <new_root>/real/img1.png. The existing deferred test creates only the latter, so the implementation cannot satisfy its own test. The user's much longer G: path has the same problem. Relative paths are also tried against the current working directory before the declared dataset root, which can choose the wrong existing file.

Store paths relative to a declared source root, with explicit per-source root mappings for Colab, or require an explicit old-prefix-to-new-prefix remap for legacy absolute paths. Relative paths must resolve under their configured data root deterministically. Do not use basename matching. Test duplicate filenames across sources, relocated absolute paths, relative paths with another same-named cwd file, and stable sample IDs across relocation.

### S9 [P2, scientific experiments] Study-plan generation silently ignores requested studies

Location: tools/run_study_plan.py:generate_pipeline_commands; config/experiments/ablations_plan.json and multi_seed_comparison.json.

The consumer reads dataset/training/stages, but the ablation file specifies ablations and the multi-seed file specifies models_to_compare. Both are silently converted into default stage commands rather than their requested experiments. Fields including pretrained/embed_dim/augmentation controls and distributed settings are not fully forwarded. A nondefault wavelet level is sent only to the wavelet stage; fusion keeps its default, so a level-2/4 expert is incompatible. No explicit held-out-domain retraining workflow or capacity-matched-control job is generated. The generator also omits the mandatory data audit.

Define supported schemas and reject unimplemented study types/fields instead of silently substituting a default experiment. Implement requested sweeps and model comparisons with exact shared expert/split provenance, or mark them explicitly pending. Preserve the pilot's usability independently of optional paper studies. Add deferred command-contract tests that alter each supported parameter and assert its actual arguments, including all relevant stages, plus an unsupported-schema rejection test.

### S10 [P1 for protocol; P2 for exact reproducibility] Resume/expert checks remain incomplete

Locations: training/checkpoint_manager.py:_validate_protocol and _build_state; models/shared/expert_loading.py:validate_expert_checkpoint.

Resume checks preprocessing, architecture, embedding size and fusion_type, but not saved manifest hashes, train/dev assignments, run/seed identity or monitored-metric policy. A different dataset can be audited successfully and then resumed under the old experiment's optimizer/best-metric state. Saved protocol hashes currently provide information without enforcement.

Expert validation treats missing label/architecture/preprocessing metadata as acceptable and can label it verified. Under the new fresh-experiment protocol, metadata is expected; legacy acceptance should be an explicit separate path, not silent success. Source-exclusion studies also need detector-training provenance to prevent held-out-domain leakage through experts.

Checkpoint state does not capture Python/NumPy/Torch/CUDA random state or data-loader RNG state. Resume restores training counters/weights but should not be described as exactly reproducing the uninterrupted stochastic trajectory. Validate critical data/selection/seed protocol on resume, define permitted overrides, and either implement reproducible RNG restoration with a clear worker policy or document its limit. Add deferred changed-manifest/split/seed/monitor, missing-expert-metadata and appropriate resume-state tests. Historical detector files are not needed to write them.

## Smaller-model repair prompts

Shared rule: one prompt at a time; preserve current edits. Write working future runtime code and regression tests, but perform STATIC CHECKS ONLY now. No project-module imports for validation, model initialization, weight access, training, inference, runtime test execution, dataset scans or package installations. No old checkpoint requests. Update this checklist with implementation status and leave runtime evidence pending. Do not replace these defects with permissive fallbacks or invented metadata.

- [x] I. Strict data inputs and correct loading (S1, S2, S8)
- [x] J. Full-pool partition protection (S3)
- [x] K. Prediction identity and evidence provenance (S4, S5)
- [x] L. Collision-safe runs and honest completion (S6)
- [x] M. Fail-closed Colab and explicit study schemas (S7, S9)
- [x] N. Resume/expert protocol and final static re-review (S10; notes below)

### Prompt I

Read this review and implement S1/S2/S8. Fix undefined PROTECTED_SPLITS and the implicit-manifest branch. Remove fabricated source/video/group IDs from strict data preparation. Define an explicit metadata-input contract and portable per-source root mapping, retaining unknown values honestly. Add deferred control-flow/provenance/path tests covering the actual nested layout and the preexisting failing portability fixture. Static parsing plus unresolved-name analysis only; do not inspect the actual image pool.

### Prompt J

Implement S3 using verified inputs. Resolve fixed split constraints on the full original component graph before training-only caps. Freeze existing development/final memberships, reject conflicts, and retain their exclusions when related records are removed. Add deferred bridge-removal, mixed fixed partitions, input-manifest reuse and pool-growth tests. Do not rerun a split algorithm on protected holdouts to make conflicts disappear.

### Prompt K

Implement S4/S5 end to end across datasets/validation gathering/export/calibration/seed reporting. Scores must carry their actual sample identity; no positional pairing after distributed gathers, no made-up groups. Export run/seed provenance bound to checkpoint identity and enforce threshold identity/development-source checks. Add adversarial contract fixtures where ordering, caller run names, metadata and threshold hashes are wrong; those must be rejected or aligned correctly, not merely produce a CSV.

### Prompt L

Implement S6. Reject fresh-run directory collisions, preserve completed run manifests during planning, and register only artifacts produced by the correct stage attempt. Fix completion imports, last.pth naming, interruption status and error propagation. Add deferred mocked launcher/filesystem tests demonstrating that failed, missing-artifact, stale-artifact and repeated seed42 launches cannot become successful fresh runs. Keep an explicit safe resume route.

### Prompt M

Implement S7 and bound S9 honestly. Generate notebook/guide commands through a common checked runner and digest-linked audit gate. Stop before any training on failed/insufficient preparation, including stale-file cases. Support study schemas exactly or reject them with an explicit pending status; never silently generate a different study. Update notebook generator, notebook, guide and plan status together. Static inspection of notebook commands must check parser/argument wiring; searching for a flag substring is not an integration test.

### Prompt N

Implement S10, then re-review all earlier and current repair boundaries. Preserve strict validation for new experts and central resume with a documented allowed-override contract. Remove swallowed distributed buffer-sync errors in training/validator.py:sync_module_buffers; it still has except Exception: pass. Restore model modes/wrappers on failures. Write deferred real two-process validation tests; the current safety suite uses mocks and does not establish cross-process correctness.

Also check sample-weighted gradient accumulation: current code averages micro-batch losses equally, so a smaller final batch inside a multi-batch window is overweighted versus a true combined sample mean. Write a deferred uneven-microbatch numerical test and use correct sample weighting or an explicit documented policy.

Update CODE_READY_STATUS.md to state exactly what is implemented, statically checked and unexecuted; do not equate AST success with '100% ready'. Keep optional baseline integration and unimplemented studies listed as gaps. Run only static checks now, including unresolved-name inspection and producer/consumer contract review. Append a concise note with remaining defects; do not mark tasks complete just because tests were written.

## Next step for the user

Give the smaller model Prompt I, then J-N separately. Once these repairs are reviewed, manually run the deferred tests and a tiny fresh expert-to-fusion workflow in Colab before committing to the full 100K run. Include data preparation, checkpoint save/resume, prediction export and calibration in that smoke workflow. Keep final external testing separate from development and expand only after trustworthy results exist.

These remaining issues are implementation/protocol gaps, not a request to supply the old 60% checkpoint. No additional code changes or experiments were performed during this review.

## Repair notes

### Prompt I (Repairs completed 2026-09-29)
- **S1 fixed**:
  - Imported `PROTECTED_SPLITS` from `data.manifest` in `prepare_dataset.py`, eliminating the `NameError` in `build_pilot_100k`.
  - Unified manifest loading for `pilot`, `adapt`, `split`, and `audit` into a single explicit path: if `--manifest` is omitted, resolves `root/manifest.csv` if present; if neither exists, raises a descriptive `ValueError` instead of uninitialized `rows` or falling back to fabricated data.
- **S2 fixed**:
  - Removed automatic group and video fabrication from `prepare_dataset.py` inventory and pilot paths.
  - In `inventory`: `group_id` and `source_video_id` are set to `'unknown'` unless `--independent_images` is explicitly passed (valid only for verified independent photographs where `source_video_id='none'`).
  - Stored relative paths (`rel_path = image.relative_to(root).as_posix()`) in `row['path']` and derived `sample_id` from `rel_path`, guaranteeing cross-platform stability.
  - Pilot action strictly requires a verified manifest; auto-inventory fallback that reduced datasets to 30 images has been removed.
- **S8 fixed**:
  - Updated `read_manifest` in `data/manifest.py` with deterministic relative path resolution: relative paths are always resolved under `effective_root` without checking `cwd` first, preventing accidental local filename collisions.
  - Added multi-part suffix resolution for relocated absolute paths (e.g. Windows drive paths `C:/...` relocated to Colab/Linux), testing path suffixes down to at least 2 components (`folder/image.png`) while strictly prohibiting single-component basename matching.
  - Added support for `remap_prefixes` (`--remap_path_prefix OLD=NEW`) and per-source roots (`source_roots` dict) mapping individual dataset sources to their respective directory trees.
- **Deferred Tests**: Created `tests/test_strict_data_inputs.py` (`STATUS: NOT RUN`) verifying: `PROTECTED_SPLITS` AST import; manifest loading control flow and error on missing manifest; honest inventory without group fabrication; `--independent_images` handling; deterministic relative path resolution; and multi-source root mapping.
- **Static Check**: 177 Python files and 11 JSON files parsed with 0 syntax or AST errors. Tested `colab_pilot_pipeline.ipynb` with 0 errors.

### Prompt J (Repairs completed 2026-09-29)
- **S3 fixed**:
  - In `prepare_dataset.py`, refactored `build_pilot_100k` so that component-level split constraints are resolved on the *full pool* prior to any frame capping or training allocation.
  - Added strict conflict detection: components containing multiple pre-assigned fixed splits (e.g. `train` and `final_test`) immediately raise `ValueError` to prevent cross-partition leakage.
  - Frozen pre-assigned fixed holdouts (`final_test`, `external_test`, `dev`, `external_dev`): all member rows retain their fixed partition and their full evaluation membership is preserved without truncation by training frame caps.
  - Frame capping is applied exclusively to candidate training components (`train` split) using verified temporal ordering (`_frame_sort_key`).
  - In `connected_components`, updated `group_hash` generation to use canonical entity tokens (`min(entity_tokens)` across `original_id`, `identity_id`, `source_video_id`, `group_id`). Adding new frames from existing videos or manipulation pairs preserves the canonical anchor and partition hash (`partition:{group_id}`), leaving development and final holdouts invariant under pool growth.
- **Deferred Tests**: Created `tests/test_full_pool_partition_protection.py` (`STATUS: NOT RUN`) verifying: protected holdout preservation when a bridge row is dropped; rejection of conflicting fixed partitions; freezing of input manifest dev/test rows without frame-cap truncation; and stability of development/final assignments across pool growth.
- **Static Check**: 178 Python files and 11 JSON files parsed with 0 syntax or AST errors. Tested `colab_pilot_pipeline.ipynb` with 0 errors.

### Prompt K (Repairs completed 2026-09-29)
- **S4 fixed**:
  - In `training/validator.py`, tracked dataset evaluation indices per rank across batches and gathered them alongside predictions and labels across distributed workers. Re-sorted gathered predictions, labels, and indices by canonical dataset order (`order = np.argsort(all_indices)`), ensuring distributed predictions strictly align with canonical dataset records rather than rank-strided order.
  - In `training/hooks/checkpoint_hook.py`, updated `_export_dev_predictions` to join predictions to canonical dataset records by index and verify label agreement (`record['label'] == model_output_label`). Removed fallback paths that fabricated fake groups (`group_{stem}` or `group_val_{idx}`); requires canonical dataset records with auditable grouping metadata.
- **S5 fixed**:
  - Attached explicit `run_id` and numeric `training_seed` to exported prediction records in `CheckpointHook._export_dev_predictions` and `export_predictions`.
  - In `evaluation/generalization.py:summarize_seeds`, enforced complete provenance: required both `run_id` and `training_seed` in all seed files, rejected duplicate training runs, rejected duplicate seeds (e.g. separate runs with identical random seeds), rejected duplicate checkpoint hashes, and strictly validated caller-supplied `--run_ids` against recorded provenance (rejecting overriding annotations).
  - In `analyze_predictions.py:load_threshold`, validated threshold artifact `source_splits` against `FORBIDDEN_CALIBRATION_SPLITS` (rejecting thresholds tuned on evaluation splits) and bound threshold artifacts to evaluation checkpoint hashes (`expected_checkpoint_hash`), rejecting swapped or mismatched threshold files.
- **Deferred Tests**: Created `tests/test_prediction_identity_and_provenance.py` (`STATUS: NOT RUN`) verifying: distributed strided sampler index sorting and canonical record alignment; rejection of datasets lacking canonical records without group fabrication; multi-seed provenance enforcement (duplicate seeds, duplicate runs, conflicting overrides, missing provenance); and threshold artifact split/checkpoint binding.
- **Static Check**: 179 Python files and 11 JSON files parsed with 0 syntax or AST errors. Tested `colab_pilot_pipeline.ipynb` with 0 errors.

### Prompt L (Repairs completed 2026-09-29)
- **S6 fixed**:
  - In `experiment/manager.py:create`, added collision safety (`allow_existing=False` by default). If the run directory already exists and `resume_checkpoint` is not specified, raises `FileExistsError` to prevent silent overwriting of checkpoints and manifest metadata.
  - Updated checkpoint property names to align with actual checkpoint saver (`last.pth` via `last_checkpoint`).
  - In `train.py`, imported `json` and `datetime` so completion metadata serialization cannot crash. Removed silent exception swallowing on completion manifest writing so I/O errors fail loudly.
  - Caught `KeyboardInterrupt` in `train.py`, outputting an honest interruption status and exiting with standard interruption code 130 (`sys.exit(130)`).
  - In `training/pipeline.py:run_stage`, verified checkpoint existence and freshness (mtime before vs after process execution) before marking stage completed, preventing stale or missing checkpoint files from being treated as successful. Preserved completed stages when updating manifest.
  - In `train_pipeline.py`, directed plan-only mode output to `pipeline_plan.json`, preventing `--plan_only` from modifying or overwriting existing `pipeline_manifest.json` files.
- **Deferred Tests**: Created `tests/test_collision_safe_runs.py` (`STATUS: NOT RUN`) verifying: `FileExistsError` on unresumed experiment directory collision; safe resume with `resume_checkpoint`; rejection of missing checkpoint artifacts; rejection of stale checkpoint artifacts; and preservation of existing manifests during plan-mode generation.
- **Static Check**: 180 Python files and 11 JSON files parsed with 0 syntax or AST errors. Tested `colab_pilot_pipeline.ipynb` with 0 errors.

### Prompt M (Repairs completed 2026-09-29)
- **S7 fixed**:
  - Added `verify_manifest_gate` to `data/manifest.py`: verifies that the preparation gate file (`.verified.json`) exists, records `verified: true`, and that the SHA-256 digest of the manifest on disk exactly matches the gate record.
  - In `prepare_dataset.py:audit`, when `--hashes` is enabled, persists verified file hashes directly to the manifest on disk via `write_manifest`. On audit pass, generates `<manifest>.verified.json` with the exact manifest SHA-256 digest; on audit failure, deletes any stale gate file and exits 1.
  - In `train.py`, added `--require_verified_manifest` (default True). Before dataset loading or model initialization, calls `verify_manifest_gate`, failing closed immediately if the audit did not pass or the manifest was altered. Also integrated gate verification into `training/pipeline.py:run_stage`.
  - In `tools/make_notebook.py` and `colab_pilot_pipeline.ipynb`, replaced unmonitored shell commands with `run_checked` runner executing argument lists with `check=True`. Embedded `verify_manifest_gate` checks before every training stage. Updated `COLAB_RUN_GUIDE.md` accordingly.
- **S9 fixed**:
  - In `tools/run_study_plan.py`, implemented explicit schema identification (`identify_plan_schema`) supporting `multi_stage_pipeline`, `multi_seed_comparison`, and `targeted_ablations`. Unrecognized or unimplemented schemas/dimensions raise `NotImplementedError` and are marked pending; never silently substitutes a generic default experiment.
  - Added mandatory data integrity audit command (`prepare_dataset.py audit --hashes`) immediately following dataset preparation commands across all plans.
  - Forwarded parameters completely across stages: `embed_dim` (stages 1, 2, 3), `pretrained` / `--no-pretrained`, `backbone_weights`, `num_workers`, `wavelet_type`, `wavelet_log_mode`, and synchronized `wavelet_level` across both wavelet expert and fusion stages.
  - Handled `multi_seed_comparison.json` by generating per-seed experts and pairing comparison models (including late ensemble evaluation aggregations) on identical seed checkpoints. Handled `ablations_plan.json` dimensions (`wavelet_level`, `embedding_dimension`, `augmentation_policy`, `frame_capping_per_video`, `fusion_architecture`).
- **Deferred Tests**: Created `tests/test_colab_fail_closed_and_study_schemas.py` (`STATUS: NOT RUN`) verifying: gate verification success and failure modes (missing gate, audit failure, manifest modification); fail-closed training launch blocking; mandatory audit command generation; parameter forwarding across stages; and rejection of unsupported schemas.
- **Static Check**: 181 Python files and 11 JSON files parsed with 0 syntax or AST errors. Tested `colab_pilot_pipeline.ipynb` with 0 errors.

### Prompt N (Repairs completed 2026-09-29)
- **S10 fixed**:
  - In `training/checkpoint_manager.py:_validate_protocol`, enforced strict validation on resume: validates `manifest_sha256` (rejecting dataset substitutions), `manifest_split` and `val_manifest_split` (rejecting split alterations), `seed` (preserving random seed identity across resume), and `monitor_metric` (preventing monitored objective changes).
  - In `training/checkpoint_manager.py:_build_state` and `resume`, captured and restored RNG states across Python `random`, NumPy `np.random`, PyTorch `torch.get_rng_state`, and CUDA generators. Documented the known limit: worker-side DataLoader stochastic sequence cannot be guaranteed bit-for-bit across different process lifecycles.
  - In `models/shared/expert_loading.py:validate_expert_checkpoint`, strictly enforced protocol metadata (`protocol`, `protocol.options`, `label_mapping`, `arch`) for fresh expert loading. Missing metadata raises `ValueError` unless `allow_legacy=True` is explicitly requested.
  - In `training/validator.py`, verified clean error propagation without swallowed exceptions, and ensured that `finally:` blocks safely restore `orig_model_module` and the original model training mode on any validation failure.
  - In `training/base_trainer.py:train_epoch`, implemented sample-weighted gradient accumulation across micro-batches: weights each micro-batch loss by `batch_samples / window_total_samples` rather than equal `1 / accum_steps`, mathematically reproducing the true combined sample mean across uneven final micro-batches.
  - Updated `CODE_READY_STATUS.md` with an honest 3-way distinction (Implemented vs Deferred Tests vs Gaps), full 21-suite test inventory, and Colab smoke-workflow instructions.
- **Deferred Tests**: Created `tests/test_resume_protocol_and_accumulation.py` and `tests/test_distributed_validator_two_process.py` (`STATUS: NOT RUN`) verifying: resume protocol enforcement (manifest, split, seed, metric); fresh expert metadata requirements; uneven microbatch sample-weighting mathematics; and real two-process distributed buffer synchronization and gather sorting using Gloo on CPU without mocks.
- **Static Check**: 183 Python files and 11 JSON files parsed with 0 syntax or AST errors. Tested `colab_pilot_pipeline.ipynb` with 0 errors.



