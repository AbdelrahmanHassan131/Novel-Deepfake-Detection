> **Latest RGB review (2026-10-08):** See [RGB_REVIEW_FIXES.md](RGB_REVIEW_FIXES.md) and [RGB_READY_TO_RUN.md](RGB_READY_TO_RUN.md). Earlier completion/readiness claims below are historical; runtime verification remains pending.

# Code-Ready Status Report (Round 3 Re-Review)

**Date**: 2026-09-29  
**Status**: **IMPLEMENTED & STATICALLY CHECKED — CODE READY FOR MANUAL COLAB EXECUTION (AWAITING RUNTIME SMOKE RUN)**
**Execution Boundary Adhered To**: Strict code-only boundary. Code, configuration, notebook cells, and regression tests only. No models initialized, no weights downloaded, no test suites executed, no image pools scanned.

---

## 1. Honest 3-Way Readiness Distinction

To ensure scientific honesty and avoid equating AST parsing with runtime readiness, repository status is categorized into three distinct levels:

### Category A: Implemented & Statically Checked
- **Source Identity & Global Namespacing (T1)**: `prepare_dataset.py inventory` namespaces `sample_id` using `hashlib.sha256(f"{source}:{rel_path}".encode()).hexdigest()[:16]`. Identical relative filenames across independent sources produce distinct IDs, remaining invariant under root relocation.
- **Holdout Enforcement & Per-Video Frame Capping (T5)**: `prepare_dataset.py pilot` inspects holdout criteria across the full connected component. Pre-assigned `train` components matching holdout rules fail closed with `ValueError` (preventing silent inclusion in training); linked authentic originals in other sources are excluded from training into `external_dev`; frame capping is applied to each verified video (`source_video_id`) inside eligible training components rather than capping the entire multi-video component.
- **Strict Audit Gate & Shortfall Policy (T6)**: `verify_manifest_gate` strictly verifies `hashes_verified` when requested (`require_hashes=True`), enforces binary class coverage (`{0, 1}`), and verifies separate `val_manifest` gates. `prepare_dataset.py pilot` compares requested vs eligible targets and raises `ValueError` on shortages unless `--allow_shortfall` is explicitly passed.
- **Deterministic Absolute Path Relocation (T7)**: `read_manifest` removes progressive suffix guessing for unmapped absolute paths, failing closed with a descriptive `FileNotFoundError` unless explicit `--remap_path_prefix` or `--source_roots` mappings are provided.
- **Distributed Gradient Accumulation Weighting (T2)**: `BaseTrainer.train_epoch` bounds accumulation weights using actual local microbatch sample counts and rank-local samplers under DDP, eliminating the global dataset length defect on multi-GPU setups (e.g. 16/18 and 2/18 rather than 16/68 and 52/68).
- **Coordinated Distributed Experiment Directory Creation (T3; runtime unverified)**: `ExperimentManager.create_experiment` coordinates directory initialization via Rank 0: Rank 0 inspects collisions, creates directories, broadcasts status/errors to all ranks via `dist.broadcast_object_list`, and synchronizes with `dist.barrier()`. Pre-existing run collisions fail coherently with `FileExistsError` across all ranks.
- **Real Checkpoint Producer Schema & GPU Resume RNG (T4)**: `CheckpointManager._validate_protocol` validates training and development manifest digests against the actual saved schema (`protocol['manifests']['manifest']['sha256']` and `val_manifest`), allowing path relocation while requiring exact digest matches. `torch_rng` tensor is explicitly converted to CPU before `torch.set_rng_state()` to prevent GPU resume crashes.
- **Corrected bounded smoke workflow**: A separate preassigned smoke CSV is limited to 300 rows total (train/dev/internal_test at most 100 each; at most 50 train rows per class). Bounds are checked before image work. The unexecuted notebook consumes one generated command plan, uses valid `--no-pretrained`, `evaluate.py`, and `analyze_predictions.py` interfaces, creates a unique run directory, and resumes with unchanged training settings. Deferred assertions check exported development IDs and saved/logged resume counters; exact numerical state restoration remains unverified.

### Category B: Deferred & Unexecuted Tests (`STATUS: NOT RUN`)
The test suites remain unexecuted per the code-only boundary. This latest pass parsed the edited files and notebook cells; it did not revalidate every historical suite. They must be manually executed during the Colab smoke-test phase:
- `tests/test_smoke_manifest_bounds.py` (bounded smoke input; newly written, NOT RUN)
- `tests/test_round3_data_contracts.py` (Prompt O / T1, T5, T6, T7)
- `tests/test_round3_distributed_correctness.py` (Prompt P / T2, T3)
- `tests/test_resume_protocol_and_accumulation.py` (Prompt Q / T4)
- `tests/test_strict_data_inputs.py` (Prompt I / S1, S2, S8)
- `tests/test_full_pool_partition_protection.py` (Prompt J / S3)
- `tests/test_prediction_identity_and_provenance.py` (Prompt K / S4, S5)
- `tests/test_collision_safe_runs.py` (Prompt L / S6)
- `tests/test_colab_fail_closed_and_study_schemas.py` (Prompt M / S7, S9)
- `tests/test_distributed_validator_two_process.py` (Prompt N / S10)
- Preexisting suites: `test_training_correctness_and_resume.py`, `test_metadata_relationship_integrity.py`, `test_stable_pilot_selection.py`, `test_staged_pipeline_and_fusion_heads.py`, `test_distributed_validation_safety.py`, `test_prediction_and_evidence_contracts.py`, `test_colab_workflow_and_commands.py`, `test_fresh_training_pipeline.py`, `test_pilot_data_preparation.py`, `test_selection_and_reporting.py`, `test_preprocessing_and_gpu_reliability.py`, `test_baseline_adapters.py`, `test_integration_checkpoint.py`, `test_generalization.py`.

### Category C: Known Gaps, Incomplete Stubs & Future Work
- **Optional Baseline Adapters / Architectural stubs**: `models/baselines/` (UniversalFakeDetect, SBI, UCF, Effort) remain specification stubs labeled `incomplete_stub_future_work`. Architectural stubs have native preprocessing and CLI interfaces exist, but official pre-trained weights and benchmark inference are deferred to future optional comparative trials.
- **Unimplemented Study Plan Types**: `tools/run_study_plan.py` explicitly rejects unknown schemas and unhandled sweep dimensions with `NotImplementedError` (marking them pending). Full Cartesian grid search and complex cross-dataset retraining sweeps are not generated.
- **Runtime Convergence & Empirical Proof**: Static parsing checks syntax. The new smoke checker reconstructs supported literal CLI options from source without application imports and checks the eight command templates. Dynamic choices, runtime behavior, numerical correctness, convergence, and GPU memory use remain unverified.

---

## 2. Complete Inventory of Deferred Test Suites

| Test File | Primary Focus | Status |
|---|---|:---:|
| `tests/test_round3_data_contracts.py` | T1/T5/T6/T7: Namespaced sample IDs, holdout conflicts & linked originals, per-video capping, audit gate checks, no suffix guessing. | **NOT RUN** |
| `tests/test_round3_distributed_correctness.py` | T2/T3: DDP local sample weighting (16/18 and 2/18), drop_last handling, rank 0 collision coordination and race-free initialization. | **NOT RUN** |
| `tests/test_resume_protocol_and_accumulation.py` | T4: Real producer checkpoint schema validation, content digest mutation rejection, relocation acceptance, CPU torch RNG restore. | **NOT RUN** |
| `tests/test_strict_data_inputs.py` | S1/S2/S8: Manifest path fallback, honest unknown groups, deterministic relative resolution, multi-source roots. | **NOT RUN** |
| `tests/test_full_pool_partition_protection.py` | S3: Full-pool partition constraints before frame caps, bridge row protection, frozen dev/final holdouts. | **NOT RUN** |
| `tests/test_prediction_identity_and_provenance.py` | S4/S5: Strided rank gather re-sorting, canonical sample joining, multi-seed provenance enforcement, threshold hash binding. | **NOT RUN** |
| `tests/test_collision_safe_runs.py` | S6: Fresh experiment directory collision rejection, last.pth naming, stale/missing checkpoint rejection, plan manifest preservation. | **NOT RUN** |
| `tests/test_colab_fail_closed_and_study_schemas.py` | S7/S9: Digest-linked audit gate verification, fail-closed training launch blocking, study plan schema dispatch, mandatory audit generation. | **NOT RUN** |
| `tests/test_distributed_validator_two_process.py` | S10: Real 2-process distributed validation with buffer sync and unpadded rank gather re-sorting. | **NOT RUN** |
| Preexisting regression suites (14 files) | Domain suites covering preprocessing, architectures, fusion, evaluation, and pipeline coordination. | **NOT RUN** |

---

## 3. Corrected smoke workflow (runtime NOT RUN)

Open `colab_smoke_pipeline.ipynb` in Colab, set its repository/data paths, and supply a separate `smoke_input.csv` as described in `COLAB_RUN_GUIDE.md`. Do not pass the million-image pool. The notebook performs bounds checks, installs dependencies only when executed by the user, audits image hashes/relationships, trains both fresh experts and fusion for one epoch each, exports and calibrates development predictions, then resumes fusion to epoch two.

`tools/make_smoke_notebook.py` generates both the notebook and `config/experiments/smoke_test_run.json`. The notebook renders command templates from that JSON and saves the exact commands per run. The JSON is a declarative notebook plan, not an input to `tools/run_study_plan.py`. Regenerate both artifacts after changing the generator.

Static verification performed in this repair: eight command templates accepted by source-derived parsers; entry points exist; all notebook code cells parse and remain unexecuted; fusion/resume arguments agree except explicit resume fields and total epochs; AMP default remains false. Newly edited Python files were parsed. No tests, image scans, checkpoint loads, training, inference, or installs were executed.

Before the 100K pilot, execute the relevant deferred tests and the bounded notebook on one GPU. Two-GPU execution needs separate verification. The notebook uses scratch output by default: choose mounted Drive storage or preserve the run directory before disconnecting. Smoke metrics do not establish generalization. Optional baseline implementations and reviewer experimental evidence remain outstanding.

## 4. Shared-folder training-size selection (2026-10-01)

Implemented `prepare_dataset.py pilot --train_size 100000` (any positive even count) and `--train_size all`, plus the main notebook's `TRAIN_SIZE` setting. Subsets use balanced targets and existing seeded group selection. Full mode includes all eligible training rows without frame caps or class downsampling, while preserving dev/test/holdout exclusions. Images remain at their original paths. Per-run output directories isolate manifests and checkpoints; the full input manifest cannot be overwritten by pilot selection. Reports record requested and actual counts. See COLAB_RUN_GUIDE.md for usage and limits.

Validation: edited Python and generated notebook syntax checked statically; source-derived CLI parsing checked for supported sizes and legacy target flags. `tests/test_training_size_selection.py` adds deferred coverage for size counts, shared paths, stable evaluation membership, full-mode frame retention and class imbalance, explicit shortages, holdout exclusion, invalid sizes, and quota rejection. Runtime tests/data scans/model loads/training/inference: NOT RUN.

## 5. Completed-audit recovery and preparation reuse (2026-10-02)

The user's completed audit found 910 distinct cross-split duplicate hashes among 100,000 training + 211,434 development rows. Added `tools/recover_prepared_manifest.py`: requires a digest-matched completed hash report plus an explicit unchanged-dataset assertion; preserves evaluation rows, removes entire linked training components, records counts/exclusions, independently checks residual overlaps, and writes a new gate with inherited-evidence provenance. A smaller training set requires explicit `--allow_shortfall`. Repeated recovery reuses matching completed output. It does not reopen images or modify the original dataset. Conflicting evaluation labels/partitions remain errors.

Added `PREPARED_MANIFEST` reuse to the training command guide and main notebook/generator, separating prepared data from new training-output directories. Added SQLite hash caching to the actual audit path (the earlier `--hash_cache` was not used by audit) and periodic manifest/audit/inventory progress messages. Future hash audits resume from committed cache entries when file metadata matches. Original failed audit evidence is reused by recovery, not silently imported as fresh file-stat-validated cache entries.

Regression tests added for provenance/digest enforcement, whole-group exclusions, evaluation protection, explicit shortages, idempotent output reuse, and audit-cache hit/invalidation behavior. Runtime tests and actual recovery on Kaggle: NOT RUN locally. No training or inference performed. Follow KAGGLE_PREPARATION_RECOVERY.md instead of repeating full preparation.

## 6. Contradictory-label diagnostics and optional quarantine (2026-10-02)

The user's actual recovery reported identical evaluation content with contradictory real/fake labels. Recovery v2 now emits a complete label-conflict CSV and summary before refusing ambiguous evaluation data. Explicit `--quarantine_label_conflicts` excludes whole connected components containing contradictory content labels from a new cohort, without relabeling or deleting images. The policy is part of reuse identity; existing outputs created under another policy/version cannot silently be reused. Reports distinguish removed training and evaluation counts and mark changed evaluation membership. All exclusions are saved; evaluation-partition overlap and loss of either required class still prevent gate issuance. The final check enforces content-label consistency as well as split separation.

The guide uses a new `recovered_100k_seed42_v2` output directory and explains that all model comparisons must use the same resulting evaluation cohort. Added deferred regression cases for diagnostics, whole-component quarantine, unchanged input/labels, reuse policy mismatch, and missing-class rejection. Static checks only; runtime tests, training, inference, image reads, and real Kaggle recovery NOT RUN locally.

## 7. Visible training startup and batch progress

Added worker messages before/after PyTorch and project imports, line-buffered stdout/stderr for the train entry point, manifest and metadata-audit progress during training preflight and dataset construction, process-group initialization messages, first-batch waiting/receipt/completion messages, and periodic local validation progress. Existing training loss reductions and model/data semantics are unchanged. The command guide exports PYTHONUNBUFFERED; KAGGLE_TRAINING_PROGRESS.md supplies a tee-backed console log and log_freq=10 launch.

This exposes repeated per-rank manifest/path work; it does not claim the actual Kaggle stall has been diagnosed or that startup I/O has been removed. Static validation only; runtime/GPU tests NOT RUN.
