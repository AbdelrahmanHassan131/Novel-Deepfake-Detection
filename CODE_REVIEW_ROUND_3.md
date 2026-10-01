# Third review: focused remaining integration fixes

Date: 2026-09-29
Status: Some repairs remain. Do not launch the full 100K experiment yet.

This is a static re-review after Prompts I-N. Historical detector weights are not required. Only this review document was created; application code and datasets were not changed.

## Confirmed improvements

The undefined globals reported last time are fixed. Strict pilot input now requires a metadata manifest; the automatic fabricated-group fallback is removed. Default unpadded distributed validation now gathers and sorts indices with predictions. The exporter preserves canonical records and records run/seed metadata. Fresh-run collisions are rejected in single-process execution. Notebook subprocesses use check=True. Optional baseline gaps and unexecuted tests are documented more honestly.

Static parsing: 88 changed/new Python files, 6 JSON files and 1 notebook passed. Targeted symbol-table analysis found no unresolved global candidates in the inspected entry points and core modules. Notebook execution counts and outputs are empty. These checks do not establish runtime correctness. No project imports, tests, models, checkpoint reads, dataset scans, training or inference were executed.

## Findings requiring fixes

### T1 [P1] Different sources generate identical sample IDs

Location: prepare_dataset.py:506.

Inventory hashes only rel_path. Two independently inventoried sources both containing real/0001.png produce exactly the same sample_id. Combining the sources into the requested mixed-source pilot then fails duplicate-ID validation, or corrupts identity joins if a downstream merge deduplicates them.

Include an immutable verified source identifier in the ID input, together with the source-relative path; do not use a machine-specific absolute root. Preserve these IDs through copying, enrichment, splitting and export. Add a deferred test that combines two sources with identical relative filenames and verifies distinct IDs, plus relocation stability for each source.

### T2 [P1 for two GPUs] Gradient weights use the full dataset rather than local samples

Location: training/base_trainer.py:240-282.

train_epoch chooses len(train_loader.dataset) before considering the sampler. Under DDP the dataset is the full pool but num_batches is the per-rank loader length. For N=100, two ranks, batch=16, each rank has four batches of 16/16/16/2. The current formula estimates the last batch as 100 - 3*16 = 52 rather than 2. With accumulation=2, the final window is weighted 16/68 and 52/68 instead of 16/18 and 2/18. This changes the optimizer update substantially.

Use actual micro-batch sample counts for each accumulation window, or an accurately bounded rank-local sampler/batch contract. Handle drop_last, weighted/subset samplers and unequal final batches. Do not infer counts from the global dataset. Add a deferred update-equivalence test that actually calls the trainer; a standalone arithmetic test does not exercise this defect.

### T3 [P1 for two GPUs] Fresh-run collision detection races between ranks

Location: experiment/manager.py:97-106.

Every rank checks whether the shared experiment directory exists; only rank 0 creates it. On a fresh run, rank 0 can create the directory before rank 1 reaches the existence check, causing rank 1 to reject the directory created by its own distributed job as a collision. Other ranks can then be left waiting in collectives.

Rank 0 must atomically decide and create the fresh run, broadcast success/failure and paths, and synchronize before any rank uses the directory. Real preexisting-run collisions must still fail coherently on every rank. Do not swallow distributed setup failures. Add a deferred real two-process test with deliberate timing skew and a second launch against the same run ID.

### T4 [P1] Resume digest checks do not read the metadata actually saved

Locations: config/protocol.py:checkpoint_metadata; training/checkpoint_manager.py:267 onward; tests/test_resume_protocol_and_accumulation.py:_base_valid_checkpoint.

The writer saves protocol['manifests']['manifest']['sha256'] and protocol['manifests']['val_manifest']['sha256']. The reader looks for options['manifest_sha256'] or protocol['manifest_sha256']; neither is written by the normal pipeline. The check is skipped on real produced checkpoints. The new test creates those nonexistent flat fields itself, so it does not verify the writer-to-reader contract. A changed dataset may still resume using old optimizer/best-metric state. Separate development-manifest changes are also unchecked.

Validate both hashes using the actual schema, requiring expected metadata and readable current inputs. Allow explicit path relocation when the intended data identity is preserved; do not bypass identity checks silently. Write a deferred test that obtains metadata from checkpoint_metadata/_build_state, then changes the actual manifest contents and verifies rejection. Use the same path for unchanged-data acceptance. Do not handcraft a substitute checkpoint layout.

Additional resume limitation: torch.load maps all tensors to model.device, including the saved CPU torch RNG tensor. Restore that RNG tensor on CPU explicitly; otherwise GPU resume can warn and skip the remaining RNG restoration. Distributed RNG state needs a per-rank policy, not an unqualified exact-reproducibility claim.

### T5 [P1 for source-exclusion studies; P2 for pilot quotas] Holdout selection and frame caps changed meaning

Locations: prepare_dataset.py:181-205 and :229-241.

Pre-assigned train components are accepted and the loop continues before holdout_source/holdout_generator is checked. Thus asking to hold out a source in an already assigned manifest can leave that source in training. Fail explicitly on this conflict or implement a separate derivation that excludes all linked training records while preserving final evaluation partitions. Never label such a run leave-one-source-out unless experts exclude that source too.

The new frame-cap code caps the entire connected component, replacing the existing per-video implementation. A connected component can contain many videos and both original/manipulated records. Capping that whole component to 15 may discard most of a source or change class/generator coverage, despite the CLI/guide claiming 15 per video. Keep component assignment for leakage prevention and apply the documented training cap to each verified video inside eligible components. Decide separately whether a component-wide budget is wanted; do not silently substitute it.

Add deferred tests for a train-labeled held-out source, linked originals, and one component containing two videos with more than 15 frames each. Require the reported policy and selected counts to agree.

### T6 [P1 for trusting the training gate] The audit gate does not enforce all its claimed checks

Locations: data/manifest.py:178 onward; prepare_dataset.py:audit; train.py:256-259.

verify_manifest_gate checks the CSV digest and verified flag but ignores hashes_verified and its own enforce_class_coverage argument. An audit without --hashes can issue an accepted gate. Pilot shortages are not part of the gate, so a smaller balanced subset can be accepted without the user choosing to accept a shortfall. train.py checks only the primary manifest's gate even when val_manifest is a different file. Train.py does check class coverage later, but that does not supply the missing hash/shortage/development-audit guarantees.

Define explicit audit policy and bind its evidence to both training and development inputs. Make strict mode require the intended hash/overlap checks and class coverage. Report pilot target/actual counts and require an explicit allow-shortfall policy; tiny smoke runs should declare their small targets rather than pretending to be 100K. Verify a separate development manifest and cross-manifest overlap. Add deferred negative tests for each missing guarantee; do not merely add more boolean fields without checking them.

### T7 [P2, wrong-file risk] Absolute-path relocation still guesses which image to use

Location: data/manifest.py:read_manifest absolute-path suffix fallback.

The code tries progressively shorter suffixes until an existing file is found. Requiring two components does not establish identity: old/source_A/real/0001.png can silently resolve to new_root/real/0001.png from source B. Source-root mapping is optional, and the matched image is not checked against a known content hash by the loader. This is a silent data-selection error, not a safe portability guarantee.

Require explicit old-root remapping/per-source roots, or known content-hash verification with unambiguous candidates. For unresolved absolute paths, raise a useful error rather than selecting the first suffix match. Keep relative paths resolved under their declared root. Update portability tests to supply the correct mapping instead of rewarding guessing.

## Required user input before the eventual Colab smoke run

The notebook now requires DATA_ROOT/manifest.csv implicitly, but it does not create or enrich that verified source manifest. This is an expected data input, not evidence that the old checkpoint is missing. Add an explicit POOL_MANIFEST setting, pass --manifest in preparation, document its required columns and source-root mappings, and stop before expensive preparation when it is absent or incomplete. Provide the inventory/enrichment/combination instructions needed to create it honestly.

Use a persistent output location or an explicit copy-out step before long training. The current notebook defaults to /content/experiments and leaves Drive mounting commented out. Do not describe those defaults as persistent storage.

## Three bounded repair prompts

Do not restart earlier broad tasks or redesign the architecture. Preserve existing edits. Follow the current code-only rule: write implementation and deferred tests; perform static checks only. No model construction, weight access, training/inference, test-suite execution, data scanning, installs or historical-checkpoint requests. Runtime verification is for the user's later Colab phase.

- [x] O. Source identity, selection and audit input contract (T1, T5, T6, T7)
- [x] P. Distributed training correctness (T2, T3)
- [x] Q. Real checkpoint contract and final handoff (T4)

### Prompt O

Fix only T1/T5/T6/T7 and the documented pool-manifest input. Namespaced stable sample IDs, explicit relocation, per-video caps, source-exclusion checks and a genuinely enforced audit policy must form one consistent data contract. Update notebook/generator/guide together. Add deferred cases listed above and include a multi-source inventory-to-pilot-to-audit fixture that uses the actual production functions. Do not invent provenance or accept unknown groups to make it pass. Report unresolved input metadata separately from code defects.

### Prompt P

Fix T2/T3. Compute accumulation weights from actual local micro-batch samples and coordinate fresh experiment creation through rank 0 with coherent failure propagation. Write deferred numerical trainer and timed two-process tests, including uneven tails, drop_last, and preexisting-run rejection. Preserve current single-GPU behavior. Do not claim DDP verified until those tests are later executed.

### Prompt Q

Fix T4 using the checkpoint metadata actually written by the application, including separate validation-manifest identity and correct CPU RNG restoration. Review current tests for invented schemas, stale assumptions and assertions that only repeat a formula/string; update them to exercise the real producer/consumer boundaries. In particular, recheck older baseline tests against the current incomplete-stub statuses and prior integration fixtures against stricter expert/audit requirements. Do not weaken production safeguards to preserve outdated fixtures.

Complete a bounded static pass over O-Q changes and update CODE_READY_STATUS.md: implemented, statically checked, runtime NOT RUN, known optional gaps. Write exact valid manual smoke-test commands in a separate unexecuted smoke notebook/config, with small targets, one epoch, no pretrained downloads where supported, a unique run ID and isolated outputs. It must cover data audit, both experts, a fusion head, best-checkpoint export/calibration and resume. Avoid literal ellipses in executable commands. Do not run it now.

## What follows these repairs

The next phase should be actual small runtime verification, not another claim that parsing proves readiness. Start with the deferred relevant tests and a tiny single-GPU Colab run; then check two-GPU behavior separately if needed. Only after those pass should the user start the 100K experiment. Optional modern-baseline implementations and full reviewer evidence remain separate unfinished work.

This review does not predict a particular external accuracy. It checks whether the code can support a trustworthy experiment.
