# RGB review fixes — 2026-10-08

The follow-up review found additional integration bugs after the earlier repair pass. They were fixed directly in the working tree. This is **source-reviewed, runtime verification pending**; it is not evidence of improved unseen accuracy or two-GPU scaling.

## Main fixes made in this review

1. **RGB policies were discarded by configuration conversion.** ModelConfig and config_to_opt now preserve head type, dropout, fine-tuning policy, backbone LR multiplier, BatchNorm policy and decay policy. Previously R2/R3 commands could silently run the default full model.
2. **Explicit augmentation settings were overwritten.** CLI provenance preserves zero-valued overrides, resolution is idempotent, the missing copy import is fixed, and saved/programmatic values remain authoritative. Historical RGB dropout no longer inherits the unrelated fusion dropout.
3. **Parity verification was not sufficiently independent or bounded.** The real-checkpoint diagnostic now constructs the production validation and evaluation loaders, runs Validator and InferenceRunner, compares tensors/logits/probabilities and checks selected image content/checkpoint/label identities. Only selected image paths are probed. Manifest metadata can still be read in full. Synthetic mode remains a synthetic contract check.
4. **Production performance instrumentation was disconnected.** Normal training now records startup preflight, loader construction, first-batch/data wait, model initialization, training, validation/metrics and checkpoint/export timing. CUDA events sample at most 64 phase calls per epoch, synchronized at epoch boundaries. Reports distinguish CPU wall/enqueue time from sampled GPU duration. These are measurements to collect, not a speedup claim.
5. **The benchmark duplicated a divergent training loop and crashed when printing its report.** It now uses BaseTrainer.train_step, the chosen RGB recipe/policy, exact sample weighting for partial accumulation windows, warmup gradient clearing, asynchronous utilization sampling, per-rank measurements and guaranteed runtime cleanup. Validation uses the safe production Validator with an exact global sample cap. Its random initialization and one-window buffering are disclosed limitations.
6. **Metadata was repeatedly checked during loader construction.** Training reuses rows already checked in that process, and removes temporary metadata references before worker creation/config persistence. Single-GPU device selection now also selects the CUDA current device; DDP wrapping supports CPU test fixtures.
7. **Resume could silently change the experiment or guess optimizer-state ordering.** Resume validates model, augmentation, optimizer, scheduler and selection policies. Historical one-group full-model optimizers restore their original groups/LR/decay; arbitrary layout changes fail. Group names/order are checked on direct restore.
8. **Calibration overstated its evidence.** Missing reference files and unsupported precision certification fail. Group/video/original/identity/hash overlaps are checked. Unbound reference files only establish an overlap check, not certified independence; artifacts say so. Both threshold consumers reject absent/forbidden split provenance.
9. **Prediction export could silently fail or fabricate groups.** Failures now reach every distributed rank; canonical records are required. Exported precision comes from the actual validation path. Saved-best writing remains a single finalized save linked to predictions.
10. **Validation precision settings could be ignored.** Options now reach the validation context; the result records effective precision. Standalone evaluation is explicitly FP32, overrides saved worker/batch settings and rejects a threshold known to use another precision.
11. **Source monitoring and distributed error handling had gaps.** Missing source values are checked, worst-source monitoring uses the readiness checks, and rank-zero metric exceptions propagate before peers wait on a broadcast. These checks validate declarations; they cannot verify invented provenance.
12. **Representative subset selection used costly repeated row scans and could exceed quotas.** Component membership uses a set, quotas/class targets are caps, row-order invariance is preserved, and shortfalls remain explicit. Content-label contradictions are now caught by metadata audit. Plateau construction no longer relies on the obsolete verbose keyword.

## Verification actually performed

- Parsed all 74 changed/untracked Python files with the standard-library AST parser; no application modules imported.
- Parsed the guide's Python setup cell and checked documented command flags against parser declarations.
- Reviewed configuration forwarding, checkpoint lifecycle, calibration consumers, distributed validation and benchmark control flow.
- Added tests/test_rgb_review_final_repairs.py with CLI/config persistence, explicit zero augmentation, legacy optimizer state/hyperparameters, quota/component preservation, calibration evidence and resume-policy regressions. Updated the earlier calibration test to require honest evidence status.

**Not executed here:** pytest, synthetic model creation, checkpoint loads, inference, training, image scans, CUDA benchmarks or dependency installation. Existing claims of runtime tests passing were not independently verified. AST success proves syntax only.

## What remains before trusting a pilot

1. Run the bounded tests and synthetic/DDP checks in RGB_PILOT_RUN_GUIDE.md. Include the existing distributed validator and training correctness suites before relying on resume/DDP results.
2. Benchmark one and two GPUs with the same effective batch and recipe; choose worker/batch settings from measurements. Both GPUs can participate without being continuously busy when CPU/image loading or validation dominates.
3. Run a small RGB pilot and inspect opt.json to verify the intended policy was actually saved. Preserve linked checkpoint, predictions and threshold artifacts.
4. Evaluate on external development with a fixed development threshold, reporting AUC, balanced accuracy and both class recalls. Reserve a fresh untouched final test.
5. The recovered diffgan-only metadata supports an engineering comparison, not proof of source-held-out generalization. Verified source annotations and a defensible held-out cohort still require dataset work. Independent calibration is explicitly unverified until reference membership is bound to the actual checkpoint's training/selection cohorts.

No model/checkpoint/dataset contents were modified. No training was launched. The next runnable steps are in RGB_PILOT_RUN_GUIDE.md; do not rerun full inventory merely to use these code changes.
