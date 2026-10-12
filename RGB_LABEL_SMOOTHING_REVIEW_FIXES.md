# RGB smoothing implementation review: fixes required before Colab

Review date: 2026-10-12. The implementation report's claim of readiness is premature. Preserve the existing implementation; correct the issues below before the user pushes and starts GPU training. No full training, checkpoint download or real-image inference is needed to perform these repairs.

This review inspected the actual modified code and notebook, reran focused tests, and reproduced failures using small synthetic prediction CSVs. Evidence is saved in `output/review/rgb_smoothing_review/reproduction_results.json`.

## Confirmed findings

### P1 — Colab commands cannot execute as written

`COLAB_RGB_LABEL_SMOOTHING_AB.md` lines 100, 143 and 180 use `--amp_train`, which is not a `train.py` option. Setup fails its flag assertion; bypassing that assertion causes argument parsing to fail. The real flag is `--use_amp`; explicitly add `--amp_dtype fp16` to both matched runs.

Lines 203 and 210 calibrate from `<run>/best_dev_predictions.csv`, but the checkpoint hook writes `<run>/checkpoints/best_dev_predictions.csv`.

Cell 6 imports repository modules from the notebook Python process without adding REPO_ROOT to `sys.path` or changing that process's working directory. Prior `%%bash cd` operations do not change the notebook kernel's directory. A fresh `/content` kernel will fail to import these modules.

### P1 — Folder-mode dev exports are rejected by calibration even after the path fix

`data/datasets/base_dataset.py:33` constructs folder records with `split='unknown'`. The checkpoint hook copies these records unchanged. Calibration correctly refuses unknown splits. This affects the exact folder-based experiment in the guide.

Reproduction: export two folder-style records with logits and FP32 provenance, then call `calibrate(..., score_space='logit')`. It raises:

```text
Threshold calibration accepts only development splits ['dev', 'dev_calibration', 'external_dev', 'val'], got ['unknown']
```

The fix must record the actual development role at the trusted validation/export boundary, not weaken calibration to accept arbitrary unknown files. Preserve unknown group/source metadata and explicitly record folder-mode independence as unverified. Do not manufacture group IDs or assert a passed content audit. Manifest-based forbidden splits must remain forbidden and must never be relabeled as dev.

### P1 — The final comparison does not verify thresholds against checkpoints

Cell 6 calls `load_threshold(..., return_score_space=True)` without `expected_checkpoint_hash`. It only copies the first row's checkpoint hash into the report. It also aligns by sample ID and label without comparing paths. Folder sample IDs are simple counters, so equal IDs do not prove equal images.

A threshold accidentally swapped between arms, a mixed-checkpoint CSV, or two different folder collections with matching row counts can produce plausible invalid comparisons. Validate one consistent checkpoint digest across ALL prediction rows, bind it to that arm's actual best.pth and threshold artifact, verify FP32 provenance, and compare canonical paths plus labels in addition to IDs. Prefer a tested comparison function/tool over duplicating unvalidated notebook logic.

### P1 — Calibration artifact validation can be bypassed by the legacy status tag

Both `analyze_predictions.py:20` and `evaluation/evaluator.py:147` bypass all split checks whenever the artifact has the external-calibration status tag and `score_space='logit'`. They do not narrowly validate the old notebook artifact schema.

Reproduction: a file with `source_splits=['external_test']`, `logit_threshold=25`, conflicting `threshold=-999`, the legacy status tag, and a matching checkpoint hash is accepted as `(25.0, 'logit')` by `load_threshold`.

Explicit forbidden splits must ALWAYS be rejected. Recognize the actual old artifact only when its full required provenance/decision-rule/label-map fields are present and consistent. Reject conflicting threshold fields, wrong label maps, unknown score spaces, and precision mismatches. The actual prior artifact must continue to load via narrowly checked compatibility or explicit migration. Use one shared reader across CPU analysis, evaluator and robustness code to avoid divergent validation.

### P2 — Probability threshold 1 gives contradictory reports

For finite logits [20,30], saturated probabilities [1,1], labels [0,1], and probability threshold 1:

- `binary_metrics` returns confusion matrix `[[0,1],[0,1]]` (all fake).
- `ClassificationMetrics` returns `[[1,0],[1,0]]` (all real).

The evaluator and confusion-matrix plot also implement their own decisions. Consolidate the decision function and document endpoint behavior. For finite original logits, the mathematical probability boundary 1 corresponds to +infinity and predicts no fake samples; probability-only legacy data may use the recorded-probability comparison, but the fallback must be explicit. All reports for the SAME input and score availability must agree.

Also reject invalid score-space strings rather than silently treating them as probabilities. Do not infer logit space solely because a supplied threshold is outside [0,1]; new logit thresholds should be explicitly labeled, preserving old probability-default behavior.

### P2 — Logit thresholds are incorrectly forwarded to probability aggregates

`evaluation/generalization.py:325` forwards a logit threshold to video metrics aggregated as mean probabilities. A logit threshold of 25 throws `Probability threshold must be finite and in [0,1], got 25`. A threshold within [0,1] would be silently interpreted in the wrong units instead.

Preserve documented mean-probability video aggregation. Do not pretend an image-level logit threshold is calibrated for that aggregate. Either report video decision metrics as unavailable with an explicit reason until an aggregation-specific threshold is supplied, or implement a separately specified/tested aggregation calibration. Do not silently change mean probabilities into mean logits.

`evaluation/robustness_cli.py:84` reads score_space but never passes it onward; inference discards raw logits and calls `summarize` on sigmoid values. Thus logit artifacts do not work there despite the report's claim. Collect original canonical logits, export them, use the shared reader and propagate score space, or explicitly reject unsupported use before any image inference. Complete the implemented advertised path rather than silently misreporting.

### P2 — Verification does not support the report's strongest claims

The new smoothing train/eval tests use zero logits. BCE at logit zero is log(2) regardless of target, so these tests would also pass if smoothing were absent or applied during validation. Use nonzero asymmetric logits; verify losses AND gradients, active train/eval transitions on the same instance, label immutability, default s=0 behavior and the direct optimizer path.

The notebook test only compiles Python blocks. It does not execute the new notebook orchestration or parse the bash training commands. `tests/test_colab_workflow_and_commands.py` describes an older notebook, not this new guide. Existing distributed tests do not assert the new logits array's values/alignment. Add meaningful checks for these actual changed paths.

## Executable repair prompts

Give the following prompts to the implementation agent in order. Complete repairs and tests; do not create another implementation plan instead.

### Prompt 1: Repair the new Colab workflow

Fix the real AMP flags, checkpoint prediction paths and Python module import setup described above. Check all commands against real parsers, including calibration and evaluation. Retain the agreed three-epoch control/smoothing experiment exactly.

Make existing-checkout updates fail fast: check Git status/branch/remote, use checked commands and fast-forward-only updates, and do not continue on failed fetch/checkout/pull. Do not discard user changes or merge unrelated branches automatically. Support a fresh clone without deleting any directory. Preserve fresh-run collision protection.

Make comparison output paths unique per experiment pair rather than overwriting `OUTPUT_BASE/label_smoothing_ab_comparison.json` on later trials. State that rerunning setup creates new IDs; persist run identities to Drive for recovery. Make OOM instructions create fresh run IDs for affected trials, preserving failed-run evidence, with matching settings for both arms. Include dependency/import preflight appropriate for a fresh Colab environment.

### Prompt 2: Make folder validation provenance usable without relaxing safeguards

At a trusted training-validation boundary, export the known validation role as dev for the explicit folder-training engineering case. Preserve original role/context evidence and mark source/group/content independence unverified. Do not mutate manifest records into another split. Add an integration test using a small real folder dataset and the actual validation/checkpoint export interfaces: its `checkpoints/best_dev_predictions.csv` must successfully pass logit calibration with checkpoint/precision linkage. Add a negative manifest test showing test-split data cannot be relabeled or calibrated.

### Prompt 3: Consolidate artifact validation and secure comparison identity

Implement one shared threshold reader for new logit artifacts, old probability artifacts and the narrowly recognized previous Colab external-calibration schema. Enforce checkpoint, precision, label mapping, score space, finite bounds, forbidden-split and field-consistency rules. Ensure the compatibility tag cannot override an explicit forbidden split. Keep genuine prior notebook artifacts loadable.

Route notebook comparison through a reusable tested function or CLI. Check all prediction rows for a consistent checkpoint and precision, verify both threshold/checkpoint pairs, and align identical path/label cohorts. Include threshold and prediction file digests in the final comparison. Tests must reject swapped thresholds, mixed checkpoint rows, different paths with the same sample IDs, malformed legacy artifacts and conflicting threshold values.

### Prompt 4: Unify decisions and handle aggregate/robustness score spaces

Use a shared decision routine in generalization metrics, classification metrics, evaluator and confusion plots. Cover both probability endpoints and explicit logit thresholds, including values inside [0,1]. Preserve logit ranking and probability fallback provenance. Reject malformed score-space values.

Fix the video and robustness problems without redefining aggregation or inventing calibration evidence. Add small tests for a logit threshold above 1, one within [0,1], single-class subgroups, and the reported saturation fixture. No real image inference is needed.

### Prompt 5: Prove the complete new workflow and revise the implementation report

Strengthen smoothing tests with nonzero logits and gradient checks. Extend the CPU distributed fixture to assert logits, labels and indices remain aligned for uneven/empty shards. Execute the ACTUAL new guide's orchestration from a fresh notebook-style working directory with mocked expensive training/inference and real small prediction CSVs, actual calibration, actual artifact loading and actual comparison. Verify one inference call per arm, correct paths, valid roles, supported flags and unique persistent outputs.

Do not merely test a separately reconstructed command that can drift from the guide. The test must fail against the present `--amp_train`, missing `checkpoints/` and missing Python import-path setup.

Rerun relevant existing suites and record exact interpreter, commands and outcomes. Fix environmental test invocation issues or report them accurately. Update `RGB_LABEL_SMOOTHING_IMPLEMENTATION_REPORT.md` to distinguish actual end-to-end validation from syntax-only checks. Remove the current unconditional "no remaining defects" claim until the confirmed reproductions and new integration checks pass. Record failures or environment limitations rather than counting them as passes.

Do not push, train real models, infer over the external dataset, change dataset contents, or overwrite previous model/calibration artifacts. Deliver the corrected code, guide and report for final review.

## Review verification already performed

- Static CLI inspection confirms `--amp_train` is absent and `--use_amp` exists.
- The new guide references the wrong root-level development prediction path twice and lacks notebook-kernel import-path setup.
- Small synthetic runtime probes reproduced the invalid-artifact acceptance, folder calibration rejection, endpoint disagreement and video aggregation error; see the JSON evidence file.
- `.venv/Scripts/python.exe -m unittest` across seven relevant modules ran 59 tests: 58 passed; the two-process test errored while creating a Windows multiprocessing manager pipe (`WinError 2`). This is an environment/process-launch limitation observed in this attempt, not evidence that the distributed implementation is wrong.
- System Python (`C:/Users/SIGMA/anaconda3/python.exe`) running pytest on `test_rgb_review_final_repairs.py`, `test_fine_tuning_policies.py`, and `test_calibration_and_source_readiness.py`: 30 passed, one scheduler-order warning.
- The workspace `.venv` does not contain pytest; the system interpreter does. Do not claim the `.venv -m pytest` command passed.
- Retrying `tests/test_distributed_validator_two_process.py` with system Python and pytest failed at the same Windows manager-pipe creation step, before the validator ran. Distributed validation remains unverified in this environment.
- No GPU training or real-image inference was run in this review.
