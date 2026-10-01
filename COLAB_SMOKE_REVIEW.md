# Latest review: finish the smoke workflow before Colab

Reviewed 2026-09-29. **Update: the four corrections below have now been implemented by Codex. Static command/notebook checks passed; runtime remains NOT RUN. The findings and prompt below are retained as review history, not pending delegation.** Scope: Round 3 O–Q changes and the new smoke workflow. This is a targeted static review, not certification of the entire repository.

## Verdict

The changes improve the experiment's reliability. Source-namespaced sample IDs, explicit holdout conflicts, per-video training caps, stricter manifest gates, removal of suffix-based path guessing, and reading the actual checkpoint manifest metadata are present. Distributed accumulation and coordinated experiment creation also have new implementations; their numerical and multiprocess behavior remains unverified.

Eleven relevant Python files parsed successfully. Smoke notebook code cells parsed after excluding notebook magics, and notebook/config JSON parsed. No application imports, tests, image scans, model loads, training, inference, installations, or downloads were performed. The smoke notebook contains no executed cells.

**Original review verdict (before the repair):** The smoke workflow needed the four corrections below. They are now implemented; the next phase is user-run verification on Colab.

## Required corrections

### 1. Invalid RGB training option

`train.py` defines `--pretrained` using `argparse.BooleanOptionalAction`; its negative option is `--no-pretrained`. The notebook, its generator, smoke JSON commands, and status document instead pass `--no_pretrained`. Training stops at argument parsing.

Correct callers of `train.py`. Do not globally replace flags for other entry points without checking their parsers. For the smoke workflow, verify all stages avoid unintended pretrained downloads.

### 2. Evaluation command does not exist

The notebook/generator, smoke JSON, and status document invoke `eval.py`, which is absent. They also invent `--model_path`, `--calibrate_threshold`, and `--export_predictions` options that the actual `evaluate.py` CLI does not expose.

Use the existing interfaces:

- `evaluate.py --checkpoint <fresh-fusion-best> --arch MHA_128 --dataroot <root> --manifest <smoke-manifest> --split dev --output_dir <evaluation-dir>`.
- Disable optional plots, t-SNE, Grad-CAM, and profiling with its supported flags for a bounded smoke run. Supply expert paths if required by the existing checkpoint loader.
- For that single-model invocation, the evaluator writes `<evaluation-dir>/MHA_128/predictions.csv`.
- Then use `analyze_predictions.py calibrate --predictions <that-CSV> --output <threshold-JSON>`.

These are interface descriptions, not commands to execute now. Replace placeholders with notebook variables in the final executable cells. Check checkpoint metadata supplies the fusion configuration; `evaluate.py` has no `--fusion_type` option. Calibration must use development predictions only.

### 3. The dataset size is not bounded

`prepare_dataset.py` deliberately preserves full evaluation membership when producing a pilot. Targets of 50 real and 50 fake limit training selection only. Passing the full pool with `dev_ratio=0.2` and `test_ratio=0.2` can retain a very large development/test set; auditing and epoch validation can consequently be expensive.

Require a separate, explicitly bounded smoke input manifest before expensive work. Either prepare a dedicated subset from verified metadata or require the user to supply one. Keep original/derivative links and identities in one split, retain genuine provenance, require both classes in train and dev, and assert/report strict per-split and total row limits before opening or hashing images. For example, permit at most 300 total smoke rows, including at most 100 train and 100 dev rows. Fail clearly if independent groups or both classes cannot fit. Never change the full pilot's protected split policy just to make the smoke run small. Smoke data is for wiring checks, not accuracy claims.

### 4. Reruns and resume claims need correction

All stages hard-code `smoke_seed42` and fixed output directories. A second fresh run collides with the new collision protection. Generate one fresh smoke run identifier/output root per session and derive every stage path from it. Reuse the existing identifier only for that session's intentional resume.

The resume command drops the original batch-size and accumulation settings. Build it from the original fusion arguments and change only resume fields and total epochs. Make the documented settings (including worker count and precision) agree with the actual arguments; the JSON currently declares settings that its commands do not pass.

The notebook prints that state recovery was verified merely because the subprocess returned successfully. Add deferred checks of the saved epoch/global-step progression and resume evidence, or narrow the message to what was actually checked. Exact optimizer/RNG/numerical continuation remains a separate deferred test requirement.

## One prompt for the implementation LLM

> Fix only the four smoke-workflow issues in COLAB_SMOKE_REVIEW.md. Update tools/make_smoke_notebook.py, colab_smoke_pipeline.ipynb, config/experiments/smoke_test_run.json, CODE_READY_STATUS.md, and relevant COLAB_RUN_GUIDE.md instructions together. Use actual train.py/evaluate.py/analyze_predictions.py interfaces; do not invent new evaluation entry points or weaken data/checkpoint safeguards. Make the smoke input and all splits strictly bounded before image work, provide unique fresh-run paths with coherent resume paths/configuration, and distinguish successful execution from verified state restoration. Add static checks of entry-point existence and CLI flags plus deferred runtime assertions where useful. Preserve all other edits. Only write code/documents and perform standard-library static checks: no application imports, tests, model construction, training/inference, dataset scans, installs, downloads, or historical-checkpoint requests. Report exact changes and leave runtime status NOT RUN.

## After these corrections

1. Review the corrected commands statically, then run the relevant deferred tests and the bounded smoke workflow on a single Colab GPU. These runtime steps are for the user's later Colab phase, not authorized by this review to run locally.
2. Confirm both experts, fusion, checkpoint export, development prediction export/calibration, and resume work. Check actual split counts, label mapping, audit evidence, and saved artifacts. Persist artifacts before a Colab session ends. Test two-GPU behavior separately before using it for an expensive run.
3. Only then start the balanced 100K pilot with verified sources and independent groups. Choose models/thresholds on development data and preserve a genuinely untouched external test set. Previously inspected failure sets are development evidence if used to guide changes.
4. Consider 1M training after the pilot demonstrates useful cross-dataset performance. Reviewer responses still need measured baselines, ablations, uncertainty, and the other evidence tracked in the reviewer closure matrix. Code changes alone do not close those requests.

No static review can establish that external accuracy will rise from 60% to over 85%. That remains an experimental target.
