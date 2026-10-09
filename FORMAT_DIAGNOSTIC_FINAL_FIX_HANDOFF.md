# Fix and execute the bounded format diagnostic

## Review verdict

The latest `output/review/archive_investigation/REVIEW_RESPONSE.md` contains useful, reproducible confusion counts, but its diagnostic is not ready to run and its AUC values are incorrect. This handoff supersedes its execution command and causal claims. Preserve original evidence. Perform the following fixes and then the bounded run; do not return another untested command or start training.

Repository: `G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection`.
Python: repository `.venv\Scripts\python.exe`.
Checkpoint: `F:\Discovery AI\Second Expirement 100K RGB model\best.pth`.
Checkpoint SHA256: `e598e1b32871f460756118ba6d4a5f1ad3679a3b9727dff1afa04d4d0de96a28`.

## Established results and interpretation

Independent recomputation from the saved development predictions is in `output/review/archive_investigation/independent_metric_review.json`:

| Stratum | Real / fake | FP / FN | Balanced accuracy | Correct probability AUC |
|---|---|---|---|---|
| All development | 7,448 / 7,387 | 272 / 141 | 0.97219627 | 0.9961728423 |
| Numeric | 5,763 / 392 | 5 / 0 | 0.99956620 | 0.9999969014 |
| Video-pattern | 1,685 / 6,995 | 267 / 141 | 0.91069289 | 0.9828972369 |

Thus real recall differs strongly across the observed strata. This supports investigating confounding but DOES NOT establish that numeric performance is “solely because” of compression, that models “inherently learn” format shortcuts, or that every PNG is predicted fake. Source, content, crop and resolution also change between these strata. Numeric fake mechanism and named source identities remain unverified. Uniform resolutions were measured on samples, not all numeric archive images. The 69.9% video-fake share refers to the full archive training population; the actual R2 training sample contains 47,247 video-pattern fake images out of 100K (94.494% of its fake class).

## Prompt 1 — Repair the metric calculation and evidence claims

In `tools/compute_subcollection_metrics.py`, replace double-argsort AUC: its ranks start at zero but its formula assumes one-based ranks, and it gives different ranks to tied scores. Use a trusted ROC AUC implementation or a tested tie-aware formula. Test perfect separation (1), reversed separation (0), all ties (0.5), mixed ties, single-class undefined, and invariance to row order. Reject nonfinite probabilities, invalid labels, duplicate sample IDs and mixed/missing checkpoint hashes. Verify saved prediction IDs, labels, image hashes and full paths against the original development rows in the saved 100K manifest before aggregation. Do not join by basename.

Regenerate the diagnostic metric report and update its explanatory prose, removing hardcoded causal conclusions. Match the independent results above within numerical precision. Keep real/fake denominators, error counts and sample/population scope explicit. Do not tune an external threshold.

## Prompt 2 — Fix the paired-preprocessing implementation

Review `tools/diagnose_format_sensitivity.py` against production `RGBDataset`, `CheckpointLoader` and `InferenceRunner`.

Required fixes:

1. **Strict input contract:** Remove the broad exception fallback from `read_manifest` to raw CSV. Do not bypass invalid labels, missing splits, duplicate IDs or failed path resolution. Reject empty and >128-row manifests before loading a model. Respect explicit absolute/full relative paths. Verify saved image hashes when supplied and record actual hashes.
2. **Production preprocessing:** The current helper forces bilinear resizing and cropping, ignoring checkpoint `rz_interp`, `no_resize`, `no_crop` and fallback size options. Reuse the production RGB dataset's evaluation transform (or a shared factory) with a copied eval configuration, rather than maintaining a competing implementation. Verify tensor parity against the production path for original images, including no-resize/no-crop and nondefault interpolation cases. Preserve the production decoder's RGB conversion policy.
3. **Canonical scores:** Honor the loaded model's `score_sign` and output convention exactly as production inference does. Restrict supported architectures explicitly to the RGB Wang models if that is the actual implementation scope. Reject unexpected output shapes; do not silently take the first element of an arbitrary vector/tuple. Record checkpoint hash, effective evaluation options, precision and runtime versions.
4. **Output directory:** The current code does not create its output directory and fails when writing results. Create a fresh destination before inference, verify it is writable, and avoid overwriting existing reports. Keep image files unchanged and use in-memory interventions.
5. **Honest controls:** `batch_size` and `workers` are currently accepted but unused. Implement actual batching or remove the misleading batch option. Either implement workers or explicitly support only zero workers. Record batch behavior. No implicit directory scans.
6. **Interventions:** Keep original, PNG RGB-equality control, JPEG95 and JPEG75. Record Pillow/encoder version, JPEG subsampling=0, quality, metadata policy and the intervention point before production transforms. Fail a PNG pixel/tensor equality violation as a control failure. Do not relax controls to obtain a pass.
7. **Reports:** Preserve sample ID, class, observable subcollection and cohort (training diagnostic versus external development), image hash, raw and canonical logit, probability, pixel/tensor differences and fixed-0.5 decision. Summarize signed and absolute score shifts and confusion counts separately by cohort and class, not only pooled flip counts. Never call quality with the highest external accuracy an optimized deployment setting.

## Prompt 3 — Meaningful tests and a runnable bounded manifest

Existing tests exercise helper functions but never execute a successful end-to-end diagnostic: the mock model is unused, output creation and score direction are untested. Add CPU tests using a deterministic dummy model/loader to exercise the complete path, newly created report directory, exactly four condition records per sample ID, nondefault preprocessing, reversed score sign, unchanged input bytes, invalid manifests and output bounds. Use seeded fixtures; avoid treating a JPEG error ordering on random images as a universal invariant.

Run tests with unittest if pytest is unavailable. Do not install dependencies unnecessarily. Include test names, counts and failures in the final report.

Create one valid manifest with at most **128 distinct images**: 64 archived training examples (16 each numeric-real, numeric-fake, video-pattern-real, video-pattern-fake) and 64 external-development examples (32 per class). Reuse the already extracted archive samples and full-path mappings in the investigation records. Do not use the proposed 256-row input or silently truncate it. Record deterministic seed 42 and sampling quotas; do not infer source labels from strata. If an input is missing, use only the explicitly selected archived members with the established 512 MiB scratch ceiling, or report the missing files. No full extraction or hashing.

## Prompt 4 — Run the bounded diagnostic and return the decision evidence

After tests pass, execute the repaired diagnostic on that <=128-image manifest using the supplied checkpoint, cuda:0 if available, otherwise CPU. This authorizes only this bounded diagnostic (four interventions per image), not retraining or a 60K-image reevaluation. Leave the original checkpoint and dataset untouched. Record exact commands and elapsed time. If a control/identity check fails, report it and do not interpret the result as model robustness.

Produce `output/review/archive_investigation/FORMAT_DIAGNOSTIC_REVIEW_RESPONSE.md` linking corrected metrics, test results, manifest and paired results. Explain whether JPEG changes external-real and external-fake behavior differently, whether PNG controls are exact, and whether effects also occur in training strata. A lack of change does not rule out historical compression/source shortcuts; a large change establishes sensitivity to this intervention, not its sole causal role in generalization failure.

Finish with one evidence-qualified pilot proposal and the specific missing provenance needed for a source-held-out claim. A four-stratum/group-aware engineering pilot may be considered without pretending the strata are verified sources. Do not promise 85%, silently resume the old checkpoint on a changed split, or recommend larger datasets merely because capacity is available. No training is authorized by this handoff. Preserve external data as external development and retain an untouched future evaluation set for any final claim.
