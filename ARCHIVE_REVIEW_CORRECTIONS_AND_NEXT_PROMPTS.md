# Review of archive findings and corrected agent prompts

Date: 2026-10-09. This review supersedes the scientific conclusions and experiment proposals in `output/review/archive_investigation/ANSWERS_FOR_REVIEW.md` and `NEXT_IMPLEMENTATION_PROMPTS.md`. Preserve those original reports as the investigation record. Do not treat their unsupported claims as established facts.

## What the evidence supports

- Archive metadata reports 805,421 training images and 211,434 validation images, with no explicit provenance documents.
- The reported recognized filename-family intersection between archive train and val is zero. This is useful but narrower than video, identity, original/derivative, exact-content and near-duplicate independence.
- The internally generated development split did share apparent frame families with training. The existing replacement fixes that recognized overlap without claiming verified provenance.
- Archive extension counts correlate strongly with labels. Bounded decoded samples confirm actual JPEG/PNG differences for the sampled images, not every image in the archive.
- Contact sheets show differences in crop, image quality and framing. Some training-real crops show only partial faces or non-face patches. This warrants a crop-quality investigation; it does not justify silently deleting examples or changing labels.
- Sampled archived image bytes agree with saved preparation hashes for 251 matching paths. Five lacked parent records; none of the compared hashes mismatched. This does not establish full-archive integrity.

## Corrections required before another experiment

1. **Incorrect count:** The repaired dev cohort contains real = 6,950 JPG + 550 PNG; fake = **7,500 PNG and zero JPG**. The claimed 550 fake JPG files contradict `output/review/rgb_family_repaired_100k_seed42/repair_report.json`.
2. **Unverified source identities:** Contact sheets cannot confirm FaceForensics++, CelebA, FFHQ, StyleGAN, ProGAN or diffusion provenance. Both the source evidence and metadata reports say provenance is unknown. Keep those names only as possible user-supplied origins, not findings. Do not infer whether a fake was generated from scratch or edited from a real image solely from appearance.
3. **Training is not demonstrated to be exclusively video manipulation:** The archive includes 31,111 numeric fake training files. Independently counting the actual R2 training manifest gives 47,247 filename-pattern video fake images and **2,753 numeric fake images**, alongside 11,430 video-pattern real and 38,570 numeric real images. The numeric fake subset must be examined before declaring two entirely different tasks. Its source remains unknown.
4. **Biased visual coverage:** The training contact sheets shown in the report contain only `vid_` examples, despite numeric images being included in the profiling sample. They do not visually represent both training subcollections. Separate sheets for numeric real/fake are needed.
5. **Not a certified clean test:** Zero common recognized filename tokens does not prove zero leakage. No full cross-partition content/near-duplicate/original-derivative audit was shown. Do not call the archive val partition pristine, a final test, or guaranteed independent. Earlier experiments also used a 209,871-image dev cohort derived from the archive val population, so the project has not universally ignored it. R2 omitted archive val from its own preparation; keep that narrower claim.
6. **External overlap is unverified:** The statement that external data has zero overlap with the archive is unsupported by the supplied cross-dataset hash evidence. Report unknown, not zero.
7. **AUC does not identify learned features:** AUC above 0.90 on archive val would show discrimination on that cohort, not prove the detector learned manipulation features. High performance after re-encoding also cannot prove absence of shortcuts. Persistent external failure would not establish that specialized architectures or multi-task adaptation are necessary.
8. **Encoding is not a container-only intervention:** PNG is lossless compression, not an uncompressed format. Converting decoded JPEG pixels to PNG preserves JPEG artifacts. JPEG re-encoding changes pixels; unchanged RGB pixels saved/reloaded as PNG should be a pipeline-control check. A model receiving RGB tensors does not directly read filename extensions or container headers. Remove the unsupported phrase “PNG demosaicing artifacts.”
9. **Verification scope:** Population extension counts are not population decoded-format counts. Sample resolution ranges do not establish full-population ranges. Report each scope explicitly.

## Agent prompt 1 — Correct the evidence record and recover available provenance

Read this review and the original investigation artifacts. Create `output/review/archive_investigation/CORRECTED_ANSWERS.md`, retaining useful numeric evidence while correcting all nine points above. Attach a claim-to-evidence table and a section of unknowns. Check class/extension/pattern counts from the cached listing and actual R2 manifest; use full relative paths. Do not rerun archive extraction or image hashing for completed checks.

Look within the repository's existing dataset preparation/download scripts, notebooks and project documentation for explicit construction records linking the numeric and `vid_` subsets to their sources. Search relevant files only; do not crawl unrelated drives or private browser history. File names and comments suggesting a source are leads, not verified mapping. Record exact file/line evidence and confidence. If unavailable, ask the user for the preparation script or original download links for each subset and external set. Do not invent provenance to unblock source-held-out training.

## Agent prompt 2 — Fill the numeric-subset inspection gap

Reuse the already extracted 256 archive samples and `sample_members.csv`. Create and inspect separate contact sheets for numeric real, numeric fake, video-pattern real, and video-pattern fake, selecting deterministically at most 24 examples each. Keep labels as supplied; describe crop/framing/content coverage without inferring named sources or generator mechanisms. If image viewing is unavailable, mark visual review unperformed.

Report dimensions, decoded formats and crop observations separately for these four strata. No new large extraction is needed. If sampled files are gone, selectively retrieve only those named members, preserving the original 512 MiB scratch ceiling and 1 GiB free-space reserve. Produce `subcollection_review.md` with representative member paths, sample counts, and unknowns. Do not infer numeric images are independent photographs merely because their names are numeric.

## Agent prompt 3 — Diagnose existing predictions by subcollection before new inference

Read R2 `best_dev_predictions.csv` and its original 100K manifest from `F:\Discovery AI\Second Expirement 100K RGB model\my_dataset_archive.zip`. Reuse existing external predictions. Join by sample ID/full path, verifying checkpoint identity. Compute counts, real recall, fake recall and probability quantiles for numeric versus video-pattern rows and class/extension combinations. Where both classes occur, report ROC AUC and balanced accuracy at the existing 0.5 threshold; report AUC as undefined for one-class strata. Include error counts and class denominators, not only percentages. Use raw-logit AUC only where logits already exist.

These are diagnostic slices of already inspected development data. Do not fit or export an external threshold, relabel provenance, or claim an unbiased final-test estimate. Produce `subcollection_metrics.json` and a short interpretation. Determine whether the aggregate development score hides a weak numeric fake or PNG-real subset. This requires CSV analysis, not model inference or new training.

## Agent prompt 4 — Implement a bounded paired-preprocessing diagnostic

Only after the preceding evidence corrections, implement a reusable diagnostic utility and tests, without starting training or a full external evaluation. Reuse the production checkpoint loader, RGB transforms and score direction. Allow an explicit checkpoint plus a supplied manifest of at most 128 images; never silently scan directories. Default workers to zero for Windows. Do not alter original images.

Compare original decoded RGB against (a) lossless PNG round-trip as an RGB-equality control and (b) fixed JPEG quality 95 and 75 round-trips as pixel interventions, using identical metadata/color-conversion policy and recorded encoder settings. Perform interventions at a documented common point before deterministic evaluation transforms; no random augmentations. Preserve per-sample IDs and record transformed-tensor differences, raw logits, probabilities, existing-threshold decisions, and group summaries. Never select a best threshold or “best quality” from the external results. If PNG changes decoded RGB, investigate that pipeline/metadata issue instead of describing it as compression invariance.

Use tiny synthetic fixtures for tests of PNG RGB equality, unchanged original files, ID alignment, bounds and score calculations. Actual-model execution should be an explicit user command delivered in the report, not an automatic large evaluation. Passing this diagnostic cannot establish universal generalization or eliminate historical compression confounds.

## Agent prompt 5 — Return a constrained next training proposal

Create `output/review/archive_investigation/REVIEW_RESPONSE.md`. Answer what corrections were made, what the numeric subset contributes, what existing per-subcollection predictions show, and what provenance is still missing. Link test results and give the exact bounded diagnostic command if implemented.

Recommend a next pilot only to the extent supported by these findings. Preserve recognized-family separation and both-class coverage, avoid letting the abundant video-frame subset overwhelm a smaller numeric subset, and explicitly distinguish observable sampling strata from verified source domains. Do not claim a source-held-out experiment is feasible without source annotations. Re-encoding alone, more images, or an arbitrary AUC target is not an established fix.

Do not extract the proposed 10K validation cohort yet, do not relabel archive val as an untouched final test, and do not start training. Keep output small, preserve original reports and checkpoints, and return `REVIEW_RESPONSE.md` for the next review. No other-chat messaging is needed.
