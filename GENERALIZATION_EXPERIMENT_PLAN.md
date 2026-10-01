# Generalization Experiment Plan: Wang2020-128 + WolterWavelet2021-128

## Purpose and scope

This plan turns the current two-stream project into a defensible generalization study. The fixed expert models are:

- **Spatial expert:** `Wang2020_128` (RGB input, 128-D embedding)
- **Frequency expert:** `WolterWavelet2021_128` (wavelet-packet input, 128-D embedding)

The fusion models to evaluate are:

1. **`Fusion_128`** - concatenation + MLP. This is the required simple-fusion control.
2. **`MHA_128` (corrected)** - an attention model that can genuinely choose between more than one source token.

The aim is not to guarantee a high score on every unknown dataset. The aim is to measure, improve, and report cross-generator and cross-dataset performance without contaminating the final test sets.

## Non-negotiable scientific rules

1. A final external test dataset must never be used to choose the architecture, epoch, threshold, augmentation, or training-set size.
2. Training, validation, and internal test splits must be grouped by source video and identity before frames are sampled.
3. Every result must identify its exact checkpoint, dataset manifest version, split manifest version, preprocessing configuration, random seed, and decision threshold.
4. A high in-domain score is not evidence of generalization. External results are the deciding evidence.
5. Do not claim deployment readiness, fairness, or open-world robustness unless the corresponding experiments support the claim.

---

## 1. Technical issues found in the current project

### P0 - Current two-stream attention is not dynamic routing

`models/mha/mha_classifier.py` converts each 128-D embedding to a sequence with exactly one token:

```python
query = query.unsqueeze(1)        # [B, 1, 128]
key_value = key_value.unsqueeze(1) # [B, 1, 128]
```

For each attention head there is only one key. Softmax over one key is always 1.0. Therefore, the module cannot assign a variable attention weight between RGB and wavelet evidence as claimed. It is still a learnable residual transformation of the other stream, but it is not evidence of sample-specific attention routing.

**Required correction:** replace this mechanism before making an attention-novelty claim. The first corrected version should stack a learned `[CLS]` token, RGB token, and wavelet token into `[B, 3, 128]`, add learned token-type embeddings, apply Transformer self-attention, and classify from the `[CLS]` output. The `[CLS]` token has multiple keys available, so its attention can vary per image. Return attention weights for analysis.

**Later, stronger version:** expose multiple spatial feature-map tokens and multiple wavelet sub-band tokens, then apply bidirectional cross-attention. This is a larger change and should be attempted only after the corrected lightweight token model is evaluated.

### P0 - Nested dataset folders can produce non-binary labels

The current loader builds a separate `ImageFolder` for every top-level class and concatenates them. If a layout is nested, for example:

```text
train/
  fake/ADM/...     fake/DDIM/...
  real/CelebA/...
```

`ImageFolder` assigns labels from the **nested source folder names**, not necessarily `fake=1` and `real=0`. `fake/ADM` can receive label 0 while `fake/DDIM` receives label 1; this is invalid for `BCEWithLogitsLoss` binary training. The weighted sampler has the same risk because it balances those derived labels.

**Required correction:** introduce a manifest-driven binary dataset. Each manifest row must carry an explicit binary `label`; it must not be inferred from nested folder ordering. Add a preflight report that prints and asserts exactly two labels, their counts, and example paths for each label.

### P0 - The training data can contain a dataset-source shortcut

Using only CelebA for real images while all fake images originate from other repositories can teach the model source style, camera pipeline, crop style, or compression style instead of manipulation artifacts. This may yield excellent internal accuracy and poor external accuracy.

**Required correction:** diversify the real class and, where possible, include authentic source material corresponding to each video manipulation dataset. No source repository or preprocessing pipeline should be nearly synonymous with one class.

### P0 - Fusion can silently load untrained experts

Both fusion trainers print a warning and continue with an unweighted backbone when a checkpoint path is missing or invalid. A final experiment must fail immediately instead.

**Required correction:** make expert checkpoints mandatory for fusion training/evaluation; record their absolute paths, hashes, validation metrics, and output shape. Assert that both embeddings have shape `[batch, 128]`, both expert models are frozen, and no warning was emitted.

### P1 - The current robustness attack does not fully attack the fused input

`evaluate_robustness.py` attacks the normalized RGB tensor but clamps it to `[0, 1]`, which is not the valid range after ImageNet normalization. For fusion models, it also keeps the supplied wavelet input clean instead of recomputing it from the perturbed image. The reported attack is therefore not a full end-to-end attack on both streams.

**Required correction:** perturb raw pixels in `[0, 1]`, clamp there, normalize only before the RGB expert, and recompute the wavelet representation from the same perturbed pixels. Keep a clearly labeled realistic JPEG/blur/noise robustness suite separate from white-box FGSM/PGD.

### P1 - Evaluation is not yet a complete generalization protocol

The existing evaluator is reusable, but it only receives paths and binary labels. It has no generator/source/identity group information, no confidence intervals, no per-group metrics, and no protected final-test mechanism.

**Required correction:** add manifest-aware evaluation, per-source/per-generator breakdowns, bootstrap confidence intervals, balanced accuracy, AUC, EER, and FPR at fixed TPR. Select the classification threshold using the development set only.

---

## 2. Data plan for the 100K pilot

The pilot target is **50K real + 50K fake samples**. This is sufficient to validate the protocol and compare designs; it is not automatically the final paper scale.

### Fake-class quotas

Use approximately 10K samples per fake generator/manipulation family. Select well-spaced frames and diverse identities/source videos. Do not collect 10K nearly adjacent frames from a small set of videos.

### Real-class quotas

Do not use 50K CelebA images alone. Spread real samples over multiple authentic sources, including original real videos from any video-manipulation sources represented in the fake class when licensing permits. The exact quotas depend on available metadata, but every major source style should occur in both the experiment or be explicitly held out as a source-domain test.

### Manifest schema

Create a CSV or Parquet manifest with at least these columns:

```text
sample_id, path, label, split, dataset_source, source_video_id,
identity_id, generator, manipulation_type, frame_index,
is_original_source, width, height, checksum
```

Unknown fields must be written as `unknown`; they must not be invented.

### Split protocol

- Split source videos/identities first; sample frames only after the split.
- For the initial 100K pool, use a group-stratified 80/10/10 train/development/internal-test split where data availability permits.
- Verify that source-video IDs, identity IDs, checksums, and perceptual hashes do not cross splits.
- Preserve generator and source distributions deliberately. Write the resulting counts as a paper table.

---

## 3. Model experiment matrix

All fusion models below must use the **same frozen Wang2020-128 and WolterWavelet2021-128 checkpoints**, same dataset manifest, same split, same augmentations, same optimizer schedule, and same seeds.

| ID | Model | Purpose |
|---|---|---|
| E1 | Wang2020-128 alone | Spatial expert baseline |
| E2 | WolterWavelet2021-128 alone | Frequency expert baseline |
| F1 | `Fusion_128` | Mandatory concatenation control |
| F2 | Gated fusion | Tests learned per-sample scalar/vector weighting without attention |
| F3 | Late score ensemble | Tests score-level fusion |
| F4 | Corrected token `MHA_128` | Main lightweight attention model |
| F5 | Token-rich cross-attention (optional) | Stronger architectural contribution if F4 is promising |

`Fusion_128` is not a weak dummy baseline: tune it fairly and report its parameter count. The corrected MHA must beat or match it consistently on the development and external-development protocols, not merely on an internal validation set.

### Corrected lightweight MHA design

1. Project RGB and wavelet embeddings to a shared dimension (initially 128).
2. Form token sequence `[CLS, RGB, Wavelet]`.
3. Add learned token-type/position embeddings.
4. Apply 1-2 Transformer encoder layers with 4 heads, residual connections, layer normalization, and dropout.
5. Use the `[CLS]` output for binary classification.
6. Save per-head `[CLS] -> RGB` and `[CLS] -> Wavelet` attention weights for qualitative analysis. Do not present attention weights as causal proof.

This design remains small enough for an L4 or two T4 GPUs and has a real choice between source tokens.

---

## 4. Evaluation protocol

### Development protocols (used to make decisions)

1. **Internal grouped test:** unseen source videos and identities from the same data collection.
2. **Leave-one-generator-out:** train without one generator/manipulation family and evaluate on that held-out family.
3. **Leave-one-source-out:** hold out an entire dataset/source when metadata permits.
4. **External-development datasets:** no fine-tuning; use only to choose data composition/model settings.

### Final protocols (used once after choices are frozen)

- At least two independent external datasets not used during development.
- Report a separate result per dataset, not only a pooled result.
- Keep all external preprocessing deterministic and documented: face detection/alignment, crop, resize, RGB order, normalization, wavelet transform, and label mapping.

### Required metrics

- ROC AUC and PR AUC
- Balanced accuracy and normal accuracy
- EER
- FPR at fixed TPR (for example, TPR = 95%)
- Per-class precision, recall, and F1
- Per-generator and per-source results
- 95% bootstrap confidence intervals
- Three independent seeds for final comparisons

An external accuracy above 80% is promising only when AUC, balanced accuracy, and error rates agree across multiple untouched external datasets. A single 80% accuracy number does not establish generalization.

---

## 5. External-result diagnostic playbook

When external performance is low, run these checks before changing the model:

| Symptom | Check | Interpretation/action |
|---|---|---|
| Accuracy around 40%, AUC near 0 | Label direction and class mapping | Likely inverted labels or score convention |
| Low accuracy but useful AUC | Threshold chosen on external test, imbalance, calibration | Use development-set threshold; report balanced accuracy/AUC |
| Every model collapses | Face crop, resize, RGB/BGR, JPEG, source shift | Audit preprocessing and source/domain shift |
| Only fusion collapses | Exact expert checkpoint, 128-D extraction, wavelet configuration | Run expert-only and fusion prediction parity tests |
| High internal score, poor external score | Duplicates, source shortcut, generator leakage | Re-run manifest/split audit and diversify training sources |

Save false-positive and false-negative galleries by source/generator. They are often more informative than one aggregate metric.

---

## 6. Training sequence and GPU use

### Phase A - Preflight (no expensive training)

1. Implement manifest-driven binary loading and split audit.
2. Verify labels on 100 random samples; assert labels are only 0 and 1.
3. Verify a fixed batch produces the same expert embeddings after checkpoint reload.
4. Verify fusion training fails if either expert checkpoint is absent.
5. Run one small smoke training/evaluation to confirm end-to-end reporting.

### Phase B - 100K pilot

1. Train Wang2020-128 and WolterWavelet2021-128 on the grouped training split.
2. Choose epochs using the development split, not external final tests.
3. Freeze the selected expert checkpoints.
4. Train F1-F4 under identical conditions.
5. Run internal, leave-generator-out, and external-development evaluations.

### Phase C - Scale only if justified

Train the best protocol at 200K and then 400K while preserving the same source/generator balance. Scale to 1M only if external-development results consistently improve. More near-duplicate frames are not a reason to scale.

### Hardware recommendation

- Use the **L4 (24 GB)** first for development and the corrected token fusion model; it gives more headroom for wavelet processing and debugging.
- Use **two T4 GPUs** with distributed training for repeated expert runs after the pipeline is stable.
- Enable mixed precision and determine the largest stable per-GPU batch size by measurement; do not assume a batch size in advance.
- Train experts separately, then train frozen fusion heads. Deterministic embedding caching is acceptable for fast baseline evaluation, but do not use cached embeddings to replace augmentation-dependent fusion training unless that limitation is explicitly controlled.

---

## 7. Definition of success

The project is ready for final-paper experiments when all of these are true:

- Binary labels and group separation have passed automated audits.
- The MHA is technically genuine attention over multiple tokens.
- Fusion comparisons include at least concatenation, gated fusion, late ensemble, and corrected MHA.
- Claims match observed evidence; no unsupported robustness/deployment language remains.
- External test results are reported independently, with confidence intervals and no target-set tuning.
- Training scale is selected by external-development learning curves, not by internal accuracy alone.

## Immediate implementation order

1. Manifest-driven binary dataset + label/split audit.
2. Fail-fast checkpoint loading + embedding parity test.
3. Corrected token `MHA_128` and gated/late-fusion baselines.
4. Manifest-aware evaluator with per-group metrics and bootstrap confidence intervals.
5. Robustness-pipeline correction.
6. 100K pilot training and external-development diagnosis.

Do not start expensive 100K training until steps 1 and 2 pass. Otherwise the result may be expensive but scientifically unusable.

---

## 8. Reviewer-comment closure matrix

This section converts every substantive Reviewer 5 comment into a required experiment and a specific acceptance criterion. The revised manuscript must report the evidence in the Results section. It is not sufficient to move an item to "future work" or mention it as a limitation.

| Reviewer concern | Required experimental response | Evidence that must appear in the revised paper | Claim permitted only if this passes |
|---|---|---|---|
| Limited methodological novelty | Correct the MHA so attention operates over multiple tokens; compare it with concatenation, gated fusion, score-level late ensemble, and a parameter-matched Transformer/self-attention baseline | Architecture diagram; parameter/FLOP table; identical-protocol results across internal, leave-generator-out, and external-development tests; attention visualizations | "Attention-based fusion improves over static fusion under the evaluated protocols" |
| MHA may not be necessary | Train every fusion baseline from exactly the same frozen Wang2020-128 and Wolter checkpoints using the same manifests, epochs selected on development data, augmentations, seeds, and optimizer schedule | Mean and 95% CI over 3 seeds; paired test statistics on the same predictions; per-dataset metric table | "The attention design is necessary" only when it improves reliably beyond the controls, not when it merely matches them |
| Generalization/open-world claims unsupported | Run grouped internal testing, leave-one-generator-out, leave-one-source-out where possible, and no-fine-tuning evaluation on at least two final independent external datasets | Separate external result table for every dataset, including dataset provenance, class counts, preprocessing, AUC, balanced accuracy, EER, FPR@TPR, and CIs | "Cross-dataset generalization under these benchmarks"; never "universal" or "open-world" robustness |
| Models may learn source/compression/generator shortcuts | Manifest every sample; audit source video, identity, generator, frame, hashes, and preprocessing; diversify real-source composition; sample spaced video frames rather than dense adjacent frames | Dataset provenance and split tables; overlap/duplicate-audit report; source and generator distributions per split | "Leakage-aware evaluation" after automated audit passes |
| Existing baselines are insufficient | Add reproducible contemporary deepfake-detection baselines selected before final testing. At minimum, include the existing Wang/Xception/Wolter controls and several recent generalization-oriented baselines with available code/checkpoints. Retrain when feasible under the same protocol. | A protocol table that states backbone, initialization, training data, augmentation, epochs, parameters, and whether each baseline is retrained. Do not mix published numbers with local results in the same ranking. | "Competitive with the evaluated baselines" only |
| Cross-generator evidence is missing | For each generator family in the manifest, train without that family, select the epoch without it, and evaluate on the held-out family. Repeat for every feasible generator. | A leave-one-generator-out matrix: training generators, unseen test generator, metrics and CIs | "Unseen-generator performance" only for generators excluded from all training and development choices |
| Robustness is incomplete | Evaluate clean inputs plus predefined JPEG qualities, blur levels, noise levels, resize/recompression settings, and corrected end-to-end FGSM/PGD tests. Do this for both experts and all selected fusion controls. | Robustness curves/tables showing absolute scores and score drops, attack parameters, and whether wavelets were recomputed from the perturbed input | "Robust under the tested perturbations"; not generally "robust" |
| Fairness is incomplete | Use a legally accessible benchmark with documented, ground-truth demographic labels. Evaluate groups and relevant intersections for gender, age band, skin tone/ethnicity where the benchmark supplies those labels. Do not infer labels with a face-analysis model. | Group sample counts; group AUC, FPR, FNR, balanced accuracy, confidence intervals, and maximum gap; stated statistical uncertainty | "Fairness evaluated on [named benchmark]". A claim of fairness requires small, statistically supported gaps, not aggregate accuracy alone. |
| Dataset detail is insufficient | Release or archive a data manifest (or a reproducible manifest-generation script where redistribution is restricted), split manifests, inclusion/exclusion rules, generator labels, sampling cap per video, face-crop process, and exact class counts | Reproducibility appendix and public repository/archival link; data card; split audit output | "Reproducible split protocol" |
| Strong practical/deployment language exceeds evidence | Rewrite abstract, introduction, results, and conclusion after final results are locked. Remove claims about practical deployment, dynamic routing, general fairness, or open-world readiness if any required test fails. | Claim-to-evidence audit before submission | Claims that exactly match the completed experiments |

### Baseline-selection rule

Before training, make a short written baseline registry. It must list the chosen current baselines, their licenses/code availability, input requirements, expected compute, and whether retraining under the common protocol is possible. The final set should contain at least four non-proposed comparator families in addition to Wang2020-128 and WolterWavelet2021-128. A candidate cannot be called a fair head-to-head baseline unless it ran on the same split and evaluation implementation.

### Fairness requirement is a real dependency

Fairness cannot be solved by wording. To fully close this reviewer point, the project needs a suitable, demographically annotated dataset with permissions compatible with the study. If such a dataset cannot be obtained, the paper must say fairness was not empirically evaluated and it will remain a valid limitation. Therefore, acquiring and documenting this dataset is a mandatory project dependency, not an optional polish step.

---

## 9. Submission gate: when the future-work items are truly complete

Do not submit the revised paper until this checklist is complete:

- [ ] The source code passes binary-label, duplicate, source-video, identity, and generator-separation audits.
- [ ] The corrected MHA has multiple tokens and exported attention weights; the old single-key MHA is not presented as dynamic routing.
- [ ] Concatenation, gated fusion, late ensemble, and attention/Transformer fusion all have matched-protocol results.
- [ ] The selected modern baselines have a documented, controlled comparison protocol.
- [ ] Every leave-one-generator-out and planned leave-one-source-out experiment is complete.
- [ ] At least two untouched external final datasets have separate no-fine-tuning results.
- [ ] Robustness tests use the corrected end-to-end perturbation pipeline and predefined severities.
- [ ] Demographic fairness tests use documented ground-truth labels and report uncertainty; otherwise the manuscript explicitly retains fairness as a limitation.
- [ ] Final tables contain confidence intervals and all model-selection decisions were made without the final external test results.
- [ ] The manuscript was rewritten so every claim has a direct table, figure, or experiment supporting it.
- [ ] The paper retains only limitations that are genuinely outside the study scope; none of Reviewer 5's requested experiments are merely promised as future work.

If a checklist item fails, the correct action is either to complete it or narrow the paper's claim. Do not hide a negative result. A well-documented negative external result can still strengthen the paper if the analysis identifies why the shortcut failed and which corrected method improves it.

---

## 10. Full peer-review history: additional closure requirements

The earlier reviews add requirements beyond Reviewer 5's final rejection letter. This table covers the remaining experimental, reproducibility, and presentation items. All row outcomes must be recorded in the project before manuscript revision begins.

| Review theme | Required work | Deliverable / pass condition |
|---|---|---|
| Repeated-run stability | Run each final selected model with three independent seeds. Use the same data manifests and fixed evaluation protocol. | Mean, standard deviation, 95% bootstrap CI, and paired comparison test for MHA versus every fusion control. State the test and number of runs explicitly. |
| Wavelet-level choice | Ablate wavelet packet levels 2, 3, and 4 using the same Wolter architecture/training protocol as far as input size permits. | Table with channel count, parameters, latency, internal-grouped and external-development metrics. Choose level 3 only if evidence supports it. |
| Embedding-dimension choice | Ablate 64-D, 128-D, and 256-D expert embeddings with the same corrected fusion family. | Parameter count, training stability, and generalization metrics. Do not call 128-D optimal unless it wins or presents a documented accuracy-efficiency trade-off. |
| Augmentation design | Evaluate a predefined training augmentation policy: JPEG compression over stated quality ranges, resize/downscale-reencode, blur, noise, and limited color/exposure variation only when appropriate for forensic signals. | Exact probabilities/ranges and an ablation table. Validation/external evaluation must remain deterministic with augmentations off. |
| Frame-rate redundancy | Compare a small set of source-video sampling strides or frame caps, for example spaced samples equivalent to 1 FPS, 5 FPS, and 10 FPS. | Table of unique source videos, frames/video, training cost, and external-development performance. Select a cap/stride by diversity and external results, not raw image count. |
| Frozen-expert protocol | Separate expert training, expert development selection, fusion training, fusion development selection, internal test, and external test. | A flow diagram and split table showing data access at every phase. Fusion training must not tune on an expert validation/test subset that is later reported as an independent fusion test. |
| Architecture reproducibility | Specify input resolution, RGB normalization, wavelet family, level, boundary/padding mode, log scaling, all layer shapes, heads, hidden dimensions, dropout, optimizer, scheduler, loss, batch size, epochs/early stopping, seed, and frozen/unfrozen state. | Formal equations, layer table, configuration file, and matching source-code version. The table must describe the model actually trained. |
| Hardware reproducibility | Do not describe `2 x T4 + RTX 3060` as one synchronized heterogeneous training job unless that job was actually implemented and documented. Treat development, two-T4 distributed training, and L4 verification as separate runs. | Hardware table per experiment: GPU type/count, CUDA, PyTorch, precision mode, distributed strategy, effective batch size, training duration, and seed. |
| Inference cost | Profile both experts plus each fusion model end-to-end, not the fusion head alone. | Parameters, checkpoint size, FLOPs/MACs, GPU peak memory, latency/image, throughput, hardware, batch size, warm-up protocol, and preprocessing inclusion/exclusion. |
| Strong baselines | Create adapters for selected modern generalization-oriented detectors after checking their official repositories and licenses. | Same-manifest evaluation for every runnable baseline. If a method cannot run under the protocol, explain why and do not rank its published result against local results. |
| Robustness completeness | Use the corrected pipeline for JPEG, blur, noise, resize/recompression, exposure/brightness where justified, FGSM, and PGD. Include face re-rendering or purification only if a reproducible, realistic protocol is available. | A compact main-text table plus a complete appendix. Every robustness claim names its tested transformations and severities. |
| Fairness completeness | Obtain an appropriate, documented demographic benchmark and evaluate relevant group intersections. | Group counts, uncertainty, FPR/FNR/AUC/balanced accuracy gaps, and a limitations statement covering unrepresented groups. |
| Dataset truth and consistency | Generate every manuscript dataset count from one authoritative manifest rather than typing totals manually. | A versioned `dataset_summary` table automatically generated from the final manifest. The abstract, methods, tables, supplement, code archive, and response letter must all match it exactly. |
| Figures/tables and result integrity | Rebuild every table and figure from saved evaluation outputs. Review captions, numbering, axes, units, bolding, class-label direction, and stated conclusions. | A figure/table audit checklist. Remove contradictory prose such as defending an unclear axis rather than fixing the figure. |
| Interpretability | Produce readable, annotated Grad-CAM examples and attention-weight summaries for correct and incorrect predictions across several generators/datasets. | Side-by-side visual figure with input, prediction, ground truth, source/generator, and clearly labeled visualization. Treat these as qualitative analysis, not proof of causality. |
| Writing and references | Perform professional copy editing after results are locked. Rebuild related work around spatial, frequency, and multi-domain methods; include only directly relevant suggested references and verify every bibliographic entry from its primary source. | Clean final manuscript; no placeholders, duplicate captions, incorrect venue/year/title, or unsupported "state of the art" label. |
| Conclusion and limitations | Rewrite conclusion only after all results are final. Explain the exact contribution, the observed failure modes, evidence boundaries, computational cost, fairness scope, and remaining limitations. | Conclusion reports findings; it does not promise unperformed experiments as evidence of the present model. |

### Why the earlier revision did not satisfy the final reviewer

The paper did respond to several early requests: it added architecture detail, an embedding ablation, a simple-fusion comparison, a parameter table, reproducibility text, and limited FGSM analysis. However, the final reviewer still found that these additions did not answer the central empirical questions:

1. The original attention implementation does not provide the claimed adaptive choice between the two streams.
2. The simple-fusion comparison alone cannot establish novelty or necessity, especially when simple fusion has slightly higher accuracy.
3. The main results remain an in-domain split of a pooled dataset rather than independent cross-dataset evidence.
4. Split claims were described but not independently auditable at source/generator/near-duplicate level.
5. Robustness and fairness were mostly discussed, not measured comprehensively.

The new work must solve these issues experimentally. Rewording them as limitations is useful for honesty but does not close them.

---

## 11. Final experimental record and manuscript package

For each completed experiment, save a machine-readable run record containing:

```text
run_id, git_commit, manifest_version, split_version, model_id,
expert_checkpoint_hashes, fusion_checkpoint_hash, seed,
hardware, software_versions, preprocessing_config, augmentation_config,
training_duration, selected_epoch, threshold_source, evaluation_dataset,
metrics, confidence_intervals
```

The final submission package must contain:

1. A data card and manifest-generation instructions.
2. Split manifests and automated leakage-audit output.
3. Exact training/evaluation configuration files and environment lockfile.
4. All paper tables generated from saved run records, not manual transcription.
5. A point-by-point response in which every reviewer request links to a manuscript section, table, figure, appendix item, or clearly justified out-of-scope limitation.
6. A concise claim-evidence table used by the authors before submission.

This package directly addresses the repeated concerns about reproducibility, inconsistent counts, unclear architecture, unstable numbers, and manuscript organization.
