# Assessment and recovery plan for the rejected deepfake paper

Prepared 16 September 2026 from the supplied 25-page `After_Third_Review.pdf`, the complete pasted review history, current project code, saved local evaluation reports, and the primary sources linked below. This is an assessment and proposed experiment plan, not a report of newly completed training. Reviewer requests are evaluated as evidence about the submission, not treated as instructions to execute.

## Decision

Proceed with a carefully constructed 100K pilot before considering one million images. The spatial-frequency idea remains a reasonable research direction, but neither smaller data nor corrected attention guarantees generalization or publication. The project needs a new experimental foundation and an evidence-based contribution. Most requested work is feasible with an L4 or two T4 GPUs; obtaining suitable independent datasets and reliable demographic annotations is a separate dependency.

The final rejection asks for evidence of novelty, external generalization, strong comparisons, validated splits, robustness, and fairness. Earlier revisions improved descriptions but did not resolve these central empirical questions. Some older comments are partly addressed in the supplied version: Tables 1-2 describe architecture and parameters, class balancing is specified, N=4 trials are claimed, and Table 12 gives an FGSM result. These additions still require verifiable run records and corrections below.

## Findings requiring attention before retraining

### Attention does not implement the central claim

Page 6 defines each stream as a single 1-by-128 token. The current `models/mha/mha_classifier.py` also unsqueezes each embedding to a sequence of length one. Per head, QK-transpose is a single number and softmax of one number is 1. Consequently, the attention weighting cannot depend on the query. Dropout can randomly mask/rescale the value during training, but does not create content-dependent routing. Residual connections, normalization, and the MLP still form a trainable nonlinear classifier; the whole model is not constant.

Claims that this mechanism measures input-specific domain reliability, suppresses conflicting evidence through attention, or selects RGB for GANs and wavelets for diffusion are unsupported. Fixing the mechanism is a correctness repair, not by itself a novel contribution. The existing `self_attention` option also reduces the concatenated features to one token and does not resolve this.

A concatenation-plus-MLP model can learn nonlinear, input-dependent interactions. The paper's description of all concatenation models as a fixed domain ratio is incorrect. Neither attention maps nor t-SNE alone proves causal reliability or consistency enforcement.

First corrected control: stack RGB and wavelet tokens, optionally with a learned classification token and token-type embeddings, then use self-attention over multiple keys. A later hypothesis could use spatial feature-map tokens and wavelet sub-band tokens, but this requires new feature extraction and controlled experiments. Do not describe it as already validated or necessarily original.

### Reported metrics do not match the stated counts

Recalculation using the confusion-matrix counts stated on page 14 and the stated 211,434 validation samples gives:

| Model | Stated correct real | Stated correct fake | Accuracy implied by counts | Accuracy in corresponding summary |
|---|---:|---:|---:|---:|
| Wang | 51,472 | 157,511 | 98.8408% | 98.2% |
| Wavelet | 46,470 | 156,702 | 96.0924% | 95.42% |
| MHA | 51,288 | 157,955 | 98.9637% | 98.82% |

Table 11 additionally lists modified Wang accuracy as 98.73%. Some precision/F1 values also fail to follow the stated counts. The stated Wang macro-F1 equals its listed real-class F1, rather than the average of its two listed class F1 scores. These are inconsistencies, not grounds to invent replacement results: recover the exact checkpoint and prediction file for every table, then regenerate the complete metrics together.

The current local MHA reports concern 32,844 samples, unlike the paper's 211,434. They cannot be assumed to be the raw evidence for the manuscript. Table 4 totals add up, but the counts imply approximately 79.21%/20.79%, not an exact 80/20 split. Approximate grouped splits are acceptable; report actual percentages.

Page 20 reports N=4 and MHA mean/SD but does not supply all per-run fusion results, CI endpoints, or the calculation procedure. An EER improvement of 1.32% to 1.27% is 0.05 percentage points; the reported MHA run SD is 0.15 percentage points. Paired results are essential before claiming a stable advantage. Non-overlap of unspecified marginal intervals is not a substitute for a documented paired comparison.

### Data and implementation risks

- `data/builders/dataset_factory.py` uses ImageFolder separately inside top-level folders when nested folders exist. Depending on actual layout, generator/source folders become target classes. Plain `fake/real` gives fake=0, real=1; `0_real/1_fake` gives the opposite. Audit each historical run and introduce explicit binary labels. Do not simply reverse labels on an external set to improve scores.
- The paper says most authentic faces come from CelebA while many fakes are video crops or other repositories. Source/crop/compression shortcuts are therefore plausible, but cannot be proven without the actual manifests and samples.
- `models/mha/trainer.py` permits missing expert paths and continues with untrained backbones. Mandatory checkpoints, hashes, architecture checks, and reload prediction parity should precede fusion experiments.
- `train.py` copies training options into validation, and augmentation functions do not check `isTrain`. Nonzero blur/JPEG probabilities can therefore persist in training-time validation. Disable random augmentation and class-balanced resampling for validation/test explicitly.
- Evaluation can reconstruct preprocessing from defaults. Persist and restore the exact resize/crop, normalization, pixel scale, wavelet family/level/mode, log scaling, and coefficient ordering used by each checkpoint. Verify CPU/GPU wavelet agreement before switching implementations.
- Precomputed wavelets can remain fixed while RGB undergoes random augmentation. Both streams must correspond to the same transformed image, or this difference must be an explicit experimental condition.
- In `evaluate_robustness.py`, FGSM/PGD clamp normalized RGB inputs to [0,1] and preserve clean wavelets in fusion inputs. This is not a valid end-to-end pixel-bounded attack on the detector. Existing numbers need revalidation against the actual version used.

## Constructing the 100K pilot

Interpret 100K as 100K training images: 50K authentic and 50K manipulated/generated, with additional independent development and internal-test groups. If the storage/data budget is 100K total, use approximately 80K/10K/10K and accurately call it an 80K training experiment.

1. Inventory available sources before selecting quotas. Record exact dataset/version, manipulation or generator, real/fake class, original source video, identity if known, frame, preprocessing, and hash. Zenodo is a hosting platform, not a generator label. Preserve unknown metadata as unknown.
2. Include authentic originals from the same video sources as manipulated examples whenever available. Diversify real images across source styles; do not make the real class synonymous with photographs and the fake class synonymous with video crops.
3. The cited [DiffFace part1 record](https://zenodo.org/records/10865300) lists `Real.tar` alongside ADM, DDIM, DDPM, and DiffSwap archives. Inspect its provenance and overlap with CelebA before using it. The archive name alone does not establish authenticity, sample pairing, identity independence, or source matching.
4. Use roughly balanced quotas across verified fake families, while preserving the distinction between fully synthetic faces, face swapping, reenactment, and editing. For example, five eligible families could contribute roughly 10K each. Do not treat all video manipulations as GAN generation, or two sampling methods as independent generators without checking their parent model and training source.
5. Group originals, derived fakes, recompressions, and identity-linked samples before splitting. Build connected groups where multiple source identities/videos contribute to one fake. Run exact-duplicate and near-duplicate audits across repositories as well as splits. Video-disjoint does not automatically mean identity-disjoint.
6. Cap well-spaced frames per source video so a few videos cannot dominate. Compare predefined sparse and dense sampling policies at controlled image counts and report the number of independent videos/identities. Ten FPS does not by itself establish useful diversity.
7. Apply the same face detection, crop policy, RGB handling, and resize conventions to both classes. Record detection failures and exclusions. A frame-level face detector should report its eligible face subset and coverage on general web-image benchmarks.
8. Audit source-by-class distributions. A 50/50 class total does not eliminate source confounding. For a balanced pilot, ordinary shuffled training may suffice; if weighting is used, document sampling probabilities and effective exposure per source/class. Inverse-frequency random sampling produces balance in expectation, not exactly in every batch.

## Development and final evaluation must have separate roles

Choose an external-development set for debugging and selecting augmentation, model, threshold, and training scale. Existing external sets repeatedly inspected to improve the method belong in development. Reserve distinct final datasets or genuinely independent held-out domains before the pilot. Do not test at 100K, tune until it works, and then call those same datasets untouched at 1M.

Use an internal grouped test, leave-one-generator-family-out, and leave-one-source-out where metadata supports them. Every leave-one-out fold must exclude the held-out family/source from both expert training and fusion development. Reusing experts previously trained on the held-out generator invalidates the claim. Distinguish ordinary in-domain splits, where generators can occur in both partitions, from experiments specifically designed to measure unseen generators.

The paper already uses Celeb-DF-related and FakeAVCeleb material. These names cannot automatically serve as independent external tests. Audit identities and original media across all training sources. If the claim covers both synthetic faces and facial manipulations, include a final evaluation for each; DFDC alone does not establish unseen diffusion synthesis detection. The [Deepfake-Eval-2024 paper](https://arxiv.org/abs/2503.02857) documents an in-the-wild benchmark, but its scope and eligible subset must be matched to this face-image detector.

Primary comparison: per-dataset ROC AUC, balanced accuracy at a development-selected threshold, and real-image false-positive rate. Also report PR AUC/AP with the precise definition, per-class precision/recall/F1, EER, and TPR at predefined low FPR or FPR at a predefined TPR. EER is a descriptive curve statistic, not an optimal deployment threshold. AUC measures ranking and is not proof of calibration, corruption robustness, or generalization by itself.

For video-derived data, aggregate frame probabilities using a fixed rule and report video-level results as well as frame-level results. Bootstrap independent source-video/identity groups, keeping related frames together. Report paired metric differences on the same test groups, plus variability across independent training seeds. Sample uncertainty and seed variability are different quantities. Use at least three full-pipeline seeds for final comparisons, or four if retaining the manuscript's four-trial claim; repeated inference from the same checkpoint is not an independent training run.

## Model and training sequence

| Stage | Work | Decision/evidence |
|---|---|---|
| A | Fix label/preprocessing/checkpoint/metric issues; run a small smoke experiment | Valid labels, disjoint groups, deterministic evaluation, exact checkpoint reload |
| B | Train RGB and wavelet experts on the 100K training pool | Expert-only internal and external-development results; class/source error galleries |
| C | Freeze one selected expert pair per seed; compare concatenation MLP, gated fusion, late score ensemble, corrected multiple-token attention, and a comparable-capacity Transformer/control | Same data access, expert weights, tuning allowance, augmentations, epoch-selection protocol, and paired predictions |
| D | Test leave-one-generator/source-out and selected strong baselines | Retrain experts for held-out folds; document any baseline-specific auxiliary supervision |
| E | Run targeted wavelet-level, embedding-size, augmentation, frame-sampling, and tokenization ablations | Evidence of what changes external-development performance and cost |
| F | Expand only the strongest justified protocols to 200K, then 400K; consider 1M if gains justify cost | Fixed development/test groups and source balance; nested training pools where practical |
| G | Freeze choices, run independent final external tests, robustness and subgroup audits, rebuild manuscript | Fully traceable results and bounded claims |

Retain the original single-key model only as a clearly labeled legacy/control configuration. A good baseline may win. If the experts both fail externally, start with data composition and representation learning. If an expert works but fusion fails, inspect expert restoration, feature scaling, wavelet parity, and head overfitting. If gated or late fusion matches attention, use that result rather than claiming attention is necessary.

Modern baseline shortlist: a frozen pretrained-feature detector such as [UniversalFakeDetect](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html); a facial-forgery generalization baseline such as [UCF](https://arxiv.org/abs/2304.13949) or [SBI](https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html); and a more recent adaptation method such as [Effort](https://arxiv.org/abs/2411.15633). These cover different scientific questions; use appropriate tasks and follow their prescribed training needs. [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) provides implementations including UCF, SBI, LSDA, and Effort. Freeze the final shortlist after code/license/resource checks and a current literature check. Do not call this shortlist an exhaustive 2026 ranking. Keep published-checkpoint and same-data retraining comparisons separately labeled.

A possible stronger research direction is to test whether spatial-region and wavelet-band interactions improve transfer under controlled degradation. Another is whether preserving broad pretrained RGB features avoids the dataset-specific bottleneck learned by the current expert. These are hypotheses. More layers or a new loss do not establish novelty without a prior-work comparison and ablations.

Wavelet levels 2/3/4 yield 48/192/768 channels, but spatial packet dimensions shrink as channels increase. Do not infer total coefficient memory solely from channel count. At 224-pixel Haar input, level 4 gives 14-by-14 packets: the current four 2-by-2 max-pools reduce 14 to 7 to 3 to 1 to zero. A controlled pooling/input adjustment is required for that ablation; it must be documented, not interpreted as evidence that level 4 is inherently inferior.

## Complete reviewer coverage

R2/R3 numbers refer to their initial numbered comments; R4 distinguishes minor and major lists. R5 refers to the later rounds and final rejection.

| Concern and review source | Required response | What would count as evidence |
|---|---|---|
| Novelty and necessity: R3.1, R4 major a/d, R5 final | Correct attention; compare concat, gate, late ensemble, genuine cross/self-attention and capacity controls | External paired comparisons and ablations; novel mechanism or a clearly bounded empirical contribution |
| Cross-dataset and cross-generator: R2.1, R3 opening, R4 major b, R5 all rounds | Separate development domains from final tests; run source/generator holdouts | Per-dataset results with training-exposure audit, including appropriate synthesis/manipulation coverage |
| Phase overlap: R2.2, R5 later protocol table | Record sample/group access for experts, head, epoch selection, thresholds, test | Versioned split manifests; test untouched by all training and selection. Shared development data is allowed if reported as development |
| Balance and counts: R2.3/12, R3.4, R4 minor 3 | Generate counts and source distributions from one manifest; specify sampler | Consistent totals, actual split percentages, unique groups and class exposure |
| Repeated runs and statistics: R2.4, R4 major f, R5 statistical comment | Retrieve four historical run records or replace unsupported claims with new independent runs | Per-run metrics for both methods, seed mean/SD, group CIs and paired difference/test method |
| Formal architecture: R2.5, R3.2, R4 major e | Align equations, tensor shapes, diagram, parameters, code and checkpoint | Correct multiple-key attention; complete preprocessing and expert/head specifications |
| Wavelet levels: R2.6 | Compare 2/3/4 under a feasible controlled architecture | Generalization and computational table; no unsupported theoretical optimum |
| Embedding dimension: R2.7, R4 major d | Reproduce 64/128/256 at expert and head stages | Numerical table and saved runs; current prose-only ablation is insufficient |
| Hardware/software: R2.8, R4 major e | Distinguish per-run hardware, effective batch, precision, versions and distributed settings | Actual environment lockfiles and logs; no pooling of unrelated GPU configurations |
| Augmentation: R2.9, R5 robustness rounds | Reconcile fixed JPEG 75 versus random 30/75 and blur/noise claims; disable evaluation randomness | Exact policy and train-only ablations, plus separate inference-time corruption tests |
| Ten-FPS redundancy: R2.10 | Compare frame caps/spacing at controlled budgets | Unique videos/identities, sampling policy, external results and cost |
| Figure/caption/axis issues: R2.11/13, R3.5, R4 minor 2 | Rebuild plots, diagram and captions from saved evidence | Readable axes, consecutive references, annotated errors and consistent labels; no prose defending ambiguous plots |
| Numeric contradictions and bolding: R3.6/7, R4 dataset concerns | Recompute every table and narrative comparison | Counts reproduce all reported metrics; correct winners, units and macro/per-class definitions |
| Demographic fairness: R2.14, R5 final | Acquire suitable annotated evaluation data; audit age/gender/ethnicity/skin tone where legitimately supplied | Per-group counts, FPR/FNR/AUC/balanced accuracy and uncertainty at the same threshold; small/missing groups disclosed |
| Attacks/anti-forensics: R2.15, R4 minor 5, R5 final | Correct pixel-space FGSM/PGD; recompute all modalities; define corruption severities | Clean/JPEG/blur/noise/resize/recompression results and attack budgets/steps/restarts; rerendering/purification only with a reproducible justified setup |
| Inference efficiency: R2.16/17, R3.8, R4 minor 5 | Profile experts, transforms and fusion end-to-end | Total versus trainable parameters, model size, peak memory, latency/throughput, FLOPs/MACs and measured hardware |
| Recent baselines: R3.3, R4 major c, R5 final | Evaluate recent generalization methods under controlled protocols | Comparable data access/pretraining/tuning; no ranking local metrics against unrelated published scores |
| Writing/conclusion: R1, R4 minor 1/5, R5 clarity | Rewrite after experiments; remove repetition and unsupported universal/deployment claims | Claim-evidence table, clear limitations, precise contributions and careful copy edit |
| References: R4 minor 4, R5 suggestions, editor note | Verify every retained reference and direct relevance | Correct bibliographic records and in-text targets. The editor explicitly says unrelated suggested additions are discretionary |

Fairness measurement and fairness improvement are different outcomes. If annotations cannot be obtained, record this as an unresolved reviewer concern and narrow claims. Do not fabricate demographic labels or promise that balancing classes guarantees subgroup fairness. Likewise, a broader experimental study does not guarantee a journal will judge the method novel enough.

## Manuscript corrections beyond the review matrix

- Figure 3 has repeated Linear(256-to-128) boxes despite an intervening 128-D representation, and labels 64 wavelet channels instead of clearly distinguishing 64 bands per color from 192 total channels. Redraw from the implemented network.
- Figure 6 is a training-loss plot, whereas surrounding prose attributes validation accuracy/precision or AP behavior to it. Figure 5 also needs its long connecting segments checked against chronological logging. Plot the actual metric named in each claim.
- Page 25 reproduction commands are visibly clipped; the repository URL is a placeholder and the requirements-file route must match files actually shipped. Main reproduction commands should recreate the two-stream paper model, not a different multi-expert variant.
- Page 10 cites CelebA as reference 15, while reference 15 is Deepfake-Eval-2024. Check all citation targets. The full bibliography has not been independently validated in this assessment.
- The paper alternates direct 224 resize and 256 resize plus 224 crop. Resolve this from run configs. The supplied code defaults do not by themselves prove what historical training used.
- Do not claim freezing experts guarantees generalization; it preserves both useful features and their existing biases. Adding a third expert is not automatically plug-and-play in the current fixed two-branch architecture.
- Separate documented observations from causal interpretations. Grad-CAM cannot alone identify a highlighted region as a specific diffusion-noise artifact or prove attention chooses reliable evidence.

## GPU feasibility and scaling decisions

An [L4 has 24 GB](https://www.nvidia.com/en-au/data-center/l4/); a [T4 has 16 GB](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf). Start development on the L4 for per-device memory headroom and a simpler setup. Two T4s can run independent expert/seed jobs or distributed data-parallel training; ordinary data parallelism retains a full replica on each device and does not create one 32 GB memory pool. Measure both options rather than promising that either is always faster.

The current ResNet50/wavelet experts and compact heads are plausible workloads on these devices with suitable batch sizes and mixed precision. This is a feasibility estimate, not a benchmark. Full fine-tuning of larger foundation models requires separate memory checks; frozen-feature baselines and efficient adaptation are useful alternatives.

Benchmark sustained throughput after warm-up, including image decoding, augmentation and wavelets. Estimate epoch time as training samples divided by measured samples/second, then add validation/checkpoint overhead. Avoid extrapolating from the existing fusion-only FLOPs or inference rates. A million images mainly increases steps, storage and I/O, not parameter memory, when batch and architecture stay fixed.

For screening, use one seed on a small set of sensible candidates. Repeat shortlisted full pipelines with independent seeds before claims. Deterministic embedding caching speeds frozen-head comparisons, but must not silently remove augmentation or compare cached and augmented models as if protocols were identical.

Scale from 100K to 200K/400K only if source-held-out/external-development results justify it and added data increases meaningful diversity. Keep final test sets locked. There is no universal 80% accuracy pass rule: consider per-dataset AUC/balanced accuracy, false positives, uncertainty, and comparison with strong baselines. A successful 100K study may be sufficient without 1M. If gains plateau, spend compute on error analysis and stronger data/representations. If attention does not improve reliably, revise the contribution around the simpler successful method or a rigorous analysis of why fusion fails.

Immediate next work: build the source inventory and explicit-label/group audit, recover the historical prediction/run records, enforce expert checkpoint restoration, and correct the evaluation pipeline. Then run the 100K expert pilot and matched fusion controls. Large retraining and manuscript rewriting should follow this evidence.
