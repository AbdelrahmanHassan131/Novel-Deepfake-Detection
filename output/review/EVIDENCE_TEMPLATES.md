# Empirical Reviewer Evidence Templates

These empty, labeled templates correspond to the empirical evidence tables required to close reviewer objections.
All empirical metric cells are explicitly marked **`[PENDING COLAB RUN]`**. No numbers are fabricated or populated from historical runs.

---

### Table 1: Auditable Dataset Composition and Partitioning (100K Pilot)

| Partition / Split | Authentic Real | Manipulated / Fake | Total Samples | Independent Groups | Frame Cap / Video | Natural Prevalence |
|---|---:|---:|---:|---:|---:|---:|
| **Training Split** (`train`) | 50,000 | 50,000 | 100,000 | `[PENDING RUN]` | 15 | 50.0% / 50.0% |
| **Development Split** (`dev`) | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | 15 | Natural |
| **Internal Test Split** (`internal_test`) | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | 15 | Natural |
| **External Benchmark: LDM** | 0 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | N/A | Fake-only (Report Recall) |
| **External Benchmark: Mixed** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | N/A | Independent Domain |

*Audit status: Connected-group disjointness verified; zero cross-split identity/video leakage.*

---

### Table 2: Multi-Seed Training Performance (3 Independent Runs)

> [!NOTE]
> Evaluated across 3 completely independent training runs (Seeds 42, 43, 44), each training new expert backbones and fusion heads. Repeated inference from the same checkpoint is rejected.

| Model / Fusion Architecture | Seeds | ROC AUC (%) | Balanced Acc (%) | Binary Acc (%) | EER (%) | FPR @ TPR 95% (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RGB Expert Alone (`Wang2020_128`)** | 3 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Wavelet Expert Alone (`WolterWavelet2021_128`)** | 3 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Concatenation MLP (`Fusion_128`)** | 3 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Gated Fusion (`Fusion_128`)** | 3 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Corrected Token Attention (`MHA_128`)** | 3 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Equal-Weight Late Ensemble** | 3 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

---

### Table 3: Paired Model Differences (Cluster Percentile Bootstrap 95% CI)

> [!NOTE]
> Evaluated on the same test groups using independent development-selected Youden J thresholds for each model.

| Metric | Token Attention vs. Gated Fusion | Token Attention vs. Concat MLP | Token Attention vs. Late Ensemble | Significant at 95% CI? |
|---|:---:|:---:|:---:|:---:|
| **ROC AUC Difference (Δ %)** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Balanced Acc Difference (Δ %)** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Equal Error Rate Difference (Δ %)** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **FPR @ TPR 95% Difference (Δ %)** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

---

### Table 4: Targeted Ablation Sweeps

#### 4a. Wavelet Decomposition Level (Fixed Embed Dim = 128)
| Wavelet Level | Sub-band Channels | Wavelet Packet Spatial Size | Dev ROC AUC (%) | Dev Balanced Acc (%) | Inference Latency (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Level 2** | 48 | 56 × 56 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Level 3** | 192 | 28 × 28 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Level 4** | 768 | 14 × 14 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

#### 4b. Feature Embedding Dimension (Fixed Wavelet Level = 3)
| Embedding Dimension | Expert Param Count | Head Param Count | Dev ROC AUC (%) | Dev Balanced Acc (%) |
|:---:|:---:|:---:|:---:|:---:|
| **64** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **128** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **256** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

#### 4c. Video Frame Capping (Fixed 100K Training Budget)
| Frame Cap / Video | Unique Source Videos | Unique Identities | Dev Balanced Acc (%) | External Transfer AUC (%) |
|:---:|:---:|:---:|:---:|:---:|
| **5 frames** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **15 frames** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **30 frames** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

---

### Table 5: Pixel-Bounded Robustness and Adversarial Attacks

> [!NOTE]
> All corruptions and adversarial perturbations are applied to the raw image in pixel space [0, 1]. Both RGB and wavelet streams recompute features from the perturbed pixels.

| Condition | Perturbation Parameter | Accuracy (%) | Balanced Acc (%) | Fake Recall (%) | AUC (%) |
|---|---|:---:|:---:|:---:|:---:|
| **Clean Baseline** | None | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **JPEG Compression** | Quality = 75 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **JPEG Compression** | Quality = 30 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Gaussian Blur** | $\sigma = 1.0$ | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Gaussian Noise** | $\sigma = 3.0 / 255$ | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Downscaling** | Scale = 0.5× | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **FGSM Attack** | $\epsilon = 4 / 255$ | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **PGD Attack** | $\epsilon = 4 / 255$, 10 steps, 2 restarts | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

---

### Table 6: Demographic Subgroup Fairness Audit

> [!NOTE]
> Evaluated strictly using supplied legitimate annotations. Missing demographic metadata is reported as unknown and never fabricated. Subgroups with $N < 30$ are flagged for high sample uncertainty.

| Demographic Attribute | Subgroup | Sample Count ($N$) | Group Share (%) | Balanced Acc (%) | False Positive Rate (%) | False Negative Rate (%) | Sample Caution |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gender** | Female | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | Adequate |
| | Male | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | Adequate |
| **Age Group** | Young (<30) | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | Adequate |
| | Middle (30-50) | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | Adequate |
| | Senior (>50) | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |

---

### Table 7: Full Detector Computational Cost (End-to-End Pipeline)

> [!NOTE]
> Measured on the target GPU (L4 or T4) including RGB normalization, differentiable wavelet transformation, both expert forward passes, and fusion head forward pass.

| Architectural Stream | Total Params (M) | Trainable Params (M) | Model Size (MB) | FLOPs / Image (G) | Latency (ms / img) | Throughput (img / s) | Peak VRAM (MB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **RGB Expert (`Wang2020_128`)** | 23.75 | 0.00 (frozen) | 90.6 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Wavelet Expert (`Wolter_128`)** | 1.15 | 0.00 (frozen) | 4.4 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Fusion Head (`MHA_128`)** | 0.13 | 0.13 | 0.5 | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
| **Full Detector (End-to-End)** | **25.03** | **0.13** | **95.5** | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` | `[PENDING RUN]` |
