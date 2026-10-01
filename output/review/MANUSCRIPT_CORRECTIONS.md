# Manuscript Corrections: Equations, Architecture, Claims & Tables

This document records all required corrections to mathematical formulas, architectural diagrams, empirical claims, and reported numbers in the manuscript (`After_Third_Review.pdf`).

---

## 1. Attention & Fusion Claims (Core Conceptual Repair)

### Erroneous Manuscript Claim:
> *"The proposed Multi-Head Attention mechanism performs dynamic routing between spatial RGB and frequency wavelet domains, adaptively measuring input-specific domain reliability and filtering out conflicting cues, whereas concatenation-based fusion is inherently non-adaptive and imposes a static domain ratio."*

### Mathematical Reality & Required Correction:
1. **Single-Key Attention Flaw**:
   - In the historical codebase and Eq. 3 of page 6, each stream embedding was unsqueezed into a sequence of length 1 ($1 \times 128$).
   - For a single key vector ($N=1$), the attention score matrix $QK^T \in \mathbb{R}^{1 \times 1}$ is a scalar.
   - The softmax of a single scalar is identically $1.0$ ($\text{softmax}([s]) = 1.0$), regardless of query, key, or input content.
   - Therefore, single-key attention mathematically cannot perform content-dependent routing, selective weighting, or suppression of domain cues.
2. **Concatenation MLP Adaptiveness**:
   - Describing concatenation followed by an MLP as "inherently nonadaptive" is mathematically false. An MLP with non-linear activations (ReLU/GELU) over concatenated features $[z_{rgb}; z_{wav}]$ learns non-linear, input-dependent interactions between spatial and frequency features.
3. **Corrected Manuscript Description**:
   - Delete all claims that concatenation models enforce a "fixed ratio" or are "non-adaptive".
   - State clearly that the corrected model implements **multiple-key token attention** ($\{t_{rgb}, t_{wav}, t_{cls}\}$), where true self-attention over multiple tokens produces non-trivial query-dependent attention distributions.
   - Frame the corrected attention mechanism as an empirical hypothesis tested against strong controls (Gated Fusion, Concat MLP, Late Ensemble), not an axiomatic superiority.

---

## 2. Mathematical Equations Alignment

### Replace Historical Single-Token Formulation:
Historical Eq. (3):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
Where $Q, K, V \in \mathbb{R}^{1 \times d}$ resulted in trivial scalar identity.

### Corrected Multiple-Token Formulation:
Let $z_{rgb} \in \mathbb{R}^{d}$ and $z_{wav} \in \mathbb{R}^{d}$ be the linear projection embeddings of the frozen RGB and Wavelet backbones. Construct the sequence of tokens:
$$T = \begin{bmatrix} z_{cls} + e_{cls} \\ z_{rgb} + e_{rgb} \\ z_{wav} + e_{wav} \end{bmatrix} \in \mathbb{R}^{3 \times d}$$
Where $z_{cls} \in \mathbb{R}^{d}$ is a learned classification token, and $e_i \in \mathbb{R}^{d}$ are learned modality-type embeddings.
For each attention head $h \in \{1, \dots, H\}$:
$$Q_h = T W_Q^{(h)}, \quad K_h = T W_K^{(h)}, \quad V_h = T W_V^{(h)} \in \mathbb{R}^{3 \times d_k}$$
$$A_h = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{3 \times 3}$$
Since $N=3 \ge 2$, $A_h$ is a genuine $3 \times 3$ attention matrix whose off-diagonal interaction terms vary as a function of the input representations.

---

## 3. Historical Inconsistencies & Numerical Contradictions

The numbers reported in the historical manuscript cannot be defended or patched; they contain mathematical contradictions between confusion counts and reported summary statistics:

| Model | Stated Correct Real ($TN$) | Stated Correct Fake ($TP$) | Total Stated Samples ($N$) | Exact Accuracy Implied by Counts | Accuracy Claimed in Manuscript Summary | Discrepancy |
|---|---:|---:|---:|---:|---:|---|
| **Wang2020** | 51,472 | 157,511 | 211,434 | **98.8408%** | 98.20% | $-0.64\%$ discrepancy |
| **Wavelet** | 46,470 | 156,702 | 211,434 | **96.0924%** | 95.42% | $-0.67\%$ discrepancy |
| **MHA** | 51,288 | 157,955 | 211,434 | **98.9637%** | 98.82% | $-0.14\%$ discrepancy |

Additional historical discrepancies:
- Table 11 additionally claimed modified Wang accuracy was 98.73%.
- Stated Wang macro-F1 was reported as equal to its real-class F1, rather than the arithmetic mean of real and fake class F1 scores.
- **Resolution**: Mark all historical numbers as **UNREPRODUCED**. Regenerate the complete metrics, confusion counts, ROC AUC, balanced accuracy, and F1 scores together from the new 100K pilot runs.

---

## 4. Diagram Corrections (Figures 3, 5, 6)

1. **Figure 3 (System Architecture)**:
   - *Error in old submission*: Visibly duplicated `Linear(256 -> 128)` boxes despite an intervening 128-D representation; labeled "64 wavelet channels" instead of distinguishing 64 sub-bands per color channel (192 total channels).
   - *Correction*: Redraw directly from `models/mha/token_fusion.py` showing:
     - RGB Backbone (`ResNet50` -> 2048-D -> `Linear(2048 -> 128)` -> 128-D).
     - Wavelet Backbone (`WaveletPacketCNN` -> 192 input channels -> 512-D -> `Linear(512 -> 128)` -> 128-D).
     - Stacking into 3 tokens ($[z_{cls}, z_{rgb}, z_{wav}]$) -> Multi-Head Attention -> Classification Head.
2. **Figure 5 (Training Convergence)**:
   - Replace coarse line segments with actual epoch-by-epoch loss and validation AUC curves logged chronologically from the ExperimentLogger.
3. **Figure 6 (Loss vs. Metric Ambiguity)**:
   - The old caption and prose described Figure 6 as demonstrating validation accuracy / average precision behavior, but plotted cross-entropy loss.
   - Ensure the plotted curve exactly matches the named metric in the caption and text.

---

## 5. Scope of Claims and Generalization Integrity

1. **No Universal Detection Claims**:
   - Deepfake detection is an open, adversarial problem. The method must be presented as an empirical investigation of spatial-frequency token fusion on 100K pilot benchmarks, not a universal solution for all generative models.
2. **Fake Recall on LDM Benchmark**:
   - The LDM external dataset is fake-only (0 real samples).
   - Never report binary accuracy or ROC AUC for LDM (they are undefined). Report **Fake Recall / Miss Rate** at development-calibrated thresholds.
3. **Demographic Fairness**:
   - Fairness reporting must strictly rely on legitimately supplied annotations. Disclose missing demographic annotations as unknown; never fabricate labels.
