# Review Note: Corrected Interpretations and Boundary Conditions

**Document:** `CORRECTED_DIAGNOSTIC_INTERPRETATION_NOTE.md`  
**Location:** `output/review/archive_investigation/CORRECTED_DIAGNOSTIC_INTERPRETATION_NOTE.md`  
**Reference:** [RGB_NEXT_CONTROLLED_PILOT_HANDOFF.md](file:///g:/Master's%20Of%20Science%20Computer%20Engineering/thesis%20deepfake%20detection/Thesis%20Revisions%20SP/First%20revesion/Novel%20Deepfake%20Detection/RGB_NEXT_CONTROLLED_PILOT_HANDOFF.md)  
**Date:** 2026-10-09  

---

## 1. Formal Qualifications and Corrections

Following reviewer audit of the bounded format diagnostic outputs, the following qualifications are formally established:

1. **External Data Status (`F:\val to be deleted 2`):**
   - The dataset located at `F:\val to be deleted 2` has been repeatedly probed, evaluated, and inspected across multiple diagnostic iterations.
   - It is strictly **external development data**, not an untouched, pristine unseen holdout set. Any future scientific generalization claims will require evaluation on a completely uninspected, unoptimized evaluation benchmark.

2. **Retraction of Speculative Domain Claims (Demographics and Frequency Components):**
   - The diagnostic demonstrated that JPEG re-encoding does not resolve false positives on external real face crops. However, no demographic analysis was conducted, and the diagnostic did not perform spectral/frequency decomposition.
   - Statements asserting that "demographics or high-frequency spectral bands dominate the failure" are formally retracted. The experiment establishes observational sensitivity to compression interventions, but does not identify the specific underlying visual or semantic features driving model predictions.

3. **Limits of One-Step Compression Interventions:**
   - A single-step in-memory JPEG compression round-trip (e.g. quality 75 or 95) cannot simulate or replicate the unknown, multi-stage compression and resizing history of an unverified source dataset.
   - The observation that a single JPEG round-trip fails to eliminate external real false positives does not logically exclude all historical compression-related confounding.

4. **Scope of Lossless Controls:**
   - The exact 0.0 pixel and tensor equality demonstrated by the PNG control applies strictly to the specific in-memory Pillow 12.2.0 round-trip and PyTorch evaluation pipeline tested. It should not be generalized to claim that every image decoder, container, or operating system produces identical floating-point tensors.

5. **Existing Augmentation Baseline:**
   - The R2 model already employed the `rgb_v1` augmentation recipe during training (JPEG compression with probability 0.5 and qualities 50..95; Gaussian blur with probability 0.5 and sigma 0..3).
   - Online JPEG augmentation is therefore not a novel proposed intervention, and must remain fixed as a baseline invariant in the controlled mixture experiment.

6. **Scope of Diagnostic Training Samples:**
   - The 64 archived-training samples evaluated in the format diagnostic were drawn deterministically from the extracted archive training pool (`Preapred Dataset/train`), not verified as members of R2's specific 100K training subset.
