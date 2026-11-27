# -*- coding: utf-8 -*-
"""Render crisp publication-grade Figure 1 (Decoupled Dual-Stream Architecture) via SVG + headless Chrome."""
import os
import tempfile
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATHS = [
    REPO_ROOT / "manuscript" / "paper1_discover_ai" / "figures" / "fig1_architecture.png",
    REPO_ROOT / "manuscript" / "paper1_discover_ai" / "overleaf_ready" / "figures" / "fig1_architecture.png",
]
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; background: #ffffff; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
  svg text { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
</style>
</head>
<body>
<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="630" viewBox="0 0 1360 630">
  <defs>
    <marker id="arr-dark" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155"/>
    </marker>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#1d4ed8"/>
    </marker>
    <marker id="arr-orange" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#c2410c"/>
    </marker>
    <marker id="arr-grey" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b"/>
    </marker>
  </defs>

  <rect width="1360" height="630" fill="#ffffff"/>

  <!-- Main Title -->
  <text x="680" y="34" text-anchor="middle" font-size="18.5" font-weight="700" fill="#0f172a">
    Decoupled Dual-Stream Spatial–Frequency Architecture with Multi-Token Attention (MHA_128) &amp; Gated Fusion
  </text>

  <!-- Input Box -->
  <rect x="20" y="255" width="155" height="120" rx="12" fill="#f8fafc" stroke="#475569" stroke-width="1.8"/>
  <text x="97.5" y="296" text-anchor="middle" font-size="14" font-weight="700" fill="#0f172a">Input RGB Image</text>
  <text x="97.5" y="324" text-anchor="middle" font-size="12" fill="#334155">224 × 224 × 3</text>
  <text x="97.5" y="344" text-anchor="middle" font-size="11.5" fill="#475569">Forensic Aug (Train)</text>

  <!-- Phase 1 Dashed Container -->
  <rect x="205" y="56" width="585" height="546" rx="14" fill="none" stroke="#64748b" stroke-width="1.5" stroke-dasharray="7,5"/>
  <text x="497.5" y="80" text-anchor="middle" font-size="13.5" font-weight="700" fill="#334155">
    Phase 1: Independent Expert Training → Strictly Frozen in Phase 2
  </text>

  <!-- Spatial Stream Box -->
  <rect x="228" y="110" width="225" height="135" rx="12" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.8"/>
  <text x="340.5" y="142" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">Spatial Stream</text>
  <text x="340.5" y="160" text-anchor="middle" font-size="13.5" font-weight="700" fill="#1e3a8a">Wang2020_128</text>
  <text x="340.5" y="186" text-anchor="middle" font-size="11.8" fill="#334155">ResNet-50 (ImageNet)</text>
  <text x="340.5" y="205" text-anchor="middle" font-size="11.8" fill="#334155">GAP → 2048-D</text>
  <text x="340.5" y="224" text-anchor="middle" font-size="11.8" fill="#334155">23.51M params</text>

  <!-- Spatial 128-D Bottleneck -->
  <rect x="498" y="115" width="268" height="125" rx="12" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.8"/>
  <text x="632" y="148" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">128-D Bottleneck</text>
  <text x="632" y="176" text-anchor="middle" font-size="11.8" fill="#1e293b">Linear(2048 → 128) + ReLU</text>
  <text x="632" y="196" text-anchor="middle" font-size="11.8" fill="#334155">Dropout(0.5) [Phase 1 Train]</text>
  <text x="632" y="217" text-anchor="middle" font-size="11.8" font-weight="600" fill="#1e3a8a">Frozen: 23,770,304 params</text>

  <!-- 3-Level Haar WPD Box -->
  <rect x="228" y="290" width="225" height="135" rx="12" fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/>
  <text x="340.5" y="320" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">3-Level Haar WPD</text>
  <text x="340.5" y="338" text-anchor="middle" font-size="13" font-weight="700" fill="#9a3412">(GPU, db1, reflect)</text>
  <text x="340.5" y="365" text-anchor="middle" font-size="11.8" fill="#334155">signed_log1p(C)</text>
  <text x="340.5" y="384" text-anchor="middle" font-size="11.8" fill="#334155">3 × 4³ = 192 sub-bands</text>
  <text x="340.5" y="403" text-anchor="middle" font-size="11.8" fill="#334155">Tensor: 192 × 28 × 28</text>

  <!-- Wavelet CNN Backbone -->
  <rect x="228" y="458" width="225" height="122" rx="12" fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/>
  <text x="340.5" y="488" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">Wavelet CNN Backbone</text>
  <text x="340.5" y="514" text-anchor="middle" font-size="11.5" fill="#334155">4×[Conv3×3 + BN + ReLU + Pool]</text>
  <text x="340.5" y="534" text-anchor="middle" font-size="11.8" fill="#334155">192 → 64 → 128 → 256 → 512</text>
  <text x="340.5" y="554" text-anchor="middle" font-size="11.8" fill="#334155">GAP → 512-D</text>

  <!-- Frequency 128-D Bottleneck -->
  <rect x="498" y="458" width="268" height="122" rx="12" fill="#fed7aa" stroke="#c2410c" stroke-width="1.8"/>
  <text x="632" y="489" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">128-D Bottleneck</text>
  <text x="632" y="516" text-anchor="middle" font-size="11.8" fill="#1e293b">Linear(512 → 128) + ReLU</text>
  <text x="632" y="536" text-anchor="middle" font-size="11.8" fill="#334155">Dropout(0.5) [Phase 1 Train]</text>
  <text x="632" y="557" text-anchor="middle" font-size="11.8" font-weight="600" fill="#9a3412">Frozen: 1,727,424 params</text>

  <!-- Phase 2 Dashed Container -->
  <rect x="860" y="56" width="480" height="546" rx="14" fill="none" stroke="#15803d" stroke-width="1.6" stroke-dasharray="7,5"/>
  <text x="1100" y="80" text-anchor="middle" font-size="13.5" font-weight="700" fill="#166534">
    Phase 2: Cross-Fitted Sample-Dependent Fusion
  </text>

  <!-- MHA_128 Block -->
  <rect x="886" y="106" width="428" height="222" rx="12" fill="#f0fdf4" stroke="#15803d" stroke-width="1.8"/>
  <text x="1100" y="136" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
    MHA_128 (TokenAttentionFusion, 199,425 params)
  </text>
  <text x="1100" y="170" text-anchor="middle" font-size="11.8" fill="#1e293b">1. LayerNorm(X_s), LayerNorm(X_f) → [CLS, X_s, X_f] (L=3)</text>
  <text x="1100" y="195" text-anchor="middle" font-size="11.8" fill="#1e293b">2. Add Learnable Token-Type Embeddings (3 × 128)</text>
  <text x="1100" y="220" text-anchor="middle" font-size="11.8" fill="#1e293b">3. 4-Head Self-Attention (D_k=32, softmax over L=3)</text>
  <text x="1100" y="245" text-anchor="middle" font-size="11.8" fill="#1e293b">4. Residual + LN → FFN(128→512→128, GELU) → Residual + LN</text>
  <text x="1100" y="270" text-anchor="middle" font-size="11.8" fill="#1e293b">5. Extract [CLS] token → Linear(128 → 1) Forgery Logit</text>
  <text x="1100" y="302" text-anchor="middle" font-size="11.3" font-style="italic" fill="#166534">Best calibration (unseen ECE = 0.039) &amp; #1 PNG-only control (98.90%)</text>

  <!-- Matched Fusion Controls Block -->
  <rect x="886" y="356" width="428" height="226" rx="12" fill="#faf5ff" stroke="#7e22ce" stroke-width="1.8"/>
  <text x="1100" y="386" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
    Matched Fusion Controls (Same Frozen X_s, X_f)
  </text>
  <text x="1100" y="418" text-anchor="middle" font-size="11.7" fill="#1e293b">• GatedFusion (41,987 params): Softmax gate g(X_s‖X_f)</text>
  <text x="1100" y="436" text-anchor="middle" font-size="11.5" fill="#475569">weighted sum → MLP(128→64→1) [Best unseen Acc: 85.71%]</text>
  <text x="1100" y="462" text-anchor="middle" font-size="11.7" fill="#1e293b">• ConcatMatchedMLP (198,017 params): 256→384→256→1</text>
  <text x="1100" y="486" text-anchor="middle" font-size="11.7" fill="#1e293b">• NoAttnFFN (198,657 params): Proj(256→128)+ResFFN</text>
  <text x="1100" y="510" text-anchor="middle" font-size="11.7" fill="#1e293b">• LinearMix (198,785 params): Linear cross-mix + FFN</text>
  <text x="1100" y="534" text-anchor="middle" font-size="11.7" fill="#1e293b">• Original Fusion_128 (107,009 params): 256→256→128→64→1</text>

  <!-- Phase 1 Internal Arrows -->
  <line x1="175" y1="292" x2="225" y2="188" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
  <line x1="175" y1="335" x2="225" y2="355" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
  <line x1="340.5" y1="425" x2="340.5" y2="454" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
  <line x1="453" y1="178" x2="494" y2="178" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>
  <line x1="453" y1="519" x2="494" y2="519" stroke="#334155" stroke-width="2" marker-end="url(#arr-dark)"/>

  <!-- Clean Routed Cross-Phase Arrows (zero overlap with any label!) -->
  <!-- X_s (blue) -> MHA_128 -->
  <line x1="766" y1="168" x2="882" y2="168" stroke="#1d4ed8" stroke-width="2.1" marker-end="url(#arr-blue)"/>
  <!-- X_s (grey) -> Controls -->
  <path d="M 766 196 L 816 196 L 816 426 L 882 426" fill="none" stroke="#64748b" stroke-width="1.8" marker-end="url(#arr-grey)"/>
  <!-- X_f (orange) -> MHA_128 -->
  <path d="M 766 502 L 834 502 L 834 256 L 882 256" fill="none" stroke="#c2410c" stroke-width="2.1" marker-end="url(#arr-orange)"/>
  <!-- X_f (grey) -> Controls -->
  <line x1="766" y1="534" x2="882" y2="534" stroke="#64748b" stroke-width="1.8" marker-end="url(#arr-grey)"/>

  <!-- Clean Pill Labels placed completely clear of arrows and dashed borders -->
  <rect x="780" y="132" width="76" height="22" rx="6" fill="#ffffff" stroke="#bfdbfe" stroke-width="1"/>
  <text x="818" y="147" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1d4ed8">X_s (128-D)</text>

  <rect x="780" y="548" width="76" height="22" rx="6" fill="#ffffff" stroke="#fed7aa" stroke-width="1"/>
  <text x="818" y="563" text-anchor="middle" font-size="11.5" font-weight="700" fill="#c2410c">X_f (128-D)</text>
</svg>
</body>
</html>"""

def main():
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        html_path = td_p / "fig1.html"
        raw_png = td_p / "raw.png"
        html_path.write_text(HTML, encoding="utf-8")
        subprocess.run([
            CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            "--window-size=1420,680", "--force-device-scale-factor=3",
            "--default-background-color=FFFFFFFF", f"--screenshot={raw_png}", html_path.as_uri()
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        im = Image.open(raw_png).convert("RGB")
        arr = np.array(im)
        nw = (arr < 252).any(axis=2)
        ys = np.where(nw.any(axis=1))[0]
        xs = np.where(nw.any(axis=0))[0]
        pad = 45
        cropped = Image.fromarray(arr[max(0,ys[0]-pad):min(arr.shape[0],ys[-1]+pad), max(0,xs[0]-pad):min(arr.shape[1],xs[-1]+pad), :])
        for p in OUT_PATHS:
            p.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(p, dpi=(300, 300))
            print("Saved:", p, cropped.size)

if __name__ == "__main__":
    main()
