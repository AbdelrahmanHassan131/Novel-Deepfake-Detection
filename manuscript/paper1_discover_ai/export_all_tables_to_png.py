# -*- coding: utf-8 -*-
"""
Script to extract and render all 14 tables from main.tex into high-resolution,
publication-ready PNG images with exact captions and academic styling.
"""

import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Directory to save generated PNGs
OUTPUT_DIR = os.path.abspath("extracted_tables")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common HTML template with publication-grade styling
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    background-color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    margin: 0;
    padding: 24px;
    display: inline-block;
  }}
  .table-box {{
    background: #ffffff;
    padding: 24px;
    border-radius: 4px;
    display: inline-block;
  }}
  .caption {{
    font-size: 14.5px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 14px;
    line-height: 1.45;
    max-width: {caption_width}px;
  }}
  .caption .tab-num {{
    font-weight: 700;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13.5px;
    color: #1f2937;
  }}
  th {{
    border-top: 2px solid #111827;
    border-bottom: 1.5px solid #111827;
    padding: 9px 14px;
    font-weight: 600;
    text-align: left;
    background-color: #f9fafb;
    vertical-align: bottom;
  }}
  td {{
    padding: 8px 14px;
    border-bottom: 1px solid #e5e7eb;
    line-height: 1.4;
    vertical-align: middle;
  }}
  tr:last-child td {{
    border-bottom: 2px solid #111827;
  }}
  .text-center {{ text-align: center; }}
  .text-right {{ text-align: right; }}
  .bold {{ font-weight: 600; }}
  .bg-sub {{ background-color: #f3f4f6; font-weight: 600; }}
  .bg-highlight {{ background-color: #f9fafb; font-weight: 600; }}
  .code {{ font-family: "Consolas", "Menlo", monospace; font-size: 12.5px; }}
</style>
</head>
<body>
<div class="table-box" id="capture">
  <div class="caption"><span class="tab-num">Table {num}.</span> {caption}</div>
  <table>
    {table_html}
  </table>
</div>
</body>
</html>"""

TABLES_DATA = [
    {
        "num": 1,
        "name": "table1_detailed_architectural_specifications",
        "caption": "Detailed architectural specifications and layer-by-layer trainable parameter counts for the proposed Multi-Head Attention (MHA) fusion model.",
        "caption_width": 850,
        "html": """
    <thead>
      <tr>
        <th>Module / Layer</th>
        <th>Specifications</th>
        <th class="text-center">Output Shape</th>
        <th class="text-right">Trainable Parameters</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Spatial Stream (<i>X<sub>s</sub></i>)</td>
        <td>Wang2020-128 Backbone (Frozen)</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">&mdash;</td>
      </tr>
      <tr>
        <td>Frequency Stream (<i>X<sub>f</sub></i>)</td>
        <td>Wolter2021-128 Backbone (Frozen)</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">&mdash;</td>
      </tr>
      <tr class="bg-sub">
        <td>Cross-Attention Blocks</td>
        <td colspan="3" class="text-center">(Parallel spatial <i>s &rarr; f</i> and frequency <i>f &rarr; s</i> paths)</td>
      </tr>
      <tr>
        <td>Multihead Attention (<i>A</i>)</td>
        <td><i>N</i><sub>heads</sub> = 4, <i>D</i> = 128, <i>D<sub>k</sub></i> = 32, bias = True</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">2 &times; 66,048 = 132,096</td>
      </tr>
      <tr>
        <td>LayerNorm 1</td>
        <td>features = 128</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">2 &times; 256 = 512</td>
      </tr>
      <tr>
        <td>Feed-Forward Network</td>
        <td>Linear(128 &rarr; 512 &rarr; 128), dropout = 0.1</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">2 &times; 131,712 = 263,424</td>
      </tr>
      <tr>
        <td>LayerNorm 2</td>
        <td>features = 128</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">2 &times; 256 = 512</td>
      </tr>
      <tr class="bg-sub">
        <td>Fusion &amp; Classification</td>
        <td colspan="3" class="text-center">(Concatenation and MLP Classification Head)</td>
      </tr>
      <tr>
        <td>Concatenation ([<i>Z<sub>s</sub></i> &parallel; <i>Z<sub>f</sub></i>])</td>
        <td>Combine refined spatial and frequency tokens</td>
        <td class="text-center">1 &times; 256</td>
        <td class="text-right">&mdash;</td>
      </tr>
      <tr>
        <td>Linear Fusion</td>
        <td>Linear(256 &rarr; 128), ReLU, dropout = 0.1</td>
        <td class="text-center">1 &times; 128</td>
        <td class="text-right">32,896</td>
      </tr>
      <tr>
        <td>Classification FC1</td>
        <td>Linear(128 &rarr; 64), ReLU, dropout = 0.1</td>
        <td class="text-center">1 &times; 64</td>
        <td class="text-right">8,256</td>
      </tr>
      <tr>
        <td>Classification FC2</td>
        <td>Linear(64 &rarr; 1)</td>
        <td class="text-center">1 &times; 1</td>
        <td class="text-right">65</td>
      </tr>
      <tr class="bg-highlight">
        <td colspan="2"><b>Total Trainable Core: MHA Fusion Classifier Module</b></td>
        <td class="text-center">&mdash;</td>
        <td class="text-right"><b>437,761</b></td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 2,
        "name": "table2_computational_footprint_comparison",
        "caption": "Computational footprint comparison across sub-model experts, fusion heads, and the complete end-to-end dual-stream MHA pipeline (224 &times; 224 input resolution, FP32 precision on NVIDIA T4 GPU).",
        "caption_width": 800,
        "html": """
    <thead>
      <tr>
        <th>Model Component</th>
        <th>Architecture Base</th>
        <th class="text-right">Parameters</th>
        <th class="text-right">Size (MB)</th>
        <th class="text-right">GFLOPs</th>
        <th class="text-right">Latency (ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Spatial Expert (<span class="code">Wang2020_128</span>)</td>
        <td>ResNet-50 + 128-D Proj.</td>
        <td class="text-right">23,770,433</td>
        <td class="text-right">90.68</td>
        <td class="text-right">4.120</td>
        <td class="text-right">8.4</td>
      </tr>
      <tr>
        <td>Frequency Expert (<span class="code">Wolter2021_128</span>)</td>
        <td>3-Level WPD + CNN + 128-D</td>
        <td class="text-right">1,727,553</td>
        <td class="text-right">6.59</td>
        <td class="text-right">0.480</td>
        <td class="text-right">4.1</td>
      </tr>
      <tr>
        <td>Simple Fusion Head (<span class="code">Fusion_128</span>)</td>
        <td>Concatenation + MLP</td>
        <td class="text-right">107,009</td>
        <td class="text-right">0.41</td>
        <td class="text-right">0.0002</td>
        <td class="text-right">0.3</td>
      </tr>
      <tr>
        <td>Proposed MHA Head (<span class="code">MHA_128</span>)</td>
        <td>Dual Cross-Attention + MLP</td>
        <td class="text-right">437,761</td>
        <td class="text-right">1.67</td>
        <td class="text-right">0.0009</td>
        <td class="text-right">0.6</td>
      </tr>
      <tr class="bg-highlight">
        <td><b>Full Dual-Stream MHA Pipeline</b></td>
        <td><b>Both Experts + MHA Head</b></td>
        <td class="text-right"><b>25,935,747</b></td>
        <td class="text-right"><b>98.94</b></td>
        <td class="text-right"><b>4.601</b></td>
        <td class="text-right"><b>13.1</b></td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 3,
        "name": "table3_authoritative_dataset_distribution",
        "caption": "Authoritative dataset distribution across Real and Fake (synthetic) classes under strict video-level and identity-level 80/20 partitioning.",
        "caption_width": 650,
        "html": """
    <thead>
      <tr>
        <th>Partition Split</th>
        <th class="text-center">Real Images (Authentic)</th>
        <th class="text-center">Fake Images (GAN + Diffusion)</th>
        <th class="text-right">Total Images</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Training (80%)</td>
        <td class="text-center">211,188</td>
        <td class="text-center">594,233</td>
        <td class="text-right">805,421</td>
      </tr>
      <tr>
        <td>Validation (20%)</td>
        <td class="text-center">52,765</td>
        <td class="text-center">158,669</td>
        <td class="text-right">211,434</td>
      </tr>
      <tr class="bg-highlight">
        <td><b>Total</b></td>
        <td class="text-center"><b>263,953</b></td>
        <td class="text-center"><b>752,902</b></td>
        <td class="text-right"><b>1,016,855</b></td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 4,
        "name": "table4_staged_training_protocol_summary",
        "caption": "Summary of the staged training and evaluation protocol, detailing parameter freezing states and strict partition isolation across Phase 1, Phase 2, and Phase 3.",
        "caption_width": 750,
        "html": """
    <thead>
      <tr>
        <th>Phase</th>
        <th>Component</th>
        <th>Parameter State</th>
        <th>Dataset Partition Used</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Phase 1</td>
        <td>Spatial RGB Expert (<span class="code">Wang2020_128</span>)</td>
        <td>Unfrozen (End-to-End)</td>
        <td>80% Train Split (<i>n</i> = 805,421)</td>
      </tr>
      <tr>
        <td>Phase 1</td>
        <td>Frequency Expert (<span class="code">Wolter2021_128</span>)</td>
        <td>Unfrozen (End-to-End)</td>
        <td>80% Train Split (<i>n</i> = 805,421)</td>
      </tr>
      <tr>
        <td>Phase 2</td>
        <td>Spatial &amp; Frequency Experts</td>
        <td><b>Strictly Frozen</b></td>
        <td>80% Train Split (Feature Extraction Only)</td>
      </tr>
      <tr>
        <td>Phase 2</td>
        <td>MHA Fusion Module (<span class="code">MHA_128</span>)</td>
        <td>Unfrozen (Trained)</td>
        <td>80% Train Split (<i>n</i> = 805,421)</td>
      </tr>
      <tr>
        <td>Phase 3</td>
        <td>Complete Dual-Stream Pipeline</td>
        <td><b>Strictly Frozen</b> (Inference)</td>
        <td>Held-Out 20% Val Split (<i>n</i> = 211,434)</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 5,
        "name": "table5_quantitative_evaluation_wang2020_spatial_expert",
        "caption": "Quantitative evaluation of the modified <span class='code'>Wang2020_128</span> spatial expert on the held-out validation partition (<i>n</i> = 211,434).",
        "caption_width": 550,
        "html": """
    <thead>
      <tr>
        <th>Metric</th>
        <th class="text-center">Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Accuracy (Milestone / Final Converged)</td>
        <td class="text-center">98.20% / 98.73%</td>
      </tr>
      <tr>
        <td>ROC AUC</td>
        <td class="text-center">0.9987</td>
      </tr>
      <tr>
        <td>Average Precision</td>
        <td class="text-center">0.9977</td>
      </tr>
      <tr>
        <td>Equal Error Rate (EER)</td>
        <td class="text-center">1.54%</td>
      </tr>
      <tr>
        <td>F1 Score (Macro)</td>
        <td class="text-center">0.9796</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 6,
        "name": "table6_class_wise_performance_wang2020_spatial_expert",
        "caption": "Class-wise performance breakdown for the modified <span class='code'>Wang2020_128</span> spatial expert.",
        "caption_width": 600,
        "html": """
    <thead>
      <tr>
        <th>Class</th>
        <th class="text-center">Precision</th>
        <th class="text-center">Recall</th>
        <th class="text-center">F1 Score</th>
        <th class="text-right">Support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Real</td>
        <td class="text-center">0.9838</td>
        <td class="text-center">0.9755</td>
        <td class="text-center">0.9796</td>
        <td class="text-right">52,765</td>
      </tr>
      <tr>
        <td>Fake</td>
        <td class="text-center">0.9889</td>
        <td class="text-center">0.9927</td>
        <td class="text-center">0.9908</td>
        <td class="text-right">158,669</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 7,
        "name": "table7_quantitative_evaluation_wolter2021_wavelet_expert",
        "caption": "Quantitative evaluation of the modified <span class='code'>Wolter2021_128</span> wavelet-packet frequency expert on the held-out validation partition (<i>n</i> = 211,434).",
        "caption_width": 550,
        "html": """
    <thead>
      <tr>
        <th>Metric</th>
        <th class="text-center">Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Accuracy</td>
        <td class="text-center">95.42%</td>
      </tr>
      <tr>
        <td>ROC AUC</td>
        <td class="text-center">0.9864</td>
      </tr>
      <tr>
        <td>Average Precision</td>
        <td class="text-center">0.9787</td>
      </tr>
      <tr>
        <td>Equal Error Rate (EER)</td>
        <td class="text-center">5.92%</td>
      </tr>
      <tr>
        <td>F1 Score (Macro)</td>
        <td class="text-center">0.9453</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 8,
        "name": "table8_class_wise_performance_wolter2021_frequency_expert",
        "caption": "Class-wise performance breakdown for the modified <span class='code'>Wolter2021_128</span> frequency expert.",
        "caption_width": 600,
        "html": """
    <thead>
      <tr>
        <th>Class</th>
        <th class="text-center">Precision</th>
        <th class="text-center">Recall</th>
        <th class="text-center">F1 Score</th>
        <th class="text-right">Support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Real</td>
        <td class="text-center">0.9699</td>
        <td class="text-center">0.8807</td>
        <td class="text-center">0.9232</td>
        <td class="text-right">52,765</td>
      </tr>
      <tr>
        <td>Fake</td>
        <td class="text-center">0.9479</td>
        <td class="text-center">0.9876</td>
        <td class="text-center">0.9673</td>
        <td class="text-right">158,669</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 9,
        "name": "table9_quantitative_evaluation_mha_fusion_model",
        "caption": "Quantitative evaluation of the proposed <span class='code'>MHA_128</span> fusion model on the held-out validation partition (<i>n</i> = 211,434).",
        "caption_width": 550,
        "html": """
    <thead>
      <tr>
        <th>Metric</th>
        <th class="text-center">Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Accuracy</td>
        <td class="text-center">98.82%</td>
      </tr>
      <tr>
        <td>ROC AUC</td>
        <td class="text-center">0.9992</td>
      </tr>
      <tr>
        <td>Average Precision</td>
        <td class="text-center">0.9984</td>
      </tr>
      <tr>
        <td>Equal Error Rate (EER)</td>
        <td class="text-center">1.27%</td>
      </tr>
      <tr>
        <td>F1 Score (Macro)</td>
        <td class="text-center">0.9862</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 10,
        "name": "table10_class_wise_performance_mha_fusion_model",
        "caption": "Class-wise performance breakdown for the proposed <span class='code'>MHA_128</span> fusion model.",
        "caption_width": 600,
        "html": """
    <thead>
      <tr>
        <th>Class</th>
        <th class="text-center">Precision</th>
        <th class="text-center">Recall</th>
        <th class="text-center">F1 Score</th>
        <th class="text-right">Support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Real</td>
        <td class="text-center">0.9900</td>
        <td class="text-center">0.9720</td>
        <td class="text-center">0.9809</td>
        <td class="text-right">52,765</td>
      </tr>
      <tr>
        <td>Fake</td>
        <td class="text-center">0.9873</td>
        <td class="text-center">0.9955</td>
        <td class="text-center">0.9914</td>
        <td class="text-right">158,669</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 11,
        "name": "table11_quantitative_comparison_mha_vs_baselines",
        "caption": "Quantitative comparison of the proposed Bidirectional Cross-Attention model (<span class='code'>MHA_128</span>) against static concatenation (<span class='code'>Fusion_128</span>) and retrained baselines on the held-out validation split (<i>n</i> = 211,434). The best value in each column is highlighted in bold.",
        "caption_width": 750,
        "html": """
    <thead>
      <tr>
        <th>Model</th>
        <th class="text-center">Accuracy</th>
        <th class="text-center">F1 (Real)</th>
        <th class="text-center">ROC AUC</th>
        <th class="text-center">EER</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color: #f9fafb;">
        <td><b>Our MHA (<span class='code'>MHA_128</span>, Proposed)</b></td>
        <td class="text-center">98.82%</td>
        <td class="text-center">0.9809</td>
        <td class="text-center"><b>0.9992</b></td>
        <td class="text-center"><b>1.27%</b></td>
      </tr>
      <tr>
        <td>Our Simple Fusion (<span class='code'>Fusion_128</span>)</td>
        <td class="text-center"><b>98.84%</b></td>
        <td class="text-center"><b>0.9810</b></td>
        <td class="text-center">0.9900</td>
        <td class="text-center">1.32%</td>
      </tr>
      <tr>
        <td>Our Wang (<span class='code'>Wang2020_128</span>)</td>
        <td class="text-center">98.73%</td>
        <td class="text-center">0.9796</td>
        <td class="text-center">0.9987</td>
        <td class="text-center">1.54%</td>
      </tr>
      <tr>
        <td>Our Wolter (<span class='code'>Wolter2021_128</span>)</td>
        <td class="text-center">95.42%</td>
        <td class="text-center">0.9232</td>
        <td class="text-center">0.9864</td>
        <td class="text-center">5.92%</td>
      </tr>
      <tr>
        <td>Wang2020 (Original, Re-trained)</td>
        <td class="text-center">98.73%</td>
        <td class="text-center">0.9796</td>
        <td class="text-center">0.9987</td>
        <td class="text-center">1.54%</td>
      </tr>
      <tr>
        <td>Wolter2021 (Original, Re-trained)</td>
        <td class="text-center">94.40%</td>
        <td class="text-center">0.9062</td>
        <td class="text-center">0.9752</td>
        <td class="text-center">8.03%</td>
      </tr>
      <tr>
        <td>Xception (Re-trained)</td>
        <td class="text-center">98.51%</td>
        <td class="text-center">0.9781</td>
        <td class="text-center">0.9984</td>
        <td class="text-center">1.57%</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 12,
        "name": "table12_multi_run_statistical_variance_analysis",
        "caption": "Multi-run statistical comparison (<i>N</i> = 4 independent seeds) between static concatenation (<span class='code'>Fusion_128</span>) and the proposed bidirectional Multi-Head Attention model (<span class='code'>MHA_128</span>), reporting Mean &plusmn; SD, 95% Confidence Intervals, and two-sided paired <i>t</i>-test significance levels.",
        "caption_width": 800,
        "html": """
    <thead>
      <tr>
        <th>Metric (<i>N</i> = 4 Independent Runs)</th>
        <th class="text-center">Simple Fusion (<span class='code'>Fusion_128</span>)</th>
        <th class="text-center">Proposed MHA (<span class='code'>MHA_128</span>)</th>
        <th class="text-center">Paired <i>t</i>-Test (<i>p</i>-value)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Accuracy (%, Mean &plusmn; SD)</td>
        <td class="text-center"><b>98.84 &plusmn; 0.03</b></td>
        <td class="text-center">98.82 &plusmn; 0.02</td>
        <td class="text-center"><i>p</i> = 0.284 (Not Sig.)</td>
      </tr>
      <tr>
        <td>Accuracy 95% Confidence Interval</td>
        <td class="text-center">[98.79%, 98.89%]</td>
        <td class="text-center">[98.79%, 98.85%]</td>
        <td class="text-center">Overlapping CIs</td>
      </tr>
      <tr>
        <td>ROC-AUC (Mean &plusmn; SD)</td>
        <td class="text-center">0.9900 &plusmn; 0.0009</td>
        <td class="text-center"><b>0.9992 &plusmn; 0.0004</b></td>
        <td class="text-center"><i>p</i> = 0.0004 (<i>p</i> &lt; 0.001)</td>
      </tr>
      <tr>
        <td>ROC-AUC 95% Confidence Interval</td>
        <td class="text-center">[0.9886, 0.9914]</td>
        <td class="text-center">[0.9986, 0.9998]</td>
        <td class="text-center">Non-overlapping CIs</td>
      </tr>
      <tr>
        <td>EER (%, Mean &plusmn; SD)</td>
        <td class="text-center">1.32 &plusmn; 0.04</td>
        <td class="text-center"><b>1.27 &plusmn; 0.02</b></td>
        <td class="text-center"><i>p</i> = 0.0082 (<i>p</i> &lt; 0.01)</td>
      </tr>
      <tr>
        <td>EER 95% Confidence Interval</td>
        <td class="text-center">[1.26%, 1.38%]</td>
        <td class="text-center">[1.24%, 1.30%]</td>
        <td class="text-center">Statistically Significant</td>
      </tr>
      <tr>
        <td>Macro F1-Score (Mean &plusmn; SD)</td>
        <td class="text-center">0.9861 &plusmn; 0.0004</td>
        <td class="text-center"><b>0.9862 &plusmn; 0.0003</b></td>
        <td class="text-center"><i>p</i> = 0.612 (Not Sig.)</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 13,
        "name": "table13_inference_time_robustness_fgsm",
        "caption": "Inference-time robustness evaluation comparing baseline validation performance against white-box FGSM adversarial perturbation.",
        "caption_width": 650,
        "html": """
    <thead>
      <tr>
        <th>Evaluation Setting</th>
        <th class="text-center">Accuracy</th>
        <th class="text-center">ROC AUC</th>
        <th class="text-center">EER</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Unperturbed Validation Baseline (<i>Q</i> = 75)</td>
        <td class="text-center">0.9882</td>
        <td class="text-center">0.9992</td>
        <td class="text-center">0.0127</td>
      </tr>
      <tr>
        <td>White-Box FGSM Adversarial Attack (&epsilon; = 8/255)</td>
        <td class="text-center">0.9070</td>
        <td class="text-center">0.8920</td>
        <td class="text-center">0.1626</td>
      </tr>
    </tbody>
        """
    },
    {
        "num": 14,
        "name": "table14_software_dependency_specifications",
        "caption": "Complete software dependency and package version specifications.",
        "caption_width": 750,
        "html": """
    <thead>
      <tr>
        <th>Library Name</th>
        <th class="text-center">Minimum Version</th>
        <th>Primary Purpose in Pipeline</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><span class="code">torch</span></td><td class="text-center">2.5.1</td><td>Core deep learning framework and tensor operations</td></tr>
      <tr><td><span class="code">torchvision</span></td><td class="text-center">0.20.1</td><td>Data loading utilities and image transformations</td></tr>
      <tr><td><span class="code">numpy</span></td><td class="text-center">2.4.4</td><td>Multi-dimensional numerical arrays and calculations</td></tr>
      <tr><td><span class="code">pytorch-wavelets</span></td><td class="text-center">1.3.0</td><td>GPU-accelerated Discrete Wavelet Transforms</td></tr>
      <tr><td><span class="code">pywavelets</span></td><td class="text-center">1.9.0</td><td>CPU-based reference wavelet packet decompositions</td></tr>
      <tr><td><span class="code">scikit-learn</span></td><td class="text-center">1.9.0</td><td>Evaluation metric calculations (EER, AUC, F1)</td></tr>
      <tr><td><span class="code">opencv-python</span></td><td class="text-center">5.0.0.93</td><td>Real-time image decoding and augmentations</td></tr>
      <tr><td><span class="code">pillow</span></td><td class="text-center">12.2.0</td><td>Image processing and basic loading checks</td></tr>
      <tr><td><span class="code">matplotlib</span></td><td class="text-center">3.11.0</td><td>Metric visualization (ROC, PR, Confusion Matrix)</td></tr>
      <tr><td><span class="code">scipy</span></td><td class="text-center">1.17.1</td><td>Mathematical utilities and signal processing algorithms</td></tr>
      <tr><td><span class="code">pyyaml</span></td><td class="text-center">6.0.3</td><td>Parsing configuration files and CLI overrides</td></tr>
      <tr><td><span class="code">tensorboard</span></td><td class="text-center">2.21.0</td><td>Logging training progress and metric visualization</td></tr>
      <tr><td><span class="code">tqdm</span></td><td class="text-center">4.68.3</td><td>Console-based execution progress tracking</td></tr>
    </tbody>
        """
    }
]

def main():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--force-device-scale-factor=2.5')
    options.add_argument('--window-size=1920,1400')

    driver = webdriver.Chrome(options=options)

    generated_files = []

    for t in TABLES_DATA:
        html = HTML_TEMPLATE.format(
            num=t["num"],
            caption=t["caption"],
            caption_width=t["caption_width"],
            table_html=t["html"]
        )
        temp_html_path = os.path.abspath(f"temp_render_table_{t['num']}.html")
        with open(temp_html_path, "w", encoding="utf-8") as f:
            f.write(html)

        driver.get(f"file:///{temp_html_path}")
        el = driver.find_element("id", "capture")

        # Save table screenshot exclusively in extracted_tables/
        snake_name = f"{t['name']}.png"
        out_path_sub = os.path.join(OUTPUT_DIR, snake_name)

        el.screenshot(out_path_sub)

        generated_files.append(snake_name)
        print(f"Generated: {snake_name}")

        try:
            os.remove(temp_html_path)
        except OSError:
            pass

    driver.quit()
    print(f"\nAll {len(generated_files)} tables rendered successfully!")

if __name__ == "__main__":
    main()
