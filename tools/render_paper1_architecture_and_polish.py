"""Generate crisp Figure 1 (exact architecture diagram) and polished Figure 4 (4-seed bars) for Paper 1 Revision."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("manuscript/paper1_discover_ai")
METRICS_DIR = ROOT / "revision_artifacts/metrics"
RAW_FIGS = ROOT / "revision_artifacts/figures"
OUT_FIGS = ROOT / "figures"
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, title, subtitle="", fc="#eef5ff", ec="#2b5c8f", title_size=8.5, sub_size=7.0):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="#111111")
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=sub_size, color="#333333")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color="#111111")


def draw_arrow(ax, x0, y0, x1, y1, color="#333333"):
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=6,head_width=3.5", color=color, linewidth=1.3)
    ax.add_patch(arr)


def render_architecture_diagram():
    """Exact layer-by-layer dual-stream architecture diagram matching Table 1 & Table 2 (fixes R1 #25 & R2 #10)."""
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Title banner
    ax.text(0.5, 0.965, "Decoupled Dual-Stream Spatial-Frequency Architecture with Multi-Token Attention (MHA_128) & Gated Fusion",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1a1a1a")

    # Input
    draw_box(ax, 0.015, 0.42, 0.115, 0.16, "Input RGB Image", "224 x 224 x 3\nForensic Aug (Train)", fc="#f5f5f5", ec="#555555")

    # Phase 1 dashed container
    p1_rect = FancyBboxPatch((0.155, 0.06), 0.44, 0.85, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor="none", edgecolor="#888888", linestyle="--", linewidth=1.2)
    ax.add_patch(p1_rect)
    ax.text(0.375, 0.885, "Phase 1: Independent Expert Training -> Strictly Frozen in Phase 2",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#444444")

    # Top branch: Spatial Expert (Wang2020_128)
    draw_box(ax, 0.175, 0.64, 0.16, 0.18, "Spatial Stream\nWang2020_128",
             "ResNet-50 (ImageNet)\nGAP -> 2048-D\n23.51M params", fc="#e6f2ff", ec="#1f5a96")
    draw_box(ax, 0.375, 0.65, 0.20, 0.16, "128-D Bottleneck",
             "Linear(2048->128) + ReLU\nDropout(0.5) [Train]\nFrozen: 23,770,304 params", fc="#d4e8ff", ec="#1f5a96")

    # Bottom branch: Frequency Expert (WolterWavelet2021_128)
    draw_box(ax, 0.175, 0.28, 0.16, 0.22, "3-Level Haar WPD\n(GPU, db1, reflect)",
             "signed_log1p(C)\n3 x 4^3 = 192 sub-bands\nTensor: 192 x 28 x 28", fc="#fff3e0", ec="#c66900")
    draw_box(ax, 0.175, 0.085, 0.16, 0.16, "Wavelet CNN Backbone",
             "4x[Conv3x3+BN+ReLU+Pool]\n192->64->128->256->512\nGAP -> 512-D", fc="#ffe8cc", ec="#c66900")
    draw_box(ax, 0.375, 0.105, 0.20, 0.16, "128-D Bottleneck",
             "Linear(512->128) + ReLU\nDropout(0.5) [Train]\nTotal: 1,727,553 params", fc="#ffdcad", ec="#c66900")

    # Phase 2 dashed container
    p2_rect = FancyBboxPatch((0.625, 0.06), 0.36, 0.85, boxstyle="round,pad=0.01,rounding_size=0.02",
                             facecolor="none", edgecolor="#2e7d32", linestyle="--", linewidth=1.2)
    ax.add_patch(p2_rect)
    ax.text(0.805, 0.885, "Phase 2: Cross-Fitted Sample-Dependent Fusion",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1b5e20")

    # Primary MHA_128 TokenAttentionFusion block (top right)
    draw_box(ax, 0.645, 0.50, 0.32, 0.33, "MHA_128 (TokenAttentionFusion, 199,425 params)",
             "1. LayerNorm(X_s), LayerNorm(X_f) -> [CLS, X_s, X_f] (L=3)\n"
             "2. Add Learnable Token-Type Embeddings (3 x 128)\n"
             "3. 4-Head Self-Attention (D_k=32, softmax over L=3)\n"
             "4. Residual + LN -> FFN(128->512->128, GELU) -> Residual + LN\n"
             "5. Extract [CLS] token -> Linear(128->1) Forgery Logit",
             fc="#e8f5e9", ec="#2e7d32", title_size=8.0, sub_size=6.8)

    # Alternative GatedFusion + Capacity-Matched Controls (bottom right)
    draw_box(ax, 0.645, 0.11, 0.32, 0.31, "Matched Fusion Controls (Same Frozen X_s, X_f)",
             "- GatedFusion (41,987 params): Softmax gate g(X_s||X_f)\n"
             "  weighted sum -> MLP(128->64->1)\n"
             "- ConcatMatchedMLP (198,017 params): 256->384->256->1\n"
             "- NoAttnFFN (198,657 params): Proj+FFN(128->640->128)\n"
             "- LinearMix (198,785 params): Linear cross-mix + FFN\n"
             "- Original Fusion_128 (107,009 params): 256->256->128->64->1",
             fc="#f3e5f5", ec="#6a1b9a", title_size=8.0, sub_size=6.8)

    # Arrows
    draw_arrow(ax, 0.13, 0.54, 0.175, 0.72)
    draw_arrow(ax, 0.13, 0.46, 0.175, 0.39)
    draw_arrow(ax, 0.255, 0.28, 0.255, 0.245)
    draw_arrow(ax, 0.335, 0.73, 0.375, 0.73)
    draw_arrow(ax, 0.335, 0.18, 0.375, 0.18)

    # Embeddings to Phase 2 (clean white pill labels so arrows and dashed border never overlap text)
    draw_arrow(ax, 0.575, 0.73, 0.645, 0.73, color="#1f5a96")
    ax.text(0.610, 0.785, "X_s (128-D)", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1f5a96",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8), zorder=5)
    draw_arrow(ax, 0.575, 0.20, 0.645, 0.58, color="#c66900")
    ax.text(0.610, 0.105, "X_f (128-D)", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#c66900",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8), zorder=5)
    draw_arrow(ax, 0.575, 0.69, 0.645, 0.33, color="#777777")
    draw_arrow(ax, 0.575, 0.16, 0.645, 0.20, color="#777777")

    fig.tight_layout()
    fig.savefig(OUT_FIGS / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_polished_ablation_bars():
    """Re-render 4-seed bar chart with exact non-overlapping value labels above error bar caps (R1 #27)."""
    ab = json.loads((METRICS_DIR / "paper1_ablation_4seed_report.json").read_text(encoding="utf-8"))
    labels_plot = list(ab["ablation_runs"].keys())
    short_labels = ["MHA_128", "ConcatMatched", "NoAttnFFN", "LinearMix", "GatedFusion", "Fusion_128"]

    accs_mu, accs_sd, aucs_mu, aucs_sd, eers_mu, eers_sd = [], [], [], [], [], []
    for lbl in labels_plot:
        rs = [r["eval"]["unseen_q85"]["macro_4domain"] for r in ab["ablation_runs"][lbl]]
        accs_mu.append(np.mean([x["accuracy"] * 100 for x in rs]))
        accs_sd.append(np.std([x["accuracy"] * 100 for x in rs], ddof=1))
        aucs_mu.append(np.mean([x["roc_auc"] * 100 for x in rs]))
        aucs_sd.append(np.std([x["roc_auc"] * 100 for x in rs], ddof=1))
        eers_mu.append(np.mean([x["eer"] * 100 for x in rs]))
        eers_sd.append(np.std([x["eer"] * 100 for x in rs], ddof=1))

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 300})
    x = np.arange(len(labels_plot))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})

    b1 = ax1.bar(x - 0.19, accs_mu, 0.36, yerr=accs_sd, capsize=3.5, label="4-Domain Mean Acc (%)", color="#1f77b4")
    b2 = ax1.bar(x + 0.19, aucs_mu, 0.36, yerr=aucs_sd, capsize=3.5, label="4-Domain Mean ROC-AUC (%)", color="#ff7f0e")
    for bars, sds in ((b1, accs_sd), (b2, aucs_sd)):
        for bar, sd in zip(bars, sds):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd + 0.35,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=18, ha="right")
    ax1.set_ylim(78, 97.5)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Cross-Fitted 4-Seed Unseen Generalization (Higher is Better)")
    ax1.legend(fontsize=8.5, loc="upper right")

    b3 = ax2.bar(x, eers_mu, 0.55, yerr=eers_sd, capsize=3.5, color="#e07a5f")
    for bar, sd in zip(b3, eers_sd):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sd + 0.35,
                 f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=18, ha="right")
    ax2.set_ylim(0, 17.5)
    ax2.set_ylabel("4-Domain Mean EER (%)")
    ax2.set_title("Equal Error Rate — Separate Axis (Lower is Better)")

    fig.tight_layout()
    fig.savefig(OUT_FIGS / "fig4_ablation_4seed_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_remaining_figures():
    mapping = {
        "fig_train_val_convergence.png": "fig2_train_val_convergence.png",
        "fig_roc_pr_reliability.png": "fig3_roc_pr_reliability.png",
        "fig_robustness_perturb_fgsm.png": "fig5_robustness_perturb_fgsm.png",
        "fig_tsne_silhouette.png": "fig6_tsne_silhouette.png",
        "fig_gradcam_masking.png": "fig7_gradcam_masking.png",
    }
    for src_name, dst_name in mapping.items():
        shutil.copy2(RAW_FIGS / src_name, OUT_FIGS / dst_name)


if __name__ == "__main__":
    render_architecture_diagram()
    render_polished_ablation_bars()
    copy_remaining_figures()
    print("Generated and copied all 7 publication figures to", OUT_FIGS)
