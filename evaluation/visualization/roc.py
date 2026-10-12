"""
ROC Curve — evaluation wrapper.

Delegates to the existing ``experiment.visualization.plot_roc_curve``
and adds the EER marker that the evaluation pipeline requires.
"""

import os
import numpy as np


def plot_evaluation_roc_curve(probabilities, labels, save_path=None,
                              show=False, title='ROC Curve', logits=None):
    """
    Plot ROC curve with AUC and EER point marked.

    This wraps the existing visualization and augments it with EER.

    Args:
        probabilities (array-like): Predicted probabilities (N,).
        labels (array-like): Ground-truth binary labels (N,).
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display interactively.
        title (str): Plot title.
        logits (array-like, optional): Raw unscaled logits (N,).

    Returns:
        str or None: Path to the saved plot.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve as sk_roc_curve, auc
    from evaluation.generalization import compute_interpolated_eer

    labels = np.asarray(labels).ravel()
    if logits is not None:
        scores = np.asarray(logits).ravel()
    else:
        scores = np.asarray(probabilities).ravel()

    fpr, tpr, thresholds = sk_roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Compute continuous interpolated EER
    eer, _, _, _ = compute_interpolated_eer(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Mark EER point
    if eer is not None:
        ax.plot(eer, 1 - eer, 'ro', markersize=8,
                label=f'EER = {eer:.4f}')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    fig.tight_layout()

    result_path = None
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        result_path = save_path

    if show:
        plt.show()
    plt.close(fig)

    return result_path
