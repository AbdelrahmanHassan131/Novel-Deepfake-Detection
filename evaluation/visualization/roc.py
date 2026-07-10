"""
ROC Curve — evaluation wrapper.

Delegates to the existing ``experiment.visualization.plot_roc_curve``
and adds the EER marker that the evaluation pipeline requires.
"""

import os
import numpy as np


def plot_evaluation_roc_curve(probabilities, labels, save_path=None,
                              show=False, title='ROC Curve'):
    """
    Plot ROC curve with AUC and EER point marked.

    This wraps the existing visualization and augments it with EER.

    Args:
        probabilities (array-like): Predicted probabilities (N,).
        labels (array-like): Ground-truth binary labels (N,).
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display interactively.
        title (str): Plot title.

    Returns:
        str or None: Path to the saved plot.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve as sk_roc_curve, auc

    probabilities = np.asarray(probabilities).ravel()
    labels = np.asarray(labels).ravel()

    fpr, tpr, thresholds = sk_roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Compute EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='darkorange', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    # Mark EER point
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
