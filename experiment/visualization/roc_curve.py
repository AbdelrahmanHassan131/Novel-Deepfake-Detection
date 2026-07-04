"""
roc_curve — post-training ROC curve visualisation.

Receives pre-computed predictions and labels and produces a ROC
curve plot with AUC annotation.

Usage::

    from Refactored.experiment.visualization import plot_roc_curve
    import numpy as np

    preds = np.array([0.9, 0.1, 0.8, 0.3])
    labels = np.array([1, 0, 1, 0])
    plot_roc_curve(preds, labels, save_path='roc.png')
"""

import os
import numpy as np


def plot_roc_curve(predictions, labels, save_path=None, show=False,
                   title='ROC Curve'):
    """
    Plot the Receiver Operating Characteristic curve.

    Requires ``sklearn`` for ``roc_curve`` computation.

    Args:
        predictions (array-like): Predicted probabilities, shape ``(N,)``.
        labels (array-like): Ground-truth binary labels, shape ``(N,)``.
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display the plot interactively.
        title (str): Plot title.

    Returns:
        str or None: Path to the saved plot, or None if not saved.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve as sk_roc_curve, auc

    predictions = np.asarray(predictions).ravel()
    labels = np.asarray(labels).ravel()

    fpr, tpr, _ = sk_roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
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
