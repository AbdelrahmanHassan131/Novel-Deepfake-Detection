"""
precision_recall — post-training Precision-Recall curve visualisation.

Receives pre-computed predictions and labels and produces a
Precision-Recall curve plot with Average Precision annotation.

Usage::

    from experiment.visualization import plot_precision_recall_curve
    import numpy as np

    preds = np.array([0.9, 0.1, 0.8, 0.3])
    labels = np.array([1, 0, 1, 0])
    plot_precision_recall_curve(preds, labels, save_path='pr.png')
"""

import os
import numpy as np


def plot_precision_recall_curve(predictions, labels, save_path=None,
                                show=False, title='Precision-Recall Curve'):
    """
    Plot the Precision-Recall curve.

    Requires ``sklearn`` for ``precision_recall_curve`` computation.

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
    from sklearn.metrics import precision_recall_curve as sk_pr_curve
    from sklearn.metrics import average_precision_score

    predictions = np.asarray(predictions).ravel()
    labels = np.asarray(labels).ravel()

    precision, recall, _ = sk_pr_curve(labels, predictions)
    avg_precision = average_precision_score(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, 'b-', linewidth=2,
            label=f'PR Curve (AP = {avg_precision:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
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
