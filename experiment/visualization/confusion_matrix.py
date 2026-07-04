"""
confusion_matrix — post-training confusion matrix visualisation.

Receives pre-computed predictions and labels (from a saved experiment
or a fresh evaluation) and produces a confusion matrix heatmap.

Usage::

    from experiment.visualization import plot_confusion_matrix
    import numpy as np

    preds = np.array([0.9, 0.1, 0.8, 0.3])
    labels = np.array([1, 0, 1, 0])
    plot_confusion_matrix(preds, labels, save_path='cm.png')
"""

import os
import numpy as np

from experiment.metrics import MetricsCalculator


def plot_confusion_matrix(predictions, labels, save_path=None,
                          show=False, threshold=0.5,
                          class_names=None, title='Confusion Matrix'):
    """
    Plot a 2x2 confusion matrix heatmap.

    Args:
        predictions (array-like): Predicted probabilities, shape ``(N,)``.
        labels (array-like): Ground-truth binary labels, shape ``(N,)``.
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display the plot interactively.
        threshold (float): Decision threshold.  Defaults to 0.5.
        class_names (list of str, optional): Class labels for axes.
            Defaults to ``['Real', 'Fake']``.
        title (str): Plot title.

    Returns:
        str or None: Path to the saved plot, or None if not saved.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = ['Real', 'Fake']

    calc = MetricsCalculator(threshold=threshold)
    metrics = calc.compute(predictions, labels)
    cm = np.array(metrics['confusion_matrix'])

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted',
        ylabel='True',
        title=title,
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=16)

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
