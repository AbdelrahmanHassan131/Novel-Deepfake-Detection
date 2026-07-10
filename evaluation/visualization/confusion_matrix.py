"""
Confusion Matrix — evaluation wrapper.

Delegates to the existing
``experiment.visualization.plot_confusion_matrix``.
"""

from experiment.visualization import (
    plot_confusion_matrix as _existing_plot_cm,
)


def plot_evaluation_confusion_matrix(probabilities, labels, save_path=None,
                                     show=False, threshold=0.5,
                                     class_names=None,
                                     title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap.

    Directly reuses the existing implementation.

    Args:
        probabilities (array-like): Predicted probabilities (N,).
        labels (array-like): Ground-truth binary labels (N,).
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display interactively.
        threshold (float): Decision threshold.
        class_names (list, optional): Class labels for axes.
        title (str): Plot title.

    Returns:
        str or None: Path to the saved plot.
    """
    return _existing_plot_cm(
        predictions=probabilities,
        labels=labels,
        save_path=save_path,
        show=show,
        threshold=threshold,
        class_names=class_names,
        title=title,
    )
