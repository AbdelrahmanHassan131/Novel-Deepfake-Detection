"""
Precision-Recall Curve — evaluation wrapper.

Delegates to the existing
``experiment.visualization.plot_precision_recall_curve``.
"""

from experiment.visualization import plot_precision_recall_curve


def plot_evaluation_pr_curve(probabilities, labels, save_path=None,
                             show=False, title='Precision-Recall Curve'):
    """
    Plot Precision-Recall curve with Average Precision.

    Directly reuses the existing implementation.

    Args:
        probabilities (array-like): Predicted probabilities (N,).
        labels (array-like): Ground-truth binary labels (N,).
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display interactively.
        title (str): Plot title.

    Returns:
        str or None: Path to the saved plot.
    """
    return plot_precision_recall_curve(
        predictions=probabilities,
        labels=labels,
        save_path=save_path,
        show=show,
        title=title,
    )
