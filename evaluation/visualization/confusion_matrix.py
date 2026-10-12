"""
Confusion Matrix — evaluation wrapper.

Delegates to the existing
``experiment.visualization.plot_confusion_matrix``.
"""

from experiment.visualization import (
    plot_confusion_matrix as _existing_plot_cm,
)
from evaluation.generalization import compute_binary_predictions


def plot_evaluation_confusion_matrix(probabilities, labels, save_path=None,
                                     show=False, threshold=0.5,
                                     class_names=None,
                                     title='Confusion Matrix',
                                     logits=None, score_space='probability'):
    """
    Plot confusion matrix heatmap.

    Directly reuses the existing implementation with unified binary decision logic.

    Args:
        probabilities (array-like): Predicted probabilities (N,).
        labels (array-like): Ground-truth binary labels (N,).
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display interactively.
        threshold (float): Decision threshold.
        class_names (list, optional): Class labels for axes.
        title (str): Plot title.
        logits (array-like, optional): Raw unscaled logits (N,).
        score_space (str): 'probability' or 'logit'.

    Returns:
        str or None: Path to the saved plot.
    """
    pred_binary, _ = compute_binary_predictions(
        probabilities=probabilities,
        logits=logits,
        threshold=threshold,
        score_space=score_space,
    )

    return _existing_plot_cm(
        predictions=pred_binary.astype(float),
        labels=labels,
        save_path=save_path,
        show=show,
        threshold=0.5,
        class_names=class_names,
        title=title,
    )
