"""
Visualization subpackage.

All visualization tools are **post-training** utilities.  They read
saved experiment files and produce plots.  Nothing in this package
is called during training.

Public API:
    from experiment.visualization import (
        plot_training_curves,
        compare_runs,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_precision_recall_curve,
    )
"""

from .plot_training import plot_training_curves
from .compare_runs import compare_runs
from .confusion_matrix import plot_confusion_matrix
from .roc_curve import plot_roc_curve
from .precision_recall import plot_precision_recall_curve

__all__ = [
    'plot_training_curves',
    'compare_runs',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
]
