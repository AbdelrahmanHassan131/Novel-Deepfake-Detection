"""
Evaluation Visualization subpackage.

Provides evaluation-specific visualizations.  Reuses the existing
``experiment.visualization`` modules for ROC, PR, and
confusion matrix plots.  Adds t-SNE and Grad-CAM as new modules.
"""

from .tsne import plot_tsne
from .gradcam import GradCAM, GradCAMPlusPlus, generate_gradcam_figure
from .roc import plot_evaluation_roc_curve
from .precision_recall import plot_evaluation_pr_curve
from .confusion_matrix import plot_evaluation_confusion_matrix

__all__ = [
    'plot_tsne',
    'GradCAM',
    'GradCAMPlusPlus',
    'generate_gradcam_figure',
    'plot_evaluation_roc_curve',
    'plot_evaluation_pr_curve',
    'plot_evaluation_confusion_matrix',
]
