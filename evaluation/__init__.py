"""
Evaluation Pipeline.

Provides a unified, architecture-agnostic evaluation system for all
models registered in the model registry.

Public API::

    from evaluation import Evaluator, EvaluationResult
    from evaluation import InferenceRunner, InferenceResult
    from evaluation import CheckpointLoader
    from evaluation.metrics import ClassificationMetrics, PerformanceProfiler
    from evaluation.reports import EvaluationReportGenerator
    from evaluation.visualization import (
        plot_tsne, GradCAM, GradCAMPlusPlus, generate_gradcam_figure,
        plot_evaluation_roc_curve, plot_evaluation_pr_curve,
        plot_evaluation_confusion_matrix
    )

Usage::

    evaluator = Evaluator(checkpoint_path, dataroot, output_dir)
    results = evaluator.run()
"""

from .evaluator import Evaluator, EvaluationResult
from .inference import InferenceRunner, InferenceResult
from .checkpoint_loader import CheckpointLoader

__all__ = [
    'Evaluator',
    'EvaluationResult',
    'InferenceRunner',
    'InferenceResult',
    'CheckpointLoader',
]
