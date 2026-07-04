"""
Experiment Management System.

Public API::

    # Core
    from experiment import ExperimentManager, Experiment

    # Logging
    from experiment import ExperimentLogger

    # Metrics
    from experiment import MetricsCalculator, MetricAccumulator

    # Post-training utilities
    from experiment import HistoryLoader, ReportGenerator

    # Post-training visualisation
    from experiment.visualization import (
        plot_training_curves, compare_runs,
        plot_confusion_matrix, plot_roc_curve,
        plot_precision_recall_curve,
    )
"""

from .manager import ExperimentManager
from .experiment import Experiment
from .logger import ExperimentLogger
from .metrics import MetricsCalculator, MetricAccumulator
from .utils import HistoryLoader, ReportGenerator

__all__ = [
    # Core
    'ExperimentManager',
    'Experiment',
    # Logging
    'ExperimentLogger',
    # Metrics
    'MetricsCalculator',
    'MetricAccumulator',
    # Utils
    'HistoryLoader',
    'ReportGenerator',
]
