"""
Experiment Management System.

Public API::

    # Core
    from Refactored.experiment import ExperimentManager, Experiment

    # Logging
    from Refactored.experiment import ExperimentLogger

    # Metrics
    from Refactored.experiment import MetricsCalculator, MetricAccumulator

    # Post-training utilities
    from Refactored.experiment import HistoryLoader, ReportGenerator

    # Post-training visualisation
    from Refactored.experiment.visualization import (
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
