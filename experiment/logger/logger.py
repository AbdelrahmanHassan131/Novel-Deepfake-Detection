"""
ExperimentLogger — unified facade for structured experiment logging.

Delegates to :class:`JsonLogger` and :class:`CsvLogger` so that
callers never interact with file formats directly.

Design principles:
    - Structured data only — no formatted text dumps.
    - Incremental writes — data is flushed after every log call.
    - No duplicated information — JSON stores the full history,
      CSV stores the flat tabular subset.

Usage::

    from experiment.logger import ExperimentLogger

    logger = ExperimentLogger(experiment)
    logger.log_epoch(epoch=1, train_loss=0.42, lr=1e-3, elapsed=12.5)
    logger.log_validation(epoch=1, val_loss=0.38, accuracy=0.91, ...)
    logger.close()
"""

from .json_logger import JsonLogger
from .csv_logger import CsvLogger


class ExperimentLogger:
    """
    Unified experiment logger.

    Writes training and validation data to both ``history.json``
    (complete history) and ``metrics.csv`` (tabular summary).

    Args:
        experiment: An :class:`Experiment` instance that provides
            file paths.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        self._json_logger = JsonLogger(experiment.history_path)
        self._csv_logger = CsvLogger(experiment.metrics_csv_path)

    # ------------------------------------------------------------------
    # Training logging
    # ------------------------------------------------------------------

    def log_epoch(self, epoch, train_loss, lr, elapsed,
                  global_step=None, num_batches=None):
        """
        Log end-of-epoch training data.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Average training loss for the epoch.
            lr (float): Current learning rate.
            elapsed (float): Epoch wall-clock time in seconds.
            global_step (int, optional): Total training steps so far.
            num_batches (int, optional): Batches processed this epoch.
        """
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': lr,
            'elapsed_seconds': round(elapsed, 2),
        }
        if global_step is not None:
            record['global_step'] = global_step
        if num_batches is not None:
            record['num_batches'] = num_batches

        self._json_logger.log_train(record)

    # ------------------------------------------------------------------
    # Validation logging
    # ------------------------------------------------------------------

    def log_validation(self, epoch, val_loss, accuracy, precision,
                       recall, f1, roc_auc, num_samples=None):
        """
        Log validation results.

        Args:
            epoch (int): Epoch at which validation was run.
            val_loss (float): Average validation loss.
            accuracy (float): Validation accuracy.
            precision (float): Validation precision.
            recall (float): Validation recall.
            f1 (float): Validation F1 score.
            roc_auc (float): Validation ROC AUC.
            num_samples (int, optional): Number of validation samples.
        """
        record = {
            'epoch': epoch,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
        }
        if num_samples is not None:
            record['num_samples'] = num_samples

        self._json_logger.log_validation(record)
        self._csv_logger.write_row(record)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Flush and close all underlying loggers."""
        self._json_logger.close()
        self._csv_logger.close()
