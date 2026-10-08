"""
Scheduler Hook.

Steps the learning-rate scheduler at the end of each epoch.

Events handled:
    - on_epoch_end: calls ``scheduler.step()`` (or ``scheduler.step(metric)``
      for ReduceLROnPlateau).
    - on_validation_end: stashes the latest metric for plateau schedulers.
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau


class SchedulerHook:
    """
    Hook that advances the LR scheduler once per epoch.

    For ``ReduceLROnPlateau`` schedulers, the hook uses the selected
    validation metric (or loss for mode='min'), stepping ONLY when fresh
    validation occurred in the current epoch.

    Args:
        scheduler: A PyTorch LR scheduler, or ``None`` (no-op).
        monitor_metric (str): Monitored metric name. Defaults to 'auc'.
    """

    def __init__(self, scheduler, monitor_metric='auc'):
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self._fresh_metric = None
        self._validation_occurred = False

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        self._fresh_metric = None
        self._validation_occurred = False

    def on_epoch_end(self, trainer):
        if self.scheduler is None:
            return

        is_plateau = isinstance(self.scheduler, ReduceLROnPlateau) or hasattr(self.scheduler, 'is_better') or hasattr(self.scheduler, 'mode')
        if is_plateau:
            # Plateau schedulers require a metric and must only step when new validation occurred
            if self._validation_occurred and self._fresh_metric is not None:
                self.scheduler.step(self._fresh_metric)
        else:
            self.scheduler.step()

    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        mode = getattr(self.scheduler, 'mode', 'max') if hasattr(self.scheduler, 'mode') else 'max'
        metric_name = getattr(getattr(trainer, 'opt', None), 'monitor_metric', self.monitor_metric)

        if mode == 'min':
            metric = getattr(result, 'loss', None)
        else:
            if metric_name == 'source_macro_auc' and getattr(result, 'source_macro_auc', None) is not None:
                metric = result.source_macro_auc
            elif metric_name == 'worst_source_recall_05' and getattr(result, 'worst_source_recall_05', None) is not None:
                metric = result.worst_source_recall_05
            elif metric_name == 'auc' and getattr(result, 'auc', None) is not None:
                metric = result.auc
            elif metric_name == 'balanced_accuracy' and getattr(result, 'balanced_accuracy', None) is not None:
                metric = result.balanced_accuracy
            else:
                metric = getattr(result, 'accuracy', None)

        self._fresh_metric = metric
        self._validation_occurred = True
