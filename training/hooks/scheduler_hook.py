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

    For ``ReduceLROnPlateau`` schedulers, the hook uses the most recent
    validation metric (defaulting to ``accuracy``) when calling
    ``scheduler.step(metric)``.

    Args:
        scheduler: A PyTorch LR scheduler, or ``None`` (no-op).
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._last_metric = None

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            # Need a metric to decide whether to reduce.
            metric = self._last_metric
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        # Stash the metric so on_epoch_end can use it for plateau.
        self._last_metric = result.accuracy
