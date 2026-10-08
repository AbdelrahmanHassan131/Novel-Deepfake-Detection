"""
Early Stopping Hook.

Monitors a validation metric and halts training when the metric stops
improving after a configurable patience period.
Broadcasts stop decisions from rank 0 to all distributed processes to
prevent hanging or race conditions in DDP collectives.
"""

import torch


class EarlyStoppingHook:
    """
    Hook that halts training when a monitored validation metric stops improving.

    Args:
        monitor_metric (str): Metric to monitor ('auc', 'source_macro_auc', 'worst_source_recall_05',
            'balanced_accuracy', 'accuracy', or 'loss'). Defaults to 'auc'.
        patience (int): Number of validation intervals with no improvement before stopping. Defaults to 5.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement. Defaults to 0.0.
        min_epochs (int): Minimum epochs to complete before early stopping can trigger. Defaults to 0.
        mode (str, optional): 'min' or 'max'. If None, inferred ('min' for loss, 'max' otherwise).
        runtime: Optional DistributedRuntime instance for broadcasting stop decisions.
    """

    def __init__(self, monitor_metric='auc', patience=5, min_delta=0.0, min_epochs=0, mode=None, runtime=None):
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        if mode is None:
            self.mode = 'min' if monitor_metric == 'loss' else 'max'
        else:
            self.mode = mode
        self.runtime = runtime

        self.best_score = None
        self.wait_count = 0
        self.stopped_epoch = 0

    def state_dict(self):
        """Return early stopping state for checkpointing."""
        return {
            'best_score': self.best_score,
            'wait_count': self.wait_count,
            'stopped_epoch': self.stopped_epoch,
            'monitor_metric': self.monitor_metric,
            'patience': self.patience,
            'mode': self.mode,
            'min_delta': self.min_delta,
            'min_epochs': self.min_epochs,
        }

    def load_state_dict(self, state):
        """Restore early stopping state from checkpoint."""
        if not state:
            return
        if 'monitor_metric' in state and state['monitor_metric'] != self.monitor_metric:
            import warnings
            warnings.warn(
                f"[EarlyStopping] Resuming with different monitor_metric: "
                f"saved={state['monitor_metric']!r}, current={self.monitor_metric!r}. Resetting wait_count."
            )
            return
        self.best_score = state.get('best_score')
        self.wait_count = state.get('wait_count', 0)
        self.stopped_epoch = state.get('stopped_epoch', 0)

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        if self.patience <= 0:
            return

        is_main = self.runtime.is_main if self.runtime is not None else getattr(trainer, 'rank', 0) == 0
        should_stop = False
        error_msg = None

        if is_main:
            try:
                score = None
                if self.monitor_metric == 'loss':
                    score = getattr(result, 'loss', None)
                elif self.monitor_metric == 'source_macro_auc':
                    score = getattr(result, 'source_macro_auc', None)
                elif self.monitor_metric == 'worst_source_recall_05':
                    score = getattr(result, 'worst_source_recall_05', None)
                elif self.monitor_metric == 'auc':
                    score = getattr(result, 'auc', None)
                    if score is None:
                        score = getattr(result, 'accuracy', None)
                elif self.monitor_metric == 'balanced_accuracy':
                    score = getattr(result, 'balanced_accuracy', None)
                    if score is None:
                        score = getattr(result, 'accuracy', None)
                else:
                    score = getattr(result, 'accuracy', None)

                if score is None:
                    raise ValueError(
                        f"[EarlyStopping] Monitored metric '{self.monitor_metric}' is not available in validation result."
                    )

                current_epoch = getattr(trainer, 'current_epoch', 0)
                if current_epoch >= self.min_epochs:
                    if self.best_score is None:
                        self.best_score = score
                    else:
                        if self.mode == 'max':
                            improved = (score > self.best_score + self.min_delta)
                        else:
                            improved = (score < self.best_score - self.min_delta)

                        if improved:
                            self.best_score = score
                            self.wait_count = 0
                        else:
                            self.wait_count += 1
                            if self.wait_count >= self.patience:
                                should_stop = True
                                self.stopped_epoch = current_epoch
            except Exception as e:
                error_msg = str(e)

        # Broadcast stop decision and errors across all ranks in DDP
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            comm_container = [should_stop, error_msg]
            dist.broadcast_object_list(comm_container, src=0)
            should_stop, error_msg = comm_container

        if error_msg:
            raise RuntimeError(f"[EarlyStopping] Error evaluating metric on rank 0: {error_msg}")

        if should_stop:
            trainer.should_stop = True
