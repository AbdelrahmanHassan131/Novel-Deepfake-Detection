"""
Checkpoint Hook.

Saves checkpoints at configurable intervals during training.

Events handled:
    - on_epoch_end: saves ``last.pth`` every epoch, ``model_epoch_N.pth``
      every ``save_epoch_freq`` epochs, and ``best.pth`` when the
      monitored metric improves.

Interaction with the Training Engine:
    Registered by the ``Trainer`` as the last hook.  Delegates all
    file I/O to :class:`CheckpointManager`.  Reads ``trainer.best_metric``
    to decide whether to save ``best.pth``.

    Passes the AMP scaler state (via ``trainer.amp_state_dict()``)
    through to the checkpoint so that mixed precision training can
    be resumed exactly.
"""


class CheckpointHook:
    """
    Hook that delegates checkpoint writes to :class:`CheckpointManager`.

    Args:
        checkpoint_manager: A ``CheckpointManager`` instance.
        save_epoch_freq (int): Save a numbered checkpoint every N epochs.
            Defaults to 1 (every epoch).
    """

    def __init__(self, checkpoint_manager, save_epoch_freq=1):
        self.ckpt = checkpoint_manager
        self.save_epoch_freq = save_epoch_freq

    # ---- helpers ----

    @staticmethod
    def _get_amp_state(trainer):
        """Safely extract AMP state from the trainer."""
        if hasattr(trainer, 'amp_state_dict'):
            return trainer.amp_state_dict()
        return None

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        """Save last (always) and epoch checkpoint (every N epochs)."""
        amp_state = self._get_amp_state(trainer)

        self.ckpt.save_last(
            epoch=trainer.current_epoch,
            best_metric=trainer.best_metric,
            global_step=trainer.global_step,
            scheduler=trainer.scheduler,
            amp_state=amp_state,
        )

        if self.save_epoch_freq > 0 and (
            trainer.current_epoch % self.save_epoch_freq == 0
        ):
            self.ckpt.save_epoch(
                epoch=trainer.current_epoch,
                best_metric=trainer.best_metric,
                global_step=trainer.global_step,
                scheduler=trainer.scheduler,
                amp_state=amp_state,
            )

    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        """Save ``best.pth`` when the monitored metric improves."""
        current_metric = result.accuracy

        if trainer.best_metric is None or current_metric > trainer.best_metric:
            trainer.best_metric = current_metric
            amp_state = self._get_amp_state(trainer)
            self.ckpt.save_best(
                epoch=trainer.current_epoch,
                best_metric=trainer.best_metric,
                global_step=trainer.global_step,
                scheduler=trainer.scheduler,
                amp_state=amp_state,
            )
