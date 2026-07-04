"""
Validation Hook.

Runs validation at configurable intervals during training.

Events handled:
    - on_epoch_end: runs the Validator every ``val_epoch_freq`` epochs.
"""

from Refactored.training.validator import Validator


class ValidationHook:
    """
    Hook that runs validation at the end of selected epochs.

    Args:
        val_loader: Validation DataLoader.
        val_epoch_freq (int): Validate every N epochs. Defaults to 1.
    """

    def __init__(self, val_loader, val_epoch_freq=1):
        self.val_loader = val_loader
        self.val_epoch_freq = val_epoch_freq
        self.validator = Validator()
        self.last_result = None

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        """Run validation if this epoch matches the frequency."""
        if self.val_epoch_freq <= 0:
            return

        if trainer.current_epoch % self.val_epoch_freq == 0:
            result = self.validator.validate(trainer.model, self.val_loader)
            self.last_result = result

            # Notify all hooks (including CheckpointHook) about val result
            trainer._fire_validation_end(result)

    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        # This hook triggers validation; it does not react to itself.
        pass
