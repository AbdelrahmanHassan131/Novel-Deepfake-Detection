"""
Automatic Mixed Precision (AMP) integration.

Provides :class:`AmpMixin`, a mixin class that the Trainer can use to
transparently enable ``torch.cuda.amp.autocast`` and ``GradScaler``
when ``opt.use_amp == True``.

Design:
    AMP is implemented as a mixin rather than a separate layer so that
    it integrates directly into the training step without requiring
    the model implementations to know about mixed precision.

When AMP is disabled (``opt.use_amp`` is False or absent), all methods
are identity operations — zero overhead, zero behavioural change.

Usage::

    class Trainer(AmpMixin, BaseTrainer):
        def train_step(self, batch):
            with self.amp_autocast():
                self.model.set_input(batch)
                self.model.forward()
                loss = self.model.get_loss()
            self.amp_backward(loss)
            self.amp_step()
            return loss.item()

Interaction with the Training Engine:
    The ``Trainer`` inherits from ``AmpMixin`` and calls
    ``amp_autocast()`` / ``amp_backward()`` / ``amp_step()`` inside
    ``train_step()``.  No other component needs to know about AMP.

Checkpoint integration:
    ``amp_state_dict()`` / ``amp_load_state_dict()`` are called by the
    ``CheckpointManager`` during save/resume to persist the GradScaler
    state across restarts.
"""

import torch


class AmpMixin:
    """
    Mixin that adds AMP support to a Trainer.

    Reads ``opt.use_amp`` to decide whether to enable mixed precision.
    When disabled, all methods are no-ops.

    Attributes:
        _amp_enabled (bool): Whether AMP is active.
        _grad_scaler (torch.cuda.amp.GradScaler or None): The scaler
            used for loss scaling in mixed precision.

    Inputs:
        - ``opt.use_amp`` (bool): Enable/disable AMP.

    Outputs:
        - ``amp_autocast()`` context manager.
        - ``amp_backward(loss)`` — scaled backward pass.
        - ``amp_step()`` — scaler step + update.
        - ``amp_state_dict()`` — serialisable scaler state.
        - ``amp_load_state_dict(state)`` — restore scaler state.
    """

    def _init_amp(self, opt):
        """
        Initialize AMP state.

        Must be called during ``Trainer.__init__`` after the device
        is known.

        Args:
            opt: Options namespace.  Reads ``opt.use_amp``.
        """
        self._amp_enabled = getattr(opt, 'use_amp', False)

        if self._amp_enabled and not torch.cuda.is_available():
            print('[AMP] CUDA not available — disabling AMP.')
            self._amp_enabled = False

        if self._amp_enabled:
            self._grad_scaler = torch.cuda.amp.GradScaler()
            print('[AMP] Enabled with GradScaler.')
        else:
            self._grad_scaler = None

    # ------------------------------------------------------------------
    # Training integration
    # ------------------------------------------------------------------

    def amp_autocast(self):
        """
        Return an ``autocast`` context manager.

        When AMP is disabled, returns a no-op context manager.

        Returns:
            Context manager for mixed-precision forward pass.
        """
        if self._amp_enabled:
            return torch.cuda.amp.autocast()
        return _NullContext()

    def amp_backward(self, loss):
        """
        Perform a (possibly scaled) backward pass.

        When AMP is enabled, uses ``GradScaler.scale(loss).backward()``.
        When disabled, calls ``loss.backward()`` directly.

        Args:
            loss (torch.Tensor): The scalar loss tensor.
        """
        if self._amp_enabled:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def amp_step(self, optimizer):
        """
        Perform a (possibly scaled) optimizer step and update the scaler.

        When AMP is enabled, uses ``GradScaler.step()`` followed by
        ``GradScaler.update()``.  When disabled, calls
        ``optimizer.step()`` directly.

        Args:
            optimizer: The ``torch.optim.Optimizer`` to step.
        """
        if self._amp_enabled:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
        else:
            optimizer.step()

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------

    def amp_state_dict(self):
        """
        Return the GradScaler state dict for checkpointing.

        Returns:
            dict or None if AMP is disabled.
        """
        if self._amp_enabled and self._grad_scaler is not None:
            return self._grad_scaler.state_dict()
        return None

    def amp_load_state_dict(self, state):
        """
        Restore the GradScaler state from a checkpoint.

        Args:
            state (dict or None): State dict from a previous
                ``amp_state_dict()`` call.
        """
        if state is not None and self._amp_enabled and self._grad_scaler is not None:
            self._grad_scaler.load_state_dict(state)
            print('[AMP] GradScaler state restored from checkpoint.')


class _NullContext:
    """Minimal no-op context manager for when AMP is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
