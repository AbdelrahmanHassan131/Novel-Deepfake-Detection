"""
Automatic Mixed Precision (AMP) integration.

Provides :class:`AmpMixin`, a mixin class that the Trainer uses to
transparently enable modern ``torch.amp`` (or legacy ``torch.cuda.amp``)
autocasting and ``GradScaler`` when ``opt.use_amp == True``.

Features:
    - Supports FP16 (standard on T4 GPUs).
    - Supports BF16 only when runtime-supported (checked via torch.cuda.is_bf16_supported()).
    - Tracks optimizer steps attempted vs successful (detecting AMP inf/nan skips).
    - Serializes GradScaler state dictionary for seamless checkpoint resumption.
"""

from contextlib import nullcontext
import torch


class AmpMixin:
    """
    Mixin that adds AMP support and step monitoring to a Trainer.
    """

    def _init_amp(self, opt):
        """
        Initialize AMP state, precision, scaler, and step accounting.
        """
        self._amp_enabled = getattr(opt, 'use_amp', False)
        self.optimizer_steps_attempted = 0
        self.optimizer_steps_successful = 0

        device = getattr(self, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if self._amp_enabled and (not torch.cuda.is_available() or torch.device(device).type != 'cuda'):
            print('[AMP] CUDA not available — disabling AMP.')
            self._amp_enabled = False

        if self._amp_enabled:
            # Check requested dtype
            requested_dtype = str(getattr(opt, 'amp_dtype', 'fp16')).lower()
            if requested_dtype == 'bf16':
                if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                    self._amp_dtype = torch.bfloat16
                    print('[AMP] Enabled with bfloat16 on supported CUDA hardware.')
                else:
                    print('[AMP] bfloat16 requested but not supported on this CUDA device. Falling back to float16.')
                    self._amp_dtype = torch.float16
            else:
                self._amp_dtype = torch.float16

            # Modern torch.amp vs legacy torch.cuda.amp
            device_type = 'cuda'
            self._device_type = device_type
            use_modern = hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast')

            if use_modern:
                self._autocast_fn = lambda: torch.amp.autocast(device_type, dtype=self._amp_dtype)
            else:
                self._autocast_fn = lambda: torch.cuda.amp.autocast(dtype=self._amp_dtype)

            # GradScaler (only enabled for float16)
            scaler_enabled = (self._amp_dtype == torch.float16)
            if use_modern and hasattr(torch.amp, 'GradScaler'):
                self._grad_scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
            else:
                self._grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

            dtype_name = 'bfloat16' if self._amp_dtype == torch.bfloat16 else 'float16'
            print(f'[AMP] Initialized: dtype={dtype_name}, GradScaler enabled={scaler_enabled}.')
        else:
            self._amp_dtype = None
            self._grad_scaler = None
            self._autocast_fn = None

    # ------------------------------------------------------------------
    # Training integration
    # ------------------------------------------------------------------

    def amp_autocast(self):
        """Return autocast context manager or nullcontext."""
        if self._amp_enabled and self._autocast_fn is not None:
            return self._autocast_fn()
        return nullcontext()

    def amp_backward(self, loss):
        """Perform (scaled) backward pass."""
        if self._amp_enabled and self._grad_scaler is not None and self._grad_scaler.is_enabled():
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def amp_step(self, optimizer):
        """
        Perform optimizer step and record whether step succeeded or was skipped by GradScaler.

        Returns:
            bool: True if step was executed, False if skipped due to Inf/NaN.
        """
        self.optimizer_steps_attempted += 1

        if self._amp_enabled and self._grad_scaler is not None and self._grad_scaler.is_enabled():
            scale_before = self._grad_scaler.get_scale()
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
            scale_after = self._grad_scaler.get_scale()

            # If scale decreased, inf/nan was encountered and optimizer.step() was skipped
            skipped = (scale_after < scale_before)
            if not skipped:
                self.optimizer_steps_successful += 1
            return not skipped
        else:
            optimizer.step()
            self.optimizer_steps_successful += 1
            return True

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------

    def amp_state_dict(self):
        """Return GradScaler state dict for checkpointing."""
        if self._amp_enabled and self._grad_scaler is not None and self._grad_scaler.is_enabled():
            return self._grad_scaler.state_dict()
        return None

    def amp_load_state_dict(self, state):
        """Restore GradScaler state from a checkpoint."""
        if state is not None and self._amp_enabled and self._grad_scaler is not None and self._grad_scaler.is_enabled():
            self._grad_scaler.load_state_dict(state)
