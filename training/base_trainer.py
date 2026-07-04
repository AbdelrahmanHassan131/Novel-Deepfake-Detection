"""
BaseTrainer — abstract training lifecycle manager.

Owns the entire training lifecycle::

    fit()  →  train_epoch()  →  train_step()
                    ↕
               validate()
                    ↕
          scheduler.step()  →  save_checkpoint()

Architecture-agnostic.  Knows nothing about Wang, Wavelet, Fusion,
Xception, or any specific model.  All model interaction happens through
the ``BaseModel`` interface (``set_input``, ``forward``,
``optimize_parameters``, ``get_loss``).

DDP-ready:
    - Delegates all distributed logic to :class:`DistributedRuntime`.
    - ``sampler.set_epoch(epoch)`` called automatically.
    - Only rank 0 writes checkpoints and logs.
    - Compatible with ``DistributedSampler``.

AMP-ready:
    - Integrates :class:`AmpMixin` for automatic mixed precision.
    - Enabled via ``opt.use_amp = True``.
"""

import os
import torch
from abc import ABC, abstractmethod

from Refactored.training.checkpoint_manager import CheckpointManager
from Refactored.training.validator import Validator, ValidationResult
from Refactored.training.runtime.distributed_runtime import DistributedRuntime
from Refactored.training.runtime.amp import AmpMixin
from Refactored.training.runtime.seed import seed_everything


class BaseTrainer(AmpMixin, ABC):
    """
    Abstract base class that owns the training loop.

    Subclasses must implement:
        - :meth:`configure_optimizer`  – return the optimizer to use.
        - :meth:`configure_scheduler` – return the scheduler to use
          (or ``None``).

    The default :meth:`train_step` delegates to
    ``model.set_input → model.optimize_parameters`` which already
    contains forward + loss + backward + step.  Override it only if you
    need custom logic.

    The ``DistributedRuntime`` is created automatically.  The Trainer
    never touches ``torch.distributed`` directly — all DDP logic is
    encapsulated in the runtime.

    Args:
        model: A ``BaseModel`` instance.
        train_loader: Training DataLoader.
        opt: Project options namespace.
        val_loader: Validation DataLoader (optional).
        runtime: A pre-built ``DistributedRuntime`` (optional).
            If ``None``, one is created automatically from ``opt``.
    """

    def __init__(self, model, train_loader, opt, val_loader=None,
                 runtime=None):
        # --- runtime ---
        if runtime is not None:
            self.runtime = runtime
        else:
            self.runtime = DistributedRuntime(opt)

        # Convenience aliases (the Trainer queries these, not DDP)
        self.rank = self.runtime.rank
        self.device = self.runtime.device

        # --- seed ---
        base_seed = getattr(opt, 'seed', None)
        deterministic = getattr(opt, 'deterministic', False)
        if base_seed is not None:
            seed_everything(base_seed, rank=self.rank,
                            deterministic=deterministic)

        # --- model ---
        self.model = model
        self.runtime.wrap_model(model)

        # --- dataloaders ---
        self.train_loader = self.runtime.wrap_loader(
            train_loader, is_train=True
        )
        if val_loader is not None:
            self.val_loader = self.runtime.wrap_loader(
                val_loader, is_train=False
            )
        else:
            self.val_loader = None

        self.opt = opt

        # --- state ---
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.epoch_loss = 0.0
        self.epoch_batches = 0
        self.last_batch_loss = 0.0

        # --- AMP ---
        self._init_amp(opt)

        # --- components ---
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

        # Save dir
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            model=self.model,
            rank=self.rank,
        )

        # Validator
        self.validator = Validator()

        # Hooks
        self._hooks = []

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def configure_optimizer(self):
        """Return the optimizer.  Called once during __init__."""

    @abstractmethod
    def configure_scheduler(self):
        """Return the LR scheduler (or None).  Called once during __init__."""

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hook(self, hook):
        """Add a hook that will receive trainer events."""
        self._hooks.append(hook)

    def _fire(self, event_name, *args, **kwargs):
        """Call ``event_name`` on every registered hook."""
        for hook in self._hooks:
            fn = getattr(hook, event_name, None)
            if fn is not None:
                fn(*args, **kwargs)

    def _fire_validation_end(self, result):
        """
        Notify hooks about a completed validation.

        Called by :class:`ValidationHook` so that other hooks
        (e.g. :class:`CheckpointHook`) can react to the metric.
        """
        self._fire('on_validation_end', self, result)

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def fit(self, num_epochs=None):
        """
        Run the full training loop.

        Args:
            num_epochs (int): Override the number of epochs.
                Defaults to ``opt.niter + opt.niter_decay``.

        Returns:
            dict with final ``best_metric``, ``global_step``.
        """
        if num_epochs is None:
            num_epochs = getattr(self.opt, 'niter', 100) + getattr(
                self.opt, 'niter_decay', 0
            )

        start_epoch = self.current_epoch + 1
        end_epoch = start_epoch + num_epochs

        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch

            # DDP: set epoch on sampler (for DistributedSampler)
            if hasattr(self.train_loader, 'sampler'):
                sampler = self.train_loader.sampler
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)

            self._fire('on_epoch_start', self)

            self.train_epoch()

            # Hooks fire validation, scheduler, checkpoint, logging
            self._fire('on_epoch_end', self)

            # Barrier: ensure all ranks finish the epoch before proceeding
            self.runtime.barrier()

        return {
            'best_metric': self.best_metric,
            'global_step': self.global_step,
        }

    def train_epoch(self):
        """Run one epoch of training."""
        self.model.train()
        self.epoch_loss = 0.0
        self.epoch_batches = 0

        for batch in self.train_loader:
            self._fire('on_batch_start', self)

            batch_loss = self.train_step(batch)

            self.last_batch_loss = batch_loss
            self.epoch_loss += batch_loss
            self.epoch_batches += 1
            self.global_step += 1
            self.model.total_steps = self.global_step

            self._fire('on_batch_end', self)

    def train_step(self, batch):
        """
        Execute one training iteration.

        When AMP is enabled, the forward pass runs under
        ``torch.cuda.amp.autocast`` and the backward pass uses
        ``GradScaler`` for loss scaling.

        When AMP is disabled, this delegates to the model's existing
        ``set_input`` → ``optimize_parameters`` → ``get_loss``
        pipeline to preserve all current training behaviour.

        Args:
            batch: A batch from the DataLoader (tuple of tensors).

        Returns:
            float – the scalar loss for this batch.
        """
        if self._amp_enabled:
            # AMP-aware training step
            self.model.set_input(batch)
            self.model.model.zero_grad()

            with self.amp_autocast():
                self.model.forward()
                loss = self.model.get_loss()

            self.amp_backward(loss)
            self.amp_step(self.model.optimizer)
            self.model.loss = loss
            return loss.item()
        else:
            # Original training step — zero behavioural change
            self.model.set_input(batch)
            self.model.optimize_parameters()
            return self.model.loss.item()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, dataloader=None):
        """
        Run validation and return a ``ValidationResult``.

        Uses ``self.val_loader`` when *dataloader* is ``None``.
        """
        if dataloader is None:
            dataloader = self.val_loader
        if dataloader is None:
            raise ValueError('No validation dataloader provided.')

        result = self.validator.validate(self.model, dataloader)

        # Notify hooks
        self._fire_validation_end(result)
        return result

    # ------------------------------------------------------------------
    # Checkpoint convenience wrappers
    # ------------------------------------------------------------------

    def save_checkpoint(self):
        """Save ``last.pth`` via the CheckpointManager."""
        self.checkpoint_manager.save_last(
            epoch=self.current_epoch,
            best_metric=self.best_metric,
            global_step=self.global_step,
            scheduler=self.scheduler,
            amp_state=self.amp_state_dict(),
        )

    def load_checkpoint(self, filepath=None):
        """Load a checkpoint via the CheckpointManager."""
        info = self.checkpoint_manager.resume(
            filepath=filepath,
            scheduler=self.scheduler,
        )
        self.current_epoch = info['epoch']
        self.best_metric = info['best_metric']
        self.global_step = info['global_step']

        # Restore AMP scaler state
        amp_state = info.get('amp_state', None)
        if amp_state is not None:
            self.amp_load_state_dict(amp_state)

    def resume_training(self, filepath=None):
        """
        Resume training from a checkpoint.

        Restores epoch, optimizer, scheduler, learning rate, best metric,
        global step, and AMP scaler state so that training continues
        exactly where it stopped.
        """
        self.load_checkpoint(filepath)
        if self.runtime.is_main:
            print(
                f'[BaseTrainer] Resumed: epoch={self.current_epoch}, '
                f'global_step={self.global_step}, '
                f'best_metric={self.best_metric}'
            )
