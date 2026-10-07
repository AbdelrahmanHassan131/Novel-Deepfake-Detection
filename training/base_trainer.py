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
from contextlib import nullcontext

from training.checkpoint_manager import CheckpointManager
from training.validator import Validator, ValidationResult
from training.runtime.distributed_runtime import DistributedRuntime
from training.runtime.amp import AmpMixin
from training.runtime.seed import seed_everything


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
        self.grad_accum_steps = max(1, getattr(opt, 'grad_accum_steps', 1))

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
            total_target_epochs = getattr(self.opt, 'niter', 100) + getattr(
                self.opt, 'niter_decay', 0
            )
        else:
            total_target_epochs = num_epochs

        additional_epochs = getattr(self.opt, 'additional_epochs', None)
        if additional_epochs is not None and additional_epochs > 0:
            end_epoch = self.current_epoch + additional_epochs + 1
        else:
            end_epoch = total_target_epochs + 1

        start_epoch = self.current_epoch + 1
        if start_epoch >= end_epoch:
            if self.runtime.is_main:
                print(
                    f'[BaseTrainer] Target total epochs ({end_epoch - 1}) already '
                    f'reached at current epoch {self.current_epoch}. No epochs to run.'
                )
            return {
                'best_metric': self.best_metric,
                'global_step': self.global_step,
            }

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
        """Run one epoch of training with sample-weighted gradient accumulation."""
        self.model.train()
        self.epoch_loss = 0.0
        self.epoch_batches = 0
        accum_steps = self.grad_accum_steps
        num_batches = len(self.train_loader)

        # Sizing and sample accounting:
        # Determine rank-local sample count without confusing global dataset size with per-rank batches under DDP.
        nominal_batch_size = getattr(self.train_loader, 'batch_size', getattr(self.opt, 'batch_size', 32))
        is_drop_last = getattr(self.train_loader, 'drop_last', False)

        local_num_samples = None
        if not is_drop_last:
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, '__len__'):
                try:
                    local_num_samples = len(self.train_loader.sampler)
                except Exception:
                    local_num_samples = None
            elif not getattr(self.runtime, 'is_distributed', False):
                if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, '__len__'):
                    try:
                        local_num_samples = len(self.train_loader.dataset)
                    except Exception:
                        local_num_samples = None

        if (not is_drop_last) and local_num_samples is not None and num_batches > 0:
            computed_tail = local_num_samples - (num_batches - 1) * nominal_batch_size
            last_batch_size = max(1, min(nominal_batch_size, computed_tail))
        else:
            last_batch_size = nominal_batch_size

        if hasattr(self.model, 'optimizer') and self.model.optimizer is not None:
            self.model.optimizer.zero_grad()

        if self.runtime.is_main:
            print(f'[Train] Epoch {self.current_epoch}: {num_batches:,} batches per rank; '
                  'waiting for first batch from the image loader...', flush=True)
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx == 0 and self.runtime.is_main:
                print('[Train] First batch received; starting forward/backward computation...', flush=True)
            self._fire('on_batch_start', self)

            window_start = (batch_idx // accum_steps) * accum_steps
            window_end = min(window_start + accum_steps, num_batches)
            current_window_size = window_end - window_start
            is_accum_end = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)

            # Sample weighting across micro-batches in the accumulation window:
            # Guarantees that smaller tail micro-batches are not overweighted compared to true combined sample mean.
            if accum_steps <= 1:
                loss_weight = 1.0
            else:
                if window_end < num_batches:
                    # Non-terminal accumulation window: all batches have full nominal size
                    loss_weight = 1.0 / max(1, current_window_size)
                else:
                    # Terminal accumulation window: includes the final (potentially partial) batch
                    cur_last_size = last_batch_size
                    if batch_idx == num_batches - 1:
                        if isinstance(batch, dict) and 'label' in batch and hasattr(batch['label'], 'shape'):
                            actual_sz = batch['label'].shape[0]
                            if actual_sz > 0:
                                cur_last_size = actual_sz
                        elif isinstance(batch, (tuple, list)) and len(batch) > 0 and hasattr(batch[0], 'shape'):
                            actual_sz = batch[0].shape[0]
                            if actual_sz > 0:
                                cur_last_size = actual_sz

                    window_batch_sizes = [
                        (cur_last_size if i == num_batches - 1 else nominal_batch_size)
                        for i in range(window_start, window_end)
                    ]
                    window_total_samples = sum(window_batch_sizes)
                    current_batch_samples = (cur_last_size if batch_idx == num_batches - 1 else nominal_batch_size)
                    loss_weight = current_batch_samples / max(1, window_total_samples)

            batch_loss = self.train_step(
                batch,
                is_accum_end=is_accum_end,
                accum_steps=current_window_size,
                loss_weight=loss_weight,
            )

            self.last_batch_loss = batch_loss
            self.epoch_loss += batch_loss
            self.epoch_batches += 1
            if is_accum_end:
                self.global_step += 1
                self.model.total_steps = self.global_step

            self._fire('on_batch_end', self)

    def train_step(self, batch, is_accum_end=True, accum_steps=1, loss_weight=None):
        """
        Execute one training iteration with optional AMP and sample-weighted gradient accumulation.

        When AMP is enabled, the forward pass runs under
        ``torch.cuda.amp.autocast`` and the backward pass uses
        ``GradScaler`` for loss scaling.

        When AMP is disabled, delegates to forward+backward with loss scaled
        by loss_weight, stepping optimizer at accumulation boundaries.

        Args:
            batch: A batch from the DataLoader (tuple of tensors).
            is_accum_end: Whether this step completes an accumulation window.
            accum_steps: Number of micro-batches in the current accumulation window.
            loss_weight: Explicit sample-proportional loss scale (defaults to 1.0 / accum_steps).

        Returns:
            float – the scalar loss for this batch.
        """
        self.model.set_input(batch)
        raw_model = self.model.model

        # In DDP, avoid unnecessary gradient synchronization during intermediate accumulation steps
        sync_context = raw_model.no_sync() if (
            self.runtime.is_distributed and not is_accum_end and hasattr(raw_model, 'no_sync')
        ) else nullcontext()

        effective_scale = loss_weight if loss_weight is not None else (1.0 / max(1, accum_steps))

        with sync_context:
            if self._amp_enabled:
                with self.amp_autocast():
                    self.model.forward()
                    loss = self.model.get_loss()

                scaled_loss = loss * effective_scale
                self.amp_backward(scaled_loss)

                if is_accum_end:
                    self.amp_step(self.model.optimizer)
                    self.model.optimizer.zero_grad()

                self.model.loss = loss
                return loss.item()
            else:
                if accum_steps > 1 or loss_weight is not None:
                    self.model.forward()
                    loss = self.model.get_loss()
                    scaled_loss = loss * effective_scale
                    scaled_loss.backward()

                    if is_accum_end:
                        self.model.optimizer.step()
                        self.model.optimizer.zero_grad()

                    self.model.loss = loss
                    return loss.item()
                else:
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
