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

    def __init__(self, model, train_loader=None, opt=None, val_loader=None,
                 runtime=None):
        # Flexible argument detection for test/legacy signature: BaseTrainer(opt, model, runtime)
        if (hasattr(model, 'batch_size') or hasattr(model, 'niter')) and hasattr(train_loader, 'train') and (hasattr(opt, 'is_distributed') or hasattr(opt, 'world_size')):
            actual_opt = model
            actual_model = train_loader
            actual_runtime = opt
            train_loader = None
            opt = actual_opt
            model = actual_model
            runtime = actual_runtime

        # --- runtime ---
        if runtime is not None:
            self.runtime = runtime
        else:
            self.runtime = DistributedRuntime(opt)

        # Convenience aliases (the Trainer queries these, not DDP)
        self.rank = getattr(self.runtime, 'rank', 0)
        self.device = getattr(self.runtime, 'device', 'cpu')

        # --- seed ---
        base_seed = getattr(opt, 'seed', None) if opt is not None else None
        deterministic = getattr(opt, 'deterministic', False) if opt is not None else False
        if base_seed is not None:
            seed_everything(base_seed, rank=self.rank,
                            deterministic=deterministic)

        # --- model ---
        self.model = model
        if hasattr(self.runtime, 'wrap_model'):
            self.runtime.wrap_model(model)

        # --- dataloaders ---
        if train_loader is not None and hasattr(self.runtime, 'wrap_loader'):
            self.train_loader = self.runtime.wrap_loader(
                train_loader, is_train=True
            )
        else:
            self.train_loader = train_loader

        if val_loader is not None and hasattr(self.runtime, 'wrap_loader'):
            self.val_loader = self.runtime.wrap_loader(
                val_loader, is_train=False
            )
        else:
            self.val_loader = val_loader

        self.opt = opt
        from training.runtime.profiler import PhaseProfiler
        self.profiler = PhaseProfiler(rank=self.rank, world_size=getattr(self.runtime, 'world_size', 1),
                                      device=self.device)

        # --- state ---
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.epoch_loss = 0.0
        self.epoch_batches = 0
        self.last_batch_loss = 0.0
        self.optimizer_steps_attempted = 0
        self.optimizer_steps_successful = 0
        self.grad_accum_steps = max(1, getattr(opt, 'grad_accum_steps', 1)) if opt is not None else 1
        self.should_stop = False

        # --- AMP ---
        self._init_amp(opt)

        # --- components ---
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

        # Save dir
        checkpoints_dir = getattr(opt, 'checkpoints_dir', './checkpoints') if opt is not None else './checkpoints'
        opt_name = getattr(opt, 'name', 'experiment') if opt is not None else 'experiment'
        self.save_dir = os.path.join(checkpoints_dir, opt_name)
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            model=self.model,
            rank=self.rank,
        )
        self.checkpoint_manager.trainer = self

        # Validator
        self.validator = Validator()

        # Hooks
        self._hooks = []

    # ------------------------------------------------------------------
    # Lifecycle hook points (can be overridden by subclasses)
    # ------------------------------------------------------------------

    def configure_optimizer(self):
        """Return the optimizer. Defaults to model.optimizer if available."""
        if hasattr(self, 'model') and hasattr(self.model, 'optimizer'):
            return self.model.optimizer
        return None

    def configure_scheduler(self):
        """Return the LR scheduler (or None). Defaults to None."""
        return None

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

            with self.profiler.phase('training_epoch_host'):
                self.train_epoch()

            # Hooks fire validation, scheduler, checkpoint, logging
            self._fire('on_epoch_end', self)

            # Barrier: ensure all ranks finish the epoch before proceeding
            self.runtime.barrier()
            cuda_timings = self.profiler.sync_cuda_events()
            performance = self.profiler.get_global_summary(
                getattr(self.opt, 'batch_size', 32), self.grad_accum_steps)
            performance['sampled_cuda_seconds_per_phase_call'] = cuda_timings
            performance['note'] = ('Cumulative end-to-end throughput includes startup/validation/checkpoints. '
                                   'Host phase timers measure wall/enqueue time; CUDA samples are bounded '
                                   'to the first 64 phase calls per epoch, not steady-state GPU utilization.')
            import json
            with open(os.path.join(self.save_dir, f'performance_rank{self.rank}.json'), 'w', encoding='utf-8') as stream:
                json.dump(performance, stream, indent=2)

            if getattr(self, 'should_stop', False):
                if self.runtime.is_main:
                    print(
                        f'[EarlyStopping] Early stopping triggered at epoch {epoch}. '
                        'Terminating training loop gracefully across all ranks.',
                        flush=True
                    )
                break

        return {
            'best_metric': self.best_metric,
            'global_step': self.global_step,
        }

    def train_epoch(self):
        """Run one epoch of training with sample-weighted gradient accumulation."""
        self.model.train()
        self.epoch_loss = 0.0
        self.epoch_batches = 0
        total_epoch_samples = 0
        device = getattr(self.runtime, 'device', 'cpu')
        epoch_loss_sum_tensor = torch.tensor(0.0, device=device)
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
        def timed_batches():
            with self.profiler.phase('loader_iterator_startup'):
                iterator = iter(self.train_loader)
            for index in range(num_batches):
                with self.profiler.phase('first_batch_latency' if index == 0 else 'steady_data_wait'):
                    batch = next(iterator)
                yield batch

        for batch_idx, batch in enumerate(timed_batches()):
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

            batch_loss_tensor = self.train_step(
                batch,
                is_accum_end=is_accum_end,
                accum_steps=current_window_size,
                loss_weight=loss_weight,
            )

            cur_samples = int(batch['label'].shape[0] if isinstance(batch, dict) else batch[-1].shape[0])
            self.profiler.add_samples(cur_samples, is_optimizer_step=is_accum_end)
            if not isinstance(batch_loss_tensor, torch.Tensor):
                batch_loss_tensor = torch.tensor(batch_loss_tensor, device=device)

            epoch_loss_sum_tensor = epoch_loss_sum_tensor + (batch_loss_tensor * cur_samples)
            total_epoch_samples += cur_samples
            self.epoch_batches += 1

            # Only synchronize CPU scalar at logging boundaries or final batch
            log_freq = getattr(self.opt, 'log_freq', 50)
            should_sync = (
                batch_idx == 0 or
                (log_freq > 0 and self.epoch_batches % log_freq == 0) or
                (batch_idx == num_batches - 1)
            )
            if should_sync:
                self.last_batch_loss = float(batch_loss_tensor.item())

            if is_accum_end:
                self.global_step += 1
                self.model.total_steps = self.global_step

            self._fire('on_batch_end', self)

        # Finalize sample-weighted epoch loss
        if total_epoch_samples > 0:
            avg_loss = float((epoch_loss_sum_tensor / total_epoch_samples).item())
        else:
            avg_loss = 0.0
        self.sample_weighted_epoch_loss = avg_loss
        self.epoch_loss = avg_loss * max(1, self.epoch_batches)

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
        with self.profiler.phase('host_to_device_transfer', sample_cuda=True):
            self.model.set_input(batch)
        raw_model = self.model.model

        # In DDP, avoid unnecessary gradient synchronization during intermediate accumulation steps
        sync_context = raw_model.no_sync() if (
            self.runtime.is_distributed and not is_accum_end and hasattr(raw_model, 'no_sync')
        ) else nullcontext()

        effective_scale = loss_weight if loss_weight is not None else (1.0 / max(1, accum_steps))

        with sync_context, self.profiler.phase('forward_backward_optimizer', sample_cuda=True):
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
                return loss.detach()
            else:
                if accum_steps > 1 or loss_weight is not None:
                    self.model.forward()
                    loss = self.model.get_loss()
                    scaled_loss = loss * effective_scale
                    scaled_loss.backward()

                    if is_accum_end:
                        self.model.optimizer.step()
                        self.model.optimizer.zero_grad()
                        self.optimizer_steps_attempted += 1
                        self.optimizer_steps_successful += 1

                    self.model.loss = loss
                    return loss.detach()
                else:
                    self.model.optimize_parameters()
                    self.optimizer_steps_attempted += 1
                    self.optimizer_steps_successful += 1
                    return self.model.loss.detach()

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
        self.optimizer_steps_attempted = info.get('optimizer_steps_attempted', self.global_step)
        self.optimizer_steps_successful = info.get('optimizer_steps_successful', self.global_step)

        # Restore early stopping state if present
        if info.get('early_stopping_state'):
            for hook in self._hooks:
                if hasattr(hook, 'load_state_dict'):
                    hook.load_state_dict(info['early_stopping_state'])

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
