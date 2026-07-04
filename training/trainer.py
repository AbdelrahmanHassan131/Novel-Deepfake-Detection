"""
Trainer — concrete, batteries-included Training Engine.

Wires together :class:`BaseTrainer` with the hook system and the
existing ``opt`` object.  This is the *only* class most callers need.

Usage::

    from models import build_model
    from training import Trainer

    model = build_model(opt)
    trainer = Trainer(model, train_loader, opt, val_loader=val_loader)
    trainer.fit()

The Trainer automatically:
    - Creates a :class:`DistributedRuntime` from ``opt`` (or accepts
      a pre-built one).
    - Wraps the model in DDP if distributed mode is active.
    - Wraps DataLoaders with ``DistributedSampler`` if needed.
    - Initializes AMP when ``opt.use_amp == True``.
    - Seeds all RNGs when ``opt.seed`` is set.
    - Builds the optimizer from the model (reuses the model's own
      optimizer by default — no behavioural change).
    - Builds a scheduler from ``opt`` (via :func:`build_scheduler`).
    - Registers the four standard hooks (Logger, Scheduler, Validation,
      Checkpoint) with sensible defaults that can be overridden through
      ``opt``.
    - Cleans up the distributed process group after training.

Hook registration order matters:  **Logger → Scheduler → Validation →
Checkpoint**.  Validation fires ``on_validation_end`` which Checkpoint
listens to for ``save_best``.  Scheduler steps after training but
*before* the metric is read by Checkpoint, matching the original
project flow.

Public API:
    Single-GPU users launch training exactly as before::

        python train.py --arch Wang2020_128 ...

    Multi-GPU users launch with ``torchrun``::

        torchrun --nproc_per_node=4 train.py --arch Wang2020_128 ...

    No code changes required between modes.
"""

from training.base_trainer import BaseTrainer
from training.scheduler_factory import build_scheduler
from training.hooks.logger_hook import LoggerHook
from training.hooks.scheduler_hook import SchedulerHook
from training.hooks.validation_hook import ValidationHook
from training.hooks.checkpoint_hook import CheckpointHook


class Trainer(BaseTrainer):
    """
    Concrete trainer for all deepfake-detection models.

    Accepts any ``BaseModel`` subclass.  No architecture-specific logic.

    Args:
        model: A ``BaseModel`` instance (already constructed via
               :func:`build_model`).
        train_loader: Training DataLoader.
        opt: Options namespace.
        val_loader: Validation DataLoader (optional).
        runtime: A pre-built ``DistributedRuntime`` (optional).
            If ``None``, one is created automatically from ``opt``.
    """

    def __init__(self, model, train_loader, opt, val_loader=None,
                 runtime=None, experiment_logger=None):
        # BaseTrainer.__init__ calls configure_optimizer / configure_scheduler
        # so we need self.opt available first.
        self._pre_opt = opt
        self._pre_model = model
        self.experiment_logger = experiment_logger

        super().__init__(
            model=model,
            train_loader=train_loader,
            opt=opt,
            val_loader=val_loader,
            runtime=runtime,
        )

        # --- register default hooks ---
        self._setup_hooks()

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def configure_optimizer(self):
        """
        Return the optimizer.

        Reuses the optimizer already created by the model trainer
        (e.g. ``Wang2020RawTrainer.__init__`` sets ``self.optimizer``).
        This means **zero behavioural change** — each model keeps its
        own optimizer configuration including weight_decay, momentum,
        betas, etc.
        """
        return getattr(self._pre_model, 'optimizer', None)

    def configure_scheduler(self):
        """
        Build a scheduler from ``opt``.

        If ``opt.lr_policy`` is not set, defaults to ``'none'`` (no
        scheduler) to preserve the existing manual
        ``adjust_learning_rate`` behaviour.
        """
        if self.optimizer is None:
            return None

        # Default to 'none' to avoid changing any model's current
        # training behaviour.  Users opt-in by setting opt.lr_policy.
        policy = getattr(self._pre_opt, 'lr_policy', 'none')
        if policy == 'none':
            return None
        return build_scheduler(self._pre_opt, self.optimizer)

    # ------------------------------------------------------------------
    # Hook setup
    # ------------------------------------------------------------------

    def _setup_hooks(self):
        """Register the four standard hooks."""
        opt = self.opt

        # 1. Logger — always on, rank-safe
        log_freq = getattr(opt, 'loss_freq', getattr(opt, 'log_freq', 50))
        self.register_hook(LoggerHook(
            log_freq=log_freq,
            rank=self.rank,
            experiment_logger=self.experiment_logger,
        ))

        # 2. Scheduler
        if self.scheduler is not None:
            self.register_hook(SchedulerHook(self.scheduler))

        # 3. Validation
        if self.val_loader is not None:
            val_freq = getattr(opt, 'val_epoch_freq', 1)
            self.register_hook(ValidationHook(
                val_loader=self.val_loader,
                val_epoch_freq=val_freq,
            ))

        # 4. Checkpoint — must come after Validation so it can react
        #    to on_validation_end with the updated best metric.
        save_freq = getattr(opt, 'save_epoch_freq', 1)
        self.register_hook(CheckpointHook(
            checkpoint_manager=self.checkpoint_manager,
            save_epoch_freq=save_freq,
        ))

    # ------------------------------------------------------------------
    # Lifecycle override
    # ------------------------------------------------------------------

    def fit(self, num_epochs=None):
        """
        Run the full training loop and clean up distributed resources.

        Extends ``BaseTrainer.fit`` to call ``runtime.cleanup()``
        after training completes, ensuring the process group is
        destroyed cleanly.
        """
        try:
            result = super().fit(num_epochs=num_epochs)
        finally:
            self.runtime.cleanup()
        return result
