"""
Refactored Training Engine.

Public API::

    from Refactored.training import Trainer
    from Refactored.training import Validator, ValidationResult
    from Refactored.training import CheckpointManager
    from Refactored.training import build_optimizer, build_scheduler

    # Distributed runtime
    from Refactored.training import DistributedRuntime
    from Refactored.training import seed_everything

Usage::

    # Single GPU — unchanged
    model = build_model(opt)
    trainer = Trainer(model, train_loader, opt, val_loader=val_loader)
    trainer.fit()

    # Multi-GPU via torchrun — no code changes needed
    # torchrun --nproc_per_node=4 train.py ...
"""

from .trainer import Trainer
from .validator import Validator, ValidationResult
from .checkpoint_manager import CheckpointManager
from .optimizer_factory import build_optimizer
from .scheduler_factory import build_scheduler
from .runtime import DistributedRuntime, seed_everything

__all__ = [
    'Trainer',
    'Validator',
    'ValidationResult',
    'CheckpointManager',
    'build_optimizer',
    'build_scheduler',
    'DistributedRuntime',
    'seed_everything',
]
