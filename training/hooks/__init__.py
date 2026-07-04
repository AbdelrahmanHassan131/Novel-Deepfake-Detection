"""
Training Hooks.

Lightweight event hooks for extending trainer behavior.

Available hooks:
    - CheckpointHook: saves checkpoints on epoch end
    - ValidationHook: runs validation on epoch end
    - LoggerHook: logs training metrics
    - SchedulerHook: steps the learning rate scheduler
"""

from .checkpoint_hook import CheckpointHook
from .validation_hook import ValidationHook
from .logger_hook import LoggerHook
from .scheduler_hook import SchedulerHook

__all__ = [
    'CheckpointHook',
    'ValidationHook',
    'LoggerHook',
    'SchedulerHook',
]
