"""
Distributed Training Runtime.

Public API::

    from training.runtime import DistributedRuntime
    from training.runtime import AmpMixin
    from training.runtime import seed_everything

Usage::

    runtime = DistributedRuntime(opt)
    device = runtime.device
    model = runtime.wrap_model(model)
    loader = runtime.wrap_loader(train_loader, is_train=True)
    runtime.cleanup()
"""

from .distributed_runtime import DistributedRuntime
from .amp import AmpMixin
from .seed import seed_everything

__all__ = [
    'DistributedRuntime',
    'AmpMixin',
    'seed_everything',
]
