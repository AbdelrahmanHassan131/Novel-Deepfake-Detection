"""
Scheduler Factory.

Centralizes learning-rate scheduler construction.

The existing codebase uses a manual ``adjust_learning_rate`` method
(divide LR by 10) triggered at specific epochs.  This factory wraps
that pattern in standard PyTorch schedulers so the Training Engine can
call ``scheduler.step()`` uniformly.

Usage:
    scheduler = build_scheduler(opt, optimizer)
"""
from torch.optim import lr_scheduler


def build_scheduler(opt, optimizer):
    """
    Build a learning-rate scheduler from the project options object.

    Reads ``opt.lr_policy`` (if present) to decide which scheduler to
    create.  Falls back to ``'step'`` to match the existing
    ``adjust_learning_rate`` behaviour (LR / 10 every ``opt.lr_decay_iters``
    epochs).

    Supported policies:
        - ``'step'``: :class:`StepLR` – multiply LR by ``gamma`` every
          ``step_size`` epochs.  Defaults:  ``step_size = opt.lr_decay_iters``
          or 10, ``gamma = 0.1``.
        - ``'plateau'``: :class:`ReduceLROnPlateau` – reduce LR when a
          monitored metric stops improving.
        - ``'cosine'``: :class:`CosineAnnealingLR`.
        - ``'none'``: No scheduler (returns ``None``).

    Args:
        opt: Options namespace.
        optimizer: A ``torch.optim.Optimizer``.

    Returns:
        A scheduler instance, or ``None`` if policy is ``'none'``.

    Raises:
        ValueError: If the policy string is not recognised.
    """
    policy = getattr(opt, 'lr_policy', 'step').lower()

    if policy == 'step':
        step_size = getattr(opt, 'lr_decay_iters', 10)
        gamma = getattr(opt, 'lr_gamma', 0.1)
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif policy == 'plateau':
        patience = getattr(opt, 'lr_patience', 5)
        factor = getattr(opt, 'lr_gamma', 0.1)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=factor,
            patience=patience,
            verbose=True,
        )

    elif policy == 'cosine':
        n_epochs = getattr(opt, 'niter', 100) + getattr(opt, 'niter_decay', 0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
        )

    elif policy == 'none':
        return None

    else:
        raise ValueError(
            f"Unsupported lr_policy '{policy}'. "
            f"Supported: ['step', 'plateau', 'cosine', 'none']"
        )
