"""
Optimizer Factory.

Centralizes optimizer construction. Supports every optimizer already
used by the project (Adam, SGD) with the exact same default parameters
each model trainer currently uses.

Usage:
    optimizer = build_optimizer(opt, model_or_params)
"""
import torch


def build_optimizer(opt, model_or_params):
    """
    Build an optimizer from the project options object.

    Reads ``opt.optim`` to select the optimizer family and uses the
    existing option attributes (``opt.lr``, ``opt.beta1``, ``opt.weight_decay``,
    ``opt.momentum``) to configure it.

    This function is a *drop-in replacement* for the per-model optimizer
    creation that was previously duplicated in every trainer ``__init__``.

    Args:
        opt: Options namespace.  Required attributes:
            - ``optim`` (str): ``'adam'`` or ``'sgd'``.
            - ``lr`` (float): learning rate.
            - ``beta1`` (float): Adam beta1 (only for adam).
          Optional attributes (with safe defaults):
            - ``weight_decay`` (float): L2 regularisation term.  Defaults to 0.
            - ``momentum`` (float): SGD momentum.  Defaults to 0.
        model_or_params: Either an ``nn.Module`` (uses ``.parameters()``)
            or an iterable of parameters / param-groups.

    Returns:
        A ``torch.optim.Optimizer`` instance.

    Raises:
        ValueError: If ``opt.optim`` is not one of the supported values.
    """
    # Accept both nn.Module and raw parameter iterables.
    if hasattr(model_or_params, 'parameters'):
        params = model_or_params.parameters()
    else:
        params = model_or_params

    weight_decay = getattr(opt, 'weight_decay', 0.0)
    optim_name = opt.optim.lower()

    if optim_name == 'adam':
        return torch.optim.Adam(
            params,
            lr=opt.lr,
            betas=(opt.beta1, 0.999),
            weight_decay=weight_decay,
        )
    elif optim_name == 'sgd':
        momentum = getattr(opt, 'momentum', 0.0)
        return torch.optim.SGD(
            params,
            lr=opt.lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer '{opt.optim}'. "
            f"Supported: ['adam', 'sgd']"
        )
