"""
Configuration Loader.

Provides a single entry-point ``load_config`` that:
    1. Converts a legacy ``opt`` object to a ``Config``.
    2. Runs validation checks.
    3. Returns a frozen, ready-to-use ``Config``.

This is the recommended way for all Refactored modules to obtain a
``Config`` instance.

Usage::

    from config import load_config

    config = load_config(opt)
    # config is validated and frozen
"""

from config.configuration import Config
from config.validator import ConfigValidator


def load_config(opt, validate=True, freeze=True):
    """Load and validate a Config from a legacy opt object.

    This is the primary entry-point for obtaining a ``Config`` instance
    in the refactored codebase.

    Args:
        opt: A legacy ``argparse.Namespace`` or similar options object.
        validate (bool): Whether to run validation checks.
            Defaults to ``True``.  Set to ``False`` for testing or
            when validation is handled separately.
        freeze (bool): Whether to freeze the config after creation.
            Defaults to ``True``.  Frozen configs raise
            ``AttributeError`` on mutation attempts.

    Returns:
        Config: A fully populated, optionally validated and frozen
        configuration object.

    Raises:
        ConfigurationError: If validation is enabled and fatal errors
            are found.
    """
    config = Config.from_opt(opt)

    if validate:
        ConfigValidator.validate_or_raise(config)

    if freeze:
        config.freeze()

    return config


def load_config_from_defaults(validate=False, freeze=False):
    """Load a Config with all default values.

    Convenience function for testing and development.

    Args:
        validate (bool): Whether to run validation.  Defaults to
            ``False`` since default values may reference non-existent
            paths.
        freeze (bool): Whether to freeze the config.  Defaults to
            ``False`` for test flexibility.

    Returns:
        Config: A config with all default values.
    """
    config = Config.from_defaults()

    if validate:
        ConfigValidator.validate_or_raise(config)

    if freeze:
        config.freeze()

    return config
