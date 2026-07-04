"""
Experiment — lightweight container for a single experiment's paths.

An ``Experiment`` is created by :class:`ExperimentManager` and exposes
every file path and directory path that other modules need.  No other
module should construct experiment paths manually.

Usage::

    exp = manager.create('wang2020_progan', opt)
    print(exp.checkpoint_dir)
    print(exp.history_path)
    print(exp.metrics_csv_path)
"""

import os


class Experiment:
    """
    Immutable path container for a single experiment.

    Args:
        name (str): Human-readable experiment name.
        experiment_id (str): Unique experiment ID (timestamp).
        root_dir (str): Absolute path to the experiment directory.
    """

    def __init__(self, name, experiment_id, root_dir):
        self.name = name
        self.experiment_id = experiment_id
        self.root_dir = root_dir

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    @property
    def history_path(self):
        """Path to ``history.json``."""
        return os.path.join(self.root_dir, 'history.json')

    @property
    def metrics_csv_path(self):
        """Path to ``metrics.csv``."""
        return os.path.join(self.root_dir, 'metrics.csv')

    @property
    def opt_txt_path(self):
        """Path to ``opt.txt``."""
        return os.path.join(self.root_dir, 'opt.txt')

    @property
    def opt_json_path(self):
        """Path to ``opt.json``."""
        return os.path.join(self.root_dir, 'opt.json')

    # ------------------------------------------------------------------
    # Directory paths
    # ------------------------------------------------------------------

    @property
    def checkpoint_dir(self):
        """Path to the ``checkpoints/`` subdirectory."""
        return os.path.join(self.root_dir, 'checkpoints')

    @property
    def plot_dir(self):
        """Path to the ``plots/`` subdirectory."""
        return os.path.join(self.root_dir, 'plots')

    @property
    def report_dir(self):
        """Path to the ``reports/`` subdirectory."""
        return os.path.join(self.root_dir, 'reports')

    @property
    def log_dir(self):
        """Path to the ``logs/`` subdirectory."""
        return os.path.join(self.root_dir, 'logs')

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f'Experiment(name={self.name!r}, '
            f'id={self.experiment_id!r}, '
            f'root={self.root_dir!r})'
        )
