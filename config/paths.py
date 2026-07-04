"""
Path Manager.

Centralizes all filesystem path construction for experiments,
checkpoints, logs, and metrics.  No other module in the refactored
codebase should manually concatenate paths — use ``PathManager``
instead.

Constructed from a ``Config`` instance, it derives all paths from
``config.experiment.checkpoints_dir`` and ``config.experiment.name``.

Usage::

    from Refactored.config import PathManager

    paths = PathManager(config)

    # or from explicit values
    paths = PathManager.from_values(
        checkpoints_dir='./checkpoints',
        experiment_name='wang2020_progan',
    )

    print(paths.experiment_dir)    # ./checkpoints/wang2020_progan
    print(paths.best_checkpoint)   # ./checkpoints/wang2020_progan/best.pth
    print(paths.history_json)      # ./checkpoints/wang2020_progan/history.json
"""

import os


class PathManager:
    """Centralized filesystem path manager.

    All paths are derived lazily from ``experiment_dir`` which is
    ``{checkpoints_dir}/{experiment_name}``.

    Args:
        config: A ``Config`` instance.  Reads
            ``config.experiment.checkpoints_dir`` and
            ``config.experiment.name``.
    """

    def __init__(self, config):
        self._checkpoints_dir = config.experiment.checkpoints_dir
        self._experiment_name = config.experiment.name

    @classmethod
    def from_values(cls, checkpoints_dir, experiment_name):
        """Construct a PathManager from explicit values.

        Useful when a ``Config`` object is not available.

        Args:
            checkpoints_dir (str): Root checkpoint directory.
            experiment_name (str): Experiment name.

        Returns:
            PathManager instance.
        """
        # Create a lightweight proxy to avoid importing Config
        class _Proxy:
            class experiment:
                pass
        proxy = _Proxy()
        proxy.experiment.checkpoints_dir = checkpoints_dir
        proxy.experiment.name = experiment_name
        return cls(proxy)

    # ------------------------------------------------------------------
    # Directory paths
    # ------------------------------------------------------------------

    @property
    def experiment_dir(self):
        """Root directory for this experiment.

        ``{checkpoints_dir}/{experiment_name}``
        """
        return os.path.join(
            self._checkpoints_dir, self._experiment_name)

    @property
    def checkpoint_dir(self):
        """Checkpoints subdirectory.

        ``{experiment_dir}/checkpoints``
        """
        return os.path.join(self.experiment_dir, 'checkpoints')

    @property
    def plots_dir(self):
        """Plots subdirectory.

        ``{experiment_dir}/plots``
        """
        return os.path.join(self.experiment_dir, 'plots')

    @property
    def logs_dir(self):
        """Logs subdirectory.

        ``{experiment_dir}/logs``
        """
        return os.path.join(self.experiment_dir, 'logs')

    @property
    def metrics_dir(self):
        """Metrics subdirectory.

        ``{experiment_dir}/metrics``
        """
        return os.path.join(self.experiment_dir, 'metrics')

    @property
    def reports_dir(self):
        """Reports subdirectory.

        ``{experiment_dir}/reports``
        """
        return os.path.join(self.experiment_dir, 'reports')

    @property
    def train_log_dir(self):
        """TensorBoard training log directory.

        ``{experiment_dir}/train``
        """
        return os.path.join(self.experiment_dir, 'train')

    @property
    def val_log_dir(self):
        """TensorBoard validation log directory.

        ``{experiment_dir}/val``
        """
        return os.path.join(self.experiment_dir, 'val')

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    @property
    def history_json(self):
        """Path to ``history.json``.

        ``{experiment_dir}/history.json``
        """
        return os.path.join(self.experiment_dir, 'history.json')

    @property
    def metrics_csv(self):
        """Path to ``metrics.csv``.

        ``{experiment_dir}/metrics.csv``
        """
        return os.path.join(self.experiment_dir, 'metrics.csv')

    @property
    def best_checkpoint(self):
        """Path to ``best.pth``.

        ``{experiment_dir}/best.pth``
        """
        return os.path.join(self.experiment_dir, 'best.pth')

    @property
    def last_checkpoint(self):
        """Path to ``last.pth``.

        ``{experiment_dir}/last.pth``
        """
        return os.path.join(self.experiment_dir, 'last.pth')

    @property
    def opt_txt(self):
        """Path to ``opt.txt``.

        ``{experiment_dir}/opt.txt``
        """
        return os.path.join(self.experiment_dir, 'opt.txt')

    @property
    def opt_json(self):
        """Path to ``opt.json``.

        ``{experiment_dir}/opt.json``
        """
        return os.path.join(self.experiment_dir, 'opt.json')

    # ------------------------------------------------------------------
    # Dynamic paths
    # ------------------------------------------------------------------

    def epoch_checkpoint(self, epoch):
        """Path to an epoch-specific checkpoint.

        Args:
            epoch: Epoch number or label (e.g. ``5``, ``'latest'``).

        Returns:
            ``{experiment_dir}/model_epoch_{epoch}.pth``
        """
        return os.path.join(
            self.experiment_dir, f'model_epoch_{epoch}.pth')

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------

    def ensure_dirs(self):
        """Create all directories if they do not exist.

        Safe to call multiple times.  Uses ``os.makedirs`` with
        ``exist_ok=True``.
        """
        for dir_path in [
            self.experiment_dir,
            self.checkpoint_dir,
            self.plots_dir,
            self.logs_dir,
            self.metrics_dir,
            self.reports_dir,
            self.train_log_dir,
            self.val_log_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f'PathManager(\n'
            f'  experiment_dir={self.experiment_dir!r},\n'
            f'  checkpoint_dir={self.checkpoint_dir!r},\n'
            f'  plots_dir={self.plots_dir!r},\n'
            f'  logs_dir={self.logs_dir!r},\n'
            f'  metrics_dir={self.metrics_dir!r},\n'
            f'  best_checkpoint={self.best_checkpoint!r},\n'
            f'  last_checkpoint={self.last_checkpoint!r},\n'
            f')'
        )
