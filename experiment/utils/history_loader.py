"""
HistoryLoader — reads saved experiment files for post-training analysis.

All post-training tools (plotting, reports, comparisons) use this
loader to read experiment data.  No tool should parse JSON / CSV
directly.

Usage::

    from Refactored.experiment.utils import HistoryLoader

    loader = HistoryLoader('experiments/wang2020_progan_20260703_120000')
    history = loader.load_history()
    csv_data = loader.load_csv()
    opt = loader.load_opt()
"""

import json
import csv
import os


class HistoryLoader:
    """
    Read-only loader for saved experiment data.

    Args:
        experiment_dir (str): Path to an experiment directory
            (the one containing ``history.json``, ``metrics.csv``, etc.).
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_history(self):
        """
        Load the full JSON history.

        Returns:
            dict with ``'train'`` and ``'validation'`` lists.

        Raises:
            FileNotFoundError: If ``history.json`` is missing.
        """
        path = os.path.join(self.experiment_dir, 'history.json')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'history.json not found in {self.experiment_dir}')
        with open(path, 'r') as f:
            return json.load(f)

    def load_csv(self):
        """
        Load ``metrics.csv`` as a list of dicts.

        Returns:
            list of dicts, one per row.

        Raises:
            FileNotFoundError: If ``metrics.csv`` is missing.
        """
        path = os.path.join(self.experiment_dir, 'metrics.csv')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'metrics.csv not found in {self.experiment_dir}')
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)

    def load_opt(self):
        """
        Load the saved options as a dict.

        Tries ``opt.json`` first, falls back to parsing ``opt.txt``.

        Returns:
            dict of option key-value pairs.
        """
        json_path = os.path.join(self.experiment_dir, 'opt.json')
        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)

        txt_path = os.path.join(self.experiment_dir, 'opt.txt')
        if os.path.isfile(txt_path):
            return self._parse_opt_txt(txt_path)

        return {}

    def get_train_losses(self):
        """Return a list of per-epoch training losses."""
        history = self.load_history()
        return [r['train_loss'] for r in history.get('train', [])
                if 'train_loss' in r]

    def get_val_metrics(self, metric_name):
        """
        Extract a single validation metric across all epochs.

        Args:
            metric_name (str): Key name (e.g. ``'accuracy'``,
                ``'f1'``, ``'roc_auc'``).

        Returns:
            list of float values, one per validation epoch.
        """
        history = self.load_history()
        return [r[metric_name] for r in history.get('validation', [])
                if metric_name in r]

    def get_epochs(self):
        """Return the list of epoch numbers from training records."""
        history = self.load_history()
        return [r['epoch'] for r in history.get('train', [])
                if 'epoch' in r]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_opt_txt(filepath):
        """Best-effort key: value parser for opt.txt."""
        result = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    key, _, value = line.partition(':')
                    result[key.strip()] = value.strip()
        return result
