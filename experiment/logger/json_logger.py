"""
JsonLogger — incremental JSON history writer.

Maintains the complete experiment history as a JSON file with
the structure::

    {
        "train": [
            {"epoch": 1, "train_loss": 0.42, "lr": 0.001, ...},
            ...
        ],
        "validation": [
            {"epoch": 1, "val_loss": 0.38, "accuracy": 0.91, ...},
            ...
        ]
    }

Writes are incremental: the file is rewritten after every ``log_*``
call to guarantee crash-safe persistence.
"""

import json
import os


class JsonLogger:
    """
    JSON-based experiment history logger.

    Args:
        filepath (str): Path to the JSON history file.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self._history = self._load_or_create()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(self, record):
        """
        Append a step-level training record and flush to disk.

        Args:
            record (dict): Step-level metrics (step, epoch, batch, loss, lr).
        """
        if 'steps' not in self._history:
            self._history['steps'] = []
        self._history['steps'].append(record)
        self._flush()

    def log_train(self, record):
        """
        Append a training record and flush to disk.

        Args:
            record (dict): Training data for one epoch.
        """
        self._history['train'].append(record)
        self._flush()

    def log_validation(self, record):
        """
        Append a validation record and flush to disk.

        Args:
            record (dict): Validation metrics for one epoch.
        """
        self._history['validation'].append(record)
        self._flush()

    @property
    def history(self):
        """Return the full history dict (read-only view)."""
        return self._history

    def close(self):
        """Final flush."""
        self._flush()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_or_create(self):
        """Load existing history or create an empty structure."""
        if os.path.isfile(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                # Validate minimal structure
                if 'train' in data and 'validation' in data:
                    if 'steps' not in data:
                        data['steps'] = []
                    return data
            except (json.JSONDecodeError, KeyError):
                pass
        return {'steps': [], 'train': [], 'validation': []}

    def _flush(self):
        """Write the full history to disk atomically."""
        tmp_path = self.filepath + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self._history, f, indent=2)
        # Atomic rename (best-effort on Windows)
        if os.path.exists(self.filepath):
            os.replace(tmp_path, self.filepath)
        else:
            os.rename(tmp_path, self.filepath)
