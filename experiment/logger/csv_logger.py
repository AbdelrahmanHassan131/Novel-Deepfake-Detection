"""
CsvLogger — incremental CSV metrics writer.

Appends one row per validation epoch.  The CSV format makes it
trivial to open in Excel / Google Sheets for quick inspection.

The header row is written on creation (or skipped if the file
already exists with data).  Each ``write_row()`` appends a single
line and flushes immediately.
"""

import csv
import os


# Column order in the CSV file.
_CSV_COLUMNS = [
    'epoch',
    'train_loss',
    'lr',
    'val_loss',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc',
    'num_samples',
]


class CsvLogger:
    """
    CSV metrics logger.

    Args:
        filepath (str): Path to the CSV file.
    """

    def __init__(self, filepath, columns=None):
        self.filepath = filepath
        self.columns = columns if columns is not None else _CSV_COLUMNS
        self._file = None
        self._writer = None
        self._open(filepath)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_row(self, record):
        """
        Append a single metrics row.

        Args:
            record (dict): Keys should match ``self.columns``.
                Missing keys are written as empty strings.
        """
        row = [record.get(col, '') for col in self.columns]
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """Flush and close the underlying file."""
        if self._file is not None and not self._file.closed:
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open(self, filepath):
        """Open the CSV file, writing a header if it is new."""
        write_header = not os.path.isfile(filepath) or os.path.getsize(filepath) == 0

        self._file = open(filepath, 'a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)

        if write_header:
            self._writer.writerow(self.columns)
            self._file.flush()
