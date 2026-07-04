"""
ReportGenerator — produces a structured text summary of an experiment.

Reads saved experiment files via :class:`HistoryLoader` and writes
a Markdown report to the experiment's ``reports/`` directory.

This is a **post-training** utility.  It is never called during
training.

Usage::

    from experiment.utils import ReportGenerator

    report = ReportGenerator('experiments/wang2020_progan_20260703_120000')
    report.generate()
"""

import os
from datetime import datetime

from .history_loader import HistoryLoader


class ReportGenerator:
    """
    Generates a Markdown summary report for a completed experiment.

    Args:
        experiment_dir (str): Path to the experiment directory.
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.loader = HistoryLoader(experiment_dir)

    def generate(self, filename='summary_report.md'):
        """
        Generate and save the report.

        Args:
            filename (str): Report filename.  Saved inside the
                experiment's ``reports/`` directory.

        Returns:
            str: Absolute path to the generated report.
        """
        report_dir = os.path.join(self.experiment_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, filename)

        content = self._build_report()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return report_path

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_report(self):
        """Assemble the full Markdown report string."""
        sections = [
            self._header(),
            self._options_section(),
            self._training_summary(),
            self._validation_summary(),
            self._best_metrics(),
        ]
        return '\n\n'.join(s for s in sections if s)

    def _header(self):
        name = os.path.basename(self.experiment_dir)
        return (
            f'# Experiment Report: {name}\n\n'
            f'Generated: {datetime.now().isoformat()}'
        )

    def _options_section(self):
        opt = self.loader.load_opt()
        if not opt:
            return None

        lines = ['## Experiment Options\n']
        lines.append('| Key | Value |')
        lines.append('|-----|-------|')
        for k in sorted(opt.keys()):
            lines.append(f'| {k} | {opt[k]} |')
        return '\n'.join(lines)

    def _training_summary(self):
        try:
            history = self.loader.load_history()
        except FileNotFoundError:
            return None

        train = history.get('train', [])
        if not train:
            return None

        num_epochs = len(train)
        losses = [r['train_loss'] for r in train if 'train_loss' in r]

        lines = ['## Training Summary\n']
        lines.append(f'- **Total epochs**: {num_epochs}')
        if losses:
            lines.append(f'- **Initial loss**: {losses[0]:.6f}')
            lines.append(f'- **Final loss**: {losses[-1]:.6f}')
            lines.append(f'- **Min loss**: {min(losses):.6f}')

        total_time = sum(r.get('elapsed_seconds', 0) for r in train)
        if total_time > 0:
            lines.append(f'- **Total training time**: {total_time:.1f}s')

        return '\n'.join(lines)

    def _validation_summary(self):
        try:
            history = self.loader.load_history()
        except FileNotFoundError:
            return None

        val = history.get('validation', [])
        if not val:
            return None

        lines = ['## Validation History\n']
        lines.append('| Epoch | Loss | Acc | Prec | Rec | F1 | AUC |')
        lines.append('|-------|------|-----|------|-----|----|----|')
        for r in val:
            lines.append(
                f"| {r.get('epoch', '?')} "
                f"| {r.get('val_loss', 0):.4f} "
                f"| {r.get('accuracy', 0):.4f} "
                f"| {r.get('precision', 0):.4f} "
                f"| {r.get('recall', 0):.4f} "
                f"| {r.get('f1', 0):.4f} "
                f"| {r.get('roc_auc', 0):.4f} |"
            )
        return '\n'.join(lines)

    def _best_metrics(self):
        try:
            history = self.loader.load_history()
        except FileNotFoundError:
            return None

        val = history.get('validation', [])
        if not val:
            return None

        best_acc = max(val, key=lambda r: r.get('accuracy', 0))
        best_f1 = max(val, key=lambda r: r.get('f1', 0))
        best_auc = max(val, key=lambda r: r.get('roc_auc', 0))

        lines = ['## Best Metrics\n']
        lines.append(
            f"- **Best Accuracy**: {best_acc.get('accuracy', 0):.4f} "
            f"(epoch {best_acc.get('epoch', '?')})"
        )
        lines.append(
            f"- **Best F1**: {best_f1.get('f1', 0):.4f} "
            f"(epoch {best_f1.get('epoch', '?')})"
        )
        lines.append(
            f"- **Best ROC AUC**: {best_auc.get('roc_auc', 0):.4f} "
            f"(epoch {best_auc.get('epoch', '?')})"
        )
        return '\n'.join(lines)
