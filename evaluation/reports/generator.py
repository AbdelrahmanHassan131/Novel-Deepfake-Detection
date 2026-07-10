"""
Evaluation Report Generator.

Exports evaluation metrics, performance profiles, checkpoint metadata,
and visualization links into clean Markdown and JSON reports.

Usage::

    from evaluation.reports import EvaluationReportGenerator

    generator = EvaluationReportGenerator(output_dir='evaluation_results')
    generator.generate(metadata, metrics, performance, plot_paths)
"""

import os
import json
from datetime import datetime


class EvaluationReportGenerator:
    """
    Generates comprehensive evaluation reports in Markdown and JSON.

    Args:
        output_dir (str): Directory where reports and artifacts are saved.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, metadata, metrics, performance, plot_paths=None,
                 report_name='evaluation_report'):
        """
        Generate both Markdown (.md) and JSON (.json) evaluation reports.

        Args:
            metadata (dict): Checkpoint and run metadata (e.g., arch, epoch).
            metrics (dict): Classification metrics dictionary.
            performance (dict): Performance profiling metrics dictionary.
            plot_paths (dict, optional): Dictionary mapping plot type to relative
                or absolute file path.
            report_name (str): Base filename for the report files.

        Returns:
            dict: Paths to generated report files ``{'markdown': path, 'json': path}``.
        """
        plot_paths = plot_paths or {}

        json_path = self._save_json(metadata, metrics, performance,
                                    plot_paths, report_name)
        md_path = self._save_markdown(metadata, metrics, performance,
                                      plot_paths, report_name)

        return {
            'markdown': md_path,
            'json': json_path,
        }

    # ------------------------------------------------------------------
    # JSON Generation
    # ------------------------------------------------------------------

    def _save_json(self, metadata, metrics, performance, plot_paths, report_name):
        json_path = os.path.join(self.output_dir, f'{report_name}.json')

        payload = {
            'generated_at': datetime.now().isoformat(),
            'metadata': self._make_serializable(metadata),
            'classification_metrics': self._make_serializable(metrics),
            'performance_metrics': self._make_serializable(performance),
            'plots': self._make_serializable(plot_paths),
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        return json_path

    # ------------------------------------------------------------------
    # Markdown Generation
    # ------------------------------------------------------------------

    def _save_markdown(self, metadata, metrics, performance, plot_paths, report_name):
        md_path = os.path.join(self.output_dir, f'{report_name}.md')

        lines = [
            "# Model Evaluation Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. Checkpoint Metadata",
            "| Field | Value |",
            "| :--- | :--- |",
        ]

        for k, v in metadata.items():
            lines.append(f"| `{k}` | {v} |")

        lines.extend([
            "",
            "## 2. Classification Summary",
            "| Metric | Value |",
            "| :--- | :--- |",
        ])

        # Key metrics table
        summary_keys = [
            ('accuracy', 'Accuracy'),
            ('roc_auc', 'ROC AUC'),
            ('pr_auc', 'PR AUC'),
            ('f1_score', 'F1 Score'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('eer', 'Equal Error Rate (EER)'),
            ('far', 'False Accept Rate (FAR)'),
            ('frr', 'False Reject Rate (FRR)'),
            ('specificity', 'Specificity'),
            ('sensitivity', 'Sensitivity'),
        ]

        for key, label in summary_keys:
            if key in metrics:
                val = metrics[key]
                formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
                lines.append(f"| **{label}** | {formatted} |")

        lines.extend([
            "",
            "### Per-Class Performance",
            "| Metric | Class 0 (Real) | Class 1 (Fake) |",
            "| :--- | :--- | :--- |",
        ])

        p0 = metrics.get('precision_class_0', 0.0)
        p1 = metrics.get('precision_class_1', 0.0)
        r0 = metrics.get('recall_class_0', 0.0)
        r1 = metrics.get('recall_class_1', 0.0)
        f0 = metrics.get('f1_class_0', 0.0)
        f1 = metrics.get('f1_class_1', 0.0)
        s0 = metrics.get('support_class_0', 0)
        s1 = metrics.get('support_class_1', 0)

        lines.append(f"| Precision | {p0:.4f} | {p1:.4f} |")
        lines.append(f"| Recall | {r0:.4f} | {r1:.4f} |")
        lines.append(f"| F1 Score | {f0:.4f} | {f1:.4f} |")
        lines.append(f"| Support | {s0} | {s1} |")

        # Confusion matrix
        cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        lines.extend([
            "",
            "### Confusion Matrix",
            "| | Predicted Real (0) | Predicted Fake (1) |",
            "| :--- | :--- | :--- |",
            f"| **Actual Real (0)** | {cm[0][0]} | {cm[0][1]} |",
            f"| **Actual Fake (1)** | {cm[1][0]} | {cm[1][1]} |",
        ])

        if performance:
            lines.extend([
                "",
                "## 3. Performance & Efficiency Profile",
                "| Metric | Value |",
                "| :--- | :--- |",
            ])
            for k, v in performance.items():
                lines.append(f"| `{k}` | {v} |")

        if plot_paths:
            lines.extend([
                "",
                "## 4. Visualizations",
            ])
            for name, path in plot_paths.items():
                rel_path = os.path.relpath(path, self.output_dir) if path else ""
                lines.append(f"- **{name}**: [{os.path.basename(path)}]({rel_path})")

        lines.append("")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return md_path

    @staticmethod
    def _make_serializable(data):
        if isinstance(data, dict):
            return {k: EvaluationReportGenerator._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [EvaluationReportGenerator._make_serializable(x) for x in data]
        elif hasattr(data, 'tolist'):
            return data.tolist()
        elif hasattr(data, 'item'):
            return data.item()
        return data
