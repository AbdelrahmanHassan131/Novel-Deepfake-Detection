"""
Multi-Model Evaluation and Comparison Engine.

Evaluates multiple deepfake detection models on a validation dataset,
generates individual evaluation results, and produces comparative
benchmarks, overlaid ROC/PR graphs, metric bar charts, and JSON summary files.

Usage::

    from evaluation.comparison import ModelComparisonEvaluator

    models_to_eval = {
        'Wang2020Raw': ('checkpoints/wang_raw.pth', 'Wang2020Raw'),
        'Wang2020_128': ('checkpoints/wang_128.pth', 'Wang2020_128'),
        'WolterWavelet2021Raw': ('checkpoints/wolter_raw.pth', 'WolterWavelet2021Raw'),
    }
    comparator = ModelComparisonEvaluator(
        models_dict=models_to_eval,
        dataroot='dataset/val',
        output_dir='evaluation_comparison_output'
    )
    results = comparator.run()
"""

import os
import json
from datetime import datetime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .evaluator import Evaluator


class ModelComparisonEvaluator:
    """
    Orchestrator for comparing multiple deepfake detection models.

    Args:
        models_dict (dict): Mapping of model display names to tuples of
            ``(checkpoint_path, architecture_name)``. E.g.
            ``{'Wang2020Raw': ('path.pth', 'Wang2020Raw')}``.
        dataroot (str): Path to validation dataset root directory.
        output_dir (str): Root directory for output reports and plots.
        batch_size (int): Batch size for inference.
        device (str): Target evaluation device.
        collect_embeddings (bool): Whether to collect feature embeddings for t-SNE.
        run_profiling (bool): Whether to profile latency/FLOPs.
        generate_plots (bool): Whether to generate individual and comparison plots.
    """

    def __init__(self, models_dict, dataroot, output_dir='evaluation_comparison_results',
                 batch_size=32, device=None, collect_embeddings=True,
                 run_profiling=True, generate_plots=True, generate_gradcam=True,
                 tsne_dataroot=None):
        self.models_dict = models_dict
        self.dataroot = dataroot
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.collect_embeddings = collect_embeddings
        self.run_profiling = run_profiling
        self.generate_plots = generate_plots
        self.generate_gradcam = generate_gradcam
        self.tsne_dataroot = tsne_dataroot

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, opt_overrides=None):
        """
        Execute evaluation across all registered models and generate comparison reports.

        Returns:
            dict: Summary containing individual results, comparison JSON path,
                  comparison Markdown report, and comparison plot paths.
        """
        print("==========================================================")
        print("          MULTI-MODEL EVALUATION & COMPARISON             ")
        print("==========================================================")
        print(f"Validation Dataset: {self.dataroot}")
        print(f"Output Directory:   {self.output_dir}")
        print(f"Models to Evaluate: {list(self.models_dict.keys())}")
        print("==========================================================")

        eval_results = {}
        inference_results = {}

        for display_name, entry in self.models_dict.items():
            if isinstance(entry, (tuple, list)):
                ckpt_path, arch = entry[0], entry[1]
            else:
                ckpt_path, arch = entry, None

            print(f"\n---> Evaluating [{display_name}] (ckp: {ckpt_path}, arch: {arch})")
            model_out_dir = os.path.join(self.output_dir, "models", display_name)

            evaluator = Evaluator(
                checkpoint_path=ckpt_path,
                dataroot=self.dataroot,
                output_dir=model_out_dir,
                arch=arch,
                batch_size=self.batch_size,
                device=self.device,
                collect_embeddings=self.collect_embeddings,
                run_profiling=self.run_profiling,
                generate_plots=self.generate_plots,
                generate_gradcam=self.generate_gradcam,
                tsne_dataroot=self.tsne_dataroot
            )
            result = evaluator.run(opt_overrides=opt_overrides)
            eval_results[display_name] = result
            inference_results[display_name] = result.inference_result

        # Generate comparison outputs
        print("\n==========================================================")
        print("Generating Multi-Model Comparative Benchmark & Reports...")
        print("==========================================================")

        comp_dir = os.path.join(self.output_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)

        json_path = self._save_comparison_json(eval_results, comp_dir)
        plot_paths = self._generate_comparison_plots(eval_results, inference_results, comp_dir)
        md_path = self._save_comparison_markdown(eval_results, plot_paths, comp_dir)

        print("\n==========================================================")
        print("          MULTI-MODEL COMPARISON COMPLETE                 ")
        print("==========================================================")
        print(f"Summary JSON:     {json_path}")
        print(f"Markdown Report:  {md_path}")
        if plot_paths:
            print("Comparison Plots Generated:")
            for k, p in plot_paths.items():
                print(f"  - {k}: {p}")
        print("==========================================================")

        return {
            'eval_results': eval_results,
            'comparison_json': json_path,
            'comparison_markdown': md_path,
            'comparison_plots': plot_paths,
        }

    def _save_comparison_json(self, eval_results, comp_dir):
        """Save clean numerical metrics to JSON."""
        summary = {
            'generated_at': datetime.now().isoformat(),
            'dataset': self.dataroot,
            'models': {}
        }

        for name, res in eval_results.items():
            metrics = res.metrics
            perf = res.performance

            summary['models'][name] = {
                'architecture': res.metadata.get('arch', name),
                'checkpoint_path': res.metadata.get('checkpoint_path', ''),
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'f1_score': float(metrics.get('f1_score', 0.0)),
                'roc_auc': float(metrics.get('roc_auc', 0.0)),
                'pr_auc': float(metrics.get('pr_auc', 0.0)),
                'eer': float(metrics.get('eer', 0.0)),
                'precision': float(metrics.get('precision', 0.0)),
                'recall': float(metrics.get('recall', 0.0)),
                'specificity': float(metrics.get('specificity', 0.0)),
                'sensitivity': float(metrics.get('sensitivity', 0.0)),
                'far': float(metrics.get('far', 0.0)),
                'frr': float(metrics.get('frr', 0.0)),
                'true_positives': int(metrics.get('true_positives', 0)),
                'false_positives': int(metrics.get('false_positives', 0)),
                'true_negatives': int(metrics.get('true_negatives', 0)),
                'false_negatives': int(metrics.get('false_negatives', 0)),
                'total_samples': int(metrics.get('total_samples', 0)),
                'performance': {
                    'flops_g': float(perf.get('flops_g', 0.0)),
                    'latency_ms_per_sample': float(perf.get('latency_ms_per_sample', 0.0)),
                    'throughput_samples_per_sec': float(perf.get('throughput_samples_per_sec', 0.0)),
                }
            }

        json_path = os.path.join(comp_dir, "comparison_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        return json_path

    def _generate_comparison_plots(self, eval_results, inference_results, comp_dir):
        """Generate combined comparison graphs across all evaluated models."""
        if not self.generate_plots:
            return {}

        plot_paths = {}

        # 1. Combined ROC Curves
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.5000)')

        for name, inf_res in inference_results.items():
            probs = inf_res.probabilities
            labels = inf_res.labels
            if len(np.unique(labels)) >= 2:
                fpr, tpr, _ = roc_curve(labels, probs)
                roc_auc_val = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_val:.4f})")

        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate (FPR)', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate (TPR)', fontsize=12)
        ax_roc.set_title('ROC Curves Comparison Across Evaluated Models', fontsize=14, fontweight='bold')
        ax_roc.legend(loc="lower right", fontsize=10)
        ax_roc.grid(True, alpha=0.3)
        roc_path = os.path.join(comp_dir, "comparison_roc_curves.png")
        fig_roc.savefig(roc_path, dpi=150, bbox_inches='tight')
        plt.close(fig_roc)
        plot_paths['roc_curves'] = roc_path

        # 2. Combined Precision-Recall Curves
        fig_pr, ax_pr = plt.subplots(figsize=(9, 7))
        for name, inf_res in inference_results.items():
            probs = inf_res.probabilities
            labels = inf_res.labels
            if len(np.unique(labels)) >= 2:
                precision, recall, _ = precision_recall_curve(labels, probs)
                pr_auc_val = auc(recall, precision)
                ax_pr.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc_val:.4f})")

        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        ax_pr.legend(loc="lower left", fontsize=10)
        ax_pr.grid(True, alpha=0.3)
        pr_path = os.path.join(comp_dir, "comparison_pr_curves.png")
        fig_pr.savefig(pr_path, dpi=150, bbox_inches='tight')
        plt.close(fig_pr)
        plot_paths['pr_curves'] = pr_path

        # 3. Grouped Bar Chart of Key Metrics (Accuracy, F1, ROC AUC, EER)
        models = list(eval_results.keys())
        n_models = len(models)
        metrics_keys = ['accuracy', 'f1_score', 'roc_auc', 'eer']
        metrics_labels = ['Accuracy', 'F1 Score', 'ROC AUC', 'EER (lower=better)']

        fig_bar, ax_bar = plt.subplots(figsize=(max(10, n_models * 2), 6))
        x = np.arange(len(metrics_keys))
        width = min(0.8 / max(1, n_models), 0.2)

        for i, name in enumerate(models):
            m_data = eval_results[name].metrics
            vals = [float(m_data.get(k, 0.0)) for k in metrics_keys]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax_bar.bar(x + offset, vals, width, label=name)
            for bar in bars:
                h = bar.get_height()
                ax_bar.annotate(f'{h:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=8, rotation=45)

        ax_bar.set_ylabel('Score', fontsize=12)
        ax_bar.set_title('Performance Metrics Comparison Across Models', fontsize=14, fontweight='bold')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metrics_labels, fontsize=11)
        ax_bar.set_ylim([0, 1.15])
        ax_bar.legend(fontsize=10)
        ax_bar.grid(axis='y', alpha=0.3)
        bar_path = os.path.join(comp_dir, "comparison_metrics_bar.png")
        fig_bar.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close(fig_bar)
        plot_paths['metrics_bar'] = bar_path

        # 4. Error Rates Bar Chart (EER, FAR, FRR)
        fig_err, ax_err = plt.subplots(figsize=(max(8, n_models * 1.8), 5))
        err_keys = ['eer', 'far', 'frr']
        err_labels = ['EER', 'FAR', 'FRR']
        x_e = np.arange(len(err_keys))

        for i, name in enumerate(models):
            m_data = eval_results[name].metrics
            vals = [float(m_data.get(k, 0.0)) for k in err_keys]
            offset = (i - n_models / 2 + 0.5) * width
            ax_err.bar(x_e + offset, vals, width, label=name)

        ax_err.set_ylabel('Error Rate', fontsize=12)
        ax_err.set_title('Error Rates Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax_err.set_xticks(x_e)
        ax_err.set_xticklabels(err_labels, fontsize=11)
        ax_err.legend(fontsize=10)
        ax_err.grid(axis='y', alpha=0.3)
        err_path = os.path.join(comp_dir, "comparison_error_rates.png")
        fig_err.savefig(err_path, dpi=150, bbox_inches='tight')
        plt.close(fig_err)
        plot_paths['error_rates'] = err_path

        return plot_paths

    def _save_comparison_markdown(self, eval_results, plot_paths, comp_dir):
        """Save comprehensive comparison Markdown table and summary."""
        md_path = os.path.join(comp_dir, "evaluation_comparison_report.md")
        lines = [
            "# Multi-Model Deepfake Detection Evaluation & Comparison Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Validation Dataset:** `{self.dataroot}`",
            "",
            "## 1. Executive Metrics Comparison Table",
            "",
            "| Model | Accuracy | F1 Score | ROC AUC | PR AUC | EER | Precision | Recall | Specificity |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        ]

        for name, res in eval_results.items():
            m = res.metrics
            acc = float(m.get('accuracy', 0.0))
            f1 = float(m.get('f1_score', 0.0))
            roc = float(m.get('roc_auc', 0.0))
            pr = float(m.get('pr_auc', 0.0))
            eer = float(m.get('eer', 0.0))
            prec = float(m.get('precision', 0.0))
            rec = float(m.get('recall', 0.0))
            spec = float(m.get('specificity', 0.0))
            lines.append(
                f"| **{name}** | {acc:.4f} | {f1:.4f} | {roc:.4f} | {pr:.4f} | {eer:.4f} | {prec:.4f} | {rec:.4f} | {spec:.4f} |"
            )

        lines.extend([
            "",
            "## 2. Performance Profiling Comparison",
            "",
            "| Model | FLOPs (G) | Latency (ms/sample) | Throughput (samples/sec) |",
            "| :--- | :--- | :--- | :--- |"
        ])

        for name, res in eval_results.items():
            perf = res.performance
            flops = float(perf.get('flops_g', 0.0))
            lat = float(perf.get('latency_ms_per_sample', 0.0))
            tput = float(perf.get('throughput_samples_per_sec', 0.0))
            lines.append(f"| **{name}** | {flops:.2f} | {lat:.2f} | {tput:.1f} |")

        if plot_paths:
            lines.extend([
                "",
                "## 3. Comparison Visualizations",
                "",
            ])
            for k, p in plot_paths.items():
                rel_p = os.path.basename(p)
                lines.append(f"- **{k}**: `comparison/{rel_p}`")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return md_path
