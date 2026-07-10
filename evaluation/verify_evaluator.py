"""
Verify Evaluator Implementation.

Self-contained verification script that tests:
1. Imports of all evaluation pipeline components.
2. ClassificationMetrics calculations on synthetic data (ensuring EER, FAR, FRR, AUC work properly).
3. EvaluationReportGenerator Markdown and JSON export.
4. Visualization wrappers (ROC, PR, CM, t-SNE) on synthetic data.
5. CheckpointLoader API and metadata extraction.

Run with:
    python -m evaluation.verify_evaluator
"""

import os
import shutil
import tempfile
import numpy as np


def verify_all():
    print("==========================================================")
    print("Verifying Refactored Evaluation Pipeline Components")
    print("==========================================================")

    # 1. Verify Imports
    print("\n[1/5] Testing Imports...")
    from evaluation import (
        Evaluator, EvaluationResult,
        InferenceRunner, InferenceResult,
        CheckpointLoader
    )
    from evaluation.metrics import ClassificationMetrics, PerformanceProfiler
    from evaluation.reports import EvaluationReportGenerator
    from evaluation.visualization import (
        plot_tsne, GradCAM, GradCAMPlusPlus, generate_gradcam_figure,
        plot_evaluation_roc_curve, plot_evaluation_pr_curve,
        plot_evaluation_confusion_matrix
    )
    print("  -> All classes and functions imported successfully.")

    # Create temporary directory for test artifacts
    tmp_dir = tempfile.mkdtemp(prefix='eval_verify_')
    try:
        # 2. Verify ClassificationMetrics
        print("\n[2/5] Testing ClassificationMetrics...")
        np.random.seed(42)
        y_true = np.array([0]*50 + [1]*50)
        # Create synthetic probabilities with reasonable separation
        y_prob_0 = np.random.uniform(0.0, 0.6, size=50)
        y_prob_1 = np.random.uniform(0.4, 1.0, size=50)
        y_prob = np.concatenate([y_prob_0, y_prob_1])

        calc = ClassificationMetrics(threshold=0.5)
        metrics = calc.compute_all(y_prob, y_true)

        required_keys = [
            'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc',
            'eer', 'far', 'frr', 'specificity', 'sensitivity', 'pr_auc',
            'confusion_matrix'
        ]
        for key in required_keys:
            assert key in metrics, f"Missing required metric: {key}"
        print(f"  -> Metrics computed: Acc={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}, EER={metrics['eer']:.4f}")

        # 3. Verify Visualizations
        print("\n[3/5] Testing Visualization generation...")
        plots_dir = os.path.join(tmp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        roc_path = plot_evaluation_roc_curve(
            y_prob, y_true, save_path=os.path.join(plots_dir, 'roc.png')
        )
        assert os.path.isfile(roc_path), "ROC plot was not created."

        pr_path = plot_evaluation_pr_curve(
            y_prob, y_true, save_path=os.path.join(plots_dir, 'pr.png')
        )
        assert os.path.isfile(pr_path), "PR plot was not created."

        cm_path = plot_evaluation_confusion_matrix(
            y_prob, y_true, save_path=os.path.join(plots_dir, 'cm.png')
        )
        assert os.path.isfile(cm_path), "CM plot was not created."

        # Synthetic embeddings for t-SNE
        embeddings = np.random.randn(100, 32)
        tsne_path = plot_tsne(
            embeddings, y_true, save_path=os.path.join(plots_dir, 'tsne.png'),
            perplexity=10, max_iter=250
        )
        assert os.path.isfile(tsne_path), "t-SNE plot was not created."
        print("  -> All plots (ROC, PR, CM, t-SNE) generated and saved successfully.")

        # 4. Verify Report Generator
        print("\n[4/5] Testing EvaluationReportGenerator...")
        report_gen = EvaluationReportGenerator(output_dir=tmp_dir)
        metadata = {
            'arch': 'Wang2020Raw',
            'epoch': 10,
            'best_metric': 0.95
        }
        performance = {
            'avg_latency_per_image_ms': 5.2,
            'throughput_images_per_sec': 192.3
        }
        plot_paths = {
            'roc_curve': roc_path,
            'precision_recall_curve': pr_path,
            'confusion_matrix': cm_path,
            'tsne': tsne_path
        }
        report_paths = report_gen.generate(metadata, metrics, performance, plot_paths)

        assert os.path.isfile(report_paths['markdown']), "Markdown report not created."
        assert os.path.isfile(report_paths['json']), "JSON report not created."
        print(f"  -> Generated reports: {os.path.basename(report_paths['markdown'])}, {os.path.basename(report_paths['json'])}")

        # 5. Verify CheckpointLoader metadata handling
        print("\n[5/6] Testing CheckpointLoader name mapping...")
        assert CheckpointLoader._model_name_to_arch('Wang2020Raw') == 'Wang2020Raw'
        assert CheckpointLoader._model_name_to_arch('Wang2020_128') == 'Wang2020_128'
        assert CheckpointLoader._model_name_to_arch('WolterWaveletRaw') == 'WolterWavelet2021Raw'
        print("  -> Architecture detection mapping verified.")

        # 6. Verify PerformanceProfiler FLOPs estimation
        print("\n[6/6] Testing PerformanceProfiler FLOPs estimation...")
        import torch
        import torch.nn as nn
        class DummyModel:
            def __init__(self):
                self.model = nn.Sequential(
                    nn.Conv2d(3, 16, 3),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(16, 1)
                )
                self.device = 'cpu'
        dummy_model = DummyModel()
        profiler = PerformanceProfiler(dummy_model, device='cpu')
        flops_est = profiler._estimate_flops((1, 3, 256, 256))
        assert 'flops' in flops_est and 'macs' in flops_est and 'flops_source' in flops_est
        print(f"  -> FLOPs estimation verified (source: {flops_est['flops_source']}, FLOPs: {flops_est['flops']})")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n==========================================================")
    print("ALL EVALUATION PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")


if __name__ == '__main__':
    verify_all()
