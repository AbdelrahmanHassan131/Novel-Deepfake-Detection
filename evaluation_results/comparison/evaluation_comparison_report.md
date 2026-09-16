# Multi-Model Deepfake Detection Evaluation & Comparison Report
**Generated:** 2026-07-14 20:22:10  
**Validation Dataset:** `F:\to be deleted again val`

## 1. Executive Metrics Comparison Table

| Model | Accuracy | F1 Score | ROC AUC | PR AUC | EER | Precision | Recall | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wang2020_128** | 0.9998 | 0.9987 | 1.0000 | 1.0000 | 0.0002 | 0.9974 | 1.0000 | 0.9997 |
| **WolterWavelet2021_128** | 0.9944 | 0.9706 | 0.9998 | 0.9983 | 0.0060 | 0.9487 | 0.9935 | 0.9945 |

## 2. Performance Profiling Comparison

| Model | FLOPs (G) | Latency (ms/sample) | Throughput (samples/sec) |
| :--- | :--- | :--- | :--- |
| **Wang2020_128** | 0.00 | 0.00 | 0.0 |
| **WolterWavelet2021_128** | 0.00 | 0.00 | 0.0 |

## 3. Comparison Visualizations

- **roc_curves**: `comparison/comparison_roc_curves.png`
- **pr_curves**: `comparison/comparison_pr_curves.png`
- **metrics_bar**: `comparison/comparison_metrics_bar.png`
- **error_rates**: `comparison/comparison_error_rates.png`