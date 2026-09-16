# Model Evaluation Report
**Generated:** 2026-07-14 17:49:40

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | MHA_WWXC |
| `epoch` | 1 |
| `best_metric` | None |
| `global_step` | 12585 |
| `model_name` | MHA_WWXC |
| `checkpoint_path` | F:\Thesis Revesions SP\Final Expirments\mha_wwxc_experiment_20260714_015042\checkpoints\model_epoch_1.pth |
| `checkpoint_size_mb` | 11.117037773132324 |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.9998 |
| **ROC AUC** | 1.0000 |
| **PR AUC** | 1.0000 |
| **F1 Score** | 0.9990 |
| **Precision** | 1.0000 |
| **Recall** | 0.9980 |
| **Equal Error Rate (EER)** | 0.0000 |
| **False Accept Rate (FAR)** | 0.0000 |
| **False Reject Rate (FRR)** | 0.0020 |
| **Specificity** | 1.0000 |
| **Sensitivity** | 0.9980 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 0.9998 | 1.0000 |
| Recall | 1.0000 | 0.9980 |
| F1 Score | 0.9999 | 0.9990 |
| Support | 29790 | 3054 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 29790 | 0 |
| **Actual Fake (1)** | 6 | 3048 |

## 3. Performance & Efficiency Profile
| Metric | Value |
| :--- | :--- |
| `avg_latency_per_image_ms` | 9.5145 |
| `avg_latency_per_batch_ms` | 304.4637 |
| `throughput_images_per_sec` | 105.1 |
| `total_batches_timed` | 50 |
| `total_samples_timed` | 1600 |
| `flops` | 1919104 |
| `macs` | 959552 |
| `flops_source` | torch.utils.flop_counter |
| `total_parameters` | 965633 |
| `trainable_parameters` | 965633 |
| `model_size_mb` | 3.6835975646972656 |
| `checkpoint_size_mb` | None |
| `gpu_allocated_mb` | 575.21 |
| `gpu_reserved_mb` | 2494.0 |
| `gpu_peak_mb` | 1225.99 |

## 4. Visualizations
- **roc_curve**: [roc_curve.png](plots\roc_curve.png)
- **precision_recall_curve**: [precision_recall_curve.png](plots\precision_recall_curve.png)
- **confusion_matrix**: [confusion_matrix.png](plots\confusion_matrix.png)
