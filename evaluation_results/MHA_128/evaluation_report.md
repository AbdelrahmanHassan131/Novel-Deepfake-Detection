# Model Evaluation Report
**Generated:** 2026-07-14 20:14:43

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | MHA_128 |
| `epoch` | 1 |
| `best_metric` | 0.9792322899817437 |
| `global_step` | 12585 |
| `model_name` | MHA_128 |
| `checkpoint_path` | F:\Thesis Revesions SP\Final Expirments\mha_128_singlegpu_20260713_083003\checkpoints\best.pth |
| `checkpoint_size_mb` | 5.0397539138793945 |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.9999 |
| **ROC AUC** | 1.0000 |
| **PR AUC** | 1.0000 |
| **F1 Score** | 0.9997 |
| **Precision** | 0.9993 |
| **Recall** | 1.0000 |
| **Equal Error Rate (EER)** | 0.0000 |
| **False Accept Rate (FAR)** | 0.0001 |
| **False Reject Rate (FRR)** | 0.0000 |
| **Specificity** | 0.9999 |
| **Sensitivity** | 1.0000 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 1.0000 | 0.9993 |
| Recall | 0.9999 | 1.0000 |
| F1 Score | 1.0000 | 0.9997 |
| Support | 29790 | 3054 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 29788 | 2 |
| **Actual Fake (1)** | 0 | 3054 |

## 3. Performance & Efficiency Profile
| Metric | Value |
| :--- | :--- |
| `avg_latency_per_image_ms` | 2.099 |
| `avg_latency_per_batch_ms` | 67.1682 |
| `throughput_images_per_sec` | 476.42 |
| `total_batches_timed` | 50 |
| `total_samples_timed` | 1600 |
| `flops` | 869504 |
| `macs` | 434752 |
| `flops_source` | torch.utils.flop_counter |
| `total_parameters` | 437761 |
| `trainable_parameters` | 437761 |
| `model_size_mb` | 1.6699256896972656 |
| `checkpoint_size_mb` | None |
| `gpu_allocated_mb` | 149.14 |
| `gpu_reserved_mb` | 794.0 |
| `gpu_peak_mb` | 467.62 |

## 4. Visualizations
- **roc_curve**: [roc_curve.png](plots\roc_curve.png)
- **precision_recall_curve**: [precision_recall_curve.png](plots\precision_recall_curve.png)
- **confusion_matrix**: [confusion_matrix.png](plots\confusion_matrix.png)
