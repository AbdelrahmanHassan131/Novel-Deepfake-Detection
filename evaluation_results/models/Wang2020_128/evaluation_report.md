# Model Evaluation Report
**Generated:** 2026-07-14 20:19:44

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | Wang2020_128 |
| `epoch` | 2 |
| `best_metric` | None |
| `global_step` | 25170 |
| `model_name` | Wang2020_128 |
| `checkpoint_path` | F:\Thesis Revesions SP\Final Expirments\wang2020_128_singlegpu_20260712_211828\checkpoints\model_epoch_2.pth |
| `checkpoint_size_mb` | 272.47157192230225 |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.9998 |
| **ROC AUC** | 1.0000 |
| **PR AUC** | 1.0000 |
| **F1 Score** | 0.9987 |
| **Precision** | 0.9974 |
| **Recall** | 1.0000 |
| **Equal Error Rate (EER)** | 0.0002 |
| **False Accept Rate (FAR)** | 0.0003 |
| **False Reject Rate (FRR)** | 0.0000 |
| **Specificity** | 0.9997 |
| **Sensitivity** | 1.0000 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 1.0000 | 0.9974 |
| Recall | 0.9997 | 1.0000 |
| F1 Score | 0.9999 | 0.9987 |
| Support | 29790 | 3054 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 29782 | 8 |
| **Actual Fake (1)** | 0 | 3054 |

## 3. Performance & Efficiency Profile
| Metric | Value |
| :--- | :--- |
| `avg_latency_per_image_ms` | 1.7073 |
| `avg_latency_per_batch_ms` | 54.6332 |
| `throughput_images_per_sec` | 585.72 |
| `total_batches_timed` | 50 |
| `total_samples_timed` | 1600 |
| `flops` | 10677125376 |
| `macs` | 5338562688 |
| `flops_source` | torch.utils.flop_counter |
| `total_parameters` | 23770433 |
| `trainable_parameters` | 23770433 |
| `model_size_mb` | 90.67700576782227 |
| `checkpoint_size_mb` | None |
| `gpu_allocated_mb` | 506.78 |
| `gpu_reserved_mb` | 954.0 |
| `gpu_peak_mb` | 716.27 |

## 4. Visualizations
- **roc_curve**: [roc_curve.png](plots\roc_curve.png)
- **precision_recall_curve**: [precision_recall_curve.png](plots\precision_recall_curve.png)
- **confusion_matrix**: [confusion_matrix.png](plots\confusion_matrix.png)
