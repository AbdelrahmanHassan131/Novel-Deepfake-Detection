# Model Evaluation Report
**Generated:** 2026-07-14 20:22:09

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | WolterWavelet2021_128 |
| `epoch` | 2 |
| `best_metric` | 0.9023525071653565 |
| `global_step` | 25170 |
| `model_name` | WolterWavelet2021_128 |
| `checkpoint_path` | F:\Thesis Revesions SP\Final Expirments\wolter2021_128_singlegpu_blur_20260713_044343\checkpoints\model_epoch_2.pth |
| `checkpoint_size_mb` | 19.805140495300293 |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.9944 |
| **ROC AUC** | 0.9998 |
| **PR AUC** | 0.9983 |
| **F1 Score** | 0.9706 |
| **Precision** | 0.9487 |
| **Recall** | 0.9935 |
| **Equal Error Rate (EER)** | 0.0060 |
| **False Accept Rate (FAR)** | 0.0055 |
| **False Reject Rate (FRR)** | 0.0065 |
| **Specificity** | 0.9945 |
| **Sensitivity** | 0.9935 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 0.9993 | 0.9487 |
| Recall | 0.9945 | 0.9935 |
| F1 Score | 0.9969 | 0.9706 |
| Support | 29790 | 3054 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 29626 | 164 |
| **Actual Fake (1)** | 20 | 3034 |

## 3. Performance & Efficiency Profile
| Metric | Value |
| :--- | :--- |
| `avg_latency_per_image_ms` | 0.2582 |
| `avg_latency_per_batch_ms` | 8.2639 |
| `throughput_images_per_sec` | 3872.26 |
| `total_batches_timed` | 50 |
| `total_samples_timed` | 1600 |
| `flops` | 5435949312 |
| `macs` | 2717974656 |
| `flops_source` | torch.utils.flop_counter |
| `total_parameters` | 1727553 |
| `trainable_parameters` | 1727553 |
| `model_size_mb` | 6.590091705322266 |
| `checkpoint_size_mb` | None |
| `gpu_allocated_mb` | 292.17 |
| `gpu_reserved_mb` | 954.0 |
| `gpu_peak_mb` | 716.27 |

## 4. Visualizations
- **roc_curve**: [roc_curve.png](plots\roc_curve.png)
- **precision_recall_curve**: [precision_recall_curve.png](plots\precision_recall_curve.png)
- **confusion_matrix**: [confusion_matrix.png](plots\confusion_matrix.png)
- **gradcam**: [gradcam_explanations.png](plots\gradcam_explanations.png)
