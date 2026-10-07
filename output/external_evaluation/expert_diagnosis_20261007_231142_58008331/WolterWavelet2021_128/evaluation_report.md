# Model Evaluation Report
**Generated:** 2026-10-07 23:21:05

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | WolterWavelet2021_128 |
| `epoch` | 2 |
| `best_metric` | 0.9726977395007578 |
| `global_step` | 3082 |
| `model_name` | WolterWavelet2021_128 |
| `checkpoint_path` | F:\Discovery AI\First expirement 100K models\train_100000_seed42_f5c32ae782\wavelet128_seed42\checkpoints\best.pth |
| `checkpoint_size_mb` | 19.817160606384277 |
| `checkpoint_sha256` | 7c4f9d5a912e6ffabceb58a11aa26be7cd1c3a32298570062cdacd33771e720d |
| `protocol` | {'format_version': 2, 'label_mapping': {'real': 0, 'fake': 1}, 'options': {'dataroot': '/kaggle/input/datasets/abdelrahmanhassani/prepareddatasetdiffgan/Preapred Dataset', 'val_root': None, 'manifest': '/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv', 'manifest_split': 'train', 'val_manifest': '/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv', 'val_manifest_split': 'dev', 'audit_hashes': False, 'allow_folder_training': False, 'noise_prob': 0.0, 'noise_std': [0.0, 3.0], 'downscale_prob': 0.0, 'downscale_range': [0.5, 1.0], 'cropSize': 224, 'loadSize': 256, 'batch_size': 16, 'serial_batches': False, 'no_flip': False, 'no_crop': False, 'no_resize': False, 'class_bal': False, 'mode': 'binary', 'classes': ['fake', 'real'], 'resize_or_crop': 'scale_and_crop', 'compute_wavelets': True, 'train_split': 'train', 'val_split': 'val', 'blur_prob': 0.5, 'blur_sig': [0.0, 3.0], 'jpg_prob': 0.0, 'jpg_method': ['cv2'], 'jpg_qual': [75], 'rz_interp': ['bilinear'], 'data_aug': False, 'wavelet_backend': 'cpu', 'wavelet_type': 'haar', 'wavelet_level': 3, 'wavelet_mode': 'reflect', 'use_log_packets': True, 'wavelet_log_mode': 'signed_log1p', 'precomputed_dir': None, 'arch': 'WolterWavelet2021_128', 'pretrained': False, 'num_classes': 1, 'init_type': 'normal', 'init_gain': 0.02, 'embed_dim': 128, 'num_heads': 4, 'dropout': 0.1, 'fusion_type': 'token_attention', 'freeze_base_models': True, 'rgb_model_path': None, 'wavelet_model_path': None, 'xception_model_path': None, 'convnext_model_path': None, 'backbone_weights': None, 'niter': 2, 'niter_decay': 0, 'lr': 0.0001, 'optim': 'adam', 'beta1': 0.9, 'weight_decay': 0.0, 'momentum': 0.0, 'lr_policy': 'none', 'lr_decay_iters': 10, 'lr_gamma': 0.1, 'lr_patience': 5, 'earlystop_epoch': 5, 'use_amp': True, 'isTrain': True, 'continue_train': False, 'new_optim': False, 'epoch_count': 1, 'last_epoch': -1, 'grad_accum_steps': 2, 'monitor_metric': 'auc', 'resume_checkpoint': None, 'additional_epochs': None, 'dist_backend': None, 'dist_url': 'env://', 'find_unused_parameters': False, 'name': '', 'run_id': 'seed42', 'checkpoints_dir': '/kaggle/working/deepfake_experiments/train_100000_seed42_f5c32ae782/wavelet128_seed42/checkpoints', 'epoch': 'latest', 'suffix': '', 'log_freq': 50, 'loss_freq': 400, 'val_epoch_freq': 1, 'save_epoch_freq': 1, 'save_latest_freq': 2000, 'gpu_ids': [0, 1], 'num_threads': 2, 'num_workers': 2, 'seed': 42, 'deterministic': False, 'pin_memory': True}, 'manifests': {'manifest': {'path': '/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv', 'sha256': '407726e461ba9513e8155b655baff3822adcc5cd5ef59323d1cad7a22bd94ae2'}, 'val_manifest': {'path': '/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv', 'sha256': '407726e461ba9513e8155b655baff3822adcc5cd5ef59323d1cad7a22bd94ae2'}}, 'experts': {}, 'head_class': 'DistributedDataParallel', 'head_params': 1727553, 'head_trainable_params': 1727553} |
| `decision_threshold_source` | G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection\output\external_evaluation\expert_diagnosis_20261007_231142_58008331\Wavelet_dev_threshold.json |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.5209 |
| **ROC AUC** | 0.7507 |
| **PR AUC** | 0.7286 |
| **F1 Score** | 0.6756 |
| **Precision** | 0.5107 |
| **Recall** | 0.9978 |
| **Equal Error Rate (EER)** | 0.3151 |
| **False Accept Rate (FAR)** | 0.9560 |
| **False Reject Rate (FRR)** | 0.0022 |
| **Specificity** | 0.0440 |
| **Sensitivity** | 0.9978 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 0.9524 | 0.5107 |
| Recall | 0.0440 | 0.9978 |
| F1 Score | 0.0842 | 0.6756 |
| Support | 30000 | 30000 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 1321 | 28679 |
| **Actual Fake (1)** | 66 | 29934 |

## 4. Visualizations
- **roc_curve**: [roc_curve.png](plots\roc_curve.png)
- **precision_recall_curve**: [precision_recall_curve.png](plots\precision_recall_curve.png)
- **confusion_matrix**: [confusion_matrix.png](plots\confusion_matrix.png)
