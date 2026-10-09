# Model Evaluation Report
**Generated:** 2026-10-09 20:28:27

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | Wang2020_128 |
| `epoch` | 5 |
| `best_metric` | 0.9961728423245355 |
| `global_step` | 7815 |
| `model_name` | Wang2020_128 |
| `checkpoint_path` | F:\Discovery AI\Second Expirement 100K RGB model\best.pth |
| `checkpoint_size_mb` | 207.1795301437378 |
| `checkpoint_sha256` | e598e1b32871f460756118ba6d4a5f1ad3679a3b9727dff1afa04d4d0de96a28 |
| `protocol` | {'format_version': 2, 'label_mapping': {'real': 0, 'fake': 1}, 'options': {'dataroot': '/content/dataset/Preapred Dataset/train', 'val_root': None, 'manifest': '/content/rgb_session_lt7k3ki6/prepared_data/sizes/100000/selected_manifest.csv', 'manifest_split': 'train', 'val_manifest': '/content/rgb_session_lt7k3ki6/prepared_data/sizes/100000/selected_manifest.csv', 'val_manifest_split': 'dev', 'audit_hashes': False, 'allow_folder_training': False, 'noise_prob': 0.0, 'noise_std': [0.0, 3.0], 'downscale_prob': 0.0, 'downscale_range': [0.5, 1.0], 'cropSize': 224, 'loadSize': 256, 'batch_size': 32, 'serial_batches': False, 'no_flip': False, 'no_crop': False, 'no_resize': False, 'class_bal': False, 'mode': 'binary', 'classes': ['fake', 'real'], 'resize_or_crop': 'scale_and_crop', 'compute_wavelets': False, 'train_split': 'train', 'val_split': 'val', 'val_batch_size': 32, 'val_num_workers': None, 'crop_policy': 'scale_and_crop', 'blur_prob': 0.5, 'blur_sig': [0.0, 3.0], 'jpg_prob': 0.5, 'jpg_method': ['cv2'], 'jpg_qual': [50, 60, 70, 80, 90, 95], 'rz_interp': ['bilinear'], 'data_aug': False, 'aug_recipe': 'rgb_v1', 'wavelet_backend': 'cpu', 'wavelet_type': 'haar', 'wavelet_level': 3, 'wavelet_mode': 'reflect', 'use_log_packets': True, 'wavelet_log_mode': 'signed_log1p', 'precomputed_dir': None, 'arch': 'Wang2020_128', 'pretrained': True, 'num_classes': 1, 'init_type': 'normal', 'init_gain': 0.02, 'embed_dim': 128, 'num_heads': 4, 'dropout': 0.1, 'fusion_type': 'token_attention', 'freeze_base_models': True, 'rgb_model_path': None, 'wavelet_model_path': None, 'xception_model_path': None, 'convnext_model_path': None, 'backbone_weights': None, 'rgb_head_type': '128d', 'rgb_dropout': 0.5, 'fine_tune_policy': 'layer4_and_head', 'backbone_lr_mult': 0.1, 'bn_policy': 'frozen', 'decay_bias_norm': False, 'niter': 5, 'niter_decay': 0, 'lr': 0.0001, 'optim': 'adam', 'beta1': 0.9, 'weight_decay': 0.0, 'momentum': 0.0, 'lr_policy': 'cosine', 'lr_decay_iters': 10, 'lr_gamma': 0.1, 'lr_patience': 5, 'earlystop_epoch': 5, 'early_stopping': True, 'early_stopping_patience': 2, 'early_stopping_min_delta': 0.0, 'early_stopping_min_epochs': 2, 'eligible_sources': None, 'allow_aggregate_sources': True, 'allow_source_overlap': True, 'use_amp': True, 'amp_dtype': 'fp16', 'val_precision': 'fp32', 'isTrain': True, 'continue_train': False, 'new_optim': False, 'epoch_count': 1, 'last_epoch': -1, 'grad_accum_steps': 2, 'monitor_metric': 'auc', 'resume_checkpoint': None, 'additional_epochs': None, 'dist_backend': None, 'dist_url': 'env://', 'find_unused_parameters': False, 'name': '', 'run_id': 'seed42_fe805782', 'checkpoints_dir': '/content/rgb_pilots/rgb_r2_100000_seed42_fe805782/checkpoints', 'epoch': 'latest', 'suffix': '', 'log_freq': 50, 'loss_freq': 400, 'val_epoch_freq': 1, 'save_epoch_freq': 0, 'save_latest_freq': 2000, 'gpu_ids': [0], 'num_threads': 4, 'num_workers': 4, 'seed': 42, 'deterministic': False, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True, 'channels_last': False}, 'manifests': {'manifest': {'path': '/content/rgb_session_lt7k3ki6/prepared_data/sizes/100000/selected_manifest.csv', 'sha256': 'b23acd537028bb6fc817eb8e5f08226a9471ff88798e1adaec9da29db12ec0d9'}, 'val_manifest': {'path': '/content/rgb_session_lt7k3ki6/prepared_data/sizes/100000/selected_manifest.csv', 'sha256': 'b23acd537028bb6fc817eb8e5f08226a9471ff88798e1adaec9da29db12ec0d9'}}, 'experts': {}, 'head_class': 'ResNet', 'head_params': 23770433, 'head_trainable_params': 15227137} |
| `decision_threshold_source` | fixed 0.5; no test calibration |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.5055 |
| **ROC AUC** | 0.7242 |
| **PR AUC** | 0.6742 |
| **F1 Score** | 0.6691 |
| **Precision** | 0.5028 |
| **Recall** | 1.0000 |
| **Equal Error Rate (EER)** | 0.3461 |
| **False Accept Rate (FAR)** | 0.9889 |
| **False Reject Rate (FRR)** | 0.0000 |
| **Specificity** | 0.0111 |
| **Sensitivity** | 1.0000 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 0.9970 | 0.5028 |
| Recall | 0.0111 | 1.0000 |
| F1 Score | 0.0220 | 0.6691 |
| Support | 30000 | 30000 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 333 | 29667 |
| **Actual Fake (1)** | 1 | 29999 |
