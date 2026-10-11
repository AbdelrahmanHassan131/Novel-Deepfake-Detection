# Model Evaluation Report
**Generated:** 2026-10-10 04:13:54

## 1. Checkpoint Metadata
| Field | Value |
| :--- | :--- |
| `arch` | Wang2020_128 |
| `epoch` | 5 |
| `best_metric` | 0.9634619999999999 |
| `global_step` | 1565 |
| `model_name` | Wang2020_128 |
| `checkpoint_path` | F:\Discovery AI\arm b expirement RGB model\best.pth |
| `checkpoint_size_mb` | 207.17946910858154 |
| `checkpoint_sha256` | 79256d43c65a2fcb95edc68503e696c3f6d37d4c6621a521bac62ae552d6b0ac |
| `protocol` | {'format_version': 2, 'label_mapping': {'real': 0, 'fake': 1}, 'options': {'dataroot': '/content/dataset/Preapred Dataset/train', 'val_root': None, 'manifest': '/content/Novel_Deepfake_Detection/output/controlled_mixture/manifests_v2/arm_b/selected_manifest.csv', 'manifest_split': 'train', 'val_manifest': '/content/Novel_Deepfake_Detection/output/controlled_mixture/manifests_v2/arm_b/selected_manifest.csv', 'val_manifest_split': 'dev', 'audit_hashes': False, 'allow_folder_training': False, 'noise_prob': 0.0, 'noise_std': [0.0, 3.0], 'downscale_prob': 0.0, 'downscale_range': [0.5, 1.0], 'cropSize': 224, 'loadSize': 256, 'batch_size': 32, 'serial_batches': False, 'no_flip': False, 'no_crop': False, 'no_resize': False, 'class_bal': False, 'mode': 'binary', 'classes': ['fake', 'real'], 'resize_or_crop': 'scale_and_crop', 'compute_wavelets': False, 'train_split': 'train', 'val_split': 'val', 'val_batch_size': 32, 'val_num_workers': None, 'crop_policy': 'scale_and_crop', 'blur_prob': 0.5, 'blur_sig': [0.0, 3.0], 'jpg_prob': 0.5, 'jpg_method': ['cv2'], 'jpg_qual': [50, 60, 70, 80, 90, 95], 'rz_interp': ['bilinear'], 'data_aug': False, 'aug_recipe': 'rgb_v1', 'wavelet_backend': 'cpu', 'wavelet_type': 'haar', 'wavelet_level': 3, 'wavelet_mode': 'reflect', 'use_log_packets': True, 'wavelet_log_mode': 'signed_log1p', 'precomputed_dir': None, 'arch': 'Wang2020_128', 'pretrained': True, 'num_classes': 1, 'init_type': 'normal', 'init_gain': 0.02, 'embed_dim': 128, 'num_heads': 4, 'dropout': 0.1, 'fusion_type': 'token_attention', 'freeze_base_models': True, 'rgb_model_path': None, 'wavelet_model_path': None, 'xception_model_path': None, 'convnext_model_path': None, 'backbone_weights': None, 'rgb_head_type': '128d', 'rgb_dropout': 0.5, 'fine_tune_policy': 'layer4_and_head', 'backbone_lr_mult': 0.1, 'bn_policy': 'frozen', 'decay_bias_norm': False, 'niter': 5, 'niter_decay': 0, 'lr': 0.0001, 'optim': 'adam', 'beta1': 0.9, 'weight_decay': 0.0, 'momentum': 0.0, 'lr_policy': 'cosine', 'lr_decay_iters': 10, 'lr_gamma': 0.1, 'lr_patience': 5, 'earlystop_epoch': 5, 'early_stopping': False, 'early_stopping_patience': 5, 'early_stopping_min_delta': 0.0, 'early_stopping_min_epochs': 0, 'eligible_sources': None, 'allow_aggregate_sources': True, 'allow_source_overlap': True, 'use_amp': True, 'amp_dtype': 'fp16', 'val_precision': 'fp32', 'isTrain': True, 'continue_train': False, 'new_optim': False, 'epoch_count': 1, 'last_epoch': -1, 'grad_accum_steps': 2, 'monitor_metric': 'auc', 'resume_checkpoint': None, 'additional_epochs': None, 'dist_backend': None, 'dist_url': 'env://', 'find_unused_parameters': False, 'name': '', 'run_id': 'seed42_20261009_222408_6a914779', 'checkpoints_dir': '/content/experiments/arm_b_seed42_20261009_222408_6a914779/checkpoints', 'epoch': 'latest', 'suffix': '', 'log_freq': 50, 'loss_freq': 400, 'val_epoch_freq': 1, 'save_epoch_freq': 1, 'save_latest_freq': 2000, 'gpu_ids': [0], 'num_threads': 4, 'num_workers': 4, 'seed': 42, 'deterministic': False, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True, 'channels_last': False}, 'manifests': {'manifest': {'path': '/content/Novel_Deepfake_Detection/output/controlled_mixture/manifests_v2/arm_b/selected_manifest.csv', 'sha256': '1b316052b2fcae5d713b344dbd345afad77b42e45869aa31c21964a23f94765d'}, 'val_manifest': {'path': '/content/Novel_Deepfake_Detection/output/controlled_mixture/manifests_v2/arm_b/selected_manifest.csv', 'sha256': '1b316052b2fcae5d713b344dbd345afad77b42e45869aa31c21964a23f94765d'}}, 'experts': {}, 'head_class': 'ResNet', 'head_params': 23770433, 'head_trainable_params': 15227137} |
| `decision_threshold_source` | G:\Master's Of Science Computer Engineering\thesis deepfake detection\Thesis Revisions SP\First revesion\Novel Deepfake Detection\output\external_evaluation\mixture_pilot_20261010_015657_8ec9ae4f\arm_b\threshold.json |

## 2. Classification Summary
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 0.5023 |
| **ROC AUC** | 0.8180 |
| **PR AUC** | 0.7872 |
| **F1 Score** | 0.6677 |
| **Precision** | 0.5012 |
| **Recall** | 1.0000 |
| **Equal Error Rate (EER)** | 0.2671 |
| **False Accept Rate (FAR)** | 0.9954 |
| **False Reject Rate (FRR)** | 0.0000 |
| **Specificity** | 0.0046 |
| **Sensitivity** | 1.0000 |

### Per-Class Performance
| Metric | Class 0 (Real) | Class 1 (Fake) |
| :--- | :--- | :--- |
| Precision | 1.0000 | 0.5012 |
| Recall | 0.0046 | 1.0000 |
| F1 Score | 0.0092 | 0.6677 |
| Support | 30000 | 30000 |

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
| :--- | :--- | :--- |
| **Actual Real (0)** | 138 | 29862 |
| **Actual Fake (1)** | 0 | 30000 |
