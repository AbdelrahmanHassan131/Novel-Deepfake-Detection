"""
Validator.

Runs model evaluation on a validation dataloader and returns a
structured ``ValidationResult``.

The Validator is completely model-agnostic: it delegates to the model's
``set_input`` / ``forward`` / ``get_loss`` interface that every
``BaseModel`` subclass already exposes.

Improvements:
    - Explicit precision controls: fp32 (reference), amp, fp16, bf16.
    - Pure tensor-based DDP gathering with variable-length support (handling 0-sample ranks).
    - Expensive metric evaluation on rank 0 with lightweight broadcast to other ranks.
    - Bit-level preservation of unpadded EvaluationSampler and exact dataset index alignment.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch


@dataclass
class ValidationResult:
    """Structured container for validation metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    auc: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    best_threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    num_samples: int = 0
    predictions: Optional[np.ndarray] = field(default=None, repr=False)
    labels: Optional[np.ndarray] = field(default=None, repr=False)
    indices: Optional[np.ndarray] = field(default=None, repr=False)
    real_recall: Optional[float] = None
    fake_recall: Optional[float] = None
    source_metrics: Optional[dict] = field(default=None)
    source_macro_auc: Optional[float] = None
    worst_source_recall_05: Optional[float] = None
    worst_source_name: Optional[str] = None
    eval_precision: str = 'fp32'


def verify_source_readiness(train_rows, dev_rows, eligible_sources=None, monitor_metric=None,
                            allow_aggregate_sources=False, allow_source_overlap=False):
    """
    Verify scientific source provenance, class coverage, and held-out evaluation status before training.

    Enforces that:
    1. Dataset sources are known and specific (not placeholder 'diffgan' or 'unknown') unless
       explicitly allowed via allow_aggregate_sources=True.
    2. Requested eligible evaluation domains are present and have both real and fake classes in dev.
    3. Eligible evaluation sources are held out from training unless allow_source_overlap=True.
    4. If monitor_metric='source_macro_auc' without explicit eligible_sources, dev data contains
       at least 2 distinct sources with balanced classes.
    """
    train_sources = {r.get('dataset_source') or 'unknown' for r in train_rows}
    dev_sources = {r.get('dataset_source') or 'unknown' for r in dev_rows}

    aggregate_or_unknown = {'diffgan', 'aggregate', 'diffgan_aggregate', 'unknown', ''}
    has_agg_or_unknown = bool(
        train_sources.intersection(aggregate_or_unknown)
        or dev_sources.intersection(aggregate_or_unknown)
        or (not train_sources and not dev_sources)
    )

    if has_agg_or_unknown and not allow_aggregate_sources:
        raise ValueError(
            f"Scientific source readiness check failed: Dataset contains aggregate or unknown source metadata "
            f"(train sources: {sorted(str(s) for s in train_sources)}, dev sources: {sorted(str(s) for s in dev_sources)}). "
            f"Explicit source provenance is required for generalization evaluation. "
            f"Pass --allow_aggregate_sources for engineering diagnostic runs."
        )

    if eligible_sources:
        eligible_set = set(eligible_sources)
        missing_eligible = eligible_set - dev_sources
        if missing_eligible:
            raise ValueError(
                f"Scientific source readiness check failed: Requested eligible source(s) {sorted(missing_eligible)} "
                f"are absent from development data. Available dev sources: {sorted(str(s) for s in dev_sources)}."
            )
        for src in eligible_sources:
            src_labels = {r['label'] for r in dev_rows if r.get('dataset_source') == src}
            if src_labels != {0, 1}:
                raise ValueError(
                    f"Scientific source readiness check failed: Eligible source '{src}' lacks both classes in development data "
                    f"(found classes: {sorted(src_labels)}). Source-macro AUC requires both real and fake samples."
                )
        overlapping = eligible_set.intersection(train_sources)
        if overlapping and not allow_source_overlap:
            raise ValueError(
                f"Scientific source readiness check failed: Eligible evaluation source(s) {sorted(overlapping)} "
                f"overlap with training sources {sorted(str(s) for s in train_sources)}. "
                f"Evaluation domains must be held out from training. Pass --allow_source_overlap to override."
            )
    elif monitor_metric in ('source_macro_auc', 'worst_source_recall_05'):
        if len(dev_sources) < 2:
            raise ValueError(
                f"Scientific source readiness check failed: monitor_metric='source_macro_auc' requires at least 2 "
                f"distinct dataset sources in development data, or explicit --eligible_sources. "
                f"Found dev sources: {sorted(str(s) for s in dev_sources)}."
            )
        for src in dev_sources:
            src_labels = {r['label'] for r in dev_rows if r.get('dataset_source') == src}
            if src_labels != {0, 1}:
                raise ValueError(
                    f"Scientific source readiness check failed: Development source '{src}' lacks both classes "
                    f"(found classes: {sorted(src_labels)}). Specify --eligible_sources with balanced sources."
                )
        overlapping = dev_sources.intersection(train_sources)
        if overlapping and not allow_source_overlap:
            raise ValueError(
                f"Scientific source readiness check failed: Development sources {sorted(overlapping)} overlap with "
                f"training sources {sorted(str(s) for s in train_sources)}. "
                f"Pass --allow_source_overlap if non-held-out evaluation is intentional."
            )


def sync_module_buffers(module):
    """Synchronize buffers (e.g. BatchNorm running stats) from rank 0 across all ranks."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _safe_auc(labels, preds):
    """Fallback AUC via trapezoidal integration if scikit-learn is unavailable."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, preds))
    except Exception:
        pass
    # Simple rank sum fallback
    pos_mask = (labels == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_order = np.argsort(preds)
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(len(preds)) + 1
    pos_ranks = ranks[pos_mask]
    return float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _compute_validation_metrics(all_preds, all_labels, all_indices, avg_loss, dataset=None, opt=None):
    """
    Computes overall and source-level validation metrics.
    Guarantees that single-class sources report defined class recall without
    inventing AUC or balanced accuracy.
    """
    num_samples = len(all_labels) if all_labels is not None else 0
    unique_labels = np.unique(all_labels) if num_samples > 0 else np.array([])
    both_classes = len(unique_labels) == 2

    best_threshold = 0.5
    auc = None
    balanced_accuracy = None

    if both_classes and num_samples > 0:
        try:
            from sklearn.metrics import roc_curve, roc_auc_score
            fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
            eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= 1.0) & (thresholds >= 0.0))
            if len(eligible) > 0:
                best_idx = eligible[np.argmax((tpr - fpr)[eligible])]
                best_threshold = float(thresholds[best_idx])
            auc = float(roc_auc_score(all_labels, all_preds))
        except Exception:
            auc = _safe_auc(all_labels, all_preds)

    bin_preds = (all_preds >= 0.5).astype(np.int64) if num_samples > 0 else np.array([], dtype=np.int64)
    int_labels = all_labels.astype(np.int64) if num_samples > 0 else np.array([], dtype=np.int64)
    accuracy = float(np.mean(bin_preds == int_labels)) if num_samples > 0 else 0.0

    real_recall = None
    fake_recall = None
    if num_samples > 0:
        pos_mask = (int_labels == 1)
        neg_mask = (int_labels == 0)
        n_pos = int(np.sum(pos_mask))
        n_neg = int(np.sum(neg_mask))
        if n_neg > 0:
            real_recall = float(np.mean(bin_preds[neg_mask] == 0))
        if n_pos > 0:
            fake_recall = float(np.mean(bin_preds[pos_mask] == 1))
        if both_classes and real_recall is not None and fake_recall is not None:
            balanced_accuracy = float(0.5 * (real_recall + fake_recall))

    tp = int(np.sum((bin_preds == 1) & (int_labels == 1)))
    fp = int(np.sum((bin_preds == 1) & (int_labels == 0)))
    fn = int(np.sum((bin_preds == 0) & (int_labels == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Source-level evaluation
    actual_dataset = dataset
    if hasattr(actual_dataset, 'dataset') and not hasattr(actual_dataset, 'records'):
        actual_dataset = actual_dataset.dataset
    records = getattr(actual_dataset, 'records', None)

    source_metrics = None
    source_macro_auc = None
    worst_source_recall_05 = None
    worst_source_name = None

    if records is not None and num_samples > 0:
        source_to_indices = {}
        for i, idx in enumerate(all_indices):
            rec_idx = int(idx) if idx < len(records) else i
            rec = records[rec_idx]
            src = rec.get('dataset_source', rec.get('source', 'unknown'))
            source_to_indices.setdefault(src, []).append(i)

        declared_eligible = getattr(opt, 'eligible_sources', None)
        if declared_eligible is not None:
            missing_eligible = set(declared_eligible) - set(source_to_indices.keys())
            if missing_eligible:
                raise ValueError(
                    f"Scientific evaluation failed: declared eligible source(s) {sorted(missing_eligible)} "
                    f"are absent from validation data."
                )
        source_metrics = {}
        eligible_macro_aucs = []
        source_class_recalls = []

        for src, idx_list in source_to_indices.items():
            s_idx = np.array(idx_list, dtype=np.int64)
            s_preds = all_preds[s_idx]
            s_labels = all_labels[s_idx]
            s_n = len(s_preds)

            s_real_mask = (s_labels == 0)
            s_fake_mask = (s_labels == 1)
            n_real = int(np.sum(s_real_mask))
            n_fake = int(np.sum(s_fake_mask))

            s_bin_preds = (s_preds >= 0.5).astype(np.int64)
            s_real_recall = float(np.mean(s_bin_preds[s_real_mask] == 0)) if n_real > 0 else None
            s_fake_recall = float(np.mean(s_bin_preds[s_fake_mask] == 1)) if n_fake > 0 else None

            has_both = (n_real > 0 and n_fake > 0)
            if has_both:
                s_auc = _safe_auc(s_labels, s_preds)
                s_ba = float(0.5 * (s_real_recall + s_fake_recall))
                defined_recall = s_ba
            else:
                if declared_eligible is not None and src in declared_eligible:
                    raise ValueError(
                        f"Scientific evaluation failed: declared eligible source '{src}' lacks both classes in validation data "
                        f"(real={n_real}, fake={n_fake}). Cannot compute source AUC."
                    )
                s_auc = None
                s_ba = None
                defined_recall = s_real_recall if n_real > 0 else s_fake_recall

            if s_real_recall is not None:
                source_class_recalls.append((f"{src}_real", s_real_recall))
            if s_fake_recall is not None:
                source_class_recalls.append((f"{src}_fake", s_fake_recall))

            is_eligible = (src in declared_eligible) if declared_eligible is not None else True
            if has_both and is_eligible and s_auc is not None:
                eligible_macro_aucs.append((src, s_auc))

            if s_n > 0:
                q25 = float(np.percentile(s_preds, 25))
                q50 = float(np.percentile(s_preds, 50))
                q75 = float(np.percentile(s_preds, 75))
            else:
                q25 = q50 = q75 = 0.5

            source_metrics[src] = {
                'num_samples': s_n,
                'real_count': n_real,
                'fake_count': n_fake,
                'has_both_classes': has_both,
                'auc': s_auc,
                'balanced_accuracy': s_ba,
                'real_recall': s_real_recall,
                'fake_recall': s_fake_recall,
                'defined_class_recall': defined_recall,
                'score_quantiles': {'p25': q25, 'p50': q50, 'p75': q75},
            }

        source_macro_auc = float(np.mean([a for _, a in eligible_macro_aucs])) if eligible_macro_aucs else None
        if source_class_recalls:
            worst_name, worst_val = min(source_class_recalls, key=lambda x: x[1])
            worst_source_recall_05 = float(worst_val)
            worst_source_name = worst_name

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'auc': auc,
        'balanced_accuracy': balanced_accuracy,
        'best_threshold': best_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_samples': num_samples,
        'real_recall': real_recall,
        'fake_recall': fake_recall,
        'source_metrics': source_metrics,
        'source_macro_auc': source_macro_auc,
        'worst_source_recall_05': worst_source_recall_05,
        'worst_source_name': worst_source_name,
    }


class Validator:
    """
    Model-agnostic validation runner with precision controls and tensor collectives.
    """

    def __init__(self, device=None, precision=None):
        self.device = device
        self.precision = precision

    def _resolve_precision_context(self, model, precision_override=None):
        prec = precision_override or self.precision or getattr(getattr(model, 'opt', None), 'val_precision', 'fp32')
        prec = str(prec).lower()
        device = getattr(model, 'device', 'cpu')
        device_type = torch.device(device).type if isinstance(device, (str, torch.device)) else 'cpu'
        self.effective_precision = 'fp32'
        if prec not in ('fp32', 'amp', 'fp16', 'bf16'):
            raise ValueError(f'Unsupported validation precision: {prec}')

        if prec in ('amp', 'fp16') and device_type == 'cuda' and torch.cuda.is_available():
            self.effective_precision = 'fp16'
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                return torch.amp.autocast('cuda', dtype=torch.float16)
            return torch.cuda.amp.autocast(dtype=torch.float16)
        elif prec == 'bf16' and device_type == 'cuda' and torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            self.effective_precision = 'bf16'
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                return torch.amp.autocast('cuda', dtype=torch.bfloat16)
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        return nullcontext()

    @torch.no_grad()
    def validate(self, model, dataloader, precision=None):
        """
        Run validation over dataloader with unpadded exact alignment and clean tensor gathering.
        """
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if is_dist else 0
        world_size = dist.get_world_size() if is_dist else 1
        log_progress = (rank == 0)

        batch_count_str = f"{len(dataloader):,}" if hasattr(dataloader, '__len__') else 'unknown'
        if log_progress:
            print(f'[Validation] Starting {batch_count_str} local batches; synchronizing model buffers...', flush=True)

        # Synchronize buffers and unwrap DDP if needed
        ddp_wrapped = False
        orig_model_module = None
        if hasattr(model, 'model') and hasattr(torch.nn.parallel, 'DistributedDataParallel'):
            if isinstance(model.model, torch.nn.parallel.DistributedDataParallel):
                ddp_wrapped = True
                orig_model_module = model.model
                sync_module_buffers(orig_model_module.module)
                model.model = orig_model_module.module

        orig_training_mode = getattr(model, 'is_train', getattr(model, 'isTrain', True))
        model.eval()

        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'indices'):
            rank_indices = list(sampler.indices)
        else:
            dataset_len = len(dataloader.dataset) if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__') else 0
            rank_indices = list(range(dataset_len))

        all_preds = []
        all_labels = []
        all_indices = []
        local_loss_sum = 0.0
        local_sample_count = 0
        cursor = 0

        autocast_ctx = self._resolve_precision_context(model, precision)

        try:
            for batch_number, batch in enumerate(dataloader, 1):
                with autocast_ctx:
                    model.set_input(batch)
                    model.forward()

                batch_size = model.label.shape[0] if hasattr(model, 'label') and model.label is not None else 0
                if batch_size > 0:
                    loss = model.get_loss()
                    local_loss_sum += float(loss.item()) * batch_size
                    local_sample_count += batch_size

                    output = model.output
                    if output.dim() > 1:
                        output = output.squeeze(1)
                    score_sign = getattr(model, 'score_sign', 1.0)
                    if not isinstance(score_sign, (int, float, torch.Tensor)):
                        score_sign = 1.0
                    probs = torch.sigmoid((output * score_sign).float())

                    all_preds.append(probs.detach().cpu().numpy())
                    all_labels.append(model.label.float().detach().cpu().numpy())
                    batch_idx = rank_indices[cursor : cursor + batch_size]
                    all_indices.extend(batch_idx)
                    cursor += batch_size

                if log_progress and (batch_number == 1 or batch_number % 100 == 0):
                    print(f'[Validation] Rank 0 progress: {batch_number:,}/{batch_count_str} batches.', flush=True)
        finally:
            if ddp_wrapped:
                model.model = orig_model_module
            if orig_training_mode and hasattr(model, 'train') and callable(model.train):
                model.train()

        local_preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([], dtype=np.float32)
        local_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([], dtype=np.float32)
        local_indices = np.array(all_indices, dtype=np.int64) if len(all_indices) > 0 else np.array([], dtype=np.int64)

        if is_dist:
            device = model.device if dist.get_backend() == 'nccl' else 'cpu'

            # 1. Gather sample counts per rank
            local_n = len(local_preds)
            local_len_t = torch.tensor([local_n], dtype=torch.long, device=device)
            gathered_lens = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
            dist.all_gather(gathered_lens, local_len_t)
            lengths = [int(l.item()) for l in gathered_lens]
            max_len = max(lengths) if lengths else 0

            # 2. Gather predictions, labels, and indices via padded tensors
            if max_len > 0:
                p_pad = torch.zeros(max_len, dtype=torch.float32, device=device)
                l_pad = torch.zeros(max_len, dtype=torch.float32, device=device)
                i_pad = torch.zeros(max_len, dtype=torch.long, device=device)

                if local_n > 0:
                    p_pad[:local_n] = torch.as_tensor(local_preds, dtype=torch.float32, device=device)
                    l_pad[:local_n] = torch.as_tensor(local_labels, dtype=torch.float32, device=device)
                    i_pad[:local_n] = torch.as_tensor(local_indices, dtype=torch.long, device=device)

                gathered_p = [torch.zeros(max_len, dtype=torch.float32, device=device) for _ in range(world_size)]
                gathered_l = [torch.zeros(max_len, dtype=torch.float32, device=device) for _ in range(world_size)]
                gathered_i = [torch.zeros(max_len, dtype=torch.long, device=device) for _ in range(world_size)]

                dist.all_gather(gathered_p, p_pad)
                dist.all_gather(gathered_l, l_pad)
                dist.all_gather(gathered_i, i_pad)

                if rank == 0:
                    valid_p = [gathered_p[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]
                    valid_l = [gathered_l[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]
                    valid_i = [gathered_i[r][:lengths[r]].cpu().numpy() for r in range(world_size) if lengths[r] > 0]

                    all_preds = np.concatenate(valid_p) if valid_p else np.array([], dtype=np.float32)
                    all_labels = np.concatenate(valid_l) if valid_l else np.array([], dtype=np.float32)
                    all_indices = np.concatenate(valid_i) if valid_i else np.array([], dtype=np.int64)

                    if len(all_indices) > 0:
                        sort_order = np.argsort(all_indices)
                        all_preds = all_preds[sort_order]
                        all_labels = all_labels[sort_order]
                        all_indices = all_indices[sort_order]
                else:
                    all_preds = None
                    all_labels = None
                    all_indices = None
            else:
                all_preds = np.array([], dtype=np.float32) if rank == 0 else None
                all_labels = np.array([], dtype=np.float32) if rank == 0 else None
                all_indices = np.array([], dtype=np.int64) if rank == 0 else None

            # 3. Sum total sample-weighted loss across ranks
            stats = torch.tensor([local_loss_sum, float(local_sample_count)], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_loss_sum = stats[0].item()
            global_sample_count = int(stats[1].item())
            avg_loss = float(global_loss_sum / global_sample_count) if global_sample_count > 0 else 0.0

            # 4. Rank 0 calculates metrics and broadcasts dictionary
            metrics_payload, metric_error = None, None
            if rank == 0:
                try:
                    metrics_payload = _compute_validation_metrics(
                        all_preds, all_labels, all_indices, avg_loss,
                        dataset=getattr(dataloader, 'dataset', None),
                        opt=getattr(model, 'opt', None),
                    )
                except Exception as exc:
                    metric_error = f'{type(exc).__name__}: {exc}'
            broadcast_container = [metrics_payload, metric_error]
            dist.broadcast_object_list(broadcast_container, src=0)
            if broadcast_container[1]:
                raise ValueError(f'Validation metric calculation failed: {broadcast_container[1]}')
            res_dict = broadcast_container[0]

            return ValidationResult(
                eval_precision=self.effective_precision,
                loss=res_dict['loss'],
                accuracy=res_dict['accuracy'],
                auc=res_dict['auc'],
                balanced_accuracy=res_dict['balanced_accuracy'],
                best_threshold=res_dict['best_threshold'],
                precision=res_dict['precision'],
                recall=res_dict['recall'],
                f1=res_dict['f1'],
                num_samples=res_dict['num_samples'],
                predictions=all_preds,
                labels=all_labels,
                indices=all_indices,
                real_recall=res_dict['real_recall'],
                fake_recall=res_dict['fake_recall'],
                source_metrics=res_dict['source_metrics'],
                source_macro_auc=res_dict['source_macro_auc'],
                worst_source_recall_05=res_dict['worst_source_recall_05'],
                worst_source_name=res_dict['worst_source_name'],
            )

        else:
            all_preds = local_preds
            all_labels = local_labels
            all_indices = local_indices
            num_samples = len(all_labels)
            avg_loss = float(local_loss_sum / local_sample_count) if local_sample_count > 0 else 0.0

            res_dict = _compute_validation_metrics(
                all_preds, all_labels, all_indices, avg_loss,
                dataset=getattr(dataloader, 'dataset', None),
                opt=getattr(model, 'opt', None),
            )

            return ValidationResult(
                eval_precision=self.effective_precision,
                loss=res_dict['loss'],
                accuracy=res_dict['accuracy'],
                auc=res_dict['auc'],
                balanced_accuracy=res_dict['balanced_accuracy'],
                best_threshold=res_dict['best_threshold'],
                precision=res_dict['precision'],
                recall=res_dict['recall'],
                f1=res_dict['f1'],
                num_samples=res_dict['num_samples'],
                predictions=all_preds,
                labels=all_labels,
                indices=all_indices,
                real_recall=res_dict['real_recall'],
                fake_recall=res_dict['fake_recall'],
                source_metrics=res_dict['source_metrics'],
                source_macro_auc=res_dict['source_macro_auc'],
                worst_source_recall_05=res_dict['worst_source_recall_05'],
                worst_source_name=res_dict['worst_source_name'],
            )
