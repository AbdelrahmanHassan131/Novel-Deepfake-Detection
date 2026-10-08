"""
Checkpoint Hook.

Saves checkpoints at configurable intervals during training.

Events handled:
    - on_epoch_end: saves ``last.pth`` every epoch, ``model_epoch_N.pth``
      every ``save_epoch_freq`` epochs, and ``best.pth`` when the
      monitored metric improves.

Interaction with the Training Engine:
    Registered by the ``Trainer`` as the last hook.  Delegates all
    file I/O to :class:`CheckpointManager`.  Reads ``trainer.best_metric``
    to decide whether to save ``best.pth``.

    Passes the AMP scaler state (via ``trainer.amp_state_dict()``)
    through to the checkpoint so that mixed precision training can
    be resumed exactly.
"""


import json
import os


class CheckpointHook:
    """
    Hook that delegates checkpoint writes to :class:`CheckpointManager`.

    Args:
        checkpoint_manager: A ``CheckpointManager`` instance.
        save_epoch_freq (int): Save a numbered checkpoint every N epochs.
            Defaults to 1 (every epoch).
        monitor_metric (str): Metric to monitor for best checkpoint ('auc', 'balanced_accuracy', 'accuracy').
            Defaults to 'auc', falling back to 'accuracy' if AUC is unavailable.
    """

    def __init__(self, checkpoint_manager=None, save_epoch_freq=1, monitor_metric='auc', **kwargs):
        self.ckpt = checkpoint_manager
        self.save_epoch_freq = save_epoch_freq
        self.monitor_metric = monitor_metric
        self._pending_best_result = None
        self._pending_best_metric_name = None
        self._pending_best_metric_val = None

    # ---- helpers ----

    @staticmethod
    def _get_amp_state(trainer):
        """Safely extract AMP state from the trainer."""
        if hasattr(trainer, 'amp_state_dict'):
            return trainer.amp_state_dict()
        return None

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        self._pending_best_result = None
        self._pending_best_metric_name = None
        self._pending_best_metric_val = None

    def on_epoch_end(self, trainer):
        from contextlib import nullcontext
        import torch.distributed as dist
        profiler = getattr(trainer, 'profiler', None)
        error = None
        try:
            with profiler.phase('checkpoint_and_prediction_export') if profiler else nullcontext():
                self._save_epoch_artifacts(trainer)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        if dist.is_available() and dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
            error = next((e for e in errors if e), None)
        if error:
            raise RuntimeError(f'Checkpoint/prediction export failed: {error}')

    def _save_epoch_artifacts(self, trainer):
        """Save best (if improved, finalized after scheduler step), last (always) and epoch checkpoint (every N epochs)."""
        amp_state = self._get_amp_state(trainer)

        if self._pending_best_result is not None:
            # Save finalized best checkpoint with post-scheduler/optimizer state of the epoch
            self.ckpt.save_best(
                epoch=trainer.current_epoch,
                best_metric=trainer.best_metric,
                global_step=trainer.global_step,
                scheduler=trainer.scheduler,
                amp_state=amp_state,
            )

            # Persist selection metadata and threshold provenance on rank 0 against the final best.pth
            if getattr(self.ckpt, 'rank', 0) == 0:
                best_ckpt_path = os.path.join(self.ckpt.save_dir, 'best.pth')
                from data.manifest import sha256
                best_sha256 = sha256(best_ckpt_path) if os.path.exists(best_ckpt_path) else 'unknown'

                result = self._pending_best_result
                metric_name = self._pending_best_metric_name
                current_metric = self._pending_best_metric_val
                meta_path = os.path.join(self.ckpt.save_dir, 'best_selection_metadata.json')
                meta = {
                    'epoch': trainer.current_epoch,
                    'global_step': trainer.global_step,
                    'monitored_metric': metric_name,
                    'metric_value': float(current_metric),
                    'best_threshold': float(getattr(result, 'best_threshold', 0.5)),
                    'accuracy': float(getattr(result, 'accuracy', 0.0)),
                    'auc': float(result.auc) if getattr(result, 'auc', None) is not None else None,
                    'balanced_accuracy': float(result.balanced_accuracy) if getattr(result, 'balanced_accuracy', None) is not None else None,
                    'source_macro_auc': float(result.source_macro_auc) if getattr(result, 'source_macro_auc', None) is not None else None,
                    'worst_source_recall_05': float(result.worst_source_recall_05) if getattr(result, 'worst_source_recall_05', None) is not None else None,
                    'worst_source_name': result.worst_source_name if getattr(result, 'worst_source_name', None) is not None else None,
                    'loss': float(getattr(result, 'loss', 0.0)),
                    'selection_criterion': f'development_{metric_name}_maximum',
                    'best_checkpoint_sha256': best_sha256,
                }
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)

                # Export dev_predictions.csv linked to the finalized best checkpoint
                if getattr(result, 'predictions', None) is not None and len(result.predictions) > 0:
                    self._export_dev_predictions(trainer, result, best_sha256)

            self._pending_best_result = None
            self._pending_best_metric_name = None
            self._pending_best_metric_val = None

        self.ckpt.save_last(
            epoch=trainer.current_epoch,
            best_metric=trainer.best_metric,
            global_step=trainer.global_step,
            scheduler=trainer.scheduler,
            amp_state=amp_state,
        )

        if self.save_epoch_freq > 0 and (
            trainer.current_epoch % self.save_epoch_freq == 0
        ):
            self.ckpt.save_epoch(
                epoch=trainer.current_epoch,
                best_metric=trainer.best_metric,
                global_step=trainer.global_step,
                scheduler=trainer.scheduler,
                amp_state=amp_state,
            )

    def on_batch_start(self, trainer):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        """Stage ``best.pth`` when the monitored metric improves; finalized at on_epoch_end."""
        metric_name = self.monitor_metric
        current_metric = None

        if metric_name == 'source_macro_auc':
            if getattr(result, 'source_macro_auc', None) is None:
                raise ValueError(
                    "Cannot monitor 'source_macro_auc': validation result does not have source-macro AUC. "
                    "Ensure validation dataset provides verified source metadata ('dataset_source') and contains sources with both classes."
                )
            current_metric = result.source_macro_auc
        elif metric_name == 'worst_source_recall_05':
            if getattr(result, 'worst_source_recall_05', None) is None:
                raise ValueError(
                    "Cannot monitor 'worst_source_recall_05': validation result does not have worst source recall. "
                    "Ensure validation dataset provides verified source metadata."
                )
            current_metric = result.worst_source_recall_05
        elif metric_name == 'auc' and getattr(result, 'auc', None) is not None:
            current_metric = result.auc
        elif metric_name == 'balanced_accuracy' and getattr(result, 'balanced_accuracy', None) is not None:
            current_metric = result.balanced_accuracy
        else:
            current_metric = getattr(result, 'accuracy', 0.0)
            metric_name = 'accuracy'

        if trainer.best_metric is None or current_metric > trainer.best_metric:
            trainer.best_metric = current_metric
            self._pending_best_result = result
            self._pending_best_metric_name = metric_name
            self._pending_best_metric_val = current_metric

    def _export_dev_predictions(self, trainer, result, best_sha256):
        """Export development predictions linked to the selected best checkpoint."""
        val_loader = getattr(trainer, 'val_loader', None)
        val_dataset = getattr(val_loader, 'dataset', None) if val_loader is not None else None
        if hasattr(val_dataset, 'dataset') and not hasattr(val_dataset, 'records'):
            val_dataset = val_dataset.dataset

        predictions = result.predictions
        labels = getattr(result, 'labels', None)
        indices = getattr(result, 'indices', None)
        num_preds = len(predictions) if predictions is not None else 0

        if num_preds == 0:
            return

        if val_dataset is None:
            raise ValueError(
                "[CheckpointHook] Exporting dev predictions requires validation dataset to provide canonical 'records' "
                "(with verified sample_id, label, and group_id). Do not export predictions with fabricated groups."
            )

        if hasattr(val_dataset, 'records'):
            canonical_records = val_dataset.records
        else:
            raise ValueError(
                "[CheckpointHook] Exporting dev predictions requires validation dataset to provide canonical 'records' "
                "(with verified sample_id, label, and group_id). Do not export predictions with fabricated groups."
            )
        if len(canonical_records) != num_preds:
            raise ValueError(
                f"[CheckpointHook] Prediction count ({num_preds}) does not match canonical records count ({len(canonical_records)})."
            )

        trainer_opt = getattr(trainer, 'opt', None)
        run_id = (getattr(trainer_opt, 'run_id', None) or getattr(trainer_opt, 'name', 'default_run')) if trainer_opt is not None else 'default_run'
        training_seed = getattr(trainer_opt, 'seed', 42) if trainer_opt is not None else 42

        aligned_records = []
        for i in range(num_preds):
            idx = int(indices[i]) if indices is not None and len(indices) == num_preds else i
            rec = dict(canonical_records[idx])
            if labels is not None and len(labels) == num_preds:
                expected_label = int(rec['label'])
                actual_label = int(labels[i])
                if expected_label != actual_label:
                    raise ValueError(
                        f"[CheckpointHook] Label mismatch for sample '{rec.get('sample_id')}' at dataset index {idx}: "
                        f"dataset record label={expected_label}, model output label={actual_label}."
                    )
            rec['run_id'] = str(run_id)
            rec['training_seed'] = int(training_seed)
            aligned_records.append(rec)

        from evaluation.generalization import export_predictions
        # Authoritative export inside run directory
        primary_file = os.path.join(self.ckpt.save_dir, 'best_dev_predictions.csv')
        alias_file = os.path.join(self.ckpt.save_dir, 'dev_predictions.csv')
        precision = getattr(result, 'eval_precision', 'unknown')
        export_predictions(primary_file, aligned_records, predictions, checkpoint_hash=best_sha256, eval_precision=precision)
        if primary_file != alias_file:
            export_predictions(alias_file, aligned_records, predictions, checkpoint_hash=best_sha256, eval_precision=precision)
        print(f"[CheckpointHook] Exported authoritative checkpoint-linked dev predictions to {primary_file}")

