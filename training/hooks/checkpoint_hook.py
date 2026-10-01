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

    def __init__(self, checkpoint_manager, save_epoch_freq=1, monitor_metric='auc'):
        self.ckpt = checkpoint_manager
        self.save_epoch_freq = save_epoch_freq
        self.monitor_metric = monitor_metric

    # ---- helpers ----

    @staticmethod
    def _get_amp_state(trainer):
        """Safely extract AMP state from the trainer."""
        if hasattr(trainer, 'amp_state_dict'):
            return trainer.amp_state_dict()
        return None

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        """Save last (always) and epoch checkpoint (every N epochs)."""
        amp_state = self._get_amp_state(trainer)

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
        """Save ``best.pth`` when the monitored metric improves."""
        metric_name = self.monitor_metric
        current_metric = None

        if metric_name == 'auc' and getattr(result, 'auc', None) is not None:
            current_metric = result.auc
        elif metric_name == 'balanced_accuracy' and getattr(result, 'balanced_accuracy', None) is not None:
            current_metric = result.balanced_accuracy
        else:
            current_metric = getattr(result, 'accuracy', 0.0)
            metric_name = 'accuracy'

        if trainer.best_metric is None or current_metric > trainer.best_metric:
            trainer.best_metric = current_metric
            amp_state = self._get_amp_state(trainer)
            self.ckpt.save_best(
                epoch=trainer.current_epoch,
                best_metric=trainer.best_metric,
                global_step=trainer.global_step,
                scheduler=trainer.scheduler,
                amp_state=amp_state,
            )

            # Persist selection metadata and threshold provenance on rank 0
            if getattr(self.ckpt, 'rank', 0) == 0:
                best_ckpt_path = os.path.join(self.ckpt.save_dir, 'best.pth')
                from data.manifest import sha256
                best_sha256 = sha256(best_ckpt_path) if os.path.exists(best_ckpt_path) else 'unknown'

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
                    'loss': float(getattr(result, 'loss', 0.0)),
                    'selection_criterion': f'development_{metric_name}_maximum',
                    'best_checkpoint_sha256': best_sha256,
                }
                try:
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2)
                except Exception as ex:
                    print(f"[CheckpointHook] Warning: Could not write selection metadata: {ex}")

                # Export dev_predictions.csv linked to the newly selected best checkpoint
                if getattr(result, 'predictions', None) is not None and len(result.predictions) > 0:
                    try:
                        self._export_dev_predictions(trainer, result, best_sha256)
                    except Exception as ex:
                        print(f"[CheckpointHook] Warning: Could not export dev predictions: {ex}")

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

        if val_dataset is None or not hasattr(val_dataset, 'records'):
            raise ValueError(
                "[CheckpointHook] Exporting dev predictions requires validation dataset to provide canonical 'records' "
                "(with verified sample_id, label, and group_id). Do not export predictions with fabricated groups."
            )

        canonical_records = val_dataset.records
        if len(canonical_records) != num_preds:
            raise ValueError(
                f"[CheckpointHook] Prediction count ({num_preds}) does not match canonical records count ({len(canonical_records)})."
            )

        run_id = getattr(trainer.opt, 'run_id', None) or getattr(trainer.opt, 'name', 'default_run')
        training_seed = getattr(trainer.opt, 'seed', 42)

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
        target_dirs = [self.ckpt.save_dir]
        parent_dir = os.path.dirname(self.ckpt.save_dir.rstrip('/\\'))
        if parent_dir and os.path.exists(parent_dir) and parent_dir != self.ckpt.save_dir:
            target_dirs.append(parent_dir)

        for d in target_dirs:
            out_file = os.path.join(d, 'dev_predictions.csv')
            export_predictions(out_file, aligned_records, predictions, checkpoint_hash=best_sha256)
            print(f"[CheckpointHook] Exported checkpoint-linked dev predictions to {out_file}")

