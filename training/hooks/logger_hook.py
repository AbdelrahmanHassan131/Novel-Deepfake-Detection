"""
Logger Hook.

Prints training and validation statistics to stdout.
Preserves the current logging style; a full logging redesign is planned
for a later step.

Events handled:
    - on_epoch_end: prints epoch-level training summary.
    - on_batch_end: prints batch-level stats every ``log_freq`` steps.
    - on_validation_end: prints validation metrics.
"""

import time


class LoggerHook:
    """
    Simple stdout logger for the Training Engine.

    Args:
        log_freq (int): Print batch-level stats every N steps.
            Set to 0 to disable per-batch logging.
        rank (int): Only rank 0 prints (DDP-safe).
    """

    def __init__(self, log_freq=50, rank=0, experiment_logger=None):
        self.log_freq = log_freq
        self.rank = rank
        self.experiment_logger = experiment_logger
        self._epoch_start_time = None

    def _should_log(self):
        return self.rank == 0

    # ---- hook interface ----

    def on_epoch_start(self, trainer):
        if not self._should_log():
            return
        self._epoch_start_time = time.time()
        lr = trainer.model.optimizer.param_groups[0]['lr']
        print(
            f'\n=== Epoch {trainer.current_epoch} '
            f'| LR: {lr:.2e} '
            f'| Global step: {trainer.global_step} ==='
        )

    def _get_world_size_and_reduce_loss(self, loss_val):
        """Average scalar loss across all DDP ranks and get world size."""
        world_size = 1
        try:
            import torch
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_world_size() > 1:
                world_size = dist.get_world_size()
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                tensor_loss = torch.tensor([loss_val], dtype=torch.float32, device=device)
                dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
                loss_val = (tensor_loss[0] / world_size).item()
        except Exception:
            pass
        return loss_val, world_size

    def on_epoch_end(self, trainer):
        avg_loss = (
            trainer.epoch_loss / trainer.epoch_batches
            if trainer.epoch_batches > 0 else 0
        )
        avg_loss, world_size = self._get_world_size_and_reduce_loss(avg_loss)
        if not self._should_log():
            return
        elapsed = time.time() - self._epoch_start_time if self._epoch_start_time else 0
        total_batches = trainer.epoch_batches * world_size
        batch_str = f"{total_batches} ({trainer.epoch_batches}/GPU)" if world_size > 1 else f"{total_batches}"
        print(
            f'--- Epoch {trainer.current_epoch} finished '
            f'| avg_loss: {avg_loss:.6f} '
            f'| batches: {batch_str} '
            f'| time: {elapsed:.1f}s ---'
        )
        if self.experiment_logger is not None:
            lr = trainer.model.optimizer.param_groups[0]['lr'] if getattr(trainer.model, 'optimizer', None) else 0.0
            # If there is no val_loader, we write CSV immediately from log_epoch
            write_csv = (trainer.val_loader is None)
            self.experiment_logger.log_epoch(
                epoch=trainer.current_epoch,
                train_loss=avg_loss,
                lr=lr,
                elapsed=elapsed,
                global_step=trainer.global_step,
                num_batches=total_batches,
                write_csv=write_csv
            )

    def on_batch_end(self, trainer):
        if self.log_freq <= 0:
            return
        if trainer.epoch_batches % self.log_freq == 0:
            loss = trainer.last_batch_loss
            loss, world_size = self._get_world_size_and_reduce_loss(loss)
            if not self._should_log():
                return
            total_batches = trainer.epoch_batches * world_size
            batch_str = f"{total_batches} ({trainer.epoch_batches}/GPU)" if world_size > 1 else f"{total_batches}"
            print(
                f'  [step {trainer.global_step}] '
                f'batch {batch_str} '
                f'| loss: {loss:.6f}'
            )
            if self.experiment_logger is not None:
                lr = trainer.model.optimizer.param_groups[0]['lr'] if getattr(trainer.model, 'optimizer', None) else 0.0
                self.experiment_logger.log_step(
                    global_step=trainer.global_step,
                    epoch=trainer.current_epoch,
                    batch=total_batches,
                    loss=loss,
                    lr=lr,
                )

    def on_batch_start(self, trainer):
        pass

    def on_validation_end(self, trainer, result):
        if not self._should_log():
            return
        print(
            f'--- Validation @ epoch {trainer.current_epoch} '
            f'| loss: {result.loss:.6f} '
            f'| acc: {result.accuracy:.4f} '
            f'| auc: {result.auc:.4f} '
            f'| f1: {getattr(result, "f1", 0.0):.4f} '
            f'| prec: {result.precision:.4f} '
            f'| rec: {result.recall:.4f} '
            f'| samples: {result.num_samples} ---'
        )
        if self.experiment_logger is not None:
            self.experiment_logger.log_validation(
                epoch=trainer.current_epoch,
                val_loss=result.loss,
                accuracy=result.accuracy,
                precision=result.precision,
                recall=result.recall,
                f1=getattr(result, 'f1', 0.0),
                roc_auc=result.auc,
                num_samples=result.num_samples
            )
