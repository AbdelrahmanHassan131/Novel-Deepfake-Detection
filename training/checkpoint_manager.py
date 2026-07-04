"""
Checkpoint Manager.

Centralizes all checkpoint save / load / resume logic.

Responsibilities:
    - save_last()      – overwrite ``last.pth``
    - save_best()      – overwrite ``best.pth``
    - save_epoch(n)    – write ``model_epoch_N.pth``
    - resume()         – restore full training state
    - latest_checkpoint() – discover the newest checkpoint on disk

The checkpoint dict stores:
    - model_state_dict
    - optimizer_state_dict
    - scheduler_state_dict  (may be None)
    - amp_state_dict        (may be None)
    - epoch
    - best_metric
    - global_step
    - model_name

Uses the existing ``BaseModel.save_checkpoint`` / ``load_checkpoint``
internally where possible, but adds the higher-level conveniences the
Training Engine needs.

Interaction with the Training Engine:
    Created by ``BaseTrainer.__init__``.  The ``CheckpointHook``
    delegates save operations to this manager.  The ``BaseTrainer``
    calls ``resume()`` to restore full training state.

Inputs:
    - ``save_dir``: Directory path for checkpoint files.
    - ``model``: A ``BaseModel`` instance.
    - ``rank``: Process rank (only rank 0 writes to disk).

Outputs:
    - Checkpoint ``.pth`` files written to ``save_dir``.
    - ``resume()`` returns a dict with restored training counters.
"""

import os
import re
import torch


class CheckpointManager:
    """
    Manages checkpoint persistence for the Training Engine.

    Args:
        save_dir (str): Directory where checkpoints are written.
        model: A ``BaseModel`` instance (has ``.model``, ``.optimizer``,
               ``.device``, etc.).
        rank (int): The current process rank.  Only rank 0 actually
                    writes to disk (DDP-safe).
    """

    def __init__(self, save_dir, model, rank=0):
        self.save_dir = save_dir
        self.model = model
        self.rank = rank
        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _build_state(self, epoch, best_metric, global_step, scheduler,
                     amp_state=None):
        """
        Assemble the checkpoint dictionary.

        Args:
            epoch (int): Current epoch number.
            best_metric: Best validation metric so far.
            global_step (int): Total training steps.
            scheduler: LR scheduler (or None).
            amp_state (dict or None): GradScaler state from AMP.

        Returns:
            dict — the complete checkpoint state.
        """
        # Unwrap DDP if needed
        raw_model = self.model.model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        state = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': (
                self.model.optimizer.state_dict()
                if hasattr(self.model, 'optimizer') and self.model.optimizer is not None
                else None
            ),
            'scheduler_state_dict': (
                scheduler.state_dict() if scheduler is not None else None
            ),
            'amp_state_dict': amp_state,
            'epoch': epoch,
            'best_metric': best_metric,
            'global_step': global_step,
            'model_name': self.model.name(),
        }
        return state

    def _save(self, filename, epoch, best_metric, global_step, scheduler,
              amp_state=None):
        """Write a checkpoint to ``save_dir / filename``."""
        if self.rank != 0:
            return
        filepath = os.path.join(self.save_dir, filename)
        state = self._build_state(
            epoch, best_metric, global_step, scheduler, amp_state
        )
        torch.save(state, filepath)
        print(f'[CheckpointManager] Saved {filepath}')

    def save_last(self, epoch, best_metric, global_step, scheduler=None,
                  amp_state=None):
        """Save as ``last.pth`` (overwrites every epoch)."""
        self._save('last.pth', epoch, best_metric, global_step, scheduler,
                   amp_state)

    def save_best(self, epoch, best_metric, global_step, scheduler=None,
                  amp_state=None):
        """Save as ``best.pth``."""
        self._save('best.pth', epoch, best_metric, global_step, scheduler,
                   amp_state)

    def save_epoch(self, epoch, best_metric, global_step, scheduler=None,
                   amp_state=None):
        """Save as ``model_epoch_<N>.pth``."""
        filename = f'model_epoch_{epoch}.pth'
        self._save(filename, epoch, best_metric, global_step, scheduler,
                   amp_state)

    # ------------------------------------------------------------------
    # Resume / load
    # ------------------------------------------------------------------

    def resume(self, filepath=None, scheduler=None):
        """
        Restore full training state from a checkpoint file.

        If *filepath* is ``None``, falls back to ``latest_checkpoint()``.

        Returns:
            dict with keys ``epoch``, ``best_metric``, ``global_step``,
            ``amp_state``.

        Raises:
            FileNotFoundError: If no checkpoint can be located.
        """
        if filepath is None:
            filepath = self.latest_checkpoint()
        if filepath is None or not os.path.isfile(filepath):
            raise FileNotFoundError(
                f'No checkpoint found at {filepath}'
            )

        print(f'[CheckpointManager] Resuming from {filepath}')
        checkpoint = torch.load(filepath, map_location=self.model.device)

        # --- model weights ---
        raw_model = self.model.model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module
        raw_model.load_state_dict(checkpoint['model_state_dict'])

        # --- optimizer state ---
        if (hasattr(self.model, 'optimizer')
                and self.model.optimizer is not None
                and checkpoint.get('optimizer_state_dict') is not None):
            self.model.optimizer.load_state_dict(
                checkpoint['optimizer_state_dict']
            )
            # Move optimizer tensors to the correct device
            for state in self.model.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.model.device)

        # --- scheduler state ---
        if (scheduler is not None
                and checkpoint.get('scheduler_state_dict') is not None):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # --- counters ---
        epoch = checkpoint.get('epoch', 0)
        best_metric = checkpoint.get('best_metric', None)
        global_step = checkpoint.get('global_step',
                                     checkpoint.get('total_steps', 0))

        self.model.total_steps = global_step

        return {
            'epoch': epoch,
            'best_metric': best_metric,
            'global_step': global_step,
            'amp_state': checkpoint.get('amp_state_dict', None),
        }

    def latest_checkpoint(self):
        """
        Return the path of the most recent checkpoint in ``save_dir``.

        Priority order:
            1. ``last.pth``
            2. Highest-numbered ``model_epoch_N.pth``

        Returns:
            Absolute path string, or ``None`` if nothing is found.
        """
        last_path = os.path.join(self.save_dir, 'last.pth')
        if os.path.isfile(last_path):
            return last_path

        # Fall back to epoch checkpoints
        pattern = re.compile(r'model_epoch_(\d+)\.pth$')
        best_epoch = -1
        best_file = None
        for fname in os.listdir(self.save_dir):
            m = pattern.match(fname)
            if m:
                ep = int(m.group(1))
                if ep > best_epoch:
                    best_epoch = ep
                    best_file = fname

        if best_file is not None:
            return os.path.join(self.save_dir, best_file)
        return None
