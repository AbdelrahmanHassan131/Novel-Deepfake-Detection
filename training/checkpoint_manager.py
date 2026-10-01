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


def _raw_model(model):
    raw = model.model
    return raw.module if hasattr(raw, 'module') else raw


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
        raw_model = _raw_model(self.model)

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
        from config.protocol import checkpoint_metadata
        state['protocol'] = checkpoint_metadata(self.model)
        state['expert_state_dicts'] = {name: getattr(self.model, name).state_dict() for name in ('rgb_model', 'wavelet_model', 'xception_model', 'convnext_model') if hasattr(self.model, name)}
        import random
        import numpy as np
        state['rng_state'] = {
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        state['rng_policy_note'] = (
            "RNG state captures generator states at checkpoint save. "
            "Note: multi-worker DataLoader stochastic sequence cannot be guaranteed bit-for-bit across process lifecycles."
        )
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
        checkpoint = torch.load(filepath, map_location=self.model.device,
                                weights_only=False)
        self._validate_protocol(checkpoint)

        # --- model weights ---
        raw_model = _raw_model(self.model)
        raw_model.load_state_dict(checkpoint['model_state_dict'])

        # Fusion experts are frozen, but their exact weights are still part of
        # the trained protocol.  Restore them when available so a resume cannot
        # silently combine a trained head with different expert features.
        for attr, state in checkpoint.get('expert_state_dicts', {}).items():
            if hasattr(self.model, attr):
                getattr(self.model, attr).load_state_dict(state, strict=True)

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

        rng_state = checkpoint.get('rng_state')
        if rng_state:
            try:
                import random
                import numpy as np
                if 'python_rng' in rng_state and rng_state['python_rng']:
                    random.setstate(rng_state['python_rng'])
                if 'numpy_rng' in rng_state and rng_state['numpy_rng']:
                    np.random.set_state(rng_state['numpy_rng'])
                if 'torch_rng' in rng_state and rng_state['torch_rng'] is not None:
                    t_rng = rng_state['torch_rng']
                    if hasattr(t_rng, 'cpu'):
                        t_rng = t_rng.cpu()
                    torch.set_rng_state(t_rng)
                if 'cuda_rng' in rng_state and rng_state['cuda_rng'] is not None and torch.cuda.is_available():
                    c_rng = rng_state['cuda_rng']
                    if isinstance(c_rng, list):
                        c_rng = [t.cpu() if hasattr(t, 'cpu') else t for t in c_rng]
                    elif hasattr(c_rng, 'cpu'):
                        c_rng = c_rng.cpu()
                    torch.cuda.set_rng_state_all(c_rng)

                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                        if self.rank != 0:
                            print(f"[CheckpointManager] Rank {self.rank}: Distributed resume aligns RNG via sampler epoch.")
                except Exception:
                    pass
            except Exception as e:
                print(f"[CheckpointManager] Warning: Could not restore full RNG state: {e}")

        self.model.total_steps = global_step

        return {
            'epoch': epoch,
            'best_metric': best_metric,
            'global_step': global_step,
            'amp_state': checkpoint.get('amp_state_dict', None),
        }

    def _validate_protocol(self, checkpoint):
        """Reject a resume whose architecture or data protocol has changed."""
        protocol = checkpoint.get('protocol')
        if not protocol:
            raise ValueError('Cannot resume a checkpoint without protocol metadata. '
                             'Use an explicitly migrated checkpoint; starting fresh is safer.')
        if protocol.get('label_mapping') != {'real': 0, 'fake': 1}:
            raise ValueError('Resume requires the canonical label mapping real=0, fake=1.')
        if checkpoint.get('model_name') and checkpoint['model_name'] != self.model.name():
            raise ValueError(f"Resume architecture mismatch: checkpoint is {checkpoint['model_name']}, "
                             f'current model is {self.model.name()}.')
        saved = protocol.get('options', {})
        if not saved:
            raise ValueError('Cannot resume: checkpoint protocol is missing saved options.')
        from config.protocol import PREPROCESSING_KEYS
        required = list(PREPROCESSING_KEYS) + ['arch', 'embed_dim', 'fusion_type']
        for key in required:
            if key in saved and hasattr(self.model.opt, key) and saved[key] != getattr(self.model.opt, key):
                raise ValueError(f'Resume protocol mismatch for {key}: '
                                 f'{saved[key]!r} != {getattr(self.model.opt, key)!r}')

        # 2. Dataset manifest identity enforcement (training and validation)
        from pathlib import Path
        import hashlib

        saved_manifests = protocol.get('manifests', {})
        train_meta = saved_manifests.get('manifest', {})
        saved_train_sha = train_meta.get('sha256') or saved.get('manifest_sha256') or protocol.get('manifest_sha256')
        if saved_train_sha:
            current_manifest = getattr(self.model.opt, 'manifest', None)
            if not current_manifest or not Path(current_manifest).is_file():
                raise FileNotFoundError(
                    f"Cannot verify manifest identity for resume: current manifest '{current_manifest}' not found. "
                    "Data identity verification is required before resuming."
                )
            current_train_sha = hashlib.sha256(Path(current_manifest).read_bytes()).hexdigest()
            if current_train_sha != saved_train_sha:
                raise ValueError(
                    f"Resume dataset manifest mismatch: checkpoint used manifest SHA256 "
                    f"'{saved_train_sha}', but current manifest has '{current_train_sha}'."
                )

        val_meta = saved_manifests.get('val_manifest', {})
        saved_val_sha = val_meta.get('sha256') or saved.get('val_manifest_sha256') or protocol.get('val_manifest_sha256')
        if saved_val_sha:
            current_val_manifest = getattr(self.model.opt, 'val_manifest', None) or getattr(self.model.opt, 'manifest', None)
            if not current_val_manifest or not Path(current_val_manifest).is_file():
                raise FileNotFoundError(
                    f"Cannot verify validation manifest identity for resume: val_manifest '{current_val_manifest}' not found."
                )
            current_val_sha = hashlib.sha256(Path(current_val_manifest).read_bytes()).hexdigest()
            if current_val_sha != saved_val_sha:
                raise ValueError(
                    f"Resume validation manifest mismatch: checkpoint used val_manifest SHA256 "
                    f"'{saved_val_sha}', but current val_manifest has '{current_val_sha}'."
                )

        # 3. Split assignment enforcement
        for split_key in ('manifest_split', 'val_manifest_split'):
            if split_key in saved and hasattr(self.model.opt, split_key):
                if saved[split_key] != getattr(self.model.opt, split_key):
                    raise ValueError(
                        f"Resume split mismatch for {split_key}: checkpoint used '{saved[split_key]}', "
                        f"current options specify '{getattr(self.model.opt, split_key)}'."
                    )

        # 4. Seed identity enforcement
        if 'seed' in saved and hasattr(self.model.opt, 'seed'):
            saved_seed = saved['seed']
            opt_seed = getattr(self.model.opt, 'seed')
            if saved_seed is not None and opt_seed is not None and saved_seed != opt_seed:
                raise ValueError(
                    f"Resume random seed mismatch: checkpoint trained with seed {saved_seed}, "
                    f"cannot resume with seed {opt_seed}. Seed must remain invariant across resume."
                )

        # 5. Monitored metric policy enforcement
        if 'monitor_metric' in saved and hasattr(self.model.opt, 'monitor_metric'):
            if saved['monitor_metric'] != getattr(self.model.opt, 'monitor_metric'):
                raise ValueError(
                    f"Resume monitor metric mismatch: checkpoint monitored '{saved['monitor_metric']}', "
                    f"current options specify '{getattr(self.model.opt, 'monitor_metric')}'."
                )

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
