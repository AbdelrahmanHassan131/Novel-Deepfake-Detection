"""
DistributedRuntime — encapsulates all distributed training logic.

Responsible for:
    - Detecting the execution environment (CPU / single-GPU / DDP).
    - Initializing and tearing down ``torch.distributed``.
    - Assigning the correct device to each process.
    - Wrapping models in ``DistributedDataParallel``.
    - Wrapping DataLoaders with ``DistributedSampler``.
    - Providing rank-aware queries (``is_main``, ``should_log``,
      ``should_save``).

Design principle:
    The Trainer never touches ``torch.distributed`` directly.
    It asks the runtime *what to do* via simple boolean/property queries.

The runtime supports three execution modes transparently:
    1. **CPU** — ``opt.gpu_ids`` is empty.
    2. **Single GPU** — ``opt.gpu_ids`` has one element, no ``torchrun``.
    3. **Multi-GPU DDP** — launched via ``torchrun``; environment
       variables ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE`` are set
       automatically by ``torchrun``.

Usage::

    from training.runtime import DistributedRuntime

    runtime = DistributedRuntime(opt)
    model = build_model(opt)
    model = runtime.wrap_model(model)
    train_loader = runtime.wrap_loader(train_loader, is_train=True)

    # ... training loop ...

    runtime.cleanup()

Interaction with the Training Engine:
    The ``Trainer`` creates the runtime in ``__init__``, uses it to
    wrap the model and DataLoader, queries ``runtime.is_main`` before
    logging or saving, and calls ``runtime.cleanup()`` after training.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class DistributedRuntime:
    """
    Central runtime that encapsulates all distributed execution logic.

    The runtime auto-detects whether the process was launched via
    ``torchrun`` (by checking for ``RANK`` / ``LOCAL_RANK`` /
    ``WORLD_SIZE`` environment variables).  If those variables are
    absent, the runtime falls back to single-GPU or CPU mode.

    Args:
        opt: Project options namespace.  Used to read ``gpu_ids``
            and optional ``dist_backend`` / ``dist_url``.

    Attributes:
        device (torch.device): The device assigned to this process.
        rank (int): Global rank (0 in non-distributed mode).
        local_rank (int): Local rank on the current node (0 in
            non-distributed mode).
        world_size (int): Total number of processes (1 in
            non-distributed mode).
        is_distributed (bool): ``True`` when DDP is active.
        is_main (bool): ``True`` when this process is rank 0.
    """

    def __init__(self, opt):
        self._opt = opt
        self._distributed_initialized = False

        # --- detect environment ---
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1
        self.is_main = (self.rank == 0)

        # --- initialize process group ---
        if self.is_distributed:
            self._init_distributed(opt)

        # --- device assignment ---
        self.device = self._resolve_device(opt)

    # ------------------------------------------------------------------
    # Distributed initialization
    # ------------------------------------------------------------------

    def _init_distributed(self, opt):
        """
        Initialize the ``torch.distributed`` process group.

        Called automatically when ``WORLD_SIZE > 1``.  Uses the NCCL
        backend for CUDA, GLOO for CPU.

        Args:
            opt: Options namespace.  Optional attributes:
                - ``dist_backend`` (str): Override backend (default
                  auto-detected).
                - ``dist_url`` (str): Override init URL (default
                  ``'env://'``).
        """
        backend = getattr(opt, 'dist_backend', None)
        if backend is None:
            backend = 'nccl' if (torch.cuda.is_available() and os.name != 'nt') else 'gloo'

        init_url = getattr(opt, 'dist_url', 'env://')

        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_url,
                world_size=self.world_size,
                rank=self.rank,
            )
            self._distributed_initialized = True

            if self.is_main:
                print(
                    f'[DistributedRuntime] Initialized: '
                    f'backend={backend}, '
                    f'world_size={self.world_size}, '
                    f'rank={self.rank}, '
                    f'local_rank={self.local_rank}'
                )

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def _resolve_device(self, opt):
        """
        Determine the correct device for this process.

        Priority:
            1. DDP mode → ``cuda:{local_rank}``
            2. Single-GPU → ``cuda:{gpu_ids[0]}``
            3. CPU → ``cpu``

        Args:
            opt: Options namespace with ``gpu_ids`` attribute.

        Returns:
            torch.device for this process.
        """
        gpu_ids = getattr(opt, 'gpu_ids', [])

        if self.is_distributed and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            return torch.device(f'cuda:{self.local_rank}')

        if gpu_ids and torch.cuda.is_available():
            return torch.device(f'cuda:{gpu_ids[0]}')

        return torch.device('cpu')

    # ------------------------------------------------------------------
    # Model wrapping
    # ------------------------------------------------------------------

    def wrap_model(self, model):
        """
        Wrap a model's inner ``nn.Module`` in ``DistributedDataParallel``.

        This method wraps ``model.model`` (the raw ``nn.Module``)
        in DDP *in-place*, without disturbing the ``BaseModel``
        trainer wrapper.  The model is also moved to the correct
        device before wrapping.

        In non-distributed mode this is a no-op (the model is moved
        to the device but not wrapped in DDP).

        Args:
            model: A ``BaseModel`` instance whose ``.model`` attribute
                is the raw ``nn.Module``.

        Returns:
            The same ``model`` reference (mutated in-place).

        Interaction:
            Called by the Trainer during ``__init__``, after the model
            is constructed but before the training loop starts.
        """
        # Update the model's device reference
        model.device = self.device

        # Move the inner nn.Module to the correct device
        model.model.to(self.device)

        if self.is_distributed:
            model.model = DDP(
                model.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=getattr(
                    self._opt, 'find_unused_parameters', False
                ),
            )
            if self.is_main:
                print(
                    f'[DistributedRuntime] Model wrapped in DDP '
                    f'(device_ids=[{self.local_rank}])'
                )

        return model

    # ------------------------------------------------------------------
    # DataLoader wrapping
    # ------------------------------------------------------------------

    def wrap_loader(self, loader, is_train=True):
        """
        Wrap a DataLoader with a ``DistributedSampler`` for DDP.

        When DDP is **disabled**, the original loader is returned
        unchanged, preserving existing shuffle and sampler behavior.

        When DDP is **enabled**:
            - A ``DistributedSampler`` replaces any existing sampler.
            - ``shuffle`` on the DataLoader is set to ``False`` (the
              sampler handles shuffling).
            - The caller must call ``sampler.set_epoch(epoch)`` before
              each epoch (handled automatically by the Trainer).

        Args:
            loader: A ``torch.utils.data.DataLoader``.
            is_train (bool): Whether this is a training loader.
                Training loaders are shuffled by the sampler; validation
                loaders are not.

        Returns:
            A new ``DataLoader`` with the distributed sampler attached,
            or the original loader if DDP is not active.

        Interaction:
            Called by the Trainer during ``__init__`` for both the
            training and validation DataLoaders.
        """
        if not self.is_distributed:
            return loader

        dataset = loader.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
        )

        # Reconstruct the loader with the distributed sampler.
        # We preserve all original loader settings except sampler/shuffle.
        new_loader = DataLoader(
            dataset=dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=getattr(loader, 'drop_last', False),
            collate_fn=loader.collate_fn,
        )

        if self.is_main:
            print(
                f'[DistributedRuntime] DataLoader wrapped with '
                f'DistributedSampler (is_train={is_train})'
            )

        return new_loader

    # ------------------------------------------------------------------
    # Rank-safe queries
    # ------------------------------------------------------------------

    @property
    def should_log(self):
        """
        Whether this process should write log output.

        Only rank 0 writes logs to avoid duplicated output.

        Returns:
            bool
        """
        return self.is_main

    @property
    def should_save(self):
        """
        Whether this process should save checkpoints.

        Only rank 0 persists checkpoints to disk.

        Returns:
            bool
        """
        return self.is_main

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------

    def barrier(self):
        """
        Block until all processes reach this point.

        No-op in non-distributed mode.

        Interaction:
            Used by the Trainer after checkpoint saves to ensure all
            processes wait for rank 0 to finish writing before
            proceeding.
        """
        if self.is_distributed:
            dist.barrier()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """
        Destroy the distributed process group.

        Must be called after training completes.  Safe to call
        even when DDP was never initialized (no-op).

        Interaction:
            Called by the Trainer at the end of ``fit()`` or by the
            caller after training is complete.
        """
        if self._distributed_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._distributed_initialized = False
            if self.is_main:
                print('[DistributedRuntime] Process group destroyed.')

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
        mode = 'DDP' if self.is_distributed else (
            'CUDA' if self.device.type == 'cuda' else 'CPU'
        )
        return (
            f'DistributedRuntime(mode={mode}, device={self.device}, '
            f'rank={self.rank}/{self.world_size}, '
            f'is_main={self.is_main})'
        )
