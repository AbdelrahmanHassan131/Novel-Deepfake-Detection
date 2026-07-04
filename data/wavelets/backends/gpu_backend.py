"""
GPUWaveletBackend — wavelet packet computation on CUDA using pytorch_wavelets.

Implements the full wavelet packet decomposition tree on GPU by iteratively
applying ``DWTForward`` from the ``pytorch_wavelets`` library.  This
replicates the coefficient ordering produced by PyWavelets
``WaveletPacket2D`` so that downstream models receive tensors with
**exactly the same shape and semantic layout** as the CPU backend.

Responsibility:
    Compute wavelet packet coefficients entirely on CUDA tensors, avoiding
    CPU↔GPU transfers during the computation.

Expected inputs:
    A ``torch.Tensor`` of shape ``(3, H, W)`` (single image) or
    ``(B, 3, H, W)`` (batched).  The tensor may reside on CPU or CUDA;
    if on CPU it is moved to the target device automatically.

Expected outputs:
    ``torch.Tensor`` of shape ``(C, H', W')`` for a single image, or
    ``(B, C, H', W')`` for a batch, where ``C = 3 * 4^level``
    (e.g. 192 for level 3).  The tensor resides on the same CUDA device
    as the input (or the device specified at construction time).

Interaction with the data pipeline:
    Used by ``WaveletDataset`` and ``FusionDataset`` when
    ``opt.wavelet_backend == 'gpu'``.  The dataset converts the augmented
    PIL Image to a tensor and calls ``backend(tensor)`` inside
    ``__getitem__``.  The resulting wavelet tensor is already on GPU so
    the training loop avoids an extra ``.to(device)`` transfer.

DDP compatibility:
    Each DDP process creates its own ``GPUWaveletBackend`` instance
    targeting its assigned CUDA device (obtained from the runtime).
    No inter-GPU synchronization is needed.  The ``DWTForward`` module
    is stateless (no learnable parameters) so it does not need to be
    wrapped in ``DistributedDataParallel``.

Memory efficiency:
    - No numpy conversions during GPU execution.
    - No unnecessary CPU↔GPU round-trips.
    - Intermediate tensors are discarded as the tree is built level by
      level so peak memory is bounded.

Data flow::

    Image → CPU augmentations → ToTensor → GPU transfer
          → DWTForward (iterative tree) → log-scale (optional)
          → Wavelet Tensor (on GPU)

Algorithm
---------
PyWavelets ``WaveletPacket2D`` produces packets by recursively applying
a 2-D DWT to every sub-band at the previous level.  At each DWT step
the input is split into four sub-bands: *LL* (approximation), *LH*
(horizontal detail), *HL* (vertical detail), *HH* (diagonal detail).

We replicate this tree with ``DWTForward(J=1)``, which performs a
**single-level** DWT.  Starting from the three colour channels we
build the tree level-by-level:

    level 0:  [channel]                                     →  1  node
    level 1:  [LL, LH, HL, HH]                             →  4  nodes
    level 2:  [LL→(LL,LH,HL,HH), LH→(…), HL→(…), HH→(…)] → 16  nodes
    level 3:                                                → 64  nodes

At the target level we have ``4^level`` packets per colour channel, and
``3 * 4^level`` packets in total — matching the CPU backend exactly.

The **ordering** of packets must match PyWavelets' convention.  ``pywt``
uses the path labels ``'a'``, ``'h'``, ``'v'``, ``'d'`` which map to
LL, LH, HL, HH respectively.  When we expand each node in that order,
the alphabetical ordering of path strings (``'aaa'``, ``'aah'``,
``'aav'``, ``'aad'``, ``'aha'``, …) naturally produces the same
sequence.  Our implementation preserves this by always expanding nodes
in ``[LL, LH, HL, HH]`` order.
"""

import torch
import torch.nn as nn
import numpy as np

from .base import WaveletBackend


class GPUWaveletBackend(WaveletBackend):
    """
    GPU-based wavelet packet backend using ``pytorch_wavelets.DWTForward``.

    Computes the full wavelet packet decomposition tree on CUDA by
    iteratively applying a single-level DWT.

    Args:
        wavelet: Wavelet family (default ``'haar'``).
        level: Decomposition level (default 3).
        mode: Signal extension mode (default ``'reflect'``).
            Mapped to ``pytorch_wavelets`` padding modes automatically.
        log_scale: Apply log-scaling to coefficients (default ``True``).
        device: Target CUDA device (default ``None`` → ``'cuda'``).
            In DDP this should be set to the local device, e.g.
            ``torch.device('cuda:1')``.
    """

    # Map pywt mode names to pytorch_wavelets mode names
    _MODE_MAP = {
        'reflect': 'reflect',
        'symmetric': 'symmetric',
        'zero': 'zero',
        'constant': 'zero',
        'periodization': 'periodization',
        'periodic': 'periodization',
    }

    def __init__(self, wavelet='haar', level=3, mode='reflect',
                 log_scale=True, device=None):
        super().__init__(wavelet=wavelet, level=level, mode=mode,
                         log_scale=log_scale)

        self.device = device or torch.device('cuda')

        # Resolve padding mode
        pt_mode = self._MODE_MAP.get(mode, 'reflect')

        # Import pytorch_wavelets lazily so the rest of the project
        # never hard-depends on it unless the GPU backend is selected.
        from pytorch_wavelets import DWTForward

        # Single-level DWT — we iterate manually to build the full
        # packet tree.  Using J=1 keeps memory bounded.
        self._dwt = DWTForward(J=1, wave=wavelet, mode=pt_mode).to(
            self.device
        )
        # DWTForward has no learnable parameters; eval mode for safety.
        self._dwt.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, data):
        """
        Compute wavelet packets on GPU.

        Args:
            data: ``torch.Tensor`` of shape ``(3, H, W)`` or
                ``(B, 3, H, W)``.  Accepted on any device (moved to
                the target CUDA device automatically).

        Returns:
            torch.Tensor: Wavelet coefficients.

                - Single image input ``(3, H, W)`` →
                  ``(3 * 4^level, H', W')``.
                - Batched input ``(B, 3, H, W)`` →
                  ``(B, 3 * 4^level, H', W')``.
        """
        # --- input normalisation ---
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        if not isinstance(data, torch.Tensor):
            # PIL Image — fallback
            data = torch.from_numpy(np.array(data)).float()
            # (H, W, 3) → (3, H, W)
            if data.ndim == 3 and data.shape[2] == 3:
                data = data.permute(2, 0, 1)

        single = data.ndim == 3
        if single:
            data = data.unsqueeze(0)  # (1, 3, H, W)

        # Move to target CUDA device if necessary
        if data.device != self.device:
            data = data.to(self.device)

        # Ensure float
        if data.dtype != torch.float32:
            data = data.float()

        B, C, H, W = data.shape

        # --- build wavelet packet tree per channel ---
        all_channel_packets = []
        for c in range(C):
            # Start with single channel: (B, 1, H, W)
            channel = data[:, c:c+1, :, :]
            packets = self._build_packet_tree(channel)
            all_channel_packets.append(packets)

        # Concatenate all channels: (B, C * 4^level, H', W')
        result = torch.cat(all_channel_packets, dim=1)

        # --- optional log-scaling ---
        if self.log_scale:
            result = self._log_scale(result)

        if single:
            result = result.squeeze(0)  # back to (C', H', W')

        return result

    @property
    def name(self):
        """Return ``'gpu'``."""
        return 'gpu'

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_packet_tree(self, x):
        """
        Build the full wavelet packet tree for a single-channel batch.

        Args:
            x: ``(B, 1, H, W)`` — single colour channel.

        Returns:
            ``(B, 4^level, H', W')`` — all leaf packets.
        """
        # nodes is a list of tensors, each (B, 1, h, w)
        nodes = [x]

        for _ in range(self.level):
            next_nodes = []
            for node in nodes:
                # DWTForward with J=1 returns:
                #   yl: (B, 1, h', w')       — LL (approximation)
                #   yh: list of length 1, each (B, 1, 3, h', w')
                #       — [LH, HL, HH] stacked along dim 2
                yl, yh = self._dwt(node)

                # yh[0] has shape (B, 1, 3, h', w')
                detail = yh[0]  # (B, 1, 3, h', w')

                lh = detail[:, :, 0, :, :]  # (B, 1, h', w')
                hl = detail[:, :, 1, :, :]  # (B, 1, h', w')
                hh = detail[:, :, 2, :, :]  # (B, 1, h', w')

                # Order: a(LL), h(LH), v(HL), d(HH) — matches pywt
                next_nodes.extend([yl, lh, hl, hh])

            nodes = next_nodes

        # Each node is (B, 1, H', W') — concatenate along channel dim
        return torch.cat(nodes, dim=1)  # (B, 4^level, H', W')

    @staticmethod
    def _log_scale(tensor, epsilon=1e-10):
        """
        Apply log-scaling: ``sign(x) * log(|x| + ε)``.

        Matches the CPU backend's :func:`log_scale_packets` exactly.

        Args:
            tensor: Input tensor (any shape).
            epsilon: Small constant to avoid ``log(0)``.

        Returns:
            Log-scaled tensor (same shape and device).
        """
        return torch.sign(tensor) * torch.log(torch.abs(tensor) + epsilon)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'wavelet={self.wavelet!r}, level={self.level}, '
            f'mode={self.mode!r}, log_scale={self.log_scale}, '
            f'device={self.device})'
        )
