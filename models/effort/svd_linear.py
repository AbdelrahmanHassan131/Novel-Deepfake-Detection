"""Orthogonal SVD Residual Adaptation (EFFORT-style) for Vision Foundation Models.

Decomposes pretrained linear weights W = U @ diag(S) @ V^T into:
  - Frozen primary semantic subspace W_main (top k - r singular components)
  - Trainable orthogonal residual subspace U_res @ diag(S_res) @ V_res (bottom r components)

At initialization, W_main + U_res @ diag(S_res) @ V_res == W exactly (zero perturbation).
During training, only (U_res, S_res, V_res) are updated while orthogonality regularization
keeps the learned forgery subspace orthogonal to the frozen semantic subspace.
"""
from typing import Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDResidualLinear(nn.Module):
    """Drop-in replacement for nn.Linear with frozen primary SVD subspace and trainable residual.

    Exposes ``.weight`` and ``.bias`` properties so modules that read ``out_proj.weight``
    directly (such as ``torch.nn.MultiheadAttention``) work seamlessly with full autograd.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        weight_init: Optional[torch.Tensor] = None,
        bias_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        min_dim = min(self.in_features, self.out_features)
        if min_dim <= 1:
            raise ValueError(f"SVDResidualLinear requires min(in, out) > 1, got ({in_features}, {out_features}).")
        self.rank = max(1, min(int(rank), min_dim - 1))
        self.main_rank = min_dim - self.rank

        # Frozen primary weight matrix (stored in state_dict for exact fast checkpoint reload)
        self.register_buffer("W_main", torch.zeros(self.out_features, self.in_features, dtype=torch.float32))
        # Non-persistent semantic basis buffers for orthogonality loss (avoid checkpoint bloat)
        self.register_buffer("_U_main", None, persistent=False)
        self.register_buffer("_Vh_main", None, persistent=False)

        # Trainable orthogonal residual factors (bottom `rank` singular components)
        self.U_res = nn.Parameter(torch.zeros(self.out_features, self.rank, dtype=torch.float32))
        self.S_res = nn.Parameter(torch.zeros(self.rank, dtype=torch.float32))
        self.V_res = nn.Parameter(torch.zeros(self.rank, self.in_features, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if weight_init is not None:
            self.initialize_from_weight(weight_init, bias_init)

    @torch.no_grad()
    def initialize_from_weight(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        """Compute exact SVD of ``weight`` and partition into frozen main + trainable residual."""
        w_fp32 = weight.detach().to(dtype=torch.float32)
        U, S, Vh = torch.linalg.svd(w_fp32, full_matrices=False)

        r_main = self.main_rank
        U_main = U[:, :r_main]
        S_main = S[:r_main]
        Vh_main = Vh[:r_main, :]

        U_res = U[:, r_main:r_main + self.rank]
        S_res = S[r_main:r_main + self.rank]
        Vh_res = Vh[r_main:r_main + self.rank, :]

        w_main = (U_main * S_main.unsqueeze(0)) @ Vh_main
        self.W_main.copy_(w_main)
        self._U_main = U_main.contiguous()
        self._Vh_main = Vh_main.contiguous()

        self.U_res.copy_(U_res)
        self.S_res.copy_(S_res)
        self.V_res.copy_(Vh_res)

        if self.bias is not None and bias is not None:
            self.bias.copy_(bias.detach().to(dtype=torch.float32))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 16) -> "SVDResidualLinear":
        """Construct an SVDResidualLinear from an existing nn.Linear without perturbing its output."""
        device = linear.weight.device
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=(linear.bias is not None),
            weight_init=linear.weight,
            bias_init=linear.bias,
        )
        return layer.to(device=device)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct effective weight W_main + U_res @ diag(S_res) @ V_res with full autograd."""
        delta_w = (self.U_res * self.S_res.unsqueeze(0)) @ self.V_res
        return self.W_main.to(dtype=delta_w.dtype) + delta_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def _ensure_semantic_basis(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Lazily recover semantic basis after checkpoint resume if orthogonality loss is requested."""
        if self._U_main is not None and self._Vh_main is not None:
            return self._U_main, self._Vh_main
        if self.W_main.abs().sum().item() == 0.0:
            return None, None
        with torch.no_grad():
            U, _, Vh = torch.linalg.svd(self.W_main.float(), full_matrices=False)
            self._U_main = U[:, :self.main_rank].contiguous()
            self._Vh_main = Vh[:self.main_rank, :].contiguous()
        return self._U_main, self._Vh_main

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize non-orthonormality of residual basis and overlap with frozen semantic subspace.

        Computes in FP32 for AMP stability:
            L_ortho = ||U_res^T U_res - I||_F^2 / r^2
                    + ||V_res V_res^T - I||_F^2 / r^2
                    + ||U_main^T U_res||_F^2 / (r_main * r)
                    + ||V_res Vh_main^T||_F^2 / (r * r_main)
        """
        U_r = self.U_res.float()
        V_r = self.V_res.float()
        eye = torch.eye(self.rank, device=U_r.device, dtype=torch.float32)

        gram_u = U_r.t() @ U_r - eye
        gram_v = V_r @ V_r.t() - eye
        loss = (gram_u.pow(2).mean() + gram_v.pow(2).mean())

        U_main, Vh_main = self._ensure_semantic_basis()
        if U_main is not None and Vh_main is not None:
            cross_u = U_main.to(device=U_r.device, dtype=torch.float32).t() @ U_r
            cross_v = V_r @ Vh_main.to(device=V_r.device, dtype=torch.float32).t()
            loss = loss + cross_u.pow(2).mean() + cross_v.pow(2).mean()
        return loss


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def apply_svd_residual_adaptation(
    encoder: nn.Module,
    rank: int = 16,
    last_n_blocks: int = 12,
    target_suffixes: Tuple[str, ...] = ("attn.out_proj", "mlp.c_fc", "mlp.c_proj", "proj"),
) -> List[str]:
    """Replace target Linear layers in the last ``last_n_blocks`` transformer blocks with SVDResidualLinear.

    Freezes all non-SVD parameters in ``encoder`` so only residual subspaces train.
    Works with OpenAI CLIP ``VisionTransformer`` (``visual.transformer.resblocks``)
    and offline test backbones.

    Returns:
        List of dotted module names that were adapted.
    """
    for param in encoder.parameters():
        param.requires_grad = False

    if rank <= 0:
        return []

    adapted: List[str] = []

    # Case 1: OpenAI CLIP VisionTransformer wrapped in CLIPViTBackbone (`encoder.visual.transformer.resblocks`)
    visual = getattr(encoder, "visual", encoder)
    transformer = getattr(visual, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None) if transformer is not None else None

    if resblocks is not None and len(resblocks) > 0:
        total_blocks = len(resblocks)
        start_idx = max(0, total_blocks - int(last_n_blocks)) if last_n_blocks > 0 else 0
        for idx in range(start_idx, total_blocks):
            block = resblocks[idx]
            for name, mod in list(block.named_modules()):
                if not isinstance(mod, nn.Linear):
                    continue
                if any(name == s or name.endswith(f".{s}") for s in target_suffixes):
                    if min(mod.in_features, mod.out_features) <= 1:
                        continue
                    svd_mod = SVDResidualLinear.from_linear(mod, rank=rank)
                    _set_submodule(block, name, svd_mod)
                    adapted.append(f"resblocks.{idx}.{name}")
        return adapted

    # Case 2: Generic / Dummy backbone fallback (adapt eligible nn.Linear layers)
    for name, mod in list(encoder.named_modules()):
        if isinstance(mod, nn.Linear) and min(mod.in_features, mod.out_features) > 1:
            if not target_suffixes or any(name == s or name.endswith(f".{s}") for s in target_suffixes):
                svd_mod = SVDResidualLinear.from_linear(mod, rank=rank)
                _set_submodule(encoder, name, svd_mod)
                adapted.append(name)
    return adapted


def collect_orthogonality_loss(module: nn.Module) -> torch.Tensor:
    """Compute average EFFORT orthogonality regularization across all SVDResidualLinear layers."""
    losses = [
        m.orthogonality_loss()
        for m in module.modules()
        if isinstance(m, SVDResidualLinear)
    ]
    if not losses:
        device = next(module.parameters()).device
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    return torch.stack(losses).mean()


def svd_trainable_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    """Yield trainable parameters belonging to SVDResidualLinear modules."""
    for m in module.modules():
        if isinstance(m, SVDResidualLinear):
            for p in m.parameters():
                if p.requires_grad:
                    yield p
