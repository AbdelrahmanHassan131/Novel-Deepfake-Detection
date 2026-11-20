"""EFFORT_128 Model: Multi-Layer Intermediate CLIP Fusion + Orthogonal SVD Adaptation.

Combines two complementary cross-domain deepfake generalization mechanisms:
  1. Multi-layer intermediate ViT [CLS] feature fusion (RINE-style): taps early, middle,
     and late transformer blocks so low-level generative/VAE texture cues are preserved
     alongside high-level semantics, aggregated via learnable softmax layer weights.
  2. Orthogonal SVD residual adaptation (EFFORT-style): keeps primary pretrained CLIP
     singular directions strictly frozen while adapting only the orthogonal residual
     subspace in the last K transformer blocks.

Contract compliance:
  - Output: single raw binary logit [B, 1] (real=0, fake=1)
  - Penultimate feature: 128-D embedding [B, 128] via forward_features(x)
  - Exact classifier parity: logits == fc(forward_features(x)) in eval mode
"""
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.drct.backbone import build_drct_backbone
from models.effort.svd_linear import (
    SVDResidualLinear,
    apply_svd_residual_adaptation,
    collect_orthogonality_loss,
)


class DummyMultiLayerBackbone(nn.Module):
    """Lightweight multi-block ViT-like backbone for fast offline CPU unit testing."""

    def __init__(self, width: int = 64, num_blocks: int = 4, out_features: int = 64):
        super().__init__()
        self.width = width
        self.out_features = out_features
        self.conv1 = nn.Conv2d(3, width, kernel_size=16, stride=16)

        class _FakeBlock(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.ln = nn.LayerNorm(d)
                self.mlp = nn.Module()
                self.mlp.c_fc = nn.Linear(d, d * 2)
                self.mlp.c_proj = nn.Linear(d * 2, d)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Sequence-first [L, B, D] matching OpenAI CLIP VisionTransformer
                h = self.ln(x)
                h = self.mlp.c_proj(F.gelu(self.mlp.c_fc(h)))
                return x + h

        self.visual = nn.Module()
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.ModuleList([_FakeBlock(width) for _ in range(num_blocks)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.ln_post = nn.LayerNorm(width)
        self.head_proj = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        tokens = self.conv1(x).flatten(2).permute(2, 0, 1)  # [L_patches, B, D]
        cls = self.cls_token.expand(1, b, -1)
        seq = torch.cat([cls, tokens], dim=0)  # [1 + L_patches, B, D]
        for blk in self.visual.transformer.resblocks:
            seq = blk(seq)
        return self.head_proj(self.ln_post(seq[0]))


def _resolve_layer_indices(total_blocks: int, num_taps: int = 4, explicit: Optional[Sequence[int]] = None) -> List[int]:
    """Choose sorted unique 0-based transformer block indices to tap."""
    if total_blocks <= 0:
        return []
    if explicit:
        indices = sorted({int(i) % total_blocks if int(i) < 0 else int(i) for i in explicit})
        for idx in indices:
            if idx < 0 or idx >= total_blocks:
                raise ValueError(f"Layer index {idx} out of range for backbone with {total_blocks} blocks.")
        return indices
    k = max(1, min(int(num_taps), total_blocks))
    if k == 1:
        return [total_blocks - 1]
    # Evenly spaced indices ending at the final block (e.g. for 24 blocks, k=4 -> [5, 11, 17, 23])
    indices = [round((i + 1) * total_blocks / k) - 1 for i in range(k)]
    return sorted(set(indices))


def _infer_block_width(encoder: nn.Module, fallback_dim: int) -> int:
    visual = getattr(encoder, "visual", encoder)
    if hasattr(visual, "ln_post") and hasattr(visual.ln_post, "normalized_shape"):
        return int(visual.ln_post.normalized_shape[0])
    if hasattr(visual, "conv1") and hasattr(visual.conv1, "out_channels"):
        return int(visual.conv1.out_channels)
    if hasattr(encoder, "width"):
        return int(encoder.width)
    return int(fallback_dim)


class EFFORT128Model(nn.Module):
    """Multi-Layer + Orthogonal SVD CLIP detector with 128-D bottleneck and single raw logit."""

    def __init__(
        self,
        backbone_name: str = "clip-ViT-L/14",
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        allow_download: bool = False,
        embed_dim: int = 128,
        svd_rank: int = 16,
        svd_last_n_blocks: int = 12,
        multi_layer: bool = True,
        num_tapped_layers: int = 4,
        layer_indices: Optional[Sequence[int]] = None,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.3,
        encoder_module: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
    ):
        super().__init__()
        if embed_dim != 128:
            raise ValueError(f"EFFORT_128 strictly requires embed_dim=128, got embed_dim={embed_dim}.")

        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.svd_rank = int(svd_rank)
        self.svd_last_n_blocks = int(svd_last_n_blocks)
        self.multi_layer = bool(multi_layer)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)

        # 1. Build visual encoder
        if encoder_module is not None:
            self.encoder = encoder_module
            self.backbone_dim = int(backbone_dim or getattr(encoder_module, "out_features", 64))
        elif str(backbone_name).strip().lower() in ("dummy", "tiny", "mock", "dummy_multilayer"):
            self.encoder = DummyMultiLayerBackbone(width=64, num_blocks=4, out_features=64)
            self.backbone_dim = 64
        else:
            self.encoder, self.backbone_dim = build_drct_backbone(
                name=backbone_name,
                pretrained=pretrained,
                weights_path=weights_path,
                freeze=True,
                allow_download=allow_download,
            )

        # 2. Apply Orthogonal SVD Residual Adaptation (or freeze completely if svd_rank == 0)
        self.adapted_layers = apply_svd_residual_adaptation(
            self.encoder,
            rank=self.svd_rank,
            last_n_blocks=self.svd_last_n_blocks,
        )

        # 3. Register multi-layer [CLS] hooks if resblocks are available and multi_layer is enabled
        visual = getattr(self.encoder, "visual", self.encoder)
        transformer = getattr(visual, "transformer", None)
        resblocks = getattr(transformer, "resblocks", None) if transformer is not None else None

        self.tapped_indices: List[int] = []
        self._hook_handles = []
        self._hook_outputs: Dict[int, torch.Tensor] = {}

        if self.multi_layer and resblocks is not None and len(resblocks) > 0:
            self.tapped_indices = _resolve_layer_indices(
                total_blocks=len(resblocks),
                num_taps=num_tapped_layers,
                explicit=layer_indices,
            )
            self.tap_dim = _infer_block_width(self.encoder, self.backbone_dim)
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.tap_dim) for _ in self.tapped_indices])
            # Learnable layer-importance logits (uniform at init)
            self.layer_weights = nn.Parameter(torch.zeros(len(self.tapped_indices), dtype=torch.float32))
            self._register_layer_hooks(resblocks)
            # Fuse weighted multi-layer CLS (tap_dim) + final CLIP visual embedding (backbone_dim)
            fused_in_dim = self.tap_dim + self.backbone_dim
        else:
            self.tap_dim = self.backbone_dim
            self.layer_norms = nn.ModuleList()
            self.register_parameter("layer_weights", None)
            fused_in_dim = self.backbone_dim

        self.final_norm = nn.LayerNorm(self.backbone_dim)

        # 4. 128-D forensic bottleneck projection (pre-dropout feature feeds both fc and MHA_128)
        self.projection = nn.Sequential(
            nn.Linear(fused_in_dim, self.head_hidden_dim),
            nn.LayerNorm(self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=self.head_dropout),
            nn.Linear(self.head_hidden_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.head_drop = nn.Dropout(p=self.head_dropout)
        self.fc = nn.Linear(self.embed_dim, 1)

        self._init_head()

    def _init_head(self, init_gain: float = 0.02) -> None:
        for module in (self.projection, self.fc):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0.0)

    def _register_layer_hooks(self, resblocks: nn.ModuleList) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        for slot, block_idx in enumerate(self.tapped_indices):
            def _make_hook(s: int):
                def _hook(_module, _inp, out):
                    self._hook_outputs[s] = out
                return _hook

            handle = resblocks[block_idx].register_forward_hook(_make_hook(slot))
            self._hook_handles.append(handle)

    def train(self, mode: bool = True) -> "EFFORT128Model":
        """Keep encoder dropout/norm in eval mode so pretrained statistics remain stable."""
        super().train(mode)
        self.encoder.eval()
        return self

    def get_layer_importance(self) -> Dict[int, float]:
        """Return normalized softmax importance weight per tapped transformer block."""
        if self.layer_weights is None or not self.tapped_indices:
            return {}
        probs = torch.softmax(self.layer_weights.detach().float(), dim=0).cpu().tolist()
        return {int(idx): float(p) for idx, p in zip(self.tapped_indices, probs)}

    def orthogonality_loss(self) -> torch.Tensor:
        """Compute average EFFORT orthogonality regularization over all SVD-adapted layers."""
        return collect_orthogonality_loss(self.encoder)

    def _extract_fused_representation(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        self._hook_outputs.clear()

        needs_encoder_grad = (len(self.adapted_layers) > 0) or x.requires_grad
        if not needs_encoder_grad:
            with torch.no_grad():
                final_feat = self.encoder(x)
        else:
            final_feat = self.encoder(x)

        if isinstance(final_feat, (tuple, list)):
            final_feat = final_feat[0]
        final_feat = self.final_norm(final_feat.float())

        if self.tapped_indices:
            weights = torch.softmax(self.layer_weights.float(), dim=0)
            weighted_cls = None
            for slot in range(len(self.tapped_indices)):
                raw_out = self._hook_outputs.get(slot)
                if raw_out is None:
                    raise RuntimeError(f"Missing intermediate hook output for tapped slot {slot}.")
                # OpenAI CLIP resblocks output sequence-first [L, B, D]; support [B, L, D] defensively.
                if raw_out.dim() == 3:
                    if raw_out.shape[1] == b:
                        cls_tok = raw_out[0]
                    elif raw_out.shape[0] == b:
                        cls_tok = raw_out[:, 0]
                    else:
                        raise RuntimeError(
                            f"Unexpected transformer block output shape {tuple(raw_out.shape)} for batch_size={b}."
                        )
                elif raw_out.dim() == 2:
                    cls_tok = raw_out
                else:
                    raise RuntimeError(f"Unsupported hook tensor dim: {raw_out.dim()}")

                normed = self.layer_norms[slot](cls_tok.float())
                term = weights[slot] * normed
                weighted_cls = term if weighted_cls is None else (weighted_cls + term)

            self._hook_outputs.clear()
            return torch.cat([weighted_cls, final_feat], dim=-1)

        self._hook_outputs.clear()
        return final_feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-D forensic embedding (exact feature feeding fc in eval mode)."""
        fused = self._extract_fused_representation(x)
        return self.projection(fused)

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning raw binary logit [B, 1] (and optional 128-D feature)."""
        feat_128 = self.forward_features(x)
        logits = self.fc(self.head_drop(feat_128))
        if return_feature:
            return logits, feat_128
        return logits

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.2, weight_decay: float = 1e-2) -> List[Dict]:
        """Build AdamW parameter groups separating SVD residual params and head params (no decay on norms/biases)."""
        svd_decay, svd_no_decay = [], []
        head_decay, head_no_decay = [], []

        svd_param_ids = set()
        for m in self.encoder.modules():
            if isinstance(m, SVDResidualLinear):
                for name, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    svd_param_ids.add(id(p))
                    if p.ndim <= 1 or name == "S_res":
                        svd_no_decay.append(p)
                    else:
                        svd_decay.append(p)

        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in svd_param_ids:
                continue
            if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "layer_weights" in name:
                head_no_decay.append(p)
            else:
                head_decay.append(p)

        groups = []
        if svd_decay:
            groups.append({"params": svd_decay, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay})
        if svd_no_decay:
            groups.append({"params": svd_no_decay, "lr": lr * backbone_lr_mult, "weight_decay": 0.0})
        if head_decay:
            groups.append({"params": head_decay, "lr": lr, "weight_decay": weight_decay})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": lr, "weight_decay": 0.0})
        return groups
