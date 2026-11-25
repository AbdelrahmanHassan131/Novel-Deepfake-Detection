"""Capacity-matched and structural ablation fusion heads for Paper 1 Revision.

Directly addresses Reviewer 1 (#1, #4, #7) and Reviewer 2 (#5):
  - TokenAttentionFusion (MHA_128): 199,425 trainable params ([CLS, RGB, Wavelet], L=3, 4 heads)
  - ConcatMatchedMLP:              ~198,017 trainable params (1:1 capacity-matched pure MLP)
  - NoAttnFFNControl:              ~198,529 trainable params (1:1 capacity-matched residual FFN without attention)
  - LinearMixControl:              ~199,041 trainable params (Linear(256->128) + residual linear mixing + FFN, no softmax)
  - GatedFusion:                    ~41,923 trainable params (sample-dependent softmax gating over modalities)
  - ConcatenationFusionClassifier:  107,009 trainable params (original Fusion_128 baseline)
"""
import torch
from torch import nn

from models.fusion.fusion_classifier import ConcatenationFusionClassifier
from models.mha.token_fusion import GatedFusion, TokenAttentionFusion


class ConcatMatchedMLP(nn.Module):
    """Parameter-matched pure-MLP concatenation control (~198K params, ratio ~1.007x vs TokenAttentionFusion)."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.wavelet_norm = nn.LayerNorm(embed_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, 3 * embed_dim),   # 256 -> 384 (98,688)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * embed_dim, 2 * embed_dim),   # 384 -> 256 (98,560)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, 1),               # 256 -> 1   (257)
        )

    def forward(self, rgb: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
        x = torch.cat((self.rgb_norm(rgb), self.wavelet_norm(wavelet)), dim=-1)
        return self.net(x)


class NoAttnFFNControl(nn.Module):
    """Capacity-matched ablated control retaining ONLY residual FFN refinement (no attention, ~198.5K params)."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.wavelet_norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(2 * embed_dim, embed_dim)  # 32,896
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 5 * embed_dim),            # 128 -> 640 (82,560)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(5 * embed_dim, embed_dim),            # 640 -> 128 (82,048)
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, rgb: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.in_proj(torch.cat((self.rgb_norm(rgb), self.wavelet_norm(wavelet)), dim=-1)))
        h = self.norm2(h + self.ffn(h))
        return self.classifier(h)


class LinearMixControl(nn.Module):
    """Linear(128->128) cross-mixing + FFN without softmax attention (R1 #4(iii) & #7, ~199K params)."""

    def __init__(self, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.wavelet_norm = nn.LayerNorm(embed_dim)
        self.rgb_lin = nn.Linear(embed_dim, embed_dim)      # 16,512
        self.wav_lin = nn.Linear(embed_dim, embed_dim)      # 16,512
        self.mix_proj = nn.Linear(2 * embed_dim, embed_dim) # 32,896
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),            # 128 -> 512 (66,048)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),            # 512 -> 128 (65,664)
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, rgb: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
        r = self.rgb_norm(rgb)
        w = self.wavelet_norm(wavelet)
        # Fixed linear cross-mixing without softmax attention weights
        r_mix = r + self.wav_lin(w)
        w_mix = w + self.rgb_lin(r)
        h = self.norm1(self.mix_proj(torch.cat((r_mix, w_mix), dim=-1)))
        h = self.norm2(h + self.ffn(h))
        return self.classifier(h)


def build_ablation_head(name: str, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.2) -> nn.Module:
    """Instantiate any Paper 1 revision fusion head by canonical name."""
    key = name.strip().lower()
    if key in ("mha_128", "token_attention", "mha"):
        return TokenAttentionFusion(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    if key in ("gated", "gated_fusion"):
        return GatedFusion(embed_dim=embed_dim, dropout=dropout)
    if key in ("fusion_128", "concat", "concat_orig"):
        return ConcatenationFusionClassifier(embed_dim=embed_dim, dropout=dropout)
    if key in ("concat_matched", "mlp_matched"):
        return ConcatMatchedMLP(embed_dim=embed_dim, dropout=dropout)
    if key in ("no_attn_ffn", "ffn_only"):
        return NoAttnFFNControl(embed_dim=embed_dim, dropout=dropout)
    if key in ("linear_mix", "linear_no_softmax"):
        return LinearMixControl(embed_dim=embed_dim, dropout=dropout)
    raise ValueError(f"Unknown ablation head: {name!r}")


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
