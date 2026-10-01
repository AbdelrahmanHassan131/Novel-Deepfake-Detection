"""Multiple-key attention and input-dependent gating controls."""
import torch
from torch import nn


class TokenAttentionFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.wavelet_norm = nn.LayerNorm(embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.types = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * embed_dim, embed_dim), nn.Dropout(dropout))
        self.classifier = nn.Linear(embed_dim, 1)
        self.last_attention = None

    def forward(self, rgb, wavelet):
        tokens = torch.cat((self.cls.expand(rgb.shape[0], -1, -1),
                            self.rgb_norm(rgb).unsqueeze(1), self.wavelet_norm(wavelet).unsqueeze(1)), 1)
        tokens = tokens + self.types
        attended, weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        self.last_attention = weights.detach()
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return self.classifier(tokens[:, 0])


class GatedFusion(nn.Module):
    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        self.rgb_norm, self.wavelet_norm = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.ReLU(),
                                  nn.Linear(embed_dim, 2))
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.last_gate = None

    def forward(self, rgb, wavelet):
        rgb, wavelet = self.rgb_norm(rgb), self.wavelet_norm(wavelet)
        weights = self.gate(torch.cat((rgb, wavelet), -1)).softmax(-1)
        self.last_gate = weights.detach()
        return self.classifier(weights[:, :1] * rgb + weights[:, 1:] * wavelet)
