"""
Concatenation Fusion Classifier for 4-model WWXC fusion.

Fuses 4 streams (Wang2020_128, Wolter_128, Xception_128, ConvNeXt_128)
via concatenation (128×4 = 512-D) followed by an MLP head.
"""
import torch
import torch.nn as nn


class ConcatenationFusionWWXCClassifier(nn.Module):
    """
    Concatenation Fusion classifier for 4-model WWXC fusion.

    Input: four 128-D embedding vectors (wang, wolter, xception, convnext).
    Architecture:
        concat(4×128 = 512) → Linear(512, 256) → ReLU → Dropout
        → Linear(256, 128) → ReLU → Dropout
        → Linear(128, 64) → ReLU → Dropout → Linear(64, 1)
    """

    def __init__(self, embed_dim=128, dropout=0.1):
        super(ConcatenationFusionWWXCClassifier, self).__init__()

        self.embed_dim = embed_dim
        num_streams = 4
        concat_dim = embed_dim * num_streams  # 512

        # Fusion MLP: 512 → 256 → 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),     # 512 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(concat_dim // 2, embed_dim),       # 256 → 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Final classifier: 128 → 64 → 1
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, wang_embed, wolter_embed, xception_embed, convnext_embed):
        """
        Args:
            wang_embed:     [B, 128] - embeddings from Wang2020_128
            wolter_embed:   [B, 128] - embeddings from Wolter_128
            xception_embed: [B, 128] - embeddings from Xception_128
            convnext_embed: [B, 128] - embeddings from ConvNeXt_128
        Returns:
            [B, 1] - logits for binary classification
        """
        # Concatenate all 4 streams: [B, 512]
        concat = torch.cat(
            [wang_embed, wolter_embed, xception_embed, convnext_embed], dim=1
        )

        # Fusion MLP
        fused = self.fusion_layer(concat)  # [B, 128]

        # Final classification
        output = self.classifier(fused)  # [B, 1]

        return output
