"""
Concatenation Fusion Classifier module.

Simple Concatenation Fusion model for combining RGB and Wavelet models.
Instead of attention, uses straightforward concatenation and MLP fusion.

Preserved exactly from MyModels/networks/Fusion_128/Trainer_Fusion_128.py.
"""
import torch
import torch.nn as nn


class ConcatenationFusionClassifier(nn.Module):
    """
    Simple Concatenation Fusion model for combining RGB and Wavelet models.
    Instead of attention, uses straightforward concatenation and MLP fusion.
    """

    def __init__(self, embed_dim=128, dropout=0.1):
        super(ConcatenationFusionClassifier, self).__init__()

        self.embed_dim = embed_dim

        # Simple concatenation fusion with MLP
        # Input: concatenated embeddings of size embed_dim * 2 (256)
        # Output: fused embedding of size embed_dim (128)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, rgb_embed, wavelet_embed):
        """
        Args:
            rgb_embed: [B, 128] - embeddings from RGB model
            wavelet_embed: [B, 128] - embeddings from Wavelet model
        Returns:
            [B, 1] - logits for binary classification
        """
        # Simple concatenation
        concat = torch.cat([rgb_embed, wavelet_embed], dim=1)  # [B, 256]

        # Fusion through MLP
        fused = self.fusion_layer(concat)  # [B, 128]

        # Final classification
        output = self.classifier(fused)  # [B, 1]

        return output
