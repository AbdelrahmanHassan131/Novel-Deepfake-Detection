"""
Multi-Head Attention Fusion Classifier modules.

Contains CrossAttentionFusion and MHAFusionClassifier.
Preserved exactly from MyModels/networks/MHA_128/Trainer_MHA_128.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module to fuse embeddings from two different models.
    Allows one embedding to attend to the other.
    """

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key_value):
        """
        Args:
            query: [B, embed_dim] - embeddings from one model
            key_value: [B, embed_dim] - embeddings from another model
        Returns:
            [B, embed_dim] - attended features
        """
        # Add sequence dimension for attention: [B, 1, embed_dim]
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)

        # Cross attention
        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        # Remove sequence dimension: [B, embed_dim]
        attn_output = attn_output.squeeze(1)
        query = query.squeeze(1)

        # Residual connection and normalization
        x = self.norm1(query + attn_output)

        # Feed-forward network with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class MHAFusionClassifier(nn.Module):
    """
    Multi-Head Attention Fusion model for combining RGB and Wavelet models.
    """

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1, fusion_type='cross_attention'):
        super(MHAFusionClassifier, self).__init__()

        self.fusion_type = fusion_type
        self.embed_dim = embed_dim

        if fusion_type == 'cross_attention':
            # Cross-attention: RGB attends to Wavelet and vice versa
            self.rgb_to_wavelet_attn = CrossAttentionFusion(
                embed_dim, num_heads, dropout)
            self.wavelet_to_rgb_attn = CrossAttentionFusion(
                embed_dim, num_heads, dropout)

            # Combine attended features
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'self_attention':
            # Concatenate and apply self-attention
            self.input_projection = nn.Linear(embed_dim * 2, embed_dim)

            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            self.norm = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )

        elif fusion_type == 'concat':
            # Simple concatenation with MLP
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

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
        if self.fusion_type == 'cross_attention':
            # RGB attends to Wavelet
            rgb_attended = self.rgb_to_wavelet_attn(rgb_embed, wavelet_embed)

            # Wavelet attends to RGB
            wavelet_attended = self.wavelet_to_rgb_attn(
                wavelet_embed, rgb_embed)

            # Concatenate attended features
            fused = torch.cat([rgb_attended, wavelet_attended], dim=1)
            fused = self.fusion_layer(fused)

        elif self.fusion_type == 'self_attention':
            # Concatenate and project
            concat = torch.cat([rgb_embed, wavelet_embed], dim=1)
            x = self.input_projection(concat)

            # Add sequence dimension
            x = x.unsqueeze(1)  # [B, 1, embed_dim]

            # Self-attention
            attn_output, _ = self.self_attn(x, x, x)
            attn_output = attn_output.squeeze(1)

            # Residual and FFN
            x = x.squeeze(1)
            x = self.norm(x + attn_output)
            fused = x + self.ffn(x)

        else:  # concat
            concat = torch.cat([rgb_embed, wavelet_embed], dim=1)
            fused = self.fusion_layer(concat)

        # Final classification
        output = self.classifier(fused)

        return output
