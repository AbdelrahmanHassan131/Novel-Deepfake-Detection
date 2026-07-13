"""
Multi-Head Attention Fusion Classifier for 4-model WWXC fusion.

Fuses 4 streams (Wang2020_128, Wolter_128, Xception_128, ConvNeXt_128)
using pairwise cross-attention, self-attention, or concatenation fusion.
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
            query:     [B, embed_dim] - embeddings from one model
            key_value: [B, embed_dim] - embeddings from another model
        Returns:
            [B, embed_dim] - attended features
        """
        query = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)

        attn_output, _ = self.multihead_attn(query, key_value, key_value)

        attn_output = attn_output.squeeze(1)
        query = query.squeeze(1)

        x = self.norm1(query + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class MHAFusionWWXCClassifier(nn.Module):
    """
    Multi-Head Attention Fusion classifier for 4-model WWXC fusion.

    For 'cross_attention':
        Each of the 4 streams attends to a combined representation
        of the other 3 streams (mean-pooled), producing 4 attended
        128-D vectors which are concatenated (4×128=512) and fused.

    For 'self_attention':
        All 4 embeddings are stacked into a 4-token sequence and
        processed by multi-head self-attention.

    For 'concat':
        Simple concatenation (4×128=512) + MLP.
    """

    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1,
                 fusion_type='cross_attention'):
        super(MHAFusionWWXCClassifier, self).__init__()

        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        self.num_streams = 4

        if fusion_type == 'cross_attention':
            # Each stream has its own cross-attention attending to others
            self.cross_attns = nn.ModuleList([
                CrossAttentionFusion(embed_dim, num_heads, dropout)
                for _ in range(self.num_streams)
            ])

            # Combine the 4 attended features
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * self.num_streams, embed_dim * 2),  # 512 → 256
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),  # 256 → 128
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'self_attention':
            # Stack as 4-token sequence and apply self-attention
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

            # Pool the 4 attended tokens into a single vector
            self.pool_layer = nn.Sequential(
                nn.Linear(embed_dim * self.num_streams, embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'concat':
            # Simple concatenation with MLP
            self.fusion_layer = nn.Sequential(
                nn.Linear(embed_dim * self.num_streams, embed_dim * 2),  # 512 → 256
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),  # 256 → 128
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
        embeds = [wang_embed, wolter_embed, xception_embed, convnext_embed]

        if self.fusion_type == 'cross_attention':
            attended = []
            for i in range(self.num_streams):
                # Mean-pool the other 3 streams as key/value
                others = [embeds[j] for j in range(self.num_streams) if j != i]
                kv = torch.stack(others, dim=0).mean(dim=0)  # [B, 128]
                attended_i = self.cross_attns[i](embeds[i], kv)
                attended.append(attended_i)

            fused = torch.cat(attended, dim=1)  # [B, 512]
            fused = self.fusion_layer(fused)     # [B, 128]

        elif self.fusion_type == 'self_attention':
            # Stack into [B, 4, embed_dim]
            x = torch.stack(embeds, dim=1)

            # Self-attention
            attn_output, _ = self.self_attn(x, x, x)  # [B, 4, embed_dim]

            # Residual + FFN
            x = self.norm(x + attn_output)
            x = x + self.ffn(x)

            # Flatten and pool: [B, 4*embed_dim] → [B, embed_dim]
            x = x.reshape(x.size(0), -1)
            fused = self.pool_layer(x)

        else:  # concat
            concat = torch.cat(embeds, dim=1)  # [B, 512]
            fused = self.fusion_layer(concat)   # [B, 128]

        # Final classification
        output = self.classifier(fused)

        return output
