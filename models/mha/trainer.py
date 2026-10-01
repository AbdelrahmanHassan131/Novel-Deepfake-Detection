"""Frozen RGB/wavelet fusion with explicit legacy checkpoint support."""
from models.shared.two_stream_trainer import TwoStreamTrainer
from .mha_classifier import MHAFusionClassifier
from .token_fusion import TokenAttentionFusion, GatedFusion


class MHAFusionTrainer(TwoStreamTrainer):
    def name(self):
        return 'MHA_128'

    def make_head(self, opt):
        strategy = getattr(opt, 'fusion_type', 'token_attention')
        dim, heads, dropout = self.embed_dim, getattr(opt, 'num_heads', 4), getattr(opt, 'dropout', 0.1)
        if strategy == 'token_attention':
            return TokenAttentionFusion(dim, heads, dropout)
        if strategy == 'gated':
            return GatedFusion(dim, dropout)
        if strategy == 'concat':
            raise ValueError(
                "fusion_type 'concat' is not supported for arch 'MHA_128'. "
                "Use '--arch Fusion_128 --fusion_type concat'."
            )
        if strategy in ('cross_attention', 'self_attention') and not getattr(opt, '_loading_checkpoint', False):
            raise ValueError('Legacy single-key attention is evaluation-only. Use token_attention for new training.')
        if strategy in ('cross_attention', 'self_attention'):
            return MHAFusionClassifier(dim, heads, dropout, strategy)
        raise ValueError(
            f"Unsupported fusion_type '{strategy}' for arch 'MHA_128'. "
            f"Supported types: ['token_attention', 'gated']."
        )

