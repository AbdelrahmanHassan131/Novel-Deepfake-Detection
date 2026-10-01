"""Controlled fusion trainer supporting concatenation and gated heads."""
from models.shared.two_stream_trainer import TwoStreamTrainer
from .fusion_classifier import ConcatenationFusionClassifier
from models.mha.token_fusion import GatedFusion


class ControlledFusionTrainer(TwoStreamTrainer):
    def name(self):
        return 'Fusion_128'

    def make_head(self, opt):
        strategy = getattr(opt, 'fusion_type', 'concat')
        dim = self.embed_dim
        dropout = getattr(opt, 'dropout', 0.1)
        if strategy == 'concat':
            return ConcatenationFusionClassifier(dim, dropout)
        elif strategy == 'gated':
            return GatedFusion(dim, dropout)
        else:
            raise ValueError(
                f"Unsupported fusion_type '{strategy}' for arch 'Fusion_128'. "
                f"Supported types: ['concat', 'gated']."
            )


# Backward-compatibility alias
ConcatenationFusionTrainer = ControlledFusionTrainer

