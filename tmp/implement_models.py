from pathlib import Path

Path('models/mha/trainer.py').write_text('''"""Frozen RGB/wavelet fusion with explicit legacy checkpoint support."""
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
        if strategy in ('cross_attention', 'self_attention') and not getattr(opt, '_loading_checkpoint', False):
            raise ValueError('Legacy single-key attention is evaluation-only. Use token_attention for new training.')
        return MHAFusionClassifier(dim, heads, dropout, strategy)
''', encoding='utf-8')
Path('models/fusion/trainer.py').write_text('''"""Concatenation control using the identical frozen expert protocol."""
from models.shared.two_stream_trainer import TwoStreamTrainer
from .fusion_classifier import ConcatenationFusionClassifier


class ConcatenationFusionTrainer(TwoStreamTrainer):
    def name(self):
        return 'Fusion_128'

    def make_head(self, opt):
        return ConcatenationFusionClassifier(self.embed_dim, getattr(opt, 'dropout', 0.1))
''', encoding='utf-8')
# All other fusion families must also fail before accepting random experts.
for file in ('models/mha_wwxc/trainer.py', 'models/fusion_wwxc/trainer.py'):
    p = Path(file)
    s = p.read_text(encoding='utf-8')
    anchor = '        # Fusion parameters'
    if anchor not in s:
        anchor = '        self.embed_dim ='
        pos = s.index(anchor)
    else:
        pos = s.index(anchor)
    guard = '''        import os
        for expert in ('rgb', 'wavelet', 'xception', 'convnext'):
            path = getattr(opt, expert + '_model_path', None)
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f'{expert} expert checkpoint required: {path!r}')
        if not getattr(opt, 'freeze_base_models', True):
            raise ValueError('Controlled fusion requires frozen experts.')

'''
    s = s[:pos] + guard + s[pos:]
    p.write_text(s, encoding='utf-8')
