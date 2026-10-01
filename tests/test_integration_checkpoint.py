"""CPU integration fixtures for the frozen two-stream checkpoint protocol."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from config import Config, config_to_opt
from config.protocol import checkpoint_metadata
from evaluation.checkpoint_loader import CheckpointLoader
from models.base.base_model import BaseModel
from models.shared.resnet import resnet50
from models.wolter2021.wavelet_cnn import WaveletPacketCNN128
from training.checkpoint_manager import CheckpointManager
from training.validator import Validator

torch.set_num_threads(2)


def _opt(root, arch='MHA_128', fusion_type='token_attention'):
    opt = config_to_opt(Config.from_defaults())
    opt.arch, opt.fusion_type = arch, fusion_type
    opt.isTrain, opt.gpu_ids, opt.pretrained = True, [], False
    opt.checkpoints_dir, opt.name = str(root), 'run'
    opt.embed_dim, opt.num_heads, opt.dropout = 8, 2, 0.0
    opt.wavelet_level, opt.cropSize, opt.loadSize = 2, 32, 32
    opt.optim, opt.lr, opt.weight_decay = 'adam', 1e-3, 0.123
    opt.freeze_base_models = True
    return opt


def _write_experts(root, opt):
    rgb = resnet50(pretrained=False, num_classes=1)
    rgb.fc = nn.Sequential(nn.Linear(rgb.fc.in_features, opt.embed_dim), nn.ReLU(),
                           nn.Dropout(0.5), nn.Linear(opt.embed_dim, 1))
    wavelet = WaveletPacketCNN128(3 * 4 ** opt.wavelet_level, embed_dim=opt.embed_dim)
    protocol = {'options': {key: getattr(opt, key) for key in (
        'cropSize', 'loadSize', 'no_crop', 'no_resize', 'rz_interp',
        'wavelet_type', 'wavelet_level', 'wavelet_mode', 'use_log_packets',
        'wavelet_log_mode')}}
    rgb_path, wavelet_path = root / 'rgb.pth', root / 'wavelet.pth'
    torch.save({'model_state_dict': rgb.state_dict(), 'protocol': protocol}, rgb_path)
    torch.save({'model_state_dict': wavelet.state_dict(), 'protocol': protocol}, wavelet_path)
    opt.rgb_model_path, opt.wavelet_model_path = str(rgb_path), str(wavelet_path)
    return rgb_path, wavelet_path


class _TinyModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def name(self):
        return 'TinyProtocolModel'


class CheckpointIntegrationTests(unittest.TestCase):
    def test_fusion_heads_roundtrip_without_expert_files(self):
        for arch, fusion_type in (('Fusion_128', 'concat'), ('MHA_128', 'token_attention'),
                                  ('MHA_128', 'gated')):
            with self.subTest(arch=arch, fusion_type=fusion_type), tempfile.TemporaryDirectory() as directory:
                root, opt = Path(directory), _opt(directory, arch, fusion_type)
                rgb_path, wavelet_path = _write_experts(root, opt)
                from models import build_model
                model = build_model(opt)
                self.assertTrue(all(not p.requires_grad for p in model.rgb_model.parameters()))
                self.assertTrue(all(not p.requires_grad for p in model.wavelet_model.parameters()))
                self.assertEqual(model.optimizer.param_groups[0]['weight_decay'], opt.weight_decay)

                batch = (torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32),
                         torch.tensor([0, 1]))
                model.set_input(batch)
                model.optimize_parameters()
                model.eval()
                with torch.no_grad():
                    expected = model.forward().detach().clone()
                manager = CheckpointManager(str(root / 'checkpoint'), model)
                manager.save_last(epoch=1, best_metric=.5, global_step=1)

                # Evaluation must be self-contained; no original expert files.
                rgb_path.unlink()
                wavelet_path.unlink()
                loaded = CheckpointLoader(str(root / 'checkpoint' / 'last.pth'), device='cpu').load()
                loaded.set_input(batch)
                with torch.no_grad():
                    actual = loaded.forward()
                torch.testing.assert_close(actual, expected)
                self.assertEqual(loaded.opt.classes, ['real', 'fake'])
                self.assertEqual(loaded.opt.wavelet_level, 2)
                self.assertTrue(all(not p.requires_grad for p in loaded.rgb_model.parameters()))

    def test_expert_preprocessing_mismatch_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root, opt = Path(directory), _opt(directory)
            _write_experts(root, opt)
            opt.wavelet_level = 3
            from models.mha.trainer import MHAFusionTrainer
            with self.assertRaisesRegex(ValueError, 'preprocessing mismatch'):
                MHAFusionTrainer(opt)
            opt.wavelet_level, opt.embed_dim = 2, 4
            with self.assertRaisesRegex(ValueError, '4-D embedding'):
                MHAFusionTrainer(opt)

    def test_validator_uses_legacy_score_direction_with_canonical_labels(self):
        class LegacyDirectionModel:
            score_sign = -1.0
            model = nn.Identity()
            def eval(self):
                return self
            def set_input(self, batch):
                self.label = batch[1].float()
                # Historical positive logits mean real (old label 1).
                self.output = batch[0].float().unsqueeze(1)
            def forward(self):
                return self.output
            def get_loss(self):
                return torch.tensor(0.)
        result = Validator().validate(LegacyDirectionModel(), [
            (torch.tensor([4., -4.]), torch.tensor([0, 1]))])
        self.assertEqual(result.accuracy, 1.0)

    def test_resume_restores_state_and_rejects_protocol_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            opt = _opt(directory, arch='TinyProtocolModel')
            model = _TinyModel(opt)
            scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, step_size=1)
            model.optimizer.zero_grad()
            model.model(torch.ones(1, 2)).sum().backward()
            model.optimizer.step()
            manager = CheckpointManager(directory, model)
            manager.save_last(epoch=4, best_metric=.7, global_step=19, scheduler=scheduler)

            restored = _TinyModel(copy.deepcopy(opt))
            restored_scheduler = torch.optim.lr_scheduler.StepLR(restored.optimizer, step_size=1)
            info = CheckpointManager(directory, restored).resume(scheduler=restored_scheduler)
            self.assertEqual((info['epoch'], info['global_step']), (4, 19))
            self.assertEqual(restored.optimizer.state_dict()['state'].keys(), model.optimizer.state_dict()['state'].keys())

            changed = _TinyModel(copy.deepcopy(opt))
            changed.opt.cropSize = 99
            with self.assertRaisesRegex(ValueError, 'Resume protocol mismatch for cropSize'):
                CheckpointManager(directory, changed).resume()


if __name__ == '__main__':
    unittest.main()
