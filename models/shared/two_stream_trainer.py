"""Common frozen-expert trainer; gradients to input remain available for attacks."""
import torch
from torch import nn
from models.base.base_model import BaseModel
from models.shared.expert_loading import load_expert


class TwoStreamTrainer(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.embed_dim = getattr(opt, 'embed_dim', 128)
        self.freeze_base_models = getattr(opt, 'freeze_base_models', True)
        if not self.freeze_base_models:
            raise ValueError('This controlled fusion trainer requires frozen experts. Train experts separately.')
        self.rgb_model = load_expert(opt, 'rgb', self.device)
        self.wavelet_model = load_expert(opt, 'wavelet', self.device)
        self.model = self.make_head(opt).to(self.device)
        self.head_class_name = self.model.__class__.__name__
        self.head_param_count = sum(p.numel() for p in self.model.parameters())
        self.head_trainable_param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.loss_fn = nn.BCEWithLogitsLoss()
        if self.isTrain:
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr,
                    betas=(opt.beta1, 0.999), weight_decay=getattr(opt, 'weight_decay', 0.0))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr,
                    momentum=getattr(opt, 'momentum', 0.9), weight_decay=getattr(opt, 'weight_decay', 0.0))
            else:
                raise ValueError(opt.optim)

    def _get_gpu_wavelet_backend(self):
        if not hasattr(self, '_gpu_wavelet'):
            from data.wavelets.backends.gpu_backend import GPUWaveletBackend
            self._gpu_wavelet = GPUWaveletBackend(wavelet=getattr(self.opt, 'wavelet_type', 'haar'),
                level=getattr(self.opt, 'wavelet_level', 3), mode=getattr(self.opt, 'wavelet_mode', 'reflect'),
                log_scale=getattr(self.opt, 'use_log_packets', True), device=self.device)
            self._gpu_wavelet.log_mode = getattr(self.opt, 'wavelet_log_mode', 'signed_log1p')
        return self._gpu_wavelet

    def set_input(self, batch):
        self.rgb_input = batch[0].to(self.device)
        wavelet = batch[1].to(self.device)
        self.wavelet_input = self._get_gpu_wavelet_backend()(wavelet) if wavelet.shape[1] == 3 else wavelet
        self.label = batch[2].to(self.device).float()

    def forward(self, rgb=None, wavelet=None):
        rgb = self.rgb_input if rgb is None else rgb
        wavelet = self.wavelet_input if wavelet is None else wavelet
        self.rgb_embed = self.rgb_model(rgb)
        self.wavelet_embed = self.wavelet_model(wavelet)
        expected = (rgb.shape[0], self.embed_dim)
        if self.rgb_embed.shape != expected or self.wavelet_embed.shape != expected:
            raise ValueError(f'Expert embeddings must both have shape {expected}')
        self.output = self.model(self.rgb_embed, self.wavelet_embed)
        return self.output

    def train(self, mode=True):
        super().train(mode)
        self.model.train(mode)
        self.rgb_model.eval()
        self.wavelet_model.eval()
        return self

    def eval(self):
        return self.train(False)

    def get_loss(self):
        return self.loss_fn(self.output.flatten(), self.label)

    def optimize_parameters(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.forward()
        self.loss = self.get_loss()
        self.loss.backward()
        self.optimizer.step()

    def adjust_learning_rate(self, min_lr=1e-6):
        for group in self.optimizer.param_groups:
            group['lr'] /= 10
        return min(g['lr'] for g in self.optimizer.param_groups) >= min_lr
