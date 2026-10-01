"""Differentiable raw-pixel inference for complete detector robustness tests."""
import torch
from torch import nn

MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def normalize(pixels):
    return (pixels - pixels.new_tensor(MEAN)[None, :, None, None]) / pixels.new_tensor(STD)[None, :, None, None]


def denormalize(rgb):
    return rgb * rgb.new_tensor(STD)[None, :, None, None] + rgb.new_tensor(MEAN)[None, :, None, None]


class PixelDetector(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper
        self.opt = wrapper.opt
        self.arch = self.opt.arch
        self.wavelet = None
        if 'Wolter' in self.arch or hasattr(wrapper, 'wavelet_model'):
            from data.wavelets.backends.gpu_backend import GPUWaveletBackend
            self.wavelet = GPUWaveletBackend(wavelet=getattr(self.opt, 'wavelet_type', 'haar'),
                level=getattr(self.opt, 'wavelet_level', 3), mode=getattr(self.opt, 'wavelet_mode', 'reflect'),
                log_scale=getattr(self.opt, 'use_log_packets', True), device=wrapper.device)
            self.wavelet.log_mode = getattr(self.opt, 'wavelet_log_mode', 'signed_log1p')

    def forward(self, pixels):
        rgb = normalize(pixels)
        w = self.wrapper
        if hasattr(w, 'rgb_model'):
            streams = [w.rgb_model(rgb), w.wavelet_model(self.wavelet(pixels * 255.0))]
            streams += [getattr(w, name)(rgb) for name in ('xception_model', 'convnext_model') if hasattr(w, name)]
            logits = w.model(*streams)
        elif 'Wolter' in self.arch:
            logits = w.model(self.wavelet(pixels * 255.0))
        else:
            logits = w.model(rgb)
        return logits.flatten() * getattr(w, 'score_sign', 1.0)


def projected_attack(model, pixels, labels, eps=4/255, steps=10, step_size=1/255, random_start=True, restarts=1):
    if not 0 <= eps <= 1 or steps < 1 or step_size <= 0 or restarts < 1:
        raise ValueError('Invalid raw-pixel attack parameters')
    if pixels.min() < 0 or pixels.max() > 1:
        raise ValueError('Attack inputs must be raw [0,1] pixels')
    clean = pixels.detach()
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    with torch.no_grad():
        best_loss = loss_fn(model(clean), labels.float())
    best = clean.clone()
    for _ in range(restarts):
        adv = (clean + torch.empty_like(clean).uniform_(-eps, eps)).clamp(0, 1) if random_start else clean.clone()
        for _ in range(steps):
            adv.requires_grad_(True)
            losses = loss_fn(model(adv), labels.float())
            grad, = torch.autograd.grad(losses.sum(), adv)
            if not torch.isfinite(grad).all():
                raise ValueError('Nonfinite attack gradient; robustness result would be invalid')
            with torch.no_grad():
                candidate = adv + step_size * grad.sign()
                adv = torch.maximum(torch.minimum(candidate, clean + eps), clean - eps).clamp(0, 1)
                current = loss_fn(model(adv), labels.float())
                improve = current > best_loss
                best[improve], best_loss[improve] = adv[improve], current[improve]
    return best.detach()
