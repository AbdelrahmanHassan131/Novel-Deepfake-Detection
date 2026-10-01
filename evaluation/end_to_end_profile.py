"""Profile the complete pixel-to-score network, including every frozen expert."""
import time
import torch
from evaluation.pixel_pipeline import PixelDetector


def profile_pipeline(wrapper, batch_size=1, warmup=3, repeats=10):
    if repeats < 1 or batch_size < 1:
        raise ValueError('Positive repeats and batch_size required')
    wrapper.eval()
    pipeline = PixelDetector(wrapper)
    size = getattr(wrapper.opt, 'cropSize', 224)
    device = wrapper.device
    pixels = torch.rand(batch_size, 3, size, size, device=device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(warmup):
            pipeline(pixels)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repeats):
            pipeline(pixels)
        sync()
        elapsed = time.perf_counter() - start
    report = dict(total_parameters=sum(p.numel() for p in wrapper.parameters()),
                  trainable_parameters=sum(p.numel() for p in wrapper.parameters() if p.requires_grad),
                  parameter_size_mb=sum(p.numel() * p.element_size() for p in wrapper.parameters()) / 1024**2,
                  latency_ms_per_image=elapsed * 1000 / (repeats * batch_size),
                  throughput_images_per_second=repeats * batch_size / elapsed,
                  batch_size=batch_size, input_shape=list(pixels.shape), warmup=warmup, repeats=repeats,
                  device=str(device), torch_version=str(torch.__version__),
                  scope='normalization, differentiable wavelets, all experts and fusion; excludes file decode and face detection',
                  gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            pipeline(pixels)
        report['flops_per_image_counted'] = counter.get_total_flops() / batch_size
        report['flops_note'] = 'PyTorch counted operations; unsupported operators may be omitted'
    except Exception as error:
        report['flops_per_image_counted'] = None
        report['flops_note'] = str(error)
    return report
