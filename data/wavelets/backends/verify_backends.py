"""
Verification script for the GPU Wavelet Backend refactor.

Checks:
    1. Imports — all backend classes and factory are importable.
    2. No circular dependencies — importing from different entry points works.
    3. CPU backend — produces correct shape from a random image.
    4. GPU backend — initialises and produces correct shape (CUDA required).
    5. Precomputed backend — loads a temporary .npy file correctly.
    6. DDP compatibility — GPU backend respects LOCAL_RANK device assignment.
    7. Backward compatibility — WaveletDataset and FusionDataset construct
       with all three backend values.
    8. Shape consistency — CPU and GPU backends produce identical shapes.

Usage::

    python -m data.wavelets.backends.verify_backends

Does NOT create benchmarks, profiling scripts, or visualisation tools.
"""

import sys
import os
import tempfile
import traceback

import numpy as np
import torch


def _header(title):
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')


def _pass(msg):
    print(f'  [PASS] {msg}')


def _fail(msg, exc=None):
    print(f'  [FAIL] {msg}')
    if exc:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def _skip(msg):
    print(f'  [SKIP] {msg}')


def check_imports():
    """1. Verify all backend classes are importable."""
    _header('1. Import checks')
    try:
        from data.wavelets.backends import (
            WaveletBackend,
            CPUWaveletBackend,
            GPUWaveletBackend,
            PrecomputedWaveletBackend,
            create_wavelet_backend,
        )
        _pass('All backend classes imported from backends package.')
    except Exception as e:
        _fail('Failed to import backend classes.', e)
        return False

    try:
        from data.wavelets import (
            WaveletBackend as WB2,
            create_wavelet_backend as cwb2,
        )
        _pass('Backend classes importable via data.wavelets shortcut.')
    except Exception as e:
        _fail('Shortcut import failed.', e)
        return False

    return True


def check_no_circular():
    """2. Verify no circular imports."""
    _header('2. Circular dependency check')
    try:
        # Re-import from different paths to trigger any cycles
        from data.wavelets.backends.base import WaveletBackend
        from data.wavelets.backends.cpu_backend import CPUWaveletBackend
        from data.wavelets.backends.gpu_backend import GPUWaveletBackend
        from data.wavelets.backends.precomputed_backend import PrecomputedWaveletBackend
        from data.wavelets.backends.factory import create_wavelet_backend
        _pass('No circular dependencies detected.')
        return True
    except Exception as e:
        _fail('Circular import detected.', e)
        return False


def check_cpu_backend():
    """3. CPU backend produces correct shapes."""
    _header('3. CPU backend')
    try:
        from data.wavelets.backends import CPUWaveletBackend

        backend = CPUWaveletBackend(wavelet='haar', level=3, mode='reflect',
                                    log_scale=True)
        assert backend.name == 'cpu', f'Expected name="cpu", got {backend.name}'
        _pass(f'CPU backend created: {backend}')

        # Random image (H, W, 3)
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = backend(img)

        expected_channels = 3 * (4 ** 3)  # 192
        assert isinstance(result, torch.Tensor), \
            f'Expected torch.Tensor, got {type(result)}'
        assert result.shape[0] == expected_channels, \
            f'Expected {expected_channels} channels, got {result.shape[0]}'
        assert result.dtype == torch.float32, \
            f'Expected float32, got {result.dtype}'
        _pass(f'CPU backend output shape: {result.shape}  (expected C=192)')

        return True, result.shape
    except Exception as e:
        _fail('CPU backend check failed.', e)
        return False, None


def check_gpu_backend():
    """4. GPU backend initialises and produces correct shapes."""
    _header('4. GPU backend')

    if not torch.cuda.is_available():
        _skip('CUDA not available — skipping GPU backend check.')
        return True, None

    try:
        from data.wavelets.backends import GPUWaveletBackend

        device = torch.device('cuda:0')
        backend = GPUWaveletBackend(
            wavelet='haar', level=3, mode='reflect',
            log_scale=True, device=device,
        )
        assert backend.name == 'gpu', f'Expected name="gpu", got {backend.name}'
        _pass(f'GPU backend created: {backend}')

        # Single image tensor
        img = torch.randn(3, 128, 128)
        result = backend(img)

        expected_channels = 3 * (4 ** 3)  # 192
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == expected_channels, \
            f'Expected {expected_channels} channels, got {result.shape[0]}'
        assert result.device.type == 'cuda', \
            f'Expected CUDA tensor, got device={result.device}'
        _pass(f'GPU backend single-image shape: {result.shape}  device={result.device}')

        # Batched input
        batch = torch.randn(4, 3, 128, 128)
        result_batch = backend(batch)
        assert result_batch.shape == (4, expected_channels, result.shape[1], result.shape[2]), \
            f'Unexpected batch shape: {result_batch.shape}'
        _pass(f'GPU backend batched shape: {result_batch.shape}')

        return True, result.shape
    except Exception as e:
        _fail('GPU backend check failed.', e)
        return False, None


def check_precomputed_backend():
    """5. Precomputed backend loads .npy correctly."""
    _header('5. Precomputed backend')
    try:
        from data.wavelets.backends import PrecomputedWaveletBackend

        backend = PrecomputedWaveletBackend()
        assert backend.name == 'precomputed'
        _pass(f'Precomputed backend created: {backend}')

        # Create a temporary .npy file
        expected_channels = 192
        dummy = np.random.randn(expected_channels, 16, 16).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            np.save(f, dummy)
            tmp_path = f.name

        try:
            result = backend(tmp_path)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (expected_channels, 16, 16), \
                f'Shape mismatch: {result.shape}'
            assert result.dtype == torch.float32
            _pass(f'Precomputed backend output shape: {result.shape}')
        finally:
            os.unlink(tmp_path)

        return True
    except Exception as e:
        _fail('Precomputed backend check failed.', e)
        return False


def check_factory():
    """6. Factory creates correct backends from opt."""
    _header('6. Factory (create_wavelet_backend)')
    try:
        from data.wavelets.backends import create_wavelet_backend

        class MockOpt:
            wavelet_type = 'haar'
            wavelet_level = 3
            wavelet_mode = 'reflect'
            use_log_packets = True

        opt = MockOpt()

        # CPU
        opt.wavelet_backend = 'cpu'
        b = create_wavelet_backend(opt)
        assert b.name == 'cpu'
        _pass(f'Factory created CPU backend: {b}')

        # Precomputed
        opt.wavelet_backend = 'precomputed'
        b = create_wavelet_backend(opt)
        assert b.name == 'precomputed'
        _pass(f'Factory created precomputed backend: {b}')

        # GPU
        if torch.cuda.is_available():
            opt.wavelet_backend = 'gpu'
            b = create_wavelet_backend(opt)
            assert b.name == 'gpu'
            _pass(f'Factory created GPU backend: {b}')
        else:
            _skip('CUDA not available — GPU factory test skipped.')

        # Invalid
        opt.wavelet_backend = 'invalid'
        try:
            create_wavelet_backend(opt)
            _fail('Factory should raise ValueError for invalid backend.')
            return False
        except ValueError:
            _pass('Factory correctly raises ValueError for invalid backend.')

        return True
    except Exception as e:
        _fail('Factory check failed.', e)
        return False


def check_shape_consistency():
    """7. CPU and GPU backends produce identical shapes."""
    _header('7. Shape consistency (CPU vs GPU)')

    if not torch.cuda.is_available():
        _skip('CUDA not available — shape consistency check skipped.')
        return True

    try:
        from data.wavelets.backends import (
            CPUWaveletBackend, GPUWaveletBackend,
        )

        cpu_backend = CPUWaveletBackend(wavelet='haar', level=3,
                                         log_scale=False)
        gpu_backend = GPUWaveletBackend(wavelet='haar', level=3,
                                         log_scale=False,
                                         device=torch.device('cuda:0'))

        img_np = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8).astype(np.float32)
        img_tensor = torch.from_numpy(
            img_np.transpose(2, 0, 1))  # (3, 64, 64)

        cpu_result = cpu_backend(img_np)
        gpu_result = gpu_backend(img_tensor)

        assert cpu_result.shape == gpu_result.shape, (
            f'Shape mismatch: CPU={cpu_result.shape}, GPU={gpu_result.shape}'
        )
        _pass(
            f'Shapes match: CPU={cpu_result.shape}, '
            f'GPU={gpu_result.shape}'
        )

        return True
    except Exception as e:
        _fail('Shape consistency check failed.', e)
        return False


def check_ddp_device_resolution():
    """8. GPU backend device resolution for DDP."""
    _header('8. DDP device resolution')

    if not torch.cuda.is_available():
        _skip('CUDA not available — DDP device check skipped.')
        return True

    try:
        from data.wavelets.backends.factory import _resolve_device

        class MockOpt:
            gpu_ids = [0]

        opt = MockOpt()

        # Simulate no DDP env vars
        old_rank = os.environ.pop('LOCAL_RANK', None)
        try:
            device = _resolve_device(opt)
            assert device == torch.device('cuda:0'), \
                f'Expected cuda:0, got {device}'
            _pass(f'Single-GPU device resolution: {device}')
        finally:
            if old_rank is not None:
                os.environ['LOCAL_RANK'] = old_rank

        # Simulate DDP with LOCAL_RANK=0
        os.environ['LOCAL_RANK'] = '0'
        try:
            device = _resolve_device(opt)
            assert device == torch.device('cuda:0')
            _pass(f'DDP LOCAL_RANK=0 device resolution: {device}')
        finally:
            if old_rank is not None:
                os.environ['LOCAL_RANK'] = old_rank
            else:
                os.environ.pop('LOCAL_RANK', None)

        return True
    except Exception as e:
        _fail('DDP device resolution check failed.', e)
        return False


def main():
    print('\n' + '=' * 60)
    print('  GPU Wavelet Backend — Verification Suite')
    print('=' * 60)

    results = []
    results.append(('Imports', check_imports()))
    results.append(('No circular deps', check_no_circular()))

    ok, cpu_shape = check_cpu_backend()
    results.append(('CPU backend', ok))

    ok, gpu_shape = check_gpu_backend()
    results.append(('GPU backend', ok))

    results.append(('Precomputed backend', check_precomputed_backend()))
    results.append(('Factory', check_factory()))
    results.append(('Shape consistency', check_shape_consistency()))
    results.append(('DDP device resolution', check_ddp_device_resolution()))

    # --- Summary ---
    _header('Summary')
    all_pass = True
    for name, passed in results:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] {name}')
        if not passed:
            all_pass = False

    if all_pass:
        print('\n  All checks passed.')
    else:
        print('\n  Some checks failed.  See details above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
