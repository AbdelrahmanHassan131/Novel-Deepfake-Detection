# STATUS: NOT RUN (runtime tests deferred; static syntax check only)
"""Windows/spawn regression: dataset transforms must survive worker serialization."""
import pickle
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data.datasets.fusion_dataset import FusionDataset
from data.datasets.rgb_dataset import RGBDataset
from data.datasets.wavelet_dataset import WaveletDataset
from data.datasets.wavelet_rgb_dataset import WaveletRGBDataset


@pytest.mark.parametrize('dataset_class', [FusionDataset, RGBDataset, WaveletDataset, WaveletRGBDataset])
@pytest.mark.parametrize('no_spatial_transforms', [False, True])
def test_spawn_matches_single_process(tmp_path, dataset_class, no_spatial_transforms):
    for name, color in [('real', (20, 80, 150)), ('fake', (190, 30, 90))]:
        folder = tmp_path / name
        folder.mkdir()
        Image.new('RGB', (40, 40), color).save(folder / 'sample.png')
    opt = SimpleNamespace(
        dataroot=str(tmp_path), isTrain=False, manifest=None,
        no_crop=no_spatial_transforms, no_resize=no_spatial_transforms,
        no_flip=True, cropSize=32, loadSize=36, rz_interp=['bilinear'],
        wavelet_backend='cpu', wavelet_type='haar', wavelet_level=1,
        wavelet_mode='reflect', wavelet_log_mode='signed_log1p',
        use_log_packets=True, precomputed_dir=None,
    )
    dataset = dataset_class(opt, str(tmp_path))
    # This alone reproduces the original lambda failure, without spawning.
    restored = pickle.loads(pickle.dumps(dataset))
    expected = list(DataLoader(dataset, batch_size=2, num_workers=0))
    actual = list(DataLoader(restored, batch_size=2, num_workers=1,
                             multiprocessing_context='spawn', timeout=60))
    assert len(actual) == len(expected) == 1
    assert len(actual[0]) == len(expected[0])
    for received, reference in zip(actual[0], expected[0]):
        torch.testing.assert_close(received, reference, rtol=0, atol=0)
