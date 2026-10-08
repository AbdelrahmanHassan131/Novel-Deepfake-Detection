from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from data.manifest import read_manifest

class BaseDataset(datasets.ImageFolder):
    """
    Base class for all future datasets.
    Provides shared dataset initialization logic.
    """
    def __init__(self, opt, root, transform=None):
        self.opt = opt
        manifest = getattr(opt, 'manifest', None)
        if manifest:
            VisionDataset.__init__(self, root, transform=transform)
            cached_records = getattr(opt, '_manifest_records', None)
            self.records = (cached_records if cached_records is not None else
                            read_manifest(manifest, root, getattr(opt, 'manifest_split', None),
                                          progress_every=10000))
            self.samples = [(r['path'], r['label']) for r in self.records]
            self.targets = [label for _, label in self.samples]
            self.imgs = self.samples
            self.loader = default_loader
            self.classes = ['real', 'fake']
            self.class_to_idx = {'real': 0, 'fake': 1}
        else:
            super().__init__(root, transform=transform)
            # ImageFolder alphabetizes directory names (``fake``, ``real``),
            # while find_classes deliberately maps them to canonical labels.
            # Keep visualization/report labels in that same canonical order.
            self.classes = ['real', 'fake']
            self.class_to_idx = {'real': 0, 'fake': 1}
            self.records = [dict(sample_id=str(i), path=p, label=y, split='unknown',
                                 dataset_source='unknown', group_id='unknown')
                            for i, (p, y) in enumerate(self.samples)]

    def find_classes(self, directory):
        classes, _ = super().find_classes(directory)
        aliases = {'real': 0, 'fake': 1, '0_real': 0, '1_fake': 1}
        if set(classes) not in ({'real', 'fake'}, {'0_real', '1_fake'}):
            raise ValueError('Binary root must contain real/fake or 0_real/1_fake. '
                             'Use an explicit manifest for other layouts; generator folders are not labels.')
        return classes, {name: aliases[name] for name in classes}
