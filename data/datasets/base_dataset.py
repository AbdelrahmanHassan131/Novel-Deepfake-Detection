from torchvision import datasets

class BaseDataset(datasets.ImageFolder):
    """
    Base class for all future datasets.
    Provides shared dataset initialization logic.
    """
    def __init__(self, opt, root, transform=None):
        self.opt = opt
        super().__init__(root, transform=transform)
