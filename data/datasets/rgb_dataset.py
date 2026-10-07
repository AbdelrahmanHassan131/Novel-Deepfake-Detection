from functools import partial
from torchvision import transforms
from .base_dataset import BaseDataset
from ..transforms.augmentations import data_augment
from ..transforms.resize import custom_resize
from ..transforms.identity import identity_image

class RGBDataset(BaseDataset):
    """
    Standard RGB Dataset.
    Returns: (image_tensor, label)
    Represents behavior used by: Wang2020Raw, Wang2020_128, XceptionRaw.
    """
    def __init__(self, opt, root):
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(identity_image)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(identity_image)

        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(identity_image)
        else:
            rz_func = transforms.Lambda(partial(custom_resize, opt=opt))

        image_transform = transforms.Compose([
            rz_func,
            transforms.Lambda(partial(data_augment, opt=opt)),
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        super().__init__(opt, root, transform=image_transform)

    # Uses standard __getitem__ from ImageFolder
