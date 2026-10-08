from functools import partial
from torchvision import transforms
from .base_dataset import BaseDataset
from ..transforms.augmentations import data_augment, resolve_augmentation_recipe, validate_augmentation_params
from ..transforms.resize import custom_resize
from ..transforms.identity import identity_image


class RGBDataset(BaseDataset):
    """
    Standard RGB Dataset.
    Returns: (image_tensor, label)
    Represents behavior used by: Wang2020Raw, Wang2020_128, XceptionRaw.
    """
    def __init__(self, opt, root):
        # Validate parameters (resolved once upstream in config/preflight without mutating shared opt)
        validate_augmentation_params(opt)

        crop_policy = getattr(opt, 'crop_policy', 'scale_and_crop')
        crop_size = getattr(opt, 'cropSize', getattr(opt, 'crop_size', 224))

        if opt.isTrain:
            if crop_policy == 'random_resized_crop':
                crop_func = transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0))
            elif crop_policy == 'patch_crop':
                crop_func = transforms.RandomCrop(crop_size, pad_if_needed=True)
            else:
                crop_func = transforms.RandomCrop(crop_size)
        elif getattr(opt, 'no_crop', False):
            crop_func = transforms.Lambda(identity_image)
        else:
            crop_func = transforms.CenterCrop(crop_size)

        if opt.isTrain and not getattr(opt, 'no_flip', False):
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(identity_image)

        if not opt.isTrain and getattr(opt, 'no_resize', False):
            rz_func = transforms.Lambda(identity_image)
        elif opt.isTrain and crop_policy == 'random_resized_crop':
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
