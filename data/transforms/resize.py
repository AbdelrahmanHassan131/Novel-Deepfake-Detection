import torchvision.transforms.functional as TF
from PIL import Image
from .augmentations import sample_discrete

rz_dict = {
    'bilinear': Image.BILINEAR,
    'bicubic': Image.BICUBIC,
    'lanczos': Image.LANCZOS,
    'nearest': Image.NEAREST
}

def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
