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
    rz_interp = getattr(opt, 'rz_interp', ['bilinear'])
    if isinstance(rz_interp, str):
        rz_interp = [x.strip() for x in rz_interp.split(',') if x.strip()]
    interp = sample_discrete(rz_interp) if getattr(opt, 'isTrain', False) else rz_interp[0]
    interp_flag = rz_dict.get(interp, Image.BILINEAR)
    load_size = getattr(opt, 'loadSize', getattr(opt, 'image_size', 256))
    return TF.resize(img, load_size, interpolation=interp_flag)
