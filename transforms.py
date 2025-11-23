import torch
from torchvision.transforms import v2
from typing import Dict, List, Optional, Tuple


class FixedAffine:
    """Deterministic affine transform with fixed parameters"""
    def __init__(self, dx=3, dy=3):
        self.dx = dx
        self.dy = dy
    
    def __call__(self, img):
        return v2.functional.affine(img, angle=0, translate=(self.dx, self.dy), scale=1.0, shear=0)

def get_transforms(config: Dict, dataset_name: str, mean: List[float], std: List[float], train: bool = True, model_name: str = None) -> Tuple[v2.Transform, Optional[List[v2.Transform]]]:
    transforms_config = config.get('transforms', {})
    
    if dataset_name.lower() == 'mnist':
        dataset_img_size = 28
        in_channels = 1
    elif 'cifar' in dataset_name.lower():
        dataset_img_size = 32
        in_channels = 3
    else:
        dataset_img_size = 224
        in_channels = 3
    
    if model_name and model_name.lower() == 'mlp':
        required_img_size = 224
    else:
        required_img_size = dataset_img_size
    
    img_size = required_img_size
    needs_resize = required_img_size != dataset_img_size
    
    normalize = v2.Normalize(mean, std, inplace=True)
    
    if train:
        train_config = transforms_config.get('train', {})
        transform_list = []
        
        if train_config.get('random_resized_crop', {}).get('enabled', False):
            crop_config = train_config['random_resized_crop']
            size = tuple(crop_config.get('size', [img_size, img_size]))
            scale = tuple(crop_config.get('scale', [0.7, 1.0]))
            p = crop_config.get('p', 0.4)
            transform_list.append(
                v2.RandomApply([v2.RandomResizedCrop(size=size, scale=scale)], p=p)
            )
        
        if train_config.get('random_translate', {}).get('enabled', False):
            translate_config = train_config['random_translate']
            translate = tuple(translate_config.get('translate', [0.1, 0.2]))
            p = translate_config.get('p', 0.4)
            transform_list.append(
                v2.RandomApply([v2.RandomAffine(degrees=0, translate=translate)], p=p)
            )
        
        if train_config.get('random_rotation', {}).get('enabled', False):
            rotation_config = train_config['random_rotation']
            degrees = rotation_config.get('degrees', 15)
            p = rotation_config.get('p', 0.4)
            transform_list.append(
                v2.RandomApply([v2.RandomRotation(degrees)], p=p)
            )
        
        if train_config.get('random_horizontal_flip', {}).get('enabled', False):
            flip_config = train_config['random_horizontal_flip']
            p = flip_config.get('p', 0.4)
            transform_list.append(v2.RandomHorizontalFlip(p=p))
        
        if train_config.get('random_vertical_flip', {}).get('enabled', False):
            flip_config = train_config['random_vertical_flip']
            p = flip_config.get('p', 0.4)
            transform_list.append(v2.RandomVerticalFlip(p=p))
        
        if needs_resize:
            transform_list.append(v2.Resize((img_size, img_size), antialias=True))
        
        train_transforms = v2.Compose(transform_list)
        
        return train_transforms
    
    else:
        test_config = transforms_config.get('test', {})
        tta_transforms_list = []
        
        base_tta_transforms = []
        if needs_resize:
            base_tta_transforms.append(v2.Resize((img_size, img_size), antialias=True))
        base_tta_transforms.append(normalize)
        
        tta_transforms_list.append(v2.Compose(base_tta_transforms))
        
        if test_config.get('resized_crop', {}).get('enabled', False):
            crop_config = test_config['resized_crop']
            crop_size = crop_config.get('crop_size', dataset_img_size)
            size = tuple(crop_config.get('size', [img_size, img_size]))
            tta_transform = [v2.CenterCrop(size=crop_size), v2.Resize(size=size, antialias=True)]
            if needs_resize:
                tta_transform.append(v2.Resize((img_size, img_size), antialias=True))
            tta_transform.append(normalize)
            tta_transforms_list.append(v2.Compose(tta_transform))
        
        if test_config.get('fixed_affine', {}).get('enabled', False):
            affine_config = test_config['fixed_affine']
            translations = affine_config.get('translations', [[3, 3], [-3, -3]])
            for dx, dy in translations:
                tta_transform = [FixedAffine(dx=dx, dy=dy)]
                if needs_resize:
                    tta_transform.append(v2.Resize((img_size, img_size), antialias=True))
                tta_transform.append(normalize)
                tta_transforms_list.append(v2.Compose(tta_transform))
        
        if test_config.get('random_rotation', {}).get('enabled', False):
            rotation_config = test_config['random_rotation']
            degrees = rotation_config.get('degrees', 15)
            tta_transform = [v2.RandomRotation(degrees)]
            if needs_resize:
                tta_transform.append(v2.Resize((img_size, img_size), antialias=True))
            tta_transform.append(normalize)
            tta_transforms_list.append(v2.Compose(tta_transform))
        
        if test_config.get('random_horizontal_flip', {}).get('enabled', False):
            tta_transform = [v2.RandomHorizontalFlip(p=1.0)]
            if needs_resize:
                tta_transform.append(v2.Resize((img_size, img_size), antialias=True))
            tta_transform.append(normalize)
            tta_transforms_list.append(v2.Compose(tta_transform))
        
        if test_config.get('random_vertical_flip', {}).get('enabled', False):
            tta_transform = [v2.RandomVerticalFlip(p=1.0)]
            if needs_resize:
                tta_transform.append(v2.Resize((img_size, img_size), antialias=True))
            tta_transform.append(normalize)
            tta_transforms_list.append(v2.Compose(tta_transform))

        return tta_transforms_list


def get_cutmix_mixup(config: Dict, num_classes: int, epoch: int = 0) -> Optional[Tuple[v2.Transform, float]]:
    transforms_config = config.get('transforms', {})
    train_config = transforms_config.get('train', {})
    cutmix_config = train_config.get('cutmix_mixup', {})
    
    if not cutmix_config.get('enabled', False):
        return None
    
    target_alpha = cutmix_config.get('alpha', 0.75)
    max_ramp_epochs = cutmix_config.get('max_ramp_epochs', 25)
    if epoch < max_ramp_epochs:
        alpha = target_alpha * (epoch / max_ramp_epochs)
    else:
        alpha = target_alpha

    cutmix = v2.CutMix(num_classes=num_classes, alpha=alpha)
    mixup = v2.MixUp(num_classes=num_classes, alpha=alpha)
    
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    p = cutmix_config.get('p', 0.5)
    return cutmix_or_mixup, p

