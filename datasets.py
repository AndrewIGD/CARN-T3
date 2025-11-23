from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torch
from transforms import get_transforms as get_transforms_from_config


def get_base_transforms():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def load_mnist(root='./data', batch_size=64, train=True):
    base_transform = get_base_transforms()
    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=base_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)


def load_cifar10(root='./data', batch_size=64, train=True):
    base_transform = get_base_transforms()
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=base_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)


def load_cifar100(root='./data', batch_size=64, train=True):
    base_transform = get_base_transforms()
    dataset = datasets.CIFAR100(
        root=root,
        train=train,
        download=True,
        transform=base_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)


def load_oxford_iiit_pet(root='./data', batch_size=64, split='trainval'):
    base_transform = get_base_transforms()
    dataset = datasets.OxfordIIITPet(
        root=root,
        split=split,
        target_types='category',
        download=True,
        transform=base_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'trainval'), num_workers=2, pin_memory=True)


def load_dataset(dataset_name, batch_size=64):
    root = './resources/datasets'
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        train_loader = load_mnist(root=root, train=True, batch_size=batch_size)
        test_loader = load_mnist(root=root, train=False, batch_size=batch_size)
        mean = [0.1307]
        std = [0.3081]
        return train_loader, test_loader, 10, mean, std
    elif dataset_name == 'cifar10':
        train_loader = load_cifar10(root=root, train=True, batch_size=batch_size)
        test_loader = load_cifar10(root=root, train=False, batch_size=batch_size)
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        return train_loader, test_loader, 10, mean, std
    elif dataset_name == 'cifar100':
        train_loader = load_cifar100(root=root, train=True, batch_size=batch_size)
        test_loader = load_cifar100(root=root, train=False, batch_size=batch_size)
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        return train_loader, test_loader, 100, mean, std
    elif dataset_name == 'oxford_iiit_pet' or dataset_name == 'oxfordiiitpet':
        train_loader = load_oxford_iiit_pet(
            root=root,
            split='trainval',
            batch_size=batch_size,
        )
        test_loader = load_oxford_iiit_pet(
            root=root,
            split='test',
            batch_size=batch_size,
        )
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return train_loader, test_loader, 37, mean, std
    
    return None, None, None, None, None