import os
import torch
import torchvision

# Define a constant for the data directory based on the script's location
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_data")

NUM_CLASSES = 10


def get_train_dataloader(
    img_size: int = 32, batch_size: int = 64, num_workers: int = 4
) -> torch.utils.data.DataLoader:
    assert img_size == 32, "CIFAR10 images are of size 32x32"
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_dataloader


def get_test_dataloader(
    img_size: int = 32, batch_size: int = 32, num_workers: int = 4
) -> torch.utils.data.DataLoader:
    assert img_size == 32, "CIFAR10 images are of size 32x32"
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader
