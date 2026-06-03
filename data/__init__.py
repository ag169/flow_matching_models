import torch

from . import cifar10


def get_num_classes_from_dataset(dataset: str) -> int:
    dataset = dataset.lower()
    if dataset == "cifar10":
        num_classes = cifar10.NUM_CLASSES
    else:
        raise ValueError("Invalid dataset!")

    return num_classes


def get_train_dataloader(
    dataset: str, batch_size: int = 64, num_workers: int = 4
) -> torch.utils.data.DataLoader:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return cifar10.get_train_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
    else:
        raise ValueError("Invalid dataset!")


def get_test_dataloader(
    dataset: str, batch_size: int = 32, num_workers: int = 4
) -> torch.utils.data.DataLoader:
    dataset = dataset.lower()
    if dataset == "cifar10":
        return cifar10.get_test_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
    else:
        raise ValueError("Invalid dataset!")
