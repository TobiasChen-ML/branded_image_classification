import os
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, val_tf


def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cuda",
) -> Tuple[Dict[str, DataLoader], Dict[str, int], list]:
    train_tf, val_tf = build_transforms(img_size)

    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")

    pin_memory = device == "cuda"

    if os.path.isdir(train_root):
        train_ds = datasets.ImageFolder(train_root, transform=train_tf)
        if os.path.isdir(val_root):
            val_ds = datasets.ImageFolder(val_root, transform=val_tf)
            class_names = train_ds.classes
        else:
            full_ds = datasets.ImageFolder(train_root, transform=val_tf)
            val_len = max(1, int(0.2 * len(full_ds)))
            train_len = len(full_ds) - val_len
            train_ds, val_ds = random_split(full_ds, [train_len, val_len])
            # 重新为训练集应用训练增广
            train_ds.dataset.transform = train_tf
            class_names = full_ds.classes
    else:
        # 直接用 data_dir 作为根目录
        full_ds = datasets.ImageFolder(data_dir, transform=val_tf)
        val_len = max(1, int(0.2 * len(full_ds)))
        train_len = len(full_ds) - val_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])
        train_ds.dataset.transform = train_tf
        class_names = full_ds.classes

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }
    dataset_sizes = {"train": len(train_ds), "val": len(val_ds)}

    return dataloaders, dataset_sizes, class_names