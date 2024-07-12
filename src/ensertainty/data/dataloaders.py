from .datasets import MNIST, FMNIST, CIFAR10, CIFAR100, ImgNette, CELEBA
from .utils import get_mean_and_std, numpy_collate_fn
from typing import Literal
from torch.utils import data
import torch
from omegaconf import DictConfig
import logging

DATASETS = {"MNIST": MNIST, "FMNIST": FMNIST, "CIFAR10": CIFAR10, "CIFAR100": CIFAR100, "ImgNette": ImgNette, "CELEBA": CELEBA}


def get_dataloaders(cfg: DictConfig, seed: int, inference_mode=False):
    dataset = cfg["dataset_name"]
    bs = cfg["batch_size"]
    data_path = cfg["data_path"]
    val_frac = cfg["val_frac"]
    n_samples = None

    cls = (
        cfg.get("classes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if cfg.get("classes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) is not None
        else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    train_stats = None

    if not dataset in ["MNIST", "FMNIST", "CELEBA"]:
        train_stats = get_mean_and_std(
            data_train=DATASETS[dataset](path_root=data_path, set_purp="train", n_samples=n_samples, cls=cls),
            val_frac=val_frac,
            seed=seed,
        )
        logging.info(f"👉 Normalizing with mean = {train_stats['mean']} and  std = {train_stats['std']} ")

    data_train = DATASETS[dataset](
        path_root=data_path, set_purp="train", n_samples=n_samples, cls=cls, normalizing_stats=train_stats
    )
    data_val = DATASETS[dataset](
        path_root=data_path, set_purp="valid" if dataset == "CELEBA" else "val", n_samples=n_samples, cls=cls, normalizing_stats=train_stats
    )
    data_test = DATASETS[dataset](
        path_root=data_path, set_purp="test", n_samples=None, cls=cls, normalizing_stats=train_stats
    )

    len_val = int(len(data_train) * val_frac)
    len_train = len(data_train) - len_val

    if not dataset == "CELEBA":
        data_train, _ = data.random_split(data_train, [len_train, len_val], generator=torch.Generator().manual_seed(seed))
        _, data_val = data.random_split(data_val, [len_train, len_val], generator=torch.Generator().manual_seed(seed))

    train_loader = data.DataLoader(
        data_train,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        collate_fn=numpy_collate_fn,
        num_workers=8,
        persistent_workers=True,
    )

    val_loader = data.DataLoader(
        data_val,
        batch_size=bs,
        shuffle=False,
        drop_last=True,
        collate_fn=numpy_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    test_loader = data.DataLoader(
        data_test,
        batch_size=bs,
        shuffle=inference_mode,
        drop_last=True,
        collate_fn=numpy_collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    if inference_mode:
        return test_loader, data_test

    return train_loader, val_loader, test_loader
