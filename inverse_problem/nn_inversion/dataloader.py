from inverse_problem.nn_inversion.dataset import SpectrumDataset, PregenSpectrumDataset
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
from torch import nn
import os
from pathlib import Path
from torch.utils.data import DataLoader
from inverse_problem.nn_inversion import transforms


def make_loader(data_arr=None, filename: Path = None, pregen=False, ff=True, noise=True,
                val_split=0.1, source="database", transform_name="mlp_transform_rescale", batch_size=64) -> DataLoader:

    """

    Args:
        data_arr:
        filename:
        pregen:
        ff:
        noise:
        val_split:

    Returns:
        train and val data loader
    """

    if data_arr is None and filename is None:
        raise AssertionError('you need provide data as array or path to data')
    transform = getattr(transforms, transform_name)()

    if pregen:
        transformed_dataset = PregenSpectrumDataset(data_arr=data_arr, param_path=filename, source=source,
                                                    transform=transform, ff=ff, noise=noise)
    else:
        transformed_dataset = SpectrumDataset(data_arr=data_arr, param_path=filename, source=source,
                                              transform=transform, ff=ff, noise=noise)
    train_idx, val_idx = train_test_split(list(range(len(transformed_dataset))), test_size=val_split)
    train_dataset = Subset(transformed_dataset, train_idx)
    val_dataset = Subset(transformed_dataset, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader