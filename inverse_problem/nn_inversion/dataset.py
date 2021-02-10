from pathlib import Path
from inverse_problem.milne_edington import HinodeME
import numpy as np
from typing import Callable, List
from torch.utils.data import Dataset
from astropy.io import fits


class SpectrumDataset(Dataset):
    """
    Args:
        source (str, optional) = source of data, database or fits refer from Hinode
    """

    # todo noise generation
    def __init__(self, param_path: Path,
                 source='database', transform: Callable = None):

        self.param_path = param_path
        self.source = source
        self.transform = transform
        self._init_dataset()

    def __len__(self):
        if self.source == 'database':
            param_len = self.param_source.shape[0]
        else:
            param_len = self.param_source.data.size

        return param_len

    def __getitem__(self, item):
        if self.source == 'database':
            obj = HinodeME.from_parameters_base(item, self.param_source)
        else:
            row_id = item // self.param_source[1].data.shape[0]
            col_id = item % self.param_source[1].data.shape[1]
            obj = HinodeME.from_refer(row_id, col_id, self.param_source)
        spectrum = obj.compute_spectrum()
        sample = {'X': (spectrum, obj.cont), 'Y': obj.param_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _init_dataset(self):
        if self.source == 'database':
            self.param_source = fits.open(self.param_path)[0].data
        elif self.source == 'refer':
            self.param_source = fits.open(self.param_path)
        else:
            raise AssertionError('source parameter should be \'database\' or \'refer\'')
