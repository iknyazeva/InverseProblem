from pathlib import Path
from inverse_problem.milne_edington import HinodeME, BatchHinodeME
import numpy as np
from inverse_problem.milne_edington.data_utils import download_from_google_disc
from typing import Callable
from torch.utils.data import Dataset
from astropy.io import fits


class SpectrumDataset(Dataset):
    """
    Args:
        source (str, optional) = source of data, database or fits refer from Hinode
        download (bool): if download, for downloading parameter base use id  =
        refer = '1sRgNdX3Fzv4rtrU5TmcUwKxZ-0mniBXi'
        parameter base - '1PjseALs0r0W0ILrkMNMKw9nOIIObzM4q'
    """

    # todo update for download option
    def __init__(self, data_arr=None, param_path: Path = None,
                 source='database', transform: Callable = None, download: bool = False,
                 ff: bool = True, noise: bool = True):
        """

        Args:
            param_path (Path): path to file
            source (str):
            transform (Callable): transforms from
            download (bool):
            ff (): whether to use filling factor
            noise (): whether to add noise
        """
        if data_arr is None and param_path is None:
            raise AssertionError('you need provide data or path to data as a string')

        self.param_path = param_path
        self.source = source
        self.download = download
        self.noise = noise
        self.ff = ff
        if self.download:
            fileid = '1PjseALs0r0W0ILrkMNMKw9nOIIObzM4q'
            download_from_google_disc(fileid=fileid, dest=str(self.param_path))
        self.transform = transform
        if data_arr is not None:
            self.param_source = data_arr
        else:
            self._init_dataset()

    def __len__(self):
        if self.source == 'database':
            param_len = self.param_source.shape[0]
        else:
            param_len = self.param_source[1].data.size

        return param_len

    def __getitem__(self, item):
        if self.source == 'database':
            obj = HinodeME.from_parameters_base(item, self.param_source)
        else:
            row_id = item // self.param_source[1].data.shape[0]
            col_id = item % self.param_source[1].data.shape[1]
            obj = HinodeME.from_refer(row_id, col_id, self.param_source)

        spectrum = obj.compute_spectrum(with_ff=self.ff, with_noise=self.noise)[0]
        sample = {'X': (spectrum, obj.cont), 'Y': obj.param_vector}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _init_dataset(self):
        """
        Opens parameters source
        Raises: AssertionError if source is undefined

        """
        if self.source == 'database':
            self.param_source = fits.open(self.param_path)[0].data
        elif self.source == 'refer':
            self.param_source = fits.open(self.param_path)
        else:
            raise AssertionError('source parameter should be \'database\' or \'refer\'')


class PregenSpectrumDataset(Dataset):

    """
    Generate spectrum by the full array of dataset at first, and iterate through
    """

    def __init__(self, data_arr=None, param_path: Path = None,
                 source='database', transform: Callable = None,
                 ff: bool = True, noise: bool = True):

        if data_arr is None and param_path is None:
            raise AssertionError('you need to provide data or path to data')

        self.param_path = param_path
        self.source = source
        self.noise = noise
        self.ff = ff
        self.transform = transform
        if data_arr is not None:
            param_source = data_arr
        elif source == 'database':
            param_source = fits.open(self.param_path)[0].data
        elif self.source == 'refer':
            param_list = [1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13]
            refer = fits.open(self.param_path)
            param_source = np.array([refer[i].data for i in param_list], dtype='float').swapaxes(0, 2).swapaxes(0, 1).reshape(-1, 11)
        else:
            raise AssertionError('source parameter should be data or from \'database\' or \'refer\'')
        self.samples = self._init_dataset(param_source)

    def _init_dataset(self, param_source):
        obj = BatchHinodeME(param_source)
        spectrum = obj.compute_spectrum(with_ff=self.ff, with_noise=self.noise)
        samples = {'X': (spectrum, obj.cont), 'Y': obj.param_vector}

        if self.transform:
            samples = self.transform(samples)
        return samples

    def __getitem__(self, item):
        sample = {'X': (self.samples['X'][0][item, :], self.samples['X'][1][item]),
                  'Y': self.samples['Y'][item, :]}

        return sample

    def __len__(self):
        param_len = self.samples['Y'].shape[0]

        return param_len
