from pathlib import Path
import pickle
import os
import numpy as np
from typing import Callable
import torch
from torchvision import transforms


class Normalize:
    """ Basic class for spectrum normalization
    """
    def __init__(self, norm_output, **kwargs):
        project_path = Path(__file__).resolve().parents[1].parent
        filename = os.path.join(project_path, 'inverse_problem/nn_inversion/spectrumRanges.pickle')
        with open(filename, 'rb') as handle:
            spectrum_dict = pickle.load(handle)
        self.spectrum_dict = spectrum_dict
        self.norm_output = norm_output
        if norm_output:
            kwdefaults = {'mode': 'range', 'logB': True}
            for kw in kwdefaults.keys():
                kwargs.setdefault(kw, kwdefaults[kw])
        self.kwargs = kwargs

    def __call__(self, sample):
        params = sample['Y']
        if self.norm_output:
            params = normalize_output(params, mode=self.kwargs['mode'], logB=self.kwargs['logB'])
        return {'X': sample['X'],
                'Y': params}


class CenteredMean(Normalize):
    """ Extracts mean from each component"""

    def __init__(self, norm_output=True, **kwargs):
        super().__init__(norm_output, **kwargs)
        self.mean = self.spectrum_dict['mean']

    def __call__(self, sample):
        sample = super().__call__(sample)
        (spectrum, cont), params = sample['X'], sample['Y']
        spectrum_normalized = spectrum.flatten(order='F') - self.mean

        return {'X': (spectrum_normalized.reshape((56, 4), order='F'), cont),
                'Y': params}


class NormalizeStandard(Normalize):
    """ Normalize each component to mean and standard deviation"""

    def __init__(self, norm_output=True, **kwargs):
        super().__init__(norm_output, **kwargs)
        self.mean = self.spectrum_dict['mean']
        self.std = self.spectrum_dict['std']
        self.mean_cont = self.spectrum_dict['cont_mean']
        self.std_cont = self.spectrum_dict['cont_std']

    def __call__(self, sample):
        sample = super().__call__(sample)
        (spectrum, cont), params = sample['X'], sample['Y']
        spectrum = (spectrum.flatten(order='F') - self.mean) / self.std
        cont = (cont - self.mean_cont) / self.std_cont
        return {'X': (spectrum.reshape((56, 4), order='F'), cont),
                'Y': params}


class Rescale(Normalize):
    """ Multiply each spectrum component by factor, preserve spectrum shape"""

    def __init__(self, factors=None, cont_scale=None, norm_output=True, **kwargs):
        super().__init__(norm_output, **kwargs)

        if factors is None:
            self.factors = [1, 1000, 1000, 1000]
            # todo чему равно cont_scale если не задано?
            self.cont_scale = cont_scale if cont_scale is not None else 40000
            self.norm_output = norm_output
        else:
            self.factors = factors
            self.cont_scale = cont_scale if cont_scale is not None else 40000
            self.norm_output = norm_output

    def __call__(self, sample):
        # output normalization
        sample = super().__call__(sample)

        (spectrum, cont), params = sample['X'], sample['Y']
        spectrum = spectrum * np.array(self.factors).reshape((1, 4))

        return {'X': (spectrum, cont / self.cont_scale),
                'Y': params}


def normalize_output(y, mode='norm', logB=True, **kwargs):
    norm_y = y.copy()
    allowedmodes = {'norm': ['mean', 'std'],
                    'range': ['max', 'min']
                    }
    kwdefaults = {'mean': [530, 91, 89, 33, 0.31, 12, 27083, 19567, 0.04, 0.5, 0.36],
                  'std': [565, 36.4, 52.6, 9.5, 0.21, 11.82, 4112, 5927, 0.04, 0.5, 0.36],
                  'max': [5000, 180, 180, 90, 1.5, 100, 38603, 60464, 10, 1, 10],
                  'min': [0, 0, 0, 20, 0, 0.01, 0, 0, -10, 0, -10]}
    if logB:
        norm_y[0] = np.log1p(y[0])
        kwdefaults['mean'][0] = 5.67
        kwdefaults['std'][0] = 1.16
        kwdefaults['max'][0] = 8.51
        kwdefaults['min'][0] = 0
    if mode not in allowedmodes.keys():
        raise ValueError('mode should be norm or range')
    for key in kwargs:
        if key not in allowedmodes[mode]:
            raise ValueError('%s keyword not in allowed keywords %s' % (key, allowedmodes[mode]))
    for kw in allowedmodes[mode]:
        kwargs.setdefault(kw, kwdefaults[kw])
    if mode == 'norm':
        norm_y = (np.array(norm_y).reshape(1, -1) -
                  np.array(kwargs['mean']).reshape(1, -1)) / np.std(np.array(kwargs['std']).reshape(1, -1))
    if mode == 'range':
        range_ = np.array(kwargs['max']).reshape(1, -1) - np.array(kwargs['min']).reshape(1, -1)
        norm_y = (np.array(norm_y).reshape(1, -1) - np.array(kwargs['min']).reshape(1, -1)) / range_

    return norm_y


class FlattenSpectrum:
    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']
        spectrum = spectrum.flatten(order='F')

        return {'X': (spectrum, cont),
                'Y': params}


class ToTensor(object):
    """Convert np arrays intoTensors."""

    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']

        return {'X': (torch.from_numpy(spectrum).float(), torch.FloatTensor([cont])),
                'Y': torch.from_numpy(params.astype(np.float32)).flatten()}


class ToConcatMlp(object):
    """Prepare X and Y for mlp."""

    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']
        return {'X': torch.cat((spectrum, cont)),
                'Y': params}


def mlp_transform_standard(**kwargs) -> Callable:
    allowed_kwargs = {'logB', 'norm_output', 'mode'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise ValueError(f'{key} not in allowed keywords: logB, norm_output, mode')
    norm = NormalizeStandard(**kwargs)
    flat = FlattenSpectrum()
    to_tensor = ToTensor()
    to_mlp = ToConcatMlp()
    return transforms.Compose([norm, flat, to_tensor, to_mlp])


def mlp_transform_rescale(**kwargs) -> Callable:
    allowed_kwargs = {'factors', 'cont_scale', 'norm_output', 'logB', 'mode'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise ЛунError(f'{key} not in allowed keywords: factor, cont_scale')

    rescale = Rescale(**kwargs)
    flat = FlattenSpectrum()
    to_tensor = ToTensor()
    return transforms.Compose([rescale, flat, to_tensor])


class ToConv1d(object):
    """Prepare X and y for conv 1d model"""

    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']
        spectrum = np.swapaxes(spectrum, 0, 1)
        return {'X': (spectrum, cont),
                'Y': params}


def conv1d_transform_rescale(**kwargs) -> Callable:
    allowed_kwargs = {'factors', 'cont_scale', 'norm_output', 'logB', 'mode'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise KeyError(f'{key} not in allowed keywords: factor, cont_scale')

    rescale = Rescale(**kwargs)
    to_conv = ToConv1d()
    to_tensor = ToTensor()
    return transforms.Compose([rescale, to_conv, to_tensor])


def conv1d_transform_standard(**kwargs) -> Callable:
    allowed_kwargs = {'logB', 'norm_output', 'mode'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise KeyError(f'{key} not in allowed keywords: factor, cont_scale')
    norm = NormalizeStandard(**kwargs)
    to_tensor = ToTensor()
    to_conv = ToConv1d()
    return transforms.Compose([norm, to_tensor, to_conv])
