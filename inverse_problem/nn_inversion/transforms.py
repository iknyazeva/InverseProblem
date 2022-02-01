import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torchvision import transforms


class Normalize:
    """
    Basic class for spectrum normalization
    """

    def __init__(self, norm_output, **kwargs):
        project_path = Path(__file__).resolve().parents[1].parent
        filename = os.path.join(project_path, 'inverse_problem/nn_inversion/spectrumRanges.pickle')
        with open(filename, 'rb') as handle:
            spectrum_dict = pickle.load(handle)
        self.spectrum_dict = spectrum_dict
        self.norm_output = norm_output

        if norm_output:
            kw_defaults = {'mode': 'range', 'logB': True, 'angle_transformation': False}
            for kw in kw_defaults.keys():
                kwargs.setdefault(kw, kw_defaults[kw])
        self.kwargs = kwargs

    def __call__(self, sample):
        params = sample['Y']
        if self.norm_output:
            params = normalize_output(params,
                                      mode=self.kwargs['mode'],
                                      logB=self.kwargs['logB'],
                                      angle_transformation=self.kwargs['angle_transformation'])
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

    def __init__(self, norm_output=False, **kwargs):
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
        """

        Args:
            factors (list of float): factors to multitply each spectrum component for increasing input of QUV relative to I
            cont_scale (float): scaler factor for continum intensity, default 40000
            norm_output (bool):
            angle_transformation:
            **kwargs:
        """
        super().__init__(norm_output, **kwargs)

        if factors is None:
            self.factors = [1, 1000, 1000, 1000]
        else:
            self.factors = factors
        self.cont_scale = cont_scale if cont_scale is not None else 40000
        self.norm_output = norm_output

    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']

        # data rescaling
        spectrum = normalize_spectrum(spectrum, factors=self.factors)
        cont = cont / self.cont_scale
        # output normalization
        sample = super().__call__({'X': (spectrum, cont),
                                   'Y': params})
        return sample


def normalize_output(y, mode='range', logB=True, angle_transformation=False, **kwargs):
    """
    Function for output
    Args:
        y (list of float or numpy array): vector with 11 parameters
        mode (str): type of rescaling, range (rescale to min max) or norm (rescale to mean and standard deviation)
        logB (bool): apply logarithmic transform to B
        angle_transformation (bool): angle transformation for inclination and azimuth (parameters with index 1 and 2)
        **kwargs:

    Returns:
    """
    norm_y = np.array(y).reshape(-1, 11).copy()
    allowedmodes = {'norm': ['mean', 'std'],
                    'range': ['max', 'min']}

    def sine_degree(x):
        return np.sin(x * np.pi / 180)

    kw_defaults = {
        'mean': [530, 91, 89, 33, 0.31, 12, 27083, 19567, 0.04, 0.5, 0.36],
        'std': [565, 36.4, 52.6, 9.5, 0.21, 11.82, 4112, 5927, 0.04, 0.5, 0.36],
        'max': [5000, 180, 180, 90, 1.5, 100, 38603,
                60464, 10, 1, 10],
        'min': [0, 0, 0, 20, 0, 0.01, 0, 0, -10, 0, -10]
    }

    if logB:
        norm_y[:, 0] = np.log1p(norm_y[:, 0])
        kw_defaults['mean'][0] = 5.67
        kw_defaults['std'][0] = 1.16
        kw_defaults['max'][0] = 8.51
        kw_defaults['min'][0] = 0

    if angle_transformation:
        norm_y[:, 1:3] -= 90
        norm_y[:, 1:3] = sine_degree(norm_y[:, 1:3])

        kw_defaults['mean'][1:3] = sine_degree(91), sine_degree(89)
        kw_defaults['std'][1:3] = sine_degree(36.4), sine_degree(52.6)
        kw_defaults['max'][1:3] = 1, 1
        kw_defaults['min'][1:3] = -1, -1

    for key in kwargs:
        if key not in allowedmodes[mode]:
            raise ValueError('%s keyword not in allowed keywords %s' % (key, allowedmodes[mode]))

    for kw in allowedmodes[mode]:
        kwargs.setdefault(kw, kw_defaults[kw])

    if mode == 'norm':
        norm_y = (np.array(norm_y).reshape(1, -1) -
                  np.array(kwargs['mean']).reshape(1, -1)) / np.std(np.array(kwargs['std']).reshape(1, -1))
    elif mode == 'range':
        range_ = np.array(kwargs['max']).reshape(-1, 1) - np.array(kwargs['min']).reshape(-1, 1)
        norm_y = (np.array(norm_y).reshape(-1, 11).T - np.array(kwargs['min'])[:, np.newaxis]) / range_
    else:
        raise ValueError('mode should be norm or range')
    return norm_y.T


def inverse_transformation(params_to_transform, inv_logB=True, inv_angle_transformation=False):
    kw_defaults = {
        'mean': [530, 91, 89, 33, 0.31, 12, 27083, 19567, 0.04, 0.5, 0.36],
        'std': [565, 36.4, 52.6, 9.5, 0.21, 11.82, 4112, 5927, 0.04, 0.5, 0.36],
        'max': [5000, 180, 180, 90, 1.5, 100, 38603,
                60464, 10, 1, 10],
        'min': [0, 0, 0, 20, 0, 0.01, 0, 0, -10, 0, -10]
    }

    if inv_logB:
        kw_defaults['mean'][0] = 5.67
        kw_defaults['std'][0] = 1.16
        kw_defaults['max'][0] = 8.51
        kw_defaults['min'][0] = 0

    def sine_degree(x):
        return np.sin(x * np.pi / 180)

    if inv_angle_transformation:
        kw_defaults['mean'][1:3] = sine_degree(91), sine_degree(89)
        kw_defaults['std'][1:3] = sine_degree(36.4), sine_degree(52.6)
        kw_defaults['max'][1:3] = 1, 1
        kw_defaults['min'][1:3] = -1, -1

    params_range = np.array(kw_defaults['max']).reshape(-1, 1) - np.array(kw_defaults['min']).reshape(-1, 1)

    transformed_params = params_to_transform.reshape(-1, 11).T * params_range + np.array(kw_defaults['min'])[:,
                                                                                np.newaxis]

    transformed_params = transformed_params.T

    if inv_logB:
        transformed_params[:, 0] = np.exp(transformed_params[:, 0])

    if inv_angle_transformation:
        transformed_params[:, 1:3] = np.arcsin(transformed_params[:, 1:3]) * 180 / np.pi
        transformed_params[:, 1:3] += 90

    return transformed_params


def normalize_spectrum(spectrum, factors=None):
    if factors is None:
        factors = [1, 1000, 1000, 1000]

    assert len(spectrum.shape) == 2 or len(
        spectrum.shape) == 3, 'spectrum must be 2d (one sample with channel in columns) or 3d array with (samples by 0 axis)'
    assert spectrum.shape[-1] == 4, 'Need to provide four lines in last dimension'

    if len(spectrum.shape) == 2:
        spectrum = spectrum * np.array(factors).reshape((1, 4))
    else:
        spectrum = (np.swapaxes(spectrum, 0, 2) * np.array(factors).reshape(4, 1, 1)).swapaxes(0, 2)

    return spectrum


class FlattenSpectrum:
    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']
        if len(spectrum.shape) == 2:
            spectrum = spectrum.flatten(order='F')
        else:
            spectrum = spectrum.reshape(spectrum.shape[0], -1, order='F')
        return {'X': (spectrum, cont),
                'Y': params}


class ToTensor(object):
    """Convert np arrays intoTensors."""

    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']
        if isinstance(cont, float):
            cont = torch.FloatTensor([cont])
        else:
            cont = torch.from_numpy(cont).float().reshape(-1,1)
        return {'X': (torch.from_numpy(spectrum).float(), cont),
                'Y': torch.from_numpy(params.astype(np.float32)).squeeze()}


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
    allowed_kwargs = {'factors', 'cont_scale', 'norm_output', 'logB', 'mode', 'angle_transformation'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise KeyError(f'{key} not in allowed keywords')

    rescale = Rescale(**kwargs)
    flat = FlattenSpectrum()
    to_tensor = ToTensor()
    return transforms.Compose([rescale, flat, to_tensor])


class ToConv1d(object):
    """Prepare X and y for conv 1d model"""

    def __call__(self, sample):
        (spectrum, cont), params = sample['X'], sample['Y']
        if len(spectrum.shape) == 2:
            spectrum = np.swapaxes(spectrum, 0, 1)
        else:
            spectrum = np.swapaxes(spectrum, 1, 2)
        return {'X': (spectrum, cont),
                'Y': params}


def conv1d_transform_rescale(**kwargs) -> Callable:
    allowed_kwargs = {'factors', 'cont_scale', 'norm_output', 'logB', 'angle_transformation', 'mode'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise KeyError(f'{key} not in allowed keywords')

    rescale = Rescale(**kwargs)
    to_conv = ToConv1d()
    to_tensor = ToTensor()
    return transforms.Compose([rescale, to_conv, to_tensor])


def conv1d_transform_standard(**kwargs) -> Callable:
    allowed_kwargs = {'logB', 'norm_output', 'mode'}
    for key in kwargs:
        if key not in allowed_kwargs:
            raise KeyError(f'{key} not in allowed keywords')
    norm = NormalizeStandard(**kwargs)
    to_tensor = ToTensor()
    to_conv = ToConv1d()
    return transforms.Compose([norm, to_tensor, to_conv])
