from typing import List, Union
from astropy.io import fits
from inverse_problem.milne_edington.me import me_model
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import gdown


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def download_from_google_disc(fileid=None, dest=None):
    """
    Download parameter base file from google disk by fileid
    Args:
        fileid (string): fileid of the source value, default value - small parameter database
        dest (string): destination folder

    Returns:

    """
    if fileid is None:

        fileid = '19jkSXHxAPWZvfgo5oxSmvEagme5YLY33'
    url = 'https://drive.google.com/uc?id=' + fileid
    if dest is None:
        dest = Path(os.getcwd()).parent / 'data' / 'downloaded_parameters_base.fits'
    gdown.download(url, dest, quiet=False)


def create_small_dataset(filename, savename, size=10000):
    parameter_base = fits.open(filename)[0].data
    small_parameter_base = parameter_base[:size, :]
    hdul = fits.HDUList([fits.PrimaryHDU(small_parameter_base)])
    hdul.writeto(savename, overwrite=1)


def compute_mean_spectrum(filename, batch_size=None, nbatches=None):
    line_vec = (6302.5, 2.5, 1)
    line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
    # filename = '/Users/irinaknyazeva/Projects/Solar/InverseProblem/data/parameters_base.fits'
    parameter_base = fits.open(filename)[0].data
    cont_vec = parameter_base[:, 6] + line_vec[2] * parameter_base[:, 7]
    N = parameter_base.shape[0]
    if batch_size is None:
        batch_size = 10
    if nbatches is None:
        nbatches = min(2, N // batch_size - 1)
    mean_spectrum = np.zeros((nbatches, 224))
    max_spectrum = np.zeros((nbatches, 224))
    min_spectrum = np.zeros((nbatches, 224))
    std_spectrum = np.zeros((nbatches, 224))
    for idx in tqdm(range(nbatches)):
        param_vec = parameter_base[idx * batch_size: idx * batch_size + batch_size, :]
        spectrum = me_model(param_vec, line_arg, line_vec, with_ff=True).reshape((-1, 224), order='F')
        mean_spectrum[idx, :] = spectrum.mean(axis=0)
        max_spectrum[idx, :] = spectrum.max(axis=0)
        min_spectrum[idx, :] = spectrum.min(axis=0)
        std_spectrum[idx, :] = spectrum.std(axis=0)
    spectrum_dict = {'mean': mean_spectrum.mean(axis=0),
                     'max': max_spectrum.max(axis=0),
                     'min': min_spectrum.min(axis=0),
                     'std': std_spectrum.mean(axis=0),
                     'cont_mean': np.mean(cont_vec),
                     'cont_std': np.std(cont_vec)}

    return spectrum_dict


def plot_spectrum_range(spectrum_dict):
    plt.subplot(2, 2, 1)
    plt.plot(spectrum_dict['mean'][:56])
    plt.plot(spectrum_dict['max'][:56])
    plt.plot(spectrum_dict['min'][:56])

    plt.subplot(2, 2, 2)
    plt.plot(spectrum_dict['mean'][56:2 * 56])
    plt.plot(spectrum_dict['max'][56:2 * 56])
    plt.plot(spectrum_dict['min'][56:2 * 56])
    plt.subplot(2, 2, 3)
    plt.plot(spectrum_dict['mean'][2 * 56:3 * 56])
    plt.plot(spectrum_dict['min'][2 * 56:3 * 56])
    plt.plot(spectrum_dict['max'][2 * 56:3 * 56])
    plt.subplot(2, 2, 4)
    plt.plot(spectrum_dict['mean'][3 * 56:])
    plt.plot(spectrum_dict['min'][3 * 56:])
    plt.plot(spectrum_dict['max'][3 * 56:])
    plt.show()


def shift_intensity(profile, meanI):
    profile[:56] -= meanI
    return profile
