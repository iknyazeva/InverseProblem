import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
import numpy as np
import pandas as pd


def open_param_file(path, normalize=True):
    refer = fits.open(path)
    print('Open file with 36 available paramters, 11 will be selected')
    param_list = [1, 2, 3, 6, 8, 7, 33, 10, 5, 12, 13]
    names = [refer[i].header['EXTNAME'] for i in param_list]
    print('\n'.join(names))
    data = np.zeros(shape=(512, 485, 11))
    for i, idx in enumerate(param_list):
        data[:, :, i] = refer[idx].data
    if normalize:
        shape = data.shape
        data = normalize_output(data.reshape(-1, 11)).reshape(shape)

    return data, names


def compute_metrics(refer, predicted, names, save_path=None):
    r2list = []
    mselist = []
    maelist = []
    for i, _ in enumerate(names):
        r2list.append(np.corrcoef(refer[:, :, i].flatten(), predicted[:, :, i].flatten())[0][1] ** 2)
        mselist.append(mean_squared_error(refer[:, :, i].flatten(), predicted[:, :, i].flatten()))
        maelist.append(mean_absolute_error(refer[:, :, i].flatten(), predicted[:, :, i].flatten()))
    df = pd.DataFrame([r2list, mselist, maelist], columns=names, index=['r2', 'mse', 'mae']).T.round(3)
    if save_path:
        df.to_csv(save_path)
    return df


def plot_params(data):
    """Draw all 11 parameters at once
    data: np array (:, :, 11)
    """
    names = ['Field Strength',
             'Field Inclination',
             'Field Azimuth',
             'Doppler Width',
             'Damping',
             'Line Strength',
             'S_0',
             'S_1',
             'Doppler Shift',
             'Filling Factor',
             'Stray light Doppler shift']

    plt.figure(figsize=(12, 9))
    plt.axis('off')
    for i in range(11):
        plt.subplot(3, 4, i + 1)
        plt.title(names[i])
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')



def metrics(true, pred):
    print('r2', r2_score(true, pred))
    print('rmse', mean_squared_error(true, pred, squared=False))
    print('mse', mean_squared_error(true, pred))
    print('mae', mean_absolute_error(true, pred))
    print('')


def plot_spectra(pred, true):
    """ Draw prediction and reference"""
    plt.figure(figsize=(15, 15))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
    sc = axs[0].imshow(pred, cmap='gray')
    fig.colorbar(sc, ax=axs[0])
    axs[0].set_title('Predicted')

    sc = axs[1].imshow(true, cmap='gray')
    fig.colorbar(sc, ax=axs[1])
    axs[1].set_title('True')
