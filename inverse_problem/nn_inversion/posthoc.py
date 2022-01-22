import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
import numpy as np
import pandas as pd


def open_param_file(path, normalize=True, **kwargs):
    refer = fits.open(path)
    print('Open file with 36 available parameters, 11 will be selected')
    param_list = [1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13]
    names = [refer[i].header['EXTNAME'] for i in param_list]
    print('\n'.join(names))
    data = np.array([refer[i].data for i in param_list], dtype='float').swapaxes(0, 2).swapaxes(0, 1)
    if normalize:
        shape = data.shape
        data = normalize_output(data.reshape(-1, 11), **kwargs).reshape(shape)

    return data, names


def compute_metrics(refer, predicted, index=None, names=None, save_path=None):
    """
    Compute metrics
    Args:
        refer (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
        index (int): index for output, if None for all values will be computed
        predicted  (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
        names (list of str): parameter names
        save_path (str): save path to results

    Returns:
        pandas dataframe with metrics

    """
    if names is None:
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
    if index is None:
        r2list = []
        mselist = []
        maelist = []
        for i, _ in enumerate(names):
            r2list.append(np.corrcoef(refer.reshape(-1, 11)[:, i], predicted.reshape(-1, 11)[:, i])[0][1] ** 2)
            mselist.append(mean_squared_error(refer.reshape(-1, 11)[:, i], predicted.reshape(-1, 11)[:, i]))
            maelist.append(mean_absolute_error(refer.reshape(-1, 11)[:, i], predicted.reshape(-1, 11)[:, i]))
        df = pd.DataFrame([r2list, mselist, maelist], columns=names, index=['r2', 'mse', 'mae']).T.round(3)
        if save_path:
            df.to_csv(save_path)
        return df
    else:
        r2 = np.corrcoef(refer.reshape(-1, 11)[:, index], predicted.reshape(-1, 11)[:, index])[0][1] ** 2
        mae = mean_absolute_error(refer.reshape(-1, 11)[:, index], predicted.reshape(-1, 11)[:, index])
        mse = mean_squared_error(refer.reshape(-1, 11)[:, index], predicted.reshape(-1, 11)[:, index])
    return r2, mae, mse


def plot_pred_vs_refer(predicted, refer, output_index=0, name=None):
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
    if name is None:
        name = names[output_index]
    r2, mae, mse = compute_metrics(refer, predicted, output_index)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].imshow(predicted.reshape(refer.shape)[:, :, output_index], cmap='gray');
    axs[0].set_title("Predicted")
    axs[1].imshow(refer[:, :, output_index], cmap='gray');
    axs[1].set_title("True output")
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.suptitle(f"Model results for {name} \n\n Quality metrics: r2 = {r2:.3f}, mse = {mse:.5f}, mae = {mae:.5f} ",
                 fontsize=16)
    plt.tight_layout()


def plot_params(data, names=None):
    """Draw all 11 parameters at once
    data: np array (:, :, 11)
    """
    if names is None:
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

    plt.figure(figsize=(12, 10))
    plt.axis('off')
    for i in range(11):
        plt.subplot(3, 4, i + 1)
        plt.title(names[i])
        plt.imshow(data[:, :, i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()


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
