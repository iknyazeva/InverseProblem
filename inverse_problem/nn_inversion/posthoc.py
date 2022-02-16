import matplotlib.pyplot as plt
#import seaborn as sns
from matplotlib.colors import LogNorm
import os
import torch
from inverse_problem.nn_inversion import normalize_spectrum
from scipy.stats import norm
from inverse_problem import HinodeME
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
import numpy as np
import pandas as pd
import seaborn as sns
import pylab


def open_param_file(path, normalize=True, print_params=True, **kwargs):
    """

    Args:
        print_params (object): 
    """
    refer = fits.open(path)
    param_list = [1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13]
    names = [refer[i].header['EXTNAME'] for i in param_list]
    if print_params:
        print('Open file with 36 available parameters, 11 will be selected')
        print('\n'.join(names))
    data = np.array([refer[i].data for i in param_list], dtype='float').swapaxes(0, 2).swapaxes(0, 1)
    if normalize:
        shape = data.shape
        data = normalize_output(data.reshape(-1, 11), **kwargs).reshape(shape)

    return data, names


def compute_metrics(refer, predicted, index=None, names=None, save_path=None, mask=None):
    """
    Compute metrics
    Args:
        refer (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
        index (int): index for output, if None for all values will be computed
        predicted  (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
        names (list of str): parameter names
        save_path (str): save path to results
        mask (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
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

    refer = refer.reshape(-1, 11)
    predicted = predicted.reshape(-1, 11)

    if mask is not None:
        mask = mask.reshape(-1, 11)

        rows_mask = np.any(mask, axis=1)

        refer = refer[~rows_mask, :]
        predicted = predicted[~rows_mask, :]

    if index is None:
        r2list = []
        mselist = []
        maelist = []
        for i in range(len(names)):
            r2list.append(np.corrcoef(refer[:, i], predicted[:, i])[0][1] ** 2)
            mselist.append(mean_squared_error(refer[:, i], predicted[:, i]))
            maelist.append(mean_absolute_error(refer[:, i], predicted[:, i]))
        df = pd.DataFrame([r2list, mselist, maelist], columns=names, index=['r2', 'mse', 'mae']).T.round(4)
        if save_path:
            df.to_csv(save_path)
        return df
    else:
        r2 = np.corrcoef(refer[:, index], predicted[:, index])[0][1] ** 2
        mae = mean_absolute_error(refer[:, index], predicted[:, index])
        mse = mean_squared_error(refer[:, index], predicted[:, index])
    return r2, mae, mse


def plot_spectrum(sp_folder, date, path_to_refer, idx_0, idx_1):
    """
    Plot spectrum, corresponding referens values of parameters and model spectrum
    idx_0 - index of line in one spectrum file (512), idx_1 - index of spectrum file sorted by time (873 in total)
    """
    # refer, names = open_param_file(path_to_refer, print_params=False, normalize=False)
    spectra_file = open_spectrum_data(sp_folder, date, idx_1)
    real_sp = real_spectra(spectra_file)
    full_line = real_sp[idx_0, :]
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    line_type = ['I', 'Q', 'U', 'V']
    print('Real spectrum for parameters')
    # print(', '.join([names[i]+f': {refer[idx_0,idx_1, i]:.2f}' for i in range(11)]))
    cont_int = np.max(full_line)

    for i in range(4):
        ax[i // 2][i % 2].plot(full_line[i * 56:i * 56 + 56] / cont_int)
        ax[i // 2][i % 2].set_title(f'Spectral line {line_type[i]}')
    fig.suptitle(f'Real spectrum with empiric intensity {cont_int :.1f}', fontsize=16, fontweight="bold")
    fig.set_tight_layout(tight=True)

    return full_line, cont_int


def plot_model_spectrum(refer, names, idx_0, idx_1):
    param_vec = refer[idx_0, idx_1, :]
    obj = HinodeME(param_vec)
    profile = obj.compute_spectrum(with_ff=True, with_noise=True)
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    line_type = ['I', 'Q', 'U', 'V']
    print('Model spectrum for parameters')
    print(', '.join([names[i] + f': {refer[idx_0, idx_1, i]:.2f}' for i in range(11)]))

    for i in range(4):
        ax[i // 2][i % 2].plot(profile[0, :, i])
        ax[i // 2][i % 2].set_title(f'Spectral line {line_type[i]}')
    fig.set_tight_layout(tight=True)
    fig.suptitle(f'Model spectrum with estimated intensity {obj.cont:.1f}', fontsize=16, fontweight="bold")

    return profile, obj.cont


def read_spectrum_for_refer(sp_folder, date):
    real_samples = np.zeros((512, 873, 224))
    cont = np.zeros((512, 873))
    for idx_1 in range(873):
        line = real_spectra(open_spectrum_data(sp_folder, date, idx_1))
        real_samples[:, idx_1, :] = line
        cont[:, idx_1] = np.max(line, axis=1)
    real_samples = real_samples.reshape(-1, 224)
    cont = cont.reshape(-1, 1)
    return real_samples / cont, cont


def prepare_real_mlp(sp_folder, date, factors=None, cont_scale=None, device=None):
    real_samples, cont = read_spectrum_for_refer(sp_folder, date)
    norm_real_samples = normalize_spectrum(np.reshape(real_samples, (-1, 56, 4), order='F'), factors=factors)
    norm_cont = cont / cont_scale
    norm_real_samples = np.reshape(norm_real_samples, (-1, 224), order='F')
    real_x = [torch.from_numpy(norm_real_samples).float().to(device), torch.from_numpy(norm_cont).float().to(device)]
    return real_x


def plot_pred_vs_refer(predicted, refer, output_index=0, name=None, save_path='../', title=''):
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
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    axs[0].imshow(predicted.reshape(refer.shape)[:, :, output_index], cmap='gray')
    axs[0].set_title("Predicted")
    axs[1].imshow(refer[:, :, output_index], cmap='gray')
    axs[1].set_title("True output")
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    plt.suptitle(f"Model results for {name}",
                 fontsize=16)
    # plt.suptitle(f"Model results for {name} \n\n Quality metrics: r2 = {r2:.3f}, mse = {mse:.5f}, mae = {mae:.5f} ",
    #              fontsize=16)
    plt.tight_layout()
    fig.savefig(save_path + "pred_vs_refer_" + title + ".pdf")


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


def plot_analysis_graphs(refer, predicted, names, title=None, index=0, save_path=None):
    """
        draw 2d graphs:
        index = 0: (x_pred-x_true)/x_true vs x_true
        index = 1: x_pred vs x_true.
    """
    if not title:
        title = ['(x_pred-x_true)/x_true vs x_true', 'x_pred vs x_true'][index]

    refer_flat = refer.reshape(-1, 11)
    predicted_flat = predicted.reshape(-1, 11)

    fig, axs = plt.subplots(3, 4, figsize=(19, 15), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axs.flat[:-1]):
        if index == 0:
            X, Y = refer_flat[:, i], predicted_flat[:, i] - refer_flat[:, i]
        elif index == 1:
            X, Y = refer_flat[:, i], predicted_flat[:, i]
        else:
            theta = np.linspace(0, 2 * np.pi, 100)
            X = 16 * (np.sin(theta) ** 3)
            Y = 13 * np.cos(theta) - 5 * np.cos(2 * theta) - 2 * np.cos(3 * theta) - np.cos(4 * theta)
        ax.set_title(names[i], weight='bold')
        ax.plot(X, Y, 'o', color='red', alpha=0.1, markersize=4, markeredgewidth=0.0)

    if index == 0:
        fig.supxlabel(r'$x_{true}$')
        fig.supylabel(r'$(x_{pred} - x_{true})/ x_{true}$')
    elif index == 1:
        fig.supxlabel(r'$x_{true}$')
        fig.supylabel(r'$x_{pred}$')

    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])

    if save_path:
        fig.savefig(save_path + ".pdf")
    plt.show()


def plot_analysis_hist2d(refer, predicted, names=None, index=0, title=None, bins=100, save_path=None):
    """
        draw hist2d:
        index = 0: (x_pred-x_true)/x_true vs x_true,
        index = 1: x_pred vs x_true.
    """
    if not title:
        title = [r'$\left(x_{pred}-x_{true}\right) / x_{true}$ vs $x_{true}$',
                 r'$x_{pred}$ vs $x_{true}$'][index]

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

    refer_flat = refer.reshape(-1, 11)
    predicted_flat = predicted.reshape(-1, 11)

    fig, axs = plt.subplots(3, 4, figsize=(19, 15))
    fig.suptitle(title, fontsize=19)

    for i, ax in enumerate(axs.flat[:-1]):
        if index == 0:
            X, Y = refer_flat[:, i], predicted_flat[:, i] - refer_flat[:, i]
        elif index == 1:
            X, Y = refer_flat[:, i], predicted_flat[:, i]
        else:
            raise ValueError
        ax.set_title(names[i], weight='bold')
        plot_params = ax.hist2d(X, Y, bins=bins, norm=LogNorm())

        if index == 0:
            fig.supxlabel(r'$x_{true}$')
            fig.supylabel(r'$\left(x_{pred} - x_{true}\right) / x_{true}$')
        elif index == 1:
            fig.supxlabel(r'$x_{true}$')
            fig.supylabel(r'$x_{pred}$')
        else:
            raise ValueError

    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])

    if save_path:
        fig.savefig(save_path + ".png")

    plt.subplots_adjust(right=0.8)
    cax = plt.axes([0.85, 0.15, 0.05, 0.7])

    plt.colorbar(plot_params[3], cax=cax)
    plt.show()

def plot_analysis_hist2d_unc(refer, predicted_mu, predicted_sigma, names=None, index=0, mask=None, title=None, bins=100,
                            save_path=None, plot_stats=False):
    """
        draws 2d graphs:
        1. (x_true - x_pred)/sigma_pred vs x_true,
        2. (x_true - x_pred) vs sigma_pred,
        3. x_true vs sigma_pred,
    """
    if not title:
        title = [r'$\left(x_{true} - x_{pred}\right) / \sigma_{pred}$ vs $x_{true}$',
                 r'$\sigma_{pred}$ vs $x_{true} - x_{pred}$',
                 r'$\sigma_{pred}$ vs $x_{true}$'][index]

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

    if mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)

    titles_for_saving = ['x_true vs (x_true - x_pred)\sigma_pred',
                         'x_true - x_pred vs sigma_pred',
                         'x_true vs sigma_pred',
                         'x_true vs x_pred',
                         '(x_pred - x_true)/x_true vs x_true']

    fig, axs = plt.subplots(3, 4, figsize=(15, 12))

    for i, ax in enumerate(axs.flat[:-1]):
        if index == 0:
            X, Y = refer[:, i], (refer[:, i] - predicted_mu[:, i]) / predicted_sigma[:, i]
        elif index == 1:
            X, Y = refer[:, i] - predicted_mu[:, i], predicted_sigma[:, i]
        elif index == 2:
            X, Y = refer[:, i], predicted_sigma[:, i]
        elif index == 3:
            X, Y = refer[:, i], predicted_mu[:, i]
        else:
            X, Y = refer[:, i], (predicted_mu[:, i] - refer[:, i])/refer[:, i]
        if i == 7 or i == 8:
            bins = 500
        ax.set_title(names[i], weight='bold')
        plot_params = ax.hist2d(X, Y, bins=bins, norm=LogNorm())
        if plot_stats:
            grid, a, b = calculate_ab_fit(X, Y, N=500)
            ax.plot(grid, a, color='red', label='a')
            ax.plot(grid, b, color='orange', label='b')
            ax.legend(loc='upper right')
        ymin, ymax = np.percentile(Y, 0.01), np.percentile(Y, 99.99)
        xmin, xmax = np.percentile(X, 0.01), np.percentile(X, 99.99)
        ax.axis(ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
        if i == 7 or i == 4:
            ax.set_xticks(np.round(np.linspace(xmin, 0.9*xmax, 3), 2))

    font = 16
    y_position, x_position = 0.01, 0.5
    if index == 0:
        fig.text(x_position, y_position, r'$x_{true}$', ha='center', fontsize=font)
        fig.text(0.01, 0.5, r'$(x_{true} - x_{pred})/ \sigma_{pred}$', va='center', rotation='vertical', fontsize=font)
    elif index == 1:
        fig.text(x_position, y_position, r'$x_{true} - x_{pred}$', ha='center', fontsize=font)
        fig.text(0.01, 0.5, r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font)
    elif index == 2:
        fig.text(x_position, y_position, r'$x_{true}$', ha='center', fontsize=font)
        fig.text(0.01, 0.5, r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font)
    elif index == 3:
        fig.text(x_position, y_position, r'$x_{true}$', ha='center', fontsize=font)
        fig.text(0.01, 0.5, r'$x_{pred}$', va='center', rotation='vertical', fontsize=font)
    else:
        fig.text(x_position, y_position, r'$x_{true}$', ha='center', fontsize=font)
        fig.text(0.01, 0.5, r'$(x_{pred} - x_{true})/x_{true}$', va='center', rotation='vertical', fontsize=font)

    fig.set_facecolor('xkcd:white')
    axs[2][3].axis("off")
    pylab.tight_layout(pad=4)

    cax = plt.axes([0.8, 0.08, 0.02, 0.2])
    plt.colorbar(plot_params[3], ax=axs, cax=cax, shrink=1)

    fig.savefig(save_path + title + "_" + titles_for_saving[index] + ".png")
    plt.show()


def calculate_ab_fit(X, Y, N=100):
    a, b, grid = [], [], []
    shape = X.shape[0]
    grid_step = np.abs((np.max(X) - np.min(X))) // N + 1
    slice_step = shape // N + 1
    for i in range(N - 1):
        slice = Y[i * slice_step:(i + 1) * slice_step]
        if slice.shape[0] != 0:
            mu, std = norm.fit(slice)
        else:
            mu, std = None, None
        a.append(mu)
        b.append(std)
        grid.append(np.min(X) + i * grid_step)

    if len(grid) != len(a):
        raise ValueError(f'grid ({len(grid)}) and a ({len(a)}) must be of the same size')
    return np.array(grid), np.array(a), np.array(b)


def plot_hist_params_comparison(pars_arr1, pars_arr2, pars_names, plot_name='', bins=100, save_path=None):
    pars_arr1 = pars_arr1.reshape(-1, 11)
    pars_arr2 = pars_arr2.reshape(-1, 11)

    fig, axs = pylab.subplots(3, 4, figsize=(20, 12), constrained_layout=True)

    # for i, ax in enumerate(axs[:3]):
    for i, ax in enumerate(axs.flat[:-1]):
        ax.set_yscale('log')
        ax.set_title(pars_names[i], weight='bold')
        # ax.set_xlim(0, 1)

        sns.histplot(
           pars_arr1[:, i], ax=ax, bins=bins, color='blue', label="predicted"
        )
        sns.histplot(
           pars_arr2[:, i], ax=ax, bins=bins, color='red', alpha=0.5, label='refer'
        )

    fig.set_facecolor('xkcd:white')

    h, l = ax.get_legend_handles_labels()
    axs[2][3].legend(h, l, borderaxespad=0, loc='upper left')
    axs[2][3].axis("off")

    if save_path:
        fig.savefig(save_path + "hists_" + plot_name + ".pdf")
    plt.show()

def open_spectrum_data(sp_folder, date, idx):
    """
    path should start from the folder included in level1 folder, with data year
    only for this path_to_folder like this sp_20140926_170005
    """
    sp_path = os.path.join(sp_folder, date[0], date[1], date[2], 'SP3D')
    sp_path = glob.glob(f'{sp_path}/*/')[0]
    sp_lines = sorted(glob.glob(sp_path + '*.fits'))
    # print(f'Number of files: {len(sp_lines)}')
    return fits.open(sp_lines[idx])


def real_spectra(spectra_file):
    """
    Extracting and plotting spectral lines from fits
    Why multiply to numbers?
    """
    real_I = spectra_file[0].data[0][:, 56:].astype('float64') * 2
    real_Q = spectra_file[0].data[1][:, 56:].astype('float64')
    real_U = spectra_file[0].data[2][:, 56:].astype('float64')
    real_V = spectra_file[0].data[3][:, 56:].astype('float64')
    return np.concatenate((real_I, real_Q, real_U, real_V), axis=1)


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
