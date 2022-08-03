import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import os
import torch
from inverse_problem.nn_inversion import normalize_spectrum

from inverse_problem import HinodeME, me_model
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
import numpy as np
import pandas as pd
from scipy import stats


def open_param_file(path, normalize=True, print_params=True, **kwargs):
    """ Build parameters vector on the basis of the Hinode fits from e https://csac.hao.ucar.edu/fg_download.php
        we need to select date and toogle level1 (big dataset, available with delay)
        and level 2 with parameters inversion.
        After that data should be saved in directory (path_to_folder) and level1
        folder renamed as file with inverted parameters, example:
    Args:
        path (object): path to  source data
        print_params (bool): if print chosen parameters
        normalize (bool): if need to normalize as used in neural networks
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


def real_spectra(spectra_file, mult_I=2):
    """
    Extracting and plotting spectral lines from fits
    Why multiply to numbers? Multiply to 2 because in the source data it is divided by 2
    """
    real_I = spectra_file[0].data[0][:, 56:].astype('float64') * mult_I
    real_Q = spectra_file[0].data[1][:, 56:].astype('float64')
    real_U = spectra_file[0].data[2][:, 56:].astype('float64')
    real_V = spectra_file[0].data[3][:, 56:].astype('float64')
    return np.concatenate((real_I, real_Q, real_U, real_V), axis=1)


def plot_real_spectrum(sp_folder, path_to_refer, idx_0, idx_1, norm=True, plot_spectrum=True):
    """  Plot spectrum, corresponding referens values of parameters and model spectrum
    idx_0 - index of line in one spectrum file (512), idx_1 - index of spectrum file sorted by time (873 in total)
    Returns:

    Args:
       sp_folder (object): path to folder with real spectrum
       path_to_refer (object): path to folder with the parameters
       date (list of object): #date as a list with year, month,day, hour ['2014','09','26','17']
       idx_0 (int): idx_0 - index of line in one spectrum file (512)
       idx_1 (): index of spectrum file sorted by time (873 in total)

    Returns:
        full_line
        cont_int
        params
    """
    assert 0 < idx_0 < 512, "Index 0 should be in range (0, 512)"
    assert 0 < idx_1 < 873, "Index 1 should be in range (0, 873)"

    refer, names = open_param_file(path_to_refer, print_params=False, normalize=False)
    date_str = sp_folder.split("_")[-2]
    date = [date_str[:4], date_str[4:6], date_str[6:]]
    spectra_file = open_spectrum_data(sp_folder, date, idx_1)
    real_sp = real_spectra(spectra_file)
    full_line = real_sp[idx_0, :]
    params = refer[idx_0, idx_1]
    cont_int = np.max(full_line)

    if plot_spectrum:
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        line_type = ['I', 'Q', 'U', 'V']
        print('Real spectrum for parameters')
        print(', '.join([names[i] + f': {params[i]:.2f}' for i in range(11)]))

        for i in range(4):
            if norm:
                ax[i // 2][i % 2].plot(full_line[i * 56:i * 56 + 56] / cont_int)
            else:
                ax[i // 2][i % 2].plot(full_line[i * 56:i * 56 + 56])
            ax[i // 2][i % 2].set_title(f'Spectral line {line_type[i]}')
        fig.suptitle(f'Real spectrum with empiric intensity {cont_int :.1f}', fontsize=16, fontweight="bold")
        fig.set_tight_layout(tight=True)
        if norm:
            return full_line.reshape((1, 56, 4), order='F')/cont_int, cont_int, params
        else:
            return full_line.reshape((1, 56, 4), order='F'), cont_int, params


def plot_model_spectrum(path_to_refer, idx_0, idx_1,
                        with_noise=True, with_ff=True, norm=True, line_vec=None, line_arg=None,
                        plot_spectrum=True):
    refer, names = open_param_file(path_to_refer, print_params=False, normalize=False)
    param_vec = refer[idx_0, idx_1, :]
    cont = np.sum(param_vec[6:8])

    if line_vec is None:
        line_vec = (6302.5, 2.5, 1)
    if line_arg is None:
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
    profile = me_model(param_vec, line_arg, line_vec, with_ff=with_ff, with_noise=with_noise, norm=norm)
    if plot_spectrum:
        fig, ax = plt.subplots(2, 2, figsize=(10, 5))
        line_type = ['I', 'Q', 'U', 'V']
        print('Model spectrum for parameters')
        print(', '.join([names[i] + f': {refer[idx_0, idx_1, i]:.2f}' for i in range(11)]))

        for i in range(4):
            ax[i // 2][i % 2].plot(profile[0, :, i])
            ax[i // 2][i % 2].set_title(f'Spectral line {line_type[i]}')
        fig.set_tight_layout(tight=True)
        fig.suptitle(f'Model spectrum with estimated intensity {cont:.1f}', fontsize=16, fontweight="bold")

    return profile, param_vec, cont


def nlpd_metric(refer, mean_pred, sigma_pred):
    """
    The Negative Log Predictive Density (NLPD) metric calculation.
    Source: http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf
    
    Parameters:
    -----------
    refer : ndarray-like
        True observations with shape [n_observations, n_parameters].
    mean_pred : ndarray-like
        Predicted paramter mean values with shape [n_observations, n_parameters].
    sigma_pred : array-like
        Predicted paramter standard deviations with shape [n_observations, n_parameters].
        
    Returns:
    --------
    metric : float
        NLPD metrc value.
    """

    metric = (refer - mean_pred) ** 2 / (2 * sigma_pred ** 2) + np.log(sigma_pred) + 0.5 * np.log(2 * np.pi)

    return metric.mean(axis=0)


def nrmse_p_metric(refer, mean_pred, sigma_pred):
    """
    The normalized Root Mean Squared Error (nRMSEp) metric based on predicted error. 
    Source: http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf
    
    Parameters:
    -----------
    refer : ndarray-like
        True observations with shape [n_observations, n_parameters].
    mean_pred : ndarray-like
        Predicted paramter mean values with shape [n_observations, n_parameters].
    sigma_pred : array-like
        Predicted paramter standard deviations with shape [n_observations, n_parameters].
        
    Returns:
    --------
    metric : float
        nRMSEp metrc value.
    """

    metric = (refer - mean_pred) ** 2 / sigma_pred ** 2

    return np.sqrt(metric.mean(axis=0))


def picp_metric(refer, mean_pred, sigma_pred, alpha=0.90):
    """
    The Prediction Interval Coverage Probability (PICP) metric. 
    Source: https://www.sciencedirect.com/science/article/pii/S0893608006000153?via%3Dihub
    
    Parameters:
    -----------
    refer : ndarray-like
        True observations with shape [n_observations, n_parameters].
    mean_pred : ndarray-like
        Predicted paramter mean values with shape [n_observations, n_parameters].
    sigma_pred : array-like
        Predicted paramter standard deviations with shape [n_observations, n_parameters].
    alpha : float [0, 1]
        Fraction of the distribution inside confident intervals.
        
    Returns:
    --------
    metric : float
        PICP metrc value.
    """

    p_left, p_right = stats.norm.interval(alpha=alpha, loc=mean_pred, scale=sigma_pred)
    metric = (refer > p_left) * (refer <= p_right)

    return metric.mean(axis=0)


def compute_metrics(refer, predicted, sigmas=None, index=None, names=None, mask=None, save_path=None):
    """
    Compute metrics
    Args:
        refer (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
        index (int): index for output, if None for all values will be computed
        predicted  (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
        sigmas  (np.ndarray): 2d with N*num_parameters array, or (height*width*num_parameters)
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
    if sigmas is not None:
        sigmas = sigmas.reshape(-1, 11)

    if mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)
        refer = refer[~masked_rows, :]
        predicted = predicted[~masked_rows, :]
        if sigmas is not None:
            sigmas = sigmas[~masked_rows, :]

    if index is None:
        r2list = []
        mselist = []
        maelist = []
        if sigmas is not None:
            nlpdlist = []
            nrmseplist = []
            picp68list = []
            picp95list = []
        for i in range(len(names)):
            r2list.append(np.corrcoef(refer[:, i], predicted[:, i])[0][1] ** 2)
            mselist.append(mean_squared_error(refer[:, i], predicted[:, i]))
            maelist.append(mean_absolute_error(refer[:, i], predicted[:, i]))
            if sigmas is not None:
                nlpdlist.append(nlpd_metric(refer[:, i], predicted[:, i], sigmas[:, i]))
                nrmseplist.append(nrmse_p_metric(refer[:, i], predicted[:, i], sigmas[:, i]))
                picp68list.append(picp_metric(refer[:, i], predicted[:, i], sigmas[:, i], alpha=0.68268))
                picp95list.append(picp_metric(refer[:, i], predicted[:, i], sigmas[:, i], alpha=0.95450))
        if sigmas is None:
            df = pd.DataFrame([r2list, mselist, maelist], columns=names, index=['r2', 'mse', 'mae']).T.round(4)
        else:
            df = pd.DataFrame([r2list, mselist, maelist, nlpdlist, nrmseplist, picp68list, picp95list], columns=names,
                              index=['r2', 'mse', 'mae', 'nlpd', 'nrmse', 'picp68', 'picp95']).T.round(4)
        if save_path:
            df.to_csv(save_path)
        return df
    else:
        r2 = np.corrcoef(refer[:, index], predicted[:, index])[0][1] ** 2
        mae = mean_absolute_error(refer[:, index], predicted[:, index])
        mse = mean_squared_error(refer[:, index], predicted[:, index])
        if sigmas is not None:
            nlpd = nlpd_metric(refer[:, index], predicted[:, index], sigmas[:, index])
            nrmse = nrmse_p_metric(refer[:, index], predicted[:, index], sigmas[:, index])
            picp68 = picp_metric(refer[:, index], predicted[:, index], sigmas[:, index], alpha=0.68268)
            picp95 = picp_metric(refer[:, index], predicted[:, index], sigmas[:, index], alpha=0.95450)
            return r2, mae, mse, nlpd, nrmse, picp68, picp95
        return r2, mae, mse


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
    axs[0].imshow(predicted.reshape(refer.shape)[:, :, output_index], cmap='gray')
    axs[0].set_title("Predicted")
    axs[1].imshow(refer[:, :, output_index], cmap='gray')
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


def plot_analysis_hist2d(refer, predicted, names=None, index=0, mask=None, title=None, bins=100, save_path=None):
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

    if mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)
        refer_flat = refer_flat[~masked_rows, :]
        predicted_flat = predicted_flat[~masked_rows, :]

    fig, axs = plt.subplots(3, 4, figsize=(19, 15), constrained_layout=True)
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
            fig.supxlabel(r'$x_{true}$', fontsize='xx-large')
            fig.supylabel(r'$\left(x_{pred} - x_{true}\right) / x_{true}$', fontsize='xx-large')
        elif index == 1:
            fig.supxlabel(r'$x_{true}$', fontsize='xx-large')
            fig.supylabel(r'$x_{pred}$', fontsize='xx-large')
        else:
            raise ValueError

    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])

    plt.colorbar(plot_params[3], ax=axs, shrink=1)

    if save_path:
        plt.savefig(save_path + ".png")

    return fig, axs


def plot_analysis_hist2d_up(refer, predicted_mu, predicted_sigma, names=None, index=0, mask=None, title=None, bins=100,
                            save_path=None, number_of_samples=None, outliers=None):
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

    refer_flat = refer.reshape(-1, 11)
    predicted_mu_flat = predicted_mu.reshape(-1, 11)
    predicted_sigma_flat = predicted_sigma.reshape(-1, 11)

    zero_rows = np.any(predicted_sigma_flat == 0, axis=1)

    if np.any(zero_rows) and mask is not None and index == 0:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)

        masked_rows = masked_rows | zero_rows
    elif np.any(zero_rows) and index == 0:
        masked_rows = zero_rows
    elif mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)
    else:
        masked_rows = np.any(np.zeros_like(refer_flat, dtype=np.bool_), axis=1)

    refer_flat = refer_flat[~masked_rows, :]
    predicted_mu_flat = predicted_mu_flat[~masked_rows, :]
    predicted_sigma_flat = predicted_sigma_flat[~masked_rows, :]

    print(f'Objects with sigma=0: {np.any(predicted_sigma_flat == 0, axis=1).sum()}')

    if number_of_samples:
        indices = np.random.choice(refer_flat.shape[0], number_of_samples, replace=False)
        refer_flat = refer_flat[indices]
        predicted_mu_flat = predicted_mu_flat[indices]
        predicted_sigma_flat = predicted_sigma_flat[indices]

    fig, axs = plt.subplots(3, 4, figsize=(19, 15), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axs.flat[:-1]):
        if index == 0:
            X, Y = refer_flat[:, i], (refer_flat[:, i] - predicted_mu_flat[:, i]) / predicted_sigma_flat[:, i]
        elif index == 1:
            X, Y = refer_flat[:, i] - predicted_mu_flat[:, i], predicted_sigma_flat[:, i]
        elif index == 2:
            X, Y = refer_flat[:, i], predicted_sigma_flat[:, i]
        else:
            raise ValueError

        if outliers:
            y_bot = np.percentile(Y, outliers[0])
            y_top = np.percentile(Y, outliers[1])
            y_pad = 0.2 * (y_top - y_bot)

            y_min = y_bot - y_pad
            y_max = y_top + y_pad

            mask_range = (Y > y_min) & (Y < y_max)
            X = X[mask_range]
            Y = Y[mask_range]

        ax.set_title(names[i], weight='bold')
        plot_params = ax.hist2d(X, Y, bins=bins, norm=LogNorm())

    if index == 0:
        fig.supxlabel(r'$x_{true}$', fontsize='xx-large')
        fig.supylabel(r'$\left(x_{true} - x_{pred}\right)/ \sigma_{pred}$', fontsize='xx-large')
    elif index == 1:
        fig.supxlabel(r'$x_{true} - x_{pred}$', fontsize='xx-large')
        fig.supylabel(r'$\sigma_{pred}$', fontsize='xx-large')
    elif index == 2:
        fig.supxlabel(r'$x_{true}$', fontsize='xx-large')
        fig.supylabel(r'$\sigma_{pred}$', fontsize='xx-large')
    else:
        raise ValueError

    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])

    plt.colorbar(plot_params[3], ax=axs, shrink=1)

    if save_path:
        plt.savefig(save_path + ".png")

    return fig, axs


def plot_hist_params(pars_arr, pars_names=None, plot_name=None, bins=100, save_path=None):
    """
    Draw histograms for all parameters at once
    """
    if pars_names is None:
        pars_names = ['Field Strength',
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

    pars_arr = pars_arr.reshape(-1, 11)

    fig, axs = plt.subplots(3, 4, figsize=(12, 15), constrained_layout=True)

    for i, ax in enumerate(axs.flat[:-1]):
        ax.set_yscale('log')
        ax.set_title(pars_names[i], weight='bold')
        ax.hist(pars_arr[:, i], bins=bins)

    fig.set_facecolor('xkcd:white')
    fig.delaxes(axs[2][3])
    if save_path:
        fig.savefig(save_path + ".png")
    if plot_name:
        plt.suptitle(plot_name, fontsize=18)

    return fig, axs


def plot_hist_params_comparison(prediction, refer, pars_names=None, mask=None, plot_name=None, bins=100,
                                save_path=None):
    if pars_names is None:
        pars_names = ['Field Strength',
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

    pars_arr1 = prediction.reshape(-1, 11)
    pars_arr2 = refer.reshape(-1, 11)

    if mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)
        pars_arr1 = pars_arr1[~masked_rows, :]
        pars_arr2 = pars_arr2[~masked_rows, :]

    fig, axs = plt.subplots(3, 4, figsize=(19, 15), constrained_layout=True)

    for i, ax in enumerate(axs.flat[:-1]):
        ax.set_yscale('log')
        ax.set_title(pars_names[i], weight='bold')

        sns.histplot(
            pars_arr1[:, i], ax=ax, bins=bins, color='blue', label="predicted"
        )
        sns.histplot(
            pars_arr2[:, i], ax=ax, bins=bins, color='red', alpha=0.6, label='refer'
        )

    fig.set_facecolor('xkcd:white')

    h, l = ax.get_legend_handles_labels()
    axs[2][3].legend(h, l, borderaxespad=0)
    axs[2][3].axis("off")

    if plot_name:
        plt.suptitle(plot_name, fontsize=18)

    if save_path:
        plt.savefig(save_path + ".png")

    return fig, axs


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
