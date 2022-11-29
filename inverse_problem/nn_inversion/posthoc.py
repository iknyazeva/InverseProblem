import matplotlib.pyplot as plt
#import seaborn as sns
from matplotlib.colors import LogNorm
import os
import torch
from inverse_problem.nn_inversion import normalize_spectrum
from scipy.stats import norm
from inverse_problem import HinodeME, me_model
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
import numpy as np
import pandas as pd
import seaborn as sns
import pylab
from scipy import stats
import matplotlib as mpl


def open_param_file(path, normalize=True, print_params=True, **kwargs):
    """

    Args:
        print_params (object):

    1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13

    ext= 1 Field_Strength
    ext= 2 Field_Inclination
    ext= 3 Field_Azimuth
    ext= 6 Doppler_Width
    ext= 8 Damping
    ext= 7 Line_Strength
    ext= 9 Source_Function
    ext= 10 Source_Function_Gradient
    ext= 5 Doppler_Shift2
    ext= 12 Stray_Light_Fill_Factor
    ext= 13 Stray_Light_Shift

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


def open_param_unc_file(path, normalize=True, print_params=True, **kwargs):
    """

    Args:
        print_params (object):

    ext= 14 Field_Strength_Error
    ext= 15 Field_Inclination_Error
    ext= 16 Field_Azimuth_Error
    ext= 19 Doppler_Width_Error
    ext= 21 Damping_Error
    ext= 20 Line_Strength_Error
    ext= 22 Source_Function_Error
    ext= 23 Source_Function_Gradient_Error
    ext= 18 Doppler_Shift2_Error
    ext= 25 Stray_Light_Fill_Factor_Error
    ext= 26 Stray_Light_Shift_Error

    """
    refer = fits.open(path)
    param_list = [14, 15, 16, 19, 21, 20, 22, 23, 18, 25, 26]
    names = [refer[i].header['EXTNAME'] for i in param_list]
    if print_params:
        print('Open file with 36 available parameters, 11 will be selected')
        print('\n'.join(names))
    data = np.array([refer[i].data for i in param_list], dtype='float').swapaxes(0, 2).swapaxes(0, 1)
    if normalize:
        shape = data.shape
        data = normalize_output(data.reshape(-1, 11), **kwargs).reshape(shape)

    return data, names


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



def read_spectrum_from_one_file(sp_path, cont_path):
    # real_samples = np.zeros((512, 873, 224))
    # cont = np.zeros((512, 873))
    real_samples = fits.open(sp_path)[0].data
    cont = fits.open(cont_path)[0].data
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
                 'Source Function (SF)',
                 'Cont. SF Gradient',
                 'Doppler Shift (DS)',
                 'Filling Factor',
                 'Stray Light DS']

    plt.figure(figsize=(12, 8))
    plt.axis('off')
    for i in range(11):
        plt.subplot(3, 4, i + 1)
        plt.title(names[i], weight='bold', fontsize=14)
        plt.imshow(data[:, :, i], cmap='RdGy')
        plt.axis('off')
    plt.tight_layout()


def plot_params2(data, names, title=None, save_path=None, color_map='gray',inverse=False):
    """Draw all 11 parameters at once
    data: np array (:, :, 11)
    """
    fig, axs = plt.subplots(3, 4, figsize=(15, 9))

    x1_limits = [0]*11
    x2_limits = [1]*11
    if inverse:
        # x2_limits = [0]*11
        # x1_limits = [1]*11
        x2_limits = [0] * 11
        x1_limits = [1] * 11

    bar_labels = ['Gauss', 'Degree', 'Degree', 'm$\AA$', 'Dopplerwidths', 'Normalized intensity', 'Normalized intensity', 'Normalized intensity', 'kmps', 'Normalized intensity', 'kmps']
    # bar_labels = ['Certainty, %']*11

    for i, ax in enumerate(axs.flat[:-1]):
        x_min, x_max = np.min(data[:, :, i]), np.max(data[:, :, i])
        ax.set_title(names[i], weight='bold', fontsize=14)
        params = ax.imshow(data[:, :, i], cmap=truncate_colormap(color_map, x1_limits[i], x2_limits[i]), vmin=x_min, vmax=x_max, aspect='auto')
        ax.axis('off')
        clb = fig.colorbar(params, ax=ax)
        clb.ax.get_yaxis().labelpad = 15
        clb.ax.set_ylabel(bar_labels[i], rotation=270, fontsize=12)


    fig.set_facecolor('xkcd:white')
    axs[2][3].axis("off")
    fig.tight_layout(pad=2)
    fig.savefig(save_path + title + "_" + 'params' + ".png")
    fig.savefig(save_path + title + "_" + 'params' + "_H.png", dpi=100)
    # plt.show()


def plot_params3(data1, data2, data3, names, title=None, save_path=None, color_map='gray',inverse=False):
    """Draw all 11 parameters at once
    data: np array (:, :, 11)
    """
    m, n = 5, 6
    y_labels = [r'$x_{true}$', r'$x_{pred}$', r'$\sigma_{pred}$']
    fig, axs = plt.subplots(n, m, figsize=(25, 23))
    # fig, axs = plt.subplots(n, m, figsize=(20, 35))
    x1_limits = [0]*11
    x2_limits = [1]*11
    if inverse:
        x2_limits = [0] * 11
        x1_limits = [1] * 11
    for i in range(10):
        for j in range(3):
            x1_limits = [1] * 11
            bar_labels = ['Gauss', 'Degree', 'Degree', 'm$\AA$', 'Dopplerwidths', 'Relative units',
                          'Normalized intensity', 'Normalized intensity', 'kmps', 'Normalized intensity', 'kmps']
            data_r = data1[:, :, i]
            data_p = data2[:, :, i]
            x_min_r, x_max_r = np.min(data_r), np.max(data_r)
            x_min_p, x_max_p = np.min(data_p), np.max(data_p)
            x_min, x_max = min(x_min_r, x_min_p), max(x_max_r, x_max_p)
            if j % 3 == 0:
                axs[j + 3 * (i//m)][i%m].set_title(names[i], weight='bold', fontsize=20, pad=10)
                params = axs[j + 3 * (i // m)][i % m].imshow(data_r, cmap=truncate_colormap(color_map, x1_limits[i],
                                                                                          x2_limits[i]), vmin=x_min,
                                                             vmax=x_max, aspect='auto')
                axs[j + 3 * (i // m)][i % m].axis('off')
                clb = fig.colorbar(params, ax=axs[j + 3 * (i // m)][i % m])
                clb.ax.tick_params(labelsize=14)
                clb.ax.get_yaxis().labelpad = 20
                clb.ax.set_ylabel(bar_labels[i], rotation=270, fontsize=18)
            elif j % 3 == 1:
                params = axs[j + 3 * (i // m)][i % m].imshow(data_p, cmap=truncate_colormap(color_map, x1_limits[i],
                                                                                          x2_limits[i]), vmin=x_min,
                                                             vmax=x_max, aspect='auto')
                axs[j + 3 * (i // m)][i % m].axis('off')
                clb = fig.colorbar(params, ax=axs[j + 3 * (i // m)][i % m])
                clb.ax.tick_params(labelsize=14)
                clb.ax.get_yaxis().labelpad = 20
                clb.ax.set_ylabel(bar_labels[i], rotation=270, fontsize=18)
            else:
                x1_limits = [0.95] * 11
                data = data3[:, :, i]
                x_min, x_max = np.min(data), np.percentile(data, 99)
                params = axs[j + 3 * (i // m)][i % m].imshow(data, cmap=truncate_colormap(color_map, x1_limits[i],
                                                                                          x2_limits[i]), vmin=x_min,
                                                             vmax=x_max, aspect='auto')
                axs[j + 3 * (i // m)][i % m].axis('off')
                clb = fig.colorbar(params, ax=axs[j + 3 * (i // m)][i % m])
                clb.ax.tick_params(labelsize=14)
                clb.ax.get_yaxis().labelpad = 20
                clb.ax.set_ylabel(bar_labels[i], rotation=270, fontsize=18)

    y_shift, x_shift = 1/(n+0.2), 0.01
    starting_point = 0.08
    font = 24
    fig.text(x_shift, starting_point + y_shift*5, '$x_{true}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    fig.text(x_shift, starting_point + y_shift*4, '$x_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    fig.text(x_shift, starting_point + y_shift*3, r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')

    fig.text(x_shift, starting_point + y_shift*2, '$x_{true}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    fig.text(x_shift, starting_point + y_shift, '$x_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    fig.text(x_shift, starting_point, r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')

    # fig.text(x_shift, 1/(n+1)*9 + shift + 0.005, '$x_{true}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    # fig.text(x_shift, 1/(n+1)*8 + shift, '$x_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    # fig.text(x_shift, 1/(n+1)*7 + shift, r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')

    # fig.text(x_shift, 1/(n+1)*n, '$x_{true}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    # fig.text(x_shift, 1/(n+1)*(n-1), '$x_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    # fig.text(x_shift, 1/(n+1)*(n-2), r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    #
    # fig.text(x_shift, 1/(n+1)*(n-3) - shift, '$x_{true}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    # fig.text(x_shift, 1/(n+1)*(n-4) - shift, '$x_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')
    # fig.text(x_shift, 1/(n+1)*(n-5) - shift - 0.005, r'$\sigma_{pred}$', va='center', rotation='vertical', fontsize=font, weight='bold')

    fig.set_facecolor('xkcd:white')
    axs[-1][-1].axis("off")
    axs[-2][-1].axis("off")
    axs[-3][-1].axis("off")
    axs[-1][-4].set_ylabel('$x_{true}$', fontsize=12)
    fig.tight_layout(pad=3)
    fig.subplots_adjust(left=0.04, bottom=0.01, right=0.98, top=0.95)
    fig.savefig(save_path + title + "_" + "params.png", dpi=80)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def plot_analysis_hist2d_unc(refer, predicted_mu, predicted_sigma, names, index=0, mask=None, title=None, bins=100,
                            save_path=None):
    """
        draws 2d graphs:
        1. (x_true - x_pred)/sigma_pred vs x_true,
        2. (x_true - x_pred) vs sigma_pred,
        3. x_true vs sigma_pred,
    """
    xlabels = [r'$x_{true}$', r'$x_{true} - x_{pred}$', r'$x_{true}$', r'$x_{pred}$', r'$x_{true}$']
    ylabels = [r'$(x_{true} - x_{pred})/ \sigma_{pred}$', r'$x_{pred}$', r'$\sigma_{pred}$', r'$\sigma_{pred}$', r'$(x_{pred} - x_{true})/x_{true}$']
    param_labels = ['Gauss', 'Degree', 'Degree', 'm$\AA$', 'Dopplerwidths', 'Relative units', 'Normalized intensity', 'Normalized intensity', 'kmps', 'Normalized intensity', 'kmps']
    font = 19
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    # fig.suptitle(title, fontsize=font)
    for i, ax in enumerate(axs.flat[:-1]):
        if index == 0:
            X, Y = refer[:, i], (refer[:, i] - predicted_mu[:, i]) / predicted_sigma[:, i]
            # X, Y = refer[:, i], ((1 - 2 * int(i == 1 or i == 2)) * refer[:, i] - predicted_mu[:, i] + 180 * int(i == 1 or i == 2)) / predicted_sigma[:, i]
        elif index == 1:
            X, Y = refer[:, i] - predicted_mu[:, i], predicted_sigma[:, i]
            # X, Y = (1 - 2 * int(i == 1 or i == 2)) * refer[:, i] - predicted_mu[:, i] + 180 * int(i == 1 or i == 2), predicted_sigma[:, i]
        elif index == 2:
            X, Y = refer[:, i], predicted_mu[:, i]
        elif index == 3:
            X, Y = predicted_mu[:, i], predicted_sigma[:, i]
        else:
            X, Y = refer[:, i], (predicted_mu[:, i] - refer[:, i])/refer[:, i]
        if i == 3: bins = bins*3
        elif i == 0: bins = bins*3
        elif i == 7: bins = bins*3
        # if i == 8:  bins = bins*2
        elif i == 9: bins = bins//2
        elif i == 10: bins = bins//3
        ax.set_title(names[i], weight='bold', fontsize=14)
        plot_params = ax.hist2d(X, Y, bins=bins, norm=LogNorm(), cmap=truncate_colormap('Blues', 0.2, 1))
        ymin, ymax = np.percentile(Y, 0.5), np.percentile(Y, 99.5)
        xmin, xmax = np.percentile(X, 0.5), np.percentile(X, 99.5)
        xlim = np.min([abs(xmin), abs(xmax)])
        if index == 0 and i == 3: xmin = 20
        elif index == 0 and i == 4: xmin = 0
        ax.axis(ymin=ymin, ymax=ymax, xmin=-xlim, xmax=xlim)
        ax.get_xaxis().labelpad = 10
        ax.set_xlabel(param_labels[i], fontsize=12)
        ax.get_yaxis().labelpad = 10
        ax.set_ylabel(param_labels[i], fontsize=12)
        # ax.set_yticks(np.round(np.linspace(ymin, ymax, 3), 1))
        # if index == 1 and i == 4: ax.set_xticks([-0.3, 0, 0.3])
        # if index == 1 and i == 6: ax.set_xticks([-5000, 0, 2500])
        # if index == 1 and i == 7: ax.set_xticks([-4000, 0, 5000])
    fig.text(0.52, 0.01, xlabels[index], ha='center', fontsize=font)
    fig.text(0.00, 0.51, ylabels[index], va='center', rotation='vertical', fontsize=font)
    fig.set_facecolor('xkcd:white')
    axs[2][3].axis("off")
    fig.tight_layout(pad=4)
    # fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    cax = plt.axes([0.809, 0.093, 0.015, 0.204])
    # cax = plt.axes([0.855, 0.095, 0.02, 0.34])
    plt.colorbar(plot_params[3], ax=axs, cax=cax, shrink=1)
    fig.savefig(save_path + title + "_" + str(index) + ".png", dpi=400)
    plt.show()


def plot_fitting_curves_unc(refer, predicted_mu, predicted_sigma, names, mask=None, title=None, save_path=None):
    """
        draws fitting curves a, b: each point of a parameter is fitted by the N(a, b)
    """
    param_labels = ['Gauss', 'Degree', 'Degree', 'm$\AA$', 'Dopplerwidths', 'Normalized intensity',
                    'Normalized intensity', 'Normalized intensity', 'kmps', 'Normalized intensity', 'kmps']
    fig, axs = plt.subplots(3, 4, figsize=(15, 12))
    skipping = 25
    for i, ax in enumerate(axs.flat[:-1]):
        X, Y = refer[:, i], (refer[:, i] - predicted_mu[:, i]) / predicted_sigma[:, i]
        # X, Y = refer[:, i], ((1 - 2*int(i == 1 or i == 2)) * refer[:, i] - predicted_mu[:, i] + 180*int(i == 1 or i == 2)) / predicted_sigma[:, i]
        ax.set_title(names[i], weight='bold', fontsize=14)
        grid, a, b = calculate_ab_fit(X, Y, N=80)
        ax.plot(grid[skipping:], a[skipping:], color='orange', label='mean (one model)')
        ax.fill_between(grid[skipping:], a[skipping:], a[skipping:] + b[skipping:], color='orange', alpha=0.1, label='std (one model)')
        ax.fill_between(grid[skipping:], a[skipping:], a[skipping:] - b[skipping:], color='orange', alpha=0.1)
        # ax.legend(loc='upper right')
        xmin, xmax = np.percentile(grid[skipping:], 0.01), np.percentile(grid[skipping:], 99.9)
        ymin, ymax = max(-30, np.percentile(a[skipping:] - b[skipping:], 0.01)), min(30, np.percentile(a[skipping:] + b[skipping:], 99.9))

        if i == 6 or i == 7: ax.set_xticks(np.round(np.linspace(xmin, 0.9 * xmax, 3), 2))
        ax.get_xaxis().labelpad = 10
        ax.set_xlabel(param_labels[i], fontsize=12)
        ax.axis(ymin=ymin, ymax=ymax)
    font = 19
    fig.text(0.5, 0.01, r'$x_{true}$', ha='center', fontsize=font)
    fig.text(0.01, 0.5, r'$(x_{true} - x_{pred})/ \sigma_{pred}$', va='center', rotation='vertical', fontsize=font)
    fig.set_facecolor('xkcd:white')
    h, l = ax.get_legend_handles_labels()
    axs[2][3].legend(h, l, loc='upper left', borderaxespad=0)
    axs[2][3].axis("off")
    pylab.tight_layout(pad=4)
    fig.savefig(save_path + title + "_" + 'stats' + ".png")
    fig.savefig(save_path + title + "_" + 'stats' + "_H.png", dpi=400)
    plt.show()


def calculate_ab_fit(X, Y, N=100):
    data = np.array([X, Y]).T
    sorted_data = data[np.argsort(data[:, 0])].T
    X, Y = sorted_data[0], sorted_data[1]
    a, b, grid = [], [], []
    shape = X.shape[0]
    step = shape // N
    X_limit = np.abs(X[-1] - X[0])/N
    for i in range(N):
        slice = Y[i * step:(i + 1) * step]
        if slice.shape[0] != 0: mu, std = norm.fit(slice)
        else: mu, std = None, None
        if np.abs(X[i * step:(i + 1) * step][-1] - X[i * step:(i + 1) * step][0]) <= X_limit:
            a.append(mu)
            b.append(std)
            if i != N - 1: grid.append(np.mean(X[i*step : (i + 1)*step]))
            else: grid.append(np.max(X[i*step : (i + 1)*step]))
    if len(grid) != len(a):
        raise ValueError(f'grid ({len(grid)}) and a ({len(a)}) must be of the same size')
    return np.array(grid), np.array(a), np.array(b)


def plot_hist_params_comparison(pars_names, pars_arr1, pars_arr2=None, mask=None, plot_name=None, bins=75, save_path=None):
    pars_arr1 = pars_arr1.reshape(-1, 11)
    if pars_arr2 is not None:
        pars_arr2 = pars_arr2.reshape(-1, 11)
    if mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)
        pars_arr1 = pars_arr1[~masked_rows, :]
        if pars_arr2 is not None:
            pars_arr2 = pars_arr2[~masked_rows, :]
    fig, axs = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    fig.suptitle('Sigmas comparison', fontsize=18)
    for i, ax in enumerate(axs.flat[:-1]):
        ax.set_yscale('log')
        ax.set_title(pars_names[i], weight='bold')
        sns.histplot(pars_arr1[:, i], ax=ax, bins=bins, alpha=0.4, label="1")
        if pars_arr2 is not None:
            sns.histplot(pars_arr2[:, i], ax=ax, bins=bins, color='gray', alpha=0.4, label='2')
    fig.set_facecolor('xkcd:white')
    h, l = ax.get_legend_handles_labels()
    axs[2][3].legend(h, l, borderaxespad=0)
    axs[2][3].axis("off")
    if save_path:
        fig.savefig(save_path + "hists_" + plot_name + ".png")
        fig.savefig(save_path + "hists_" + plot_name + "_H.png", dpi=300)



def plot_hists(pars_arr1, pars_arr2, names, bins=50, save_path=None):
    pars_arr1 = pars_arr1.reshape(-1, 11)
    if pars_arr2 is not None:
        pars_arr2 = pars_arr2.reshape(-1, 11)
    fig, axs = plt.subplots(2, 2, figsize=(6, 5.5), constrained_layout=True)
    for i, ax in enumerate(axs.flat[:-1]):
        ax.set_yscale('log')
        ax.set_title(names[i], weight='bold')
        sns.histplot(pars_arr1[:, i], ax=ax, bins=bins, alpha=0.2, label="$x_{pred}$")
        sns.histplot(pars_arr2[:, i], ax=ax, bins=bins, color='grey', alpha=0.2, label='$x_{true}$')
    fig.set_facecolor('xkcd:white')
    h, l = ax.get_legend_handles_labels()
    axs[1][1].legend(h, l, borderaxespad=0)
    axs[1][1].axis("off")
    if save_path:
        fig.savefig(save_path + "hists.png")
        fig.savefig(save_path + "hists_H.png", dpi=300)


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

def plot_real_spectrum(sp_folder, path_to_refer, idx_0, idx_1, norm=True, plot_spectrum=False):
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

    refer, names = open_param_file(path_to_refer, print_params=False, normalize=True)
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



def plot_correlation(refer, predicted_mu, names, bins=100, title=None, save_path=None):
    """
        scatter plots refer vs predicted data
    """
    param_labels = ['Gauss', 'Degree', 'Degree', 'm$\AA$', 'Dopplerwidths', 'Relative units',
                    'Normalized intensity', 'Normalized intensity', 'kmps', 'Normalized intensity', 'kmps']
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    for i, ax in enumerate(axs.flat[:-1]):
        ax.set_title(names[i], weight='bold', fontsize=14)
        # ax.scatter(refer[:, i], predicted_mu[:, i], color='orange', label='mean (one model)')
        X, Y = refer[:, i], predicted_mu[:, i]
        if i == 3: bins = bins * 2
        elif i == 7: bins = bins * 3
        elif i == 9: bins = bins // 2
        # elif i == 10: bins = bins // 2
        plot_params = ax.hist2d(X, Y, bins=bins, norm=LogNorm(), cmap=truncate_colormap('Blues', 0.2, 1))
        ymin, ymax = np.percentile(Y, 0.05), np.percentile(Y, 99.95)
        xmin, xmax = np.percentile(X, 0.05), np.percentile(X, 99.95)
        lb, hb = np.min([ymin, xmin]), np.max([ymax, xmax])
        ax.plot(np.array([lb, hb]), np.array([lb, hb]), 'r--', linewidth=0.8)
        ax.axis(ymin=lb, ymax=hb, xmin=lb, xmax=hb)
        ax.get_yaxis().labelpad = 10
        ax.set_ylabel(param_labels[i], fontsize=12)
        ax.get_xaxis().labelpad = 10
        ax.set_xlabel(param_labels[i], fontsize=12)
    font = 19
    fig.text(0.5, 0.01, r'$x_{true}$', ha='center', fontsize=font)
    fig.text(0.01, 0.5, r'$x_{pred}$', va='center', rotation='vertical', fontsize=font)
    fig.set_facecolor('xkcd:white')
    axs[2][3].axis("off")
    pylab.tight_layout(pad=4)
    cax = plt.axes([0.809, 0.093, 0.015, 0.204])
    plt.colorbar(plot_params[3], ax=axs, cax=cax, shrink=1)
    pylab.tight_layout(pad=4)
    fig.savefig(save_path + title + "_correlation.png", dpi=400)
    plt.show()


def plot_fitting_curves_unc_double(refer, predicted_mu, predicted_sigma, refer2, predicted_mu2, predicted_sigma2, names=None, mask=None, title=None, save_path=None):
    """
        draws fitting curves a, b: each point of a parameter is fitted by the N(a, b)
    """
    if mask is not None:
        mask_flat = mask.reshape(-1, 11)
        masked_rows = np.any(mask_flat, axis=1)
        refer = refer[~masked_rows, :]
        predicted_mu = predicted_mu[~masked_rows, :]
        predicted_sigma = predicted_sigma[~masked_rows, :]
    if names is None:
        names = ['Field Strength',
                 'Field Inclination',
                 'Field Azimuth',
                 'Doppler Width',
                 'Damping',
                 'Line Strength',
                 'Source Function (SF)',
                 'Cont. SF Gradient',
                 'Doppler Shift (DS)',
                 'Filling Factor',
                 'Stray Light DS']
    fig, axs = plt.subplots(3, 4, figsize=(15, 11))
    skipping = 25
    for i, ax in enumerate(axs.flat[:-1]):
        X, Y = refer[:, i], (refer[:, i] - predicted_mu[:, i]) / predicted_sigma[:, i]
        X2, Y2 = refer2[:, i], (refer2[:, i] - predicted_mu2[:, i]) / predicted_sigma2[:, i]
        ax.set_title(names[i], weight='bold', fontsize=14)
        grid, a, b = calculate_ab_fit(X, Y, N=100)
        grid2, a2, b2 = calculate_ab_fit(X2, Y2, N=100)
        ax.plot(grid[skipping:], a[skipping:], color='orange', label='mean (one model)')
        ax.fill_between(grid[skipping:], a[skipping:], a[skipping:] + b[skipping:], color='orange', alpha=0.1, label='std (one model)')
        ax.fill_between(grid[skipping:], a[skipping:], a[skipping:] - b[skipping:], color='orange', alpha=0.1)
        ax.plot(grid2[skipping:], a2[skipping:], color='green', label='mean (ensemble)')
        ax.fill_between(grid2[skipping:], a2[skipping:], a2[skipping:] + b2[skipping:], color='green', alpha=0.1)
        ax.fill_between(grid2[skipping:], a2[skipping:], a2[skipping:] - b2[skipping:], color='green', alpha=0.1, label='std (ensemble)')
        xmin, xmax = np.percentile(grid[skipping:], 0.01), np.percentile(grid[skipping:], 99.9)
        if i == 6 or i == 7: ax.set_xticks(np.round(np.linspace(xmin, 0.9 * xmax, 3), 2))
    font = 19
    fig.text(0.5, 0.01, r'$x_{true}$', ha='center', fontsize=font)
    fig.text(0.01, 0.5, r'$(x_{true} - x_{pred})/ \sigma_{pred}$', va='center', rotation='vertical', fontsize=font)
    fig.set_facecolor('xkcd:white')
    h, l = ax.get_legend_handles_labels()
    axs[2][3].legend(h, l, loc='upper left', borderaxespad=0)
    axs[2][3].axis("off")
    pylab.tight_layout(pad=4)
    fig.savefig(save_path + title + "_" + 'stats' + ".png")
    fig.savefig(save_path + title + "_" + 'stats' + "_H.png", dpi=400)
    plt.show()