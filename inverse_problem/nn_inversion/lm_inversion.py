import scipy
import time
from tqdm import tqdm
import pandas as pd
from scipy.optimize import least_squares, curve_fit
import scipy.special
import numpy as np
from inverse_problem.milne_edington.me import me_model
from inverse_problem.nn_inversion.posthoc import real_spectra, plot_real_spectrum, plot_model_spectrum


class NlInversion:
    line_vec = (6302.5, 2.5, 1)
    line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
    with_ff = True

    def __init__(self, path_to_refer, path_to_sp):
        self.path_to_refer = path_to_refer
        self.path_to_sp = path_to_sp
        self.idx_0 = None
        self.idx_1 = None
        self.real_spectrum = None
        self.real_cont = None
        self.model_cont = None
        self.param_vec = None
        self.param_vec_norm = None

    def get_real_spectrum(self, idx_0, idx_1, plot_spectrum=True, norm=True):
        self.idx_0 = idx_0
        self.idx_1 = idx_1
        real_line, cont_int, params = plot_real_spectrum(self.path_to_sp, self.path_to_refer, idx_0, idx_1,
                                                    plot_spectrum=plot_spectrum, norm=norm)
        self.real_spectrum = real_line
        self.real_cont = cont_int
        self.param_vec = params
        self.model_cont = np.sum(params[6:8])
        self.param_vec_norm = norm_to_cont(params)
        return real_line, cont_int, params

    def get_model_spectrum(self, idx_0, idx_1, norm=False, plot_spectrum=True, with_noise=True):
        self.idx_0 = idx_0
        self.idx_1 = idx_1
        model_line, param_vec, cont = plot_model_spectrum(self.path_to_refer, idx_0, idx_1, with_noise=with_noise, norm=norm)
        self.param_vec = param_vec
        self.model_cont = np.sum(param_vec[6:8])
        self.param_vec_norm = norm_to_cont(param_vec)
        return model_line, norm_to_cont(param_vec), cont

    def inverse_line(self, line, initial="mean", start_mean=None, lm=False):

        assert isinstance(line, np.ndarray), "Line should be np.ndarray"
        if line.ndim == 3:
            line = line.reshape(1, -1, order='F').flatten()
        assert line.shape == (224,), "Expected flatten spectrum"
        start = time.time()
        upper_bounds = np.array([5000, 180, 180, 90, 1.5, 100, 1, 1, 10, 1, 10])
        lower_bounds = np.array([0, 0, 0, 20, 0, 0.01, 0, 0, -10, 0, -10])
        if start_mean is not None:
            start_mean = start_mean
        elif initial == 'mean':
            start_mean = [1000, 45, 45, 30, 1, 10, 0.5, 0.5, 0, 0.5, 0]
        elif initial == "emp_mean":
            start_mean = [530, 91, 89, 33, 0.31, 12, 0.58, 0.41, 0.04, 0.5, 0.36]
        elif initial == 'random':
            start_mean = lower_bounds + np.random.random(size=11) * (upper_bounds - lower_bounds)
        else:
            raise
        bounds = (lower_bounds, upper_bounds)
        try:
            if lm:
                out = curve_fit(me_curve_fit_prepare, xdata=self.line_arg, ydata=line, method="lm", p0=start_mean)
            else:
                out = curve_fit(me_curve_fit_prepare, xdata=self.line_arg, ydata=line, bounds=bounds, p0=start_mean)
        except:
            out = (start_mean, np.zeros((11, 11)))

        elapsed = (time.time() - start)

        pcov = out[1]
        perr = np.sqrt(np.diag(pcov))

        return (out[0], perr), elapsed

    def inv_compare_model_noise(self, param_vec, n_iter=5, lm=False, initial="emp_mean", start_mean=None):

        preds = []
        errs = []
        if initial == "true":
            start_mean = norm_to_cont(param_vec)
        for i in tqdm(range(n_iter)):
            line = me_model(param_vec, self.line_arg, self.line_vec, with_ff=self.with_ff, with_noise=True, norm=True)
            (pred, perr), elapsed = self.inverse_line(line, lm=lm, initial=initial, start_mean=start_mean)
            preds.append(pred)
            errs.append(perr)
        mean_preds = np.array(preds).mean(axis=0)
        std_preds = np.array(preds).std(axis=0)
        preds_err_mean = np.array(errs).mean(axis=0)
        data = np.concatenate([param_vec.reshape(11, -1), mean_preds.reshape(11, -1), std_preds.reshape(11, -1),
                               preds_err_mean.reshape(11, -1)],
                              axis=1)
        columns = ['target', 'mean_predicted', 'std_predicted', 'mean_pred_unc']
        return pd.DataFrame(data, columns=columns)


    def inv_line_compare_inits(self, line, param_vec, n_iter=5, lm=False):
        random_preds = []
        random_errs = []
        (mean_pred, mean_err), elapsed = self.inverse_line(line, lm=lm, initial="emp_mean")
        (true_pred, true_err), elapsed = self.inverse_line(line, lm=lm, start_mean=norm_to_cont(param_vec))
        for i in tqdm(range(n_iter)):
            (pred, perr), elapsed = self.inverse_line(line, lm=lm, initial="random")
            random_preds.append(pred)
            random_errs.append(perr)
        random_mean_preds = np.array(random_preds).mean(axis=0)
        random_std_preds = np.array(random_preds).std(axis=0)
        random_preds_err_mean = np.array(random_errs).mean(axis=0)
        data = np.concatenate([param_vec.reshape(11, -1), mean_pred.reshape(11, -1), mean_err.reshape(11, -1),
                               true_pred.reshape(11, -1), true_err.reshape(11, -1), random_mean_preds.reshape(11, -1),
                               random_std_preds.reshape(11, -1), random_preds_err_mean.reshape(11, -1)],
                              axis=1)
        columns = ['target', 'pred(init:mean)', 'err(init:mean)', 'pred(init:true)', 'err(init:true)',
                   'mean_pred(init:random)', 'std_pred(init:random)', 'mean_err(init:random)']
        return pd.DataFrame(data, columns=columns)

    def inverse_one_line_several_random_start(self, line, param_vec, n_iter, lm=False):
        preds = []
        preds_err = []

        for i in tqdm(range(n_iter)):
            (pred, perr), _ = self.inverse_line(line, lm=lm, initial="random")
            preds.append(pred)
            preds_err.append(preds_err)
        mean_preds = np.array(preds).mean(axis=0)
        err_preds = np.array(preds).std(axis=0)
        preds_err_mean = np.array(preds_err).mean(axis=0)
        data = np.concatenate([param_vec.reshape(11, -1), mean_preds.reshape(11, -1), err_preds.reshape(11, -1),
                               preds_err_mean.reshape(11, -1)],
                              axis=1)
        return pd.DataFrame(data, columns=['target', 'mean_predicted', 'std_predicted', 'mean_pred_uncertanty'])


def norm_to_cont(param_vec):
    vec = param_vec.copy()
    vec[6:8] /= vec[6:8].sum()
    return vec


def lm_inversion(spectrum, initial='mean', line_arg=None, line_vec=None):
    assert spectrum.shape == (1, 56, 4), "Spectrum should be reshaped as (1,56,4)"

    if not line_vec:
        line_vec = (6302.5, 2.5, 1)
    if not line_arg:
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])

    # lower_bounds = np.array([0, 0, 0, 5, 0.1, 0.01, 0.01, 0.01, -20, 0, -20])
    # upper_bounds = ([5000, 180, 180, 100, 2, 100, 1, 1, 20, 1, 20])
    upper_bounds = [5000, 180, 180, 90, 1.5, 100, 1, 1, 10, 1, 10]
    lower_bounds = [0, 0, 0, 20, 0, 0.01, 0, 0, -10, 0, -10]

    if initial == 'mean':
        x0 = [1000, 45, 45, 30, 1, 10, 0.5, 0.5, 0, 0.5, 0]
    elif initial == "emp_mean":
        x0 = [530, 91, 89, 33, 0.31, 12, 0.58, 0.41, 0.04, 0.5, 0.36]
    elif initial == 'random':
        x0 = lower_bounds + np.random.random(size=11) * (upper_bounds - lower_bounds)
    else:
        raise

    fun = lambda x: np.power(me_model(x, line_arg=line_arg, line_vec=line_vec, with_ff=True,
                                      with_noise=False, norm=True) - spectrum, 2).flatten()
    params = scipy.optimize.least_squares(fun, x0=x0, method='lm')

    return params.x


def me_curve_fit_prepare(line_arg, *params):
    param_vec = np.array([*params])
    line_vec = (6302.5, 2.5, 1)
    param_vec = np.array([*params])
    lines = me_model(param_vec, line_arg, line_vec, with_ff=True, with_noise=False, norm=True)
    return lines.reshape(1, -1, order='F').flatten()


def inverse_one_line(line, param_vec, lm=True, initial="emp_mean",
                     print_time=False, display_results=False):
    """

    Args:
        display_results:
        print_time:
        line (np.ndarray):
        param_vec (list of float or ndarray size 11):
        lm (bool): if using Levenber-Marquat or not
        initial (start value):

    Returns:
        inversion output and dataframe with result and expected
    """
    assert isinstance(line, np.ndarray), "Line should be np.ndarray"
    if line.ndim == 3:
        line = line.reshape(1, -1, order='F').flatten()
    assert line.shape == (224,), "Expected flatten spectrum"
    start = time.time()

    upper_bounds = np.array([5000, 180, 180, 90, 1.5, 100, 1, 1, 10, 1, 10])
    lower_bounds = np.array([0, 0, 0, 20, 0, 0.01, 0, 0, -10, 0, -10])
    if initial == 'mean':
        start_mean = [1000, 45, 45, 30, 1, 10, 0.5, 0.5, 0, 0.5, 0]
    elif initial == "emp_mean":
        start_mean = [530, 91, 89, 33, 0.31, 12, 0.58, 0.41, 0.04, 0.5, 0.36]
    elif initial == 'random':
        start_mean = lower_bounds + np.random.random(size=11) * (upper_bounds - lower_bounds)
    else:
        raise
    # start_mean = [530, 91, 89, 33, 0.31, 12, 27083, 19567, 0.04, 0.5, 0.36]
    bounds = (lower_bounds, upper_bounds)
    line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - 6302.5)
    try:
        if lm:
            out = curve_fit(me_curve_fit_prepare, xdata=line_arg, ydata=line, method="lm", p0=start_mean)
        else:
            out = curve_fit(me_curve_fit_prepare, xdata=line_arg, ydata=line, bounds=bounds, p0=start_mean)
    except:
        out = (start_mean, np.zeros(11, 11))

    elapsed = (time.time() - start)
    if print_time:
        print(f"Время обсчета одного пикселя {elapsed:.2f} секунд, а снимка: {512 * 873 * elapsed / 3600:.2f} часов")
    pcov = out[1]
    perr = np.sqrt(np.diag(pcov))
    if display_results:
        data = np.concatenate([param_vec.reshape(11, -1), out[0].reshape(11, -1), perr.reshape(11, -1)], axis=1)
        df = pd.DataFrame(data, columns=['target', 'predicted', 'sigmas'])
        return out, df
    else:
        return out


def inverse_one_line_several_random_start(line, param_vec, n_iter):
    outs = []
    for i in tqdm(range(n_iter)):
        out = inverse_one_line(line, param_vec, lm=False, initial="random")
        outs.append(out)
    mean_preds = np.array([out[0] for out in outs]).mean(axis=0)
    err_preds = np.array([out[0] for out in outs]).std(axis=0)
    data = np.concatenate([param_vec.reshape(11, -1), mean_preds.reshape(11, -1), err_preds.reshape(11, -1)], axis=1)
    return pd.DataFrame(data, columns=['target', 'mean_predicted', 'std_predicted'])


def generate_model_lines(param_vec, n_lines=10):
    model_noised_lines = []
    line_vec = (6302.5, 2.5, 1)
    line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
    for i in range(n_lines):
        line_vec = (6302.5, 2.5, 1)
        lines = me_model(param_vec, line_arg, line_vec, with_ff=True, with_noise=True, norm=True)
        model_noised_lines.append(lines.reshape(1, -1, order='F').flatten())
    return model_noised_lines


def predict_full_image(line, parameter, **kwargs):
    """ Predicts full image
    Args:
        line: array of size (n, 512)
        parameter (int): index of parameter to predict
    """
    output = np.zeros(line[0].shape[:2])
    for i in range(output.shape[0]):
        for t in range(output.shape[1]):
            predicted = lm_inversion(line[0][i, t].reshape((1, 56, 4)), **kwargs)
            output[i, t] = predicted[parameter]
    return output
