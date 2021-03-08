import scipy
import scipy.special
import numpy as np
import os
import astropy.io.fits as fits
from tqdm import tqdm

c = 3e10
mass = 9.1e-28
el_c = 4.8e-10

absolute_noise_levels = [109, 28, 28, 44]


class HinodeME(object):
    """ Compute spectrum I,Q,U,V component based on atmosphere model, class for data loader generation
    """

    # constant parameters for any instance

    line_vector = (6302.5, 2.5, 1)
    line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vector[0])

    def __init__(self, param_vec):
        """
        Parameters initialization with normalized continuum

        Args:
            param_vec (array float): list of parameters for milne_edington model, here we don't use batch option from me_model
                0. B, Field Strength, Hinode range = (0, 5000)
                1. Theta, Field Inclination, Hinode range = (0, 180)
                2. Theta, Field Azimuth, Hinode range = (0, 180)
                3. Doppler Width (Broadening),  Hinode range = (20, 90)
                4. Damping, Hinode range = (0, 1.5)
                5. Line Strength, Hinode range = (0.01, 100)
                6. Source Function S_0
                7. Source Function gradient S_1
                8. Doppler Shift, Hinode range = (-10, +10)
                9. Filling Factor, Hinode range = (0, 1)
                10. Stray light Doppler shift, Hinode range = (-10, +10)

        """
        # parameters for inversion
        param_vec = np.array(param_vec).astype(float)
        # add broadcasting param_vec.shape
        assert True == (len(param_vec) == 11), 'For ME model number of parameters should be 11 '
        # TODO : add assertion for each parameter
        self.param_vector = param_vec.astype(float)
        self.cont = self.param_vector[6] + self.line_vector[2] * self.param_vector[7]
        # self.cont = np.reshape(self.cont, (-1, 1)).astype(float)

    @classmethod
    def from_parameters_base(cls, idx, parameters_base=None):
        # file parameter base already in memory
        assert (idx >= 0) and (idx < parameters_base.shape[0])
        param_vector = parameters_base[idx].astype(float).copy()
        return cls(param_vector)

    @classmethod
    def from_refer(cls, idx_0, idx_1, refer):
        # todo может быть добавить функцию чтения всего спектра?
        # from fits file with maps for each parameters 512(idx_0)*873 (idx_1)
        assert (idx_0 >= 0) and (idx_1 < 512), 'Index should be less than 512 and greater than 0 '
        assert (idx_1 >= 0) and (idx_1 < 873), 'Index should be less than 872 and greater than 0 '

        param_list = [1, 2, 3, 6, 8, 7, 9, 10, 5, 12, 13]
        param_vec = np.array([refer[i].data[idx_0, idx_1] for i in param_list], dtype='float')
        return cls(param_vec)

    def compute_spectrum(self, with_ff=True, with_noise=True) -> np.ndarray:
        """
        Compute Milne Eddington approximation
        Args:
            with_ff (Bool): using model with filling factor
            with_noise (Bool): whether to add noise
        Returns: concatenated spectrum
        """
        lines = me_model(self.param_vector, self.line_arg, self.line_vector, with_ff=with_ff,
                         with_noise=with_noise)

        # this cont level matches better with cont level, calculated from real date (includes noise)
        self.cont = np.amax(lines) * self.cont
        return lines


def me_model(param_vec, line_arg=None, line_vec=None,
             with_ff=True, norm=True, with_noise=True, **kwargs):
    """
    Args:
        line_vec (float,float, float): specific argument for inversion for hinode (6302.5, 2.5, 1)
        line_arg (float ndarray): 1dim array with the spectral line argument, 56 in Hinode case
        param_vec (float ndarray): shape
        with_ff (Boolean): use model with filling factor
    Returns:
    spectrum lines
    """
    if line_vec is None:
        line_vec = (6302.5, 2.5, 1)
    if line_arg is None:
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
    wl0 = line_vec[0] * 1e-8
    g = line_vec[1]
    mu = line_vec[2]

    param_vec = np.array(param_vec, dtype='float')
    if len(param_vec.shape) == 1:
        param_vec = np.reshape(param_vec, (1, -1)).astype(float)

    B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift = _prepare_base_model_parameters(param_vec, line_vec, norm)
    spectrum = _compute_spectrum(B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift, line_arg, line_vec)
    if with_ff:
        B0, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift0 = _prepare_zero_model_parameters(param_vec, line_vec,
                                                                                               norm)
        zero_spectrum = _compute_spectrum(B0, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift0, line_arg, line_vec)
        ff = param_vec[:, 9].reshape(-1, 1, 1)
        quiet_spectrum = ff * spectrum + (1 - ff) * zero_spectrum
    else:
        quiet_spectrum = spectrum

    if with_noise:
        noise = generate_noise(param_vec, mu=mu, **kwargs)
        return quiet_spectrum + noise

    else:
        return quiet_spectrum


def generate_noise(param_vec, absolute_noise_levels=[109, 28, 28, 44], noise_size=None, mu=1):
    """
    Args:
        noise_size: shape of resulted noise
        param_vec (list or ndarray): list of 11 atmosphere parameters
        mu:
        absolute_noise_levels (list of numbers): magical empirical values

    Returns:
        noise as ndarray with the same shape as spectrum generated from param_vec
    """
    param_vec = np.array(param_vec)
    if len(param_vec.shape) == 1:
        param_vec = np.reshape(param_vec, (1, -1))
    if noise_size is None:
        noise_size = (param_vec.shape[0], 56, 4)
    cont = np.array(param_vec[:, 6] + mu * param_vec[:, 7]).reshape(-1, 1, 1)
    noise_level = np.array(absolute_noise_levels).reshape(1, 1, 4)/cont
    noise = noise_level * np.random.normal(size=noise_size)

    return noise


def _prepare_base_model_parameters(param_vec, line_vec, norm=True):
    """
    Args:
        line_vec:
        param_vec (ndarray): vector with 9 parameters atmosphere, size number of examples to number of parameters
        norm (Bool):
    Returns:
        9 separates parameters for ME model
    """
    # parameters for inversion
    wl0 = line_vec[0] * 1e-8
    g = line_vec[1]
    mu = line_vec[2]

    if not isinstance(param_vec, np.ndarray):
        param_vec = np.array(param_vec, dtype='float')
        if len(param_vec.shape) == 1:
            param_vec = np.reshape(param_vec, (1, -1)).astype(float)
    params = param_vec.astype(float).copy()

    if norm:
        cont = param_vec[:, 6] + mu * param_vec[:, 7]
        params[:, 6] /= cont
        params[:, 7] /= cont

    params = params.T
    B = params[0]
    theta = params[1] / 180 * np.pi  # inclination
    xi = params[2] / 180 * np.pi  # azimuth

    D = params[3] * 1e-11  # Doppler width
    gamma = params[4]  # damping
    etta_0 = params[5]  # line strength

    S_0 = params[6]  # Source function
    S_1 = params[7]  # Source function decrement  # Source function decrement

    Dop_shift = params[8] * 1e5 / c * wl0  # Doppler shift
    return B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift


def _prepare_zero_model_parameters(param_vec, line_vec, norm=True):
    """
    Args:
        param_vec: vector with 11 parameters atmosphere
    Returns:
        9 separates parameters for ME model, but with zero B and stray shift
    """
    # parameters for inversion
    if not isinstance(param_vec, np.ndarray):
        param_vec = np.array(param_vec, dtype='float')
        if len(param_vec.shape) == 1:
            param_vec = np.reshape(param_vec, (1, -1)).astype(float)
    params = param_vec[:, :9].copy()
    params[:, 0] = np.zeros(param_vec.shape[0])
    params[:, 8] = param_vec[:, 10]
    return _prepare_base_model_parameters(params, line_vec, norm)


def _compute_spectrum(B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift, line_arg, line_vec):
    """
    Args:
        B (ndarray): vector with size batch_size
        theta (ndarray): vector with size batch_size
        xi (ndarray): vector with size batch_size
        D (ndarray): vector with size batch_size
        gamma (ndarray): vector with size batch_size
        etta_0 (ndarray): vector with size batch_size
        S_0 (ndarray): vector with size batch_size
        S_1 (ndarray): vector with size batch_size
        Dop_shift (ndarray): vector with size batch_size
        line_vec:
        line_arg:
    Returns:
        spectrum lines: I, Q, U, V as a
    """
    wl0 = line_vec[0] * 1e-8
    g = line_vec[1]
    mu = line_vec[2]

    def H_function(v, a):
        return np.real(scipy.special.wofz(v + a * 1j))

    def L_function(v, a):
        return np.imag(scipy.special.wofz(v + a * 1j))

    if not isinstance(B, float):
        x = np.broadcast_to(line_arg, (len(B), len(line_arg))).T
    else:
        x = np.broadcast_to(line_arg, (1, 56)).T

    v = (x * 1e-11 - Dop_shift) / D
    a = gamma
    v_b = B * wl0 * wl0 * el_c / (4 * np.pi * mass * c * c * D)
    # Df = D * c / (wl0 * wl0)

    ka_L = etta_0 * np.sqrt(np.pi)

    etta_p = H_function(v, a) / np.sqrt(np.pi)
    etta_b = H_function(v - g * v_b, a) / np.sqrt(np.pi)
    etta_r = H_function(v + g * v_b, a) / np.sqrt(np.pi)

    rho_p = L_function(v, a) / np.sqrt(np.pi)
    rho_b = L_function(v - g * v_b, a) / np.sqrt(np.pi)
    rho_r = L_function(v + g * v_b, a) / np.sqrt(np.pi)

    h_I = 0.5 * (etta_p * np.sin(theta) * np.sin(theta) + 0.5 * (etta_b + etta_r) * (
            1 + np.cos(theta) * np.cos(theta)))
    h_Q = 0.5 * (etta_p - 0.5 * (etta_b + etta_r)) * np.sin(theta) * np.sin(theta) * np.cos(2 * xi)
    h_U = 0.5 * (etta_p - 0.5 * (etta_b + etta_r)) * np.sin(theta) * np.sin(theta) * np.sin(2 * xi)
    h_V = 0.5 * (etta_r - etta_b) * np.cos(theta)
    r_Q = 0.5 * (rho_p - 0.5 * (rho_b + rho_r)) * np.sin(theta) * np.sin(theta) * np.cos(2 * xi)
    r_U = 0.5 * (rho_p - 0.5 * (rho_b + rho_r)) * np.sin(theta) * np.sin(theta) * np.sin(2 * xi)
    r_V = 0.5 * (rho_r - rho_b) * np.cos(theta)

    k_I = ka_L * h_I
    k_Q = ka_L * h_Q
    k_U = ka_L * h_U
    k_V = ka_L * h_V
    f_Q = ka_L * r_Q
    f_U = ka_L * r_U
    f_V = ka_L * r_V

    det = np.power(1 + k_I, 4) + np.power(1 + k_I, 2) * (
            f_Q * f_Q + f_U * f_U + f_V * f_V - k_Q * k_Q - k_U * k_U - k_V * k_V) - np.power(
        k_Q * f_Q + k_U * f_U + k_V * f_V, 2)

    I = S_0 + S_1 * mu - mu * S_1 * (
            1 - (1 + k_I) * ((1 + k_I) * (1 + k_I) + f_Q * f_Q + f_U * f_U + f_V * f_V) / det)
    V = mu * S_1 * ((1 + k_I) * (1 + k_I) * k_V + f_V * (k_Q * f_Q + k_U * f_U + k_V * f_V)) / det
    U = -S_1 * mu / det * ((1 + k_I) * (1 + k_I) * k_U - (1 + k_I) * (k_V * f_Q - k_Q * f_V) + f_U * (
            k_Q * f_Q + k_U * f_U + k_V * f_V))
    Q = -S_1 * mu / det * ((1 + k_I) * (1 + k_I) * k_Q - (1 + k_I) * (k_U * f_V - k_V * f_U) + f_Q * (
            k_Q * f_Q + k_U * f_U + k_V * f_V))

    return np.transpose(np.array([I, Q, U, V]))


def read_full_spectra(files_path):
    files = os.listdir(files_path)
    files_list = [files_path + i for i in files]
    X_len = len(files_list)
    Y_len = 512
    full_spectra = np.empty((X_len, Y_len, 4 * 56))
    normalization_map = np.empty((X_len, Y_len))

    for X_count in tqdm(range(X_len)):
        spectra_file = fits.open(files_list[X_count])

        real_I = spectra_file[0].data[0][:, 56:].astype('float64') * 2
        real_Q = spectra_file[0].data[1][:, 56:].astype('float64') * 3
        real_U = spectra_file[0].data[2][:, 56:].astype('float64') * 3
        real_V = spectra_file[0].data[3][:, 56:].astype('float64') * 3

        real_sp = np.concatenate((real_I, real_Q, real_U, real_V), axis=1)

        normalization = np.reshape(np.max(real_sp, axis=1), (-1, 1))
        real_sp /= normalization

        normalization_map[X_count] = normalization.flatten()

        full_spectra[X_count] = real_sp

    return full_spectra, normalization
