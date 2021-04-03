import pytest
from inverse_problem.milne_edington.me import HinodeME, me_model, _compute_spectrum, _prepare_base_model_parameters, \
    _prepare_zero_model_parameters, generate_noise, read_full_spectra
from inverse_problem import get_project_root
import numpy as np
from astropy.io import fits
import os
from pathlib import Path
from astropy.io import fits

# todo test parameters checking from AboutParameters


class TestModelMe:

    def test_prepare_model_parameters(self):
        param_vec = [1250., 15., 20., 30., 1., 50., 0.5, 0.5, 0., 0.25, 0.]
        line_vec = (6302.5, 2.5, 1)
        B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift = _prepare_base_model_parameters(param_vec, line_vec)

        assert 1250 == pytest.approx(B)
        assert 0.26179939 == pytest.approx(theta)
        assert 0.34906585 == pytest.approx(xi)
        assert 3.e-10 == pytest.approx(D)

        B0, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift0 = _prepare_zero_model_parameters(param_vec, line_vec)

        assert 0 == pytest.approx(B0)

    def test_compute_spectrum_base(self):
        line_vec = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
        param_vec = [0., 15., 20., 30., 1., 50., 0.5, 0.5, 0]
        B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift = _prepare_base_model_parameters(param_vec, line_vec)
        spectrum = _compute_spectrum(B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift, line_arg, line_vec)
        expected_I = np.array([0.93969684,
                               0.93402373,
                               0.92755837,
                               0.92015735,
                               0.91164687])
        expected_V = np.array([-0,
                               -0,
                               -0,
                               -0,
                               -0])
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (1, 56, 4)
        assert expected_I == pytest.approx(spectrum[0, :5, 0])
        assert expected_V == pytest.approx(spectrum[0, :5, 1])

    def test_compute_params_base(self):
        line_vec = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
        # with stray shift
        param_vec = [1000., 15., 20., 30., 1., 50., 0, 0.5, 0.5, 0.7, -9]
        B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift = _prepare_base_model_parameters(param_vec, line_vec)
        spectrum = _compute_spectrum(B, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift, line_arg, line_vec)
        B0, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift0 = _prepare_zero_model_parameters(param_vec, line_vec)
        spectrum0 = _compute_spectrum(B0, theta, xi, D, gamma, etta_0, S_0, S_1, Dop_shift0, line_arg, line_vec)
        assert True

    @pytest.fixture
    def param_vec_0(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'parameters_base.fits'
        param_vec = fits.open(filename)[0].data[:2]
        return param_vec

    def test_generate_noise(self, param_vec_0):
        noise = generate_noise(param_vec_0)
        assert noise.shape[0] == param_vec_0.shape[0]
        assert noise.shape[1] == 56
        assert noise.shape[2] ==4
        noise = generate_noise(param_vec_0[0])
        assert noise.shape[0] == 1
        assert noise.shape[1] == 56
        assert noise.shape[2] == 4

    def test_me_model_no_noise(self):
        line_vec = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
        # with stray shift
        param_vec = [1000., 15., 20., 30., 1., 50., 0.5, 0.5, 0, 0.7, -9]
        spectrum = me_model(param_vec, line_arg, line_vec, with_ff=False, with_noise=False)
        spectrum_ff = me_model(param_vec, line_arg, line_vec, with_ff=True, with_noise=False)
        expected_ff_I = [0.91103614, 0.90068347, 0.88866059]
        expected_ff_QUV = [0.00003492, 0.00002593, 0.00785174]
        assert expected_ff_I == pytest.approx(spectrum_ff[0, :3, 0], rel=1e-4)
        assert expected_ff_QUV == pytest.approx(spectrum_ff[0, 0, 1:4], rel=1e-3)

    def test_me_model_with_noise(self, param_vec_0):
        line_vec = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
        spectrum_no_noise = me_model(param_vec_0[0], line_arg, line_vec, with_ff=True, with_noise=False)
        spectrum_with_noise = me_model(param_vec_0[0], line_arg, line_vec, with_ff=True, with_noise=True)
        signal_to_noise = np.mean(spectrum_with_noise / spectrum_no_noise, axis=1)
        assert np.mean(signal_to_noise, axis=1) < 1000

    def test_me_batch(self):
        parameters = np.array([[1000, 15, 20, 30, 1, 50, 0.5, 0.5, 0, 0.7, -9],
                               [1000, 15, 20, 30, 1, 50, 0.5, 0.5, 0, 0.7, -4.25],
                               [1000, 15, 20, 30, 1, 50, 0.5, 0.5, 0, 0.7, 0.5]])

        line_vec = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
        spectrum = me_model(parameters, line_arg, line_vec, with_ff=True, with_noise=False)
        expected_I = [0.91103614, 0.92981541, 0.93938165]
        expected_QUV = [0.00003492, 0.00002593, 0.00785174]
        assert expected_I == pytest.approx(spectrum[:, 0, 0])
        assert expected_QUV == pytest.approx(spectrum[0, 0, 1:], rel=1e-3)
        assert spectrum.shape == (3, 56, 4)


class TestHinodeME:
    # todo не проходит некоторые тесты из-за assertion
    def test_compute_spectrum_zero(self):
        param_vec = [0., 15., 20., 30., 1., 50., 0.5, 0.5, 0., 1., 0.]

        obj = HinodeME(param_vec)
        spectrum = obj.compute_spectrum(with_ff=False, with_noise=False)
        expected_I = np.array([0.93969684,
                               0.93402373,
                               0.92755837,
                               0.92015735,
                               0.91164687])
        expected_V = np.array([-0,
                               -0,
                               -0,
                               -0,
                               -0])
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (1, 56, 4)
        assert expected_I == pytest.approx(spectrum[:5, 0])
        assert expected_V == pytest.approx(spectrum[:5, 1])

        spectrum_ff = obj.compute_spectrum(with_ff=True, with_noise=False)
        assert isinstance(spectrum, np.ndarray)
        assert spectrum_ff.shape == (1, 56, 4)
        assert expected_I == pytest.approx(spectrum_ff[:5, 0])
        assert expected_V == pytest.approx(spectrum_ff[:5, 1])

    def test_compute_spectrum_1250(self):
        param_vec = [1250., 15., 20., 30., 1., 50., 0.5, 0.5, 0., 1., 0.]

        obj = HinodeME(param_vec)
        spectrum = obj.compute_spectrum(with_ff=False, with_noise=False)
        expected_I = np.array([0.93729352,
                               0.93119325,
                               0.92421034,
                               0.91618101,
                               0.90690729])
        expected_V = np.array([0.00007888,
                               0.00009473,
                               0.00011456,
                               0.00013955,
                               0.00017127])
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (1, 56, 4)
        assert expected_I == pytest.approx(spectrum[:5, 0], rel=1e-4)
        assert expected_V == pytest.approx(spectrum[:5, 1], rel=1e-4)

        spectrum_ff = obj.compute_spectrum(with_ff=True, with_noise=False)

        assert isinstance(spectrum_ff, np.ndarray)
        assert spectrum_ff.shape == (1, 56, 4)
        assert expected_I == pytest.approx(spectrum_ff[:5, 0], rel=1e-4)
        assert expected_V == pytest.approx(spectrum_ff[:5, 1], rel=1e-4)

    def test_compute_spectrum_ff(self):
        param_vec = [1000., 15., 20., 30., 1., 50., 0.5, 0.5, 0., 0.25, 0.]

        obj = HinodeME(param_vec)
        spectrum = obj.compute_spectrum(with_ff=True, with_noise=False)
        expected_I = np.array([0.93931485,
                               0.93357398,
                               0.92702653,
                               0.91952583,
                               0.91089422])
        expected_V = np.array([0.00001247,
                               0.00001497,
                               0.00001811,
                               0.00002207,
                               0.00002711])
        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == (1, 56, 4)
        assert expected_I == pytest.approx(spectrum[:5, 0], rel=1e-4)
        assert expected_V == pytest.approx(spectrum[:5, 1], rel=1e-3)
        assert True

    def test_from_refer(self):
        filename = Path(os.getcwd()).parent / 'data' / "20170905_030404.fits"
        refer = fits.open(filename)
        obj = HinodeME.from_refer(100, 200, refer)
        spectrum = obj.compute_spectrum(with_ff=True, with_noise=False)
        assert spectrum.shape == (1, 56, 4)

    def test_from_parameters_base(self):
        filename = Path(os.getcwd()).parent / 'data' / 'parameters_base.fits'
        parameter_base = fits.open(filename)[0].data
        obj = HinodeME.from_parameters_base(0, parameter_base)
        spectrum = obj.compute_spectrum(with_ff=True, with_noise=False)
        spectrum_noised = obj.compute_spectrum(with_ff=True, with_noise=True)
        assert True

    def test_read_full_spectra(self):
        cont_scale = 40000
        path_to_data = os.path.join(get_project_root(), 'data', '20170905_030404\\')
        full_spectra = read_full_spectra(cont_scale, files_path=path_to_data)
        flist = [path_to_data+'SP3D20170905_030404.4C.fits', ]
        full_spectra = read_full_spectra(cont_scale, files_list=flist)
        assert True


