import pytest
from inverse_problem.milne_edington.me import HinodeME, me_model, _compute_spectrum, _prepare_base_model_parameters, \
    _prepare_zero_model_parameters, generate_noise, read_full_spectra
from inverse_problem import get_project_root
from inverse_problem.nn_inversion.lm_inversion import lm_inversion, predict_one_pixel, predict_full_image
import numpy as np
from astropy.io import fits
import os
from pathlib import Path
from astropy.io import fits

class TestLM:
    def test_lm_inversion(self):
        line_vector = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vector[0])
        cont_scale = 40000
        path_to_data = os.path.join(get_project_root(), 'data', '20170905_030404\\')
        full_spectra = read_full_spectra(cont_scale, files_path=path_to_data)[0]
        spectrum = full_spectra[0].reshape((512, 56, 4))[0]
        x = lm_inversion(spectrum, line_arg=line_arg, line_vec=line_vector)
        assert True

    def test_predict_one_pixel(self):
        pass


    def test_predict_full_image(self):
        cont_scale = 40000
        path_to_data = os.path.join(get_project_root(), 'data', '20170905_030404\\')
        full_spectra = read_full_spectra(cont_scale, files_path=path_to_data)[0][:10]
        #spectrum = full_spectra[0].reshape((512, 56, 4))[0]
        x = predict_full_image(full_spectra, 0)

