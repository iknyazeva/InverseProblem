import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from inverse_problem.nn_inversion.transforms import normalize_output
from astropy.io import fits
from inverse_problem.milne_edington.data_utils import get_project_root
from inverse_problem.nn_inversion.posthoc import compute_metrics, open_param_file
import numpy as np
import pandas as pd



def test_open_param_file():
    project_path = get_project_root()
    real_path = project_path /'res_experiments'/'predictions'/'20170905_030404.fits'
    refer, names = open_param_file(real_path)
    assert refer.shape == (512, 485, 11)
    assert len(names) == 11

def test_compute_metrics():
    project_path = get_project_root()
    project_path = get_project_root()
    refer_path = project_path / 'res_experiments' / 'predictions' / '20170905_030404.fits'
    refer, names = open_param_file(refer_path)
    pred_path = project_path / 'res_experiments' / 'predictions' / '20170905_030404_common.fits'
    save_path = project_path / 'res_experiments' / 'predictions' / 'test.csv'

    predicted = fits.open(pred_path)
    predicted_data = predicted[0].data
    df = compute_metrics(refer, predicted_data, names, save_path)
    # df.to_csv(save_path)
    assert df.shape == (11, 3)