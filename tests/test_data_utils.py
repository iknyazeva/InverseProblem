import pytest
from pathlib import Path
from inverse_problem.milne_edington import compute_mean_spectrum
import numpy as np
from astropy.io import fits
from inverse_problem.milne_edington.data_utils import download_from_google_disc, get_project_root

def test_compute_mean_spectrum():

    filename = get_project_root() / 'data' / 'parameters_base.fits'
    batch_size = 1000
    nbatches = 2
    MEAN, STD = compute_mean_spectrum(filename, batch_size=batch_size, nbatches=nbatches)
    #assert 224 == len(MEAN)

def test_get_project_root():
    root_path = get_project_root()
    root_path1 = Path(__file__).parent.parent
    assert root_path1 == root_path

def test_download_from_google_disc():
    file_id = '19jkSXHxAPWZvfgo5oxSmvEagme5YLY33'
    dest_path =  get_project_root() / 'data' / 'small_parameters_base.fits'
    download_from_google_disc(fileid=file_id, dest=dest_path)
    filepath = Path(dest_path)
    parameter_base = fits.open(dest_path)[0].data
    assert True