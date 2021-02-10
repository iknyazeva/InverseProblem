import pytest
from inverse_problem.milne_edington import compute_mean_spectrum
import numpy as np
from astropy.io import fits


def test_compute_mean_spectrum():

    filename = '/Users/irinaknyazeva/Projects/Solar/InverseProblem/data/parameters_base.fits'
    batch_size = 1000
    nbatches = 2
    MEAN, STD = compute_mean_spectrum(filename, batch_size=batch_size, nbatches=nbatches)
    assert 224 == len(MEAN)