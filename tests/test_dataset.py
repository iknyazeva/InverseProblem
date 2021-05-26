import pytest
import astropy.io.fits as fits
import numpy as np
import os
from pathlib import Path
from inverse_problem.milne_edington.data_utils import get_project_root
from inverse_problem.nn_inversion import SpectrumDataset, ToTensor, NormalizeStandard, Rescale, FlattenSpectrum
from inverse_problem.nn_inversion import mlp_transform_standard, mlp_transform_rescale


class TestSpectrumDataset:
    @pytest.fixture
    def sample_from_database(self):
        # filename = '/Users/irinaknyazeva/Projects/Solar/InverseProblem/data/parameters_base.fits'
        project_path = get_project_root()
        filename =  project_path / 'data' / 'small_parameters_base.fits'
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        sample = sobj[0]
        return sample
    def test_init_dataset(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        param_array = fits.open(filename)[0].data[:10]
        sobj = SpectrumDataset(data_arr=param_array)
        assert True

    def test_init_database_dataset(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'parameters_base.fits'
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        assert sobj.param_source.shape[1] == 11
        assert isinstance(sobj[0]['X'][1], float)
        assert isinstance(sobj[0]['X'][0], np.ndarray)
        assert isinstance(sobj[0]['Y'], np.ndarray)
        assert sobj.__len__() == sobj.param_source.shape[0]
        assert 224 == sobj[0]['X'][0].size

    def test_init_refer_dataset(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'hinode_source'/'20140926_170005.fits'
        source = 'refer'
        sobj = SpectrumDataset(param_path=filename, source=source)
        assert isinstance(sobj.param_source, list)
        assert isinstance(sobj[0]['X'][1], float)
        assert 224 == sobj[0]['X'][0].size

    def test_dataset_with_scale_transforms(self):
        project_path = Path(__file__).resolve().parents[1]
        filename = os.path.join(project_path, 'data/parameters_base.fits')
        source = 'database'
        transform = mlp_transform_rescale()
        sobj = SpectrumDataset(filename, source=source, transform=transform)
        sample = sobj[1]
        assert True
