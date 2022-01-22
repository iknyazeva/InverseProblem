import pytest
import astropy.io.fits as fits
import numpy as np
import torch
import os
from pathlib import Path
from inverse_problem.milne_edington.data_utils import get_project_root
from inverse_problem.nn_inversion import SpectrumDataset, ToTensor, Rescale, FlattenSpectrum
from inverse_problem.nn_inversion import PregenSpectrumDataset
from inverse_problem.nn_inversion import conv1d_transform_rescale, mlp_transform_rescale


class TestSpectrumDataset:
    @pytest.fixture
    def sample_from_database(self):
        # filename = '/Users/irinaknyazeva/Projects/Solar/InverseProblem/data/parameters_base.fits'
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        sample = sobj[0]
        return sample

    def test_init_dataset_param_array(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        param_array = fits.open(filename)[0].data[:10]
        sobj = SpectrumDataset(data_arr=param_array)
        assert sobj[0]['X'][0].shape == (56, 4)
        assert sobj.param_source.shape[1] == 11
        assert isinstance(sobj[0]['X'][1], float)
        assert isinstance(sobj[0]['X'][0], np.ndarray)
        assert isinstance(sobj[0]['Y'], np.ndarray)
        assert sobj.__len__() == sobj.param_source.shape[0]
        assert 224 == sobj[0]['X'][0].size
        assert 11 == sobj[0]['Y'].size

    def test_init_database_dataset(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        assert sobj.param_source.shape[1] == 11
        assert isinstance(sobj[0]['X'][1], float)
        assert isinstance(sobj[0]['X'][0], np.ndarray)
        assert isinstance(sobj[0]['Y'], np.ndarray)
        assert sobj.__len__() == sobj.param_source.shape[0]
        assert 224 == sobj[0]['X'][0].size
        assert 11 == sobj[0]['Y'].size

    def test_dataset_with_mlp_scale_transforms(self):
        project_path = Path(__file__).resolve().parents[1]
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        transform = mlp_transform_rescale(angle_transformation=True)
        sobj = SpectrumDataset(param_path=filename, source=source, transform=transform)
        sample = sobj[1]
        assert sample['X'][1].shape[0] == 1
        assert pytest.approx(1, rel=0.1) == sample['X'][0][0]
        assert pytest.approx(1, rel=3) == sample['X'][1]
        assert sample['Y'].shape[0] == 11
        assert sample['Y'][1] == pytest.approx(1, rel=1)

    def test_dataset_with_conv1d_scale_transforms(self):
        project_path = Path(__file__).resolve().parents[1]
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        transform = conv1d_transform_rescale(angle_transformation=True)
        sobj = SpectrumDataset(param_path=filename, source=source, transform=transform)
        sample = sobj[1]
        assert sample['X'][0].shape == (4, 56)
        assert sample['X'][1].shape[0] == 1
        assert pytest.approx(1, rel=0.1) == sample['X'][0][0, 0]
        assert pytest.approx(1, rel=3) == sample['X'][1]
        assert sample['Y'].shape[0] == 11


class TestPregenSpectrumDataset:

    def test_init_dataset_data_array(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        param_array = fits.open(filename)[0].data[:10]
        sobj = PregenSpectrumDataset(data_arr=param_array)
        sample = sobj[0]
        assert sample['X'][0].shape == (56, 4)
        assert isinstance(sample['X'][1], float)
        assert isinstance(sample['X'][0], np.ndarray)
        assert isinstance(sample['Y'], np.ndarray)
        assert 224 == sobj[0]['X'][0].size
        assert sobj[0]['Y'].shape == (11,)

    def test_init_dataset_database(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        transform = mlp_transform_rescale(factors=[1, 1000, 1000, 1000])
        sobj = PregenSpectrumDataset(param_path=filename, source='database', transform=transform)
        sample = sobj[0]
        assert sobj.__len__() == sobj.samples['Y'].shape[0]
        assert isinstance(sample['Y'], torch.Tensor)
        assert sample['X'][0].shape == (224,)
        assert sample['Y'].shape == (11, )


    def test_dataset_with_mlp_scale_transforms(self):
        project_path = Path(__file__).resolve().parents[1]
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        transform = mlp_transform_rescale(angle_transformation=True)
        sobj = PregenSpectrumDataset(param_path=filename, source=source, transform=transform)
        sample = sobj[1]
        assert sample['Y'][1] == pytest.approx(1, rel=1)
        assert sobj.samples['X'][1].shape[1] == 1

    def test_dataset_with_conv1d_scale_transforms(self):
        project_path = Path(__file__).resolve().parents[1]
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        transform = conv1d_transform_rescale(angle_transformation=True)
        sobj = PregenSpectrumDataset(param_path=filename, source=source, transform=transform)
        sample = sobj[1]
        assert sample['X'][0].shape == (4, 56)
        assert isinstance(sample['X'][1], torch.Tensor)
        assert pytest.approx(1, rel=0.1) == sample['X'][0][0, 0]
        assert pytest.approx(1, rel=3) == sample['X'][1]
        assert sample['Y'].shape == (11, )


class TestDatasetRefer:

    def test_init_refer_dataset(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'hinode_source' / '20140926_170005.fits'
        source = 'refer'
        sobj = SpectrumDataset(param_path=filename, source=source)
        assert isinstance(sobj.param_source, list)
        assert isinstance(sobj[0]['X'][1], float)
        assert sobj.__len__() == 446976
        assert 224 == sobj[0]['X'][0].size
        assert 11 == sobj[0]['Y'].size

    def test_init_dataset_refer(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'hinode_source' / '20140926_170005.fits'
        sobj = PregenSpectrumDataset(param_path=filename, source='refer')
        sample = sobj[0]
        assert isinstance(sample['X'][1], float)
        assert isinstance(sample['X'][0], np.ndarray)
        assert isinstance(sample['Y'], np.ndarray)
        assert sobj.__len__() == sobj.param_source.shape[0]
        assert 224 == sobj[0]['X'][0].size
        assert 11 == sobj[0]['Y'].size