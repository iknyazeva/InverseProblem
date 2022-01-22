import os
from astropy.io import fits
import numpy as np
import pytest
import torch
from inverse_problem.milne_edington.me import BatchHinodeME
from inverse_problem import get_project_root
from inverse_problem.nn_inversion import SpectrumDataset, ToTensor, Rescale, FlattenSpectrum
from inverse_problem.nn_inversion import  mlp_transform_rescale, conv1d_transform_rescale
from inverse_problem.nn_inversion.transforms import normalize_output, normalize_spectrum, Normalize


class TestTransforms:
    @pytest.fixture
    def sample_from_database(self):
        project_path = get_project_root()
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        sample = sobj[2]
        return sample

    @pytest.fixture
    def batch_sample(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        param_batch = fits.open(filename)[0].data
        obj = BatchHinodeME(param_batch)
        spectrum = obj.compute_spectrum(with_ff=True, with_noise=True)
        batch_sample = {'X': (spectrum, obj.cont), 'Y': obj.param_vector}
        return batch_sample

    def test_normalize_output_range_no_angle(self, sample_from_database):
        y = sample_from_database['Y']
        sample = normalize_output(y, mode='range', logB=True)
        assert sample.min() > 0
        assert sample.max() < 1

    def test_normalize_output_range_angle(self, sample_from_database):
        y = sample_from_database['Y']
        sample = normalize_output(y, mode='range', logB=True, angle_transformation=True)
        assert sample[0][1] == pytest.approx(1, 0.1)
        assert sample.min() >= 0
        assert sample.max() <= 1

    def test_normalize_output_array_no_angle(self):
        project_path = get_project_root()
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        params = fits.open(filename)[0].data[5:11]
        sample = normalize_output(params, mode='range', logB=True)
        assert sample.min() >= 0
        assert sample.max() <= 1

    def test_normalize_output_array_angle(self):
        project_path = get_project_root()
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        params = fits.open(filename)[0].data[5:11]
        sample = normalize_output(params, mode='range', logB=True, angle_transformation=True)
        assert sample.min() >= 0
        assert sample.max() < 1
        assert sample[0, 1] == pytest.approx(np.sin(params[0, 1] * np.pi / 180), 0.01)

    def test_normalize_spectrum_one_sample(self, sample_from_database):
        spectrum = sample_from_database['X'][0]
        factors = [1, 10, 100, 1000]
        spectrum = normalize_spectrum(spectrum, factors=factors)
        assert len(spectrum.shape) == 2
        assert spectrum.shape == (56, 4)

    def test_normalize_spectrum_batch_sample(self, batch_sample):
        spectrum = batch_sample['X'][0]
        factors = [1, 10, 100, 1000]
        spectrum = normalize_spectrum(spectrum, factors=factors)
        assert len(spectrum.shape) == 3
        assert spectrum.shape == (100, 56, 4)

    def test_flatten_one_sample(self, sample_from_database):

        sample = FlattenSpectrum()(sample_from_database)
        assert len(sample['X'][0].shape) == 1

    def test_flatten_batch_sample(self, batch_sample):
        sample = FlattenSpectrum()(batch_sample)
        assert len(sample['X'][0].shape) == 2

    def test_to_tensor(self, sample_from_database):
        to_tensor = ToTensor()
        sample = to_tensor(sample_from_database)
        assert isinstance(sample['Y'], torch.Tensor)
        assert isinstance(sample['X'][0], torch.Tensor)

    def test_to_tensor_batch(self, batch_sample):
        to_tensor = ToTensor()
        sample = to_tensor(batch_sample)
        assert isinstance(sample['Y'], torch.Tensor)
        assert isinstance(sample['X'][0], torch.Tensor)

    # todo: add asserts for different inputs parameters
    def test_rescale(self, sample_from_database):
        rescale = Rescale(factors=[1, 10, 100, 1000])
        sample = rescale(sample_from_database)

        assert 0.53 == pytest.approx(sample['Y'][0, 0], rel=1e-1)
        rescale = Rescale(logB=False)
        sample = rescale(sample_from_database)
        assert 0.019 == pytest.approx(sample['Y'][0, 0], rel=1e-1)

    def test_rescale_angle(self, sample_from_database):
        rescale = Rescale(angle_transformation=True)
        sample = rescale(sample_from_database)

        assert 0.53 == pytest.approx(sample['Y'][0, 0], rel=1e-1)
        assert sample['Y'][0][1] == pytest.approx(np.sin(sample_from_database['Y'][1] * np.pi / 180), 0.01)

    def test_mlp_transfrom_rescale_no_output(self, sample_from_database):
        kwargs = {'factors': [1, 1000, 2000, 1000], 'cont_scale': 40000, 'norm_output': False}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X']

    def test_mlp_transfrom_rescale_output(self, sample_from_database):
        kwargs = {'norm_output': True}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X'][0].shape == (224,)

    def test_mlp_transfrom_rescale_output_angle(self, sample_from_database):
        kwargs = {'norm_output': True, 'angle_transformation': True, 'logB': True}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X'][0].shape == (224,)

    def test_mlp_transfrom_rescale_batch(self, batch_sample):
        kwargs = {'norm_output': True, 'angle_transformation': True, 'logB': True}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(batch_sample)
        assert transformed_sample['X'][0].shape == (100, 224)

    def test_conv1d_transform_rescale(self, sample_from_database):
        kwargs = {'norm_output': True, 'angle_transformation': True, 'logB': True}
        trsfm = conv1d_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X'][0].shape == (4, 56)


    def test_conv1_transfrom_rescale_batch(self, batch_sample):
        kwargs = {'norm_output': True, 'angle_transformation': True, 'logB': True}
        trsfm = conv1d_transform_rescale(**kwargs)
        transformed_sample = trsfm(batch_sample)
        assert transformed_sample['X'][0].shape == (100, 4, 56)
