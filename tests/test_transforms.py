import os
from astropy.io import fits
import numpy as np
import pytest
import torch
from inverse_problem.milne_edington.me import BatchHinodeME
from inverse_problem import get_project_root
from inverse_problem.nn_inversion import SpectrumDataset, ToTensor, NormalizeStandard, Rescale, FlattenSpectrum
from inverse_problem.nn_inversion import mlp_transform_standard, mlp_transform_rescale, conv1d_transform_rescale
from inverse_problem.nn_inversion.transforms import normalize_output


class TestTransforms:
    @pytest.fixture
    def sample_from_database(self):
        project_path = get_project_root()
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        sample = sobj[2]
        return sample

    def test_flatten(self):
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        param_batch = fits.open(filename)[0].data
        obj = BatchHinodeME(param_batch)
        spectrum = obj.compute_spectrum(with_ff=True, with_noise=True)
        sample = FlattenSpectrum()(spectrum[:3, :])
        assert True

    # todo: add asserts for different inputs
    def test_normalize_output_range_no_angle(self, sample_from_database):
        y = sample_from_database['Y']
        sample = normalize_output(y, mode='range', logB=True)
        assert min(sample) > 0
        assert max(sample) < 1

    def test_normalize_output_range_angle(self, sample_from_database):
        y = sample_from_database['Y']
        sample = normalize_output(y, mode='range', logB=True, angle_transformation=True)
        assert sample[0][1] == pytest.approx(1, 0.1)
        assert min(sample) > 0
        assert max(sample) < 1

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
        assert sample[0, 1] == pytest.approx(np.sin(params[0, 1]*np.pi/180), 0.01)
        assert True

    def test_to_tensor(self, sample_from_database):
        to_tensor = ToTensor()
        sample = to_tensor(sample_from_database)
        assert isinstance(sample['Y'], torch.Tensor)
        assert isinstance(sample['X'][0], torch.Tensor)

    # todo: add asserts for different inputs
    def test_normalize_standard(self, sample_from_database):
        norm = NormalizeStandard()
        sample = norm(sample_from_database)
        assert True

    # todo: add asserts for different inputs parameters
    def test_rescale(self, sample_from_database):
        rescale = Rescale()
        sample = rescale(sample_from_database)

        assert 0.53 == pytest.approx(sample['Y'][0, 0], rel=1e-1)
        rescale = Rescale(logB=False)
        sample = rescale(sample_from_database)
        assert 0.019 == pytest.approx(sample['Y'][0, 0], rel=1e-1)

    def test_rescale_angle(self, sample_from_database):
        rescale = Rescale(angle_transformation=True)
        sample = rescale(sample_from_database)

        assert 0.53 == pytest.approx(sample['Y'][0, 0], rel=1e-1)
        assert sample['Y'][0][1] == pytest.approx(np.sin(sample_from_database['Y'][1]*np.pi/180), 0.01)

    def test_mlp_transfrom_standard(self, sample_from_database):
        trsfm = mlp_transform_standard()
        transformed_sample = trsfm(sample_from_database)
        assert True

    def test_mlp_transfrom_rescale_no_output(self, sample_from_database):
        kwargs = {'factors': [1, 1000, 2000, 1000], 'cont_scale': 40000, 'norm_output': False}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X']

    def test_mlp_transfrom_rescale_output(self, sample_from_database):
        kwargs = {'norm_output': True}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X'][0].shape == 224

    def test_mlp_transfrom_rescale_output_angle(self, sample_from_database):
        kwargs = {'norm_output': True, 'angle_transformation': True, 'logB': True}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X'] == 224

    def test_conv1d_transform_rescale(self, sample_from_database):
        kwargs = {'norm_output': True, 'angle_transformation': True, 'logB': True}
        trsfm = conv1d_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X'][0].shape == (4, 56)
