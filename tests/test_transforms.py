import os
from astropy.io import fits
import pytest
import torch

from inverse_problem import get_project_root
from inverse_problem.nn_inversion import SpectrumDataset, ToTensor, NormalizeStandard, Rescale
from inverse_problem.nn_inversion import mlp_transform_standard, mlp_transform_rescale, conv1d_transform_rescale
from inverse_problem.nn_inversion.transforms import normalize_output


class TestTransforms:
    @pytest.fixture
    def sample_from_database(self):
        project_path = get_project_root()
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source)
        sample = sobj[0]
        return sample

    def test_normalize_output_array(self):
        project_path = get_project_root()
        filename = os.path.join(project_path, 'data/small_parameters_base.fits')
        params = fits.open(filename)[0].data[5:11]
        sample = normalize_output(params, mode='range', logB=True)
        assert True


    # todo: add asserts for different inputs
    def test_normalize_output(self, sample_from_database):
        y = sample_from_database['Y']
        sample = normalize_output(y, mode='range', logB=True)
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
        assert 0.56428 == pytest.approx(sample['Y'][0, 0], rel=1e-2)
        rescale = Rescale(logB=False)
        sample = rescale(sample_from_database)
        assert 0.024151 == pytest.approx(sample['Y'][0, 0], rel=1e-4)

    def test_mlp_transfrom_standard(self, sample_from_database):
        trsfm = mlp_transform_standard()
        transformed_sample = trsfm(sample_from_database)
        assert True

    def test_mlp_transfrom_rescale(self, sample_from_database):
        kwargs = {'factors': [1, 1000, 2000, 1000], 'cont_scale': 40000, 'norm_output': False}
        trsfm = mlp_transform_rescale(**kwargs)
        transformed_sample = trsfm(sample_from_database)
        assert transformed_sample['X']

    def test_conv1d_transform_rescale(self, sample_from_database):
        trsfm = conv1d_transform_rescale(factors=[1, 1, 1, 1])
        transformed_sample = trsfm(sample_from_database)
        assert True





