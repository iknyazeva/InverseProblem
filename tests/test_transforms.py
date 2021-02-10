import pytest
import numpy as np
from torchvision import transforms
import torch
import os
from pathlib import Path
from inverse_problem.nn_inversion import SpectrumDataset, ToTensor, NormalizeStandard, Rescale, FlattenSpectrum
from inverse_problem.nn_inversion import mlp_transform_standard, mlp_transform_rescale
from inverse_problem.nn_inversion.transforms import normalize_output


class TestTransforms:
    @pytest.fixture
    def sample_from_database(self):
        # filename = '/Users/irinaknyazeva/Projects/Solar/InverseProblem/data/parameters_base.fits'
        project_path = Path(__file__).resolve().parents[1]
        filename = os.path.join(project_path, 'data/parameters_base.fits')
        source = 'database'
        sobj = SpectrumDataset(filename, source=source)
        sample = sobj[0]
        return sample

    def test_to_tensor(self, sample_from_database):
        to_tensor = ToTensor()
        sample = to_tensor(sample_from_database)
        assert isinstance(sample['Y'], torch.Tensor)
        assert isinstance(sample['X'][0], torch.Tensor)

    # todo: add asserts for different inputs
    def test_normalize_output(self, sample_from_database):
        y = sample_from_database['Y']
        sample = normalize_output(y, mode='norm', logB=True)
        assert True

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
        trsfm = mlp_transform_rescale(factor=[1, 1, 1, 1])
        transformed_sample = trsfm(sample_from_database)
        assert True