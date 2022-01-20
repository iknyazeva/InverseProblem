import torch
import pytest
from inverse_problem.nn_inversion.model_mlp_common import MlpCommonNet
from inverse_problem import SpectrumDataset, PregenSpectrumDataset
import torch.nn.functional as F


class TestModelMlpCommon:
    @pytest.fixture
    def sample_x(self):
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = 20
        input_dim = 224
        x0 = torch.randn(N, input_dim, device=device, dtype=dtype)
        x1 = torch.randn(N, 1, device=device, dtype=dtype)
        return x0, x1

    def test_forward(self, sample_x):
        net = MlpCommonNet(input_dim=224, output_dim=11, hidden_dims=[200, 200, 100],
                           activation='elu', batch_norm=True, dropout=0.05, number_readout_layers=2)
        out = net(sample_x)
        assert True
