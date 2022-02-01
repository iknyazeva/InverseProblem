import torch
import pytest
from inverse_problem.nn_inversion.model_pi_mlp_conv import PIMLPConvNet


class TestPIMLPConvNet:
    @pytest.fixture
    def sample_x(self):
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 20
        x0 = torch.randn(batch_size, 4, 56, device=device, dtype=dtype)
        x1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
        return x0, x1

    def test_forward(self, sample_x):
        net = PIMLPConvNet()
        out = net(sample_x)
        assert True
