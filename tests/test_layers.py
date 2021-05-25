import torch
import pytest
from inverse_problem.nn_inversion.layers import ConvLayer, MLPlayer, MLPReadout, MLPBlock
import torch.nn.functional as F

def test_conv_layer():
    input = torch.randn(10, 4, )

def test_mlp_layer():
    input = 10*torch.randn(10, 100)
    activation = getattr(F, 'relu')
    mlp = MLPlayer(100, 20, activation, 0.05, True)
    output = mlp(input)
    assert True

def test_mlp_block():
    input = 10 * torch.randn(10, 100)
    mlp = MLPBlock(100, F.elu, 0.01, True, hidden_dims=[10, 20, 30])
    output = mlp(input)
    assert True


def test_mlp_readout():
    input = 10 * torch.randn(10, 100)
    activation = getattr(F, 'relu')
    mlp = MLPReadout(100, 20, activation, 0.05, True)
    output = mlp(input)
    assert True

