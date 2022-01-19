import torch
import pytest
from inverse_problem.nn_inversion.layers import ConvLayer, MLPlayer, MLPReadout, MLPBlock, ConvBlock
import torch.nn.functional as F


def test_mlp_layer():
    input = 10 * torch.randn(10, 100)
    activation = getattr(F, 'relu')
    mlp = MLPlayer(100, 20, activation, 0.05, True)
    output = mlp(input)
    assert output.shape == (10, 20)


def test_conv_layer():
    input = torch.randn(10, 4, 56)
    activation = getattr(F, 'relu')
    conv = ConvLayer(4, 64, 3, 1, activation, dropout=0.05, batch_norm=True, pool=None)
    output = conv(input)
    assert output.shape == (10, 64, 56)


def test_mlp_block():
    input = 10 * torch.randn(10, 100)
    mlp = MLPBlock(100, F.elu, 0.01, True, hidden_dims=[10, 20, 30])
    output = mlp(input)
    assert True


def test_conv_block():
    input = 10 * torch.randn(10, 4, 56)
    activation = getattr(F, 'elu')
    conv_output = 14
    bottom_output = 40
    conv = ConvBlock(4, 3, 2, activation, dropout=0.05, batch_norm=True,
                     out_channels=[32, 32, 64, 64, 128, 128], pool=[None, 'max', None, 'max', None],
                     conv_output=conv_output, bottom_output=bottom_output)
    output = conv(input)
    assert output.shape[1] == bottom_output


def test_mlp_readout():
    input = 10 * torch.randn(10, 100)
    activation = getattr(F, 'relu')
    mlp = MLPReadout(100, 20, activation, 0.05, True)
    output = mlp(input)
    assert True
