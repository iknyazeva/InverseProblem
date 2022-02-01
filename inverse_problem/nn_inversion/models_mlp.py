import torch.nn.functional as F
import torch
from torch import nn
import torchvision
from pathlib import Path
from inverse_problem.nn_inversion.layers import MLPBlock, MLPReadout


class MlpCommonNet(nn.Module):
    """ Arbitrary-layer MLP forward propagation with batch-norm dropout, bottom layer, takes plain input.
       """

    def __init__(self, input_dim=224, output_dim=11, hidden_dims=[200, 200, 100],
                 activation='elu', batch_norm=True, dropout=0.05, number_readout_layers=2):
        super(MlpCommonNet, self).__init__()
        self.activation = getattr(F, activation)
        self.mlp = MLPBlock(input_dim, self.activation, dropout,
                            batch_norm, hidden_dims)
        self.readout = MLPReadout(hidden_dims[-1] + 1, output_dim, self.activation, dropout,
                                  batch_norm, number_readout_layers)

    def forward(self, sample):
        x = self.mlp(sample[0], )
        x = torch.cat((x, sample[1]), dim=1)
        x = self.readout(x)
        return x


class MlpPartlyIndepNet(nn.Module):
    """ Arbitrary-layer MLP forward propagation with batch-norm dropout, bottom layer, takes plain input.
       """

    def __init__(self, input_dim=224, output_dim=11, hidden_dims=[200, 200, 200],
                 activation='elu', batch_norm=True, dropout=0.05, number_readout_layers=3):
        super(MlpPartlyIndepNet, self).__init__()
        self.activation = getattr(F, activation)
        self.common_mlp = MLPBlock(input_dim, self.activation, dropout,
                            batch_norm, hidden_dims)
        top_mlp_layers = []
        for i in range(output_dim):
            top_mlp_layers.append(nn.Sequential(
                MLPReadout(hidden_dims[-1]+1, 1, self.activation, dropout, batch_norm, number_readout_layers))
            )
        self.indep_readout = nn.ModuleList(top_mlp_layers)

    def forward(self, sample):
        x = self.common_mlp(sample[0], )
        x = torch.cat((x, sample[1]), dim=1)
        x = torch.cat(tuple(layer(x) for layer in self.indep_readout), dim=1)
        return x
