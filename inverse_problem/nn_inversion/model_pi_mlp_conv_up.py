import torch.nn.functional as F
import torch
from torch import nn
from inverse_problem.nn_inversion.layers import MLPBlock, MLPReadout


class ConvBlock(nn.Module):
    """
    Conv block with optional batch norm and dropout
    """

    def __init__(self, in_dim, out_dim, kernel_size, padding, activation, dropout, batch_norm, pool=None):
        super().__init__()
        self.in_channels = in_dim
        self.pool = pool
        self.out_channels = out_dim
        self.batch_norm = batch_norm

        self.batch_norm_1d = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)

        if pool == 'max':
            self.pool = nn.MaxPool1d(2, stride=2)
        elif pool == 'mean':
            self.pool = nn.AvgPool1d(2)
        elif pool is None:
            self.pool = pool
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        if self.batch_norm:
            x = self.batch_norm_1d(x)
        if self.activation:
            x = self.activation(x)

        x = self.dropout(x)
        return x


class PIMLPConvDistributionNet(nn.Module):
    def __init__(self, n_blocks=6, in_dim=(4, 64, 64, 128, 128, 256), out_dim=(64, 64, 128, 128, 256, 256),
                 kernel_size=(3, 3, 3, 3, 3, 3), padding=(1, 1, 1, 1, 1, 1), activation='elu', dropout=0.05,
                 batch_norm=True, pool=(None, 'max', None, 'max', None, None), hidden_dims=(100, 100),
                 bottom_output=100, number_readout_layers=2, top_output=11):
        """
        - 1D convolutions + pooling, batch normalization and dropout
        - MLP block
        - Partly independent MLP block with additional outputs for uncertainty prediction
        """
        super(PIMLPConvDistributionNet, self).__init__()

        activation = getattr(F, activation)

        conv_blocks = []
        for i in range(n_blocks):
            conv_blocks.append(ConvBlock(in_dim[i], out_dim[i], kernel_size[i], padding[i], activation,
                                         dropout, batch_norm, pool=pool[i]))

        self.conv_part = nn.ModuleList(conv_blocks)

        self.mlp = MLPBlock(256 * 14, activation, dropout, batch_norm, (*hidden_dims, bottom_output))

        top_mlp_layers = []
        for i in range(top_output):
            top_mlp_layers.append(nn.Sequential(
                MLPBlock(bottom_output + 1, activation, dropout, batch_norm, hidden_dims),
                MLPReadout(hidden_dims[-1], 1, activation, dropout, batch_norm, number_readout_layers))
            )
            top_mlp_layers.append(nn.Sequential(
                MLPBlock(bottom_output + 1, activation, dropout, batch_norm, hidden_dims),
                MLPReadout(hidden_dims[-1], 1, activation, dropout, batch_norm, number_readout_layers))
            )

        self.top_mlp_part = nn.ModuleList(top_mlp_layers)

    def forward(self, sample):
        x = sample[0]

        for conv_block in self.conv_part:
            x = conv_block(x)

        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        x = torch.cat((x, sample[1]), dim=1)
        x = torch.cat(tuple(layer(x) for layer in self.top_mlp_part), dim=1)
        return x
