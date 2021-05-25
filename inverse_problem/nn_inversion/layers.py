import torch.nn as nn
from torch.nn import Conv1d


class ConvLayer(nn.Module):
    """
    Conv layer with optional batch norm and dropout
    """

    def __init__(self, in_dim, out_dim, kernel_size, activation, dropout, batch_norm, pool=None):
        super().__init__()
        self.in_channels = in_dim
        self.pool = pool
        self.out_channels = out_dim
        self.batch_norm = batch_norm

        self.batch_normx = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = Conv1d(in_dim, out_dim, kernel_size=kernel_size)

        if pool == 'max':
            self.pool = nn.MaxPool1d(2)
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
            x = self.batch_normx(x)  # batch normalization

        if self.activation:
            x = self.activation(x)

        x = self.dropout(x)
        return x

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channels, self.out_channels)


class MLPlayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.linear = nn.Linear(in_dim, out_dim)
        self.batch_normx = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)

        if self.batch_norm:
            x = self.batch_normx(x)  # batch normalization

        if self.activation:
            x = self.activation(x)

        x = self.dropout(x)
        return x


class MLPReadout(nn.Module):

    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm, num_layers=2):
        """

        Args:
            in_dim (int): input dim of processed data in previous block
            out_dim (int): number of predicted variables
            activation (callable): F from nn.Module
            dropout (float): from 0 to 1
            batch_norm (bool): do we need batch norm
            num_layers(int):
        """
        super().__init__()
        self.num_layers = num_layers
        list_FC_layers = [MLPlayer(in_dim, in_dim // 2, activation, dropout, batch_norm)]
        list_FC_layers.extend([MLPlayer(in_dim // 2 * l, in_dim // (2 * (l + 1)), activation, dropout, batch_norm)
                               for l in range(1, num_layers)])
        list_FC_layers.append(MLPlayer(in_dim // (2 * num_layers), out_dim, None, dropout, batch_norm))
        self.FC_layers = nn.ModuleList(list_FC_layers)

    def forward(self, x):
        for layer_id in range(self.num_layers + 1):
            x = self.FC_layers[layer_id](x)
        return x

class ConvBlock(nn.Module):
    pass


class MLPBlock(nn.Module):
    def __init__(self, in_dim, activation, dropout, batch_norm, hidden_dims=[200, 300]):
        """

        Args:
            in_dim (int): input dim of processed data in previous block
            out_dim (int): number of predicted variables
            activation (callable): F from nn.Module
            dropout (float): from 0 to 1
            batch_norm (bool): do we need batch norm
            num_layers(int):
        """
        super().__init__()
        assert len(hidden_dims) >= 1
        self.num_layers = len(hidden_dims)
        list_FC_layers = [MLPlayer(in_dim, hidden_dims[0], activation, dropout, batch_norm)]
        list_FC_layers.extend([MLPlayer(hidden_dims[l], hidden_dims[l+1], activation, dropout, batch_norm)
                               for l in range(self.num_layers-1)])
        self.FC_layers = nn.ModuleList(list_FC_layers)

    def forward(self, x):
        for layer_id in range(self.num_layers):
            x = self.FC_layers[layer_id](x)
        return x

