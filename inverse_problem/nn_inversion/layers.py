import torch.nn as nn
from torch.nn import Conv1d


class ConvLayer(nn.Module):
    """
    Conv layer with optional batch norm and dropout
    """

    def __init__(self, in_dim, out_dim, kernel_size, padding, activation, dropout, batch_norm,
                 pool=None, conv_output=14, dilation=1):
        """

        Args:
            in_dim:
            out_dim:
            kernel_size:
            padding:
            activation:
            dropout:
            batch_norm:
            pool:
            adapt_size (int): size of output after applying AddaptivePooling
            dilation:
        """
        super().__init__()
        self.in_channels = in_dim
        self.pool = pool
        self.out_channels = out_dim
        self.batch_norm = batch_norm

        self.batch_norm_1d = nn.BatchNorm1d(out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=(dilation,))

        if pool == 'max':
            self.pool = nn.MaxPool1d(2, stride=2)
        elif pool == 'mean':
            self.pool = nn.AvgPool1d(2)
        elif pool == 'adapt':
            self.pool = nn.AdaptiveAvgPool1d(conv_output)
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


class MLPlayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation, dropout, batch_norm):
        super().__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        # torch.manual_seed(torch.Generator().get_state())
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
        list_FC_layers.extend([MLPlayer(in_dim // (2 * l), in_dim // (2 * (l + 1)), activation, dropout, batch_norm)
                               for l in range(1, num_layers)])
        list_FC_layers.append(MLPlayer(in_dim // (2 * num_layers), out_dim, None, dropout, batch_norm))
        self.FC_layers = nn.ModuleList(list_FC_layers)

    def forward(self, x):
        for layer_id in range(self.num_layers + 1):
            x = self.FC_layers[layer_id](x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_сhannels,  kernel_size, padding, activation, dropout, batch_norm,
                 out_channels=[64, 64, 128, 128, 256, 256], pool=[None, 'max', None, 'max', None],
                 conv_output=14, bottom_output=40, dilation=1):
        """

        Args:
            in_сhannels (int): number of input channels, in case of IQUV - 4
            kernel_size (int): kernel size
            padding (int):
            activation:
            dropout:
            batch_norm:
            out_channels (list of int):
            pool (list of None or str): type of pooling applied to each layer, should be compatible woth conv layer
            dilation:
        """
        super().__init__()
        assert len(out_channels) >= 1
        assert len(pool) == len(out_channels)-1 or len(pool) == 1
        self.num_layers = len(out_channels)
        if len(pool) == 1:
            pool = self.num_layers-1*[pool] + ['adapt']
        else:
            pool = pool+['adapt']

        list_Conv_layers = [ConvLayer(in_сhannels, out_channels[0], kernel_size,
                                      padding, activation, dropout, batch_norm, pool=pool[0], dilation=dilation)]
        list_Conv_layers.extend([ConvLayer(out_channels[l], out_channels[l + 1], kernel_size,
                                          padding, activation, dropout, batch_norm, pool=pool[l+1], dilation=dilation)
                                 for l in range(self.num_layers - 1)])
        self.Conv_layers = nn.ModuleList(list_Conv_layers)
        self.fc_layer = MLPlayer(out_channels[-1]*conv_output, bottom_output, activation, dropout, batch_norm)

    def forward(self, x):
        for layer_id in range(self.num_layers):
            x = self.Conv_layers[layer_id](x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


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
        list_FC_layers.extend([MLPlayer(hidden_dims[l], hidden_dims[l + 1], activation, dropout, batch_norm)
                               for l in range(self.num_layers - 1)])
        self.FC_layers = nn.ModuleList(list_FC_layers)

    def forward(self, x):
        for layer_id in range(self.num_layers):
            x = self.FC_layers[layer_id](x)
        return x
