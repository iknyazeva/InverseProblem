import json
import torch.nn.functional as F
import torch
from torch import nn
from pathlib import Path
import attr


@attr.s(slots=True)
class HyperParams:
    """
    Attributes:
    n_input: int, input size

    predict_ind: list, parameters to predict

    top_output: int, output size for top model, i.e. number of parameters to predict

    transform_type: str, transformation type to be applied to data

    mode:

    logB: bool,

    factors: list

    cont_scale: int? continuum scale ?

    norm_output: bool ?

    source: str, where to get data from

    hidden_size: int, hidden size for TopNet

    dropout: float, dropout rate

    bn: ?

    top_net: str, name of top net

    bottom_net: str, name of bottom net

    activation: str, activation layer in top net

    lr: float, learning rate for optimizer

    lr_decay: learning rate decay for optimizer

    weight_decay: weight rate decay for optimizer

    batch_size: int, num of batches

    n_epochs: int, num of epochs to train

    per_epoch: int, num of examples to use while training
    """
    n_input = attr.ib(default=224) # ?
    bottom_output = attr.ib(default=40) # ?
    predict_ind = attr.ib(default=[0, 1, 2])
    top_output = attr.ib(default=3)
    transform_type = attr.ib(default='mlp_transform_rescale')
    mode = attr.ib(default='range')
    logB = attr.ib(default=True)
    factors = attr.ib(default=[1, 1000, 1000, 1000])
    cont_scale = attr.ib(default=40000)
    norm_output = attr.ib(default=True)
    source = attr.ib(default='database')
    hidden_size = attr.ib(default=100)
    dropout = attr.ib(default=0.0)
    bn = attr.ib(default=1)
    top_net = attr.ib(default='TopNet')
    bottom_net = attr.ib(default='BottomSimpleConv1d')
    activation = attr.ib(default='relu')
    lr = attr.ib(default=0.0001)
    lr_decay = attr.ib(default=0.0)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=20)
    n_epochs = attr.ib(default=5)
    per_epoch = attr.ib(default=10)
    patience = attr.ib(default=3)
    absolute_noise_levels = attr.ib(default=[109, 28, 28, 44])


    @classmethod
    def from_file(cls, path_to_json: Path):
        """loads hyper params from json file"""
        with open(path_to_json) as json_file:
            params = json.loads(json_file.read())
            fields = {field.name for field in attr.fields(HyperParams)}
            return cls(**{k: v for k, v in params.items() if k in fields})


class BaseNet(nn.Module):
    """Parent class for all models"""
    def __init__(self, hps: HyperParams):
        """
        Args:
            hps: hyperparameters class
        """
        super().__init__()
        self.hps = hps
        self.activation = getattr(F, hps.activation)


class FullModel(BaseNet):
    """Creates full model using bottom and top nets"""
    def __init__(self, hps: HyperParams, bottom: BaseNet, top: BaseNet):
        """
        Args:
            hps (): HyperParams class
            bottom (): Bottom net, the leading architecture
            top (): TopNet, outer layer for all bottom models
        """
        super().__init__(hps)
        self.bottom = bottom(hps)
        self.top = top(hps)

    def forward(self, sample_x):
        x = self.bottom(sample_x[0])
        x = torch.cat((x, sample_x[1]), axis=1)
        x = self.top(x)
        return x


class BottomSimpleMLPNet(BaseNet):
    """ Two-layer MLP forward propagation with dropout, bottom layer, takes plain input.
    """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.fc1 = nn.Linear(hps.n_input, hps.hidden_size)
        self.fc2 = nn.Linear(hps.hidden_size, hps.bottom_output)
        self.dropout = nn.Dropout(hps.dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


class BottomSimpleConv1d(BaseNet):
    def __init__(self, hps: HyperParams):
        """
        Two-layer 1D convolution net with average pooling and batch normalization.
        Takes 56-channel inputs.

        Args:
            hps (): HyperParams class
        """
        super().__init__(hps)

        self.conv1 = nn.Sequential(nn.Conv1d(56, 64, 2, padding=2),
                                   nn.AvgPool1d(2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 2),
                                   nn.AvgPool1d(2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(128))
        self.linear = nn.Sequential(nn.Linear(128, hps.bottom_output), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x.squeeze())
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class TopNet(BaseNet):
    """ Two-layer forward propagation concatenation, top layer, same building block
     for all models
    """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.fc1 = nn.Linear(hps.bottom_output + 1, hps.hidden_size)
        self.fc2 = nn.Linear(hps.hidden_size, hps.top_output)
        self.dropout = nn.Dropout(hps.dropout)

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

