from .dataset import SpectrumDataset
from .transforms import mlp_transform_rescale
import json
import torch.nn.functional as F
import torch
from torch import nn, cuda
import os
from pathlib import Path
from torch.utils.data import DataLoader
import attr


@attr.s(slots=True)
class HyperParams:
    n_input = attr.ib(default=224)
    bottom_output = attr.ib(default=40)
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


    @classmethod
    def from_file(cls, path_to_json: Path):
        with open(path_to_json) as json_file:
            params = json.loads(json_file.read())
            fields = {field.name for field in attr.fields(HyperParams)}
            return cls(**{k: v for k, v in params.items() if k in fields})


class BaseNet(nn.Module):
    def __init__(self, hps: HyperParams):
        """
        Args:
            hps (nn.Module): class contains all hyperparameters
        """
        super().__init__()
        self.hps = hps
        self.activation = getattr(F, hps.activation)


class FullModel(BaseNet):
    def __init__(self, hps: HyperParams, bottom: BaseNet, top: BaseNet):
        super(FullModel, self).__init__(hps)
        self.bottom = bottom(hps)
        self.top = top(hps)

    def forward(self, sample_x):
        x = self.bottom(sample_x[0])
        x = torch.cat((x, sample_x[1]), axis=1)
        x = self.top(x)
        return x


class BottomSimpleMLPNet(BaseNet):
    """ Two-layer MLP forward propagation concatenation, bottom layer, take plain input
    """

    def __init__(self, hps: HyperParams):
        super(BottomSimpleMLPNet, self).__init__(hps)

        self.fc1 = nn.Linear(hps.n_input, hps.bottom_output)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return x


class BottomSimpleConv1d(BaseNet):
    def __init__(self, hps: HyperParams):
        super().__init__(hps)

        self.conv1 = nn.Sequential(nn.Conv1d(4, 32, 5, padding=2),
                                   nn.AvgPool1d(2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 5),
                                   nn.AvgPool1d(2),
                                   nn.ReLU())
        self.linear = nn.Sequential(nn.Linear(64 * 12, hps.bottom_output), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
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


class Conv1dModel(BaseNet):
    """ Two-layer 1D Convolution Network
    """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.hps = hps
        self.topnet = TopNet(hps)
        self.conv1 = nn.Sequential(nn.Conv1d(4, 32, 2),
                                   nn.MaxPool1d(2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 2), nn.ELU())
        self.linear = nn.Sequential(nn.Linear(64 * 26, 225), nn.ReLU())

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(-1, 64 * 26)
        x = self.linear(x)
        x = self.topnet(x)
        return x
