from .dataset import SpectrumDataset
from .transforms import mlp_transform_rescale
import torch.nn.functional as F
import torch
from torch import nn, cuda
import os
from pathlib import Path
from torch.utils.data import DataLoader
import attr


@attr.s(slots=True)
class HyperParams:
    top_input = attr.ib(default=225)
    predict_ind = attr.ib(default=[0, 1, 2])
    top_output = attr.ib(default=3)
    transform_type = attr.ib(default='mlp_transform_rescale')
    source = attr.ib(default='database')
    hidden_size = attr.ib(default=100)
    dropout = attr.ib(default=0.0)
    bn = attr.ib(default=1)
    net = attr.ib(default='TopNet')
    activation = attr.ib(default='relu')
    lr = attr.ib(default=0.0001)
    lr_decay = attr.ib(default=0.0)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=20)
    n_epochs = attr.ib(default=5)
    dropout = attr.ib(default=0.0)
    per_epoch = attr.ib(default=10)


class BaseNet(nn.Module):
    def __init__(self, hps: HyperParams):
        super().__init__()
        self.hps = hps


class TopNet(BaseNet):
    """ Two-layer forward propagation concatenation, top layer, same building block
     for all models
    """
    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.fc1 = nn.Linear(225, 3)
        self.fc1 = nn.Linear(hps.top_input, hps.hidden_size)
        self.fc2 = nn.Linear(hps.hidden_size, hps.top_output)
        self.activation = getattr(F, hps.activation)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
