from inverse_problem.nn_inversion.models import HyperParams, TopNet, BaseNet
import torch.nn.functional as F
import torch
from torch import nn, cuda


class Conv1dModel(BaseNet):
    """ Two-layer 1D Convolution Network
    """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.hps = hps
        self.topnet = TopNet(hps)
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, 2),
                                   nn.MaxPool1d(2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 2), nn.ELU())
        self.linear = nn.Sequential(nn.Linear(64 * 111, 225), nn.ReLU())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(-1, 64 * 111)
        x = self.linear(x)
        x = self.topnet(x)
        return x
