import json
import torch.nn.functional as F
import torch
from torch import nn
import torchvision
from pathlib import Path
from inverse_problem.nn_inversion.layers import MLPBlock, MLPReadout
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

    trainset: int, num of examples to use while training
    valset: int, num of examples to use while evaluation
    """
    hps_name = attr.ib(default='basic_hps')
    n_input = attr.ib(default=224)
    batch_norm = attr.ib(default=True)
    dropout = attr.ib(default=0.05)
    hidden_dims = attr.ib(default=[200, 300])
    bottom_output = attr.ib(default=40)
    predict_ind = attr.ib(default=[0, 1, 2])
    activation = attr.ib(default='relu')
    val_split = attr.ib(default=0.1)
    top_output = attr.ib(default=3)
    top_layers = attr.ib(default=2)
    transform_type = attr.ib(default='mlp_transform_rescale')
    mode = attr.ib(default='range')
    logB = attr.ib(default=True)
    factors = attr.ib(default=[1, 1000, 1000, 1000])
    cont_scale = attr.ib(default=40000)
    norm_output = attr.ib(default=False)
    source = attr.ib(default='database')
    hidden_size = attr.ib(default=100)
    bn = attr.ib(default=1)
    top_net = attr.ib(default='TopNet')
    bottom_net = attr.ib(default='BottomSimpleConv1d')
    lr = attr.ib(default=0.0001)
    lr_decay = attr.ib(default=0.0)
    weight_decay = attr.ib(default=0.0)
    batch_size = attr.ib(default=20)
    n_epochs = attr.ib(default=5)
    trainset = attr.ib(default=10)
    valset = attr.ib(default=10)
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
        self.batch_norm = hps.batch_norm
        self.dropout = hps.dropout
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


class BottomMLPNet(BaseNet):
    """ Arbitraty-layer MLP forward propagation with batch-norm dropout, bottom layer, takes plain input.
       """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.mlp = MLPBlock(hps.n_input, self.activation, self.dropout,
                            self.batch_norm, hps.hidden_dims + [hps.bottom_output])

    def forward(self, x):
        x = self.mlp(x)
        return x


class ZeroMLP(BaseNet):
    def __init__(self, hps: HyperParams):
        super().__init__(hps)

    def forward(self, x):
        return x


class TopCommonMLPNet(BaseNet):
    """
    Hard-hard sharing, only one unit for target have own weight
    """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.mlp = MLPReadout(hps.bottom_output + 1, hps.top_output, self.activation, self.dropout,
                              self.batch_norm, hps.top_layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


class TopIndependentNet(BaseNet):
    """
        Task independent block, only one unit for target have own weight
    """

    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        layer = nn.Sequential(
                    MLPBlock(hps.bottom_output+1, self.activation, self.dropout, self.batch_norm, hps.hidden_dims),
                    MLPReadout(hps.hidden_dims[-1], 1, self.activation, self.dropout, self.batch_norm, hps.top_layers)
        )
        self.task_layers = nn.ModuleList(hps.top_output * [layer])

    def forward(self, x):
        return torch.cat(tuple(task_layer(x) for task_layer in self.task_layers), dim=1)


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

        self.conv1 = nn.Sequential(nn.Conv1d(4, 32, 2, padding=2),
                                   nn.AvgPool1d(2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(32))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 2),
                                   nn.AvgPool1d(2),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64))
        self.linear = nn.Sequential(nn.Linear(64 * 14, hps.bottom_output), nn.ReLU())

    def forward(self, x):
        x = self.conv1(x.squeeze(1))
        x = self.conv2(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class BottomResNet(BaseNet):
    def __init__(self, hps: HyperParams):
        super().__init__(hps)
        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.linear = nn.Sequential(nn.Linear(1000, hps.bottom_output), nn.ReLU())

    def forward(self, x):
        x = self.resnet(x.permute(0, 2, 1, 3))
        # print(x.shape)
        x = x.view(-1, 1000)
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
