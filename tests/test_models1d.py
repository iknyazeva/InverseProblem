import torch
import os
from inverse_problem.nn_inversion.models import HyperParams, TopNet, BaseNet
from inverse_problem.nn_inversion.conv1dmodel import Conv1dModel

class TestModels1d:
    def test_base_net(self):
        hps = HyperParams()
        net = BaseNet(hps)
        assert net.hps.activation == 'relu'


    def test_top_net_init(self):
        hps = HyperParams()
        net = TopNet(hps)
        convnet = Conv1dModel(hps)
        assert True

    def test_top_net_forward(self):
        hps = HyperParams()
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = hps.batch_size
        D_in = hps.top_input
        D_out = hps.top_output
        x = torch.randn([N, 224], device=device, dtype=dtype)
        net = TopNet(hps)
        convnet = Conv1dModel(hps)
        # y = torch.rand(N, D_out, device=device, dtype=dtype)
        out = convnet(x)
        assert True


