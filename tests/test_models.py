import torch
import pytest
import torch.nn.functional as F
import os
from inverse_problem import get_project_root
from inverse_problem.nn_inversion.models import HyperParams
from inverse_problem.nn_inversion.models import BaseNet, BottomSimpleMLPNet, FullModel, BottomSimpleConv1d, BottomResNet
from inverse_problem.nn_inversion.models import BottomMLPNet, TopCommonMLPNet, TopIndependentNet
from inverse_problem.nn_inversion.main import Model
from inverse_problem.nn_inversion import models


class TestModels:

    @pytest.fixture
    def flat_x():
        hps = HyperParams()
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = hps.batch_size
        d_in = hps.n_input
        flat_x = torch.randn(N, d_in, device=device, dtype=dtype)
        return flat_x

    @pytest.fixture
    def conv_x(self):
        hps = HyperParams()
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = hps.batch_size
        d_in = hps.n_input
        x = torch.randn(N, 4, d_in // 4, device=device, dtype=dtype)
        return x

    @pytest.fixture
    def top_x(self):
        hps = HyperParams()
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = hps.batch_size
        d_in = hps.bottom_output + 1
        x = torch.randn(N, d_in, device=device, dtype=dtype)
        return x

    @pytest.fixture
    def sample_x(self):
        hps = HyperParams()
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = hps.batch_size
        d_in = hps.n_input
        x0 = torch.randn(N, d_in, device=device, dtype=dtype)
        x1 = torch.randn(N, 1, device=device, dtype=dtype)
        return x0, x1

    @pytest.fixture
    def sample_conv_x(self):
        hps = HyperParams()
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        N = hps.batch_size
        d_in = hps.n_input
        x0 = torch.randn(N, 4, d_in // 4, device=device, dtype=dtype)
        x1 = torch.randn(N, 1, device=device, dtype=dtype)
        return x0, x1

    def test_hyper_params(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        assert hps.activation == 'relu'

    def test_base_net(self):
        hps = HyperParams()
        net = BaseNet(hps)
        assert net.hps.activation == 'relu'

    def test_bottom_simple_mlp_net(self, flat_x):
        hps = HyperParams()
        net = BottomSimpleMLPNet(hps)
        out = net.forward(flat_x)
        assert True

    def test_bottom_simple_conv1_net(self, conv_x):
        hps = HyperParams()
        net = BottomSimpleConv1d(hps)
        out = net.forward(conv_x)
        assert True

    def test_model_mlp_stack(self, sample_x):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        bottom = getattr(models, hps.bottom_net)
        top = getattr(models, hps.top_net)
        net = FullModel(hps, bottom, top)
        out = net(sample_x)
        assert True

    def test_model_conv_stack(self, sample_conv_x):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        bottom = getattr(models, hps.bottom_net)
        top = getattr(models, hps.top_net)
        model = Model(hps)
        x = model.make_loader()
        x_ = next(iter(x))
        net = FullModel(hps, bottom, top)
        out = net(x_['X'])
        assert True

    def test_resnet(self, conv_x):
        hps = HyperParams()
        net = BottomResNet(hps)
        out = net.forward(conv_x.unsqueeze(2))
        assert True

    def test_resnet_stack(self, sample_conv_x):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        bottom = getattr(models, 'BottomResNet')
        top = getattr(models, hps.top_net)
        model = Model(hps)
        x = model.make_loader()
        x_ = next(iter(x))
        net = FullModel(hps, bottom, top)
        out = net(x_['X'])
        assert True


class TestMLPStack:

    @pytest.fixture
    def flat(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        N = hps.batch_size
        d_in = hps.n_input
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        flat = torch.randn(N, d_in, device=device, dtype=torch.float)
        self.hps = hps
        return flat

    def test_bottom_mlp_net(self, flat):
        net = BottomMLPNet(self.hps)
        out = net.forward(flat)
        assert net.activation == getattr(F, self.hps.activation)
        assert out.shape[1] == self.hps.bottom_output
        assert net.batch_norm == self.hps.batch_norm

    def test_top_common(self, flat):
        bot_net = BottomMLPNet(self.hps)
        out = bot_net.forward(flat)
        cont = torch.randn(20, 1, dtype=torch.float)
        top_input = torch.cat((out, cont), axis=1)
        top_net = TopCommonMLPNet(self.hps)
        top_out = top_net(top_input)
        assert True

    def test_top_common(self, flat):
        bot_net = BottomMLPNet(self.hps)
        out = bot_net.forward(flat)
        cont = torch.randn(20, 1, dtype=torch.float)
        top_input = torch.cat((out, cont), axis=1)
        top_net = TopIndependentNet(self.hps)
        top_out = top_net(top_input)
        assert True

    def test_full_mlp_stack(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        bottom = getattr(models, hps.bottom_net)
        top = getattr(models, hps.top_net)
        model = Model(hps)
        x = model.make_loader()
        x_ = next(iter(x))
        net = FullModel(hps, bottom, top)
        out = net(x_['X'])
