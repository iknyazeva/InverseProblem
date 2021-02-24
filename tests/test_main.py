import pytest
import numpy as np
from torchvision import transforms
import torch
import os
from inverse_problem.nn_inversion.main import HyperParams, Model
from inverse_problem import get_project_root
from inverse_problem.nn_inversion import models


class TestMain:

    @pytest.fixture
    def base_mlp_rescale_hps(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        return hps

    @pytest.fixture
    def base_mlp_standard_hps(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_mlp_standard.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        return hps

    def test_model_make_loader(self, base_mlp_rescale_hps):
        model = Model(base_mlp_rescale_hps)
        train_loader = model.make_loader()
        it = iter(train_loader)
        sample_batch = next(it)
        assert sample_batch['X'][0].size() == (20, 224)
        assert sample_batch['Y'].size() == (20, 11)
        assert isinstance(sample_batch['X'][0], torch.Tensor)
        assert isinstance(sample_batch['Y'][0], torch.Tensor)
        # todo add tests for other transforms

    def test_model_fit_step(self, base_mlp_rescale_hps):
        model = Model(base_mlp_rescale_hps)
        train_loader = model.make_loader()
        it = iter(train_loader)
        sample_batch = next(it)
        loss = model.fit_step(sample_batch)
        assert loss > 0

    def test_model_eval_step(self, base_mlp_rescale_hps):
        model = Model(base_mlp_rescale_hps)
        train_loader = model.make_loader()
        it = iter(train_loader)
        sample_batch = next(it)
        loss = model.eval_step(sample_batch)
        assert loss > 0

    def test_model_mlp_train(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.per_epoch = 10
        model = Model(base_mlp_rescale_hps)
        history = model.train()
        assert history[0][0] > 0

    def test_model_conv_train(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        hps.per_epoch = 10
        model = Model(hps)
        history = model.train()
        assert history[0][0] > 0
