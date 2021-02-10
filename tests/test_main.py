import pytest
import numpy as np
from torchvision import transforms
import torch
import os
from inverse_problem.nn_inversion.main import HyperParams, Model


class TestMain:

    @pytest.fixture
    def sample_batch(self):
        hps = HyperParams()
        model = Model(hps)
        train_loader = model.make_loader()
        it = iter(train_loader)
        sample_batch = next(it)
        return sample_batch


    def test_model_init(self):
        hps = HyperParams()
        model = Model(hps)
        assert True

    def test_model_make_loader(self):
        hps = HyperParams()
        model = Model(hps)
        train_loader = model.make_loader()
        it = iter(train_loader)
        sample_batch = next(it)
        assert sample_batch['X'].size() == (20, 225)
        assert sample_batch['Y'].size() == (20, 11)

        assert isinstance(sample_batch['X'][0], torch.Tensor)
        assert isinstance(sample_batch['Y'][0], torch.Tensor)

    def test_model_fit_step(self, sample_batch):
        hps = HyperParams()
        model = Model(hps)
        loss = model.fit_step(sample_batch)
        assert True

    def test_model_eval_step(self, sample_batch):
        hps = HyperParams()
        model = Model(hps)
        loss = model.eval_step(sample_batch)
        assert True

    def test_model_train(self):
        hps = HyperParams()
        hps.per_epoch = 2
        model = Model(hps)
        history = model.train()
        assert True