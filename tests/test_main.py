import pytest
import torch
import os
from inverse_problem.nn_inversion.main import HyperParams, Model
from inverse_problem import get_project_root
from inverse_problem.milne_edington.me import read_full_spectra


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
        loss = model.fit_step(train_loader)
        assert loss > 0

    def test_model_eval_step(self, base_mlp_rescale_hps):
        model = Model(base_mlp_rescale_hps)
        train_loader = model.make_loader()
        loss = model.eval_step(train_loader)
        assert loss > 0

    def test_model_mlp_train(self, base_mlp_standard_hps):
        # todo failed with standard hps
        base_mlp_standard_hps.per_epoch = 10
        model = Model(base_mlp_standard_hps)
        history = model.train()
        assert history[0][0] > 0

    def test_model_conv_train(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        hps.per_epoch = 10
        model = Model(hps)
        x = model.make_loader()
        x_ = next(iter(x))
        history = model.train()
        assert history[0][0] > 0

    def test_predict_one_pixel(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.per_epoch = 1
        base_mlp_rescale_hps.n_epochs = 1
        base_mlp_rescale_hps.batch_size = 1
        model = Model(base_mlp_rescale_hps)
        history = model.train()
        filename = os.path.join(get_project_root(), 'data', 'test_1_file\\')
        line, cont = read_full_spectra(filename)
        predicted = model.predict_one_pixel((line, cont))
        assert predicted.shape == torch.Size([512, 3])

    def test_continue_training(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.per_epoch = 50
        base_mlp_rescale_hps.n_epochs = 5
        base_mlp_rescale_hps.batch_size = 25
        model = Model(base_mlp_rescale_hps)
        path_to_save = os.path.join(get_project_root(), 'data', 'test.pt')
        history = model.train(save_model=True, path_to_save=path_to_save, save_epoch=[3])
        base_mlp_rescale_hps.per_epoch = 50
        base_mlp_rescale_hps.n_epochs = 2
        base_mlp_rescale_hps.batch_size = 25
        model.continue_training(path_to_save)

    def test_predict_full_image(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.per_epoch = 1
        base_mlp_rescale_hps.n_epochs = 1
        base_mlp_rescale_hps.batch_size = 1
        model = Model(base_mlp_rescale_hps)
        history = model.train()

        filename = os.path.join(get_project_root(), 'data', '20170905_030404\\')
        line, cont = read_full_spectra(filename)
        predicted = model.predict_full_image((line, cont), 0)
        assert predicted.shape == (512, line.shape[0])
