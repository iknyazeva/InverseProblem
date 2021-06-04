import pytest
import torch
import os
from astropy.io import fits
from inverse_problem.milne_edington.data_utils import create_small_dataset
from pathlib import Path
from inverse_problem.nn_inversion.main import HyperParams, Model
from inverse_problem import get_project_root
from inverse_problem.nn_inversion import SpectrumDataset
from inverse_problem.milne_edington.me import read_full_spectra
from inverse_problem.nn_inversion.posthoc import compute_metrics, open_param_file


class TestMain:

    @pytest.fixture
    def base_mlp_rescale_hps(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        return hps

    @pytest.fixture
    def base_mlp_standard_hps(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_mlp_standard.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        return hps

    def test_model_make_loader(self, base_mlp_rescale_hps):
        model = Model(base_mlp_rescale_hps)
        train_loader, val_loader = model.make_loader()
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
        model = Model(base_mlp_standard_hps)
        history = model.train()
        assert history[0][0] > 0

    def test_model_conv_train(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model = Model(hps)
        x = model.make_loader()
        x_ = next(iter(x))
        history = model.train()
        assert history[0][0] > 0

    def test_resnet_train(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_resnet.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        hps.trainset = 5
        model = Model(hps)
        history = model.train()
        assert history[0][0] > 0

    def test_predict_one_pixel(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.trainset = 1
        base_mlp_rescale_hps.valset = 1
        base_mlp_rescale_hps.n_epochs = 1
        base_mlp_rescale_hps.batch_size = 1
        model = Model(base_mlp_rescale_hps)
        history = model.train()
        filename = Path(os.getcwd()).parent / 'data' / "20170905_030404.fits"
        ref = fits.open(filename)
        predicted, y, x = model.predict_one_pixel(ref, 3, 4)
        assert predicted[0].shape == torch.Size([1, 3])

    def test_predict_one_pixel_conv(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        hps.trainset = 1
        hps.valset = 1
        hps.n_epochs = 1
        hps.batch_size = 1
        model = Model(hps)
        history = model.train()

        filename = Path(os.getcwd()).parent / 'data' / "20170905_030404.fits"
        ref = fits.open(filename)
        predicted, y, x, _ = model.predict_one_pixel(ref, 3, 4)
        assert predicted[0].shape == torch.Size([1, 3])

    def test_continue_training(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.trainset = 1
        base_mlp_rescale_hps.valset = 1
        base_mlp_rescale_hps.n_epochs = 5
        base_mlp_rescale_hps.batch_size = 25
        model = Model(base_mlp_rescale_hps)
        path_to_save = os.path.join(get_project_root(), 'data', 'test.pt')
        history = model.train(path_to_save=path_to_save, save_epoch=[3])
        base_mlp_rescale_hps.trainset = 1
        base_mlp_rescale_hps.valset = 1
        base_mlp_rescale_hps.n_epochs = 2
        base_mlp_rescale_hps.batch_size = 25
        model.continue_training(path_to_save)

    def test_predict_full_image(self, base_mlp_rescale_hps):
        base_mlp_rescale_hps.trainset = 1
        base_mlp_rescale_hps.valset = 1
        base_mlp_rescale_hps.n_epochs = 1
        base_mlp_rescale_hps.batch_size = 1
        model = Model(base_mlp_rescale_hps)
        history = model.train()

        filename = Path(os.getcwd()).parent / 'data' / "20170905_030404.fits"
        ref = fits.open(filename)
        predicted, params, lines, cont = model.predict_full_image(ref)
        assert predicted.shape == (ref[1].data.shape + (3,))

    def test_predict_full_image_conv(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_base_conv.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        hps.trainset = 1
        hps.valset = 1
        hps.n_epochs = 1
        hps.batch_size = 1
        model = Model(hps)
        history = model.train()

        filename = Path(os.getcwd()).parent / 'data' / "20170905_030404.fits"
        ref = fits.open(filename)
        predicted, params, lines, cont = model.predict_full_image(ref, cnn=True)
        assert predicted.shape == (ref[1].data.shape + (3,))


class TestMainMlp():
    @pytest.fixture
    def params(self):
        #filename = get_project_root() / 'data' / 'parameters_base.fits'
        savename = get_project_root() / 'data' / 'small_parameters_base.fits'
        #create_small_dataset(filename, savename, size=1000)
        params = fits.open(savename)[0].data
        return params

    @pytest.fixture
    def common_mlp_rescale_hps(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        return hps

    def test_generate_batch_spectrum(self, common_mlp_rescale_hps, params):
        model = Model(common_mlp_rescale_hps)
        data = model.generate_batch_spectrum(params, noise=False)
        project_path = get_project_root()
        filename = project_path / 'data' / 'small_parameters_base.fits'
        source = 'database'
        sobj = SpectrumDataset(param_path=filename, source=source, transform=model.transform, ff=True, noise=False)
        rec0 = sobj[0]
        rec0ind = rec0['X'][0].detach().numpy()
        rec0batch = data['X'][0][0].detach().numpy()
        assert rec0ind == pytest.approx(rec0batch)
        assert data['Y'][0].detach().numpy() == pytest.approx(rec0['Y'].detach().numpy())

    def test_generate_spectrum_by_refer(self, common_mlp_rescale_hps):
        model = Model(common_mlp_rescale_hps)
        project_path = get_project_root()
        refer_path = project_path/'res_experiments'/'predictions'/'20170905_030404.fits'
        refer, names = open_param_file(refer_path, normalize=False)
        params = refer.reshape(-1, 11)
        spectrum = model.generate_batch_spectrum(params[:10], noise=False)
        assert True

    def test_predict_from_batch(self, common_mlp_rescale_hps, params):
        model = Model(common_mlp_rescale_hps)
        predicted = model.predict_by_batches(params, batch_size=64)
        assert True

    def test_predict_by_batches(self,common_mlp_rescale_hps, params):
        model = Model(common_mlp_rescale_hps)
        predicted = model.predict_from_batch(params)
        assert True

    def test_predict_refer(self,common_mlp_rescale_hps):
        project_path = get_project_root()
        refer_path = get_project_root()/'res_experiments'/'predictions'/'20170905_030404.fits'
        model = Model(common_mlp_rescale_hps)
        predicted = model.predict_refer(refer_path)
        assert True




    def test_model_make_loader_data(self, common_mlp_rescale_hps, params):
        model = Model(common_mlp_rescale_hps)
        train_loader, val_loader = model.make_loader(data_arr=params)
        it = iter(train_loader)
        sample_batch = next(it)
        assert sample_batch['X'][0].size() == (20, 224)
        assert sample_batch['Y'].size() == (20, 11)
        assert isinstance(sample_batch['X'][0], torch.Tensor)
        assert isinstance(sample_batch['Y'][0], torch.Tensor)

    def test_model_make_loader(self, common_mlp_rescale_hps):
        model = Model(common_mlp_rescale_hps)
        train_loader, val_loader = model.make_loader(filename='../data/small_parameters_base.fits')
        it = iter(train_loader)
        sample_batch = next(it)
        assert sample_batch['X'][0].size() == (20, 224)
        assert sample_batch['Y'].size() == (20, 11)
        assert isinstance(sample_batch['X'][0], torch.Tensor)
        assert isinstance(sample_batch['Y'][0], torch.Tensor)

    def test_model_fit_step(self, common_mlp_rescale_hps):
        model = Model(common_mlp_rescale_hps)
        train_loader, val_loader = model.make_loader(filename='../data/small_parameters_base.fits')
        loss = model.fit_step(train_loader)
        loss = model.eval_step(val_loader)
        assert loss > 0

    def test_model_eval_step(self, common_mlp_rescale_hps):
        model = Model(common_mlp_rescale_hps)
        train_loader, val_loader = model.make_loader(filename='../data/small_parameters_base.fits')
        loss = model.eval_step(val_loader)
        assert loss > 0

    def test_common_model_train(self, common_mlp_rescale_hps):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model = Model(hps)
        filename = '../data/small_parameters_base.fits'
        history = model.train(filename=filename, path_to_save='../res_experiments/trained_models/test.pt',
                              logdir='../res_experiments/')
        model.save_model(path ='../res_experiments/trained_models/common.pt')
        assert True


    def test_model_ind_train(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model = Model(hps)
        filename = '../data/small_parameters_base.fits'
        history = model.train(filename=filename, path_to_save='../res_experiments/trained_models/test.pt',
                              logdir='../res_experiments/')
        # model.save_model(path_to_save='../res_experiments/trained_models/test.pt')
        # list(zip(*self.net.top.task_layers[0].named_parameters()))[1][0].grad
        params_groups_0 = list(zip(*model.net.top.task_layers[0].named_parameters()))
        params_groups_3 = list(zip(*model.net.top.task_layers[3].named_parameters()))
        assert True

    def test_model_partly_ind_train(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_partly_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model = Model(hps)
        filename = '../data/small_parameters_base.fits'
        history = model.train(filename=filename, path_to_save='../res_experiments/trained_models/partly_test.pt',
                              logdir='../res_experiments/')
        # model.save_model(path_to_save='../res_experiments/trained_models/test.pt')
        assert True

    def test_partly_pretrained_init(self):
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        path_to_model = os.path.join(get_project_root(), 'res_experiments', 'trained_models', 'test.pt')

        hps = HyperParams.from_file(path_to_json=path_to_json)
        model_common = Model(hps)
        model_common.load_model(path_to_model)
        pretrained_dict = model_common.net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'bottom' in k }
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_partly_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model_part = Model(hps)
        model_dict = model_part.net.state_dict()
        model_dict.update(pretrained_dict)
        model_part.net.load_state_dict(model_dict)
        par_0_part = list(zip(*model_part.net.bottom.mlp.named_parameters()))[1][0]
        par_0_com = list(zip(*model_common.net.bottom.mlp.named_parameters()))[1][0]


    def test_load_pretrained_bottom(self):
        path_to_pretrained_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        path_to_pretrained_model = os.path.join(get_project_root(), 'res_experiments', 'trained_models', 'test.pt')
        hps_pretrained = HyperParams.from_file(path_to_json=path_to_pretrained_json)
        model_common = Model(hps_pretrained)
        model_common.load_model(path_to_pretrained_model)
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_partly_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model_part = Model(hps)
        model_part.load_pretrained_bottom(path_to_pretrained_model, path_to_pretrained_json)
        par_0_part = list(zip(*model_part.net.bottom.mlp.named_parameters()))[1][0]
        par_0_com = list(zip(*model_common.net.bottom.mlp.named_parameters()))[1][0]
        assert True

    def test_partly_pretrained_fit_step(self):
        path_to_pretrained_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        path_to_pretrained_model = os.path.join(get_project_root(), 'res_experiments', 'trained_models', 'test.pt')
        hps_pretrained = HyperParams.from_file(path_to_json=path_to_pretrained_json)
        model_common = Model(hps_pretrained)
        model_common.load_model(path_to_pretrained_model)
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_partly_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model_part = Model(hps)
        model_part.load_pretrained_bottom(path_to_pretrained_model, path_to_pretrained_json)
        train_loader, val_loader = model_part.make_loader(filename='../data/small_parameters_base.fits')
        loss = model_part.fit_step(train_loader, pretrained_bottom=True)
        par_0_com = list(zip(*model_common.net.bottom.mlp.named_parameters()))[1][0].detach().numpy()
        par_0_par = list(zip(*model_part.net.bottom.mlp.named_parameters()))[1][0].detach().numpy()


        assert par_0_com == pytest.approx(par_0_par)

    def test_partly_pretrained_train(self):
        path_to_pretrained_json = os.path.join(get_project_root(), 'res_experiments', 'hps_common_mlp.json')
        path_to_pretrained_model = os.path.join(get_project_root(), 'res_experiments', 'trained_models', 'test.pt')
        hps_pretrained = HyperParams.from_file(path_to_json=path_to_pretrained_json)
        model_common = Model(hps_pretrained)
        model_common.load_model(path_to_pretrained_model)
        path_to_json = os.path.join(get_project_root(), 'res_experiments', 'hps_partly_independent_mlp.json')
        hps = HyperParams.from_file(path_to_json=path_to_json)
        model_part = Model(hps)
        model_part.load_pretrained_bottom(path_to_pretrained_model, path_to_pretrained_json)
        filename = '../data/small_parameters_base.fits'
        history = model_part.train(filename=filename, path_to_save='../res_experiments/trained_models/partly_test.pt',
                              pretrained_bottom=True, logdir='../res_experiments/')
        par_0_com = list(zip(*model_common.net.bottom.mlp.named_parameters()))[1][0].detach().numpy()
        par_0_par = list(zip(*model_part.net.bottom.mlp.named_parameters()))[1][0].detach().numpy()


        assert par_0_com == pytest.approx(par_0_par)






