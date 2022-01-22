from pathlib import Path
import pytest
import torch
from inverse_problem.milne_edington.data_utils import get_project_root
from inverse_problem.nn_inversion.dataloader import make_loader


def test_make_loader_spectrum_no_transform():
    project_path = get_project_root()
    filename = project_path / 'data' / 'small_parameters_base.fits'
    pregen = False
    transform_name = None
    batch_size = 20
    train_loader, val_loader = make_loader(filename=filename, transform_name=transform_name, batch_size=20)
    it = iter(train_loader)
    sample_batch = next(it)
    assert sample_batch['X'][0].size() == (20, 56, 4)
    assert sample_batch['Y'].size() == (20, 11)
    assert isinstance(sample_batch['X'][0], torch.Tensor)
    assert isinstance(sample_batch['Y'][0], torch.Tensor)


def test_make_loader_spectrum_mlp_transform_default():
    project_path = get_project_root()
    filename = project_path / 'data' / 'small_parameters_base.fits'
    pregen = False
    transform_name = "mlp_transform_rescale"
    batch_size = 20
    train_loader, val_loader = make_loader(filename=filename, batch_size=20)
    it = iter(train_loader)
    sample_batch = next(it)
    assert sample_batch['X'][0].size() == (20, 224)
    assert sample_batch['X'][1].size == (20, 1)
    assert sample_batch['Y'].size() == (20, 11)
    assert isinstance(sample_batch['X'][0], torch.Tensor)
    assert isinstance(sample_batch['Y'][0], torch.Tensor)


def test_make_loader_spectrum_mlp_transform_kwargs():
    project_path = get_project_root()
    filename = project_path / 'data' / 'small_parameters_base.fits'
    pregen = False
    transform_name = "mlp_transform_rescale"
    batch_size = 20
    train_loader, val_loader = make_loader(filename=filename, batch_size=20, transform_name=transform_name,
                                           logB=False, angle_transformation=True)
    it = iter(train_loader)
    sample_batch = next(it)
    assert sample_batch['X'][0].size() == (20, 224)
    assert sample_batch['Y'].size() == (20, 11)
    assert isinstance(sample_batch['X'][0], torch.Tensor)
    assert isinstance(sample_batch['Y'][0], torch.Tensor)


def test_make_loader_spectrum_pregen():
    project_path = get_project_root()
    filename = project_path / 'data' / 'small_parameters_base.fits'
    pregen = False
    transform_name = "mlp_transform_rescale"
    batch_size = 20
    train_loader, val_loader = make_loader(filename=filename, batch_size=20, pregen=True)
    it = iter(train_loader)
    sample_batch = next(it)
    assert sample_batch['X'][0].size() == (20, 224)
    assert sample_batch['X'][1].shape == (20, 1)
    assert sample_batch['Y'].size() == (20, 11)
    assert isinstance(sample_batch['X'][0], torch.Tensor)
    assert isinstance(sample_batch['Y'][0], torch.Tensor)
