import os
import sys

sys.path.append('..')
sys.path.append('../..')

import torch
import torch.nn as nn

import numpy as np
import numpy.ma as ma

from astropy.io import fits

from tqdm import tqdm

from inverse_problem import make_loader

from inverse_problem.nn_inversion.model_pi_mlp_conv import PIMLPConvNet

from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
filename = '../../data/parameters_base.fits'

factors, cont_scale = [1, 1000, 1000, 1000], 40000
angle_transformation, logB = True, True

transform_name = "conv1d_transform_rescale"

n_epoch = 1
batch_size = 128
num_workers = 0

model_name = 'bottleneck'

current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))


def main():
    model = PIMLPConvNet(n_blocks=6, in_dim=(4, 64, 64, 128, 128, 256), out_dim=(64, 64, 128, 128, 256, 256),
                         kernel_size=(3, 3, 3, 3, 3, 3), padding=(1, 1, 1, 1, 1, 1), activation='elu', dropout=0.05,
                         batch_norm=True, pool=(None, 'max', None, 'max', None, None), hidden_dims=(100, 100),
                         bottom_output=100, number_readout_layers=2, top_output=11)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)

    train(model, criterion, optimizer, n_epoch=n_epoch)


def params_masked_rows(pars_arr):
    max_par_values = np.array([par_arr.max() for par_arr in pars_arr.T])
    min_par_values = np.array([par_arr.min() for par_arr in pars_arr.T])

    bool_arr = (min_par_values + 1e-3 < pars_arr) & (pars_arr < max_par_values - 1e-3)
    return np.all(bool_arr, axis=1)


def create_masked_array(pars_arr):
    rows_mask = params_masked_rows(pars_arr)
    array_mask = rows_mask[:, np.newaxis] | np.zeros_like(pars_arr, dtype=bool)
    return ma.masked_array(pars_arr, mask=~array_mask)


def params_filtration(pars_arr):
    rows_mask_params = params_masked_rows(pars_arr)
    return pars_arr[rows_mask_params, :]


def fit_step(model, criterion, optimizer, dataloader, max_steps=None):
    train_loss = 0.0
    train_it = 0
    if max_steps is None:
        max_steps = float('inf')
    total = min(max_steps, len(dataloader))

    with tqdm(desc="fit_batch", total=total, position=0, leave=True) as pbar_outer:
        for i, inputs in enumerate(dataloader):
            if i == total:
                break

            x = [inputs['X'][0].to(device), inputs['X'][1].to(device)]
            y = inputs['Y'].to(device)

            outputs = model(x)

            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_it += 1

            if train_it % 10 == 0:
                pbar_outer.update(10)

        return train_loss / train_it


def eval_step(model, criterion, dataloader, max_steps=None):
    model.eval()
    val_loss = 0.0
    val_it = 0

    if max_steps is None:
        max_steps = float('inf')
    total = min(max_steps, len(dataloader))

    with tqdm(desc="val_batch", total=total, position=0, leave=True) as pbar_outer:
        for i, inputs in enumerate(dataloader):
            if i == total:
                break

            x = [inputs['X'][0].to(device), inputs['X'][1].to(device)]
            y = inputs['Y'].to(device)

            with torch.no_grad():
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_it += 1

            if val_it % 10 == 0:
                pbar_outer.update(10)
        return val_loss / val_it


def save_model(model, optimizer, path, epoch, loss):
    """
    Args:
        path (str): path to save model to
        epoch (int): optional
        loss (float): optional, validation loss
    Returns:
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        path + model_name + '_' + f'ep{epoch}.pt')


def train(model, criterion, optimizer, n_epoch, log_dir=None, path_to_save=None, max_steps=None):
    loss_history = []
    best_valid_loss = float('inf')

    params = fits.open(filename)[0].data
    filtered_params = params_filtration(params)

    train_loader, val_loader = make_loader(data_arr=filtered_params, transform_name=transform_name,
                                           factors=factors, cont_scale=cont_scale,
                                           logB=logB, angle_transformation=angle_transformation,
                                           batch_size=batch_size, num_workers=num_workers)

    sample_batch = next(iter(train_loader))

    print(f'Device: {device.type}\n')

    print('Size of spectrum batch: ', sample_batch['X'][0].shape)
    print('Size of cont batch: ', sample_batch['X'][1].shape)
    print('Size of true params batch: ', sample_batch['Y'].shape)

    print(f'\nNumber of batches for train: {len(train_loader)}, for validation: {len(val_loader)}')

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"

    with tqdm(desc="epoch", total=n_epoch, position=0, leave=True) as pbar_outer:
        for epoch in range(n_epoch):
            train_loss = fit_step(model, criterion, optimizer, train_loader, max_steps=max_steps)
            val_loss = eval_step(model, criterion, val_loader, max_steps=max_steps)
            loss_history.append((train_loss, val_loss))
            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss, v_loss=val_loss))

            if path_to_save:
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    save_model(path_to_save, optimizer, epoch, val_loss)

            if log_dir:
                with open(os.path.join(log_dir, 'history_' + model_name + '_' + current_time + '.txt'), 'w') as f:
                    for i, item in enumerate(loss_history):
                        f.write(f"Train loss in epoch {i}: {item[0]: .4f}, val_loss: {item[1]:.4f}\n")

    return loss_history


if __name__ == '__main__':
    main()
