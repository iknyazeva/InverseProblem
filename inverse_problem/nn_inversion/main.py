from inverse_problem.nn_inversion.dataset import SpectrumDataset
import torch
from inverse_problem.nn_inversion.posthoc import compute_metrics, open_param_file
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from tqdm import tqdm
from torch import nn
import os
from pathlib import Path
from torch.utils.data import DataLoader
from inverse_problem.nn_inversion.models import HyperParams, FullModel
from inverse_problem.nn_inversion import models
from inverse_problem.nn_inversion import transforms
from inverse_problem.milne_edington.me import HinodeME, me_model
from inverse_problem.nn_inversion.transforms import normalize_output
import numpy as np
from astropy.io import fits


class Model:
    """
    Model class for fitting data
    Methods:
    make_loader(): returns DataLoader
    train(): performs model training
    _init_transform(): returns transforms for data
    _init_optimizer(): returns optimizer for model training
    """

    def __init__(self, hps: HyperParams):
        """
        Args:
            hps (): HyperParams object or file to read parameters from
        """
        self.hps = hps
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.top_net = getattr(models, hps.top_net)
        self.bottom_net = getattr(models, hps.bottom_net)
        self.net = FullModel(hps, self.bottom_net, self.top_net).to(self.device)
        self.alpha = 0.00001
        self.optimizer = self._init_optimizer()
        self.transform = self._init_transform()
        self.scheduler = self._init_scheduler()

    def _init_transform(self):
        """
        Returns: Composition of transforms which will be applied to data
            transforms are taken from hps file
        """
        transform_type = self.hps.transform_type
        factors = self.hps.factors
        cont_scale = self.hps.cont_scale
        norm_output = self.hps.norm_output
        logB = self.hps.logB
        mode = self.hps.mode
        rescale_kwargs = {'factors': factors, 'cont_scale': cont_scale,
                          'norm_output': norm_output, 'logB': logB, 'mode': mode}
        normal_kwargs = {'logB': logB, 'norm_output': norm_output, 'mode': mode}
        tsfm_kwargs = {'mlp_transform_rescale': rescale_kwargs,
                       'mlp_transform_standard': normal_kwargs,
                       'conv1d_transform_rescale': rescale_kwargs,
                       'conv1d_transform_standard': normal_kwargs}
        return getattr(transforms, transform_type)(**tsfm_kwargs[transform_type])

    def _init_optimizer(self):
        """
        Returns: Adam optimizer instance
            learning rate and weight decay rate are taken from hps file
        """
        return torch.optim.Adam(self.net.parameters(),
                                lr=self.hps.lr, weight_decay=self.hps.weight_decay)

    def _init_scheduler(self):
        # todo добавить patience в hps?
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def _init_tensorboard(self, logdir=None, comment=''):
        return SummaryWriter(log_dir=logdir, comment=comment)

    def load_pretrained_bottom(self, path_to_model, path_to_json):

        hps = HyperParams.from_file(path_to_json=path_to_json)
        model_common = Model(hps)
        model_common.load_model(path_to_model)
        pretrained_dict = model_common.net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'bottom' in k}
        model_dict = self.net.state_dict()
        model_dict.update(pretrained_dict)
        self.net.load_state_dict(model_dict)

    def fit_step(self, dataloader, pretrained_bottom=False):
        train_loss = 0.0
        train_it = 0
        total = self.hps.trainset or len(dataloader)
        with tqdm(desc="batch", total=total) as pbar_outer:
            for i, inputs in enumerate(dataloader):
                if self.hps.trainset:
                    if self.hps.trainset == i:
                        break
                x = [inputs['X'][0].to(self.device), inputs['X'][1].to(self.device)]
                # print(x.shape)
                y = inputs['Y'][:, self.hps.predict_ind].to(self.device)
                if pretrained_bottom:
                    with torch.no_grad():
                        outputs = self.net.bottom(x[0])
                    outputs = torch.cat((outputs, x[1]), axis=1)
                    outputs = self.net.top(outputs)
                else:
                    outputs = self.net(x)
                if "independent" in self.hps.hps_name:
                    self.optimizer.zero_grad()
                    losses = [self.criterion(outputs[:, ind], y[:, ind]) for ind in self.hps.predict_ind]
                    loss = torch.stack(losses).mean()
                    # loss.backward()
                    torch.autograd.backward(losses)
                    self.optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()
                train_loss += loss.item()
                train_it += 1
                if train_it % 100 == 0:
                    pbar_outer.update(100)
        return train_loss / train_it

    def eval_step(self, dataloader):
        self.net.eval()
        val_loss = 0.0
        val_it = 0
        for i, inputs in enumerate(dataloader):
            if self.hps.valset:
                if self.hps.valset == i:
                    break
            x = [inputs['X'][0].to(self.device), inputs['X'][1].to(self.device)]
            y = inputs['Y'][:, self.hps.predict_ind].to(self.device)
            with torch.no_grad():
                outputs = self.net(x)
                loss = self.criterion(outputs, y)
                val_loss += loss.item()
            val_it += 1
        return val_loss / val_it

    def make_loader(self, data_arr=None, filename: Path = None, ff=True, noise=True, val_split=0.1) -> DataLoader:
        """
        Args:
            noise (bool): add noise or not
            ff (bool): with filling factor
            filename (): str, Optional; Path where to load data from
        Returns:
            DataLoader
        """
        # if filename is None:
        #    project_path = Path(__file__).resolve().parents[2]
        #    filename = os.path.join(project_path, 'data/parameters_base.fits')
        if data_arr is None and filename is None:
            raise AssertionError('you need provide data or path to data')
        transformed_dataset = SpectrumDataset(data_arr=data_arr, param_path=filename, source=self.hps.source,
                                              transform=self.transform, ff=ff, noise=noise)
        train_idx, val_idx = train_test_split(list(range(len(transformed_dataset))), test_size=val_split)
        train_dataset = Subset(transformed_dataset, train_idx)
        val_dataset = Subset(transformed_dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=self.hps.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.hps.batch_size, shuffle=True)
        return train_loader, val_loader

    def train(self, data_arr=None, filename=None, pretrained_bottom=False, path_to_save=None, save_epoch=[],
              ff=True, noise=True, scheduler=False, tensorboard=False, logdir=None, comment=''):
        """
            Function for model training
        Args:
            save_model (bool): whether to save checkpoint, if True saves every best validation loss by default
            path_to_save (str):
            save_epoch (list of ints): save checkpoint every given epoch
            scheduler (): whether to use scheduler
            filename (): str, Optional; Path where to load data from
            model_path (): str, Optional; Path to save model to
            noise (): whether to use noise
            ff (): whether to use ff
            tensorboard ():
        Returns:
            List, training process history
        """
        train_loader, val_loader = self.make_loader(data_arr, filename, ff=ff, noise=noise,
                                                    val_split=self.hps.val_split)
        best_valid_loss = float('inf')
        history = []
        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
         val_loss {v_loss:0.4f}"

        with tqdm(desc="epoch", total=self.hps.n_epochs) as pbar_outer:
            for epoch in range(self.hps.n_epochs):
                train_loss = self.fit_step(train_loader, pretrained_bottom=pretrained_bottom)
                val_loss = self.eval_step(val_loader)
                history.append((train_loss, val_loss))

                if scheduler:
                    self.scheduler.step(val_loss)

                if path_to_save:
                    if save_epoch:
                        # todo чтобы каждый чекпоинт сохранялся в свой файл
                        if epoch in save_epoch:
                            self.save_model(path_to_save, epoch, val_loss)
                    elif val_loss < best_valid_loss:
                        best_valid_loss = val_loss
                        self.save_model(path_to_save, epoch, val_loss)

                if tensorboard:
                    self.tensorboard_writer = self._init_tensorboard(logdir, comment)
                    self.tensorboard_writer.add_scalar("Loss/train", train_loss, epoch)
                    self.tensorboard_writer.add_scalar("Loss/val", val_loss, epoch)

                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                               v_loss=val_loss))
                if logdir:
                    with open(os.path.join(logdir, 'history_' + self.hps.hps_name + '.txt'), 'w') as f:
                        for i, item in enumerate(history):
                            f.write(f"Train loss in epoch {i}: {item[0]: .4f}, val_loss: {item[1]:.4f}\n")
        return history

    def save_model(self, path, epoch=None, loss=None):
        """
        Args:
            path (str): path to save model to
            epoch (int): optional
            loss (float): optional, validation loss
        Returns:
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss}, path)

    def continue_training(self, checkpoint_path, **kwargs):
        """
        Loads model from checkpoint and continues training
        Args:
            checkpoint_path (str): path to load checkpoint from
            **kwargs (): args from train()
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if epoch and loss:
            print('model was saved at {} epoch with {} validation loss'.format(epoch, loss))
        self.train(**kwargs)
        # todo беда с номером эпохи

    def load_model(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['model_state_dict'])

    def predict_one_pixel(self, refer, idx_0, idx_1, **kwargs):
        """ Predicts one pixel
        Args:
        Returns: predicted params, transformed target params, computed lines, cont
        """
        hinode = HinodeME.from_refer(idx_0, idx_1, refer)
        param_vec = hinode.param_vector
        x = hinode.compute_spectrum(**kwargs)
        # line = torch.FloatTensor(x).to(self.device)
        cont = torch.tensor(hinode.cont, dtype=torch.float).to(self.device)
        data = {'X': [x, cont], 'Y': param_vec}
        data = self.transform(data)
        self.net.eval()
        with torch.no_grad():
            predicted = self.net([data['X'][0].unsqueeze(0).to(self.device), data['X'][1].unsqueeze(0).to(self.device)])
        return predicted.cpu(), data['Y'], data['X'][0], data['X'][1]

    def predict_full_image(self, refer_path, predicted_tofits, **kwargs):
        """ Predicts full image
        Args:
            refer - path to refer
            predicted_tofits - path to save prediction or False
        """
        with fits.open(refer_path) as refer:
            out = np.zeros(refer[1].data.shape + (self.hps.top_output,))
            params = np.zeros(refer[1].data.shape + (11,))
            for i in range(out.shape[0]):
                for t in range(out.shape[1]):
                    out[i, t], params[i, t], _, _ = self.predict_one_pixel(refer, i, t, **kwargs)
            if predicted_tofits:
                hdr = fits.getheader(refer_path)
                fits.writeto(predicted_tofits, out, hdr)
            return out, params

    def predict_refer(self, refer_path):
        refer, names = open_param_file(refer_path, normalize=False)
        shape = refer.shape
        params = refer.reshape(-1, 11)
        predicted = self.predict_by_batches(params, batch_size=1000)
        return predicted.reshape(shape)

    def predict_by_batches(self, params, batch_size=100):

        length = params.shape[0]
        n_batches = length//batch_size
        predict = np.zeros((length, 11))
        for i in tqdm(range(n_batches)):
            predict[i*batch_size:batch_size*(i+1), :] = self.predict_from_batch(params[i*batch_size:batch_size*(i+1), :])
        if n_batches*batch_size < length:
            predict[n_batches*batch_size:, :] = self.predict_from_batch(params[batch_size*n_batches:, :])
        return predict



    def predict_from_batch(self, param_vector, noise=True):
        data = self.generate_batch_spectrum(param_vector, noise=noise)
        self.net.eval()
        with torch.no_grad():
            predicted = self.net(data['X'])
        return predicted.cpu().detach().numpy()

    def generate_batch_spectrum(self, param_vector, noise=True):
        line_vec = (6302.5, 2.5, 1)
        line_arg = 1000 * (np.linspace(6302.0692255, 6303.2544205, 56) - line_vec[0])
        spectrum = me_model(param_vector, line_arg, line_vec, with_ff=True, with_noise=noise)
        cont = param_vector[:, 6] + line_vec[2] * param_vector[:, 7]
        cont = cont * np.amax(spectrum.reshape(-1, 224), axis=1) / self.hps.cont_scale
        cont = torch.from_numpy(cont.reshape(-1, 1)).float().to(self.device)
        y = normalize_output(param_vector, mode=self.hps.mode, logB=self.hps.logB)
        y = torch.from_numpy(y).float().to(self.device)
        if 'rescale' in self.hps.transform_type:
            rescaled = (np.swapaxes(spectrum, 0, 2) * np.array(self.hps.factors).reshape(4, 1, 1)).swapaxes(0, 2)
            if 'mlp' in self.hps.transform_type:
                rescaled = rescaled.reshape(-1, 224, order='F')
            rescaled = torch.from_numpy(rescaled).float().to(self.device)
        else:
            NotImplementedError('Only rescale transform')
        data = {'X': [rescaled, cont], 'Y': y}
        return data

    def tensorboard_flush(self):
        self.tensorboard_writer.flush()

    def tensorboard_close(self):
        self.tensorboard_writer.close()
