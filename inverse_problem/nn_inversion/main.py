from .dataset import SpectrumDataset
import torch
from tqdm import tqdm
from torch import nn
import os
from pathlib import Path
from torch.utils.data import DataLoader
from inverse_problem.nn_inversion.models import HyperParams, FullModel
from inverse_problem.nn_inversion import models
from inverse_problem.nn_inversion import transforms
import numpy as np


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

    def fit_step(self, sample_batch):
        train_loss = 0.0
        train_it = 0
        for i, inputs in enumerate(sample_batch):
            if self.hps.per_epoch == i:
                break
            self.optimizer.zero_grad()
            x = inputs['X'][0].to(self.device) #, inputs['X'][1].to(self.device)]
            # print(x.shape)
            y = inputs['Y'][:, self.hps.predict_ind].to(self.device)
            outputs = self.net(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_it += 1
        return train_loss / train_it

    def eval_step(self, sample_batch):
        self.net.eval()
        val_loss = 0.0
        val_it = 0
        for i, inputs in enumerate(sample_batch):
            if self.hps.per_epoch == i:
                break
            x = inputs['X'][0].to(self.device) #, inputs['X'][1].to(self.device)]

            y = inputs['Y'][:, self.hps.predict_ind].to(self.device)
            with torch.no_grad():
                outputs = self.net(x)
                loss = self.criterion(outputs, y)
                val_loss += loss.item()
            val_it += 1
        return val_loss / val_it

    def make_loader(self, filename: Path = None, ff=True, noise=True) -> DataLoader:
        """
        Args:
            noise (bool):
            ff (bool):
            filename (): str, Optional; Path where to load data from

        Returns:
            DataLoader
        """
        if filename is None:
            project_path = Path(__file__).resolve().parents[2]
            filename = os.path.join(project_path, 'data/parameters_base.fits')
        transformed_dataset = SpectrumDataset(filename, source=self.hps.source,
                                              transform=self.transform, ff=ff, noise=noise)
        return DataLoader(transformed_dataset, batch_size=self.hps.batch_size, shuffle=True)

    def train(self, filename=None, save_model=False, path_to_save=None, save_epoch=[], ff=True, noise=True, scheduler=False):
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

        Returns:
            List, training process history
        """
        train_loader = self.make_loader(filename, ff=ff, noise=noise)
        val_loader = self.make_loader(filename, ff=ff, noise=noise)
        best_valid_loss = float('inf')
        history = []
        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
         val_loss {v_loss:0.4f}"

        with tqdm(desc="epoch", total=self.hps.n_epochs) as pbar_outer:
            for epoch in range(self.hps.n_epochs):
                train_loss = self.fit_step(train_loader)
                val_loss = self.eval_step(val_loader)
                history.append((train_loss, val_loss))

                if scheduler:
                    self.scheduler.step(val_loss)

                if save_model:
                    if save_epoch:
                        # todo чтобы каждый чекпоинт сохранялся в свой файл
                        if epoch in save_epoch:
                            self.save_model(path_to_save, epoch, val_loss)
                    elif val_loss < best_valid_loss:
                        best_valid_loss = val_loss
                        self.save_model(path_to_save, epoch, val_loss)

                #if tensorboard:
                    #with train_summary_writer.as_default():
                        #tf.summary.scalar('loss', loss.item(), step=globaliter)


                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                               v_loss=val_loss))
        return history

    def _init_tensorboard(self):
        pass

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
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if epoch and loss:
            print('model was saved at {} epoch with {} validation loss'.format(epoch+1, loss))
        self.train(**kwargs)
        # todo беда с номером эпохи

    def load_model(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    def predict_one_pixel(self, x):
        """ Predicts one pixel
        Args:
            x: list of torch.tensors where [0] - spectrum lines (1, 512, 224)
                [1] - cont
        Returns: torch.tensor of shape (512, n), n - number of predicted parameters

        """
        line = torch.FloatTensor(x[0])[0]
        cont = torch.FloatTensor(x[1])
        self.net.eval()
        with torch.no_grad():
            predicted = self.net((line, cont))
        return predicted

    def predict_full_image(self, x, parameter):
        """ Predicts full image
        Args:
            x (tuple): [0] array of size (n, 512), [1] continuum vector;
            parameter (int): index of parameter to predict
        """
        line = torch.FloatTensor(x[0])
        cont = torch.FloatTensor(x[1])
        output = np.zeros(line.shape[:2])
        self.net.eval()
        with torch.no_grad():
            for i in range(line.shape[0]):
                predicted = self.net((line[i], cont))
                output[i] = predicted[:, parameter]
        return output

