from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from NormalizingFlow import NormalizingFlow
from RealNVP import RealNVP



# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

class NFFitter(object):
    
    def __init__(self, var_size=2, cond_size=2, normalize_y=True, n_layers=8,  batch_size=32, n_epochs=10, lr=0.0001):
        
        self.normalize_y = normalize_y
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.loss_history = []
        
        prior = torch.distributions.MultivariateNormal(torch.zeros(var_size), torch.eye(var_size))

        layers = []
        for i in range(n_layers):
            layers.append(RealNVP(var_size=var_size, cond_size=cond_size, mask=((torch.arange(var_size) + i) % 2)))

        self.nf = NormalizingFlow(layers=layers, prior=prior)
        self.opt = torch.optim.Adam(self.nf.parameters(), lr=self.lr)
        
        
    def reshape(self, y):
        try:
            y.shape[1]
            return y
        except:
            return y.reshape(-1, 1)
    
    
    def fit(self, X, y):
        
        # reshape
        y = self.reshape(y)
        
        # normalize
        if self.normalize_y:
            self.ss_y = StandardScaler()
            y = self.ss_y.fit_transform(y)
            
        #noise = np.random.normal(0, 1, (y.shape[0], 1))
        #y = np.concatenate((y, noise), axis=1)
        
        # numpy to tensor
        y_real = torch.tensor(y, dtype=torch.float32, device=DEVICE)
        X_cond = torch.tensor(X, dtype=torch.float32, device=DEVICE)

        
        
        # tensor to dataset
        dataset_real = TensorDataset(y_real, X_cond)
        
        criterion = nn.MSELoss()
        

        # Fit GAN
        for epoch in range(self.n_epochs):
            for i, (y_batch, x_batch) in enumerate(DataLoader(dataset_real, batch_size=self.batch_size, shuffle=True)):
                
                noise = np.random.normal(0, 1, (len(y_batch), 1))
                noise = torch.tensor(noise, dtype=torch.float32, device=DEVICE)
                y_batch = torch.cat((y_batch, noise), dim=1)
                
                y_pred = self.nf.sample(x_batch)
                
                # caiculate loss
                #loss = -self.nf.log_prob(y_batch, x_batch)
                loss = criterion(y_batch, y_pred)
                
                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                    
                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())
                    
        
    def predict(self, X):
        #noise = np.random.normal(0, 1, (X.shape[0], 1))
        #X = np.concatenate((X, noise), axis=1)
        X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_pred = self.nf.sample(X).cpu().detach().numpy()#[:, 0]
        # normalize
        if self.normalize_y:
            y_pred = self.ss_y.inverse_transform(y_pred)
        return y_pred
    
    def predict_n_times(self, X, n_times=100):
        predictions = []
        for i in range(n_times):
            y_pred = self.predict(X)
            predictions.append(y_pred)
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        return mean, std

    def predict_image_n_times(self, X, n_times=100):
        predictedImage = []
        for row in X:
            predictedRow, _ = self.predict_n_times(row, n_times=n_times)
            predictedImage.append(predictedRow)
        return np.array(predictedImage)