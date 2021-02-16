import numpy as np
from torchvision import transforms
import torch
import os

from inverse_problem.nn_inversion.models import Conv1dModel
from inverse_problem.nn_inversion.main import HyperParams, Model
import matplotlib.pyplot as plt

#[20, 56, 4]
#[32, 4, 2]
hps = HyperParams()
hps.n_epochs = 50
hps.per_epoch = 100
hps.transform_type = 'conv1d_transform_standard'
model = Model(hps)
model.net = Conv1dModel(hps)
history = model.train()
_ = plt.plot(history)
plt.ylabel('loss')
plt.show()
