import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import Sine


def activation_factory(name, inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=inplace)
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'linear' or name is None:
        return nn.Identity()
    elif name == 'sine':
        return Sine()
    else:
        raise ValueError('Not supported activation:', name)