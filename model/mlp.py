import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation import activation_factory
from utils_dir.utils_visual import plot_multiple_lines


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(activation_factory(activation, inplace=False))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        # 这里是学习同一个joint的不同尺度的信息.
        # the_mlp_w = torch.sum(self.layers[0].weight, dim=0).squeeze().view(13, -1).sum(0).cpu().numpy()
        # print('the_mlp_w: ', the_mlp_w)
        # plot_multiple_lines([the_mlp_w])
        # assert 0
        for layer in self.layers:
            x = layer(x)
        return x

