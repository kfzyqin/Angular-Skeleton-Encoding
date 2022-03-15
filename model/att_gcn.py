import math
import sys

from torch.nn import TransformerEncoderLayer, TransformerEncoder

from graph.ang_adjs import get_ang_adjs
from graph.hyper_graphs import get_hyper_edge

sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.mlp import MLP
from model.activation import activation_factory


class Att_GraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=0,
                 activation='relu',
                 **kwargs):
        super().__init__()

        self.local_bone_hyper_edges = get_hyper_edge('ntu', 'local_bone')
        self.center_hyper_edges = get_hyper_edge('ntu', 'center')
        self.figure_l_hyper_edges = get_hyper_edge('ntu', 'figure_l')
        self.figure_r_hyper_edges = get_hyper_edge('ntu', 'figure_r')
        self.hand_hyper_edges = get_hyper_edge('ntu', 'hand')
        # self.foot_hyper_edges = get_hyper_edge('ntu', 'foot')

        self.hyper_edge_num = 8
        self.in_fea_mlp = MLP(self.hyper_edge_num, [50, out_channels], dropout=dropout, activation=activation)
        self.in_fea_mlp_last = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def process_hyper_edge_w(self, he_w, device):
        he_w = he_w.repeat(1, 1, 1, he_w.shape[-2])
        for i in range(he_w.shape[0]):
            for j in range(he_w.shape[1]):
                he_w[i][j] *= torch.eye(he_w.shape[-2]).to(device)
        return he_w

    def normalized_aggregate(self, w, h):
        degree_v = torch.einsum('ve,bte->btv', h, w)
        degree_e = torch.sum(h, dim=0)
        degree_v = torch.pow(degree_v, -0.5)
        # degree_v = torch.pow(degree_v, -1)
        degree_e = torch.pow(degree_e, -1)
        degree_v[degree_v == float("Inf")] = 0
        degree_v[degree_v != degree_v] = 0
        degree_e[degree_e == float("Inf")] = 0
        degree_e[degree_e != degree_e] = 0
        dh = torch.einsum('btv,ve->btve', degree_v, h)
        dhw = torch.einsum('btve,bte->btve', dh, w)
        dhwb = torch.einsum('btve,e->btve', dhw, degree_e)
        dhwbht = torch.einsum('btve,eu->btvu', dhwb, torch.transpose(h, 0, 1))
        dhwbhtd = torch.einsum('btvu,btu->btvu', dhwbht, degree_v)
        if torch.max(dhwbhtd).item() != torch.max(dhwbhtd).item():
            print('max h: ', torch.max(h).item(), 'min h: ', torch.min(h).item())
            print('max w: ', torch.max(w).item(), 'min w: ', torch.min(w).item())
            print('max degree v: ', torch.max(degree_v).item(), 'min degree v: ', torch.min(degree_v).item())
            print('max degree e: ', torch.max(degree_e).item(), 'min degree e: ', torch.min(degree_e).item())
            print('max dh: ', torch.max(dh).item(), 'min dh: ', torch.min(dh).item())
            print('max dhw: ', torch.max(dhw).item(), 'min dhw: ', torch.min(dhw).item())
            print('max dhwb: ', torch.max(dhwb).item(), 'min dhwb: ', torch.min(dhwb).item())
            print('max dhwbht: ', torch.max(dhwbht).item(), 'min dhwbht: ', torch.min(dhwbht).item())
            print('max dhwbhtd: ', torch.max(dhwbhtd).item(), 'min dhwbhtd: ', torch.min(dhwbhtd).item())
            assert 0

        # dhwbhtd[dhwbhtd != dhwbhtd] = 0
        return dhwbhtd

    def att_convolve(self, x):
        cor_w = x[:, :3, :, :]
        local_bone_w = x[:, 6, :, :].unsqueeze(1)  # 6
        center_w = x[:, 7, :, :].unsqueeze(1)  # 7
        figure_l_w = x[:, 9, :, :].unsqueeze(1)  # 9
        figure_r_w = x[:, 10, :, :].unsqueeze(1)  # 10
        hand_w = x[:, 11, :, :].unsqueeze(1)  # 11

        # Make channels more
        in_fea = torch.cat((cor_w, local_bone_w, center_w, figure_l_w, figure_r_w, hand_w), dim=1)
        in_fea = self.in_fea_mlp(in_fea)
        in_fea = self.in_fea_mlp_last(in_fea)
        in_fea = in_fea.permute(0, 2, 3, 1)

        in_fea = torch.einsum('btvm,btmu->btvu', in_fea, in_fea.permute(0, 1, 3, 2))
        in_fea = torch.softmax(in_fea, dim=-1)
        return in_fea

    def forward(self, x):
        return self.att_convolve(x)


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16, 3, 30, 25))
