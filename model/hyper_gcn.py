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


class Hyper_GraphConv(nn.Module):
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

        self.hyper_edge_num = 5

        self.mlp = MLP(in_channels * self.hyper_edge_num, [out_channels], dropout=dropout, activation=activation)

        self.fea_mlp = MLP(self.hyper_edge_num, [50, 50, self.hyper_edge_num], dropout=dropout, activation=activation)

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

    def hyper_edge_convolve(self, x):
        self.local_bone_hyper_edges = self.local_bone_hyper_edges.to(x.device)
        self.center_hyper_edges = self.center_hyper_edges.to(x.device)
        self.figure_l_hyper_edges = self.figure_l_hyper_edges.to(x.device)
        self.figure_r_hyper_edges = self.figure_r_hyper_edges.to(x.device)
        self.hand_hyper_edges = self.hand_hyper_edges.to(x.device)
        # self.foot_hyper_edges = self.foot_hyper_edges.to(x.device)

        # Not make the angular feature learnable
        # print('max x: ', torch.max(x).item(), 'min x: ', torch.min(x).item())
        local_bone_w = x[:, 6, :, :]  # 6
        center_w = x[:, 7, :, :]  # 7
        figure_l_w = x[:, 9, :, :]  # 9
        figure_r_w = x[:, 10, :, :]  # 10
        hand_w = x[:, 11, :, :]  # 11
        # foot_w = x[:, 14, :, :].unsqueeze(1)  # 14

        # Makes the angular feature learnble
        # local_bone_w = x[:, 0, :, :].unsqueeze(1)  # 6
        # center_w = x[:, 0, :, :].unsqueeze(1)  # 7
        # figure_l_w = x[:, 0, :, :].unsqueeze(1)  # 9
        # figure_r_w = x[:, 0, :, :].unsqueeze(1)  # 10
        # hand_w = x[:, 0, :, :].unsqueeze(1)  # 11
        # # foot_w = x[:, 14, :, :].unsqueeze(1)  # 14
        #
        # fea_w_cat = torch.cat((local_bone_w, center_w, figure_l_w, figure_r_w,
        #                        hand_w), dim=1)
        # fea_w_cat = self.fea_mlp(fea_w_cat)
        # local_bone_w = fea_w_cat[:, 0, :, :]
        # center_w = fea_w_cat[:, 1, :, :]
        # figure_l_w = fea_w_cat[:, 2, :, :]
        # figure_r_w = fea_w_cat[:, 3, :, :]
        # hand_w = fea_w_cat[:, 4, :, :]
        # # foot_w = fea_w_cat[:, 5, :, :]

        # local bone angle
        # local_bone_hwh = torch.einsum('ve,bte->btve', self.local_bone_hyper_edges, local_bone_w)
        # local_bone_hwh = torch.einsum('btvu,un->btvn', local_bone_hwh,
        #                               torch.transpose(self.local_bone_hyper_edges, 0, 1))
        local_bone_hwh = self.normalized_aggregate(local_bone_w, self.local_bone_hyper_edges)

        # center angle
        # center_hwh = torch.einsum('ve,bte->btve', self.center_hyper_edges, center_w)
        # center_hwh = torch.einsum('btvu,un->btvn', center_hwh,
        #                           torch.transpose(self.center_hyper_edges, 0, 1))
        center_hwh = self.normalized_aggregate(center_w, self.center_hyper_edges)

        # figure left angle
        # figure_l_hwh = torch.einsum('ve,bte->btve', self.figure_l_hyper_edges, figure_l_w)
        # figure_l_hwh = torch.einsum('btvu,un->btvn', figure_l_hwh,
        #                           torch.transpose(self.figure_l_hyper_edges, 0, 1))
        figure_l_hwh = self.normalized_aggregate(figure_l_w, self.figure_l_hyper_edges)

        # figure right angle
        # figure_r_hwh = torch.einsum('ve,bte->btve', self.figure_r_hyper_edges, figure_r_w)
        # figure_r_hwh = torch.einsum('btvu,un->btvn', figure_r_hwh,
        #                           torch.transpose(self.figure_r_hyper_edges, 0, 1))
        figure_r_hwh = self.normalized_aggregate(figure_r_w, self.figure_r_hyper_edges)

        # hand angle
        # hand_hwh = torch.einsum('ve,bte->btve', self.hand_hyper_edges, hand_w)
        # hand_hwh = torch.einsum('btvu,un->btvn', hand_hwh,
        #                         torch.transpose(self.hand_hyper_edges, 0, 1))
        hand_hwh = self.normalized_aggregate(hand_w, self.hand_hyper_edges)

        # foot angle
        # foot_hwh = torch.einsum('ve,bte->btve', self.foot_hyper_edges, foot_w)
        # foot_hwh = torch.einsum('btvu,un->btvn', foot_hwh,
        #                          torch.transpose(self.foot_hyper_edges, 0, 1))

        hwh_cat = torch.cat((local_bone_hwh, center_hwh, figure_l_hwh, figure_r_hwh,
                             hand_hwh), dim=-2)

        # Softmax normalization
        # hwh_cat = torch.softmax(hwh_cat, dim=-1)

        # Use partial features
        x = x[:, :3, :, :]
        N, C, T, V = x.shape

        support = torch.einsum('btvu,bctu->bctv', hwh_cat, x)
        support = support.view(N, C, T, self.hyper_edge_num, V)
        support = support.permute(0, 3, 1, 2, 4).contiguous().view(N, self.hyper_edge_num * C, T, V)
        out = self.mlp(support)

        return out

    def forward(self, x):
        return self.hyper_edge_convolve(x)


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16, 3, 30, 25))
