import math
import sys

from torch.nn import TransformerEncoderLayer, TransformerEncoder

from graph.ang_adjs import get_ang_adjs
from model.hyper_gcn import Hyper_GraphConv

sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.mlp import MLP
from model.activation import activation_factory


class MultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu',
                 to_use_hyper_conv=False,
                 **kwargs):
        super().__init__()
        self.num_scales = num_scales

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary)) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        self.A_powers = torch.Tensor(A_powers)

        if 'hyper_conv' in kwargs and kwargs['hyper_conv'] == 'ntu':
            hyper_adjs = get_ang_adjs('ntu')
            self.A_powers = torch.cat((self.A_powers, hyper_adjs), dim=0)
            if kwargs['hyper_conv'] == 'ntu':
                self.num_scales += 6
            elif kwargs['hyper_conv'] == 'kinetics':
                self.num_scales += 4

        # self.A_powers_param = torch.nn.Parameter(self.A_powers)

        self.use_mask = use_mask
        if use_mask:
            # NOTE: the inclusion of residual mask appears to slow down training noticeably
            self.A_res = nn.init.uniform_(nn.Parameter(torch.Tensor(self.A_powers.shape)), -1e-6, 1e-6)

        # 这个MLP根本就不是MLP, 这是卷积, 只是类似于MLP的功能.
        self.mlp = MLP(in_channels * self.num_scales, [out_channels], dropout=dropout, activation=activation)

        # Spatial Transformer Attention
        if 'to_use_spatial_transformer' in kwargs and kwargs['to_use_spatial_transformer']:
            self.to_use_spatial_trans = True
            self.trans_conv = nn.Conv2d(out_channels, 1, (1, 1), (1, 1))
            self.temporal_len = kwargs['temporal_len']
            nhead = 5
            nlayers = 2
            trans_dropout = 0.5
            encoder_layers = nn.TransformerEncoderLayer(self.temporal_len,
                                                        nhead=nhead, dropout=trans_dropout)
            self.trans_enc = nn.TransformerEncoder(encoder_layers, nlayers)

            # spatial point normalization
            self.point_norm_layer = nn.Sigmoid()

        else:
            self.to_use_spatial_trans = False

        if 'to_use_sptl_trans_feature' in kwargs and kwargs['to_use_sptl_trans_feature']:
            self.to_use_sptl_trans_feature = True
            self.fea_dim = kwargs['fea_dim']
            encoder_layers = nn.TransformerEncoderLayer(self.fea_dim,
                                                        nhead=kwargs['sptl_trans_feature_n_head'],
                                                        dropout=0.5)
            self.trans_enc_fea = nn.TransformerEncoder(encoder_layers,
                                                       kwargs['sptl_trans_feature_n_layer'])
        else:
            self.to_use_sptl_trans_feature = False

    def forward(self, x):
        N, C, T, V = x.shape
        self.A_powers = self.A_powers.to(x.device)

        A = self.A_powers.to(x.dtype)
        if self.use_mask:
            A = A + self.A_res.to(x.dtype)

        support = torch.einsum('vu,nctu->nctv', A, x)

        support = support.view(N, C, T, self.num_scales, V)
        support = support.permute(0, 3, 1, 2, 4).contiguous().view(N, self.num_scales * C, T, V)

        out = self.mlp(support)

        # 实现kernel中, 只实现了一半.
        # out = torch.einsum('nijtv,njktv->niktv', out.unsqueeze(2), out.unsqueeze(1)).view(
        #     N, self.out_channels * self.out_channels, T, V
        # )

        if self.to_use_spatial_trans:
            out_mean = self.trans_conv(out).squeeze()
            out_mean = out_mean.permute(0, 2, 1)
            out_mean = self.trans_enc(out_mean)
            out_mean = self.point_norm_layer(out_mean)
            out_mean = out_mean.permute(0, 2, 1)
            out_mean = torch.unsqueeze(out_mean, dim=1).repeat(1, out.shape[1], 1, 1)
            out = out_mean * out

        if self.to_use_sptl_trans_feature:
            out = out.permute(2, 3, 0, 1)
            for a_out_idx in range(len(out)):
                a_out = out[a_out_idx]
                a_out = self.trans_enc_fea(a_out)
                out[a_out_idx] = a_out
            out = out.permute(2, 3, 0, 1)

        return out


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph

    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = MultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16, 3, 30, 25))
