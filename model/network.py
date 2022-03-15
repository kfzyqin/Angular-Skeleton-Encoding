import copy
import sys

from model.att_gcn import Att_GraphConv
from model.hyper_gcn import Hyper_GraphConv
# from model.transformers import get_pretrained_transformer

sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.ms_gcn import MultiScale_GraphConv as MS_GCN
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.ms_gtcn import SpatialTemporal_MS_GCN, UnfoldTemporalWindows
from model.mlp import MLP
from model.activation import activation_factory


class MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_size,
                 window_stride,
                 window_dilation,
                 embed_factor=1,
                 nonlinear='relu'):
        super().__init__()

        self.window_size = window_size
        self.out_channels = out_channels
        self.embed_channels_in = self.embed_channels_out = out_channels // embed_factor
        if embed_factor == 1:
            self.in1x1 = nn.Identity()
            self.embed_channels_in = self.embed_channels_out = in_channels
            # The first STGC block changes channels right away; others change at collapse
            if in_channels == 3:
                self.embed_channels_out = out_channels
        else:
            self.in1x1 = MLP(in_channels, [self.embed_channels_in])

        self.gcn3d = nn.Sequential(
            UnfoldTemporalWindows(window_size, window_stride, window_dilation),
            SpatialTemporal_MS_GCN(
                in_channels=self.embed_channels_in,
                out_channels=self.embed_channels_out,
                A_binary=A_binary,
                num_scales=num_scales,
                window_size=window_size,
                use_Ares=True,
                activation=nonlinear
            )
        )

        self.out_conv = nn.Conv3d(self.embed_channels_out, out_channels, kernel_size=(1, self.window_size, 1))
        self.out_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, _, T, V = x.shape
        x = self.in1x1(x)
        # Construct temporal windows and apply MS-GCN
        x = self.gcn3d(x)

        # Collapse the window dimension
        x = x.view(N, self.embed_channels_out, -1, self.window_size, V)
        x = self.out_conv(x).squeeze(dim=3)
        x = self.out_bn(x)

        # no activation
        return x


class MultiWindow_MS_G3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A_binary,
                 num_scales,
                 window_sizes=[3, 5],
                 window_stride=1,
                 window_dilations=[1, 1]):
        super().__init__()
        self.gcn3d = nn.ModuleList([
            MS_G3D(
                in_channels,
                out_channels,
                A_binary,
                num_scales,
                window_size,
                window_stride,
                window_dilation
            )
            for window_size, window_dilation in zip(window_sizes, window_dilations)
        ])

    def forward(self, x):
        # Input shape: (N, C, T, V)
        out_sum = 0
        for gcn3d in self.gcn3d:
            out_sum += gcn3d(x)
        # no activation
        return out_sum

ntu_bone_angle_pairs = {
    25: (24, 12),
    24: (25, 12),
    12: (24, 25),
    11: (12, 10),
    10: (11, 9),
    9: (10, 21),
    21: (9, 5),
    5: (21, 6),
    6: (5, 7),
    7: (6, 8),
    8: (23, 22),
    22: (8, 23),
    23: (8, 22),
    3: (4, 21),
    4: (4, 4),
    2: (21, 1),
    1: (17, 13),
    17: (18, 1),
    18: (19, 17),
    19: (20, 18),
    20: (20, 20),
    13: (1, 14),
    14: (13, 15),
    15: (14, 16),
    16: (16, 16)
}

ntu_bone_adj = {
    25: 12,
    24: 12,
    12: 11,
    11: 10,
    10: 9,
    9: 21,
    21: 21,
    5: 21,
    6: 5,
    7: 6,
    8: 7,
    22: 8,
    23: 8,
    3: 21,
    4: 3,
    2: 21,
    1: 2,
    17: 1,
    18: 17,
    19: 18,
    20: 19,
    13: 1,
    14: 13,
    15: 14,
    16: 15
}


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 ablation='original',
                 to_use_final_fc=True,
                 to_fc_last=True,
                 frame_len=300,
                 nonlinear='relu',
                 **kwargs):
        super(Model, self).__init__()

        # cosine
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

        # Activation function
        self.nonlinear_f = activation_factory(nonlinear)

        # ZQ ablation studies
        self.ablation = ablation

        Graph = import_class(graph)
        A_binary = Graph().A_binary

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # channels
        # c1 = 96
        # c2 = c1 * 2  # 192
        # c3 = c2 * 2  # 384

        c1 = 96
        self.c1 = c1
        c2 = c1 * 2  # 192  # Original implementation
        self.c2 = c2
        c3 = c2 * 2  # 384  # Original implementation
        self.c3 = c3

        # r=3 STGC blocks

        # MSG3D
        self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1_msgcn = MS_GCN(num_gcn_scales, in_channels, c1, A_binary, disentangled_agg=True,
                                  **kwargs, temporal_len=frame_len, fea_dim=c1, to_use_hyper_conv=True,
                                  activation=nonlinear)
        self.sgcn1_ms_tcn_1 = MS_TCN(c1, c1, activation=nonlinear)
        self.sgcn1_ms_tcn_2 = MS_TCN(c1, c1, activation=nonlinear)
        self.sgcn1_ms_tcn_2.act = nn.Identity()

        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.tcn1 = MS_TCN(c1, c1, **kwargs,
                               section_size=kwargs['section_sizes'][0], num_point=num_point,
                               fea_dim=c1, activation=nonlinear)
        else:
            self.tcn1 = MS_TCN(c1, c1, **kwargs, fea_dim=c1, activation=nonlinear)

        # MSG3D
        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2_msgcn = MS_GCN(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True,
                                  **kwargs, temporal_len=frame_len, fea_dim=c1, activation=nonlinear)
        self.sgcn2_ms_tcn_1 = MS_TCN(c1, c2, stride=2, activation=nonlinear)
        # self.sgcn2_ms_tcn_1 = MS_TCN(c1, c2, activation=nonlinear)
        self.sgcn2_ms_tcn_2 = MS_TCN(c2, c2, activation=nonlinear)
        self.sgcn2_ms_tcn_2.act = nn.Identity()

        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.tcn2 = MS_TCN(c2, c2, **kwargs,
                               section_size=kwargs['section_sizes'][1], num_point=num_point,
                               fea_dim=c2, activation=nonlinear)
        else:
            self.tcn2 = MS_TCN(c2, c2, **kwargs, fea_dim=c2, activation=nonlinear)

        # MSG3D
        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3_msgcn = MS_GCN(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True,
                                  **kwargs, temporal_len=frame_len // 2, fea_dim=c2,
                                  activation=nonlinear)
        self.sgcn3_ms_tcn_1 = MS_TCN(c2, c3, stride=2, activation=nonlinear)
        # self.sgcn3_ms_tcn_1 = MS_TCN(c2, c3, activation=nonlinear)
        self.sgcn3_ms_tcn_2 = MS_TCN(c3, c3, activation=nonlinear)
        self.sgcn3_ms_tcn_2.act = nn.Identity()

        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.tcn3 = MS_TCN(c3, c3, **kwargs,
                               section_size=kwargs['section_sizes'][2], num_point=num_point,
                               fea_dim=c3, activation=nonlinear)
        else:
            self.tcn3 = MS_TCN(c3, c3, **kwargs, fea_dim=c3, activation=nonlinear)

        self.use_temporal_transformer = False

        self.to_use_final_fc = to_use_final_fc
        if self.to_use_final_fc:
            self.fc = nn.Linear(c3, num_class)

    def forward(self, x, set_to_fc_last=True):
        # Select channels
        x = x[:, :3, :, :]
        x = self.preprocessing(x)
        # assert 0
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        ###### First Component ######
        x = self.sgcn1_msgcn(x)
        x = self.sgcn1_ms_tcn_1(x)
        x = self.sgcn1_ms_tcn_2(x)
        x = self.nonlinear_f(x)
        x = self.tcn1(x)
        ###### End First Component ######

        ###### Second Component ######
        x = self.sgcn2_msgcn(x)
        x = self.sgcn2_ms_tcn_1(x)
        x = self.sgcn2_ms_tcn_2(x)
        x = self.nonlinear_f(x)
        x = self.tcn2(x)
        ###### End Second Component ######

        ###### Third Component ######
        x = self.sgcn3_msgcn(x)
        x = self.sgcn3_ms_tcn_1(x)
        x = self.sgcn3_ms_tcn_2(x)
        x = self.nonlinear_f(x)
        x = self.tcn3(x)
        ###### End Third Component ######

        out = x

        out_channels = out.size(1)

        t_dim = out.shape[2]
        out = out.view(N, M, out_channels, t_dim, -1)
        out = out.permute(0, 1, 3, 4, 2)  # N, M, T, V, C
        out = out.mean(3)  # Global Average Pooling (Spatial)

        out = out.mean(2)  # Global Average Pooling (Temporal)
        out = out.mean(1)  # Average pool number of bodies in the sequence

        if set_to_fc_last:
            if self.to_use_final_fc:
                out = self.fc(out)

        other_outs = {}
        return out, other_outs

    def preprocessing(self, x):
        # Extract Bone and Angular Features
        fp_sp_joint_list_bone = []
        fp_sp_joint_list_bone_angle = []
        fp_sp_joint_list_body_center_angle_1 = []
        fp_sp_joint_list_body_center_angle_2 = []
        fp_sp_left_hand_angle = []
        fp_sp_right_hand_angle = []
        fp_sp_two_hand_angle = []
        fp_sp_two_elbow_angle = []
        fp_sp_two_knee_angle = []
        fp_sp_two_feet_angle = []

        all_list = [
            fp_sp_joint_list_bone, fp_sp_joint_list_bone_angle, fp_sp_joint_list_body_center_angle_1,
            fp_sp_joint_list_body_center_angle_2, fp_sp_left_hand_angle, fp_sp_right_hand_angle,
            fp_sp_two_hand_angle, fp_sp_two_elbow_angle, fp_sp_two_knee_angle,
            fp_sp_two_feet_angle
        ]

        for a_key in ntu_bone_angle_pairs:
            a_angle_value = ntu_bone_angle_pairs[a_key]
            a_bone_value = ntu_bone_adj[a_key]
            the_joint = a_key - 1
            a_adj = a_bone_value - 1
            bone_diff = (x[:, :3, :, the_joint, :] -
                         x[:, :3, :, a_adj, :]).unsqueeze(3).cpu()
            fp_sp_joint_list_bone.append(bone_diff)

            # bone angles
            v1 = a_angle_value[0] - 1
            v2 = a_angle_value[1] - 1
            vec1 = x[:, :3, :, v1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, v2, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_bone_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # body angles 1
            vec1 = x[:, :3, :, 2 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 21 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_body_center_angle_1.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # body angles 2
            vec1 = x[:, :3, :, the_joint, :] - x[:, :3, :, 21 - 1, :]
            vec2 = x[:, :3, :, 2 - 1, :] - x[:, :3, :, 21 - 1, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_joint_list_body_center_angle_2.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # left hand angle
            vec1 = x[:, :3, :, 24 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 25 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_left_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # right hand angle
            vec1 = x[:, :3, :, 22 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 23 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_right_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # two hand angle
            vec1 = x[:, :3, :, 24 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 22 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_hand_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # two elbow angle
            vec1 = x[:, :3, :, 10 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 6 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_elbow_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # two knee angle
            vec1 = x[:, :3, :, 18 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 14 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_knee_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

            # two feet angle
            vec1 = x[:, :3, :, 20 - 1, :] - x[:, :3, :, the_joint, :]
            vec2 = x[:, :3, :, 16 - 1, :] - x[:, :3, :, the_joint, :]
            angular_feature = (1.0 - self.cos(vec1, vec2))
            angular_feature[angular_feature != angular_feature] = 0
            fp_sp_two_feet_angle.append(angular_feature.unsqueeze(2).unsqueeze(1).cpu())

        for a_list_id in range(len(all_list)):
            all_list[a_list_id] = torch.cat(all_list[a_list_id], dim=3)

        all_list = torch.cat(all_list, dim=1)
        # print('All_list:', all_list.shape)

        features = torch.cat((x, all_list.cuda()), dim=1)
        # print('features:', features.shape)
        return features


if __name__ == "__main__":
    # For debugging purposes
    import sys

    sys.path.append('..')

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 25, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))
