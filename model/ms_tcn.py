import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn

from model.activation import activation_factory


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1,
                 activation='relu',
                 **kwargs):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                activation_factory(activation),
                # 在时间轴上对每一个joint做卷积
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation),
            )
            for dilation in dilations
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            activation_factory(activation),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.act = activation_factory(activation)

        # Transformer attention
        if 'to_use_temporal_transformer' in kwargs and kwargs['to_use_temporal_transformer']:
            self.to_use_temporal_trans = True
            self.section_size = kwargs['section_size']
            self.num_point = kwargs['num_point']
            self.trans_conv = nn.Conv2d(1, 1, (self.section_size, 1), (self.section_size, 1))
            nhead = 5
            nlayers = 2
            trans_dropout = 0.5
            encoder_layers = nn.TransformerEncoderLayer(self.num_point,
                                                        nhead=nhead, dropout=trans_dropout)
            self.trans_enc = nn.TransformerEncoder(encoder_layers, nlayers)

            # frame normalization
            self.frame_norm_layer = nn.Softmax(dim=1)
            if 'frame_norm' in kwargs:
                if kwargs['frame_norm'] == 'sigmoid':
                    self.frame_norm_layer = nn.Sigmoid()

        else:
            self.to_use_temporal_trans = False

        # Transformer feature
        if 'to_use_temp_trans_feature' in kwargs and kwargs['to_use_temp_trans_feature']:
            self.to_use_temp_trans_feature = True
            self.fea_dim = kwargs['fea_dim']
            nhead = kwargs['temp_trans_feature_n_head']
            nlayers = kwargs['temp_trans_feature_n_layer']
            trans_dropout = 0.5
            encoder_layers = nn.TransformerEncoderLayer(self.fea_dim,
                                                        nhead=nhead, dropout=trans_dropout)
            self.trans_enc_fea = nn.TransformerEncoder(encoder_layers, nlayers)
        else:
            self.to_use_temp_trans_feature = False

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        tempconv_idx = 0
        for tempconv in self.branches:
            x_in = x
            if self.to_use_temporal_trans:
                x_mean = torch.mean(x, dim=1)
                x_mean = x_mean.unsqueeze(1)
                x_mean = self.trans_conv(x_mean).squeeze(1)
                x_mean = self.trans_enc(x_mean)
                x_mean = self.frame_norm_layer(x_mean)
                x_mean = torch.repeat_interleave(x_mean, self.section_size, dim=1)
                x_mean = torch.unsqueeze(x_mean, dim=1).repeat(1, x.shape[1], 1, 1)
                x_in = x * x_mean
            out = tempconv(x_in)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)

        if self.to_use_temp_trans_feature:
            out = out.permute(3, 2, 0, 1)
            for a_out_idx in range(len(out)):
                a_out = out[a_out_idx]
                a_out = self.trans_enc_fea(a_out)
                out[a_out_idx] = a_out
            out = out.permute(2, 3, 1, 0)

        out += res
        out = self.act(out)

        return out


if __name__ == "__main__":
    mstcn = MultiScale_TemporalConv(288, 288)
    x = torch.randn(32, 288, 100, 20)
    mstcn.forward(x)
    for name, param in mstcn.named_parameters():
        print(f'{name}: {param.numel()}')
    print(sum(p.numel() for p in mstcn.parameters() if p.requires_grad))