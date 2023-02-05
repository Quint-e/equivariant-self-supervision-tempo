# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8
"""
import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F


EPS = 10e-8

class ResidualBlock(nn.Module):
    def __init__(
            self,
            n_channel_in, 
            dilation_rate, 
            activation, 
            num_filters, 
            kernel_size, 
            dropout_rate=0,
            include_conv1x1_res=True
            ):
        super(ResidualBlock, self).__init__()
        self.include_conv1x1_res = include_conv1x1_res # If set to false, no 1x1 conv is applied on the residual. (cf. forward method)
        self.n_channel_in = n_channel_in
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # self.padding = padding
        self.dropout_rate = dropout_rate
        self.padding = self._padding_same(
            self.n_channel_in,
            self.kernel_size,1,
            self.dilation_rate
            )
        # Layers
        self.conv = nn.Conv1d(
            self.n_channel_in,
            self.num_filters,
            self.kernel_size, 
            dilation=self.dilation_rate, 
            padding=self.padding
            )
        self.spatial_dropout = nn.Dropout2d(p=self.dropout_rate)
        self.conv1x1 = nn.Conv1d(
            self.num_filters,
            self.num_filters,1, 
            padding=0
            )
        self.conv1x1_res = nn.Conv1d(
            self.n_channel_in, 
            self.num_filters, 
            1, 
            padding=0
            )

    def _padding_same(self,i, k, s, d):
        '''Returns the padding value to be used to produce a 'same' padding, given other convolution parameters.
        cf. https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        args:
            i = input
            k = kernel size,
            s = stride,
            d = dilation
        returns:
            p = padding'''
        p = ( i*(s-1) + k - s + (k-1)*(d-1))//2
        return p

    def forward(self, x):
        original_x = x
        x = self.conv(x)
        x = self.activation(x)
        x = self.spatial_dropout(x)
        x = self.conv1x1(x)
        if self.include_conv1x1_res:
            res_x = self.conv1x1_res(original_x) + x
        else:
            res_x = original_x + x
        return res_x, x


class TCNBlock(nn.Module):
    def __init__(
            self, 
            n_channels_in, 
            num_filters=16, 
            kernel_size=5, 
            dilations=[1,2,4,8,16,32,64,128],
            activation=nn.ELU(),
            dropout_rate=0.1
            ):
        super(TCNBlock, self).__init__()
        self.n_channels_in = n_channels_in
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.activation = activation
        self.dropout_rate = dropout_rate

        module_dict = {}
        for i, d in enumerate(self.dilations):
            ch_in = self.n_channels_in if i==0 else self.num_filters
            module_dict["TCNBlock{}".format(i)] = ResidualBlock(
                ch_in,
                d,
                self.activation,
                self.num_filters,
                self.kernel_size,
                self.dropout_rate
                )
        self.model = nn.ModuleDict(module_dict)

    def forward(self,x):
        for i, d in enumerate(self.dilations):
            x, x_skip = self.model["TCNBlock{}".format(i)](x)
            if i==0:
                tempo_skip = x_skip
            else:
                tempo_skip += x_skip
        x = self.activation(x)
        return x, tempo_skip


class TempoBlock(nn.Module):
    def __init__(
            self, 
            in_features, 
            mode='classification', 
            num_units=16,  
            output_units=300, 
            dropout_rate=0.1,
            include_top=True, 
            add_proj_head=False, 
            proj_head_dim=16
            ):
        super(TempoBlock, self).__init__()
        self.in_features = in_features
        self.num_units = num_units
        self.output_units = output_units
        self.dropout_rate = dropout_rate
        self.mode = mode
        self.validate_mode()
        self.include_top = include_top
        # Projection head params
        self.add_proj_head = add_proj_head  # Boolean to include projection head or not.
        self.proj_head_dim = proj_head_dim  # Dimension of the projection head hidden layer

        self.globalAvgPool = nn.AdaptiveAvgPool1d(1) # Average pool the time dimension, for any input size.
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.dense = nn.Linear(self.in_features,self.num_units)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        if self.add_proj_head==True:
            self.proj_head = nn.Linear(self.num_units, self.proj_head_dim)
            # Output layers for either classification or regression.
            self.output = nn.Linear(self.proj_head_dim, self.output_units)
            self.output_reg = nn.Linear(self.proj_head_dim, 1)
        else:
            # Output layers for either classification or regression.
            self.output = nn.Linear(self.num_units, self.output_units)
            self.output_reg = nn.Linear(self.num_units, 1)

    def validate_mode(self):
        permitted_modes = ['classification','regression']
        if not self.mode in permitted_modes:
            raise ValueError("TCN mode must be one of {}".format(permitted_modes))

    def forward(self,x):
        x = self.globalAvgPool(x)
        x = torch.squeeze(x, 2)
        x = self.dropout(x)
        x = self.activation(self.dense(x))
        if self.add_proj_head:
            x = self.activation(self.proj_head(x))
        if self.include_top:
            if self.mode=='classification':
                x = self.output(x)
            elif self.mode=='regression':
                x = self.output_reg(x)
        return x

class TCN(nn.Module):
    '''Main module class'''
    def __init__(
            self, 
            tempo_range=(0,300), 
            mode='classification',
            num_filters=16, 
            dropout_rate=0.1, 
            num_dilations=10,
            add_proj_head=False, 
            proj_head_dim=16
            ):
        super(TCN, self).__init__()
        self.tempo_range = tempo_range
        self.mode = mode
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.dilations = [2 ** i for i in range(num_dilations)]
        # Projection head params
        self.add_proj_head = add_proj_head # Boolean to include projection head or not.
        self.proj_head_dim = proj_head_dim # Dimension of the projection head hidden layer

        # Pre-processing
        self.bn_input = nn.BatchNorm2d(num_features=1)

        # Front-end convolution blocks
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.conv1 = nn.Conv2d(1, self.num_filters, (3, 3))
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.conv2 = nn.Conv2d(self.num_filters, self.num_filters, (3, 3))
        self.maxpool2 = nn.MaxPool2d((3, 1))
        self.conv3 = nn.Conv2d(self.num_filters, self.num_filters, (8, 1))

        # TCNBlock
        self.tcn_block = TCNBlock(
            self.num_filters,
            self.num_filters,
            5,
            self.dilations,
            self.activation,
            self.dropout_rate
            )

        # Tempo Block
        self.tempo_block = TempoBlock(
            self.num_filters,
            mode=self.mode, 
            num_units=16,
            output_units=self.tempo_range[1]-self.tempo_range[0],
            dropout_rate=self.dropout_rate,
            add_proj_head=self.add_proj_head, 
            proj_head_dim=self.proj_head_dim
            )

    def forward(self, x):
        x = self.bn_input(x)

        # Conv block 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool1(x)
        x = self.dropout(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool2(x)
        x = self.dropout(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Eliminate the frequency dimension
        x = torch.squeeze(x,2)

        # TCNBlock
        x, tempo_skip = self.tcn_block(x)

        # Tempo block
        # x = self.tempo_block(tempo_skip) # use sum of skip connections as input to tempo block
        x = self.tempo_block(x) # add tempo block at the outpout of TCN.

        return x
