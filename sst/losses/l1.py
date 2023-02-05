import torch
import torch.nn as nn


class L1Ratio(nn.Module):
    '''L1 loss between ratio of predictions and timestretching rates'''
    def __init__(self, reduction='mean'):
        super(L1Ratio, self).__init__()
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, z_i, z_j, ts_rates_i, ts_rates_j):
        # ts_ratio = torch.ones_like(z_i) * ts_rate_i / ts_rate_j
        return self.l1(z_i/z_j, ts_rates_i/ts_rates_j)


class L1Diff(nn.Module):
    '''L1 loss between z_i and z_j scaled by ratio of timestretching rates'''
    def __init__(self, reduction='mean'):
        super(L1Diff, self).__init__()
        self.l1 = nn.L1Loss(reduction=reduction)

    def forward(self, z_i, z_j, ts_rate_i, ts_rate_j):
        return self.l1(z_i, z_j*ts_rate_i/ts_rate_j)
