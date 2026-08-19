import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as n
from torch.autograd import Variable
import sys
import math
from scipy.stats import nbinom
from torch.nn.utils import weight_norm


class NBNorm(nn.Module):
    def __init__(self, c_in, c_out):
        super(NBNorm, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

        self.p_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)
        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        n = self.n_conv(x).squeeze_(-1)
        p = self.p_conv(x).squeeze_(-1)

        # Reshape
        n = n.view([B, self.out_dim, N])
        p = p.view([B, self.out_dim, N])

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n)  # Some parameters can be tuned here
        p = F.sigmoid(p)
        return n.permute([0, 2, 1]), p.permute([0, 2, 1])

    def likelihood_loss(self, y, n, p, y_mask=None):
        """
        y: true values
        y_mask: whether missing mask is given
        """
        nll = torch.lgamma(n) + torch.lgamma(y + 1) - torch.lgamma(n + y) - n * torch.log(p) - y * torch.log(1 - p)
        if y_mask is not None:
            nll = nll * y_mask
        return torch.sum(nll)

    def mean(self, n, p):
        """
        :param cat: Input data of shape (batch_size, num_timesteps, in_nodes)
        :return: Output data of shape (batch_size, 1, num_timesteps, in_nodes)
        """
        pass


# Define the Gaussian
class GaussNorm(nn.Module):
    def __init__(self, c_in, c_out):
        super(GaussNorm, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)

        self.p_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=c_out,
                                kernel_size=(1, 1),
                                bias=True)
        self.out_dim = c_out  # output horizon

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        (B, _, N, _) = x.shape  # B: batch_size; N: input nodes
        loc = self.n_conv(x).squeeze_(
            -1)  # The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
        scale = self.p_conv(x).squeeze_(-1)

        # Reshape
        loc = loc.view([B, self.out_dim, N])
        scale = scale.view([B, self.out_dim, N])

        # Ensure n is positive and p between 0 and 1
        loc = F.softplus(loc)  # Some parameters can be tuned here, count data are always positive
        scale = F.sigmoid(scale)
        return loc.permute([0, 2, 1]), scale.permute([0, 2, 1])


# Define the NB class first, not mixture version
class NBNorm_ZeroInflated(nn.Module):
    def __init__(self, c_in, c_out, his_time_step, fore_horizon, hidden):
        super(NBNorm_ZeroInflated, self).__init__()
        self.c_in = c_in
        self.hidden = hidden
        self.c_out = c_out
        self.his_time_step = his_time_step
        self.fore_horizon  = fore_horizon
        self.n_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=hidden,
                                kernel_size=3,
                                padding=1,
                                bias=True)

        self.n_conv_2 = nn.Conv2d(in_channels=hidden,
                                  out_channels=c_out,
                                  kernel_size=3,
                                  padding=1,
                                  bias=True)

        self.p_conv = nn.Conv2d(in_channels=c_in,
                                out_channels=hidden,
                                kernel_size=3,
                                padding=1,
                                bias=True)

        self.p_conv_2 = nn.Conv2d(in_channels=hidden,
                                  out_channels=c_out,
                                  kernel_size=3,
                                  padding=1,
                                  bias=True)

        self.pi_conv = nn.Conv2d(in_channels=c_in,
                                 out_channels=hidden,
                                 kernel_size=3,
                                 padding=1,
                                 bias=True)

        self.pi_conv_2 = nn.Conv2d(in_channels=hidden,
                                   out_channels=c_out,
                                   kernel_size=3,
                                   padding=1,
                                   bias=True)

        self.out_dim = c_out  # output horizon

        self.zinb_linear1 = nn.Linear(self.his_time_step, 128)
        self.zinb_relu    = nn.ReLU()
        self.zinb_linear2 = nn.Linear(128, self.fore_horizon)

    def forward(self, x):
        # x.shape: batch, residual_channel, timestep
        # x = x.permute(0, 2, 1)
        (B, _, V, T) = x.shape  # B: batch_size; N: timestep
        n = self.n_conv(x)  # (B, c_out, num_nodes, timestep)
        n = self.n_conv_2(n)
        n = n + x

        p = self.p_conv(x)
        p = self.p_conv_2(p)
        p = p + x

        pi = self.pi_conv(x)
        pi = self.pi_conv_2(pi)
        pi = pi + x

        # Reshape
        n = F.relu(self.zinb_linear1(n))
        n = self.zinb_linear2(n)

        p = F.relu(self.zinb_linear1(p))
        p = self.zinb_linear2(p)

        pi = F.relu(self.zinb_linear1(pi))
        pi = self.zinb_linear2(pi)

        # Ensure n is positive and p between 0 and 1
        n = F.softplus(n)  # Some parameters can be tuned here
        p = torch.sigmoid(p)
        pi = torch.sigmoid(pi)
        return n, p, pi
