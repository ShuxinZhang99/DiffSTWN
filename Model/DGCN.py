import numpy as np
import torch
import torch.nn as nn

from dgcn_cell import *


class DGCN(nn.Module):
    def __init__(self, num_units, o_adj_mx, d_adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh', filter_type='laplacian'):
        super().__init__()
        self.num_gcn_layers = 1
        self.dgcn_layers = nn.ModuleList(
            [DGCNCell(num_units, o_adj_mx, d_adj_mx, max_diffusion_step, num_nodes, nonlinearity, filter_type)
             for _ in range(self.num_gcn_layers)]
        )

    def forward(self, input):
        # inputs.shape: [batch_size, num_nodes, time_step]
        batch, num_nodes, time_step = input.size()

        output = input
        for layer_num, dgcn_layer in enumerate(self.dgcn_layers):
            output = dgcn_layer(output)

        return output