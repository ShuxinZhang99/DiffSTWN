import numpy
import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from Utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape, device=device))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length, device=device))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]


class DGCNCell(nn.Module):
    def __init__(self, num_units, o_adj_mx, d_adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type='laplacian', bias=True):
        """
        :param num_units
        :param o_adj_mx:
        :param d_adj_mx:
        :param max_diffusion_step:
        :param nonlinearity:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk"
        """
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._time_lag = num_units
        self._max_diffusion_step = max_diffusion_step
        self._support = []
        supports = []
        if filter_type == 'laplacian':
            supports.append(calculate_normalized_laplacian(o_adj_mx))
        elif filter_type == 'random_walk':
            supports.append(calculate_random_walk_matrix(o_adj_mx).T)
        elif filter_type == 'dual_random_walk':
            supports.append(calculate_random_walk_matrix(o_adj_mx).T)
            supports.append(calculate_random_walk_matrix(d_adj_mx).T)
        else:
            supports.append(calculate_scaled_laplacian(o_adj_mx))
        for support in supports:
            self._support.append(self._build_sparse_matrix(support))
        # self._fc_params = LayerParams(self, 'fc')
        # self._gconv_params = LayerParams(self, 'gconv')
        self.num_matrices = len(self._support) * self._max_diffusion_step + 1

        self.weight = Parameter(
            torch.FloatTensor(self.num_matrices * self._time_lag, self._num_units).type(torch.float32))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self._num_units).type(torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L.double()

    def forward(self, inputs):
        """
        :param inputs: (B, num_nodes, time_step)
        """
        output_size = self._num_units

        batch_size = inputs.shape[0]
        input_size = inputs.size(-1)

        x0 = inputs.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._support:
                x1 = torch.sparse.mm(support, x0.double())
                x = torch.cat([x, x1.unsqueeze(0)], dim=0)

                for k in range(2, self._max_diffusion_step + 1):
                    # print(support.shape, x1.shape, x0.shape, x.shape)
                    x2 = 2 * torch.sparse.mm(support, x1.double()) - x0
                    x = torch.cat([x, x2.unsqueeze(0)], dim=0)
                    x1, x0 = x2, x1

        num_matrices = len(self._support) * self._max_diffusion_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        # weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x.double(), self.weight.double())  # (batch_size * self._num_nodes, output_size)

        # biases = self._gconv_params.get_biases(output_size, bias_start=0.0)
        x += self.bias.double()
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        output = torch.reshape(x, [batch_size, self._num_nodes, output_size])
        return output.float()
