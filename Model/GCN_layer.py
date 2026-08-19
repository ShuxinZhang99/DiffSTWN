import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, num_nodes, bias=True, addaptadj=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 256), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(256, num_nodes), requires_grad=True)
        self.addaptadj = addaptadj
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).type(torch.float32))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).type(torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        if self.addaptadj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            adj = adj + adp
        support = torch.matmul(x, self.weight.type(torch.float32))
        output = torch.bmm(adj.unsqueeze(0).expand(support.size(0), *adj.size()), support)
        if self.bias is not None:
            return output + self.bias.type(torch.float32)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '('+ str(self.in_features) + ' -> ' + str(self.out_features) + ')'
