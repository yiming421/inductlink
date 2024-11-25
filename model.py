import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul
from torch_sparse.matmul import spmm_add
from util import adjoverlap

class LinearPredictor(nn.Module):
    def __init__(self, in_channels):
        super(LinearPredictor, self).__init__()
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, x_i, x_j):
        return self.linear(x_i * x_j).squeeze(-1)

class DotPredictor(nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def forward(self, x_i, x_j):
        return torch.sum(x_i * x_j, dim=-1)

class CosinePredictor(nn.Module):
    def __init__(self):
        super(CosinePredictor, self).__init__()

    def forward(self, x_i, x_j):
        x = torch.nn.functional.cosine_similarity(x_i, x_j, dim=-1)
        return x

class ManhattanPredictor(nn.Module):
    def __init__(self):
        super(ManhattanPredictor, self).__init__()

    def forward(self, x_i, x_j):
        x = torch.sum(torch.abs(x_i - x_j), dim=-1)
        return x

class LorentzPredictor(nn.Module):
    def __init__(self):
        super(LorentzPredictor, self).__init__()

    def forward(self, x_i, x_j):
        n = x_i.size(-1)
        x = torch.sum(x_i[:, 0:n//2] * x_j[:, 0:n//2], dim=-1) - torch.sum(x_i[:, n//2:] * x_j[:, n//2:], dim=-1)
        return x

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=3, res=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            if self.res:
                x = x + lin(x)
            else:
                x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()
    
class NCNPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta = 1.0,
                 res = 0.0):
        super().__init__()

        self.beta = beta

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 nn.LayerNorm(hidden_channels),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels))

    def multidomainforward(self,
                           x,
                           adj_t,
                           tar_ei):
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj_t, adj_t, tar_ei)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj_t, tar_ei):
        return self.multidomainforward(x, adj_t, tar_ei)

class inductive_GCN(MessagePassing):
    def __init__(self, layers, alpha, in_channels, out_channels, linear=False, dropout=0.2):
        super(inductive_GCN, self).__init__(aggr='add')
        self.layers = layers
        self.alpha = alpha
        self.alphas = Parameter(torch.Tensor(alpha ** np.arange(layers + 1)), requires_grad=True)
        self.linear = nn.Linear(in_channels, out_channels) if linear else nn.Identity()
        self.dropout = dropout

    def forward(self, x, adj_t, edge_weight=None):
        adj_t = gcn_norm(adj_t, edge_weight, adj_t.size(0), dtype=float)
        res = x * self.alphas[0]
        for i in range(self.layers):
            x = self.propagate(adj_t, x=x, edge_weight=edge_weight, size=None)
            res = x * self.alphas[i + 1] + res
        res = self.linear(res)
        return res
    
    def message_and_aggregate(self, adj_t, x, edge_weight):
        return matmul(adj_t, x, reduce='add')
    
class inductive_GCN_feat(nn.Module):
    def __init__(self, layers, input, hidden, dropout=0, relu=False, linear=False, conv='gcn'):
        super(inductive_GCN_feat, self).__init__()
        self.layers = layers
        if conv == 'gcn':
            self.conv1 = GCNConv(input, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif conv == 'sage':
            self.conv1 = SAGEConv(input, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        else:
            raise ValueError('Unknown convolution')
        self.layers = layers
        self.relu = relu
        self.dropout = dropout
        self.linear = linear
        self.norm = nn.LayerNorm(hidden)
        if linear:
            self.lin = nn.Linear(hidden, hidden)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        for i in range(self.layers - 1):
            if self.relu:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            if self.linear:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
                x = self.lin(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            x = self.conv2(x, adj_t)
        return x

class inductive_GCN_no_feat(nn.Module):
    def __init__(self, layers, hidden, residual=0, dropout=0, relu=False, linear=False, conv='gcn'):
        super(inductive_GCN_no_feat, self).__init__()
        self.layers = layers
        if conv == 'gcn':
            self.conv1 = GCNConv(hidden, hidden)
            self.conv2 = GCNConv(hidden, hidden)
        elif conv == 'sage':
            self.conv1 = SAGEConv(hidden, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
        else:
            raise ValueError('Unknown convolution')
        self.layers = layers
        self.relu = relu
        self.dropout = dropout
        self.linear = linear
        if linear:
            self.lin = nn.Linear(hidden, hidden)
        self.residual = residual

    def forward(self, x, adj_t):
        ori = x
        x = self.conv1(x, adj_t) + self.residual * ori
        for i in range(self.layers - 1):
            if self.relu:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            if self.linear:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
                x = self.lin(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(x)
            x = self.conv2(x, adj_t) + self.residual * ori
        return x

class inductive_GCN_light(nn.Module):
    def __init__(self, layers, hidden, alpha = 0.5, dropout=0, relu=False):
        super(inductive_GCN_light, self).__init__()
        self.layers = layers
        self.convs = [GCNConv(hidden, hidden) for _ in range(layers)]
        self.dropout = dropout
        self.relu = relu
        self.alphas = Parameter(torch.Tensor(alpha ** np.arange(layers + 1)), requires_grad=True)

    def forward(self, x, adj_t):
        res = x * self.alphas[0]
        for i in range(self.layers):
            x = self.convs[i](x, adj_t)
            res = x * self.alphas[i + 1] + res
        return res