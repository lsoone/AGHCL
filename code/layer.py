import math
import torch
import torch.nn as nn
from torch.cuda import device
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import torch.autograd
from torch.nn.modules.module import Module
from torch_geometric.nn import GATConv, GCNConv
import torch_geometric.utils as utils
import scipy.sparse as sp

def similarity_to_graph_adjacency(similarity_matrix, k):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        similarities = similarity_matrix[i]
        nearest_neighbors_indices = np.argsort(similarities)[-k:]
        for j in nearest_neighbors_indices:
            adjacency_matrix[i, j] = 1
            adjacency_matrix[j, i] = 1
    return adjacency_matrix


class CircRNAFeatureExtractor(nn.Module):
    def __init__(self, input_size, projection_layer_sizes, dropout_prob=0.5):
        super(CircRNAFeatureExtractor, self).__init__()


        self.projection_layers = nn.ModuleList()
        self.input_size = input_size


        for output_size in projection_layer_sizes:
            self.projection_layers.append(nn.Linear(self.input_size, output_size))
            self.projection_layers.append(nn.ReLU())
            self.projection_layers.append(nn.Dropout(p=dropout_prob))
            self.input_size = output_size

        self.projection_layers = nn.Sequential(*self.projection_layers)

    def forward(self, circrna_matrix):
        circrna_matrix = circrna_matrix.to(torch.float)

        for layer in self.projection_layers:
            if isinstance(layer, nn.Linear):
                layer.weight = nn.Parameter(layer.weight.to(torch.float))
                layer.bias = nn.Parameter(layer.bias.to(torch.float))

        projected_features = self.projection_layers(circrna_matrix)
        return projected_features

class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act

        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_edge) + ' -> ' + str(self.num_hidden)


class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.double))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.double))

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + 2*self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_node) + ' -> ' + str(self.num_hidden)


class decoder2(nn.Module):
    def __init__(self, dropout=0.8, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)

        z = self.act(z_node_.mm(z_hyperedge_.t()))
        return z

class decoder1(nn.Module):
    def __init__(self, dropout=0.5, act=torch.sigmoid):
        super(decoder1, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_node, z_hyperedge):
        z_node_ = z_node
        z_hyperedge_ = z_hyperedge
        z = self.act(z_node_.mm(z_hyperedge_.t()))

        return z



class HGNN1(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, n_hid_2=128,dropout=0.5):
        super(HGNN1, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv1(in_ch, n_hid)
        self.hgc2 = HGNN_conv1(n_hid, n_class)



    def forward(self, x, G):
        G = G + torch.eye(G.shape[0]).cuda()
        x = self.hgc1(x, G)
        x = torch.tanh(x)
        x = self.hgc2(x, G)
        x = torch.tanh(x)

        return x

class HGNN_conv1(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):  #
        super(HGNN_conv1, self).__init__()

        self.weight = Parameter(torch.DoubleTensor(in_ft, out_ft))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = Parameter(torch.DoubleTensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.linear_x_1 = nn.Linear(in_ft, out_ft).to(torch.double)
        self.reset_parameters()




    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):  # x: torch.Tensor, G: torch.Tensor

        # part1
        x = x.double()
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x) + x


        return x

class MultiGraphConvolution_Layer(nn.Module):

    def __init__(self, in_features, out_features):
        super(MultiGraphConvolution_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #self.view_conv1 = GATConv(in_features, out_features, heads=3, dropout=0.5, concat=False)
        #self.view_conv2 = GATConv(out_features, out_features, heads=3, dropout=0.5, concat=False)
        self.view_conv1 = GATConv(in_features, out_features)
        self.view_conv2 = GATConv(out_features, out_features)



    def forward(self, input_x, adj):
        # 确保 input_x 是 tensor，如果是 numpy 数组则转换
        if isinstance(input_x, np.ndarray):
            input_x = torch.from_numpy(input_x).float().to(self.device)  # 转换为 tensor 并移动到设备
        else:
            input_x = input_x.to(self.device)  # 如果已经是 tensor，直接移动到设备

        # 确保 adj 是 tensor，如果是 numpy 数组则转换
        if isinstance(adj, np.ndarray):
            adj_temp = torch.from_numpy(adj).to(self.device)  # 转换为 tensor 并移动到设备
        else:
            adj_temp = adj.to(self.device)  # 如果已经是 tensor，直接移动到设备


        # 将 tensor 转回 numpy 进行 COO 转换
        adj_temp = sp.coo_matrix(adj_temp.cpu().numpy())
        edge_index, edge_weight = utils.from_scipy_sparse_matrix(adj_temp)

        # 确保 edge_index 和 edge_weight 在正确的设备上
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)

        # 执行卷积操作
        input_x_view_conv1 = F.relu(self.view_conv1(input_x, edge_index, edge_weight))
        input_x_view_conv2 = F.relu(self.view_conv2(input_x_view_conv1, edge_index, edge_weight))



        return input_x_view_conv2

class MultiGraphConvolution_Layer1(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultiGraphConvolution_Layer1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用默认的float32类型初始化GCN层
        self.view_conv1 = GCNConv(in_features, out_features)
        self.view_conv2 = GCNConv(out_features, out_features)

    def forward(self, input_x, adj):
        # 确保input_x是float32类型
        if isinstance(input_x, np.ndarray):
            input_x = torch.from_numpy(input_x).float().to(self.device)  # 明确转换为float32
        else:
            input_x = input_x.to(self.device).float()  # 强制类型转换

        # 确保邻接矩阵是float32类型
        if isinstance(adj, np.ndarray):
            adj_temp = torch.from_numpy(adj).float().to(self.device)  # 添加.float()确保类型
        else:
            adj_temp = adj.to(self.device).float()  # 强制类型转换

        # 转换为COO格式的稀疏矩阵
        adj_temp = sp.coo_matrix(adj_temp.cpu().numpy())
        edge_index, edge_weight = utils.from_scipy_sparse_matrix(adj_temp)

        # 确保edge_weight是float32类型
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device).float()  # 关键修改：强制转换为float32

        # 执行卷积操作
        input_x_view_conv1 = F.relu(self.view_conv1(input_x, edge_index, edge_weight))
        input_x_view_conv2 = F.relu(self.view_conv2(input_x_view_conv1, edge_index, edge_weight))

        return input_x_view_conv2






