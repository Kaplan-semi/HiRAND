import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
# from sparsegnc_layer import *



class SparseGCN(Module):

    def __init__(self, in_features, out_features, bias=False):
        super(SparseGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, in_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        ##kaiming_uniform
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inpt, gene_adj):
        output = torch.mm(inpt, (self.weight * gene_adj))
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'







class RandPropaGCN(nn.Module):

    def __init__(
        self, drop_rate, order=1, param=False, bias=True,
         #in_features=None, out_features=None
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.order = order
        self.param = param
        self.weight = Parameter(torch.FloatTensor(10696,10696))#1603 11301 10696 10984
        self.reset_parameters()
       
    def reset_parameters(self):
         #kaiming_uniform
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
     
    def forward(self, features, A):
        n = features.shape[0]
        if self.training:
            drop_rates = torch.FloatTensor(np.ones(n) * self.drop_rate)
            masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(features)
            feats = masks * features
        else:
            feats = features * (1. - self.drop_rate)
        x1 = y1 = feats
        # x1 = y1 =torch.mm(feats,self.weight)
        for i in range(self.order):
            x1 = torch.mm(A, x1)
            y1.add_(x1)
        return y1.div(self.order+1.0)


class ShrinkMLP(nn.Module):

    def __init__(
        self, nfeat, nhid, nclass, input_droprate, hidden_droprate,
        use_bn=False
    ):
        super().__init__()

        # self.layer1 = nn.Linear(nfeat, 256)
        self.layer1 = nn.Linear(nfeat, 128)
        self.layer2 = nn.Linear(128, nhid)
        self.layer3 = nn.Linear(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(128)
        self.use_bn = use_bn
        
    def forward(self, x):
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = torch.tanh(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x



class fwgcn(Module):
    """
    feature-weighted gcn layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(fwgcn, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform
        stdv = 1. / math.sqrt(self.a.size(0))
        self.a.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        a = torch.softmax(self.a, dim=0) #softmax/attention
        output = input * self.a
        # output = torch.mm(input, self.weight) #orignal GCN
        output = torch.mm(adj, output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class HiRAND(nn.Module):
    def __init__(
        self, nfeat, nhid, nclass, input_droprate, hidden_droprate,
        use_bn, drop_rate, order, K, logsoftmax=True
    ):
        super(HiRAND, self).__init__()
        self.K = K
        self.logsoftmax = logsoftmax
        self.sgcn = SparseGCN(nfeat, nfeat)
        self.randpgcn = RandPropaGCN(drop_rate, order)
        self.fgcn1 = fwgcn(nfeat,nfeat)
        self.fgcn = ShrinkMLP(
            nfeat, nhid,nclass, input_droprate, hidden_droprate, use_bn
        )


    def forward(self, inpt, gene_adj, A):
        x = self.sgcn(inpt, gene_adj)
        x = torch.tanh(x)
        if self.training:
            output_list = []
            for _ in range(self.K):
                x = self.randpgcn(x, A)
                xi = torch.tanh(x)
                xi = self.fgcn1(xi,A)
                xi = F.relu(xi)
                xi = self.fgcn(xi)
                xi = torch.tanh(xi)
                if self.logsoftmax:
                    xi = torch.log_softmax(xi, dim=-1)
                output_list.append(xi)
            return output_list
        else:
            x = self.randpgcn(x, A)
            x =  torch.tanh(x)
            x = self.fgcn1(x,A)
            x = F.relu(x)
            x = self.fgcn(x)
            x = torch.tanh(x)
            if self.logsoftmax:
                x = torch.log_softmax(x, dim=-1)
            return x
