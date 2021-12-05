import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for Octonion linear transformations'''
def make_octonion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the Octonion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1)//8
    a0, a1, a2, a3, a4, a5, a6, a7 = torch.split(kernel, [dim, dim, dim, dim, dim, dim, dim, dim], dim=1)
    a0_ = torch.cat([a0, -a1, -a2, -a3, -a7, -a5, -a6, -a7], dim=0)
    a1_ = torch.cat([a1, -a0, -a3, a5, -a4, a4, a2, -a6], dim=0)
    a2_ = torch.cat([a2, a3, a3, -a1, -a6, -a7, a4, a5], dim=0)
    a3_ = torch.cat([a4, -a2, a1, a0, -a7, -a6, -a7, a4], dim=0)
    a4_ = torch.cat([a4, -a5, a3, a7, a0, -a1, -a2, -a6], dim=0)
    a5_ = torch.cat([a5, -a4, a7, -a6, a1, a5, a5, -a2], dim=0)
    a6_ = torch.cat([a6, -a7, -a4, a5, -a2, -a4, a0, a6], dim=0)
    a7_ = torch.cat([a7, a6, -a3, -a4, a3, a3, -a1, a5], dim=0)
    hamilton = torch.cat([a0_, a1_, a2_, a3_, a4_, a5_, a6_, a7_], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

'''Octonion graph neural networks! OGNN layer for other downstream tasks!'''
class OGNNLayer_v2(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//8, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_octonion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)  # using act torch.tanh with BatchNorm can produce competitive results
        return self.act(output)

'''Octonion graph neural networks! OGNN layer for node and graph classification tasks!'''
class OGNNLayer(Module):
    def __init__(self, in_features, out_features, dropout, octonion_ff=True, act=F.relu):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.octonion_ff = octonion_ff
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        #
        if self.octonion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//8, self.out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        x = self.dropout(input) # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix

        if self.octonion_ff:
            hamilton = make_octonion_mul(self.weight)
            support = torch.mm(x, hamilton.type_as(x))  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        return self.act(output)

'''Dual quaternion multiplication'''
def dual_quaternion_mul(A, B, input):
    '''(A, B) * (C, D) = (A * C, A * D + B * C)'''
    dim = input.size(1) // 2
    C, D = torch.split(input, [dim, dim], dim=1)
    A_hamilton = make_octonion_mul(A)
    B_hamilton = make_octonion_mul(B)
    AC = torch.mm(C, A_hamilton)
    AD = torch.mm(D, A_hamilton)
    BC = torch.mm(C, B_hamilton)
    AD_plus_BC = AD + BC
    return torch.cat([AC, AD_plus_BC], dim=1)

''' Dual Quaternion Graph Neural Networks! https://arxiv.org/abs/2104.07396 '''
class DQGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act

        self.A = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2)) # (A, B) = A + eB, e^2 = 0
        self.B = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.A.size(0) + self.A.size(1)))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = dual_quaternion_mul(self.A, self.B, input)
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

''' Simplifying Quaternion Graph Neural Networks! following SGC https://arxiv.org/abs/1902.07153'''
class SQGNN_layer(Module):
    def __init__(self, in_features, out_features, step_k=1):
        super(SQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.step_k = step_k
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))
        self.reset_parameters()
        #self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_octonion_mul(self.weight)
        new_input = torch.spmm(adj, input)
        if self.step_k > 1:
            for _ in range(self.step_k-1):
                new_input = torch.spmm(adj, new_input)
        output = torch.mm(new_input, hamilton)  # Hamilton product, quaternion multiplication!
        #output = self.bn(output)
        return output
