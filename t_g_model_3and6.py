# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 04:10:20 2020

@author: lei
"""
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer
import torch.nn.functional as F

from torch.nn import Parameter
import math
from utils import gen_A, gen_adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NetworkModel(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, adj_file, n_layers=6, n_head=8, d_k=16, d_v=16,
            d_model=49, d_inner=128, dropout=0.5, n_position=200):

        super().__init__()
################################ depthwise_conv+transformer_encoder: shape特征通道 ######################################################        
        self.depth_wise_conv = nn.Conv2d(in_channels = 30, out_channels = 30, kernel_size = 3, stride =1, padding = 1, dilation = 1, groups = 30)
        #self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        self.linear1 = nn.Linear(490, 320)
        self.linear2 = nn.Linear(980, 320)
        self.linear3 = nn.Linear(320, 160)
        self.linear4 = nn.Linear(320, 160)
        self.linear5 = nn.Linear(9,3)
        
        self.bn1 = nn.BatchNorm2d(30)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
################################################### gcn:au特征通道 ################################################################
        self.embedding = nn.Embedding(9, 40)
        self.gc1_1 = GraphConvolution(40, 80)#in_channel 节点维度
        self.gc1_2 = GraphConvolution(80, 160)
        self.grelu1_1 = nn.LeakyReLU(0.2)
        self.grelu1_2 = nn.LeakyReLU(0.2)
          
        _adj = gen_A(9, 0.1, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        self.layer_norm1 = nn.LayerNorm(9, eps=1e-6)
        
        self.init_weights()
        
    def init_weights(self):
        self.embedding.weight.data.normal_(0, 0.01)    
        
    def forward(self, src_seq, src_mask = None, return_attns=False):
################################################# shape特征通道 ######################################################################
        src_seq = self.depth_wise_conv(src_seq)
        src_seq = self.relu1(self.bn1(src_seq))
        src_seq = src_seq.view(src_seq.size()[0], src_seq.size()[1], -1)
        
        eyebrow_src_seq = src_seq[:,0:10,:]
        mouth_src_seq = src_seq[:,10:30,:]        

        for eyebrow_enc_layer in self.layer_stack:
            eyebrow_enc_output, not_use1 = eyebrow_enc_layer(eyebrow_src_seq, slf_attn_mask=src_mask)#64X10X49

        for mouth_enc_layer in self.layer_stack1:
            mouth_enc_output, not_use2 = mouth_enc_layer(mouth_src_seq, slf_attn_mask=src_mask)#64X20X49

        eyebrow_enc_output = eyebrow_enc_output.view(eyebrow_enc_output.size()[0], -1)#64X490
        mouth_enc_output = mouth_enc_output.view(mouth_enc_output.size()[0], -1)#64X980
                
        o_eye = self.dropout1(self.relu2(self.linear1(eyebrow_enc_output)))#490到120 64x480
        o_eye = self.relu4(self.linear3(o_eye))
        o_mou = self.dropout2(self.relu3(self.linear2(mouth_enc_output)))#980到240 64x960
        o_mou = self.relu5(self.linear4(o_mou))
        #o = torch.cat([o_eye, o_mou], dim=1)#64x360       
#################################################### au特征通道 ####################################################################################
        adj = gen_adj(self.A).detach()

        a = torch.LongTensor([0,1,2,3,4,5,6,7,8]).cuda()#前三个au与眉毛有关，后6个与嘴有关，比例1:2

        a1 = self.embedding(a)
      
        x1 = self.gc1_1(a1, adj)
        x1 = self.grelu1_1(x1)
 
        x1 = self.gc1_2(x1, adj)
        x1 = self.grelu1_2(x1)

        x = x1.transpose(0, 1)#160X9
        part1, part2 = x.split([3,6],dim=1)
        part1 = torch.matmul(o_eye, part1)
        part2 = torch.matmul(o_mou, part2)
        x = torch.cat([part1, part2], dim = 1)
        x = self.layer_norm1(x)
        x = self.linear5(x)

        return F.log_softmax(x, dim=1)