# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torch.nn.parameter import Parameter


class embedding(nn.Module):

    def __init__(self, vocab_size, num_units, zeros_pad=True, scale=True):

        super(embedding, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale
        self.lookup_table = Parameter(torch.Tensor(vocab_size, num_units))
        nn.init.xavier_normal(self.lookup_table.data)
        if self.zeros_pad:
            self.lookup_table.data[0, :].fill_(0)

    def forward(self, inputs):
        if self.zeros_pad:
            self.padding_idx = 0
        else:
            self.padding_idx = -1
        outputs = F.embedding(
            inputs, self.lookup_table, self.padding_idx, None, 2, False, False) 

        if self.scale:
            outputs = outputs * (self.num_units ** 0.5)

        return outputs


class layer_normalization(nn.Module):

    def __init__(self, features, epsilon=1e-8):

        super(layer_normalization, self).__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x, dim=-1):
        mean = x.mean(dim, keepdim=True)
        std = x.std(dim, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


class positional_encoding(nn.Module):

    def __init__(self, num_units, zeros_pad=True, scale=True):

        super(positional_encoding, self).__init__()
        self.num_units = num_units
        self.zeros_pad = zeros_pad
        self.scale = scale

    def forward(self, inputs):
        N, T = inputs.size()[0: 2]

        position_ind = Variable(torch.unsqueeze(torch.arange(0, T), 0).repeat(N, 1).long().cuda())
        position_enc = torch.Tensor([
            [pos / np.power(10000, 2. * i / self.num_units) for i in range(self.num_units)]
            for pos in range(T)]).cuda()

        position_enc[:, 0::2] = torch.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = torch.cos(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = Variable(position_enc)

        if self.zeros_pad:
            lookup_table = torch.cat((Variable(torch.zeros(1, self.num_units)),
                                     lookup_table[1:, :]), 0)
            padding_idx = 0
        else:
            padding_idx = -1

        outputs = F.embedding(
            position_ind, lookup_table, padding_idx, None, 2, False, False)   # copied from torch.nn.modules.sparse.py

        if self.scale:
            outputs = outputs * self.num_units ** 0.5

        return outputs

class nonlocal_attention(nn.Module):

    def __init__(self, num_units, num_heads=8, dropout_rate=0, causality=False):

        super(nonlocal_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Conv2d(self.num_units, self.num_units, 1), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Conv2d(self.num_units, self.num_units, 1), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Conv2d(self.num_units, self.num_units, 1), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

        self.normalization = layer_normalization(self.num_units)

    def reshape(self, input):
        N, T, W, H, C = input.size()
        output = input.permute(0, 1, 3, 4, 2)
        output = output.view(N, T*W*H, C)
        return output

    def reshape_back(self, input, shape):


        N, T, C, W, H = shape
        output = input.view((N, T, W, H, C))
        output = output.permute(0, 1, 4, 2, 3)
        return output

    def forward(self, queries, keys, values):

        input_shape= values.size()
        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        Q = self.reshape(Q)
        K = self.reshape(K)
        V = self.reshape(V)

        queries = self.reshape(queries)
        keys = self.reshape(keys)

        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        outputs = outputs / (K_.size()[-1] ** 0.5)

        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        padding = Variable(torch.ones(*outputs.size()).cuda() * (-2 ** 32 + 1))
        condition = key_masks.eq(0.).float()
        outputs = padding * condition + outputs * (1. - condition)

        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size()).cuda()  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)

            padding = Variable(torch.ones(*masks.size()).cuda() * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)

        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks

        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        outputs += queries

        outputs = self.normalization(outputs)  # (N, T_q, C)

        outputs = self.reshape_back(outputs, input_shape)
        return outputs


class multihead_attention(nn.Module):

    def __init__(self, num_units, num_heads=8, dropout_rate=0, causality=False):

        super(multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())

        self.output_dropout = nn.Dropout(p=self.dropout_rate)

        self.normalization = layer_normalization(self.num_units)

    def forward(self, queries, keys, values):

        Q = self.Q_proj(queries)  # (N, T_q, C)
        K = self.K_proj(keys)  # (N, T_q, C)
        V = self.V_proj(values)  # (N, T_q, C)

        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*N, T_q, C/h)

        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        outputs = outputs / (K_.size()[-1] ** 0.5)

        key_masks = torch.sign(torch.abs(torch.sum(keys, dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = torch.unsqueeze(key_masks, 1).repeat(1, queries.size()[1], 1)  # (h*N, T_q, T_k)

        padding = Variable(torch.ones(*outputs.size()).cuda() * (-2 ** 32 + 1))
        condition = key_masks.eq(0.).float()
        outputs = padding * condition + outputs * (1. - condition)

        if self.causality:
            diag_vals = torch.ones(*outputs[0, :, :].size()).cuda()  # (T_q, T_k)
            tril = torch.tril(diag_vals, diagonal=0)  # (T_q, T_k)
            masks = Variable(torch.unsqueeze(tril, 0).repeat(outputs.size()[0], 1, 1))  # (h*N, T_q, T_k)

            padding = Variable(torch.ones(*masks.size()).cuda() * (-2 ** 32 + 1))
            condition = masks.eq(0.).float()
            outputs = padding * condition + outputs * (1. - condition)

        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks

        outputs = self.output_dropout(outputs)  # (h*N, T_q, T_k)

        outputs = torch.bmm(outputs, V_)  # (h*N, T_q, C/h)

        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        outputs += queries

        outputs = self.normalization(outputs)  # (N, T_q, C)

        return outputs


class cvfeedforward(nn.Module):

    def __init__(self, in_channels, num_units=[2048, 512]):
 
        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units


        params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0],
                  'kernel_size': 1, 'stride': 1, 'bias': True}
        self.conv1 = nn.Sequential(nn.Conv2d(**params), nn.ReLU())
        params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1],
                  'kernel_size': 1, 'stride': 1, 'bias': True}
        self.conv2 = nn.Conv2d(**params)

        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        outputs += inputs

        outputs = self.normalization(outputs, dim=2)

        return outputs


class feedforward(nn.Module):

    def __init__(self, in_channels, num_units=[2048, 512]):

        super(feedforward, self).__init__()
        self.in_channels = in_channels
        self.num_units = num_units

        self.conv = False
        if self.conv:
            params = {'in_channels': self.in_channels, 'out_channels': self.num_units[0],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv1 = nn.Sequential(nn.Conv1d(**params), nn.ReLU())
            params = {'in_channels': self.num_units[0], 'out_channels': self.num_units[1],
                      'kernel_size': 1, 'stride': 1, 'bias': True}
            self.conv2 = nn.Conv1d(**params)
        else:
            self.conv1 = nn.Sequential(nn.Linear(self.in_channels, self.num_units[0]), nn.ReLU())
            self.conv2 = nn.Linear(self.num_units[0], self.num_units[1])
        self.normalization = layer_normalization(self.in_channels)

    def forward(self, inputs):
        if self.conv:
            inputs = inputs.permute(0, 2, 1)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)

        outputs += inputs

        if self.conv:
            outputs = self.normalization(outputs.permute(0, 2, 1))
        else:
            outputs = self.normalization(outputs)

        return outputs


class label_smoothing(nn.Module):

    def __init__(self, epsilon=0.1):
     
        super(label_smoothing, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs):
        K = inputs.size()[-1]
        return ((1 - self.epsilon) * inputs) + (self.epsilon / K)
