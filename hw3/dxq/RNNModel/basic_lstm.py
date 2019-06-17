# coding=utf-8

import torch
import torch.nn as nn

class LSTMCell(nn.Module):

    def __init__(self, num_hidden_in, num_hidden, tln=True):
        """Initialize the basic LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            num_hidden: number of units in output tensor.
            tln: whether to apply tensor layer normalization
        """
        super(LSTMCell, self).__init__()
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.layer_norm = tln
        self._forget_bias = 1.0

        self.x_fc = nn.Linear(self.num_hidden_in, self.num_hidden * 4)
        self.h_fc = nn.Linear(self.num_hidden, self.num_hidden * 4)

        self.cell_fc = nn.Linear(self.num_hidden * 2, self.num_hidden)

        self.layer_norm_t = nn.LayerNorm(num_hidden * 4)
        self.layer_norm_s = nn.LayerNorm(num_hidden * 4)
        self.layer_norm_x = nn.LayerNorm(num_hidden * 4)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, hidden):
        h, c = hidden

        # print(x.shape, h.shape)
        h_state = self.h_fc(h)
        x_state = self.x_fc(x)

        if self.layer_norm:
            h_state = self.layer_norm_t(h_state)
            x_state = self.layer_norm_x(x_state)

        # print(s_cc.size(), t_cc.size(), x_cc.size())
        chunk_size = x_state.size(1) // 4
        ix, fx, ox, cx = torch.split(x_state, chunk_size, 1)
        ih, fh, oh, ch = torch.split(h_state, chunk_size, 1)

        i = self.sigmoid(ix + ih)
        f = self.sigmoid(fx + fh)
        o = self.sigmoid(ox + oh)
        u = self.tanh(cx + ch)

        new_c = i * u + f * c
        new_h = o * self.tanh(new_c)

        return new_h, new_h, new_c


class BasicLstm(nn.Module):

    def __init__(self, ninput, nhid, nlayers):
        super(BasicLstm, self).__init__()

        layers = []
        layers.append(LSTMCell(ninput, nhid, True))

        for i in range(1, nlayers):
            layers.append(LSTMCell(nhid, nhid, False))

        self.layers = nn.ModuleList(layers)

    # input: size
    def forward(self, input, hidden):

        outputs = []

        # for each time step
        for x in input:
            new_hidden = []
            output = x
            for l, layer in enumerate(self.layers):
                output,  h, c = layer(output, hidden[l])
                new_hidden.append((h, c))
            outputs.append(output)
        outputs = torch.stack(outputs, 0)
        return outputs, new_hidden



