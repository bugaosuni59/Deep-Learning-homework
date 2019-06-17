import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StLSTM(nn.Module):

    def __init__(self, layer_name, num_hidden_in, num_hidden,
                 seq_shape, tln=True):
        """Initialize the basic Conv LSTM cell.
        Args:
            layer_name: layer names for different convlstm layers.
            filter_size: int tuple thats the height and width of the filter.
            num_hidden: number of units in output tensor.
            forget_bias: float, The bias added to forget gates (see above).
            tln: whether to apply tensor layer normalization
        """
        super(StLSTM, self).__init__()
        self.layer_name = layer_name
        self.num_hidden_in = num_hidden_in
        self.num_hidden = num_hidden
        self.batch = seq_shape[0]
        self.layer_norm = tln
        self._forget_bias = 1.0

        self.h_fc = nn.Linear(self.num_hidden, self.num_hidden * 4)
        self.m_fc = nn.Linear(self.num_hidden, self.num_hidden * 4)
        self.x_fc = nn.Linear(self.num_hidden, self.num_hidden * 4)
        self.cell_fc = nn.Linear(self.num_hidden * 2, self.num_hidden)

        self.layer_norm_t = nn.LayerNorm(num_hidden * 4)
        self.layer_norm_s = nn.LayerNorm(num_hidden * 4)
        self.layer_norm_x = nn.LayerNorm(num_hidden * 4)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_state(self, is_cuda):
        state = np.zeros((self.batch, self.num_hidden))
        state = torch.FloatTensor(state)
        if is_cuda:
            state = state.cuda()
        return state

    def forward(self, x, h, c, m):
        if h is None:
            h = self.init_state(x.is_cuda)
        if c is None:
            c = self.init_state(x.is_cuda)
        if m is None:
            m = self.init_state(x.is_cuda)

        # print(x.shape, h.shape)
        t_cc = self.h_fc(h)
        s_cc = self.m_fc(m)
        x_cc = self.x_fc(x)


        if self.layer_norm:
            t_cc = self.layer_norm_t(t_cc)
            s_cc = self.layer_norm_s(s_cc)
            x_cc = self.layer_norm_x(x_cc)

        # print(s_cc.size(), t_cc.size(), x_cc.size())
        chunk_size = s_cc.size(1) // 4
        i_s, g_s, f_s, o_s = torch.split(s_cc, chunk_size, 1)
        i_t, g_t, f_t, o_t = torch.split(t_cc, chunk_size, 1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, chunk_size, 1)

        # print(i_x.size(), i_t.size())

        i = F.sigmoid(i_x + i_t)
        i_ = F.sigmoid(i_x + i_s)
        g = F.tanh(g_x + g_t)
        g_ = F.tanh(g_x + g_s)
        f = F.sigmoid(f_x + f_t + self._forget_bias)
        f_ = F.sigmoid(f_x + f_s + self._forget_bias)
        o = F.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m + i_ * g_
        new_c = f * c + i * g
        cell = torch.cat((new_c, new_m),1)
        cell = self.cell_fc(cell)

        new_h = o * F.tanh(cell)

        return new_h, new_c, new_m