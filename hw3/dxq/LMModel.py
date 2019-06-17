import torch
import torch.nn as nn

from RNNModel import basic_lstm
from regulation.embed_regularize import embedded_dropout
from regulation.locked_dropout import LockedDropout
from regulation.weight_drop import WeightDrop


class AwdRNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.2, wdrop=0.2, tie_weights=False):
        super(AwdRNN, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['AWD-LSTM', 'AWD-GRU'], 'RNN type is not supported'

        if rnn_type == 'AWD-LSTM':
            self.rnns = [
                torch.nn.LSTM(ninp if l == 0 else nhid, nhid,
                              1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'AWD-GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid, 1, dropout=0) for l
                         in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)
        hidden = self.init_hidden(input.size(1))
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        result = self.decoder(result).view(output.size(0), output.size(1), -1)
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'AWD-LSTM':
            return [(weight.new(1, bsz, self.nhid).zero_(),
                    weight.new(1, bsz, self.nhid).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'AWD-GRU':
            return [weight.new(1, bsz, self.nhid).zero_()
                    for l in range(self.nlayers)]


class BasicRNN(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, rnn_type, nvoc, ninput, nhid, nlayers):
        super(BasicRNN, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.embedding = nn.Embedding(nvoc, ninput)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(ninput, nhid, nlayers)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(ninput, nhid, nlayers)
        elif rnn_type == "MyLSTM":
            self.rnn = basic_lstm.BasicLstm(ninput, nhid, nlayers)

        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.ninput = ninput
        self.nhid = nhid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.softmax = nn.Softmax()
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.embedding.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        hidden = self.init_hidden(input.size(1))
        embeddings = self.drop(self.embedding(input))
        rnn_out, hstate = self.rnn(embeddings, hidden)

        output = rnn_out
        hidden = hstate

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        elif self.rnn_type == "GRU":
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
        elif self.rnn_type == "MyLSTM":
            return [(weight.new(bsz, self.nhid).zero_(),
                     weight.new(bsz, self.nhid).zero_())
                        for l in range(self.nlayers)]

