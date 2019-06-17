import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable

class LMModel(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer. 
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding. 
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput=1024, nhid=256, nlayers=1):
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you RNN model here. You can add additional parameters to the function.
        self.rnn = nn.RNN(ninput, nhid, nlayers, nonlinearity='tanh', dropout=0.5)
        ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden=None):
        embeddings = self.drop(self.encoder(input))
        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        output = None
        output, hidden = self.rnn(embeddings, hidden)
        ########################################

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

class Attention(nn.Module):
    def __init__(self,nhid):
        super(Attention,self).__init__()
        self.nhid=nhid
        self.linear_hid=nn.Linear(nhid,nhid)
        self.linear_input=nn.Linear(nhid,nhid)
        self.w=nn.Parameter(torch.randn(1,1,nhid))
        self.w.data.uniform_(-2. / math.sqrt(nhid), 2. / math.sqrt(nhid))

    def forward(self,input,hid):
        #input [B,F]
        #hid   [T,B,F]
        t, b, f=hid.size(0), hid.size(1), hid.size(2)
        hid=self.linear_hid(hid.view(t*b,f)).view(t,b,f)
        input=self.linear_input(input).view(1,b,f)
        total=hid+input
        total=torch.tanh(total)
        attention=total*self.w
        attention=torch.sum(attention,dim=2)
        attention=F.softmax(attention,dim=0)
        return attention.view(attention.size(0),attention.size(1),1)


class LMModel_att(nn.Module):
    # Language model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self,nvoc, ninput=1024, nhid=256, nlayers=1):
        super(LMModel_att, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.encoder = nn.Embedding(nvoc, ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Construct you RNN model here. You can add additional parameters to the function.
        # self.batch_size=batch_size
        self.rnn_1 = nn.GRU(ninput,nhid,nlayers,bidirectional=False)
        self.rnn_2 = nn.GRU(nhid+ninput,nhid,nlayers,bidirectional=False)
        self.fc_layer=nn.Linear(nhid,nhid)
        self.attention=Attention(nhid)
         ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers



    def init_weights(self):
        init_uniform = 0.1
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        embeddings = self.drop(self.encoder(input))
        batch_size = embeddings.size(1)
        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # Output has the dimension of sequence_length * batch_size * number of classes
        output = None
        hidden = Variable(torch.zeros(self.nlayers,batch_size,self.nhid)).cuda()
        output,hidden=self.rnn_1(embeddings,hidden)
        timestep=input.size(0)
        predicts=[]
        hidden_2= Variable(torch.zeros(self.nlayers,batch_size,self.nhid)).cuda()
        for i in range(0,timestep):
            energy=self.attention(hidden_2,output[:i+1])
            state=torch.sum(energy*hidden[:i+1],0)
            pred,hidden_2=self.rnn_2(torch.cat((state,embeddings[i]),dim=1).view(1,state.size(0),-1),hidden_2)
            predicts.append(pred.view(-1,pred.size(2)))
        # output=output[:,:,:self.nhid]+output[:,:,self.nhid:]
        ########################################
        output=torch.stack(predicts,dim=0)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
