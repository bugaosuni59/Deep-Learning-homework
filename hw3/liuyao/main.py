# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn

import data
# from .data import Corpus
import model
# from .model import LMModel
# from src import model
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=100, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=3, help='GPU device id used')
parser.add_argument('--emsize', type=int, default=1500, help='size of word embeddings ')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer ')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers ')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights ')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    # device = torch.device(args.gpu_id)
    device = torch.device('cuda:0')
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("data/ptb", batch_size, args.max_sql)
        
# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)
nvoc = len(data_loader.vocabulary)
model = model.LMModel(rnn_type=args.model,
                      nvoc=nvoc,
                      ninput=args.emsize,
                      nhid=args.nhid,
                      nlayers=args.nlayers,
                      tie_weights=args.tied)
model = model.to(device)
lr = args.lr
best_val_loss = None
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
########################################

criterion = nn.CrossEntropyLoss()



# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    model.eval()
    data_loader.set_valid()
    end_flag = False
    total_loss = 0.

    hidden = model.init_hidden(args.eval_batch_size)
    with torch.no_grad():
        while(not end_flag):
            data, target, end_flag = data_loader.get_batch()
            data = data.to(device)
            target = target.to(device)
            # hidden = hidden.to(device)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, nvoc)
            hidden = repackage_hidden(hidden)
            # total_loss += len(data) * criterion(output_flat, target).item()
            total_loss += criterion(output_flat, target).item()
    # val_loss = total_loss / (data_loader.valid.size(0)/args.max_sql)
    val_loss = total_loss / (data_loader.valid.size(0)/args.max_sql)
    return val_loss

########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train():
    model.train()
    data_loader.set_train()
    end_flag = False
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.train_batch_size)
    batch = 0
    while(not end_flag):
        data, target, end_flag = data_loader.get_batch()
        hidden = repackage_hidden(hidden)
        data = data.to(device)
        target = target.to(device)
        # hidden = hidden.to(device)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, nvoc), target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        optimizer.step()

        total_loss += loss.item()
        # epoch_loss = total_loss / nvoc
        epoch_loss = total_loss / (data_loader.train.size(0)/args.max_sql)

        # if batch % args.log_interval == 0 and batch > 0:
        #     cur_loss = total_loss / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
        #           'loss {:5.2f} | ppl {:8.2f}'.format(
        #         epoch, batch, data_loader.train_batch_num, lr,
        #                       elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
        #     total_loss = 0.
        #     start_time = time.time()
        batch += 1
    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | time {:5.2f} | '
          'loss {:5.2f} | ppl {:8.2f}'.format(
        epoch, batch, data_loader.train_batch_num, lr,
        time.time() - start_time, epoch_loss, math.exp(epoch_loss)))

########################################


# Loop over epochs.
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val_loss = val_loss
    # else:
    #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
    #     lr /= 4.0
print('| The end | best valid loss {:5.2f} | best valid pp {:8.2f}'.format(best_val_loss, math.exp(best_val_loss)))

