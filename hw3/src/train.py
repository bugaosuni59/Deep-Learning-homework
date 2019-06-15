# coding: utf-8
import math
import os
import sys
import torch
import torch.nn as nn

import my_model
import model
import utils
import data
import time

start = time.time()

# prepare
eval_batch_size = 10
args = utils.get_args_parser()
print('\n')
print('V ' * 80)
print('\n')
print(args)
print('\n')
print('#' * 80)
print("start!====")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = utils.check_device(args)
if args.use_MyGRU:
    print("train by MyGRU!")
###############################################################################
# Load data
###############################################################################
if not os.path.exists(args.save):
    os.makedirs(args.save)

corpus = data.Corpus()
os.path.join(args.data, 'train.txt')
corpus.set_train(os.path.join(args.data, 'train.txt'), args.train_size)
corpus.set_valid(os.path.join(args.data, 'valid.txt'), args.valid_size)
corpus.save_dictionary(os.path.join(args.save, 'rnn_model_dict'))

train_data = utils.batchify(corpus.train, args.batch_size, device)
val_data = utils.batchify(corpus.valid, eval_batch_size, device)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.use_MyGRU:
    model = my_model.MyGRUModel(ntokens, args.nembed, args.nhid, args.nlayers, args.dropout)
else:
    model = model.RNNModel(args.model, ntokens, args.nembed, args.nhid, args.nlayers, args.dropout).to(device)

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def train():
    # set the flags of conponents in model to be training: self.training = mode
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    ret_ppl = -1
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = utils.get_batch(args, train_data, i)
        hidden = utils.repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            # if ret_ppl is None:
            ret_ppl = math.exp(cur_loss)
    return ret_ppl


# Loop over epochs.
lr = args.lr
best_val_loss = None

try:
    result = []
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_ppl = train()
        val_loss = utils.evaluate(val_data, model, corpus, criterion, args, eval_batch_size)
        print('-' * 80)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 80)
        # Save the model if the validation loss is the best we've seen so far.
        if best_val_loss is None or val_loss < best_val_loss:
            print('save model!')
            with open(os.path.join(args.save, 'model.pt'), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
        result.append((epoch, train_ppl, math.exp(val_loss)))
    # print for figure plotting
    print('epoch\ttrain-ppl\tvalid-ppl')
    for line in result:
        print('%d\t%.3f\t%.3f' % (line[0], line[1], line[2]))

except KeyboardInterrupt:
    print('-' * 80)
    print('Exiting from training early')
