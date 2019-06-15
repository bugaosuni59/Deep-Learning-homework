# coding: utf-8

import math
import os

import torch

from torch import nn
import time

import data

import utils

start = time.time()

# prepare
eval_batch_size = 10
args = utils.get_args_parser()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = utils.check_device(args)

###############################################################################
# Load the best saved model.
###############################################################################

with open(os.path.join(args.save, 'model.pt'), 'rb') as f:
    rnn_model = torch.load(f)
    if hasattr(rnn_model, 'rnn'):
        rnn_model.rnn.flatten_parameters()
    criterion = nn.CrossEntropyLoss()

test_corpus = data.Corpus()
test_corpus.load_dictionary(os.path.join(args.save, 'rnn_model_dict'))
test_corpus.set_test(os.path.join(args.data, 'test.txt'), test_size=args.test_size)

test_data = utils.batchify(test_corpus.test, eval_batch_size, device)
test_loss = utils.evaluate(test_data, rnn_model, test_corpus, criterion, args, eval_batch_size)

print('=' * 80)
print('| End of training | time: %5.2f s | test loss %5.2f | test ppl %8.2f'
      % ((time.time() - start), test_loss, math.exp(test_loss)))
