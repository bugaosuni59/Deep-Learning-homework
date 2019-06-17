# coding: utf-8
import argparse
import time
import math
import shutil
import numpy as np
import torch
import torch.nn as nn

import data
import LMModel
import os

from metrics import bleu, rouge
from utils import *
from Attention import Attention, AttModel
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--lr', type=float, default=0.001,
                    help='upper epoch limit')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--data_path', type=str, default="../data/ptb",
                    help='datat path')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--save_path', type=str, default="../model",
                    help='model save path')
parser.add_argument('--model_name', type=str, default="LSTM",
                    help='model name')
parser.add_argument('--suffix', type=str, default="",
                    help='model name suffix')
parser.add_argument('--add_att', action="store_true", default=False,
                    help='weather add attention')
parser.add_argument('--save_iter', type=int, default=10,
                    help='how long to save model')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--ninput', type=int, default=512,
                    help='words embeding dim')
parser.add_argument('--nhidden', type=int, default=512,
                    help='rnn hidden dim')
parser.add_argument('--nlayers', type=int, default=1,
                    help='num rnn layers')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--resume', type=str, default="",
                    help='pretrain model path')
parser.add_argument('--mode', type=str, default="train",
                    help='train or test')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = args.cuda

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
data_path = args.data_path
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus(data_path, batch_size, args.max_sql)


full_model_name = args.model_name + "_" + args.suffix
writer = SummaryWriter(log_dir="../summary/{}".format(full_model_name))

        
# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)

nvoc = len(data_loader.vocabulary)
cutoffs, tie_projs = [], [False]
if args.model_name in ["LSTM", "MyLSTM", "GRU"]:
    model = LMModel.BasicRNN(args.model_name, nvoc, args.ninput, args.nhidden, args.nlayers)
elif args.model_name in ["AWD-LSTM", "AWD-GRU"]:
    model = LMModel.AwdRNN(args.model_name, nvoc, args.ninput, args.nhidden, args.nlayers)
elif args.model_name == "GRU-ATT":
    model = Attention.GRUAtt(args.model_name, nvoc, args.ninput, args.nhidden, args.nlayers)
elif args.model_name == "LSTM-ATT":
    # model = Attention.LSTMAtt(args.model_name, nvoc, args.ninput, args.nhidden, args.nlayers)
    model = Attention.LNLSTMAtt(args.model_name, nvoc, args.ninput, args.nhidden, args.nlayers)

if use_gpu:
    model.cuda()

if args.resume != "":
    model.load_state_dict(torch.load(args.resume))

########################################
criterion = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(criterion.parameters())

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[20,40,60,80,100],
                                                gamma=0.3)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate():
    loss_sum = 0
    bleu_sum = 0
    rouge_sum = 0
    total = 0
    model.eval()
    data_loader.set_valid()

    while True:
        data, target, end_flag = data_loader.get_batch()
        # move data to gpu
        if use_gpu:
            data = data.cuda()
            target = target.cuda()

        with torch.no_grad():
            # caculate output and loss
            output, hidden = model(data)
            loss = criterion(output.view(-1, nvoc), target)
            _, predicted = torch.max(output, -1)

        len = data.size(1)
        loss_sum += len * loss
        total += len
        label = target.view(predicted.size()).data.cpu().numpy()
        predicted = predicted.data.cpu().numpy()
        bleu_sum += len * bleu.cal_blue_score(predicted, label)
        rouge_sum += len * rouge.cal_rouge_score(predicted, label)

        if end_flag:
            break
    loss = loss_sum.item() / total
    perplexity = 2 ** loss
    bleu_val = bleu_sum / total
    rouge_val = rouge_sum / total
    print("valid loss={}, ppl={}, bleu={}, rouge={}".format(loss, perplexity, bleu_val, rouge_val))
    return loss, perplexity, bleu_val, rouge_val
########################################


# WRITE CODE HERE within two '#' bar
########################################


# Train Function
def train(epoch):
    model.train()
    data_loader.set_train()
    loss_np = []
    while True:
        data, target, end_flag = data_loader.get_batch()

        # move data to gpu
        if use_gpu:
            data = data.cuda()
            target = target.cuda()

        # caculate output and loss

        if args.model_name in ["AWD-LSTM", "AWD-GRU"]:
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, return_h=True)
            loss = criterion(output.view(-1, nvoc), target)
            # Activiation Regularization
            if args.alpha: loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        elif args.model_name in ["LSTM", "MyLSTM", "GRU", "LSTM-ATT", "GRU-ATT"]:
            output, hidden = model(data)
            loss = criterion(output.view(-1, nvoc), target)

        # backward
        optimizer.zero_grad()
        loss.backward()
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        loss_np.append(loss.data)
        if end_flag:
            break
    loss = np.mean(loss_np)
    writer.add_scalar("Train/loss", loss, epoch)
    writer.add_scalar("Train/ppl", 2**loss, epoch)
    print("train epoch {}, loss={}".format(epoch, np.mean(loss_np)))

########################################

def re_eval():
    for epoch in range(1, 201):
        if epoch % 20 == 0:
            model_path = "../model/LSTM-ATT_512/model_{}.pth".format(epoch)
            model.load_state_dict(torch.load(model_path))
            loss = evaluate()
            writer.add_scalar("Val/loss", loss[0], epoch)
            writer.add_scalar("Val/ppl", loss[1], epoch)
    exit()

if args.mode == "test":
    evaluate()
    # re_eval()
    exit()

save_path = os.path.join(args.save_path, full_model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Loop over epochs.
for epoch in range(1, args.epochs+1):
    train(epoch)
    scheduler.step()
    if epoch % args.save_iter == 0:
        file_path = os.path.join(save_path, "model_{}.pth".format(epoch))
        torch.save(model.state_dict(), file_path)
    loss = evaluate()
    writer.add_scalar("Val/loss", loss[0], epoch)
    writer.add_scalar("Val/ppl", loss[1], epoch)
