import pickle
import argparse

import torch


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
    parser.add_argument('--model', type=str, default='GRU',
                        help='type of recurrent net (LSTM, GRU)')
    parser.add_argument('--nembed', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--data', type=str, default='./data/ptb',
                        help='location of the data corpus')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model_dir',
                        help='path to save the final model and dictionary')
    parser.add_argument('--train_size', type=int, default=-1, metavar='N',
                        help='train size')
    parser.add_argument('--valid_size', type=int, default=-1, metavar='N',
                        help='valid size')
    parser.add_argument('--test_size', type=int, default=-1, metavar='N',
                        help='test size')
    parser.add_argument('--use_MyGRU', action='store_true',
                        help='use MyGRU')

    args = parser.parse_args()
    return args


def list_save(content, filename, mode='w'):
    if type(content) is not list:
        raise TypeError("expect list")
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


def list_load(filename):
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content


def dict_save(content, filename, mode='wb'):
    if type(content) is not dict:
        raise TypeError("expect dict")
    f1 = open(filename, mode)
    pickle.dump(content, f1)
    f1.close()


def dict_load(filename):
    f2 = open(filename, 'rb')
    content = pickle.load(f2)
    f2.close()
    return content


def check_device(args):
    if args.use_MyGRU:
        print("MyGPU doesn't support GPU, use CPU default")
        return torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return torch.device("cuda" if args.cuda else "cpu")


def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(args, source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(data_source, model, corpus, criterion, args, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(args, data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)
