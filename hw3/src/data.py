import os
import torch
import utils


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def save_dict(self, path):
        utils.list_save(self.idx2word, path + "_dict_idx2word")
        utils.dict_save(self.word2idx, path + "_dict_word2idx")

    def load_dict(self, path):
        self.idx2word = utils.list_load(path + "_dict_idx2word")
        self.word2idx = utils.dict_load(path + "_dict_word2idx")


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
        self.train = None
        self.valid = None
        self.test = None

    def set_train(self, path, train_size=-1):
        self.train = self.tokenize(path, train_size)

    def set_valid(self, path, valid_size=-1):
        self.valid = self.tokenize(path, valid_size)

    def set_test(self, path, test_size=-1):
        self.test = self.tokenize(path, test_size)

    def tokenize(self, path, target_size):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            n_read_line = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                n_read_line += 1
                if n_read_line % 5000 == 0:
                    print("add words %d" % n_read_line)
                if n_read_line == target_size:
                    break

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            n_read_line = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
                n_read_line += 1
                if n_read_line % 5000 == 0:
                    print("tokenize %d" % n_read_line)
                if n_read_line == target_size:
                    break

        return ids

    def save_dictionary(self, dir_path):
        self.dictionary.save_dict(dir_path)

    def load_dictionary(self, dir_path):
        self.dictionary.load_dict(dir_path)
