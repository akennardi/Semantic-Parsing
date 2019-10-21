import time
import random
import math
from random import randint
import pickle as pkl
import numpy as np
import torch
import os
from operator import itemgetter

"""
Class to manage symbol and keep track index
Reference: https://github.com/Alex-Fabbri/lang2logic-PyTorch
"""


class SymbolManager:
    """class is adapted from baseline code"""

    def __init__(self, whether_add_special_tags):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0
        self.whether_add_special_tags = whether_add_special_tags
        if whether_add_special_tags:
            # start symbol = 0
            self.add_symbol('<S>')
            # end symbol = 1
            self.add_symbol('<E>')
            # UNK symbol = 2
            self.add_symbol('<U>')

    def add_symbol(self, s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size = self.vocab_size + 1
        return self.symbol2idx[s]

    def get_symbol_idx(self, s):
        if s not in self.symbol2idx:
            if self.whether_add_special_tags:
                return self.symbol2idx['<U>']
            else:
                print("this should never be reached (always add <U>")
                return 0
        return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return '<U>'
        return self.idx2symbol[idx]

    def init_from_file(self, fn, min_freq, max_vocab_size):
        print("loading vocabulary file: {}\n".format(fn))
        with open(fn, "r") as f:
            for line in f:
                l_list = line.strip().split('\t')
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size >= max_vocab_size:
                    break

    def get_symbol_idx_for_list(self, l):
        r = []
        for i in range(len(l)):
            r.append(self.get_symbol_idx(l[i]))
        return r




