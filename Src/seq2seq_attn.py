# import sys
# sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim

import random
import numpy as np
import pickle as pkl
import argparse
import os
import time

from Class.LSTM import LSTM
from Class.MiniBatchLoader import MiniBatchLoader

"""
This module was adapted from baseline model.
Reference: https://github.com/Alex-Fabbri/lang2logic-PyTorch
"""

# The RNN Class is used to wrap the encoder and decoder LSTM


class RNN(nn.Module):
    def __init__(self, opt, input_size):
        super(RNN, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
        self.embedding = nn.Embedding(input_size, self.hidden_size)
        self.lstm = LSTM(self.opt)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input_src, prev_c, prev_h):
        src_emb = self.embedding(input_src)  # batch_size * src_length * emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h)
        return prev_cy, prev_hy

    def init_embedding(self, emb_path):
        self.embedding.load_state_dict(torch.load(emb_path))
        # self.embedding.weight.requires_grad = False
        # use argument to select whether embedding will be fine-tuned

# The AttnUnit Class is used to wrap the attention layers


class AttnUnit(nn.Module):
    def __init__(self, opt, output_size):
        super(AttnUnit, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size

        self.linear_att = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, output_size)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0, 2, 1), attention)
        hid = torch.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), dec_s_top), 1)))
        h2y_in = hid
        if self.opt.dropout > 0:
            h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)
        return pred


def eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer, decoder_optimizer,
                  attention_decoder_optimizer, criterion, using_gpu, form_manager):
    """
    Perform forward pass of the model and compute loss
    :param opt: argument parser
    :param train_loader: MiniBatchLoader
    :param encoder: model encoder
    :param decoder: model decoder
    :param attention_decoder: attention layers
    :param encoder_optimizer: encoder optimizer
    :param decoder_optimizer: decoder optimizer
    :param attention_decoder_optimizer: attention layers optimizer
    :param criterion: loss function
    :param using_gpu: GPU
    :param form_manager: symbol manager
    :return: loss
    """
    # encode, decode, backward, return loss
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    attention_decoder_optimizer.zero_grad()
    enc_batch, enc_len_batch, dec_batch = train_loader.random_batch()
    # do not predict after <E>
    enc_max_len = enc_batch.size(1)
    # because you need to compare with the next token!!
    dec_max_len = dec_batch.size(1) - 1

    enc_outputs = torch.zeros((enc_batch.size(0), enc_max_len, encoder.hidden_size), requires_grad=False)
    if using_gpu:
        enc_outputs = enc_outputs.cuda()

    enc_s = {}
    for j in range(opt.enc_seq_length + 1):
        enc_s[j] = {}

    dec_s = {}
    for j in range(opt.dec_seq_length + 1):
        dec_s[j] = {}

    for i in range(1, 3):  # 1, 2 are index for cell and hidden, respectively. Initialization of hidden unit
        enc_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=False)
        dec_s[0][i] = torch.zeros((opt.batch_size, opt.rnn_size), dtype=torch.float, requires_grad=False)
        if using_gpu:
            enc_s[0][i] = enc_s[0][i].cuda()
            dec_s[0][i] = dec_s[0][i].cuda()

    for i in range(enc_max_len):  # unroll the encoder, iteration over training sequence
        enc_s[i+1][1], enc_s[i+1][2] = encoder(enc_batch[:,i], enc_s[i][1], enc_s[i][2])
        enc_outputs[:, i, :] = enc_s[i+1][2]

    loss = 0

    for i in range(opt.batch_size):
        dec_s[0][1][i, :] = enc_s[enc_len_batch[i]][1][i, :]
        dec_s[0][2][i, :] = enc_s[enc_len_batch[i]][2][i, :]

    for i in range(dec_max_len):
        dec_s[i+1][1], dec_s[i+1][2] = decoder(dec_batch[:,i], dec_s[i][1], dec_s[i][2])  # FIX IT
        pred = attention_decoder(enc_outputs, dec_s[i+1][2])
        loss += criterion(pred, dec_batch[:,i+1])

    loss = loss / opt.batch_size
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
    torch.nn.utils.clip_grad_value_(decoder.parameters(), opt.grad_clip)
    torch.nn.utils.clip_grad_value_(attention_decoder.parameters(), opt.grad_clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    attention_decoder_optimizer.step()
    return loss


def main(opt):
    """
    Main Function to perform training and save the model as a checkpoint
    :param opt: argument parser
    :return: None
    """
    # q_emb_path = '/Users/alvinkennardi/Documents/Master_of_Computing/COMP8755/Embedding/emb_layer_q_split15.pt'
    # f_emb_path = '/Users/alvinkennardi/Documents/Master_of_Computing/COMP8755/Embedding/emb_layer_f_split15.pt'
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    managers = pkl.load(open("{}/map.pkl".format(opt.data_dir), "rb"))
    word_manager, form_manager = managers
    using_gpu = False
    if opt.gpuid > -1:
        using_gpu = True
        torch.cuda.manual_seed(opt.seed)
    encoder = RNN(opt, word_manager.vocab_size)

    decoder = RNN(opt, form_manager.vocab_size)

    attention_decoder = AttnUnit(opt, form_manager.vocab_size)
    if using_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        attention_decoder = attention_decoder.cuda()
    # init parameters

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)
    for name, param in attention_decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -opt.init_weight, opt.init_weight)

    if opt.init_embed == 1:
        q_emb_path = opt.q_emb_file
        f_emb_path = opt.f_emb_file
        encoder.init_embedding(q_emb_path)  # trainable
        decoder.init_embedding(f_emb_path)  # trainable

    # load data
    train_loader = MiniBatchLoader(opt, 'train', using_gpu)

    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    optim_state = {"learningRate": opt.learning_rate, "alpha": opt.decay_rate}
    # default to RMSprop
    if opt.opt_method == 0:
        print("using RMSprop")
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=optim_state["learningRate"],
                                          alpha=optim_state["alpha"])
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=optim_state["learningRate"],
                                          alpha=optim_state["alpha"])
        attention_decoder_optimizer = optim.RMSprop(attention_decoder.parameters(), lr=optim_state["learningRate"],
                                                    alpha=optim_state["alpha"])
    criterion = nn.NLLLoss(reduction='sum', ignore_index=0)

    print("Starting training.")
    encoder.train()
    decoder.train()
    attention_decoder.train()
    iterations = opt.max_epochs * train_loader.num_batch
    eval_data = pkl.load(open("{}/test.pkl".format(opt.data_dir), "rb"))
    for i in range(iterations):
        epoch = i // train_loader.num_batch
        start_time = time.time()
        # print("iteration: {}\n".format(i))
        train_loss = eval_training(opt, train_loader, encoder, decoder, attention_decoder, encoder_optimizer,
                                   decoder_optimizer, attention_decoder_optimizer, criterion, using_gpu, form_manager)
        # exponential learning rate decay
        if opt.opt_method == 0:
            if i % train_loader.num_batch == 0 and opt.learning_rate_decay < 1:
                if epoch >= opt.learning_rate_decay_after:
                    decay_factor = opt.learning_rate_decay
                    optim_state["learningRate"] = optim_state["learningRate"] * decay_factor  # decay it
                    for param_group in encoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]
                    for param_group in attention_decoder_optimizer.param_groups:
                        param_group['lr'] = optim_state["learningRate"]

        end_time = time.time()
        if i % opt.print_every == 0:
            print("{}/{}, train_loss = {}, time/batch = {}".format(i, iterations, train_loss, (end_time - start_time) / 60))

        # on last iteration
        if i == iterations - 1:
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            checkpoint["attention_decoder"] = attention_decoder
            checkpoint["opt"] = opt
            checkpoint["i"] = i
            checkpoint["epoch"] = epoch
            torch.save(checkpoint, "{}/model_seq2seq_attention".format(opt.checkpoint_dir))

        if train_loss != train_loss:
            print('loss is NaN.  This usually indicates a bug.')
            break


if __name__ == "__main__":
    start = time.time()
    main_arg_parser = argparse.ArgumentParser(description="parser")
    main_arg_parser.add_argument('-gpuid', type=int, default=-1, help='which gpu to use. -1 = use CPU')
    main_arg_parser.add_argument('-data_dir', type=str, default='../Data/GEO', help='data path')  # default ATIS_EXP
    main_arg_parser.add_argument('-seed', type=int, default=123, help='torch manual random number generator seed')
    main_arg_parser.add_argument('-checkpoint_dir', type=str, default='checkpoint_dir',
                                 help='output directory where checkpoints get written')
    main_arg_parser.add_argument('-savefile', type=str, default='save',
                                 help='filename to autosave the checkpont to. Will be inside checkpoint_dir/')
    main_arg_parser.add_argument('-print_every', type=int, default=500,
                                 help='how many steps/minibatches between printing out the loss')
    main_arg_parser.add_argument('-rnn_size', type=int, default=200,
                                 help='size of LSTM internal state')  # change from default 200
    main_arg_parser.add_argument('-num_layers', type=int, default=1, help='number of layers in the LSTM')
    main_arg_parser.add_argument('-dropout', type=float, default=0.4,
                                 help='dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
    main_arg_parser.add_argument('-dropoutrec', type=int, default=0,
                                 help='dropout for regularization, used after each c_i. 0 = no dropout')
    main_arg_parser.add_argument('-enc_seq_length', type=int, default=50, help='number of timesteps to unroll for')
    main_arg_parser.add_argument('-dec_seq_length', type=int, default=100,
                                 help='number of timesteps to unroll for')  # Change default value from 100 to 150
    main_arg_parser.add_argument('-batch_size', type=int, default=20,
                                 help='number of sequences to train on in parallel')  # default: 80
    main_arg_parser.add_argument('-max_epochs', type=int, default=80,
                                 help='number of full passes through the training data')
    main_arg_parser.add_argument('-opt_method', type=int, default=0, help='optimization method: 0-rmsprop 1-sgd')
    main_arg_parser.add_argument('-learning_rate', type=float, default=0.01, help='learning rate')
    main_arg_parser.add_argument('-init_weight', type=float, default=0.08, help='initailization weight')
    main_arg_parser.add_argument('-learning_rate_decay', type=float, default=0.98, help='learning rate decay')
    main_arg_parser.add_argument('-learning_rate_decay_after', type=int, default=5,
                                 help='in number of epochs, when to start decaying the learning rate')
    main_arg_parser.add_argument('-restart', type=int, default=-1,
                                 help='in number of epochs, when to restart the optimization')
    main_arg_parser.add_argument('-decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
    main_arg_parser.add_argument('-grad_clip', type=int, default=5, help='clip gradients at this value')
    main_arg_parser.add_argument('-sample', type=int, default=0,
                                 help='0 to use max at each timestep (-beam_size=1), '
                                      '1 to sample at each timestep, 2 to beam search')
    main_arg_parser.add_argument('-init_embed', type=int, default=0,
                                 help= 'initialize embedding using pre-trained embedding if it is set to 1')
    main_arg_parser.add_argument('-q_emb_file', type=str , default='../Embedding/emb_layer_geo_q.pt',
                                 help='file name for question embedding')
    main_arg_parser.add_argument('-f_emb_file', type=str, default='../Embedding/emb_layer_geo_f.pt',
                                 help='file name for form embedding')

    args = main_arg_parser.parse_args()
    main(args)
    end = time.time()
    print("total time: {} minutes\n".format((end - start) / 60))
