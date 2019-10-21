import io
import pickle as pkl
import argparse
import gensim
import datetime
import os
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import warnings
import numpy as np


main_arg_parser = argparse.ArgumentParser(description="parser")
main_arg_parser.add_argument("-embed_dir", type=str, default="../Embedding/",
                             help="Embedding directory")
main_arg_parser.add_argument("-file_name", type=str, default="ATIS_EXP.vec",
                             help="Pre-trained embedding name")
main_arg_parser.add_argument("-data_dir", type=str, default="../Data/GEO_EXP",
                             help="Data set directory")
main_arg_parser.add_argument("-seed", type=int, default=123,
                             help='torch manual random number generator seed')
main_arg_parser.add_argument("-embed_dim", type=int, default=150,
                             help='embedding dimension')
main_arg_parser.add_argument('-init_weight', type=float, default=0.08,
                             help='initailization weight')

warnings.filterwarnings("ignore")  # warning comes from gensim implementation to open file. Will be removed

# Load vec file
args = main_arg_parser.parse_args()
embedding_path = os.path.join(args.embed_dir, args.file_name)
print("Start to load model at {}".format(datetime.datetime.now()))
model = KeyedVectors.load_word2vec_format(embedding_path)
print("End to load model at {}".format(datetime.datetime.now()))
print("model loaded")

# Load managers file. Managers file has the information of the index
managers = pkl.load(open("{}/map.pkl".format(args.data_dir), "rb"))
word_manager, form_manager = managers

# create question embedding
q_vocab = list(word_manager.symbol2idx.keys())
print("number of vocabulary in the question: {}".format(len(q_vocab)))

weight_matrix_q = np.zeros((len(q_vocab), args.embed_dim))
count_outside_vocab_q = 0
for i, word in enumerate(q_vocab):
    try:
        weight_matrix_q[i] = model[word]
    except KeyError:
        weight_matrix_q[i] = np.random.uniform(-args.init_weight, args.init_weight, size=(args.embed_dim, ))
        count_outside_vocab_q += 1

print("number of vocabulary outside pre-trained vocabulary (question):{}".format(count_outside_vocab_q))
emb_layer_q = nn.Embedding(len(q_vocab), args.embed_dim)
emb_layer_q.load_state_dict({'weight': torch.Tensor(weight_matrix_q)})
emb_q_path = os.path.join(args.embed_dir, 'emb_layer_geo_q.pt')
torch.save(emb_layer_q.state_dict(), emb_q_path)

# create form embedding
f_vocab = list(form_manager.symbol2idx.keys())
print("number of vocabulary in the question: {}".format(len(f_vocab)))

weight_matrix_f = np.zeros((len(f_vocab), args.embed_dim))
count_outside_vocab_f = 0
for i, word in enumerate(f_vocab):
    try:
        weight_matrix_f[i] = model[word]
    except KeyError:
        weight_matrix_f[i] = np.random.uniform(-args.init_weight, args.init_weight, size=(args.embed_dim, ))
        count_outside_vocab_f += 1

print("number of vocabulary outside pre-trained vocabulary (form):{}".format(count_outside_vocab_f))
emb_layer_f = nn.Embedding(len(f_vocab), args.embed_dim)
emb_layer_f.load_state_dict({'weight': torch.Tensor(weight_matrix_f)})
emb_f_path = os.path.join(args.embed_dir, 'emb_layer_geo_f.pt')
torch.save(emb_layer_f.state_dict(), emb_f_path)





