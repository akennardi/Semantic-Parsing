#!/usr/bin/env bash

cd ../Model

export PYTHONPATH=${PYTHONPATH}:../

python -u seq2seq_attn.py -checkpoint_dir checkpoint_dir_atis -data_dir ../Data/ATIS -print_every 500 -rnn_size 200 -dropout 0.4 -enc_seq_length 50 -dec_seq_length 150 -max_epochs 80

cd ../Evaluation

python Sample.py -data_dir ../Data/ATIS -output Output/seq2seq_attention_output_on_atis.txt -model ../Model/checkpoint_dir_atis/model_seq2seq_attention