#!/usr/bin/env bash

cd ../Src

export PYTHONPATH=${PYTHONPATH}:../

python -u seq2seq_attn.py -checkpoint_dir checkpoint_geo_default -data_dir ../Data/GEO -print_every 300 -rnn_size 150\
 -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 90 -seed 100

cd ../Evaluation

python Sample.py -data_dir ../Data/GEO -output Output/seq2seq_attention_output_on_geo.txt -model ../Src/checkpoint_geo_default/model_seq2seq_attention