#!/usr/bin/env bash

DATA_DIR=../Data/GEO_EXP_QUERY

cd ../Src

export PYTHONPATH=${PYTHONPATH}:../

python -u seq2seq_attn.py -checkpoint_dir checkpoint_geo_exp_query -data_dir $DATA_DIR -print_every 300 \
-rnn_size 150 -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 120 -seed 100 -init_embed 0

cd ../Evaluation

python Sample.py -data_dir $DATA_DIR -output Output/seq2seq_attention_output_on_geo_exp_query.txt\
 -model ../Src/checkpoint_geo_exp_query/model_seq2seq_attention