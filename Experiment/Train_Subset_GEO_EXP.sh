#!/usr/bin/env bash

declare -a subset=(../Data/GEO_EXP_10 ../Data/GEO_EXP_20 ../Data/GEO_EXP_30  ../Data/GEO_EXP_40 ../Data/GEO_EXP_50
                   ../Data/GEO_EXP_60 ../Data/GEO_EXP_70 ../Data/Geo_EXP_80  ../Data/GEO_EXP_90 ../Data/GEO_EXP)

declare -a checkpoint=(checkpoint_geo_exp_10
                       checkpoint_geo_exp_20
                       checkpoint_geo_exp_30
                       checkpoint_geo_exp_40
                       checkpoint_geo_exp_50
                       checkpoint_geo_exp_60
                       checkpoint_geo_exp_70
                       checkpoint_geo_exp_80
                       checkpoint_geo_exp_90
                       checkpoint_geo_exp)

declare -a output=(Output/seq2seq_attention_output_geo_exp_10.txt
                   Output/seq2seq_attention_output_geo_exp_20.txt
                   Output/seq2seq_attention_output_geo_exp_30.txt
                   Output/seq2seq_attention_output_geo_exp_40.txt
                   Output/seq2seq_attention_output_geo_exp_50.txt
                   Output/seq2seq_attention_output_geo_exp_60.txt
                   Output/seq2seq_attention_output_geo_exp_70.txt
                   Output/seq2seq_attention_output_geo_exp_80.txt
                   Output/seq2seq_attention_output_geo_exp_90.txt
                   Output/seq2seq_attention_output_geo_exp.txt)

declare -a model=(../Model/checkpoint_geo_exp_10/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_20/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_30/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_40/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_50/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_60/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_70/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_80/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp_90/model_seq2seq_attention
                  ../Model/checkpoint_geo_exp/model_seq2seq_attention)

cd ../Model

export PYTHONPATH=${PYTHONPATH}:../

for i in ${!subset[*]};
do
    DATA_DIR=${subset[$i]}
    CHECKPOINT=${checkpoint[$i]}
    python -u seq2seq_attn.py -checkpoint_dir $CHECKPOINT -data_dir $DATA_DIR -print_every 300 -rnn_size 150\
    -dropout 0.5 -enc_seq_length 40 -dec_seq_length 100 -max_epochs 120 -seed 1000
done

cd ../Evaluation

for i in ${!subset[*]};
do
    DATA_DIR=${subset[$i]}
    OUTPUT=${output[$i]}
    MODEL=${model[$i]}
    python Sample.py -data_dir $DATA_DIR -output $OUTPUT -model $MODEL
done